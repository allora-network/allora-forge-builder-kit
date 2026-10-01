import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import polars as pl
from numba import jit

from .data_manager_factory import DataManager
from .base_data_manager import BaseDataManager


@jit(nopython=True, cache=False)
def _extract_features_numba(
    ts_index, 
    data_values, 
    start_times_int, 
    number_of_input_bars
):
    """
    JIT-compiled core loop for feature extraction.
    Data is already resampled, so we just extract windows of bars.
    Returns features array and valid indices.
    """
    n_features = number_of_input_bars * 5
    max_results = len(start_times_int)
    features_out = np.empty((max_results, n_features), dtype=np.float64)
    valid_indices = np.empty(max_results, dtype=np.int64)
    valid_count = 0
    
    for idx in range(len(start_times_int)):
        T_int = start_times_int[idx]
        
        # Binary search for position - must use side='right' to match original
        # Find rightmost position where T_int could be inserted
        left, right = 0, len(ts_index)
        while left < right:
            mid = (left + right) // 2
            if ts_index[mid] <= T_int:
                left = mid + 1
            else:
                right = mid
        pos = left
        
        if pos - number_of_input_bars < 0:
            continue
        
        window = data_values[pos - number_of_input_bars:pos]
        
        if window.shape[0] != number_of_input_bars:
            continue
        
        # Extract OHLCV directly (no aggregation needed since data is already resampled)
        last_close = window[-1, 3]
        last_volume = window[-1, 4]
        
        # Validate last values
        if last_close == 0 or np.isnan(last_close) or last_volume == 0 or np.isnan(last_volume):
            continue
        
        # Normalize and flatten
        feature_idx = 0
        for bar_i in range(number_of_input_bars):
            for feat_i in range(5):
                val = window[bar_i, feat_i]
                if feat_i < 4:  # OHLC
                    val /= last_close
                else:  # Volume
                    val /= last_volume
                features_out[valid_count, feature_idx] = val
                feature_idx += 1
        
        valid_indices[valid_count] = idx
        valid_count += 1
    
    return features_out[:valid_count], valid_indices[:valid_count]


class AlloraMLWorkflow:
    def __init__(
        self,
        tickers,
        number_of_input_bars,
        target_bars,
        interval="5m",
        data_source="binance",  # Simple string API
        data_manager=None,  # Advanced: explicit instance
        target_type="log_return",
        **data_manager_kwargs  # Pass through to data manager (market, api_key, etc.)
    ):
        """
        High-level ML workflow built on top of DataManager.

        Args:
            tickers: List of ticker symbols
            number_of_input_bars: Number of resampled bars to use as features (at the specified interval)
            target_bars: Number of bars ahead to predict (at the specified interval)
            interval: Bar interval (e.g. "5m", "1h")
            data_source: Data source string ("binance" or "allora") - simple API
            data_manager: Optional pre-configured data manager instance - advanced API
            target_type: Type of prediction target. One of:
                - "log_return" (default): log(close[t+H] / close[t])
                - "volatility": std of 1-minute log returns over the target horizon.
                  Requires interval="1m"; target_bars defines the horizon in minutes.
                - "triple_barrier": one-hot up/neutral/down first-touch targets;
                  horizon = target_bars × interval, ATR lookback = 100 horizons.
            **data_manager_kwargs: Arguments passed to DataManager factory:
                - Binance: market="futures", batch_timeout=20, base_dir="..."
                - Allora: api_key="...", base_dir="...", max_pages=1000
        
        Examples:
            # Simple API - Binance (24 hours ahead with 1-hour bars)
            workflow = AlloraMLWorkflow(
                tickers=["BTCUSDT"],
                number_of_input_bars=24,  # Use 24 bars (24 hours at 1h interval)
                target_bars=24,  # Predict 24 bars ahead (24 hours)
                interval="1h",
                data_source="binance",
                market="futures"  # Binance-specific param
            )
            
            # Simple API - Allora (24 hours ahead with 5-min bars)
            workflow = AlloraMLWorkflow(
                tickers=["BTC/USD"],
                number_of_input_bars=288,  # Use 288 bars (24 hours of 5-min bars)
                target_bars=288,  # Predict 288 bars ahead (24 hours)
                interval="5m",
                data_source="allora",
                api_key="your-key"  # Allora-specific param
            )
            
            # Volatility target - 15-minute BTC/USD volatility (Topic 79)
            workflow = AlloraMLWorkflow(
                tickers=["btcusd"],
                number_of_input_bars=15,
                target_bars=15,  # 15-minute horizon
                interval="1m",
                target_type="volatility",
                data_source="allora",
                api_key="your-key"
            )
            
            # Advanced API - explicit instance
            dm = DataManager(source="binance", interval="5m", market="futures")
            workflow = AlloraMLWorkflow(..., data_manager=dm)
        """
        _valid_target_types = ("log_return", "volatility", "triple_barrier")
        if target_type not in _valid_target_types:
            raise ValueError(
                f"target_type must be one of {_valid_target_types}, got {target_type!r}"
            )

        self.tickers = tickers
        self.number_of_input_bars = number_of_input_bars
        self.target_bars = target_bars
        self.target_type = target_type
        self.test_targets = None
        self.interval = interval

        # Use provided data manager instance, or create one from the factory string.
        if data_manager is not None:
            # Advanced API: explicit instance
            if not isinstance(data_manager, BaseDataManager):
                raise TypeError(
                    f"data_manager must be an instance of BaseDataManager, "
                    f"got {type(data_manager)}"
                )
            self._dm = data_manager
            # Update interval if manager uses different interval
            if hasattr(self._dm, 'interval'):
                self.interval = self._dm.interval
        else:
            # Simple API: create from factory using string
            self._dm = DataManager(
                source=data_source,
                interval=interval,
                symbols=tickers,
                **data_manager_kwargs
            )

        # Validate effective interval AFTER data_manager may have overridden self.interval.
        # Checking the raw `interval` arg before this branch would raise a false ValueError
        # when a 1m data_manager is passed without explicit interval="1m", and would miss
        # the case where interval="1m" is overridden to a non-1m interval by the manager.
        if target_type == "volatility" and self.interval != "1m":
            raise ValueError(
                f"target_type='volatility' requires interval='1m' (got {self.interval!r}). "
                "Volatility targets are defined as std of 1-minute log returns."
            )
    
    def _parse_interval_to_bars_per_hour(self, interval: str) -> float:
        """Convert interval string (e.g., '5m', '1h', '15m') to bars per hour."""
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            return 60.0 / minutes
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            return 1.0 / hours
        elif interval.endswith('d'):
            days = int(interval[:-1])
            return 1.0 / (days * 24)
        else:
            raise ValueError(f"Unsupported interval format: {interval}")

    # ---------- Public orchestration ----------
    def backfill(self, start=None):
        """Backfill OHLCV data for all workflow tickers."""
        print(f"[workflow] Backfilling {self.tickers} {start} → {'now'}")
        self._dm.backfill_missing(self.tickers, start=start)

    def load_raw(self, start=None, end=None):
        """Load raw OHLCV data for all workflow tickers as a Pandas DataFrame."""
        return self._dm.load_pandas(self.tickers, start=start)

    def stream_live_predictions(self, model, feature_fn=None):
        """
        Stream live predictions by attaching to DataManager's websocket.
        Calls feature_fn if provided, otherwise uses extract_features.
        """

        def callback(open_time, snapshot):
            latest_rows = []
            for sym, bars in snapshot.items():
                df = pd.DataFrame.from_records(bars).set_index("open_time")
                if len(df) < self.number_of_input_bars:
                    continue
                feats = (feature_fn or self.extract_features)(
                    df, self.number_of_input_bars, [df.index[-1]]
                )
                if feats.empty:
                    continue
                pred = model.predict(feats)
                print(f"[live] {sym} @ {open_time} → {pred}")

        self._dm.register_batch_callback(callback)
        self._dm.live(self.tickers)

    # ---------- Live data & features ----------
    def get_live_features(self, ticker: str) -> pd.DataFrame:
        """
        Get live features for a single ticker (data-source agnostic).
        
        Uses the shared stand_alone_features_from_1min_bars function to ensure
        identical transformations for live and historical data.
        
        Process:
        1. Fetches recent 1-minute bars from the data manager
        2. Uses shared function with live_mode=True to:
           - Resample to workflow interval with live alignment
           - Extract features from resampled data
        
        Args:
            ticker: Symbol to fetch features for
            
        Returns:
            DataFrame with features for the most recent complete bar
            
        Raises:
            ValueError: If not enough historical data or no features extracted
        """
        # Calculate how much historical data we need (in hours)
        # We need enough full interval bars for features plus a small buffer for
        # live-bar alignment and incomplete-minute trimming.
        bars_per_hour = self._parse_interval_to_bars_per_hour(self.interval)
        required_hours = self.number_of_input_bars / bars_per_hour
        hours_back = int(np.ceil(required_hours)) + 2
        
        # Fetch 1-minute bars from data manager (works for both Allora and Binance)
        df_1min_pandas = self._dm.get_live_1min_data(ticker, hours_back=hours_back)
        
        if df_1min_pandas.empty:
            raise ValueError(f"No 1-minute data returned for {ticker}")
        
        # Convert to polars for shared pipeline
        df_1min_polars = pl.from_pandas(df_1min_pandas.reset_index())
        
        # Use shared function with live_mode=True
        df_with_features = self.stand_alone_features_from_1min_bars(
            df_1min_polars, 
            live_mode=True
        )
        
        # Check if we have enough bars
        if df_with_features.height < self.number_of_input_bars:
            raise ValueError(
                f"Not enough historical data. Need {self.number_of_input_bars} bars, "
                f"got {df_with_features.height}"
            )
        
        # Get the last row (most recent bar) and extract just the features
        last_row = df_with_features.tail(1)
        
        # Extract feature columns
        feature_cols = [col for col in last_row.columns if col.startswith("feature_")]
        
        if len(feature_cols) == 0:
            raise ValueError("No features returned.")
        
        # Convert to pandas and return just features with open_time as index
        features_pandas = last_row.select(["open_time"] + feature_cols).to_pandas()
        features_pandas = features_pandas.set_index("open_time")

        # Attach latest resampled close as context for inference callers
        # (keeps API backward-compatible while avoiding separate local-data reads).
        features_pandas.attrs["current_price"] = float(last_row["close"][0])

        return features_pandas

    def stand_alone_features_from_1min_bars(
        self, 
        df_1min_polars: pl.DataFrame, 
        live_mode: bool = False
    ) -> pl.DataFrame:
        """
        Shared function to extract features from 1-minute bars.
        
        This is the single source of truth for feature extraction, used by both:
        - get_live_features (with live_mode=True)
        - get_full_feature_target_dataframe (with live_mode=False)
        
        Args:
            df_1min_polars: Polars DataFrame with 1-minute OHLCV data
            live_mode: If True, aligns resampling to end at last bar
            
        Returns:
            Polars DataFrame with resampled OHLCV data and extracted features
        """
        # Resample to workflow interval
        df = self.resample_ohlcv_polars(
            df_1min_polars, 
            freq=self.interval, 
            live_mode=live_mode
        )
        
        # Normalize datetime precision to microseconds for consistent joins
        # This prevents precision mismatch errors (ns vs μs) when joining dataframes
        df = df.with_columns([
            pl.col("open_time").cast(pl.Datetime("us", "UTC"))
        ])
        
        # Extract features for all timestamps
        features = self.extract_features_polars(
            df, 
            self.number_of_input_bars, 
            df["open_time"].to_list()
        )
        
        # Join features back to resampled data
        df = df.join(features, on="open_time", how="left")
        return df

    # ---------- Historical features/targets ----------
    def get_full_feature_target_dataframe(self, start_date=None, end_date=None) -> pl.DataFrame:
        print(f"[workflow] Loading data")
        raw = self._dm.load_polars(self.tickers, start=start_date, end=end_date)

        datasets = []
        for t in self.tickers:
            df = raw.filter(pl.col("symbol") == t)
            print(f"[workflow] Processing {t} ({df.height} rows)")
            
            # Skip tickers with no data
            if df.height == 0:
                print(f"[workflow] Skipping {t} - no data available")
                continue

            minute_data = df
            df = self.stand_alone_features_from_1min_bars(df, live_mode=False)

            if self.target_type == "volatility":
                df = self.compute_volatility_target_polars(df, self.target_bars)
            elif self.target_type == "triple_barrier":
                df = self.compute_triple_barrier_target_polars(df, self.target_bars, minute_data)
            else:
                df = self.compute_target_polars(df, self.target_bars)

            df = df.with_columns([pl.lit(t).alias("ticker")])
            datasets.append(df)

        full_data = pl.concat(datasets).sort(["ticker", "open_time"])
        return full_data.to_pandas().set_index(["ticker", "open_time"])

    def resample_ohlcv_polars(
        self,
        df: pl.DataFrame,
        time_col: str = "open_time",
        freq: str = "5m",
        ohlcv_cols: dict = None,
        groupby: list = None,
        live_mode: bool = False,
    ) -> pl.DataFrame:
        """
        General OHLCV resampling for polars DataFrames.
        
        Args:
            df: Polars DataFrame with OHLCV data
            time_col: Name of the timestamp column
            freq: Polars duration string, e.g. '5m', '1h', '1d'
            ohlcv_cols: Dict mapping OHLCV column names to aggregation functions
            groupby: List of columns to group by (e.g. ['symbol'])
            live_mode: If True, drop incomplete bars and align to end at last timestamp
        
        Returns:
            Polars DataFrame with resampled OHLCV data
        """
        if ohlcv_cols is None:
            ohlcv_cols = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        if groupby is None:
            groupby = []

        # Live mode: drop incomplete 1-min bars
        if live_mode:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            last_ts = df[time_col].max()
            
            # Drop if last 1-min bar is incomplete (current second < 45)
            if last_ts > now or (last_ts.minute == now.minute and last_ts.hour == now.hour and now.second < 45):
                df = df.filter(pl.col(time_col) < last_ts)
            
            if df.is_empty():
                return df

        # Calculate offset for live mode
        # Shift timestamps forward, resample, then shift back to align end at last bar
        offset_minutes = 0
        if live_mode and freq.endswith('m'):
            last_ts = df[time_col].max()
            minute = last_ts.minute
            interval_minutes = int(freq[:-1])
            
            # Calculate offset: (last_minute + 1) % interval_minutes
            offset_minutes = (minute + 1) % interval_minutes

        # Apply offset: shift origin backwards, truncate, shift back
        # This replicates pandas resample(offset=...) behavior
        if offset_minutes != 0:
            df = df.with_columns([
                pl.col(time_col).dt.offset_by(f"-{offset_minutes}m").alias("shifted_time")
            ])
            df = df.with_columns([
                pl.col("shifted_time").dt.truncate(freq).alias("resample_time")
            ])
            df = df.with_columns([
                pl.col("resample_time").dt.offset_by(f"{offset_minutes}m").alias("resample_time")
            ])
            df = df.drop("shifted_time")
        else:
            # No offset needed
            df = df.with_columns([
                pl.col(time_col).dt.truncate(freq).alias("resample_time")
            ])

        # Group by resample_time and any additional columns (e.g. symbol)
        gb_cols = ["resample_time"] + groupby

        agg_exprs = [getattr(pl.col(col), func)().alias(col) for col, func in ohlcv_cols.items()]
        out = df.group_by(gb_cols).agg(agg_exprs).sort(gb_cols)
        
        # Rename resample_time back to the original time column name
        out = out.rename({"resample_time": time_col})
        return out

    def compute_target_polars(self, df: pl.DataFrame, target_bars: int) -> pl.DataFrame:
        """
        Compute log return to future close for polars DataFrame.
        
        Args:
            df: Polars DataFrame with OHLCV data (sorted by time)
            target_bars: Number of bars ahead to predict
            
        Returns:
            Polars DataFrame with 'future_close' and 'target' columns added
        """
        df = df.with_columns([
            pl.col("close").shift(-target_bars).alias("future_close")
        ])
        df = df.with_columns([
            (pl.col("future_close").log() - pl.col("close").log()).alias("target")
        ])
        return df

    def compute_volatility_target_polars(
        self, df: pl.DataFrame, target_bars: int
    ) -> pl.DataFrame:
        """
        Compute realised volatility target: the standard deviation of consecutive
        1-minute log returns over the next *target_bars* bars, scaled to the
        full-horizon volatility using the square-root-of-time rule.

        Definition:
            For each row at time t, let r_i = log(close[t+i] / close[t+i-1])
            for i in 1..target_bars.
            target[t] = std(r_1, ..., r_{target_bars}) × √target_bars

        The √target_bars scaling converts the per-bar (1-minute) return std
        into the volatility of the full horizon return.  If 1-minute returns
        have std σ₁, the std of the T-minute return is σ₁·√T (assuming
        uncorrelated returns).  This matches the ground-truth definition used
        by the Allora volatility reputer (domain/volatility.py), which applies
        ``standardization_ratio = √(timeframe / frequency)``.

        Args:
            df: Polars DataFrame with OHLCV data sorted by time.  Must be at
                1-minute resolution for the result to be meaningful.
            target_bars: Number of forward bars defining the volatility window.

        Returns:
            Polars DataFrame with 'target' column added.  Rows where the full
            forward window is unavailable will have null targets.
        """
        import math

        # Compute per-bar log returns: log(close[t] / close[t-1])
        log_returns = (pl.col("close").log() - pl.col("close").shift(1).log()).alias(
            "_log_return"
        )
        df = df.with_columns([log_returns])

        # Rolling std over the *next* target_bars log returns.
        # Strategy: shift the log_return column backwards by 1 so that row t
        # sees returns from t+1..t+target_bars, then apply a forward-looking
        # rolling window.  Polars rolling_std is backward-looking, so we reverse
        # the column, apply rolling_std, then reverse back.
        lr = df["_log_return"].shift(-1)  # align: row t now holds return at t+1

        # Reverse, apply backward rolling std, reverse back → forward rolling std
        lr_reversed = lr.reverse()
        vol_reversed = lr_reversed.rolling_std(window_size=target_bars, min_samples=target_bars)
        vol = vol_reversed.reverse()

        # Scale by √target_bars to convert per-bar std to horizon volatility.
        # This matches the reputer's standardization_ratio = √(timeframe/frequency).
        scaling = math.sqrt(target_bars)
        df = df.with_columns([(vol * scaling).alias("target")])
        df = df.drop("_log_return")
        return df
    
    def compute_triple_barrier_target_polars(
        self, df: pl.DataFrame, target_bars: int, minute_data: pl.DataFrame,
        barrier_multiplier: float = 0.25,
    ) -> pl.DataFrame:
        """Append one-hot first-touch targets without changing feature construction.

        A row's prediction time is its opening time plus ``self.interval``.
        ATR is the mean of trailing ``target_bars`` high/low log ranges over
        100 horizons, calculated on the already-resampled candles. Each range
        includes both endpoints (h + 1 bars), matching the reputer SQL. The
        averaging interval excludes its right endpoint; ranges at its start
        use only candles inside that interval. Missing minute coverage leaves all
        target columns null. Minute ties resolve down first.

        Metadata uses the ``tb_`` prefix and must not be used as model features.
        ``tb_exit_time`` is the end of the first-hit minute (or expiry), the
        earliest time that candle's high/low/close is observable.
        """
        if isinstance(target_bars, bool) or int(target_bars) != target_bars or target_bars < 1:
            raise ValueError("target_bars must be a positive integer")
        if not np.isfinite(barrier_multiplier) or barrier_multiplier <= 0:
            raise ValueError("barrier_multiplier must be positive and finite")
        h = int(target_bars)
        minutes_per_bar = int(round(60 / self._parse_interval_to_bars_per_hour(self.interval)))
        if minutes_per_bar < 1:
            raise ValueError("triple barrier requires bars of at least one minute")
        width = pd.Timedelta(minutes=minutes_per_bar)
        horizon = h * width
        raw = minute_data.select("open_time", "open", "high", "low", "close").to_pandas()
        raw = raw.sort_values("open_time").set_index("open_time")
        if raw.index.has_duplicates:
            raise ValueError("Duplicate minute timestamps in triple-barrier input")
        if raw.empty:
            raise ValueError("No minute candles for triple-barrier targets")
        if not (raw.index == raw.index.floor("min")).all():
            raise ValueError("Minute candle timestamps must be minute-aligned")
        grid = pd.date_range(raw.index.min(), raw.index.max(), freq="min")
        raw = raw.reindex(grid)
        valid = np.isfinite(raw).all(axis=1) & (raw > 0).all(axis=1)
        valid &= (raw.high >= raw[['open', 'close', 'low']].max(axis=1))
        valid &= (raw.low <= raw[['open', 'close', 'high']].min(axis=1))
        # Resampled prices are the workflow's native bars; minute counts validate
        # them instead of silently treating a partial bar as complete.
        native = df.select("open_time", "high", "low").to_pandas().set_index("open_time")
        native = native.reindex(pd.date_range(native.index.min(), native.index.max(), freq=width))
        counts = valid.astype(int).resample(width).sum().reindex(native.index, fill_value=0)
        native.loc[counts != minutes_per_bar, ['high', 'low']] = np.nan
        lookback = 100 * h
        # SQL RANGE BETWEEN horizon PRECEDING AND CURRENT ROW includes h+1
        # candle openings. WHERE clips history BEFORE the window calculation,
        # so the first h ranges of each lookback have lengths 1, ..., h.
        ranges = (np.log(native.high.rolling(h + 1, min_periods=1).max())
                  - np.log(native.low.rolling(h + 1, min_periods=1).min()))
        range_sum = ranges.rolling(lookback, min_periods=lookback).sum()
        for j in range(h):
            partial = (np.log(native.high.rolling(j + 1, min_periods=1).max())
                       - np.log(native.low.rolling(j + 1, min_periods=1).min()))
            range_sum += (partial - ranges).shift(lookback - 1 - j)
        # A row opens at T-width. Samples must open before T-width.
        atr = (range_sum / lookback).shift(1)
        # Also require every sample candle in the historical averaging interval.
        complete = (counts == minutes_per_bar).rolling(100 * h, min_periods=100 * h).sum().shift(1)
        atr = atr.where(complete == 100 * h)
        bad_prefix = np.r_[0, np.cumsum(~valid.to_numpy())]
        lows, highs, closes = (raw[c].to_numpy() for c in ('low', 'high', 'close'))
        first = grid[0]
        records = []
        for opening in df['open_time'].to_list():
            t = pd.Timestamp(opening) + width
            end = t + horizon - pd.Timedelta(minutes=1)
            rec = dict(target_up=None, target_neutral=None, target_down=None,
                       tb_prediction_time=t, tb_resolution_time=end,
                       tb_atr=None, tb_base=None, tb_upper=None, tb_lower=None,
                       tb_exit_time=None, tb_exit_price=None, tb_exit_reason=None)
            a = atr.get(pd.Timestamp(opening), np.nan)
            start_i = int((t - pd.Timedelta(minutes=1) - first) / pd.Timedelta(minutes=1))
            end_i = start_i + h * minutes_per_bar
            if (not np.isfinite(a) or start_i < 0 or end_i > len(raw)
                    or bad_prefix[end_i] != bad_prefix[start_i]):
                records.append(rec)
                continue
            base = closes[start_i]
            upper, lower = base * np.exp(barrier_multiplier * a), base * np.exp(-barrier_multiplier * a)
            down = lows[start_i:end_i] <= lower
            up = highs[start_i:end_i] >= upper
            hits = np.flatnonzero(down | up)
            if len(hits):
                j = int(hits[0])
                label = 'down' if down[j] else 'up'
                exit_price = lower if label == 'down' else upper
                exit_time = grid[start_i + j] + pd.Timedelta(minutes=1)
            else:
                label, exit_price, exit_time = 'neutral', closes[end_i - 1], end
            rec.update({f'target_{c}': int(c == label) for c in ('up', 'neutral', 'down')})
            rec.update(tb_atr=float(a), tb_base=float(base), tb_upper=float(upper), tb_lower=float(lower),
                       tb_exit_time=exit_time, tb_exit_price=float(exit_price),
                       tb_exit_reason='expiry' if label == 'neutral' else label)
            records.append(rec)
        metadata = pl.from_pandas(pd.DataFrame(records)).with_columns(
            pl.col('target_up', 'target_neutral', 'target_down').cast(pl.Int8),
            pl.col('tb_atr', 'tb_base', 'tb_upper', 'tb_lower', 'tb_exit_price').cast(pl.Float64),
            pl.col('tb_prediction_time', 'tb_resolution_time', 'tb_exit_time').cast(pl.Datetime('us', 'UTC')),
            pl.col('tb_exit_reason').cast(pl.String),
        )
        return df.hstack(metadata)

    def extract_features_polars(
        self,
        df: pl.DataFrame,
        number_of_input_bars: int,
        start_times: list,
        time_col: str = "open_time",
    ) -> pl.DataFrame:
        """
        Numba-optimized version of feature extraction (polars).
        Extracts features from already resampled OHLCV data.
        Returns a polars DataFrame indexed by start_times.
        
        Args:
            df: Polars DataFrame with resampled OHLCV data
            number_of_input_bars: Number of bars to use as lookback window
            start_times: List of timestamps to extract features for
            time_col: Name of the time column
            
        Returns:
            Polars DataFrame with extracted features
        """
        ts_index = df[time_col].cast(pl.Int64).to_numpy()
        data_values = df.select(["open", "high", "low", "close", "volume"]).to_numpy()
        
        # Pre-convert timestamps to integers
        start_times_int = np.array([
            int(T.timestamp() * 1_000_000) if hasattr(T, 'timestamp') 
            else int(pd.Timestamp(T).value / 1000)
            for T in start_times
        ], dtype=np.int64)
        
        # Call JIT-compiled function (first call includes compilation time)
        features_array, valid_indices = _extract_features_numba(
            ts_index, data_values, start_times_int, number_of_input_bars
        )
        
        columns = [
            f"feature_{f}_{i}"
            for i in range(number_of_input_bars)
            for f in ["open", "high", "low", "close", "volume"]
        ]
        
        if len(features_array) == 0:
            empty_df = pl.DataFrame({col: [] for col in columns})
            empty_df = empty_df.with_columns([
                pl.Series(time_col, [], dtype=pl.Datetime("us", "UTC"))
            ])
            return empty_df
        
        out_df = pl.DataFrame(features_array, schema=columns)
        valid_times = [start_times[i] for i in valid_indices]
        out_df = out_df.with_columns([
            pl.Series(time_col, valid_times, dtype=pl.Datetime("us", "UTC"))
        ])
        return out_df