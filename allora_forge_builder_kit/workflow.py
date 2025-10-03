import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import polars as pl

from .data_manager import DataManager

class AlloraMLWorkflow:
    def __init__(
        self,
        tickers,
        hours_needed,
        number_of_input_candles,
        target_length,
        interval="5m",
        market="futures",
        base_dir="parquet_data",
    ):
        """
        High-level ML workflow built on top of DataManager.
        DataManager handles ingestion/storage, but stays internal.
        """
        self.tickers = tickers
        self.hours_needed = hours_needed
        self.number_of_input_candles = number_of_input_candles
        self.target_length = target_length
        self.test_targets = None
        self.interval = interval

        # 🔹 hide DataManager internally
        self._dm = DataManager(interval=interval, market=market, base_dir=base_dir)
    
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
        Calls feature_fn if provided, otherwise uses extract_rolling_daily_features.
        """

        def callback(open_time, snapshot):
            latest_rows = []
            for sym, bars in snapshot.items():
                df = pd.DataFrame.from_records(bars).set_index("open_time")
                if len(df) < self.hours_needed * 12:
                    continue
                feats = (feature_fn or self.extract_rolling_daily_features)(
                    df, self.hours_needed, self.number_of_input_candles, [df.index[-1]]
                )
                if feats.empty:
                    continue
                pred = model.predict(feats)
                print(f"[live] {sym} @ {open_time} → {pred}")

        self._dm.register_batch_callback(callback)
        self._dm.live(self.tickers)

    # ---------- Feature engineering ----------
    def create_5_min_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.reset_index().set_index("open_time").sort_index()
        return df[["open", "high", "low", "close", "volume"]].dropna()

    def compute_target(self, df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
        df["future_close"] = df["close"].shift(freq=f"-{hours}h")
        df["target"] = np.log(df["future_close"]) - np.log(df["close"])
        return df

    def extract_rolling_daily_features(
        self, data: pd.DataFrame, lookback: int, number_of_candles: int, start_times: list
    ) -> pd.DataFrame:
        ts_index = data.index.to_numpy()
        data_values = data[["open", "high", "low", "close", "volume"]].to_numpy()
        features_list, index_list = [], []
        # Use actual bars per hour based on current interval
        bars_per_hour = self._parse_interval_to_bars_per_hour(self.interval)
        candle_length = int(lookback * bars_per_hour)

        for T in start_times:
            pos = np.searchsorted(ts_index, T, side="right")
            if pos - candle_length < 0:
                continue
            window = data_values[pos - candle_length:pos]
            try:
                reshaped = window.reshape(number_of_candles, -1, 5)
            except ValueError:
                continue
            open_ = reshaped[:, 0, 0]
            high_ = reshaped[:, :, 1].max(axis=1)
            low_ = reshaped[:, :, 2].min(axis=1)
            close_ = reshaped[:, -1, 3]
            volume_ = reshaped[:, :, 4].sum(axis=1)
            last_close = close_[-1]
            last_volume = volume_[-1]
            if last_close == 0 or np.isnan(last_close) or last_volume == 0 or np.isnan(last_volume):
                continue
            features = np.stack([open_, high_, low_, close_, volume_], axis=1)
            features[:, :4] /= last_close
            features[:, 4] /= last_volume
            features_list.append(features.flatten())
            index_list.append(T)

        if not features_list:
            return pd.DataFrame(columns=[
                f"feature_{f}_{i}"
                for i in range(number_of_candles)
                for f in ["open", "high", "low", "close", "volume"]
            ])

        features_array = np.vstack(features_list)
        columns = [
            f"feature_{f}_{i}"
            for i in range(number_of_candles)
            for f in ["open", "high", "low", "close", "volume"]
        ]
        return pd.DataFrame(features_array, index=index_list, columns=columns)

    # ---------- Historical features/targets ----------
    def get_full_feature_target_dataframe(self, start_date=None, end_date=None) -> pl.DataFrame:
        print(f"[workflow] Loading data")
        raw = self._dm.load_polars(self.tickers, start=start_date, end=end_date)  # You need to implement load_polars

        datasets = []
        for t in self.tickers:
            df = raw.filter(pl.col("symbol") == t)
            print(f"[workflow] Processing {t} ({df.height} rows)")
            
            # Skip tickers with no data
            if df.height == 0:
                print(f"[workflow] Skipping {t} - no data available")
                continue

            df = self.resample_ohlcv_polars(df, freq=self._dm.interval)
            df = self.compute_target_polars(df, self.target_length)
            features = self.extract_rolling_daily_features_polars(
                df, self.hours_needed, self.number_of_input_candles, df["open_time"].to_list()
            )
            df = df.join(features, on="open_time", how="left")
            df = df.with_columns([pl.lit(t).alias("ticker")])
            datasets.append(df)

        full_data = pl.concat(datasets).sort(["ticker", "open_time"])
        # If you need a MultiIndex, you can convert to pandas at the end:
        return full_data.to_pandas().set_index(["ticker", "open_time"])
        # return full_data.drop_nulls()

    def resample_ohlcv_polars(
        self,
        df: pl.DataFrame,
        time_col: str = "open_time",
        freq: str = "5m",
        ohlcv_cols: dict = None,
        groupby: list = None,
    ) -> pl.DataFrame:
        """
        General OHLCV resampling for polars DataFrames.
        freq: polars duration string, e.g. '5m', '1h', '1d'
        ohlcv_cols: dict mapping OHLCV column names to their aggregation functions
        groupby: list of columns to group by (e.g. ['symbol'])
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

        # Floor timestamps to the desired frequency
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

    def compute_target_polars(self, df: pl.DataFrame, hours: int = 24) -> pl.DataFrame:
        """
        Compute log return to future close for polars DataFrame.
        Assumes 'open_time' is sorted and of type datetime, and 'close' is present.
        """
        # Calculate the number of rows to shift forward (future)
        # Use actual bars per hour based on current interval
        bars_per_hour = self._parse_interval_to_bars_per_hour(self.interval)
        bars_ahead = int(hours * bars_per_hour)
        df = df.with_columns([
            pl.col("close").shift(-bars_ahead).alias("future_close")
        ])
        df = df.with_columns([
            (pl.col("future_close").log() - pl.col("close").log()).alias("target")
        ])
        return df
    
    def extract_rolling_daily_features_polars(
        self,
        df: pl.DataFrame,
        lookback: int,
        number_of_candles: int,
        start_times: list,
        time_col: str = "open_time",
    ) -> pl.DataFrame:
        """
        Polars version of rolling daily feature extraction.
        Returns a polars DataFrame indexed by start_times.
        """
        # Convert time column to Unix timestamps (int64) for timezone-safe comparison
        ts_index = df[time_col].cast(pl.Int64).to_numpy()
        data_values = df.select(["open", "high", "low", "close", "volume"]).to_numpy()
        features_list, index_list = [], []
        # Use actual bars per hour based on current interval
        bars_per_hour = self._parse_interval_to_bars_per_hour(self.interval)
        candle_length = int(lookback * bars_per_hour)

        for T in start_times:
            # Convert search timestamp to Unix timestamp for comparison
            if hasattr(T, 'timestamp'):
                T_int = int(T.timestamp() * 1_000_000)  # microseconds
            else:
                T_int = int(pd.Timestamp(T).value / 1000)  # microseconds
            pos = np.searchsorted(ts_index, T_int, side="right")
            if pos - candle_length < 0:
                continue
            window = data_values[pos - candle_length:pos]
            try:
                reshaped = window.reshape(number_of_candles, -1, 5)
            except ValueError:
                continue
            open_ = reshaped[:, 0, 0]
            high_ = reshaped[:, :, 1].max(axis=1)
            low_ = reshaped[:, :, 2].min(axis=1)
            close_ = reshaped[:, -1, 3]
            volume_ = reshaped[:, :, 4].sum(axis=1)
            last_close = close_[-1]
            last_volume = volume_[-1]
            if last_close == 0 or np.isnan(last_close) or last_volume == 0 or np.isnan(last_volume):
                continue
            features = np.stack([open_, high_, low_, close_, volume_], axis=1)
            features[:, :4] /= last_close
            features[:, 4] /= last_volume
            features_list.append(features.flatten())
            index_list.append(T)

        columns = [
            f"feature_{f}_{i}"
            for i in range(number_of_candles)
            for f in ["open", "high", "low", "close", "volume"]
        ]
        if not features_list:
            # Create empty DataFrame with proper datetime type for time column
            empty_df = pl.DataFrame({col: [] for col in columns})
            empty_df = empty_df.with_columns([
                pl.Series(time_col, [], dtype=pl.Datetime("us", "UTC"))
            ])
            return empty_df
        features_array = np.vstack(features_list)
        out_df = pl.DataFrame(features_array, schema=columns)
        out_df = out_df.with_columns([
            pl.Series(time_col, index_list)
        ])
        return out_df