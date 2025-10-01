import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import dill
import os

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

        # 🔹 hide DataManager internally
        self._dm = DataManager(interval=interval, market=market, base_dir=base_dir)

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
        candle_length = lookback * 12  # 12 bars per hour for 5m

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
    def get_full_feature_target_dataframe(self, start_date=None, end_date=None) -> pd.DataFrame:
        datasets = []
        print(f"[workflow] Loading data")
        raw = self.load_raw(start=start_date, end=end_date)

        for t in self.tickers:
            df = raw.loc[raw.index.get_level_values(0) == t].copy()
            print(f"[workflow] Processing {t} ({len(df)} rows)")

            df = self.create_5_min_bars(df)
            df = self.compute_target(df, self.target_length)
            features = self.extract_rolling_daily_features(
                df, self.hours_needed, self.number_of_input_candles, df.index.tolist()
            )
            df = df.join(features)
            df["ticker"] = t
            datasets.append(df)

        full_data = pd.concat(datasets).sort_index()
        full_data.index = pd.MultiIndex.from_frame(
            full_data.reset_index()[["ticker", "open_time"]]
        )
        full_data.index.names = ["symbol", "open_time"]

        return full_data.dropna()
