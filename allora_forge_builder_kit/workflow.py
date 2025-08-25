import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
import dill

class AlloraMLWorkflow:
    def __init__(self, data_api_key, tickers, hours_needed, number_of_input_candles, target_length):
        self.api_key = data_api_key
        self.tickers = tickers
        self.hours_needed = hours_needed  # For input window
        self.number_of_input_candles = number_of_input_candles
        self.target_length = target_length  # Target horizon in hours
        self.test_targets = None

    def compute_from_date(self, extra_hours: int = 12) -> str:
        total_hours = self.hours_needed + extra_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=total_hours)
        return cutoff_time.strftime("%Y-%m-%d")

    def list_ready_buckets(self, ticker, from_month):
        url = "https://api.allora.network/v2/allora/market-data/ohlc/buckets/by-month"
        headers = {"x-api-key": self.api_key}
        resp = requests.get(url, headers=headers, params={"tickers": ticker, "from_month": from_month}, timeout=30)
        resp.raise_for_status()
        buckets = resp.json()["data"]["data"]
        return [b for b in buckets if b["state"] == "ready"]

    def fetch_bucket_csv(self, download_url):
        return pd.read_csv(download_url)

    def fetch_ohlcv_data(self, ticker, from_date: str, max_pages: int = 1000, sleep_sec: float = 0.1) -> pd.DataFrame:
        url = "https://api.allora.network/v2/allora/market-data/ohlc"
        headers = {"x-api-key": self.api_key}
        params = {"tickers": ticker, "from_date": from_date}

        all_data = []
        pages_fetched = 0

        while pages_fetched < max_pages:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            payload = response.json()
            if not payload.get("status", False):
                raise RuntimeError("API responded with an error status.")

            all_data.extend(payload["data"]["data"])

            token = payload["data"].get("continuation_token")
            if not token:
                break

            params["continuation_token"] = token
            pages_fetched += 1
            time.sleep(sleep_sec)

        df = pd.DataFrame(all_data)
        if df.empty:
            raise ValueError("No data returned from API.")

        for col in ["open", "high", "low", "close", "volume", "volume_notional"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        return df

    def create_5_min_bars(self, df: pd.DataFrame, live_mode: bool = False) -> pd.DataFrame:
        df = df.set_index("date").sort_index()
        # print("Raw 1-min timestamps:", df.index[-10:])  # Show last 10 timestamps for debugging

        if not live_mode:
            bars = df.resample("5min").apply({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "trades_done": "sum"
            })
        else:
            last_ts = df.index[-1]
            now = datetime.utcnow().replace(tzinfo=last_ts.tzinfo)
            if last_ts > now:
                # print(f"Dropping incomplete 1-min bar at {last_ts} (future timestamp)")
                df = df.iloc[:-1]
            else:
                # Drop the last bar if the current time in seconds is < 45
                if last_ts.minute == now.minute and last_ts.hour == now.hour and now.second < 45:
                    # print(f"Dropping incomplete 1-min bar at {last_ts} (current second: {now.second})")
                    df = df.iloc[:-1]
            last_ts = df.index[-1]
            minute = last_ts.minute
            offset_minutes = (minute + 1) % 5
            offset = f"{offset_minutes}min" if offset_minutes != 0 else "0min"
            # print(f"Live mode: last minute={minute}, offset_minutes={offset_minutes}, offset={offset}")
            bars = df.resample("5min", offset=offset).apply({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "trades_done": "sum"
            })
            print("Live Mode Bars:  ", bars.tail(5))

        # print("5-min bar timestamps:", bars.index[-10:])  # Show last 10 bar timestamps for debugging
        return bars

    def compute_target(self, df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
        df["future_close"] = df["close"].shift(freq=f"-{hours}h")
        df["target"] = np.log(df["future_close"]) - np.log(df["close"])
        return df

    def extract_rolling_daily_features(
        self, data: pd.DataFrame, lookback: int, number_of_candles: int, start_times: list
    ) -> pd.DataFrame:
        # Convert index to array for quick lookup
        ts_index = data.index.to_numpy()
        data_values = data[["open", "high", "low", "close", "volume", "trades_done"]].to_numpy()
        features_list = []
        index_list = []
    
        candle_length = lookback * 12  # 12 points per hour if 5min bars
    
        for T in start_times:
            # Find the last index <= T
            pos = np.searchsorted(ts_index, T, side="right")
            if pos - candle_length < 0:
                continue
    
            window = data_values[pos - candle_length:pos]
    
            # Group window into number_of_candles equal chunks
            try:
                reshaped = window.reshape(number_of_candles, -1, 6)
            except ValueError:
                continue  # Skip if window can't be reshaped
    
            open_ = reshaped[:, 0, 0]
            high_ = reshaped[:, :, 1].max(axis=1)
            low_ = reshaped[:, :, 2].min(axis=1)
            close_ = reshaped[:, -1, 3]
            volume_ = reshaped[:, :, 4].sum(axis=1)
            trades_ = reshaped[:, :, 5].sum(axis=1)
    
            last_close = close_[-1]
            last_volume = volume_[-1]
            if last_close == 0 or np.isnan(last_close) or last_volume == 0 or np.isnan(last_volume):
                continue
    
            features = np.stack([open_, high_, low_, close_, volume_, trades_], axis=1)
            features[:, :4] /= last_close  # Normalize OHLC
            features[:, 4] /= last_volume  # Normalize volume
    
            features_list.append(features.flatten())
            index_list.append(T)
    
        if not features_list:
            return pd.DataFrame(columns=[
                f"feature_{f}_{i}" for i in range(number_of_candles) for f in ["open", "high", "low", "close", "volume", "trades_done"]
            ])
    
        features_array = np.vstack(features_list)
        columns = [f"feature_{f}_{i}" for i in range(number_of_candles) for f in ["open", "high", "low", "close", "volume", "trades_done"]]
        return pd.DataFrame(features_array, index=index_list, columns=columns)

    def get_live_features(self, ticker):
        from_date = self.compute_from_date()
        df = self.fetch_ohlcv_data(ticker, from_date)
        five_min_bars = self.create_5_min_bars(df, live_mode=True)
        if len(five_min_bars) < self.hours_needed * 12:
            raise ValueError("Not enough historical data.")
        live_time = five_min_bars.index[-2]
        features = self.extract_rolling_daily_features(five_min_bars, self.hours_needed, self.number_of_input_candles, [live_time])
        if features.empty:
            raise ValueError("No features returned.")
        return features

    def evaluate_test_data(self, predictions: pd.Series) -> dict:
        if self.test_targets is None:
            raise ValueError("Test targets not set. Run get_train_validation_test_data first.")

        if not predictions.index.equals(self.test_targets.index):
            raise ValueError("Prediction index must match test target index.")

        y_true = self.test_targets
        y_pred = predictions

        corr = np.corrcoef(y_true, y_pred)[0, 1]
        directional_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

        return {
            "correlation": corr,
            "directional_accuracy": directional_accuracy
        }

    def get_full_feature_target_dataframe(self, from_month="2025-01") -> pd.DataFrame:
        """
        Returns a DataFrame containing all features and target values for all tickers,
        with a MultiIndex of (date, ticker). Does not split into training/validation.
        """
        all_data = {}
        for t in self.tickers:
            print(f"Downloading Historical Data for {t}")
            frames = []
            for bucket in self.list_ready_buckets(t, from_month):
                df = self.fetch_bucket_csv(bucket["download_url"])
                df["bucket_start"] = bucket["start"]
                df["bucket_end"] = bucket["end"]
                df["availability"] = bucket["availability"]
                frames.append(df)
            combined_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not combined_df.empty:
                latest_ts = sorted(pd.to_datetime(combined_df["date"]).dt.date.unique())[-2]
                live_df = self.fetch_ohlcv_data(t, latest_ts.strftime("%Y-%m-%d"))
                combined_df = pd.concat([combined_df, live_df], ignore_index=True)
                combined_df["date"] = pd.to_datetime(combined_df["date"])
                combined_df = combined_df.drop_duplicates(subset="date")
            all_data[t] = combined_df
    
        datasets = []
        for t in self.tickers:
            print(f"Processing 5-minute bars for {t}")
            df = self.create_5_min_bars(all_data[t])
            df = self.compute_target(df, self.target_length)
            features = self.extract_rolling_daily_features(
                df, self.hours_needed, self.number_of_input_candles, df.index.tolist()
            )
            df = df.join(features)
            df["ticker"] = t
            datasets.append(df)
    
        full_data = pd.concat(datasets).sort_index()
        full_data.index = pd.MultiIndex.from_frame(full_data.reset_index()[["date", "ticker"]])
        full_data = full_data.dropna()  # Drop rows with missing values
    
        return full_data

    def get_train_validation_test_data(self, from_month="2025-01", validation_months=3, test_months=3, force_redownload=False):
        def generate_filename():
            """Generate a unique filename based on parameters."""
            tickers_str = "_".join(self.tickers)
            return (
                f"data_{tickers_str}_{from_month}_val{validation_months}_test{test_months}"
                f"_candles{self.number_of_input_candles}.pkl"
            )

        def save_to_disk(data, filename):
            """Save data to disk."""
            with open(filename, "wb") as f:
                dill.dump(data, f)

        def load_from_disk(filename):
            """Load data from disk."""
            with open(filename, "rb") as f:
                X_train, y_train, X_val, y_val, X_test, y_test = dill.load(f)
            self.test_targets = y_test
            return X_train, y_train, X_val, y_val, X_test, y_test

        # Generate the filename
        filename = generate_filename()

        # Check if the file exists and load it if not forcing a redownload
        if os.path.exists(filename) and not force_redownload:
            print(f"Loading data from {filename}")
            return load_from_disk(filename)

        # If file doesn't exist or force_redownload is True, proceed with data preparation
        all_data = {}
        for t in self.tickers:
            print(f"Downloading Historical Data for {t}")
            frames = []
            for bucket in self.list_ready_buckets(t, from_month):
                df = self.fetch_bucket_csv(bucket["download_url"])
                df["bucket_start"] = bucket["start"]
                df["bucket_end"] = bucket["end"]
                df["availability"] = bucket["availability"]
                frames.append(df)
            combined_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not combined_df.empty:
                latest_ts = sorted(pd.to_datetime(combined_df["date"]).dt.date.unique())[-2]
                live_df = self.fetch_ohlcv_data(t, latest_ts.strftime("%Y-%m-%d"))
                combined_df = pd.concat([combined_df, live_df], ignore_index=True)
                combined_df['date'] = pd.to_datetime(combined_df['date'])
                combined_df = combined_df.drop_duplicates(subset='date')
            all_data[t] = combined_df

        datasets = []
        for t in self.tickers:
            print(f"Processing 5-minute bars for {t}")
            df = self.create_5_min_bars(all_data[t])
            print(f"Computing target")
            df = self.compute_target(df, self.target_length)
            print(f"Extracting features")
            features = self.extract_rolling_daily_features(df, self.hours_needed, self.number_of_input_candles, df.index.tolist())
            df = df.join(features)
            df["ticker"] = t
            datasets.append(df)

        full_data = pd.concat(datasets).sort_index()
        full_data.index = pd.MultiIndex.from_frame(full_data.reset_index()[["date", "ticker"]])
        full_data = full_data.dropna()

        # Define cutoff dates for test, validation, and training sets
        test_cutoff = datetime.utcnow() - pd.DateOffset(months=test_months)
        val_cutoff_start = test_cutoff - timedelta(hours=self.target_length) - pd.DateOffset(months=validation_months)
        val_cutoff_end = test_cutoff - timedelta(hours=self.target_length)
        train_cutoff = val_cutoff_start - timedelta(hours=self.target_length)

        # Create masks for each set
        test_mask = full_data.index.get_level_values("date") >= str(test_cutoff)
        val_mask = (full_data.index.get_level_values("date") >= str(val_cutoff_start)) & \
                   (full_data.index.get_level_values("date") < str(val_cutoff_end))
        train_mask = full_data.index.get_level_values("date") < str(train_cutoff)

        # Store validation targets for evaluation
        self.validation_targets = full_data.loc[val_mask, ["target"]]

        # Split data into train, validation, and test sets
        X_train = full_data.loc[train_mask].drop(columns=["target", "future_close"])
        y_train = full_data.loc[train_mask]["target"]
        X_val = full_data.loc[val_mask].drop(columns=["target", "future_close"])
        y_val = full_data.loc[val_mask]["target"]
        X_test = full_data.loc[test_mask].drop(columns=["target", "future_close"])
        y_test = full_data.loc[test_mask]["target"]

        self.test_targets = y_test

        # Save the prepared data to disk
        print(f"Saving data to {filename}")
        save_to_disk((X_train, y_train, X_val, y_val, X_test, y_test), filename)

        return X_train, y_train, X_val, y_val, X_test, y_test

