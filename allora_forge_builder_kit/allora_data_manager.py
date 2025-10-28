#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests
import pandas as pd
import polars as pl
import numpy as np

from .base_data_manager import BaseDataManager, day_str


# -----------------------------
# AlloraDataManager
# -----------------------------

class AlloraDataManager(BaseDataManager):
    """
    Allora Network market data manager.
    
    Features:
      - Partitioned Parquet storage (inherited)
      - Backfill via monthly buckets + REST API
      - Polling-based live snapshot (no WebSocket)
      - Hot cache of recent bars (inherited)
      - Compatible with AlloraMLWorkflow
    """

    OHLC_API_URL = "https://api.allora.network/v2/allora/market-data/ohlc"
    BUCKETS_API_URL = "https://api.allora.network/v2/allora/market-data/ohlc/buckets/by-month"

    def __init__(
        self,
        api_key: str,
        base_dir: str = "parquet_data_allora",  # Source-specific default
        interval: str = "5m",
        symbols: Optional[List[str]] = None,
        cache_len: int = 1000,
        max_pages: int = 1000,
        sleep_sec: float = 0.1
    ):
        # Initialize base class
        super().__init__(base_dir=base_dir, interval=interval, symbols=symbols, cache_len=cache_len)
        
        # Allora-specific attributes
        self.api_key = api_key
        self.headers = {"x-api-key": api_key}
        self.max_pages = max_pages
        self.sleep_sec = sleep_sec

    # ---------- Helper Methods ----------
    def _interval_to_pandas_freq(self) -> str:
        """
        Convert interval format (e.g., '5m') to pandas frequency string (e.g., '5min').
        Pandas deprecated 'm' for minutes; now requires 'min' or 'T'.
        """
        interval = self.interval
        if interval.endswith('m') and not interval.endswith('min'):
            # Convert '5m' -> '5min'
            return interval[:-1] + 'min'
        return interval
    
    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        """
        Normalize ticker format for Allora API.
        
        Allora expects lowercase tickers without slashes: 'btcusd', 'ethusd', 'solusd'
        This method converts formats like 'BTC/USD' or 'BTCUSD' to 'btcusd'.
        
        Args:
            ticker: Ticker symbol in any format
            
        Returns:
            Normalized ticker (lowercase, no slash)
        """
        return ticker.replace("/", "").replace("-", "").lower()
    
    def _fetch_ohlcv_data(self, ticker: str, from_date: str, max_pages: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from Allora API with pagination.
        Returns 1-minute bars.
        
        Args:
            ticker: Symbol (e.g. "BTC/USD" or "btcusd")
            from_date: Start date in YYYY-MM-DD format
            max_pages: Maximum pages to fetch (defaults to self.max_pages)
            
        Returns:
            DataFrame with 1-minute bars
        """
        # Normalize ticker to Allora format (lowercase, no slash)
        normalized_ticker = self._normalize_ticker(ticker)
        
        max_pages = max_pages or self.max_pages
        params = {"tickers": normalized_ticker, "from_date": from_date}
        all_data = []
        pages_fetched = 0

        while pages_fetched < max_pages:
            try:
                response = requests.get(self.OHLC_API_URL, headers=self.headers, params=params, timeout=30)
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
                time.sleep(self.sleep_sec)
                
            except Exception as e:
                print(f"[Allora API error] {ticker} from {from_date}: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        
        # Parse data types
        for col in ["open", "high", "low", "close", "volume", "volume_notional"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["date"] = pd.to_datetime(df["date"], utc=True)
        
        # Drop exchange_code if present
        if "exchange_code" in df.columns:
            df.drop(columns=['exchange_code'], inplace=True)
            
        return df

    def _list_ready_buckets(self, ticker: str, from_month: str) -> List[dict]:
        """
        List available monthly data buckets for a ticker.
        
        Args:
            ticker: Symbol (e.g. "BTC/USD" or "btcusd")
            from_month: Starting month in YYYY-MM format
            
        Returns:
            List of bucket dicts with download URLs
        """
        # Normalize ticker to Allora format (lowercase, no slash)
        normalized_ticker = self._normalize_ticker(ticker)
        
        params = {"tickers": normalized_ticker, "from_month": from_month}
        try:
            resp = requests.get(self.BUCKETS_API_URL, headers=self.headers, params=params, timeout=30)
            resp.raise_for_status()
            buckets = resp.json()["data"]["data"]
            return [b for b in buckets if b["state"] == "ready"]
        except Exception as e:
            print(f"[Allora buckets error] {ticker}: {e}")
            return []

    def _fetch_bucket_csv(self, download_url: str) -> pd.DataFrame:
        """Download and parse a monthly bucket CSV."""
        try:
            df = pd.read_csv(download_url)
            if 'exchange_code' in df.columns:
                df.drop(columns=['exchange_code'], inplace=True)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            return df
        except Exception as e:
            print(f"[Allora bucket download error] {download_url}: {e}")
            return pd.DataFrame()

    def _create_interval_bars(self, df_1min: pd.DataFrame, live_mode: bool = False) -> pd.DataFrame:
        """
        Aggregate 1-minute bars to the target interval (e.g., 5-min).
        
        Args:
            df_1min: DataFrame with 1-minute bars
            live_mode: If True, drop incomplete bars
            
        Returns:
            DataFrame with aggregated bars
        """
        if df_1min.empty:
            return pd.DataFrame()
            
        df = df_1min.copy()
        df = df.set_index("date").sort_index().dropna()

        if live_mode:
            # Drop last bar if incomplete (same logic as old AlloraMLWorkflow)
            last_ts = df.index[-1]
            now = datetime.now(timezone.utc)
            
            # Ensure timezone aware
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize(timezone.utc)
            
            if last_ts > now:
                df = df.iloc[:-1]
            else:
                # Drop if current second < 45
                if last_ts.minute == now.minute and last_ts.hour == now.hour and now.second < 45:
                    df = df.iloc[:-1]
            
            if df.empty:
                return pd.DataFrame()
                
            # Calculate offset for alignment
            last_ts = df.index[-1]
            minute = last_ts.minute
            offset_minutes = (minute + 1) % int(self.interval[:-1])
            offset = f"{offset_minutes}min" if offset_minutes != 0 else "0min"
            
            bars = df.resample(self._interval_to_pandas_freq(), offset=offset).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "trades_done": "sum"
            })
        else:
            bars = df.resample(self._interval_to_pandas_freq()).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "trades_done": "sum"
            })

        bars = bars.dropna()
        return bars

    def _parse_allora_bar(self, timestamp: datetime, row: pd.Series, symbol: str) -> dict:
        """
        Parse Allora bar data into standardized format.
        
        Maps:
        - Allora's "trades_done" → standardized "n_trades"
        - Sets "quote_volume" to NaN (not available in Allora)
        """
        return {
            "symbol": symbol,
            "open_time": timestamp,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "quote_volume": np.nan,  # Not available in Allora
            "n_trades": int(row["trades_done"]),
        }

    # ---------- Backfill (Implementation of Abstract Methods) ----------
    def backfill_symbol(self, symbol: str, start: datetime, end: Optional[datetime] = None, only_days: Optional[set] = None):
        """
        Backfill historical data for a single symbol.
        Uses monthly buckets where available, then API for recent data.
        
        Args:
            symbol: Symbol to backfill (e.g. "BTC/USD")
            start: Start datetime (UTC)
            end: End datetime (UTC), defaults to now
            only_days: Optional set of date objects - if provided, only write bars from these specific days
        """
        self._in_backfill = True
        
        # Ensure start and end are timezone-aware (UTC)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(timezone.utc)
        elif end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        
        print(f"[Allora backfill] {symbol}: {start} → {end}")
        
        # Step 1: Try to use monthly buckets for bulk historical data
        from_month = start.strftime("%Y-%m")
        to_month = end.strftime("%Y-%m")
        all_buckets = self._list_ready_buckets(symbol, from_month)
        
        # Filter buckets to only those between start and end month
        # Bucket structure: {'start': '2020-01-01', 'end': '2020-01-31', ...}
        buckets = []
        for bucket in all_buckets:
            if 'start' in bucket:
                bucket_month = bucket['start'][:7]  # Get YYYY-MM from 'YYYY-MM-DD'
                if from_month <= bucket_month <= to_month:
                    buckets.append(bucket)
        
        frames = []
        if buckets:
            print(f"[Allora backfill] {symbol}: downloading {len(buckets)} monthly buckets (filtered from {len(all_buckets)})")
            for bucket in buckets:
                df = self._fetch_bucket_csv(bucket["download_url"])
                if not df.empty:
                    frames.append(df)
        
        # Combine bucket data
        if frames:
            combined_df = pd.concat(frames, ignore_index=True)
            combined_df["date"] = pd.to_datetime(combined_df["date"], utc=True)
            combined_df = combined_df.drop_duplicates(subset="date").sort_values("date")
            
            # Filter to date range
            combined_df = combined_df[
                (combined_df["date"] >= start) & 
                (combined_df["date"] <= end)
            ]
        else:
            combined_df = pd.DataFrame()
        
        # Step 2: Fill gaps with API data (especially recent data)
        # Determine what's missing
        if not combined_df.empty:
            last_bucket_date = combined_df["date"].max()
            # Fetch from last bucket date to end
            api_start = max(start, last_bucket_date)
        else:
            # No bucket data, fetch everything via API
            api_start = start
        
        # Fetch recent data via API
        if api_start < end:
            # Calculate reasonable max_pages based on date range
            # API returns ~1440 1-minute bars per page (1 day)
            days_needed = (end - api_start).days + 1
            reasonable_max_pages = min(days_needed + 5, self.max_pages)  # Add buffer of 5
            
            print(f"[Allora backfill] {symbol}: fetching API data from {api_start.date()} ({days_needed} days, max {reasonable_max_pages} pages)")
            api_df = self._fetch_ohlcv_data(symbol, api_start.strftime("%Y-%m-%d"), max_pages=reasonable_max_pages)
            
            if not api_df.empty:
                # Combine with bucket data
                if not combined_df.empty:
                    combined_df = pd.concat([combined_df, api_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset="date").sort_values("date")
                else:
                    combined_df = api_df
        
        if combined_df.empty:
            print(f"[Allora backfill] {symbol}: no data returned")
            self._in_backfill = False
            return
        
        # Step 3: Store raw 1-minute bars (NO resampling during backfill)
        # Resampling happens when loading data for the workflow
        # Rename 'date' column to 'open_time' to match expected format
        combined_df = combined_df.rename(columns={'date': 'open_time'})
        
        # Filter to only_days if specified (for efficient gap-filling)
        if only_days:
            combined_df['date'] = pd.to_datetime(combined_df['open_time']).dt.date
            combined_df = combined_df[combined_df['date'].isin(only_days)]
            combined_df = combined_df.drop(columns=['date'])
            print(f"[Allora backfill] {symbol}: filtered to {len(only_days)} incomplete days, writing {len(combined_df)} bars to Parquet")
        else:
            print(f"[Allora backfill] {symbol}: writing {len(combined_df)} bars to Parquet")
        
        # Bulk write for performance (instead of row-by-row)
        # Convert dataframe rows to bar dictionaries
        bars_list = []
        for _, row in combined_df.iterrows():
            bar = {
                'open_time': row['open_time'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0.0,
                'symbol': symbol  # Fixed: use 'symbol' not 'ticker'
            }
            bars_list.append(bar)
        
        # Group by date and write each day's data at once
        from collections import defaultdict
        from allora_forge_builder_kit.base_data_manager import day_str
        by_day = defaultdict(list)
        for bar in bars_list:
            d = day_str(bar["open_time"])
            by_day[d].append(bar)
        
        # Write each day's data in one operation
        for day, day_bars in by_day.items():
            path = self._partition_path(symbol, day)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            new_df = pl.DataFrame(day_bars)
            if os.path.exists(path):
                old_df = pl.read_parquet(path)
                df = pl.concat([old_df, new_df]).unique(subset=["open_time"], keep="last").sort("open_time")
            else:
                df = new_df.sort("open_time")
            df.write_parquet(path)
        
        print(f"[Allora backfill] {symbol}: wrote {len(by_day)} daily parquet files")
        self._in_backfill = False

    def backfill_realtime(self, symbols: List[str]):
        """
        Sync latest data from last stored bar to now.
        Overwrites the last bar to ensure clean continuation.
        
        Args:
            symbols: List of symbols to sync
        """
        self._clean_corrupt_files()
        now = datetime.now(timezone.utc)
        
        for sym in symbols:
            last = self.latest(sym)

            if not last:
                # No history, do full backfill from 2020
                start = datetime(2020, 1, 1, tzinfo=timezone.utc)
                print(f"[Allora backfill-realtime] {sym}: no history, full backfill {start} → {now}")
            else:
                # Resume from last bar (subtract interval to overwrite)
                start = last - timedelta(minutes=int(self.interval[:-1]))
                print(f"[Allora backfill-realtime] {sym}: resuming from {start} → {now}")

            self.backfill_symbol(sym, start, end=now)

    def backfill_missing(self, symbols: List[str], start: Optional[datetime] = None):
        """
        Detect gaps in local storage and backfill missing data.
        
        Args:
            symbols: List of symbols to check
            start: Start date for gap detection (defaults to 2020-01-01)
        """
        self._clean_corrupt_files()
        now = datetime.now(timezone.utc)
        
        # Allora always stores 1-minute bars (downsampling happens in code)
        # Expected bars per day for 1-minute base interval
        expected_per_day = 24 * 60  # 1440 bars per day for 1-minute bars

        # Default start = 2020-01-01
        start = start or datetime(2020, 1, 1, tzinfo=timezone.utc)

        for sym in symbols:
            last = self.latest(sym)
            print(f"[Allora backfill-missing] Checking {sym} {start} → {now}")

            import glob
            glob_path = f"{self.base_dir}/symbol={sym}/dt=*.parquet"
            files = glob.glob(glob_path)
            
            if files:
                df = (pl.scan_parquet(files)
                      .with_columns(pl.col("open_time").dt.date().alias("date"))
                      .group_by("date")
                      .agg(pl.len().alias("n_bars"))
                      .collect())
                print(f"[Allora backfill-missing] Found {len(files)} parquet files for {sym}")
                print(f"[Allora backfill-missing] Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"[Allora backfill-missing] Complete days (1440 bars): {len(df.filter(pl.col('n_bars') == expected_per_day))}")
            else:
                print(f"[Allora backfill-missing] No parquet files found for {sym} at {glob_path}")
                df = pl.DataFrame({"date": [], "n_bars": []})

            complete_days = set(
                df.filter(pl.col("n_bars") == expected_per_day)["date"].to_list()
            )

            # Collect all incomplete days
            incomplete_days = []
            d = start.date()
            # Fill any incomplete days up to yesterday (if last exists)
            while last and d < last.date():
                if d not in complete_days:
                    incomplete_days.append(d)
                d += timedelta(days=1)
            
            # If there are incomplete days, backfill them (only write those specific days)
            if incomplete_days:
                # Group incomplete days by month to minimize bucket downloads
                from collections import defaultdict
                months_needed = defaultdict(list)
                for day in incomplete_days:
                    month_key = (day.year, day.month)
                    months_needed[month_key].append(day)
                
                first_day = min(incomplete_days)
                last_day = max(incomplete_days)
                print(f"[Allora backfill-missing] {sym}: {len(incomplete_days)} incomplete days across {len(months_needed)} months, backfilling {first_day} → {last_day}")
                
                # Download only the months we need
                for (year, month), days_in_month in months_needed.items():
                    month_start = datetime(year, month, 1, tzinfo=timezone.utc)
                    # Last day of month
                    if month == 12:
                        month_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(microseconds=1)
                    else:
                        month_end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(microseconds=1)
                    
                    print(f"[Allora backfill-missing] {sym}: backfilling {len(days_in_month)} days in {year}-{month:02d}")
                    self.backfill_symbol(sym, month_start, end=month_end, only_days=set(days_in_month))

            # Fill latest day from last known → now
            if last:
                latest_day = last.date()
                latest_start = last + timedelta(milliseconds=1)
                print(f"[Allora backfill-missing] {sym}: backfilling latest {latest_day} from {latest_start} → {now}")
                self.backfill_symbol(sym, latest_start, end=now)
            else:
                print(f"[Allora backfill-missing] {sym}: no history at all, full backfill")
                self.backfill_symbol(sym, start, end=now)

    # ---------- Live Snapshot (Always fetch from API) ----------
    def get_live_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch the most recent OHLCV bar(s) directly from Allora API.
        
        IMPORTANT: Always fetches from remote - no local storage dependency.
        This allows the function to work in pickled/isolated contexts.
        
        Args:
            symbols: List of symbols to fetch
            
        Returns:
            MultiIndex DataFrame (symbol, open_time) with columns:
            open, high, low, close, volume, quote_volume, n_trades
        """
        bars = []
        
        for ticker in symbols:
            try:
                # Fetch recent 1-min data (last 2 hours to ensure we have enough)
                from_date = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%d")
                df_1min = self._fetch_ohlcv_data(ticker, from_date, max_pages=5)
                
                if df_1min.empty:
                    print(f"[Allora snapshot] {ticker}: no data returned")
                    continue
                
                # Aggregate to interval bars (e.g., 5-min) in live mode
                df_bars = self._create_interval_bars(df_1min, live_mode=True)
                
                if df_bars.empty:
                    print(f"[Allora snapshot] {ticker}: no complete bars")
                    continue
                    
                # Get most recent bar
                latest_timestamp = df_bars.index[-1]
                latest_row = df_bars.iloc[-1]
                
                bar = self._parse_allora_bar(latest_timestamp, latest_row, ticker)
                bars.append(bar)
                
            except Exception as e:
                print(f"[Allora snapshot error] {ticker}: {e}")
                continue
        
        if not bars:
            raise ValueError("No data fetched for any symbol")
        
        df = pd.DataFrame(bars)
        df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
        df = df.set_index(['symbol', 'open_time']).sort_index()
        return df

    def get_live_1min_data(self, symbol: str, hours_back: int = 2) -> pd.DataFrame:
        """
        Fetch recent 1-minute bars for a symbol (for live feature extraction).
        
        This is used by the workflow to get raw 1-minute data that can then be
        resampled to the target interval and used for feature extraction.
        
        Args:
            symbol: Symbol to fetch
            hours_back: Hours of historical 1-minute data to fetch
            
        Returns:
            DataFrame with 1-minute bars (open_time index, OHLCV columns)
        """
        # Calculate start date (Allora API uses date strings)
        from_date = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).strftime("%Y-%m-%d")
        
        # Calculate reasonable max_pages (Allora returns ~2000 bars per page)
        # Each page ≈ 1.4 days of 1-min data, so days needed = hours_back / 24
        # Add buffer (+2 pages) and set reasonable upper limit to prevent excessive API calls
        days_needed = (hours_back / 24) + 2
        max_pages = min(int(days_needed), 50)  # Cap at 50 pages (~70 days)
        
        # Fetch 1-minute bars
        df_1min = self._fetch_ohlcv_data(symbol, from_date, max_pages=max_pages)
        
        if df_1min.empty:
            return pd.DataFrame()
        
        # Return as DataFrame with open_time index and standard OHLCV columns
        df = df_1min.copy()
        df = df.rename(columns={'date': 'open_time'})
        df = df.set_index('open_time').sort_index()
        
        # Select only OHLCV columns (standard format)
        return df[['open', 'high', 'low', 'close', 'volume']]
