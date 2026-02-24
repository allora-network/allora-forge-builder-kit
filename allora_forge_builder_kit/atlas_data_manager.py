#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl
import requests

from .base_data_manager import BaseDataManager, day_str


ATLAS_BASE_URL = "https://forge-data.allora.run/api"


class AtlasDataManager(BaseDataManager):
    """
    Atlas data service manager for Allora Network.

    Uses the Atlas timeseries platform (forge-data.allora.run) which provides
    1-minute candle data sourced from Tiingo.  The Atlas service accepts the
    same API key as the legacy Allora data service.

    Features:
      - Partitioned Parquet storage (inherited)
      - Automatic dataset discovery by ticker
      - Bulk download for efficient historical backfill
      - Paginated row queries for incremental sync
      - Hot cache of recent bars (inherited)
      - Compatible with AlloraMLWorkflow
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = ATLAS_BASE_URL,
        base_dir: str = "parquet_data_allora",
        interval: str = "5m",
        symbols: Optional[List[str]] = None,
        cache_len: int = 1000,
        page_size: int = 1000,
        sleep_sec: float = 0.0,
    ):
        super().__init__(
            base_dir=base_dir, interval=interval, symbols=symbols, cache_len=cache_len
        )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}
        self.page_size = page_size
        self.sleep_sec = sleep_sec

        self._dataset_cache: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        return ticker.replace("/", "").replace("-", "").lower()

    def _interval_minutes(self) -> int:
        """Convert interval string (e.g. '5m', '1h', '1d') to minutes."""
        iv = self.interval
        if iv.endswith("h"):
            return int(iv[:-1]) * 60
        elif iv.endswith("d"):
            return int(iv[:-1]) * 1440
        elif iv.endswith("min"):
            return int(iv[:-3])
        return int(iv[:-1])

    def _interval_to_pandas_freq(self) -> str:
        interval = self.interval
        if interval.endswith("m") and not interval.endswith("min"):
            return interval[:-1] + "min"
        return interval

    def _resolve_dataset_id(self, ticker: str) -> int:
        """Resolve a ticker symbol to an Atlas dataset ID (cached)."""
        norm = self._normalize_ticker(ticker)
        if norm in self._dataset_cache:
            return self._dataset_cache[norm]

        dataset_name = f"tiingo_{norm}_1min"
        resp = requests.get(
            f"{self.base_url}/datasets/search/",
            headers=self.headers,
            params={"source": "tiingo", "ticker": norm, "frequency": "1min"},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        for ds in results:
            if ds["name"] == dataset_name:
                self._dataset_cache[norm] = ds["id"]
                return ds["id"]

        if results:
            self._dataset_cache[norm] = results[0]["id"]
            return results[0]["id"]

        raise ValueError(
            f"No Atlas dataset found for ticker '{ticker}' "
            f"(searched for '{dataset_name}')"
        )

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------
    def _fetch_rows_paginated(
        self,
        dataset_id: int,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        max_rows: int = 100_000,
    ) -> pd.DataFrame:
        """Fetch rows from Atlas via paginated REST calls."""
        params: dict = {
            "dataset": dataset_id,
            "ordering": "timestamp",
            "limit": self.page_size,
        }
        if start:
            params["start"] = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        if end:
            params["end"] = end.strftime("%Y-%m-%dT%H:%M:%SZ")

        all_rows: list = []
        offset = 0

        while len(all_rows) < max_rows:
            params["offset"] = offset
            try:
                resp = requests.get(
                    f"{self.base_url}/rows/",
                    headers=self.headers,
                    params=params,
                    timeout=60,
                )
                resp.raise_for_status()
                payload = resp.json()
            except Exception as e:
                print(f"[Atlas API error] dataset={dataset_id} offset={offset}: {e}")
                break

            results = payload.get("results", [])
            if not results:
                break

            all_rows.extend(results)
            if payload.get("next") is None:
                break

            offset += self.page_size
            time.sleep(self.sleep_sec)

        return self._rows_to_dataframe(all_rows)

    def _bulk_download(
        self,
        dataset_id: int,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Efficient bulk download using Atlas streaming endpoint."""
        params: dict = {"dataset": dataset_id, "output": "json"}
        if start:
            params["start"] = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        if end:
            params["end"] = end.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            resp = requests.get(
                f"{self.base_url}/rows/bulk_download/",
                headers=self.headers,
                params=params,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[Atlas bulk download error] dataset={dataset_id}: {e}")
            return pd.DataFrame()

        if isinstance(data, list):
            return self._rows_to_dataframe(data)
        if isinstance(data, dict) and "results" in data:
            return self._rows_to_dataframe(data["results"])
        return pd.DataFrame()

    @staticmethod
    def _rows_to_dataframe(rows: list) -> pd.DataFrame:
        """Convert Atlas row objects into a flat DataFrame."""
        if not rows:
            return pd.DataFrame()

        records = []
        for row in rows:
            vals = row.get("values", {})
            records.append(
                {
                    "date": row["timestamp"],
                    "open": float(vals["open"]) if "open" in vals else float("nan"),
                    "high": float(vals["high"]) if "high" in vals else float("nan"),
                    "low": float(vals["low"]) if "low" in vals else float("nan"),
                    "close": float(vals["close"]) if "close" in vals else float("nan"),
                    "volume": float(vals.get("volume", 0)),
                    "trades_done": int(vals.get("trades_done", 0)),
                    "volume_notional": float(vals.get("volume_notional", 0)),
                }
            )

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Interval resampling (shared with legacy manager)
    # ------------------------------------------------------------------
    def _create_interval_bars(
        self, df_1min: pd.DataFrame, live_mode: bool = False
    ) -> pd.DataFrame:
        if df_1min.empty:
            return pd.DataFrame()

        df = df_1min.copy()
        df = df.set_index("date").sort_index().dropna()

        if live_mode:
            last_ts = df.index[-1]
            now = datetime.now(timezone.utc)

            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize(timezone.utc)

            if last_ts > now:
                df = df.iloc[:-1]
            elif last_ts.minute == now.minute and last_ts.hour == now.hour and now.second < 45:
                df = df.iloc[:-1]

            if df.empty:
                return pd.DataFrame()

            last_ts = df.index[-1]
            minute = last_ts.minute
            offset_minutes = (minute + 1) % self._interval_minutes()
            offset = f"{offset_minutes}min" if offset_minutes != 0 else "0min"

            agg_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "trades_done": "sum",
            }
            bars = df.resample(self._interval_to_pandas_freq(), offset=offset).agg(agg_dict)
        else:
            agg_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "trades_done": "sum",
            }
            bars = df.resample(self._interval_to_pandas_freq()).agg(agg_dict)

        return bars.dropna()

    def _parse_atlas_bar(
        self, timestamp: datetime, row: pd.Series, symbol: str
    ) -> dict:
        return {
            "symbol": symbol,
            "open_time": timestamp,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "quote_volume": np.nan,
            "n_trades": int(row["trades_done"]),
        }

    # ------------------------------------------------------------------
    # Backfill implementations
    # ------------------------------------------------------------------
    def backfill_symbol(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        only_days: Optional[set] = None,
    ):
        self._in_backfill = True
        try:
            self._backfill_symbol_impl(symbol, start, end, only_days)
        finally:
            self._in_backfill = False

    def _backfill_symbol_impl(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime],
        only_days: Optional[set],
    ):
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(timezone.utc)
        elif end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        print(f"[Atlas backfill] {symbol}: {start.date()} -> {end.date()}")

        dataset_id = self._resolve_dataset_id(symbol)

        days_span = (end - start).days
        if days_span > 30:
            print(f"[Atlas backfill] {symbol}: using bulk download ({days_span} days)")
            combined_df = self._bulk_download(dataset_id, start=start, end=end)
        else:
            combined_df = self._fetch_rows_paginated(dataset_id, start=start, end=end)

        if combined_df.empty:
            print(f"[Atlas backfill] {symbol}: no data returned")
            return

        combined_df = combined_df.rename(columns={"date": "open_time"})

        if only_days:
            combined_df["_date"] = pd.to_datetime(combined_df["open_time"]).dt.date
            combined_df = combined_df[combined_df["_date"].isin(only_days)]
            combined_df = combined_df.drop(columns=["_date"])
            print(
                f"[Atlas backfill] {symbol}: filtered to {len(only_days)} days, "
                f"{len(combined_df)} bars"
            )
        else:
            print(f"[Atlas backfill] {symbol}: writing {len(combined_df)} bars")

        combined_df["volume"] = combined_df["volume"].fillna(0)
        combined_df["symbol"] = symbol
        combined_df["_day"] = combined_df["open_time"].dt.strftime("%Y-%m-%d")

        for day_val, group in combined_df.groupby("_day"):
            path = self._partition_path(symbol, day_val)
            os.makedirs(os.path.dirname(path), exist_ok=True)

            new_df = pl.from_pandas(
                group[["open_time", "open", "high", "low", "close", "volume", "symbol"]]
            )
            if os.path.exists(path):
                old_df = pl.read_parquet(path)
                df = (
                    pl.concat([old_df, new_df])
                    .unique(subset=["open_time"], keep="last")
                    .sort("open_time")
                )
            else:
                df = new_df.sort("open_time")
            df.write_parquet(path)

        n_days = combined_df["_day"].nunique()
        print(f"[Atlas backfill] {symbol}: wrote {n_days} daily parquet files")

    def backfill_realtime(self, symbols: List[str]):
        self._clean_corrupt_files()
        now = datetime.now(timezone.utc)

        for sym in symbols:
            last = self.latest(sym)
            if not last:
                start = datetime(2020, 1, 1, tzinfo=timezone.utc)
                print(f"[Atlas backfill-realtime] {sym}: no history, full backfill")
            else:
                start = last - timedelta(minutes=self._interval_minutes())
                print(f"[Atlas backfill-realtime] {sym}: resuming from {start}")
            self.backfill_symbol(sym, start, end=now)

    def backfill_missing(
        self, symbols: List[str], start: Optional[datetime] = None
    ):
        self._clean_corrupt_files()
        now = datetime.now(timezone.utc)
        expected_per_day = 24 * 60
        start = start or datetime(2020, 1, 1, tzinfo=timezone.utc)

        for sym in symbols:
            last = self.latest(sym)
            print(f"[Atlas backfill-missing] Checking {sym} {start.date()} -> {now.date()}")

            import glob as _glob

            safe = self._normalize_symbol_for_path(sym)
            glob_path = f"{self.base_dir}/symbol={safe}/dt=*.parquet"
            files = _glob.glob(glob_path)

            if files:
                df = (
                    pl.scan_parquet(files)
                    .with_columns(pl.col("open_time").dt.date().alias("date"))
                    .group_by("date")
                    .agg(pl.len().alias("n_bars"))
                    .collect()
                )
            else:
                df = pl.DataFrame({"date": [], "n_bars": []})

            complete_days = set(
                df.filter(pl.col("n_bars") == expected_per_day)["date"].to_list()
            )

            incomplete_days = []
            d = start.date()
            while last and d < last.date():
                if d not in complete_days:
                    incomplete_days.append(d)
                d += timedelta(days=1)

            if incomplete_days:
                months_needed: Dict[tuple, list] = defaultdict(list)
                for day in incomplete_days:
                    months_needed[(day.year, day.month)].append(day)

                first_day = min(incomplete_days)
                last_day = max(incomplete_days)
                print(
                    f"[Atlas backfill-missing] {sym}: {len(incomplete_days)} incomplete "
                    f"days, backfilling {first_day} -> {last_day}"
                )

                for (year, month), days_in_month in months_needed.items():
                    if len(days_in_month) <= 7:
                        for day in days_in_month:
                            day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
                            day_end = day_start + timedelta(days=1) - timedelta(microseconds=1)
                            self.backfill_symbol(sym, day_start, end=day_end)
                    else:
                        month_start = datetime(year, month, 1, tzinfo=timezone.utc)
                        if month == 12:
                            month_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(
                                microseconds=1
                            )
                        else:
                            month_end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(
                                microseconds=1
                            )
                        self.backfill_symbol(
                            sym, month_start, end=month_end, only_days=set(days_in_month)
                        )

            if last:
                latest_start = last + timedelta(milliseconds=1)
                self.backfill_symbol(sym, latest_start, end=now)
            else:
                self.backfill_symbol(sym, start, end=now)

    # ------------------------------------------------------------------
    # Live snapshot
    # ------------------------------------------------------------------
    def get_live_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        bars = []
        for ticker in symbols:
            try:
                dataset_id = self._resolve_dataset_id(ticker)
                from_dt = datetime.now(timezone.utc) - timedelta(hours=2)

                df_1min = self._fetch_rows_paginated(
                    dataset_id, start=from_dt, max_rows=5000
                )

                # Fallback: data may be lagging; find the actual latest row
                if df_1min.empty:
                    resp = requests.get(
                        f"{self.base_url}/rows/",
                        headers=self.headers,
                        params={"dataset": dataset_id, "limit": 1, "ordering": "-timestamp"},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    results = resp.json().get("results", [])
                    if results:
                        latest_ts = pd.to_datetime(results[0]["timestamp"], utc=True).to_pydatetime()
                        from_dt = latest_ts - timedelta(hours=2)
                        df_1min = self._fetch_rows_paginated(
                            dataset_id, start=from_dt, end=latest_ts + timedelta(minutes=1), max_rows=5000
                        )

                if df_1min.empty:
                    print(f"[Atlas snapshot] {ticker}: no data returned")
                    continue

                df_bars = self._create_interval_bars(df_1min, live_mode=True)
                if df_bars.empty:
                    print(f"[Atlas snapshot] {ticker}: no complete bars")
                    continue

                latest_ts = df_bars.index[-1]
                latest_row = df_bars.iloc[-1]
                bars.append(self._parse_atlas_bar(latest_ts, latest_row, ticker))
            except Exception as e:
                print(f"[Atlas snapshot error] {ticker}: {e}")

        if not bars:
            raise ValueError("No data fetched for any symbol")

        df = pd.DataFrame(bars)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        return df.set_index(["symbol", "open_time"]).sort_index()

    def get_live_1min_data(
        self, symbol: str, hours_back: int = 2
    ) -> pd.DataFrame:
        """
        Fetch the most recent 1-minute bars available for a symbol.

        Atlas data may have some lag relative to wall-clock time, so this
        method first finds the latest available row and then fetches
        ``hours_back`` worth of data ending at that point.
        """
        dataset_id = self._resolve_dataset_id(symbol)

        # First attempt: recent data from wall-clock time
        from_dt = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        df_1min = self._fetch_rows_paginated(
            dataset_id,
            start=from_dt,
            max_rows=hours_back * 60 + 120,
        )

        # If no data found (data lag), discover the latest available row
        # and fetch backwards from there.
        if df_1min.empty:
            try:
                resp = requests.get(
                    f"{self.base_url}/rows/",
                    headers=self.headers,
                    params={"dataset": dataset_id, "limit": 1, "ordering": "-timestamp"},
                    timeout=30,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if not results:
                    return pd.DataFrame()

                latest_ts = pd.to_datetime(results[0]["timestamp"], utc=True).to_pydatetime()
                from_dt = latest_ts - timedelta(hours=hours_back)
                df_1min = self._fetch_rows_paginated(
                    dataset_id,
                    start=from_dt,
                    end=latest_ts + timedelta(minutes=1),
                    max_rows=hours_back * 60 + 120,
                )
            except Exception as e:
                print(f"[Atlas live fallback error] {symbol}: {e}")
                return pd.DataFrame()

        if df_1min.empty:
            return pd.DataFrame()

        df = df_1min.rename(columns={"date": "open_time"})
        df = df.set_index("open_time").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------
    # Dataset discovery helpers
    # ------------------------------------------------------------------
    def list_available_datasets(
        self,
        source: str = "tiingo",
        frequency: str = "1min",
    ) -> List[Dict]:
        """List datasets available on Atlas for a given source and frequency."""
        params = {"source": source, "frequency": frequency, "limit": 250}
        resp = requests.get(
            f"{self.base_url}/datasets/search/",
            headers=self.headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    def search_datasets(self, query: str) -> List[Dict]:
        """Free-text search across Atlas dataset names and descriptions."""
        resp = requests.get(
            f"{self.base_url}/datasets/",
            headers=self.headers,
            params={"search": query, "limit": 50},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
