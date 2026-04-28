#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob as _glob
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import polars as pl
import requests

from .base_data_manager import BaseDataManager, day_str


ATLAS_BASE_URL = "https://forge-data.allora.run/api"
_ALLOWED_HOSTS = {"forge-data.allora.run"}


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

    _acquired_keys: set = set()

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
        allowed_hosts: Optional[set] = None,
        auto_acquire_tag: bool = True,
    ):
        super().__init__(
            base_dir=base_dir, interval=interval, symbols=symbols, cache_len=cache_len
        )
        effective_hosts = allowed_hosts if allowed_hosts is not None else _ALLOWED_HOSTS
        parsed = urlparse(base_url)
        if parsed.hostname not in effective_hosts:
            raise ValueError(
                f"base_url host '{parsed.hostname}' is not in the allowed list "
                f"{effective_hosts}. This protects against API key leakage."
            )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}
        self.page_size = page_size
        self.sleep_sec = sleep_sec

        self._dataset_cache: Dict[str, int] = {}

        if auto_acquire_tag and api_key not in AtlasDataManager._acquired_keys:
            self._ensure_public_tag()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_public_tag(self) -> None:
        """Acquire the 'public' tag if not already held.

        The Atlas UI does this automatically on login. API users need to call
        POST /api/tags/acquire/ once so they can see public datasets.
        This is idempotent — safe to call every time.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/tags/acquire/",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"tag_name": "public"},
                timeout=15,
            )
            if resp.status_code in (200, 201):
                print("[Atlas] Acquired 'public' tag — public datasets are now accessible.")
                AtlasDataManager._acquired_keys.add(self.api_key)
            elif resp.status_code == 409:
                AtlasDataManager._acquired_keys.add(self.api_key)  # already held
            else:
                print(f"[Atlas] Warning: could not acquire 'public' tag (HTTP {resp.status_code}): {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            print(f"[Atlas] Warning: could not acquire 'public' tag: {e}")

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

        expected_name = f"tiingo_{norm}_1min"
        try:
            resp = requests.get(
                f"{self.base_url}/datasets/",
                headers=self.headers,
                params={"search": expected_name, "limit": 10},
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Atlas API error while resolving dataset for ticker '{ticker}': {e}"
            ) from e

        match = next((ds for ds in results if ds.get("name") == expected_name), None)
        if match is None:
            raise ValueError(
                f"No Atlas dataset with expected name '{expected_name}' found for "
                f"ticker '{ticker}'. Got {len(results)} result(s)"
                + (f"; first was '{results[0].get('name')}'" if results else "")
                + ". Use list_available_datasets() or search_datasets() to discover datasets."
            )

        dataset_id = match.get("id")
        if not isinstance(dataset_id, int):
            raise TypeError(
                f"Atlas returned non-integer dataset ID {dataset_id!r} "
                f"for '{expected_name}'"
            )

        self._dataset_cache[norm] = dataset_id
        return dataset_id

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
            except requests.exceptions.RequestException as e:
                if all_rows:
                    import warnings
                    warnings.warn(
                        f"[Atlas API] Partial data: got {len(all_rows)} rows before "
                        f"error at offset {offset}: {e}",
                        stacklevel=2,
                    )
                    break
                raise

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
        raise_on_error: bool = False,
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
            if raise_on_error:
                raise
            print(f"[Atlas bulk download error] dataset={dataset_id}: {e}")
            return pd.DataFrame()

        if isinstance(data, list):
            return self._rows_to_dataframe(data)
        if isinstance(data, dict) and "results" in data:
            return self._rows_to_dataframe(data["results"])
        return pd.DataFrame()

    def _bulk_download_chunked(
        self,
        dataset_id: int,
        start: datetime,
        end: datetime,
        chunk_days: int = 30,
        min_chunk_days: int = 1,
        retries: int = 3,
    ) -> pd.DataFrame:
        """Download large ranges in resilient chunks with retry + split-on-failure.

        If a chunk fails repeatedly (timeouts/network), it is recursively split
        into smaller chunks until `min_chunk_days` is reached.
        """

        def _download_resilient(chunk_start: datetime, chunk_end: datetime, days_hint: int) -> list[pd.DataFrame]:
            last_error = None
            for attempt in range(1, retries + 1):
                try:
                    df = self._bulk_download(
                        dataset_id,
                        start=chunk_start,
                        end=chunk_end,
                        raise_on_error=True,
                    )
                    return [df] if not df.empty else []
                except Exception as e:
                    last_error = e
                    if attempt < retries:
                        time.sleep(min(5.0, attempt * 1.5))

            span_days = max((chunk_end - chunk_start).total_seconds() / 86400.0, 0.0)
            if span_days <= float(min_chunk_days):
                print(
                    f"[Atlas backfill] giving up chunk {chunk_start.date()} -> {chunk_end.date()} "
                    f"after {retries} retries: {last_error}"
                )
                return []

            midpoint = chunk_start + (chunk_end - chunk_start) / 2
            left_end = midpoint
            right_start = midpoint + timedelta(seconds=1)
            next_hint = max(min_chunk_days, int(days_hint / 2) or min_chunk_days)

            print(
                f"[Atlas backfill] splitting failed chunk {chunk_start.date()} -> {chunk_end.date()} "
                f"into {chunk_start.date()} -> {left_end.date()} and {right_start.date()} -> {chunk_end.date()}"
            )
            return _download_resilient(chunk_start, left_end, next_hint) + _download_resilient(right_start, chunk_end, next_hint)

        chunks: list[pd.DataFrame] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
            chunks.extend(_download_resilient(chunk_start, chunk_end, chunk_days))
            chunk_start = chunk_end + timedelta(seconds=1)

        if not chunks:
            return pd.DataFrame()
        combined = pd.concat(chunks, ignore_index=True)
        return combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _rows_to_dataframe(rows: list) -> pd.DataFrame:
        """Convert Atlas row objects into a flat DataFrame."""
        if not rows:
            return pd.DataFrame()

        records = []
        for row in rows:
            vals = row.get("values", None) or row
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
        if live_mode:
            bars = df.resample(self._interval_to_pandas_freq(), offset=offset).agg(agg_dict)
        else:
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
        if days_span > 14:
            print(f"[Atlas backfill] {symbol}: resilient bulk download in 14-day chunks ({days_span} days)")
            combined_df = self._bulk_download_chunked(
                dataset_id,
                start=start,
                end=end,
                chunk_days=14,
                min_chunk_days=1,
                retries=3,
            )
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
        """List datasets available on Atlas for a given source and frequency.

        Uses text search on ``/datasets/`` and filters results client-side,
        since ``/datasets/search/`` currently rejects multiple metadata filters.
        """
        resp = requests.get(
            f"{self.base_url}/datasets/",
            headers=self.headers,
            params={"search": source, "limit": 250},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [
            ds for ds in results
            if ds.get("metadata", {}).get("source") == source
            and ds.get("metadata", {}).get("frequency") == frequency
        ]

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
