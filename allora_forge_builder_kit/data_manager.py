#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import random
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import requests
import polars as pl
import websocket  # pip install websocket-client
import glob
import pandas as pd

# -----------------------------
# Utility
# -----------------------------

def to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

def day_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


# -----------------------------
# DataManager
# -----------------------------

class DataManager:
    """
    Binance market data manager:
      - Partitioned Parquet storage
      - Backfill via REST
      - Auto-resume before live streaming
      - WebSocket sharded streaming
      - Batch callback once per bar close across all tickers
      - Timeout + REST gap-fill for missing bars
      - Hot cache of recent bars
    """

    SPOT_REST = "https://data-api.binance.vision/api/v3/klines"
    FUTS_REST = "https://fapi.binance.com/fapi/v1/klines"
    SPOT_WS   = "wss://stream.binance.com:9443/stream?streams="
    FUTS_WS   = "wss://fstream.binance.com/stream?streams="

    def __init__(
        self,
        base_dir: str = "parquet_data",
        interval: str = "5m",
        market: str = "futures",  # "spot" or "futures"
        symbols: Optional[List[str]] = None,
        cache_len: int = 1000,
        batch_timeout: int = 20,
        rate_limit: float = 0.5 # seconds between REST calls
    ):
        self.base_dir = base_dir
        self.interval = interval
        self.market = market.lower()
        self.symbols = set(symbols or [])
        os.makedirs(self.base_dir, exist_ok=True)

        # cache of recent bars
        self._cache_len = cache_len
        self._bar_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=cache_len))

        # batch logic
        self._batch_callbacks = []
        self._pending: Dict[datetime, set[str]] = defaultdict(set)
        self._timers: Dict[datetime, threading.Timer] = {}
        self.batch_timeout = batch_timeout

        # live WS state
        self._ws_threads: List[threading.Thread] = []
        self._stop_event = threading.Event()

        # rate throttling
        self._last_request = 0
        self._rate_limit = rate_limit  

        # backfill flag
        self._in_backfill = False

    # ---------- File paths ----------
    def _partition_path(self, symbol: str, dt_str: str) -> str:
        return f"{self.base_dir}/symbol={symbol}/dt={dt_str}.parquet"

    # ---------- Append bar ----------
    def _append_bar(self, bar: dict, backfill: bool = False):
        sym = bar["symbol"]
        ot = bar["open_time"]
    
        # write to parquet
        d = day_str(ot)
        path = self._partition_path(sym, d)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
        new = pl.DataFrame([bar])
        if os.path.exists(path):
            old = pl.read_parquet(path)
            df = pl.concat([old, new]).unique(subset=["open_time"], keep="last")
        else:
            df = new
        df.write_parquet(path)
    
        # update cache with dedupe
        if self._bar_cache[sym] and self._bar_cache[sym][-1]["open_time"] == ot:
            self._bar_cache[sym][-1] = bar
        else:
            self._bar_cache[sym].append(bar)
    
        if backfill:
            return  # skip batch logic
    
        # batch tracking
        self._pending[ot].add(sym)
    
        if ot not in self._timers:
            t = threading.Timer(self.batch_timeout, self._batch_timeout_handler, args=[ot])
            t.daemon = True
            t.start()
            self._timers[ot] = t
    
        self._check_and_fire(ot)


    # ---------- Batch callbacks ----------
    def register_batch_callback(self, fn):
        """Register function: fn(open_time, snapshot_dict)"""
        self._batch_callbacks.append(fn)

    def _batch_timeout_handler(self, ot: datetime):
        if self._in_backfill:
            return  # skip during backfill
        missing = self.symbols - self._pending.get(ot, set())
        if missing:
            print(f"[timeout] {len(missing)} symbols missing for {ot}, gap-filling...")
            for sym in missing:
                try:
                    bar = self._fetch_rest_bar(sym, ot)
                    if bar:
                        self._bar_cache[sym].append(bar)
                        self._pending[ot].add(sym)
                except Exception as e:
                    print(f"[gap-fill error] {sym} {ot}: {e}")
        self._check_and_fire(ot)

    def _check_and_fire(self, ot: datetime):
        if self.symbols and self._pending[ot] == self.symbols:
            snapshot = {s: list(self._bar_cache[s]) for s in self.symbols}
            for fn in self._batch_callbacks:
                try:
                    fn(ot, snapshot)
                except Exception as e:
                    print(f"[batch callback error] {e}")
            self._pending.pop(ot, None)
            if ot in self._timers:
                self._timers[ot].cancel()
                del self._timers[ot]

    # ---------- REST ----------
    def _rest_url(self) -> str:
        return self.SPOT_REST if self.market == "spot" else self.FUTS_REST

    def _rest_get(self, url, params, timeout=30):
        # Global rate limit
        now = time.time()
        delta = now - self._last_request
        if delta < self._rate_limit:
            time.sleep(self._rate_limit - delta)
        self._last_request = time.time()

        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            print("[rate limit] 429 Too Many Requests, sleeping 60s...")
            time.sleep(60)
            return self._rest_get(url, params, timeout)
        elif r.status_code == 400:
            print(f"[REST error] 400 Bad Request: {r.url} -- continuing")
            return r
        r.raise_for_status()
        return r

    def backfill_symbol(self, symbol: str, start: datetime, end: Optional[datetime] = None):
        self._in_backfill = True
        url = self._rest_url()
        start_ms = to_ms(start)
        end_ms = to_ms(end) if end else None
        limit = 1000
        ts = start_ms

        while True:
            params = {"symbol": symbol.upper(), "interval": self.interval, "limit": limit, "startTime": ts}
            if end_ms:
                params["endTime"] = end_ms

            r = self._rest_get(url, params, timeout=30)
            data = r.json()

            if not data or r.status_code == 400:
                break

            for kl in data:
                bar = {
                    "symbol": symbol,
                    "open_time": from_ms(kl[0]),
                    "open": float(kl[1]),
                    "high": float(kl[2]),
                    "low": float(kl[3]),
                    "close": float(kl[4]),
                    "volume": float(kl[5]),
                    "close_time": from_ms(kl[6]),  # untouched
                    "quote_volume": float(kl[7]),
                    "n_trades": int(kl[8]),
                    "taker_base_vol": float(kl[9]),
                    "taker_quote_vol": float(kl[10]),
                }
                self._append_bar(bar, backfill=True)

            ts = data[-1][0] + 1
            if len(data) < limit:
                break

        self._in_backfill = False

    def backfill_realtime(self, symbols: List[str]):
        """Fill only from the last stored bar forward (overwrite last bar)."""
        self._clean_corrupt_files()
        now = datetime.now(timezone.utc)
        for sym in symbols:
            last = self.latest(sym)
    
            if not last:
                start = datetime(2020, 1, 1, tzinfo=timezone.utc)
                print(f"[backfill-realtime] {sym}: no history, full backfill {start} → {now}")
            else:
                start = last - timedelta(minutes=int(self.interval[:-1]))
                print(f"[backfill-realtime] {sym}: resuming from {start} → {now}")
    
            self.backfill_symbol(sym, start, end=now)

    def _fetch_rest_bar(self, symbol: str, ot: datetime):
        url = self._rest_url()
        start_ms = to_ms(ot)
        end_ms = to_ms(ot + timedelta(minutes=int(self.interval[:-1])) - timedelta(milliseconds=1))
        params = {"symbol": symbol.upper(), "interval": self.interval,
                  "startTime": start_ms, "endTime": end_ms, "limit": 1}
        r = self._rest_get(url, params, timeout=10)
        data = r.json()
        if not data:
            return None
        kl = data[0]
        return {
            "symbol": symbol,
            "open_time": from_ms(kl[0]),
            "open": float(kl[1]),
            "high": float(kl[2]),
            "low": float(kl[3]),
            "close": float(kl[4]),
            "volume": float(kl[5]),
            "close_time": from_ms(kl[6]),  # untouched
            "quote_volume": float(kl[7]),
            "n_trades": int(kl[8]),
            "taker_base_vol": float(kl[9]),
            "taker_quote_vol": float(kl[10]),
        }

    def latest(self, symbol: str) -> Optional[datetime]:
        glob_path = f"{self.base_dir}/symbol={symbol}/dt=*.parquet"
        try:
            df = (pl.scan_parquet(glob_path)
                  .select(pl.col("open_time").max().alias("last_open"))
                  .collect())
            return df["last_open"][0] if df["last_open"][0] else None
        except Exception:
            return None

    def backfill_missing(
        self,
        symbols: List[str],
        start: Optional[datetime] = None
    ):
        """
        Backfill missing bars day-by-day. If start is provided, only check from that date forward.
        """
        self._clean_corrupt_files()
        now = datetime.now(timezone.utc)
        expected_per_day = (24 * 60) // int(self.interval[:-1])  # e.g. 288 for 5m
    
        # default start = 2020-01-01
        start = start or datetime(2020, 1, 1, tzinfo=timezone.utc)
    
        for sym in symbols:
            last = self.latest(sym)
            print(f"[backfill-missing] Checking {sym} {start} → {now}")
    
            glob_path = f"{self.base_dir}/symbol={sym}/dt=*.parquet"
            files = glob.glob(glob_path)
            if files:
                df = (pl.scan_parquet(files)
                      .with_columns(pl.col("open_time").dt.date().alias("date"))
                      .group_by("date")
                      .agg(pl.len().alias("n_bars"))
                      .collect())
            else:
                df = pl.DataFrame({"date": [], "n_bars": []})
    
            complete_days = set(
                df.filter(pl.col("n_bars") == expected_per_day)["date"].to_list()
            )
    
            d = start.date()
            # Fill any incomplete days up to yesterday (if last exists)
            while last and d < last.date():
                if d not in complete_days:
                    day_start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
                    day_end = day_start + timedelta(days=1) - timedelta(milliseconds=1)
                    print(f"[backfill-missing] {sym}: day {d} incomplete, refetching whole day")
                    self.backfill_symbol(sym, day_start, end=day_end)
                d += timedelta(days=1)
    
            # Fill latest day from last known → now
            if last:
                latest_day = last.date()
                latest_start = last + timedelta(milliseconds=1)
                print(f"[backfill-missing] {sym}: backfilling latest {latest_day} from {latest_start} → {now}")
                self.backfill_symbol(sym, latest_start, end=now)
            else:
                print(f"[backfill-missing] {sym}: no history at all, full backfill")
                self.backfill_symbol(sym, start, end=now)


    # ---------- WebSocket ----------
    def _ws_base(self) -> str:
        return self.SPOT_WS if self.market == "spot" else self.FUTS_WS

    def _on_message(self, ws, msg: str):
        try:
            payload = json.loads(msg)
            k = payload.get("data", {}).get("k")
            if not k or not k.get("x"):
                return  # skip if bar not closed
    
            bar = {
                "symbol": k["s"],
                "open_time": from_ms(k["t"]),
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
                "close_time": from_ms(k["T"]),
                "quote_volume": float(k["q"]),
                "n_trades": int(k["n"]),
                "taker_base_vol": float(k["V"]),
                "taker_quote_vol": float(k["Q"]),
            }
    
            # just append bar once
            self._append_bar(bar)
    
        except Exception as e:
            print(f"[WS error in on_message] {e}")


    def _socket_worker(self, shard_syms: List[str]):
        streams = "/".join([f"{s.lower()}@kline_{self.interval}" for s in shard_syms])
        url = self._ws_base() + streams
        while not self._stop_event.is_set():
            try:
                ws = websocket.WebSocketApp(url, on_message=self._on_message)
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print(f"[WS worker exception] {e}")
            time.sleep(2 + random.random()*3)

    def _populate_cache(self, symbols: List[str]):
        for sym in symbols:
            glob_path = f"{self.base_dir}/symbol={sym}/dt=*.parquet"
            files = glob.glob(glob_path)
            if not files:
                continue
            try:
                df = (pl.scan_parquet(files)
                      .sort("open_time", descending=True)
                      .limit(self._cache_len)
                      .collect())
                bars = df.to_dicts()
                self._bar_cache[sym].clear()
                for bar in reversed(bars):
                    self._bar_cache[sym].append(bar)
                print(f"[cache] {sym}: populated with {len(bars)} bars")
            except Exception as e:
                print(f"[cache error] {sym}: {e}")

    def live(self, symbols: List[str], shard_size: int = 50, resume: bool = True):
        if resume:
            self.backfill_realtime(symbols)

        self._populate_cache(symbols)
    
        self._stop_event.clear()
        shards = [symbols[i:i+shard_size] for i in range(0, len(symbols), shard_size)]
        print(f"[LIVE] {len(shards)} shards for {len(symbols)} symbols")
    
        for shard in shards:
            t = threading.Thread(target=self._socket_worker, args=(shard,))
            t.start()
            self._ws_threads.append(t)
    
        try:
            for t in self._ws_threads:
                t.join()
        except KeyboardInterrupt:
            print("[LIVE] stopped by user")
            self.stop()

    def stop(self):
        self._stop_event.set()
        for t in self._ws_threads:
            t.join(timeout=5)
        self._ws_threads.clear()

    # ---------- Loader ----------
    def load_pandas(
        self,
        symbols: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ):
        """
        Load bars into a Pandas DataFrame with MultiIndex (symbol, open_time),
        scanning all symbols in one pass.
        """
        glob_path = f"{self.base_dir}/symbol=*/dt=*.parquet"
        try:
            df = pl.scan_parquet(glob_path)
        except Exception as e:
            print(f"[error] parquet scan failed: {e}")
            return None

        if symbols:
            df = df.filter(pl.col("symbol").is_in(symbols))
        if start:
            df = df.filter(pl.col("open_time") >= start)
        if end:
            df = df.filter(pl.col("open_time") <= end)

        # Collect to pandas
        pdf = df.collect().to_pandas()

        # 🔹 Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(pdf["open_time"]):
            pdf["open_time"] = pd.to_datetime(pdf["open_time"], utc=True)

        # 🔹 Drop duplicate (symbol, open_time) rows, keep latest
        pdf = pdf.drop_duplicates(subset=["symbol", "open_time"], keep="last")

        # 🔹 Set MultiIndex
        pdf = pdf.set_index(["symbol", "open_time"]).sort_index()

        return pdf


    def _clean_corrupt_files(self):
        bad_files = []
        for f in glob.glob(f"{self.base_dir}/symbol=*/dt=*.parquet"):
            try:
                pl.read_parquet(f, n_rows=1)
            except Exception as e:
                print(f"[corrupt] {os.path.basename(f)}: {e}, deleting...")
                bad_files.append(f)
        for f in bad_files:
            os.remove(f)
        if bad_files:
            print(f"[cleanup] Deleted {len(bad_files)} corrupt parquet files")
