#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import random
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import requests
import polars as pl
import websocket  # pip install websocket-client
import pandas as pd
import numpy as np

from .base_data_manager import BaseDataManager, to_ms, from_ms, day_str


# -----------------------------
# BinanceDataManager
# -----------------------------

class BinanceDataManager(BaseDataManager):
    """
    Binance market data manager with WebSocket streaming support.
    
    Features:
      - Partitioned Parquet storage (inherited)
      - Backfill via REST API
      - Auto-resume before live streaming
      - WebSocket sharded streaming
      - Batch callback once per bar close across all tickers
      - Timeout + REST gap-fill for missing bars
      - Hot cache of recent bars (inherited)
    """

    SPOT_REST = "https://data-api.binance.vision/api/v3/klines"
    FUTS_REST = "https://fapi.binance.com/fapi/v1/klines"
    SPOT_WS   = "wss://stream.binance.com:9443/stream?streams="
    FUTS_WS   = "wss://fstream.binance.com/stream?streams="

    def __init__(
        self,
        base_dir: str = "parquet_data_binance",  # Source-specific default
        interval: str = "5m",
        market: str = "futures",  # "spot" or "futures"
        symbols: Optional[List[str]] = None,
        cache_len: int = 1000,
        batch_timeout: int = 20,
        rate_limit: float = 0.5  # seconds between REST calls
    ):
        # Initialize base class
        super().__init__(base_dir=base_dir, interval=interval, symbols=symbols, cache_len=cache_len)
        
        # Binance-specific attributes
        self.market = market.lower()
        self.batch_timeout = batch_timeout
        self._rate_limit = rate_limit

        # Batch callback logic
        self._batch_callbacks = []
        self._pending: Dict[datetime, set[str]] = defaultdict(set)
        self._timers: Dict[datetime, threading.Timer] = {}

        # WebSocket state
        self._ws_threads: List[threading.Thread] = []
        self._stop_event = threading.Event()

        # Rate throttling
        self._last_request = 0

        # Backfill flag
        self._in_backfill = False

    # ---------- Override batch callback hook ----------
    def _on_bar_appended(self, bar: dict, ot: datetime):
        """
        Called when a bar is appended in non-backfill mode.
        Implements batch callback logic for WebSocket streaming.
        """
        sym = bar["symbol"]
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

    # ---------- REST API ----------
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

    def _parse_binance_kline(self, kl: list, symbol: str) -> dict:
        """
        Parse Binance kline array into standardized bar format.
        
        Binance kline format:
        [
            0: open_time,
            1: open,
            2: high,
            3: low,
            4: close,
            5: volume,
            6: close_time,
            7: quote_volume,
            8: n_trades,
            9: taker_base_vol,
            10: taker_quote_vol,
            11: ignore
        ]
        """
        return {
            "symbol": symbol,
            "open_time": from_ms(kl[0]),
            "open": float(kl[1]),
            "high": float(kl[2]),
            "low": float(kl[3]),
            "close": float(kl[4]),
            "volume": float(kl[5]),
            "quote_volume": float(kl[7]),
            "n_trades": int(kl[8]),
        }

    # ---------- Backfill (REST) ----------
    def backfill_symbol(self, symbol: str, start: datetime, end: Optional[datetime] = None):
        """Backfill historical data for a single symbol via REST API."""
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
                bar = self._parse_binance_kline(kl, symbol)
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
            import glob
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

    def _fetch_rest_bar(self, symbol: str, ot: datetime):
        """Fetch a single bar via REST API for gap filling."""
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
        return self._parse_binance_kline(kl, symbol)

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
                "quote_volume": float(k["q"]),
                "n_trades": int(k["n"]),
            }

            # Append bar once
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

    def live(self, symbols: List[str], shard_size: int = 50, resume: bool = True):
        """
        Start live WebSocket streaming for symbols.
        
        Args:
            symbols: List of symbols to stream
            shard_size: Number of symbols per WebSocket connection
            resume: If True, backfill from last bar to now before streaming
        """
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
        """Stop all WebSocket connections."""
        self._stop_event.set()
        for t in self._ws_threads:
            t.join(timeout=5)
        self._ws_threads.clear()

    # ---------- Live Snapshot (fetch from API) ----------
    def get_live_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch latest closed bar for each symbol via REST API.
        Always fetches from remote - no local storage dependency.
        
        Returns:
            MultiIndex DataFrame (symbol, open_time) with standardized columns
        """
        bars = []
        
        for sym in symbols:
            # Calculate most recent completed bar time
            now = datetime.now(timezone.utc)
            bar_minutes = int(self.interval[:-1])
            minutes_past = now.minute % bar_minutes
            most_recent_bar_start = now.replace(
                minute=now.minute - minutes_past,
                second=0,
                microsecond=0
            ) - timedelta(minutes=bar_minutes)
            
            # Fetch via REST
            bar = self._fetch_rest_bar(sym, most_recent_bar_start)
            if bar:
                bars.append(bar)
        
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
        
        NOTE: Binance API has a 1000-bar limit per request. For large lookbacks
        (>16 hours), this method makes multiple paginated requests.
        
        Args:
            symbol: Symbol to fetch
            hours_back: Hours of historical 1-minute data to fetch
            
        Returns:
            DataFrame with 1-minute bars (open_time index, OHLCV columns)
        """
        # Calculate start and end time
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours_back)
        start_ms = to_ms(start)
        end_ms = to_ms(now)
        
        # Fetch 1-minute bars with pagination (Binance limit = 1000 per request)
        url = self._rest_url()
        limit = 1000
        all_bars = []
        current_start_ms = start_ms
        
        while current_start_ms < end_ms:
            params = {
                "symbol": symbol.upper(),
                "interval": "1m",  # Always fetch 1-minute bars
                "startTime": current_start_ms,
                "endTime": end_ms,
                "limit": limit
            }
            
            r = self._rest_get(url, params, timeout=30)
            data = r.json()
            
            if not data:
                break
            
            # Parse bars
            for kl in data:
                bar = self._parse_binance_kline(kl, symbol)
                all_bars.append(bar)
            
            # If we got fewer bars than limit, we're done
            if len(data) < limit:
                break
            
            # Update start time for next iteration (last bar time + 1ms)
            current_start_ms = data[-1][0] + 1
        
        if not all_bars:
            return pd.DataFrame()
        
        # Return as DataFrame with open_time index
        df = pd.DataFrame(all_bars)
        df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
        df = df.set_index('open_time').sort_index()
        return df[['open', 'high', 'low', 'close', 'volume']]
