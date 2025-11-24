#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import List, Dict, Optional

import polars as pl
import pandas as pd


# -----------------------------
# Utility Functions
# -----------------------------

def to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def from_ms(ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def day_str(dt: datetime) -> str:
    """Format datetime as YYYY-MM-DD string."""
    return dt.strftime("%Y-%m-%d")


# -----------------------------
# BaseDataManager
# -----------------------------

class BaseDataManager(ABC):
    """
    Abstract base class for data managers.
    
    All implementations must:
    - Store data in partitioned Parquet files
    - Implement backfill methods to sync remote → local
    - Provide load_pandas/load_polars for querying local storage
    - Implement get_live_snapshot to fetch latest data from remote API
    
    Standardized bar format:
    {
        "symbol": str,
        "open_time": datetime (UTC),
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float,
        "quote_volume": float,  # Optional (NaN if not available)
        "n_trades": int,
    }
    """

    def __init__(
        self,
        base_dir: str = "parquet_data",
        interval: str = "5m",
        symbols: Optional[List[str]] = None,
        cache_len: int = 1000,
        **kwargs  # Allow subclass-specific parameters (e.g., market, api_key)
    ):
        """
        Initialize base data manager.
        
        Args:
            base_dir: Directory for Parquet storage
            interval: Bar interval (e.g. "5m", "1h")
            symbols: List of symbols to manage
            cache_len: Number of recent bars to keep in memory
            **kwargs: Subclass-specific parameters (ignored in base class)
        """
        self.base_dir = base_dir
        self.interval = interval
        self.symbols = set(symbols or [])
        os.makedirs(self.base_dir, exist_ok=True)

        # Hot cache of recent bars
        self._cache_len = cache_len
        self._bar_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=cache_len))

    # ---------- File paths ----------
    @staticmethod
    def _normalize_symbol_for_path(symbol: str) -> str:
        r"""
        Normalize symbol for use in file system paths.
        Removes characters that could cause path issues: / \ : * ? " < > |
        
        Examples:
            "BTC/USD" -> "BTCUSD"
            "BTC-USD" -> "BTCUSD"
        """
        return symbol.replace("/", "").replace("\\", "").replace(":", "").replace("-", "").replace(" ", "")
    
    def _partition_path(self, symbol: str, dt_str: str) -> str:
        """Get path for partitioned Parquet file."""
        safe_symbol = self._normalize_symbol_for_path(symbol)
        return f"{self.base_dir}/symbol={safe_symbol}/dt={dt_str}.parquet"

    # ---------- Storage operations (shared) ----------
    def _append_bar(self, bar: dict, backfill: bool = False):
        """
        Append a bar to Parquet storage and cache.
        Shared implementation for all data managers.
        
        Args:
            bar: Standardized bar dict (8 fields)
            backfill: If True, skip batch logic (for historical data)
        """
        sym = bar["symbol"]
        ot = bar["open_time"]

        # Write to Parquet
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

        # Update cache with dedupe
        if self._bar_cache[sym] and self._bar_cache[sym][-1]["open_time"] == ot:
            self._bar_cache[sym][-1] = bar
        else:
            self._bar_cache[sym].append(bar)

        # Subclasses may implement batch callback logic here
        if not backfill:
            self._on_bar_appended(bar, ot)

    def _on_bar_appended(self, bar: dict, ot: datetime):
        """
        Hook for subclasses to implement batch callback logic.
        Called when a bar is appended in non-backfill mode.
        Default implementation does nothing.
        """
        pass

    def _populate_cache(self, symbols: List[str]):
        """Load most recent bars from Parquet into cache."""
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

    def _clean_corrupt_files(self):
        """Detect and remove corrupt Parquet files."""
        all_files = glob.glob(f"{self.base_dir}/symbol=*/dt=*.parquet")
        bad_files = []
        
        if all_files:
            print(f"[cleanup] Checking {len(all_files)} parquet files for corruption...")
        
        for f in all_files:
            try:
                pl.read_parquet(f, n_rows=1)
            except Exception as e:
                print(f"[corrupt] {os.path.basename(f)}: {e}, deleting...")
                bad_files.append(f)
        
        for f in bad_files:
            os.remove(f)
        
        if bad_files:
            print(f"[cleanup] Deleted {len(bad_files)} corrupt parquet files")
        elif all_files:
            print(f"[cleanup] All {len(all_files)} parquet files are valid")

    # ---------- Backfill (abstract - implementation-specific) ----------
    @abstractmethod
    def backfill_symbol(self, symbol: str, start: datetime, end: Optional[datetime] = None):
        """
        Backfill historical data for a single symbol.
        
        Args:
            symbol: Symbol to backfill
            start: Start datetime (UTC)
            end: End datetime (UTC), defaults to now
        """
        pass

    @abstractmethod
    def backfill_realtime(self, symbols: List[str]):
        """
        Sync latest data from last stored bar to now.
        Overwrites the last bar to ensure clean continuation.
        
        Args:
            symbols: List of symbols to sync
        """
        pass

    @abstractmethod
    def backfill_missing(self, symbols: List[str], start: Optional[datetime] = None):
        """
        Detect gaps in local storage and backfill missing data.
        
        Args:
            symbols: List of symbols to check
            start: Start date for gap detection (defaults to 2020-01-01)
        """
        pass

    # ---------- Query local storage (shared implementation) ----------
    def latest(self, symbol: str) -> Optional[datetime]:
        """Get timestamp of most recent bar for a symbol."""
        safe_symbol = self._normalize_symbol_for_path(symbol)
        glob_path = f"{self.base_dir}/symbol={safe_symbol}/dt=*.parquet"
        try:
            df = (pl.scan_parquet(glob_path)
                  .select(pl.col("open_time").max().alias("last_open"))
                  .collect())
            return df["last_open"][0] if df["last_open"][0] else None
        except Exception:
            return None

    def load_pandas(
        self,
        symbols: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load bars into a Pandas DataFrame with MultiIndex (symbol, open_time).
        
        Args:
            symbols: List of symbols to load
            start: Start datetime filter (UTC)
            end: End datetime filter (UTC)
            
        Returns:
            MultiIndex DataFrame with columns: open, high, low, close, volume, quote_volume, n_trades
        """
        glob_path = f"{self.base_dir}/symbol=*/dt=*.parquet"
        try:
            # Handle missing columns - Allora data doesn't have quote_volume
            df = pl.scan_parquet(glob_path, missing_columns="insert")
        except Exception as e:
            print(f"[error] parquet scan failed: {e}")
            return pd.DataFrame()

        if symbols:
            df = df.filter(pl.col("symbol").is_in(symbols))
        if start:
            df = df.filter(pl.col("open_time") >= start)
        if end:
            df = df.filter(pl.col("open_time") <= end)

        # Collect to pandas
        pdf = df.collect().to_pandas()
        
        # Ensure standard columns exist (Allora data has different naming)
        if "quote_volume" not in pdf.columns:
            pdf["quote_volume"] = None
        
        # Map trades_done to n_trades for consistency
        if "trades_done" in pdf.columns and "n_trades" not in pdf.columns:
            pdf["n_trades"] = pdf["trades_done"]
        elif "n_trades" not in pdf.columns:
            pdf["n_trades"] = None

        if pdf.empty:
            return pd.DataFrame()

        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(pdf["open_time"]):
            pdf["open_time"] = pd.to_datetime(pdf["open_time"], utc=True)

        # Drop duplicate (symbol, open_time) rows, keep latest
        pdf = pdf.drop_duplicates(subset=["symbol", "open_time"], keep="last")

        # Set MultiIndex
        pdf = pdf.set_index(["symbol", "open_time"]).sort_index()

        return pdf

    def load_polars(
        self,
        symbols: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pl.DataFrame:
        """
        Load bars into a Polars DataFrame.
        
        Args:
            symbols: List of symbols to load
            start: Start datetime filter (UTC) - can be datetime object or string like "2020-01-01"
            end: End datetime filter (UTC) - can be datetime object or string like "2020-01-01"
            
        Returns:
            Polars DataFrame with columns: symbol, open_time, open, high, low, close, volume, quote_volume, n_trades
        """
        from datetime import datetime
        import pytz
        
        # Convert string dates to timezone-aware datetime objects
        if start and isinstance(start, str):
            start = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
        if end and isinstance(end, str):
            end = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
        
        glob_path = f"{self.base_dir}/symbol=*/dt=*.parquet"
        try:
            # Handle missing columns - Allora data doesn't have quote_volume
            df = pl.scan_parquet(glob_path, missing_columns="insert")
        except Exception as e:
            print(f"[error] parquet scan failed: {e}")
            return pl.DataFrame()

        if symbols:
            df = df.filter(pl.col("symbol").is_in(symbols))
        if start:
            df = df.filter(pl.col("open_time") >= start)
        if end:
            df = df.filter(pl.col("open_time") <= end)

        pdf = df.collect()
        
        # Ensure standard columns exist (Allora data has different naming)
        if "quote_volume" not in pdf.columns:
            pdf = pdf.with_columns([pl.lit(None).cast(pl.Float64).alias("quote_volume")])
        
        # Map trades_done to n_trades for consistency
        if "trades_done" in pdf.columns and "n_trades" not in pdf.columns:
            pdf = pdf.with_columns([pl.col("trades_done").alias("n_trades")])
        elif "n_trades" not in pdf.columns:
            pdf = pdf.with_columns([pl.lit(None).cast(pl.Int64).alias("n_trades")])

        # Drop duplicate (symbol, open_time) rows, keep latest
        pdf = pdf.unique(subset=["symbol", "open_time"], keep="last")
        pdf = pdf.sort(["symbol", "open_time"])

        return pdf

    # ---------- Live snapshot (abstract - always fetch from remote) ----------
    @abstractmethod
    def get_live_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch the most recent OHLCV bar(s) directly from remote API.
        
        IMPORTANT: Always fetches from remote - no local storage dependency.
        This allows the function to work in pickled/isolated contexts.
        
        Args:
            symbols: List of symbols to fetch
            
        Returns:
            MultiIndex DataFrame (symbol, open_time) with columns:
            open, high, low, close, volume, quote_volume, n_trades
        """
        pass

    @abstractmethod
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
        pass

    # ---------- Optional: Live streaming (Binance-specific) ----------
    def register_batch_callback(self, fn):
        """
        Register a callback function for live streaming.
        
        NOTE: This is optional functionality (mainly for Binance WebSocket streaming).
        Implementations without live streaming support should raise NotImplementedError.
        
        Args:
            fn: Callback function(open_time, snapshot_dict)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support live streaming callbacks. "
            "This feature is only available in WebSocket-based implementations (e.g., BinanceDataManager)."
        )

    def live(self, symbols: List[str], **kwargs):
        """
        Start live streaming for symbols.
        
        NOTE: This is optional functionality (mainly for Binance WebSocket streaming).
        Implementations without live streaming support should raise NotImplementedError.
        
        Args:
            symbols: List of symbols to stream
            **kwargs: Implementation-specific parameters
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support live streaming. "
            "This feature is only available in WebSocket-based implementations (e.g., BinanceDataManager). "
            "Consider using backfill_realtime() + get_live_snapshot() for polling-based updates."
        )
