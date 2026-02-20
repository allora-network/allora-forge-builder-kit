#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Manager Factory

Provides a simple string-based API for creating data managers::

    dm = DataManager(source="binance", interval="5m", market="futures")
    dm = DataManager(source="allora", interval="5m", api_key="...")
    dm = DataManager(source="atlas", interval="5m", api_key="...")

``"allora"`` now defaults to the Atlas backend.  Use ``"allora-legacy"``
to explicitly use the old Allora Network REST API.
"""

from typing import Optional, List
from .binance_data_manager import BinanceDataManager


def DataManager(
    source: str = "binance",
    base_dir: Optional[str] = None,
    interval: str = "5m",
    symbols: Optional[List[str]] = None,
    cache_len: int = 1000,
    **kwargs,
):
    """
    Factory function to create data managers from a source string.

    Args:
        source: Data source — one of ``"binance"``, ``"allora"`` (Atlas),
            ``"atlas"`` (explicit), or ``"allora-legacy"`` (old API).
        base_dir: Directory for Parquet storage (auto-set based on source).
        interval: Bar interval (e.g. ``"5m"``, ``"1h"``).
        symbols: List of symbols to manage.
        cache_len: Number of recent bars to cache in memory.
        **kwargs: Source-specific parameters:
            - **Binance**: ``market``, ``batch_timeout``, ``rate_limit``
            - **Allora / Atlas**: ``api_key``, ``base_url``, ``page_size``
            - **Allora-legacy**: ``api_key``, ``max_pages``, ``sleep_sec``

    Returns:
        A ``BaseDataManager`` subclass instance.

    Examples::

        dm = DataManager(source="binance", interval="5m", market="futures")
        dm = DataManager(source="allora", interval="5m", api_key="UP-...")
        dm = DataManager(source="atlas", interval="5m", api_key="UP-...")
    """
    source_lower = source.lower().strip()

    # ── Binance ──────────────────────────────────────────────────────
    if source_lower == "binance":
        if base_dir is None:
            base_dir = "parquet_data_binance"

        market = kwargs.pop("market", "futures")
        batch_timeout = kwargs.pop("batch_timeout", 20)
        rate_limit = kwargs.pop("rate_limit", 0.5)

        return BinanceDataManager(
            base_dir=base_dir,
            interval=interval,
            market=market,
            symbols=symbols,
            cache_len=cache_len,
            batch_timeout=batch_timeout,
            rate_limit=rate_limit,
        )

    # ── Atlas / Allora (new default) ─────────────────────────────────
    if source_lower in ("allora", "atlas"):
        from .atlas_data_manager import AtlasDataManager

        if base_dir is None:
            base_dir = "parquet_data_allora"

        api_key = kwargs.pop("api_key", None)
        if api_key is None:
            raise ValueError("api_key is required for the Atlas/Allora data source")

        base_url = kwargs.pop("base_url", None)
        page_size = kwargs.pop("page_size", 250)
        sleep_sec = kwargs.pop("sleep_sec", 0.05)

        ctor_kwargs = dict(
            api_key=api_key,
            base_dir=base_dir,
            interval=interval,
            symbols=symbols,
            cache_len=cache_len,
            page_size=page_size,
            sleep_sec=sleep_sec,
        )
        if base_url is not None:
            ctor_kwargs["base_url"] = base_url

        return AtlasDataManager(**ctor_kwargs)

    # ── Allora legacy API ────────────────────────────────────────────
    if source_lower == "allora-legacy":
        from .allora_data_manager import AlloraDataManager

        if base_dir is None:
            base_dir = "parquet_data_allora"

        api_key = kwargs.pop("api_key", None)
        if api_key is None:
            raise ValueError("api_key is required for the Allora data source")

        max_pages = kwargs.pop("max_pages", 1000)
        sleep_sec = kwargs.pop("sleep_sec", 0.1)

        return AlloraDataManager(
            base_dir=base_dir,
            interval=interval,
            symbols=symbols,
            cache_len=cache_len,
            api_key=api_key,
            max_pages=max_pages,
            sleep_sec=sleep_sec,
        )

    raise ValueError(
        f"Unknown data source: '{source}'.  "
        f"Supported: 'binance', 'allora', 'atlas', 'allora-legacy'"
    )


def list_data_sources():
    """List all available data sources and their parameters."""
    return {
        "binance": {
            "description": "Binance exchange (spot & futures)",
            "features": ["WebSocket streaming", "Batch callbacks", "REST API"],
            "parameters": ["market", "batch_timeout", "rate_limit"],
        },
        "allora": {
            "description": "Atlas data service (Tiingo 1-min candles via forge-data.allora.run)",
            "features": ["Bulk download", "Paginated API", "Dataset discovery", "SSE streaming"],
            "parameters": ["api_key", "base_url", "page_size"],
            "note": "This is the recommended data source for Allora topics.",
        },
        "atlas": {
            "description": "Alias for 'allora' — Atlas data service",
            "parameters": ["api_key", "base_url", "page_size"],
        },
        "allora-legacy": {
            "description": "Legacy Allora Network REST API (deprecated)",
            "features": ["Monthly buckets", "Paginated API"],
            "parameters": ["api_key", "max_pages", "sleep_sec"],
        },
    }
