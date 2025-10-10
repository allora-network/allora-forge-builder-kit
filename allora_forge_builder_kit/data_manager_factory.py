#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Manager Factory

Provides a simple string-based API for creating data managers:
    dm = DataManager(source="binance", interval="5m", market="futures")
    dm = DataManager(source="allora", interval="5m", api_key="...")

This is a factory function that returns the appropriate data manager instance.
"""

from typing import Optional, List
from .binance_data_manager import BinanceDataManager


def DataManager(
    source: str = "binance",
    base_dir: Optional[str] = None,
    interval: str = "5m",
    symbols: Optional[List[str]] = None,
    cache_len: int = 1000,
    **kwargs
):
    """
    Factory function to create data managers from a source string.
    
    Args:
        source: Data source ("binance" or "allora")
        base_dir: Directory for Parquet storage (auto-set based on source if None)
        interval: Bar interval (e.g. "5m", "1h")
        symbols: List of symbols to manage
        cache_len: Number of recent bars to cache
        **kwargs: Source-specific parameters:
            - Binance: market="futures|spot", batch_timeout=20, rate_limit=0.5
            - Allora: api_key="...", max_pages=1000
    
    Returns:
        Instance of BinanceDataManager or AlloraDataManager
        
    Examples:
        # Binance
        dm = DataManager(source="binance", interval="5m", market="futures")
        
        # Allora
        dm = DataManager(source="allora", interval="5m", api_key="your-key")
        
    Raises:
        ValueError: If source is not recognized
    """
    source_lower = source.lower()
    
    if source_lower == "binance":
        # Set default base_dir if not provided
        if base_dir is None:
            base_dir = "parquet_data_binance"
        
        # Extract Binance-specific params
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
            rate_limit=rate_limit
        )
    
    elif source_lower == "allora":
        # Import here to avoid circular imports
        from .allora_data_manager import AlloraDataManager
        
        # Set default base_dir if not provided
        if base_dir is None:
            base_dir = "parquet_data_allora"
        
        # Extract Allora-specific params
        api_key = kwargs.pop("api_key", None)
        if api_key is None:
            raise ValueError("api_key is required for Allora data source")
        
        max_pages = kwargs.pop("max_pages", 1000)
        sleep_sec = kwargs.pop("sleep_sec", 0.1)
        
        return AlloraDataManager(
            base_dir=base_dir,
            interval=interval,
            symbols=symbols,
            cache_len=cache_len,
            api_key=api_key,
            max_pages=max_pages,
            sleep_sec=sleep_sec
        )
    
    else:
        raise ValueError(
            f"Unknown data source: '{source}'. "
            f"Supported sources: 'binance', 'allora'"
        )


# Convenience function to list available sources
def list_data_sources():
    """List all available data sources."""
    sources = {
        "binance": {
            "description": "Binance exchange (spot & futures)",
            "features": ["WebSocket streaming", "Batch callbacks", "REST API"],
            "parameters": ["market", "batch_timeout", "rate_limit"],
        },
        "allora": {
            "description": "Allora Network market data API",
            "features": ["Monthly buckets", "Paginated API", "REST API"],
            "parameters": ["api_key", "max_pages", "sleep_sec"],
        }
    }
    return sources
