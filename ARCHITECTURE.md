# Allora Forge Builder Kit - Data Manager Architecture

**Version:** 2.0  
**Date:** October 2025  
**Status:** Production Ready ✅

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Quick Start](#quick-start)
5. [Data Manager Selection](#data-manager-selection)
6. [Standardized Bar Format](#standardized-bar-format)
7. [Storage Strategy](#storage-strategy)
8. [Testing](#testing)
9. [API Reference](#api-reference)
10. [Migration Guide](#migration-guide)

---

## Overview

The Allora Forge Builder Kit now features a **modular, extensible data management architecture** that supports multiple market data sources through a unified interface. The system is built on object-oriented principles with abstract base classes, concrete implementations, and a factory pattern for easy instantiation.

### Supported Data Sources

- ✅ **Binance** (Spot & Futures) - REST API + WebSocket streaming
- ✅ **Allora Network** - REST API with monthly bucket support
- 🔧 **Extensible** - Easy to add new data sources

---

## Architecture

### Class Hierarchy

```
BaseDataManager (Abstract)
├── BinanceDataManager (Concrete)
└── AlloraDataManager (Concrete)

DataManager() Factory Function
```

### Components

#### 1. **BaseDataManager** (`base_data_manager.py`)
- **Role:** Abstract base class defining the common interface
- **Responsibilities:**
  - Partitioned Parquet storage (by symbol and date)
  - Hot cache of recent bars (deque-based)
  - Data loading (Pandas & Polars)
  - Standardized bar format
  - Symbol normalization for file paths

**Shared Methods:**
- `load_pandas(symbols, start, end)` - Load data as Pandas DataFrame
- `load_polars(symbols, start, end)` - Load data as Polars DataFrame
- `latest(symbol)` - Get most recent bar timestamp
- `_append_bar(bar)` - Append bar to storage and cache
- `_write_bars(df, symbol)` - Bulk write bars to Parquet

**Abstract Methods (must be implemented):**
- `get_live_snapshot(symbols)` - Fetch latest bars from API
- `backfill_symbol(symbol, start, end)` - Historical data sync
- `backfill_realtime(symbols)` - Sync to current time
- `backfill_missing(symbols, start)` - Fill gaps in storage

#### 2. **BinanceDataManager** (`binance_data_manager.py`)
- **Role:** Binance-specific implementation
- **Features:**
  - REST API for historical data (1000 bars per request)
  - WebSocket streaming for real-time updates
  - Batch callbacks (trigger when all symbols complete a bar)
  - Support for Spot and Futures markets
  - Rate limiting

**Default Storage:** `parquet_data_binance/`

**Unique Methods:**
- `register_batch_callback(fn)` - Register callback for WebSocket updates
- `live(symbols)` - Start WebSocket streaming

#### 3. **AlloraDataManager** (`allora_data_manager.py`)
- **Role:** Allora Network-specific implementation
- **Features:**
  - REST API with pagination (1-minute bars)
  - Monthly bucket downloads for bulk historical data
  - Automatic ticker normalization (BTC/USD → btcusd)
  - Aggregation from 1-min to any interval (5m, 15m, etc.)

**Default Storage:** `parquet_data_allora/`

**Unique Methods:**
- `_fetch_ohlcv_data(ticker, from_date)` - Paginated API fetches
- `_list_ready_buckets(ticker, from_month)` - Monthly bulk data
- `_create_interval_bars(df_1min)` - Aggregate to target interval

#### 4. **DataManager Factory** (`data_manager_factory.py`)
- **Role:** User-friendly instantiation
- **Function:** `DataManager(source="binance", **kwargs)`

**Example:**
```python
# Binance
dm = DataManager(source="binance", interval="5m", symbols=["BTCUSDT"])

# Allora
dm = DataManager(source="allora", api_key="your-key", interval="5m", symbols=["BTC/USD"])
```

---

## Key Features

### 1. **Unified Interface**
All data managers share the same core methods:
```python
dm.backfill_symbol(symbol, start, end)
dm.get_live_snapshot(symbols)
dm.load_pandas(symbols, start, end)
dm.load_polars(symbols, start, end)
```

### 2. **Partitioned Parquet Storage**
- **Structure:** `base_dir/symbol=BTCUSD/dt=2025-10-06.parquet`
- **Benefits:** 
  - Fast queries (only read relevant partitions)
  - Efficient storage (columnar format with compression)
  - Easy date-based operations

### 3. **Hot Cache**
- Recent bars cached in memory (deque)
- Configurable size (default: 1000 bars)
- Reduces disk I/O for repeated queries

### 4. **Storage Isolation**
Different data sources use separate directories by default:
- Binance: `parquet_data_binance/`
- Allora: `parquet_data_allora/`

Prevents data collision when using multiple sources.

### 5. **Symbol Normalization**
- **File Paths:** `BTC/USD` → `BTCUSD` (removes `/` to prevent directory issues)
- **API Calls:** Automatic format conversion per data source
  - Binance: `BTCUSDT`
  - Allora: `btcusd`

### 6. **Flexible Integration**
Works seamlessly with `AlloraMLWorkflow`:
```python
# Simple string-based API
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=24,
    number_of_input_candles=24,
    target_length=16,
    data_source="binance"  # or "allora"
)

# Advanced: explicit data manager
custom_dm = BinanceDataManager(interval="1h", market="spot")
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=24,
    number_of_input_candles=24,
    target_length=16,
    data_manager=custom_dm
)
```

---

## Quick Start

### Installation

```bash
cd "Allora/ML Workflow"
pip install -e .
```

### Basic Usage

#### Binance (No API Key Required)

```python
from allora_forge_builder_kit import DataManager
from datetime import datetime, timedelta, timezone

# Create Binance data manager
dm = DataManager(
    source="binance",
    interval="5m",
    symbols=["BTCUSDT", "ETHUSDT"],
    market="futures"
)

# Backfill last 7 days
start = datetime.now(timezone.utc) - timedelta(days=7)
dm.backfill_missing(["BTCUSDT", "ETHUSDT"], start=start)

# Load data
df = dm.load_pandas(["BTCUSDT"], start=start)
print(df.head())

# Get live snapshot
snapshot = dm.get_live_snapshot(["BTCUSDT", "ETHUSDT"])
print(snapshot)
```

#### Allora Network (API Key Required)

```python
from allora_forge_builder_kit import DataManager

# Create Allora data manager
dm = DataManager(
    source="allora",
    api_key="your-allora-api-key",
    interval="5m",
    symbols=["BTC/USD", "ETH/USD"]
)

# Backfill last 7 days
start = datetime.now(timezone.utc) - timedelta(days=7)
dm.backfill_missing(["BTC/USD"], start=start)

# Load data
df = dm.load_pandas(["BTC/USD"], start=start)
print(df.head())

# Get live snapshot
snapshot = dm.get_live_snapshot(["BTC/USD"])
print(snapshot)
```

#### ML Workflow Integration

```python
from allora_forge_builder_kit import AlloraMLWorkflow
from datetime import datetime, timedelta, timezone

# Create workflow with Binance data
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT", "ETHUSDT"],
    hours_needed=24,
    number_of_input_candles=24,
    target_length=16,
    interval="5m",
    data_source="binance",
    market="futures"
)

# Backfill data
start = datetime.now(timezone.utc) - timedelta(days=3)
workflow.backfill(start=start)

# Extract features
df_features = workflow.get_full_feature_target_dataframe_pandas(start_date=start)
print(df_features.head())

# Get live features
live_features = workflow.get_live_features()
print(live_features)
```

---

## Data Manager Selection

### Via Factory (Recommended)

```python
from allora_forge_builder_kit import DataManager

# Binance
dm = DataManager(source="binance", interval="5m", market="futures")

# Allora
dm = DataManager(source="allora", api_key="key", interval="5m")
```

### Direct Instantiation (Advanced)

```python
from allora_forge_builder_kit import BinanceDataManager, AlloraDataManager

# Binance with custom settings
dm_binance = BinanceDataManager(
    base_dir="my_binance_data",
    interval="15m",
    market="spot",
    symbols=["BTCUSDT"],
    cache_len=2000,
    batch_timeout=30,
    rate_limit=0.3
)

# Allora with custom settings
dm_allora = AlloraDataManager(
    api_key="your-key",
    base_dir="my_allora_data",
    interval="5m",
    symbols=["BTC/USD"],
    cache_len=1500,
    max_pages=500,
    sleep_sec=0.05
)
```

### In AlloraMLWorkflow

```python
from allora_forge_builder_kit import AlloraMLWorkflow

# Simple string API
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=24,
    number_of_input_candles=24,
    target_length=16,
    data_source="allora",  # ← Simple!
    api_key="your-key"
)

# Advanced: pass explicit manager
from allora_forge_builder_kit import AlloraDataManager
custom_dm = AlloraDataManager(api_key="key", interval="1h")
workflow = AlloraMLWorkflow(
    tickers=["BTC/USD"],
    hours_needed=24,
    number_of_input_candles=24,
    target_length=16,
    data_manager=custom_dm  # ← Advanced!
)
```

---

## Standardized Bar Format

All data managers return data in a consistent format:

### Dictionary Format (Single Bar)

```python
{
    "symbol": str,          # e.g., "BTCUSDT" or "BTC/USD"
    "open_time": datetime,  # UTC timezone-aware
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": float,        # Base asset volume
    "quote_volume": float,  # Quote asset volume (NaN if unavailable)
    "n_trades": int,        # Number of trades
}
```

### DataFrame Format (MultiIndex)

```python
# Index: (symbol, open_time)
# Columns: open, high, low, close, volume, quote_volume, n_trades

                              open      high       low     close    volume  quote_volume  n_trades
symbol   open_time                                                                                
BTCUSDT  2025-10-06 12:00:00  62500.0  62800.0  62400.0  62700.0  125.45  7856234.50   1523
         2025-10-06 12:05:00  62700.0  62900.0  62600.0  62850.0  98.32   6172845.60   1321
ETHUSDT  2025-10-06 12:00:00  2450.0   2465.0   2445.0   2460.0   340.12  834894.72    892
```

---

## Storage Strategy

### Partitioned Parquet Files

```
base_dir/
├── symbol=BTCUSDT/
│   ├── dt=2025-10-01.parquet
│   ├── dt=2025-10-02.parquet
│   └── dt=2025-10-03.parquet
└── symbol=ETHUSDT/
    ├── dt=2025-10-01.parquet
    └── dt=2025-10-02.parquet
```

**Benefits:**
- Query only relevant partitions (fast)
- Automatic deduplication on load
- Columnar compression (small files)
- Date-based cleanup easy

### Backfill Strategies

#### 1. **backfill_symbol(symbol, start, end)**
Fetch specific date range for one symbol.

```python
dm.backfill_symbol("BTCUSDT", start, end)
```

#### 2. **backfill_realtime(symbols)**
Sync from last stored bar to now (overwrites last bar for clean continuation).

```python
dm.backfill_realtime(["BTCUSDT", "ETHUSDT"])
```

#### 3. **backfill_missing(symbols, start)**
Detect gaps in storage and fill them.

```python
dm.backfill_missing(["BTCUSDT"], start=start)
```

---

## Testing

### Run Tests

```bash
cd "Allora/ML Workflow"

# Unit tests only (no network, fast)
pytest tests/test_data_managers.py -v

# All tests including integration (requires network + API key)
export RUN_INTEGRATION_TESTS=1
pytest tests/test_data_managers.py -v
```

### Test Coverage

**23 Tests Total:**
- ✅ 17 Unit Tests (factory, initialization, parsing, storage)
- ✅ 6 Integration Tests (Binance + Allora live APIs)

See `tests/README.md` for detailed test documentation.

---

## API Reference

### BaseDataManager (Abstract)

#### Methods

**`load_pandas(symbols, start=None, end=None) -> pd.DataFrame`**
Load bars as Pandas DataFrame with MultiIndex (symbol, open_time).

**`load_polars(symbols, start=None, end=None) -> pl.DataFrame`**
Load bars as Polars DataFrame.

**`latest(symbol) -> Optional[datetime]`**
Get timestamp of most recent bar for a symbol.

**`get_live_snapshot(symbols) -> pd.DataFrame`**
Fetch latest bars directly from API (no local storage dependency).

**`backfill_symbol(symbol, start, end)`**
Backfill specific date range for one symbol.

**`backfill_realtime(symbols)`**
Sync from last bar to now.

**`backfill_missing(symbols, start=None)`**
Detect and fill gaps in storage.

---

### BinanceDataManager

#### Constructor

```python
BinanceDataManager(
    base_dir: str = "parquet_data_binance",
    interval: str = "5m",
    market: str = "futures",  # "spot" or "futures"
    symbols: Optional[List[str]] = None,
    cache_len: int = 1000,
    batch_timeout: int = 20,
    rate_limit: float = 0.5
)
```

#### Additional Methods

**`register_batch_callback(fn)`**
Register callback for WebSocket batch updates.
```python
def my_callback(open_time, snapshot_dict):
    print(f"Batch completed at {open_time}")
    print(snapshot_dict)

dm.register_batch_callback(my_callback)
```

**`live(symbols)`**
Start WebSocket streaming for symbols.
```python
dm.live(["BTCUSDT", "ETHUSDT"])
```

---

### AlloraDataManager

#### Constructor

```python
AlloraDataManager(
    api_key: str,
    base_dir: str = "parquet_data_allora",
    interval: str = "5m",
    symbols: Optional[List[str]] = None,
    cache_len: int = 1000,
    max_pages: int = 1000,
    sleep_sec: float = 0.1
)
```

#### Features

- Automatic ticker normalization: `BTC/USD` → `btcusd` for API
- Monthly bucket support for bulk historical downloads
- Aggregates 1-minute bars to any interval

---

### DataManager Factory

```python
DataManager(
    source: str = "binance",  # "binance" or "allora"
    interval: str = "5m",
    symbols: Optional[List[str]] = None,
    cache_len: int = 1000,
    base_dir: Optional[str] = None,
    **kwargs  # Source-specific parameters
) -> BaseDataManager
```

**List Available Sources:**
```python
from allora_forge_builder_kit import list_data_sources
print(list_data_sources())
# Output: ['binance', 'allora']
```

---

## Migration Guide

### From Old `DataManager` to New Architecture

**Before (v1.x):**
```python
from allora_forge_builder_kit import DataManager

dm = DataManager(interval="5m", symbols=["BTCUSDT"])
dm.backfill_missing(["BTCUSDT"], start=start)
```

**After (v2.0):**
```python
from allora_forge_builder_kit import DataManager

# Same code works! Defaults to Binance
dm = DataManager(source="binance", interval="5m", symbols=["BTCUSDT"])
dm.backfill_missing(["BTCUSDT"], start=start)

# Or use Allora
dm = DataManager(source="allora", api_key="key", interval="5m", symbols=["BTC/USD"])
dm.backfill_missing(["BTC/USD"], start=start)
```

### Workflow Integration

**Before:**
```python
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=24,
    number_of_input_candles=24,
    target_length=16,
    interval="5m"
)
```

**After:**
```python
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=24,
    number_of_input_candles=24,
    target_length=16,
    interval="5m",
    data_source="binance"  # ← Add this (defaults to "binance")
)
```

---

## Performance Tips

1. **Use Polars for large datasets:**
   ```python
   df = dm.load_polars(symbols, start, end)  # Faster than Pandas
   ```

2. **Adjust cache size for your use case:**
   ```python
   dm = DataManager(source="binance", cache_len=5000)  # More memory, less disk I/O
   ```

3. **Backfill strategically:**
   ```python
   # Full backfill once
   dm.backfill_symbol("BTCUSDT", start, end)
   
   # Then only sync recent data
   dm.backfill_realtime(["BTCUSDT"])  # Much faster!
   ```

4. **Use date filters when loading:**
   ```python
   # Good: Only load what you need
   df = dm.load_pandas(symbols, start=yesterday, end=today)
   
   # Avoid: Loading everything
   df = dm.load_pandas(symbols)
   ```

---

## Contributing

### Adding a New Data Source

1. Create `your_data_manager.py` inheriting from `BaseDataManager`
2. Implement abstract methods:
   - `get_live_snapshot(symbols)`
   - `backfill_symbol(symbol, start, end)`
   - `backfill_realtime(symbols)`
   - `backfill_missing(symbols, start)`
3. Add to factory in `data_manager_factory.py`
4. Create tests in `tests/test_data_managers.py`

**Example Template:**
```python
from .base_data_manager import BaseDataManager

class MyDataManager(BaseDataManager):
    def __init__(self, base_dir="parquet_data_myapi", interval="5m", **kwargs):
        super().__init__(base_dir=base_dir, interval=interval, **kwargs)
        # Your initialization
    
    def get_live_snapshot(self, symbols):
        # Implement API call
        pass
    
    def backfill_symbol(self, symbol, start, end):
        # Implement backfill
        pass
    
    # ... implement other abstract methods
```

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'polars'**
```bash
pip install polars pandas requests websocket-client
```

**2. Empty DataFrame after backfill**
- Check API availability for your date range
- Verify symbol format (BTCUSDT vs BTC/USD)
- Confirm API key is valid (for Allora)

**3. File path issues with symbols containing `/`**
- Fixed in v2.0! Symbols are automatically normalized for file paths
- `BTC/USD` → stored in `symbol=BTCUSD/`

**4. Parquet files not found**
- Check `base_dir` setting
- Verify backfill completed successfully
- Ensure symbols match exactly (case-sensitive)

---

## License

MIT License - See project root for details.

---

## Support

For issues, feature requests, or questions:
- GitHub Issues: [Create an issue](https://github.com/allora-network/allora-forge-builder-kit)
- Discord: [Join our community](https://discord.gg/allora)

---

**Built with ❤️ by the Allora Team**



