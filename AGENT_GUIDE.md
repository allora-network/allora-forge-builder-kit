# Allora Forge Builder Kit - Comprehensive Agent Guide

**Version:** 2.0  
**Date:** October 2025  
**Audience:** AI Agents, Developers, Data Scientists

---

## 📋 Table of Contents

1. [Quick Start for Agents](#quick-start-for-agents)
2. [The Workflow: Your Primary Interface](#the-workflow-your-primary-interface)
3. [Understanding the Data Flow](#understanding-the-data-flow)
4. [Local Data Storage](#local-data-storage)
5. [Live Feature Extraction](#live-feature-extraction)
6. [Data Managers: Under the Hood](#data-managers-under-the-hood)
7. [Common Use Cases](#common-use-cases)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start for Agents

### Installation

**For AI Agents - Install directly from GitHub:**
```bash
pip install git+https://github.com/allora-network/allora-forge-builder-kit.git
```

**From Source:**
```bash
git clone https://github.com/allora-network/allora-forge-builder-kit.git
cd allora-forge-builder-kit
pip install -e .
```

**With Conda (Full ML Environment):**
```bash
conda env create -f environment.yml
conda activate ml311_dev
```

### 📚 Complete Examples Available

Before diving into the guide, check out these complete working examples:

**1. Topic 69 Bitcoin Prediction** (Recommended for beginners)
- **Notebook**: `notebooks/Topic 69 - Bitcoin Price Prediction.ipynb`
- **Script**: `notebooks/example_topic_69_bitcoin_prediction.py`
- Shows complete v2.0 workflow from data loading to deployment
- Uses fast Allora monthly buckets
- Includes log return to price conversion
- Live feature extraction with offset resampling

**2. Signal Miner Grid Search** (Advanced)
- **Script**: `notebooks/example_signal_miner_grid_search.py`
- Systematic hyperparameter search
- Time-series cross-validation
- Model selection and deployment
- Production-ready best practices

**Run these first to see the framework in action!**

### Core Principle
**🎯 Work with the `AlloraMLWorkflow` - let the data managers handle everything else.**

The workflow is your main interface. Data managers work behind the scenes to:
- Fetch data from APIs (Binance, Allora, etc.)
- Store data locally in efficient Parquet files
- Load data when you need it
- Handle live data updates

### Minimal Example

```python
from allora_forge_builder_kit import AlloraMLWorkflow
from datetime import datetime, timedelta, timezone

# 1. Create workflow (automatically creates appropriate data manager)
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=32,           # Lookback period (hours)
    number_of_input_candles=8, # Number of candles in features
    target_length=16,          # Prediction horizon (hours)
    interval="5m",             # Bar interval
    data_source="binance"      # "binance" or "allora"
)

# 2. Backfill historical data (one-time or periodic)
start = datetime.now(timezone.utc) - timedelta(days=3)
workflow.backfill(start=start)

# 3. Extract features for training
df_features = workflow.get_full_feature_target_dataframe_pandas(start_date=start)

# 4. Train your model
# ... your ML code ...

# 5. Get live features for inference
live_features = workflow.get_live_features("BTCUSDT")
prediction = model.predict(live_features)
```

That's it! The workflow handles all data management automatically.

---

## The Workflow: Your Primary Interface

### What is AlloraMLWorkflow?

`AlloraMLWorkflow` is a high-level class that orchestrates the entire ML pipeline:
- Data fetching & storage (via data managers)
- Feature engineering
- Target creation
- Live inference data preparation

**You should interact with the workflow, not the data managers directly.**

### Key Configuration Parameters

```python
AlloraMLWorkflow(
    # === Data Source Configuration ===
    tickers: List[str],              # Symbols to track (e.g., ["BTCUSDT", "ETHUSDT"])
    interval: str = "5m",            # Bar interval: "1m", "5m", "15m", "1h", "4h", "1d"
    data_source: str = "binance",    # Data source: "binance" or "allora"
    
    # === Feature Configuration ===
    hours_needed: int,               # Lookback period in hours (e.g., 32)
    number_of_input_candles: int,   # How many candles in feature vector (e.g., 8)
    
    # === Target Configuration ===
    target_length: int,              # Prediction horizon in hours (e.g., 16)
    
    # === Data Source Specific (optional) ===
    # For Binance:
    market: str = "futures",         # "spot" or "futures"
    
    # For Allora:
    api_key: str = None,             # Required for Allora
    
    # === Advanced (optional) ===
    data_manager: BaseDataManager = None,  # Explicit data manager instance
    **data_manager_kwargs                   # Additional args passed to data manager
)
```

### Core Workflow Methods

#### 1. **backfill(start=None)**
Fetches and stores historical data locally.

```python
# Backfill last 7 days
start = datetime.now(timezone.utc) - timedelta(days=7)
workflow.backfill(start=start)

# Or let it figure out what's needed
workflow.backfill()
```

**When to use:**
- Initial setup (download historical data)
- After gaps in data collection
- Periodic updates (daily/weekly)

**What happens under the hood:**
1. Workflow calls `data_manager.backfill_missing(symbols, start)`
2. Data manager checks local storage for gaps
3. Fetches missing data from API
4. Stores in Parquet files partitioned by symbol and date
5. Updates in-memory cache

#### 2. **load_raw(start=None, end=None)**
Loads raw OHLCV data from local storage.

```python
# Load specific date range
df = workflow.load_raw(start=start_date, end=end_date)

# Load all available data
df = workflow.load_raw()
```

**Returns:** Pandas DataFrame with MultiIndex (symbol, open_time)

#### 3. **get_full_feature_target_dataframe_pandas(start_date=None)**
Extracts features and targets for ML training.

```python
# Get features + targets for training
df_train = workflow.get_full_feature_target_dataframe_pandas(start_date=start)

# Columns: feature_0, feature_1, ..., feature_39, target
X = df_train[[f"feature_{i}" for i in range(40)]]
y = df_train["target"]
```

**Features:**
- 8 candles × 5 OHLCV = 40 features (normalized)
- Target: log return at prediction horizon

#### 4. **get_live_features(ticker: str)** ⭐ NEW
Fetches 1-minute bars, resamples, and extracts features for live inference.

```python
# Get features for most recent complete bar
features = workflow.get_live_features("BTCUSDT")

# Use for prediction
prediction = model.predict(features)
```

**Returns:** 1-row DataFrame with 40 features, ready for `model.predict()`

**What happens:**
1. Fetches recent 1-minute bars from API
2. Drops incomplete bars (if current second < 45)
3. Resamples to workflow interval (e.g., 5m) with offset alignment
4. Extracts normalized features from last complete bar
5. Returns DataFrame ready for inference

---

## Understanding the Data Flow

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      AlloraMLWorkflow                        │
│  (Your primary interface - orchestrates everything)          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ delegates data operations to
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Manager                            │
│  (Handles all data fetching & storage)                       │
│  - BinanceDataManager   OR   AlloraDataManager              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ stores/loads from
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Local Parquet Storage                       │
│  (Efficient, partitioned, persistent)                        │
│  parquet_data_binance/ or parquet_data_allora/              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow for Training

```
1. User creates workflow
   ↓
2. workflow.backfill(start)
   ↓
3. Data Manager fetches from API
   ↓
4. Data stored in Parquet files (partitioned by symbol/date)
   ↓
5. workflow.get_full_feature_target_dataframe_pandas()
   ↓
6. Data Manager loads from Parquet
   ↓
7. Workflow extracts features + targets
   ↓
8. Returns DataFrame ready for training
```

### Data Flow for Live Inference

```
1. workflow.get_live_features(ticker)
   ↓
2. Data Manager fetches 1-minute bars from API (NO local storage)
   ↓
3. Workflow resamples to target interval
   ↓
4. Workflow extracts features
   ↓
5. Returns 1-row DataFrame ready for model.predict()
```

**Key Insight:** Live features ALWAYS fetch fresh data from API, never from local storage!

---

## Local Data Storage

### Storage Structure

Data is stored in **partitioned Parquet files** for efficiency:

```
parquet_data_binance/              # Binance data (default directory)
├── symbol=BTCUSDT/
│   ├── dt=2025-10-01.parquet      # Each day is a separate file
│   ├── dt=2025-10-02.parquet
│   ├── dt=2025-10-03.parquet
│   └── dt=2025-10-04.parquet
└── symbol=ETHUSDT/
    ├── dt=2025-10-01.parquet
    └── dt=2025-10-02.parquet

parquet_data_allora/               # Allora data (separate directory)
└── symbol=BTCUSD/                 # Note: "/" removed from BTC/USD
    ├── dt=2025-10-01.parquet
    └── dt=2025-10-02.parquet
```

### Why Parquet + Partitioning?

**Benefits:**
1. **Fast queries**: Only read relevant date partitions
2. **Small size**: Columnar compression (5-10x smaller than CSV)
3. **Type safety**: Schema enforced, no parsing errors
4. **Automatic deduplication**: Polars/Pandas handle duplicates on load
5. **Easy cleanup**: Delete old date partitions as needed

### Storage Isolation

Different data sources use **separate directories** by default:
- **Binance**: `parquet_data_binance/`
- **Allora**: `parquet_data_allora/`

**Why?**
- Prevents data collision
- Different bar formats/frequencies can coexist
- Easy to switch between sources

### Custom Storage Location

```python
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=32,
    number_of_input_candles=8,
    target_length=16,
    data_source="binance",
    base_dir="/path/to/my/custom/data"  # Custom location
)
```

### Data Lifecycle

```
1. BACKFILL (One-time or periodic)
   ├─ Fetches from API
   ├─ Writes to Parquet
   └─ Updates cache

2. TRAINING (Uses local storage)
   ├─ Loads from Parquet
   ├─ Extracts features
   └─ Returns DataFrame

3. LIVE INFERENCE (Uses API directly)
   ├─ Fetches 1-min bars from API
   ├─ Resamples on-the-fly
   └─ Extracts features
```

### Cache Layer

Data managers maintain an in-memory cache (deque) of recent bars:
- Default size: 1000 bars
- Reduces disk I/O for repeated queries
- Automatically updated during backfill

```python
# Adjust cache size if needed
workflow = AlloraMLWorkflow(
    ...,
    cache_len=5000  # Larger cache = more memory, less disk I/O
)
```

---

## Live Feature Extraction

### The Problem We Solved

For live inference, you need:
1. The most recent complete data
2. Properly aligned bars (not cut off mid-interval)
3. Features that match your training format
4. Fast, real-time execution

**Traditional approach:** Store all data locally, load last N bars
**Problem:** Stale data, alignment issues, unnecessary disk I/O

**Our approach:** Fetch 1-minute bars from API, resample on-the-fly
**Benefits:** Always fresh, properly aligned, no storage dependency

### How get_live_features() Works

```python
live_features = workflow.get_live_features("BTCUSDT")
```

**Step-by-Step:**

#### Step 1: Fetch 1-Minute Bars
```python
# Data manager fetches recent 1-minute bars
# Amount: hours_needed + 2 hours buffer
# Example: 32 hours needed → fetch 34 hours of 1-min bars
df_1min = data_manager.get_live_1min_data(ticker, hours_back=34)
```

**Why 1-minute bars?**
- Universal base frequency (works for any interval)
- Maximum flexibility for resampling
- Precise alignment control

**Data Sources:**
- **Binance**: Fetches 1-min bars via REST API with pagination (handles 10,000+ bars)
- **Allora**: Fetches 1-min bars via Allora API (native 1-min data)

#### Step 2: Drop Incomplete Bar
```python
# Check if last 1-min bar is complete
now = current_time()
last_bar_time = df_1min.index[-1]

if last_bar_time.minute == now.minute and now.second < 45:
    df_1min = df_1min[:-1]  # Drop incomplete bar
```

**Logic:** If we're in the same minute as the last bar and it's not yet second 45, the bar is incomplete.

#### Step 3: Resample to Target Interval
```python
# Resample to workflow's interval (e.g., 5m)
# WITH offset to align last bar with last 1-min bar
df_resampled = resample_with_offset(df_1min, interval="5m", live_mode=True)
```

**Offset Calculation:**
```
If last 1-min bar is at 16:38:
  - For 5-min bars to END at 16:38
  - Last bar should be labeled 16:34 (covers 16:34-16:38)
  - Offset = (38 + 1) % 5 = 4 minutes
```

**Result:**
- Last resampled bar ends exactly at last 1-minute bar
- All data included, nothing cut off
- Properly aligned for feature extraction

#### Step 4: Verify Sufficient Data
```python
required_bars = hours_needed * bars_per_hour
if len(df_resampled) < required_bars:
    raise ValueError("Not enough data")
```

#### Step 5: Extract Features
```python
# Extract features from last complete bar
features = extract_rolling_daily_features(
    data=df_resampled,
    lookback=hours_needed,
    number_of_candles=number_of_input_candles,
    start_times=[df_resampled.index[-1]]
)
```

**Features:**
- 8 candles × 5 OHLCV = 40 features
- Normalized (prices relative to last close, volume relative to last volume)
- Same format as training data

#### Step 6: Return
```python
# Returns 1-row DataFrame ready for model.predict()
return features  # Shape: (1, 40)
```

### Live Mode vs Historical Mode

| Mode | Offset Applied? | Last Bar | Use Case |
|------|----------------|----------|----------|
| `live_mode=True` | ✅ Yes | Ends at last 1-min bar | Live inference |
| `live_mode=False` | ❌ No | Standard alignment (00, 05, 10...) | Training |

**Why different modes?**
- **Training**: Want consistent, clock-aligned bars (00:00, 00:05, 00:10...)
- **Live**: Want bars that end exactly at the last complete 1-min bar (may be 00:03, 00:08, etc.)

### Example Output

```python
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=2,
    number_of_input_candles=8,
    target_length=16,
    interval="5m",
    data_source="binance"
)

features = workflow.get_live_features("BTCUSDT")
print(features)

# Output:
#                       feature_0  feature_1  feature_2  ...  feature_39
# 2025-10-10 16:34:00   1.01716    1.01776    1.01590   ...    3.92214

# Ready for prediction:
prediction = model.predict(features)
```

---

## Data Managers: Under the Hood

### When to Use Data Managers Directly

**Short answer: Usually never.**

The workflow handles everything. However, you might use data managers directly for:
- Custom data pipelines
- Non-ML use cases
- WebSocket streaming (Binance only)
- Advanced configuration

### Data Manager Responsibilities

```python
# What data managers do:
1. Fetch data from APIs (REST + WebSocket for Binance)
2. Store data in Parquet files (partitioned)
3. Load data efficiently (Pandas or Polars)
4. Maintain in-memory cache
5. Handle rate limiting & retries
6. Normalize symbol formats
7. Detect and fill data gaps
```

### Available Data Managers

#### BinanceDataManager

```python
from allora_forge_builder_kit import BinanceDataManager

dm = BinanceDataManager(
    interval="5m",
    market="futures",  # or "spot"
    symbols=["BTCUSDT", "ETHUSDT"]
)

# Backfill
start = datetime.now(timezone.utc) - timedelta(days=7)
dm.backfill_missing(["BTCUSDT"], start=start)

# Load data
df = dm.load_pandas(["BTCUSDT"], start=start)

# Live snapshot
snapshot = dm.get_live_snapshot(["BTCUSDT"])

# WebSocket streaming (unique to Binance)
def callback(open_time, snapshot_dict):
    print(f"Bar completed at {open_time}")

dm.register_batch_callback(callback)
dm.live(["BTCUSDT", "ETHUSDT"])  # Starts streaming
```

**Features:**
- No API key required
- WebSocket streaming support
- Batch callbacks (triggered when all symbols complete a bar)
- Spot & Futures markets

**Storage:** `parquet_data_binance/`

#### AlloraDataManager

```python
from allora_forge_builder_kit import AlloraDataManager

dm = AlloraDataManager(
    api_key="your-api-key",
    interval="5m",
    symbols=["BTC/USD", "ETH/USD"]
)

# Backfill
start = datetime.now(timezone.utc) - timedelta(days=7)
dm.backfill_missing(["BTC/USD"], start=start)

# Load data
df = dm.load_pandas(["BTC/USD"], start=start)

# Live snapshot
snapshot = dm.get_live_snapshot(["BTC/USD"])
```

**Features:**
- Requires API key
- Native 1-minute data
- Monthly bucket downloads (bulk historical)
- Automatic ticker normalization (BTC/USD → btcusd for API)

**Storage:** `parquet_data_allora/`

### Factory Pattern

```python
from allora_forge_builder_kit import DataManager

# Binance
dm = DataManager(source="binance", interval="5m", market="futures")

# Allora
dm = DataManager(source="allora", api_key="key", interval="5m")
```

---

## Common Use Cases

### Use Case 1: Initial Setup & Training

```python
from allora_forge_builder_kit import AlloraMLWorkflow
from datetime import datetime, timedelta, timezone
import lightgbm as lgb

# 1. Create workflow
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT", "ETHUSDT"],
    hours_needed=32,
    number_of_input_candles=8,
    target_length=16,
    interval="5m",
    data_source="binance",
    market="futures"
)

# 2. Backfill historical data (one-time)
start = datetime.now(timezone.utc) - timedelta(days=30)
workflow.backfill(start=start)

# 3. Extract features + targets
df_train = workflow.get_full_feature_target_dataframe_pandas(start_date=start)

# 4. Split and train
feature_cols = [f"feature_{i}" for i in range(40)]
X = df_train[feature_cols]
y = df_train["target"]

model = lgb.LGBMRegressor()
model.fit(X, y)

# 5. Save model
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```

### Use Case 2: Live Inference Loop

```python
import time
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Create workflow (no backfill needed for live)
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=32,
    number_of_input_candles=8,
    target_length=16,
    interval="5m",
    data_source="binance",
    market="futures"
)

# Live inference loop
while True:
    try:
        # Get live features
        features = workflow.get_live_features("BTCUSDT")
        
        # Predict
        prediction = model.predict(features)[0]
        timestamp = features.index[0]
        
        print(f"[{timestamp}] Prediction: {prediction:.6f}")
        
        # Execute trading logic here...
        
    except ValueError as e:
        print(f"Error: {e}")
    
    # Wait for next bar
    time.sleep(5 * 60)  # 5 minutes
```

### Use Case 3: Multiple Data Sources

```python
# Compare Binance vs Allora data
workflow_binance = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=24,
    number_of_input_candles=24,
    target_length=16,
    data_source="binance"
)

workflow_allora = AlloraMLWorkflow(
    tickers=["btcusd"],  # Allora format
    hours_needed=24,
    number_of_input_candles=24,
    target_length=16,
    data_source="allora",
    api_key="your-key"
)

# Both work identically!
features_binance = workflow_binance.get_live_features("BTCUSDT")
features_allora = workflow_allora.get_live_features("btcusd")
```

### Use Case 4: Periodic Data Updates

```python
import schedule

def update_data():
    """Run daily to keep data fresh"""
    workflow.backfill()  # Fills gaps since last update
    print("Data updated!")

# Schedule daily at 2 AM
schedule.every().day.at("02:00").do(update_data)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Best Practices

### 1. **Let the Workflow Handle Data Management**

❌ **Don't do this:**
```python
dm = BinanceDataManager(...)
dm.backfill_symbol(...)
df = dm.load_pandas(...)
# Now manually extract features...
```

✅ **Do this:**
```python
workflow = AlloraMLWorkflow(...)
workflow.backfill(...)
df = workflow.get_full_feature_target_dataframe_pandas()
```

### 2. **Backfill Once, Update Periodically**

❌ **Don't do this:**
```python
# Every time you run
workflow.backfill(start=30_days_ago)  # Refetches everything!
```

✅ **Do this:**
```python
# Initial setup (once)
workflow.backfill(start=30_days_ago)

# Periodic updates (daily)
workflow.backfill()  # Only fills gaps
```

### 3. **Use Correct Intervals**

Match your interval to your trading strategy:
- **High frequency**: 1m, 5m
- **Day trading**: 15m, 1h
- **Swing trading**: 4h, 1d

```python
# For 5-minute strategy
workflow = AlloraMLWorkflow(interval="5m", ...)

# For hourly strategy
workflow = AlloraMLWorkflow(interval="1h", ...)
```

### 4. **Handle Errors Gracefully**

```python
try:
    features = workflow.get_live_features(ticker)
    prediction = model.predict(features)
except ValueError as e:
    # Not enough data, API down, etc.
    print(f"Warning: {e}")
    # Use fallback logic or skip
```

### 5. **Monitor Data Freshness**

```python
# Check when data was last updated
latest_time = workflow._dm.latest("BTCUSDT")
age = datetime.now(timezone.utc) - latest_time

if age > timedelta(hours=1):
    print("Warning: Data is stale!")
    workflow.backfill()
```

### 6. **Test with Small Date Ranges**

```python
# Development: Small range
start = datetime.now(timezone.utc) - timedelta(days=3)
workflow.backfill(start=start)

# Production: Full history
start = datetime.now(timezone.utc) - timedelta(days=90)
workflow.backfill(start=start)
```

### 7. **Use Appropriate Lookback Periods**

```python
# Short-term patterns
workflow = AlloraMLWorkflow(hours_needed=8, ...)

# Medium-term patterns
workflow = AlloraMLWorkflow(hours_needed=32, ...)

# Long-term patterns
workflow = AlloraMLWorkflow(hours_needed=168, ...)  # 7 days
```

---

## Troubleshooting

### Issue: "Not enough historical data"

**Cause:** Insufficient bars for feature extraction

**Solution:**
```python
# Increase backfill period
workflow.backfill(start=earlier_date)

# OR reduce hours_needed
workflow = AlloraMLWorkflow(hours_needed=16, ...)  # Instead of 32
```

### Issue: "No 1-minute data returned"

**Cause:** API unavailable or rate limited

**Solution:**
```python
# Add retry logic
import time

max_retries = 3
for attempt in range(max_retries):
    try:
        features = workflow.get_live_features(ticker)
        break
    except ValueError:
        if attempt < max_retries - 1:
            time.sleep(10)  # Wait and retry
        else:
            raise
```

### Issue: Empty DataFrame after backfill

**Cause:** Wrong date range or symbol format

**Solution:**
```python
# Check if data exists
latest = workflow._dm.latest("BTCUSDT")
print(f"Latest bar: {latest}")

# Verify symbol format
# Binance: BTCUSDT (uppercase, no separator)
# Allora: btcusd (lowercase, no separator)
```

### Issue: Features don't match training

**Cause:** Different interval or configuration

**Solution:**
```python
# Ensure consistent config
HOURS_NEEDED = 32
NUM_CANDLES = 8
INTERVAL = "5m"

# Training
workflow_train = AlloraMLWorkflow(
    hours_needed=HOURS_NEEDED,
    number_of_input_candles=NUM_CANDLES,
    interval=INTERVAL,
    ...
)

# Inference (must match!)
workflow_live = AlloraMLWorkflow(
    hours_needed=HOURS_NEEDED,  # SAME
    number_of_input_candles=NUM_CANDLES,  # SAME
    interval=INTERVAL,  # SAME
    ...
)
```

### Issue: "ModuleNotFoundError"

**Cause:** Missing dependencies

**Solution:**
```bash
pip install polars pandas requests websocket-client lightgbm
# or
conda env create -f environment.yml
conda activate ml311_dev
```

### Issue: Parquet files taking too much space

**Solution:**
```python
# Clean old data
import glob
import os

# Delete files older than 90 days
for file in glob.glob("parquet_data_binance/*/dt=*.parquet"):
    # Parse date from filename and delete if old
    ...
```

---

## Summary: Quick Reference

### Workflow Creation

```python
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=32,
    number_of_input_candles=8,
    target_length=16,
    interval="5m",
    data_source="binance"  # or "allora"
)
```

### Common Operations

| Operation | Method | When to Use |
|-----------|--------|-------------|
| Download historical data | `workflow.backfill(start)` | Initial setup, periodic updates |
| Load raw OHLCV | `workflow.load_raw(start, end)` | Exploratory analysis |
| Get training data | `workflow.get_full_feature_target_dataframe_pandas()` | Model training |
| Get live features | `workflow.get_live_features(ticker)` | Live inference |

### Data Flow Summary

```
TRAINING FLOW:
API → Parquet Storage → workflow.get_full_feature_target_dataframe_pandas() → Model

INFERENCE FLOW:
API → workflow.get_live_features() → Model (bypasses storage)
```

### Key Principles

1. **Use the workflow** - Don't manage data manually
2. **Backfill periodically** - Keep local storage fresh
3. **Use live features for inference** - Always fresh from API
4. **Match configurations** - Training and inference must use same params
5. **Handle errors gracefully** - APIs can fail, plan for it

---

**Built with ❤️ by the Allora Team**

For more details, see:
- `ARCHITECTURE.md` - Deep dive into data manager architecture
- `tests/README.md` - Test documentation and examples
- `README.md` - Project overview


