[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![GitHub issues](https://img.shields.io/github/issues/allora-network/allora-forge-builder-kit)](https://github.com/allora-network/allora-forge-builder-kit/issues)
[![Last Commit](https://img.shields.io/github/last-commit/allora-network/allora-forge-builder-kit)](https://github.com/allora-network/allora-forge-builder-kit/commits/main)
[![Stars](https://img.shields.io/github/stars/allora-network/allora-forge-builder-kit?style=social)](https://github.com/allora-network/allora-forge-builder-kit/stargazers)

<img width="1536" height="1024" alt="forge_silicon" src="https://github.com/user-attachments/assets/f1444abf-e649-4e48-a9f0-187b78b59ccc" />

# Allora Forge Builder Kit 3.0

**A production-ready machine learning workflow for cryptocurrency prediction and trading, designed for both humans and AI agents.**

Build, train, and deploy ML models with a single unified interface that works seamlessly across multiple data sources (Atlas, Binance, Allora Network) while handling all the complexity of data management, feature engineering, evaluation, and live inference for you.

🚀 **[Launch the Builder Kit in Google Colab](https://colab.research.google.com/github/allora-network/allora-forge-builder-kit/blob/main/notebooks/Allora%20Forge%20Builder%20Kit.ipynb)**

---

## ✨ What's New in v3.0

v3.0 builds on the v2.0 architecture with new data sources, corrected evaluation metrics, topic discovery, and AI agent support.

### 1. **Atlas Data Service (New Default)**

The default `"allora"` data source now connects to the **Atlas API** (`forge-data.allora.run`) — Tiingo-sourced 1-minute OHLCV candles with low latency and broad asset coverage.

**Key Features:**
- **76 Tiingo 1-minute datasets** available (BTC, ETH, SOL, and more)
- **Paginated bulk download** with smart gap detection and incremental backfill
- **Data lag fallback**: Automatically discovers latest available data if there's API lag
- **Same API key** as the legacy Allora data service
- **Multi-source support**: Atlas, Binance (Spot & Futures), and legacy Allora API
- **Extensible design**: Add new data sources by implementing the `BaseDataManager` abstract class
- **Local Parquet storage** with remote sync, deduplication, and in-memory caching

### 2. **Updated Performance Metrics with A+ to F Grading**

Model predictions are scored using the **latest evaluation framework** with clear, actionable feedback.

**The Grading System:**
- **7 primary metrics** with pass/fail thresholds:
  - Directional Accuracy (≥ 52%)
  - DA CI Lower Bound (≥ 50%)
  - DA Statistical Significance (p < 0.05, z-test with autocorrelation-aware effective n)
  - Pearson Correlation (≥ 0.05)
  - Pearson Statistical Significance (p < 0.05)
  - WRMSE Improvement vs Baseline (≥ 5%)
  - CZAR Improvement vs Oracle (≥ 10%)

- **Auto-grading**: Get an A+ to F grade based on metrics passed + temporal coverage (max 8 points)
  - 8/8 points → A+ (100%)
  - 7/8 points → A (87.5%)
  - 6/8 points → B+ (75%)
  - 5/8 points → B (62.5%)
  - 4/8 points → C (50%)
  - < 4/8 points → D or F

- **15+ additional metrics**: MAE, MSE, RMSE, R², MAPE, Precision, Recall, F1, Specificity, Spearman correlation, and more

**Why This Matters:** Instead of guessing whether your model is "good enough," you get objective, research-aligned metrics that help you understand prediction quality and iterate faster.

### 3. **Topic Discovery & AI Agent Support**

- **`AlloraTopicDiscovery`**: Query all active Allora Network topics and their configuration via `allora_sdk`
- **`AGENTS.md`**: Structured instructions for AI agents working with this repository
- **`skills/`**: Focused skill packages for model building and data exploration
- **SDK compatibility**: Updated for `allora_sdk >= 1.0.6`

### 4. **Enhanced Workflow & Feature Engineering**

- **Standalone feature conversion**: Convert 1-minute candles into features at any interval
- **Feature engineering guidance**: Examples for SMAs, MACD, RSI, and more
- **45 tests**: 33 unit + 12 integration tests covering all data sources and evaluation metrics
- **Improved live feature extraction** with production-ready error handling

---

### Migrate to v3.0

**Binance Example:**
```python
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    number_of_input_bars=48,
    target_bars=24,
    interval="1h",
    data_source="binance",
    market="futures"  # or "spot"
)
```

**Allora Example:**
```python
workflow = AlloraMLWorkflow(
    tickers=["btcusd"],
    number_of_input_bars=48,
    target_bars=24,
    interval="1h",
    data_source="allora",
    api_key="your-api-key"
)
```

**Key Changes:**
- Added `data_source` parameter to specify data provider
- `number_of_input_bars` and `target_bars` are now measured in units of the `interval` parameter
  - Example: `interval="1h"` with `target_bars=24` means predict 24 hours ahead
  - Example: `interval="5m"` with `target_bars=24` means predict 120 minutes (2 hours) ahead
- Unified API works consistently across both Binance and Allora data sources

---

## 🚀 Quick Start

### Installation

#### From GitHub (Recommended for Agents)
```bash
pip install git+https://github.com/allora-network/allora-forge-builder-kit.git
```

#### From Source
```bash
git clone https://github.com/allora-network/allora-forge-builder-kit.git
cd allora-forge-builder-kit
pip install -e .
```

#### With Conda (Full ML Environment)
```bash
git clone https://github.com/allora-network/allora-forge-builder-kit.git
cd allora-forge-builder-kit
conda env create -f environment.yml
conda activate ml311_dev
```

### 30-Second Example (Topic 69)

```python
from allora_forge_builder_kit import AlloraMLWorkflow
from datetime import datetime, timedelta, timezone
import lightgbm as lgb
import numpy as np

# 1. Create workflow for 24-hour Bitcoin prediction
workflow = AlloraMLWorkflow(
    tickers=["btcusd"],
    number_of_input_bars=48,   # 48 hourly bars for features
    target_bars=24,            # Predict 24 hours ahead
    interval="1h",
    data_source="allora",
    api_key="your-api-key"
)

# 2. Backfill data (fast with Allora!)
start = datetime.now(timezone.utc) - timedelta(days=180)
workflow.backfill(start=start)

# 3. Get training data
df = workflow.get_full_feature_target_dataframe(start_date=start).reset_index()
feature_cols = [c for c in df.columns if c.startswith('feature_')]

# 4. Train model on log returns
model = lgb.LGBMRegressor(n_estimators=100)
model.fit(df[feature_cols], df["target"])

# 5. Live inference with price conversion
def predict():
    # Get live features (fresh 1-min data → resampled → features)
    features = workflow.get_live_features("btcusd")
    log_return = model.predict(features)[0]
    
    # Get current price
    raw = workflow.load_raw(start=datetime.now(timezone.utc) - timedelta(hours=2))
    current_price = raw["close"].iloc[-1]
    
    # Convert log return → price (Topic 69 requirement)
    predicted_price = current_price * np.exp(log_return)
    return float(predicted_price)

print(f"Predicted BTC price (24h): ${predict():,.2f}")
```

**That's it!** Complete Topic 69 pipeline in 40 lines.

💡 **See full examples**:
- [`notebooks/Allora Forge Builder Kit.ipynb`](notebooks/Allora%20Forge%20Builder%20Kit.ipynb) - Main walkthrough notebook with grid search and evaluation
- [`notebooks/example_topic_69_bitcoin_walkthrough.py`](notebooks/example_topic_69_bitcoin_walkthrough.py) - Python script version of complete walkthrough
- [`notebooks/feature_engineering_example.py`](notebooks/feature_engineering_example.py) - Feature engineering with technical indicators
- [`notebooks/Allora Wallet Creator.ipynb`](notebooks/Allora%20Wallet%20Creator.ipynb) - Create and fund Allora wallets for deployment
- [`notebooks/deploy_worker.py`](notebooks/deploy_worker.py) - Deploy your trained model to Allora Network

---

## 🔄 How It Works

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
│  AtlasDataManager (default)  |  BinanceDataManager          │
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

**Key Insight:** Training uses local storage for speed, while live inference always fetches fresh data from APIs for accuracy.

---

## 🎯 Key Features

### 1. **Data-Source Agnostic Architecture**

Work with multiple data sources through a single unified interface:

```python
# Use Binance (no API key required)
workflow_binance = AlloraMLWorkflow(
    tickers=["BTCUSDT", "ETHUSDT"],
    number_of_input_bars=48,
    target_bars=24,
    interval="1h",
    data_source="binance",
    market="futures"  # or "spot"
)

# Use Allora Network (API key required)
workflow_allora = AlloraMLWorkflow(
    tickers=["btcusd", "ethusd"],
    number_of_input_bars=48,
    target_bars=24,
    interval="1h",
    data_source="allora",
    api_key="your-allora-api-key"
)

# Both work identically!
features_binance = workflow_binance.get_live_features("BTCUSDT")
features_allora = workflow_allora.get_live_features("btcusd")
```

### 2. **Automated Data Management**

The workflow handles everything:
- ✅ Fetch historical data from APIs
- ✅ Store efficiently in partitioned Parquet files
- ✅ Detect and fill gaps automatically
- ✅ Maintain in-memory cache for fast access
- ✅ Handle rate limiting and pagination

```python
# Backfill last 30 days (detects gaps automatically)
start = datetime.now(timezone.utc) - timedelta(days=30)
workflow.backfill(start=start)

# Periodic updates (only fills gaps)
workflow.backfill()  # Smart gap detection
```

**Storage Structure:**
```
parquet_data_binance/
├── symbol=BTCUSDT/
│   ├── dt=2025-10-01.parquet
│   ├── dt=2025-10-02.parquet
│   └── dt=2025-10-03.parquet
└── symbol=ETHUSDT/
    └── dt=2025-10-01.parquet

parquet_data_allora/
└── symbol=BTCUSD/
    └── dt=2025-10-01.parquet
```

### 3. **Live Feature Extraction**

Real-time inference with fresh data:

```python
# Automatically:
# 1. Fetches recent 1-minute bars from API
# 2. Drops incomplete bars
# 3. Resamples to workflow interval (e.g., 5m)
# 4. Extracts normalized features
# 5. Returns DataFrame ready for model.predict()

features = workflow.get_live_features("BTCUSDT")
prediction = model.predict(features)
```

**What happens under the hood:**
1. Calculate required data based on `number_of_input_bars` and `interval`
2. Fetch 1-minute bars from API (bypasses local storage)
3. Drop incomplete bar (if current second < 45)
4. Resample to target interval with offset alignment
5. Extract normalized features (number_of_input_bars × 5 OHLCV)
6. Return 1-row DataFrame ready for inference

### 4. **Flexible Feature Engineering**

Normalized features from rolling windows:

```python
# Configuration
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    number_of_input_bars=48,   # Number of input bars
    target_bars=24,            # Prediction horizon
    interval="1h",             # Bar interval
    data_source="binance",
    ...
)

# Features: 48 bars × 5 OHLCV = 240 features
# Normalized: prices relative to last close, volume relative to last volume
features = workflow.get_live_features("BTCUSDT")
# Shape: (1, 240)
```

**Feature Structure:**
- `feature_open_0` to `feature_open_47`: Open prices for 48 bars
- `feature_high_0` to `feature_high_47`: High prices for 48 bars
- `feature_low_0` to `feature_low_47`: Low prices for 48 bars
- `feature_close_0` to `feature_close_47`: Close prices for 48 bars
- `feature_volume_0` to `feature_volume_47`: Volume for 48 bars

### 5. **Built-in Evaluation Metrics**

```python
from allora_forge_builder_kit import PerformanceEvaluator

evaluator = PerformanceEvaluator()
report = evaluator.evaluate(y_true=actual_returns, y_pred=predicted_returns)
evaluator.print_report(report)

# Output: grade (A+ to F), 7 primary metrics with pass/fail, additional metrics
```

### 6. **WebSocket Streaming** (Binance Only)

```python
from allora_forge_builder_kit import BinanceDataManager

dm = BinanceDataManager(interval="5m", market="futures")

# Register callback for when bars complete
def on_batch_complete(open_time, snapshot_dict):
    print(f"Bar completed at {open_time}")
    for symbol, bar in snapshot_dict.items():
        print(f"  {symbol}: close={bar['close']}")

dm.register_batch_callback(on_batch_complete)

# Start streaming
dm.live(["BTCUSDT", "ETHUSDT"])
```

### 7. **Standardized Data Format**

All data managers return data in a consistent format across sources:

**Single Bar (Dictionary):**
```python
{
    "symbol": str,          # e.g., "BTCUSDT" or "BTC/USD"
    "open_time": datetime,  # UTC timezone-aware
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": float,        # Base asset volume
    "quote_volume": float,  # Quote asset volume
    "n_trades": int         # Number of trades
}
```

**DataFrame (MultiIndex):**
```python
# Index: (symbol, open_time)
# Columns: open, high, low, close, volume, quote_volume, n_trades

                              open      high       low     close    volume  quote_volume  n_trades
symbol   open_time                                                                                
BTCUSDT  2025-10-31 12:00:00  62500.0  62800.0  62400.0  62700.0  125.45  7856234.50   1523
         2025-10-31 12:05:00  62700.0  62900.0  62600.0  62850.0  98.32   6172845.60   1321
```

**Storage Structure:**
```
parquet_data_binance/
├── symbol=BTCUSDT/
│   ├── dt=2025-10-01.parquet  # Daily partitions
│   ├── dt=2025-10-02.parquet
│   └── dt=2025-10-03.parquet
└── symbol=ETHUSDT/
    └── dt=2025-10-01.parquet
```

**Benefits:**
- Fast queries (only read relevant date partitions)
- Automatic deduplication on load
- Columnar compression (5-10x smaller than CSV)
- Type safety with schema enforcement

---

## 📚 Documentation

All documentation is consolidated in this README. For additional details:

- **Testing**: See [tests/README.md](tests/README.md) for test documentation and coverage
- **Examples**: See the [Quick Start](#-quick-start) section for links to example notebooks and scripts
- **API Reference**: All classes and methods are documented with docstrings in the source code

---

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Install test dependencies
pip install pytest

# Run unit tests (fast, no network)
pytest tests/test_data_managers.py -v

# Run all tests including integration (requires network)
export RUN_INTEGRATION_TESTS=1
pytest tests/test_data_managers.py -v
```

**Test Coverage:**
- ✅ 33 Unit Tests (factory, initialization, storage, workflow, evaluation metrics)
- ✅ 12 Integration Tests (Binance + Allora live APIs, full pipelines)

**Integration Tests:**
- `test_binance_get_live_1min_data` - Verify 1-minute data fetching
- `test_allora_get_live_1min_data` - Verify 1-minute data fetching
- `test_workflow_binance_get_live_features` - End-to-end validation
- `test_workflow_allora_get_live_features` - End-to-end validation
- `test_workflow_live_mode_offset` - Visual inspection of offset logic
- `test_workflow_get_live_features_end_to_end` - Full validation

See [tests/README.md](tests/README.md) for detailed documentation.

---

## 🌟 Why Allora Forge Builder Kit?

### For Data Scientists
- **Focus on ML**: Data management is handled automatically
- **Reproducible**: Same workflow for training and inference
- **Flexible**: Easy to experiment with different intervals and lookbacks
- **Production-ready**: Robust error handling and logging

### For AI Agents
- **Simple interface**: One class does everything
- **Self-documenting**: Clear parameters and return types
- **Easy installation**: `pip install git+https://github.com/...`
- **Comprehensive examples**: Multiple working examples included

### For Traders
- **Live inference**: Always fresh data, no stale predictions
- **Multi-source**: Compare Binance vs Allora data
- **Real-time streaming**: WebSocket support for Binance
- **Backtesting**: Historical data readily available

### For Developers
- **Modular architecture**: Easy to extend with new data sources
- **Well-tested**: 45 tests covering all functionality
- **Type-safe**: Clear interfaces and error messages
- **Open source**: Apache 2.0 license

---

## 📊 Supported Intervals

| Interval | Description | Use Case |
|----------|-------------|----------|
| `1m` | 1-minute bars | Ultra high-frequency trading |
| `5m` | 5-minute bars | High-frequency trading |
| `15m` | 15-minute bars | Intraday trading |
| `1h` | 1-hour bars | Day trading |
| `4h` | 4-hour bars | Swing trading |
| `1d` | Daily bars | Position trading |

---

## 🚨 Requirements

### Python Version
- Python 3.10 or higher

### Core Dependencies
```
pandas
numpy
polars
pyarrow
pytz
requests
lightgbm
numba
scipy
dill
allora-sdk>=1.0.6
```

### Optional Dependencies
```
websocket-client   # Binance WebSocket streaming
scikit-learn       # Cross-validation utilities
cloudpickle        # Serializing prediction functions
matplotlib         # Feature visualization
```

See `pyproject.toml` for the complete list.

---

## 🤝 Contributing

We welcome contributions! Here's how:

### Adding a New Data Source

1. Create `your_data_manager.py` inheriting from `BaseDataManager`
2. Implement abstract methods
3. Add to factory in `data_manager_factory.py`
4. Create tests in `tests/test_data_managers.py`
5. Submit PR

See [ARCHITECTURE.md - Contributing](ARCHITECTURE.md#contributing) for detailed guide.

### Reporting Issues

Found a bug? Have a feature request?
- [GitHub Issues](https://github.com/allora-network/allora-forge-builder-kit/issues)

---

## 📄 License

Apache 2.0 License - See [LICENSE](LICENSE) for details.

---

## 🌐 Community & Support

- **AI Agent Guide**: [AGENTS.md](AGENTS.md)
- **Discord**: [Join our community](https://discord.gg/allora)
- **GitHub**: [allora-network/allora-forge-builder-kit](https://github.com/allora-network/allora-forge-builder-kit)
- **Website**: [https://allora.network](https://allora.network)
- **Developer Portal**: [https://developer.allora.network](https://developer.allora.network)

---

## 🙏 Acknowledgments

Built with ❤️ by the Allora Team and contributors.

Special thanks to:
- Binance for providing free market data API
- The Allora Network community
- All open-source contributors

---

**Welcome to the Forge. Build the future of decentralized AI.**

