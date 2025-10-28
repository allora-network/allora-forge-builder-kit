[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![GitHub issues](https://img.shields.io/github/issues/allora-network/allora-forge-builder-kit)](https://github.com/allora-network/allora-forge-builder-kit/issues)
[![Last Commit](https://img.shields.io/github/last-commit/allora-network/allora-forge-builder-kit)](https://github.com/allora-network/allora-forge-builder-kit/commits/main)
[![Stars](https://img.shields.io/github/stars/allora-network/allora-forge-builder-kit?style=social)](https://github.com/allora-network/allora-forge-builder-kit/stargazers)

<img width="1536" height="1024" alt="forge_silicon" src="https://github.com/user-attachments/assets/f1444abf-e649-4e48-a9f0-187b78b59ccc" />

# Allora Forge Builder Kit 2.0

**A production-ready machine learning workflow for cryptocurrency prediction and trading, now with multi-source data management and live feature extraction.**

Build, train, and deploy ML models with a single unified interface that works seamlessly across multiple data sources (Binance, Allora Network) while handling all the complexity of data management, feature engineering, and live inference for you.

🚀 **[Launch the Builder Kit in Google Colab](https://colab.research.google.com/github/allora-network/allora-forge-builder-kit/blob/main/notebooks/Allora%20Forge%20Builder%20Kit.ipynb)**

---

## ✨ What's New in v2.0

### 🔄 **Modular Data Management Architecture**
- **Multi-source support**: Binance (Spot & Futures) + Allora Network
- **Factory pattern**: Easy instantiation with `DataManager(source="binance")`
- **Extensible**: Add new data sources by implementing `BaseDataManager`
- **Storage isolation**: Separate Parquet directories per source
- **Partitioned Parquet**: Fast queries, small files, automatic deduplication
- **Smart backfill**: Gap detection, incremental updates, pagination
- **Hot cache**: In-memory recent bars for repeated queries

### ⚡ **Live Feature Extraction**
- **Real-time inference**: Fetch 1-minute bars, resample, extract features—all on-the-fly
- **Data-source agnostic**: Same `get_live_features()` works for Binance or Allora
- **Smart alignment**: Offset resampling ensures bars end exactly at last 1-minute bar
- **No storage dependency**: Always fetches fresh data for inference

### 📏 **Official Performance Metrics**
- **8 primary metrics**: Directional Accuracy, Pearson Correlation, WRMSE/ZPTAE Improvement, Log Aspect Ratio + statistical significance
- **Auto-grading**: A+ to F grades based on metrics passed
- **Research-aligned**: Same framework used by Allora Research team
- **Comprehensive reporting**: 15+ additional metrics (MAE, MSE, R², Precision, Recall, F1)

### 🎯 **Production-Ready Testing**
- **23 comprehensive tests**: Unit + integration tests
- **Both data sources**: Validates Binance & Allora end-to-end
- **Visual inspection**: Tests output actual data for verification
- **CI/CD ready**: Separates unit tests (fast) from integration tests (network)

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

# 1. Create workflow for 1-day Bitcoin prediction
workflow = AlloraMLWorkflow(
    tickers=["btcusd"],
    hours_needed=7*24,         # 7 days lookback
    number_of_input_candles=24,# 24 candles in features
    target_length=1*24,        # 1 day prediction (Topic 69)
    interval="1h",             # 1-hour bars
    data_source="allora",      # Fast monthly buckets!
    api_key="your-api-key"
)

# 2. Backfill data (fast with Allora!)
start = datetime.now(timezone.utc) - timedelta(days=180)
workflow.backfill(start=start)

# 3. Get training data
df = workflow.get_full_feature_target_dataframe_pandas(start_date=start)
df = df.reset_index()
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

💡 **See full example**: [`notebooks/example_topic_69_bitcoin_prediction.py`](notebooks/example_topic_69_bitcoin_prediction.py)

---

## 🎯 Key Features

### 1. **Data-Source Agnostic Architecture**

Work with multiple data sources through a single unified interface:

```python
# Use Binance (no API key required)
workflow_binance = AlloraMLWorkflow(
    tickers=["BTCUSDT", "ETHUSDT"],
    data_source="binance",
    market="futures",  # or "spot"
    interval="5m",
    hours_needed=32,
    number_of_input_candles=8,
    target_length=16
)

# Use Allora Network (API key required)
workflow_allora = AlloraMLWorkflow(
    tickers=["btcusd", "ethusd"],
    data_source="allora",
    api_key="your-allora-api-key",
    interval="5m",
    hours_needed=32,
    number_of_input_candles=8,
    target_length=16
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
1. Calculate required data: `hours_needed + 2 hours buffer`
2. Fetch 1-minute bars from API (bypasses local storage)
3. Drop incomplete bar (if current second < 45)
4. Resample to target interval with offset alignment
5. Extract 40 normalized features (8 candles × 5 OHLCV)
6. Return 1-row DataFrame: shape `(1, 40)`

### 4. **Flexible Feature Engineering**

Normalized features from rolling windows:

```python
# Configuration
workflow = AlloraMLWorkflow(
    hours_needed=32,           # Lookback window
    number_of_input_candles=8, # Split into 8 segments
    interval="5m",             # Bar interval
    ...
)

# Features: 8 candles × 5 OHLCV = 40 features
# Normalized: prices relative to last close, volume relative to last volume
features = workflow.get_live_features(ticker)
# Shape: (1, 40)
```

**Feature Structure:**
- `feature_0` to `feature_4`: Candle 1 (open, high, low, close, volume)
- `feature_5` to `feature_9`: Candle 2
- ...
- `feature_35` to `feature_39`: Candle 8

### 5. **Built-in Evaluation Metrics**

```python
# Get test predictions
test_preds = model.predict(X_test[feature_cols])

# Evaluate
metrics = workflow.evaluate_test_data(test_preds)
print(metrics)

# Output:
# {'correlation': 0.042, 'directional_accuracy': 0.548}
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

---

## 📚 Documentation

### Core Documentation

| Document | Description |
|----------|-------------|
| **[AGENT_GUIDE.md](AGENT_GUIDE.md)** | 📘 **Comprehensive guide for AI agents**: How to use the workflow, understand data flow, local storage, and live feature extraction |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | 🏗️ Deep dive into data manager architecture, storage strategy, and API reference |
| **[tests/README.md](tests/README.md)** | 🧪 Test documentation with examples and coverage details |

### Quick Links

- **Installation**: See [Installation](#installation) section above
- **API Reference**: See [ARCHITECTURE.md - API Reference](ARCHITECTURE.md#api-reference)
- **Testing**: See [tests/README.md](tests/README.md)
- **Examples**: See [Example Notebooks](#example-notebooks) below

---

## 📖 Example Notebooks & Scripts

### 1. **Topic 69: Bitcoin Price Prediction** (Complete Example)

**Notebook**: [`notebooks/Topic 69 - Bitcoin Price Prediction.ipynb`](notebooks/Topic%2069%20-%20Bitcoin%20Price%20Prediction.ipynb)  
**Script**: [`notebooks/example_topic_69_bitcoin_prediction.py`](notebooks/example_topic_69_bitcoin_prediction.py)

Complete pipeline for 24-hour Bitcoin price prediction:
- v2.0 workflow with Allora data (fast loading)
- Train on log returns
- Convert to price predictions for Topic 69
- Full deployment example

**Key Features:**
- ✅ Uses Allora monthly buckets (much faster than Binance)
- ✅ Trains on log returns (ML best practice)
- ✅ Converts to actual prices (Topic 69 requirement)
- ✅ Live feature extraction with offset resampling
- ✅ Ready-to-deploy predict function

### 2. **Signal Miner: Hyperparameter Grid Search**

**Script**: [`notebooks/example_signal_miner_grid_search.py`](notebooks/example_signal_miner_grid_search.py)

Systematic model selection pipeline:
- Hyperparameter grid search (10+ models)
- Time-series cross-validation (3 folds)
- Model selection based on validation performance
- Retrain best model on full dataset
- Deploy best configuration

**Inspired by ML MCP Server's experiment tracking**

### 3. **Legacy: Basic ML Workflow**
[Allora Forge ML Workflow Example](https://github.com/allora-network/allora-forge-builder-kit/blob/main/notebooks/Allora%20Forge%20Builder%20Kit.ipynb)

Original example (pre-v2.0):
- Data fetching with old API
- Basic feature extraction
- Simple model training

---

## 🔥 Complete Training & Deployment Example

### Step 1: Setup & Data Collection

```python
from allora_forge_builder_kit import AlloraMLWorkflow
from datetime import datetime, timedelta, timezone
import lightgbm as lgb
import pickle

# Configuration
TICKERS = ["BTCUSDT", "ETHUSDT"]
HOURS_NEEDED = 32
NUM_CANDLES = 8
TARGET_LENGTH = 16
INTERVAL = "5m"

# Create workflow
workflow = AlloraMLWorkflow(
    tickers=TICKERS,
    hours_needed=HOURS_NEEDED,
    number_of_input_candles=NUM_CANDLES,
    target_length=TARGET_LENGTH,
    interval=INTERVAL,
    data_source="binance",
    market="futures"
)

# Backfill 30 days
start = datetime.now(timezone.utc) - timedelta(days=30)
print(f"Backfilling data from {start}...")
workflow.backfill(start=start)
print("Backfill complete!")
```

### Step 2: Feature Engineering & Training

```python
# Get training data
print("Extracting features...")
df_train = workflow.get_full_feature_target_dataframe_pandas(start_date=start)

# Split features and target
feature_cols = [f"feature_{i}" for i in range(40)]
X = df_train[feature_cols]
y = df_train["target"]

print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")

# Train model
print("Training model...")
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31
)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
    
print("Model trained and saved!")
```

### Step 3: Live Inference Loop

```python
import time

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Starting live inference loop...")

while True:
    try:
        # Get live features (fresh from API)
        features = workflow.get_live_features("BTCUSDT")
        
        # Predict
        prediction = model.predict(features)[0]
        timestamp = features.index[0]
        
        print(f"[{timestamp}] BTCUSDT Prediction: {prediction:.6f}")
        
        # Your trading logic here...
        
    except ValueError as e:
        print(f"Warning: {e}")
    
    # Wait for next bar (5 minutes)
    time.sleep(5 * 60)
```

### Step 4: Deploy to Allora Network

```python
from allora_sdk.worker import AlloraWorker
import asyncio

# Load workflow and model (same as above)
workflow = AlloraMLWorkflow(...)
model = pickle.load(open("model.pkl", "rb"))

# Define predict function
def predict():
    features = workflow.get_live_features("btcusd")
    prediction = model.predict(features)[0]
    return prediction

# Deploy worker
async def main():
    worker = AlloraWorker(
        predict_fn=predict,
        api_key="your-allora-api-key",
        topic_id=69  # Optional
    )

    async for result in worker.run():
        if isinstance(result, Exception):
            print(f"Error: {str(result)}")
        else:
            print(f"Prediction submitted: {result.prediction}")

# Run
asyncio.run(main())
```

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
- ✅ 17 Unit Tests (factory, initialization, storage)
- ✅ 6 Integration Tests (Binance + Allora live APIs)

**Integration Tests:**
- `test_binance_get_live_1min_data` - Verify 1-minute data fetching
- `test_allora_get_live_1min_data` - Verify 1-minute data fetching
- `test_workflow_binance_get_live_features` - End-to-end validation
- `test_workflow_allora_get_live_features` - End-to-end validation
- `test_workflow_live_mode_offset` - Visual inspection of offset logic
- `test_workflow_get_live_features_end_to_end` - Full validation

See [tests/README.md](tests/README.md) for detailed documentation.

---

## 🔧 Advanced Configuration

### Custom Data Manager

```python
from allora_forge_builder_kit import BinanceDataManager

# Create custom data manager
custom_dm = BinanceDataManager(
    base_dir="/path/to/data",
    interval="15m",
    market="spot",
    cache_len=2000,
    batch_timeout=30,
    rate_limit=0.3
)

# Pass to workflow
workflow = AlloraMLWorkflow(
    tickers=["BTCUSDT"],
    hours_needed=32,
    number_of_input_candles=8,
    target_length=16,
    data_manager=custom_dm  # Use custom manager
)
```

### Multiple Intervals

```python
# 5-minute bars for high-frequency
workflow_5m = AlloraMLWorkflow(interval="5m", ...)

# 1-hour bars for day trading
workflow_1h = AlloraMLWorkflow(interval="1h", ...)

# 4-hour bars for swing trading
workflow_4h = AlloraMLWorkflow(interval="4h", ...)
```

### Different Lookback Periods

```python
# Short-term (8 hours)
workflow_short = AlloraMLWorkflow(hours_needed=8, ...)

# Medium-term (32 hours)
workflow_medium = AlloraMLWorkflow(hours_needed=32, ...)

# Long-term (168 hours = 7 days)
workflow_long = AlloraMLWorkflow(hours_needed=168, ...)
```

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
- **Comprehensive docs**: AGENT_GUIDE.md explains everything
- **Easy installation**: `pip install git+https://github.com/...`

### For Traders
- **Live inference**: Always fresh data, no stale predictions
- **Multi-source**: Compare Binance vs Allora data
- **Real-time streaming**: WebSocket support for Binance
- **Backtesting**: Historical data readily available

### For Developers
- **Modular architecture**: Easy to extend with new data sources
- **Well-tested**: 23 tests covering all functionality
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
- Python 3.8 or higher

### Core Dependencies
```
pandas>=1.3.0
polars>=0.20.0
requests>=2.31.0
websocket-client>=1.6.0
numba>=0.58.0
```

### ML Dependencies (Optional)
```
lightgbm>=4.0.0
scikit-learn>=1.3.0
```

### Allora SDK (For Deployment)
```
allora-sdk>=0.1.0
```

See `requirements.txt` or `environment.yml` for complete list.

---

## 🗺️ Roadmap

### v2.1 (Coming Soon)
- [ ] Additional data sources (Coinbase, Kraken)
- [ ] Built-in model registry
- [ ] Advanced feature engineering templates
- [ ] Performance profiling tools

### v2.2 (Future)
- [ ] Multi-asset portfolio support
- [ ] Custom indicator library
- [ ] AutoML integration
- [ ] Dashboard for monitoring

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

- **Documentation**: [AGENT_GUIDE.md](AGENT_GUIDE.md) | [ARCHITECTURE.md](ARCHITECTURE.md)
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

