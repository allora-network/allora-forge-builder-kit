# AGENTS.md — Allora Forge Builder Kit

> Instructions for AI agents working with this repository.

## Project Overview

The **Allora Forge Builder Kit** is a Python toolkit for building,
evaluating, and deploying machine-learning models to the
[Allora Network](https://allora.network).  It provides:

- **Data management** — Fetch and store 1-minute OHLCV candle data from
  Atlas (Tiingo), Binance, or the legacy Allora API.
- **Feature engineering** — Resample bars to any interval and extract
  normalised OHLCV feature vectors with Numba-accelerated extraction.
- **Evaluation** — 7 primary metrics aligned with the Research team's
  framework, plus regression, classification, and trading-simulation
  metrics.
- **Topic discovery** — Query all active Allora Network topics and their
  configuration via the `allora_sdk` API client.
- **Deployment** — Pickle a prediction function and deploy it as an
  Allora worker with `allora_sdk >= 1.0.6`.

## Quick Start for Agents

```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install (editable + dev extras)
pip install -e ".[dev]"

# 3. Run the unit tests (fast, no network required)
pytest tests/test_data_managers.py -v -m "not integration"
```

### Typical workflow

```python
from allora_forge_builder_kit import (
    AlloraMLWorkflow,
    PerformanceEvaluator,
    AlloraTopicDiscovery,
)

# Discover available topics
discovery = AlloraTopicDiscovery(api_key="UP-...")
topics = discovery.get_all_topics()

# Train and evaluate a model
workflow = AlloraMLWorkflow(
    tickers=["btcusd"],
    number_of_input_bars=48,
    target_bars=24,
    interval="1h",
    data_source="allora",  # uses Atlas backend
    api_key="UP-...",
)
workflow.backfill(start=...)
df = workflow.get_full_feature_target_dataframe()
# ... train model ...
report = PerformanceEvaluator().evaluate(y_true, y_pred)
```

## Repository Layout

```
allora_forge_builder_kit/
  __init__.py               # Package exports (v3.0.0)
  workflow.py               # AlloraMLWorkflow — the main orchestrator
  atlas_data_manager.py     # NEW: Atlas data service (forge-data.allora.run)
  allora_data_manager.py    # Legacy Allora Network REST API
  binance_data_manager.py   # Binance REST + WebSocket
  base_data_manager.py      # Abstract base class for data managers
  data_manager_factory.py   # DataManager() factory function
  evaluation.py             # PerformanceEvaluator (7 primary metrics)
  topic_discovery.py        # AlloraTopicDiscovery via allora_sdk
  utils.py                  # Helpers (get_api_key)

notebooks/
  example_topic_69_bitcoin_walkthrough.py
  deploy_worker.py
  feature_engineering_example.py
  Allora Forge Builder Kit.ipynb

tests/
  test_data_managers.py     # 28 tests (16 unit + 12 integration)
```

## Data Sources

| Source key       | Backend                                 | Notes |
|------------------|-----------------------------------------|-------|
| `"allora"`       | Atlas (`forge-data.allora.run`)         | **Default.** Tiingo 1-min candles. |
| `"atlas"`        | Same as `"allora"`                      | Explicit alias. |
| `"allora-legacy"`| `api.allora.network` REST API           | Deprecated; kept for backward compat. |
| `"binance"`      | Binance spot/futures REST + WebSocket   | Free, no API key needed. |

Atlas requires an API key passed via `X-API-Key` header.  The same key
that worked with the legacy Allora data service works with Atlas.

## Evaluation Metrics

### 7 Primary Metrics (pass/fail thresholds)

Updated per the Research team's framework (RES-1271, RES-1293, RES-1257,
RES-1375).

| # | Metric | Threshold | Direction |
|---|--------|-----------|-----------|
| 1 | Directional Accuracy | >= 0.52 | higher is better |
| 2 | DA CI Lower Bound | >= 0.50 | higher is better |
| 3 | DA p-value (z-test, autocorr-aware) | < 0.05 | lower is better |
| 4 | Pearson Correlation (r) | >= 0.05 | higher is better |
| 5 | Pearson p-value | < 0.05 | lower is better |
| 6 | WRMSE Improvement vs baseline | >= 0.05 | higher is better |
| 7 | CZAR Improvement vs oracle | >= 0.10 | higher is better |

Key changes from the earlier (v2) framework:
- DA threshold lowered from 0.55 to 0.52.
- DA CI lower bound **kept** (>= 0.50); not removed.
- DA p-value now uses z-test with continuity correction and
  autocorrelation-aware effective sample size.
- WRMSE threshold lowered from 10% to 5%.
- ZPTAE replaced by CZAR (Cumulative Z-scored Absolute Return).
- Log Aspect Ratio moved to additional (non-scored) metrics.

### Composite Score

7 primary metrics + 1 temporal-coverage point = max 8 points.

| Points | Grade |
|--------|-------|
| 8      | A+    |
| 7      | A     |
| 6      | B+    |
| 5      | B     |
| 4      | C     |
| 3      | D     |
| < 3    | F     |

## Price vs Log-Return Topics

Allora topics come in two flavours.  Agents **must** format predictions
appropriately:

- **Price topics** — Submit an absolute price (e.g. `67000.0`).
- **Log-return topics** — Submit the predicted log return
  `log(future_price / current_price)`.

Use `AlloraTopicDiscovery` to inspect the `metadata` field of each topic
to determine its type.

## Key Conventions

- Python >= 3.10 required.
- All timestamps are UTC `datetime` objects.
- Tickers for the Allora/Atlas source should be lowercase without
  slashes: `btcusd`, `ethusd`, `solusd`.
- The package uses **Polars** for Parquet I/O and **Pandas** for the
  public-facing DataFrames.

## Testing

```bash
# Fast unit tests (no API keys needed)
pytest tests/test_data_managers.py -v -m "not integration"

# Full integration tests (requires ALLORA_API_KEY env var)
export RUN_INTEGRATION_TESTS=1
export ALLORA_API_KEY="UP-..."
pytest tests/test_data_managers.py -v
```

## Dependencies

Runtime: `pandas`, `numpy`, `lightgbm`, `requests`, `dill`, `polars`,
`numba`, `scipy`, `allora-sdk>=1.0.6`.

Optional: `websocket-client` (for Binance streaming), `cloudpickle`
(for pickling prediction functions).
