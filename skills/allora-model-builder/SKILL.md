# Allora Model Builder Skill

Build, evaluate, and deploy an ML model to the Allora Network using the
Forge Builder Kit.

## When to use

- User wants to create a price or log-return prediction model.
- User wants to deploy a model as an Allora worker.
- User asks how to contribute predictions to a topic.

## Prerequisites

```bash
pip install -e ".[dev]"
```

**API key — stop and ask the user before proceeding.**
Do not silently use a discovered key — treat it as human-confirmed input.
- If a key exists in env/file, tell the user and ask: "Should I use this key, or a different one?"
- If no key is found, prompt: "Sign up free at https://developer.allora.network and paste
  your key, or I can use `data_source='binance'` instead (no key needed)."
- **Wait for the user to respond.** Do not proceed to data/model steps without confirmation.

## Steps

### 1. Discover topics

```python
from allora_forge_builder_kit import AlloraTopicDiscovery

discovery = AlloraTopicDiscovery(api_key="UP-...")
topics = discovery.get_all_topics()
for t in topics:
    print(t.topic_id, t.metadata)
```

### 2. Initialise workflow

```python
from allora_forge_builder_kit import AlloraMLWorkflow

workflow = AlloraMLWorkflow(
    tickers=["btcusd"],
    number_of_input_bars=48,   # lookback window
    target_bars=24,            # prediction horizon (bars)
    interval="1h",
    data_source="allora",      # Atlas backend
    api_key="UP-...",
)
```

### 3. Backfill data

```python
from datetime import datetime, timedelta, timezone

start = datetime.now(timezone.utc) - timedelta(days=500)
workflow.backfill(start=start)
```

### 4. Extract features and train

```python
import numpy as np
from lightgbm import LGBMRegressor

df = workflow.get_full_feature_target_dataframe(start_date=start).reset_index()
feature_cols = [c for c in df.columns if c.startswith("feature_")]
df = df.dropna(subset=feature_cols + ["target"])

model = LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1)
model.fit(df[feature_cols], df["target"])
```

### 5. Evaluate

```python
from allora_forge_builder_kit import PerformanceEvaluator

y_pred = model.predict(df[feature_cols])
evaluator = PerformanceEvaluator()
report = evaluator.evaluate(df["target"].values, y_pred)
evaluator.print_report(report)
```

### 6. Deploy

```python
import cloudpickle

def predict(nonce=None):
    features = workflow.get_live_features("btcusd")
    log_return = model.predict(features[feature_cols].values.reshape(1, -1))[0]
    raw = workflow.load_raw()
    price = raw["close"].iloc[-1]
    return float(price * np.exp(log_return))

with open("predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
```

Then deploy with `allora_sdk`:

```python
from allora_sdk.worker import AlloraWorker

worker = AlloraWorker(
    topic_id=69,
    predict_fn=predict,
    api_key="UP-...",
)
```

## Key points

- Use `data_source="allora"` — this connects to Atlas (Tiingo candles).
- Tickers: lowercase, no slash (`btcusd`, `ethusd`).
- Evaluation uses **7 primary metrics** (DA, DA CI, DA p-value, Pearson r,
  Pearson p-value, WRMSE improvement, CZAR improvement) scored out of 7.
- For **price topics**, return an absolute price.
  For **log-return topics**, return the log return.

## Base feature normalization

Features from `get_full_feature_target_dataframe()` are **not raw prices** — they are normalized ratios:

- `feature_open_i`, `feature_high_i`, `feature_low_i`, `feature_close_i` → divided by the last bar's close (`feature_close_{N-1}` is always `1.0`)
- `feature_volume_i` → divided by the last bar's volume (`feature_volume_{N-1}` is always `1.0`)

Index `0` = oldest bar, index `N-1` = most recent bar.

**When adding engineered features** (TA indicators, log returns, etc.), normalize them consistently:
- Derive them from the already-normalized feature columns, OR
- Compute from raw data and divide by the same last-close / last-volume values

Mixing raw prices with normalized features will produce a badly conditioned model.
