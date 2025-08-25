[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![GitHub issues](https://img.shields.io/github/issues/allora-network/allora-forge-builder-kit)](https://github.com/allora-network/allora-forge-builder-kit/issues)
[![Last Commit](https://img.shields.io/github/last-commit/allora-network/allora-forge-builder-kit)](https://github.com/allora-network/allora-forge-builder-kit/commits/main)
[![Stars](https://img.shields.io/github/stars/allora-network/allora-forge-builder-kit?style=social)](https://github.com/allora-network/allora-forge-builder-kit/stargazers)

<img width="1536" height="1024" alt="forge_silicon" src="https://github.com/user-attachments/assets/f1444abf-e649-4e48-a9f0-187b78b59ccc" />


# Allora Forge Builder Kit

Welcome to **Allora Forge Builder Kit**, a cutting-edge machine learning workflow package designed to streamline your ML pipeline. Whether you're a seasoned data scientist or just starting your journey, the Forge Builder Kit provides the tools you need to build, evaluate, and deploy ML models with ease.  

---

## API Key
Navigate to https://developer.allora.network/, register, and create your free API key. 

---

## Features

### 1. **Automated Dataset Generation**
Effortlessly generate datasets with train/validation/test splits. Allora Forge takes care of the heavy lifting, ensuring your data is ready for modeling in no time.  

### 2. **Dynamic Feature Engineering**
Leverage automated feature generation based on historical bar data. The package dynamically creates features tailored to your dataset, saving you hours of manual work.  

### 3. **Built-in Evaluation Metrics**
Evaluate your models with a suite of built-in metrics, designed to provide deep insights into performance. From accuracy to precision, Allora Forge has you covered.  

### 4. **Model Export for Live Inference**
Export your trained models seamlessly for live inference. Deploy your models in production environments with confidence and minimal effort.  

---

## Example Notebook Highlights

Explore the full pipeline in action in the included Jupyter notebooks. The first is a barebones ML workflow to get a feel for how it works.

[Allora Forge ML Workflow Example](https://github.com/allora-network/allora-forge-builder-kit/blob/main/notebooks/Allora%20Forge%20Builder%20Kit.ipynb)

The second is a more robust grid search pipeline, where you evaluate many models, choose the best, and deploy it live.

[Allora Forge Signal Miner Example](https://github.com/allora-network/allora-forge-ml-workflow/blob/main/notebooks/Allora%20Forge%20Signal%20Miner.ipynb)

The example notebook included in the repository demonstrates:  
- **Dataset Creation**: Automatically split your data into train/validation/test sets.  
- **Feature Engineering**: Generate dynamic features from historical bar data.  
- **Model Training**: Train your ML models with ease.  
- **Evaluation**: Use built-in metrics to assess model performance.  
- **Export**: Save your model for live inference deployment.  

### Quickstart Example Code

```python
from allora_forge_builder_kit import AlloraMLWorkflow, get_api_key #Allora Forge
import lightgbm as lgb
import pandas as pd

tickers = ["btcusd", "ethusd", "solusd"]
hours_needed = 1*24             # Number of historical hours for feature lookback window
number_of_input_candles = 24    # Number of candles for input features
target_length = 1*24            # Number of hours into the future for target

# Instantiate the workflow
workflow = AlloraMLWorkflow(
    data_api_key=get_api_key(),
    tickers=tickers,
    hours_needed=hours_needed,
    number_of_input_candles=number_of_input_candles,
    target_length=target_length
)

# Get training, validation, and test data
X_train, y_train, X_val, y_val, X_test, y_test = workflow.get_train_validation_test_data(
    from_month="2023-01",
    validation_months=3,
    test_months=3
)

# Define feature columns and ML model
feature_cols = [f for f in list(X_train) if 'feature' in f]

# Define hyperparameters for the LightGBM model
learning_rate = 0.001
max_depth = 5
num_leaves = 8

# Initialize LightGBM model with hyperparameters
model = lgb.LGBMRegressor(
    n_estimators=50,
    learning_rate=learning_rate,
    max_depth=max_depth,
    num_leaves=num_leaves
)

model.fit(
    pd.concat([X_train[feature_cols], X_val[feature_cols]]), 
    pd.concat([y_train, y_val])
)

# Evaluate on the test data
test_preds = model.predict(X_test[feature_cols])
test_preds = pd.Series(test_preds, index=X_test.index)

# Show test metrics
metrics = workflow.evaluate_test_data(test_preds)
print(metrics)
```

> {'correlation': 0.038930690096235177, 'directional_accuracy': 0.5414329504839673}

### Model Deployment for Live Inference on the Allora Network

```python
from allora_sdk.worker import AlloraWorker

# Final predict function
def predict() -> pd.Series:
    live_features = workflow.get_live_features("btcusd")
    preds = model.predict(live_features)
    return pd.Series(preds, index=live_features.index)

# Pickle the function
with open("predict.pkl", "wb") as f:
    dill.dump(predict, f)

# Load the pickled predict function
with open("predict.pkl", "rb") as f:
    predict_fn = dill.load(f)


def my_model():
    # Call the function and get predictions
    tic = time.time()
    prediction = predict_fn()
    toc = time.time()

    print("predict time: ", (toc - tic) )
    print("prediction: ", prediction )
    return prediction

async def main():
    worker = AlloraWorker(
        # topic_id=69,  ### THIS IS OPTIONAL -- TOPIC 69 IS OPEN TO EVERYONE
        predict_fn=my_model,
        api_key="<your API key>",
    )

    async for result in worker.run():
        if isinstance(result, Exception):
            print(f"Error: {str(result)}")
        else:
            print(f"Prediction submitted to Allora: {result.prediction}")

# IF RUNNING IN A NOTEBOOK:
await main()

# OR IF RUNNING FROM THE TERMINAL
asyncio.run(main())
```
> predict time:  0.49544739723205566
> prediction:  2025-08-05 17:15:00+00:00    0.002185
---

## Get Started

Dive into the future of machine learning workflows with Allora Forge. Check out the example notebook to see the magic in action and start building your next ML project today!  

**Welcome to the Forge.**  

---

# AlloraMLWorkflow Documentation

The `AlloraMLWorkflow` class provides methods to fetch, preprocess, and prepare financial time-series data for machine learning workflows.

---

## Class Initialization

```python
AlloraMLWorkflow(data_api_key, tickers, hours_needed, number_of_input_candles, target_length)
```

**Arguments:**

- `data_api_key` (`str`): API key for accessing market data.
- `tickers` (`list[str]`): List of ticker symbols to fetch data for.
- `hours_needed` (`int`): Lookback window (in hours) for feature extraction.
- `number_of_input_candles` (`int`): Number of candles to segment the lookback window into.
- `target_length` (`int`): Target horizon in hours for predictive modeling.

---

## Methods

### `compute_from_date(extra_hours: int = 12) -> str`

Compute a starting date string based on the lookback window.

**Arguments:**

- `extra_hours` (`int`, default=12): Additional buffer hours before the cutoff.

**Returns:**

- `str` – Date string in format `YYYY-MM-DD`.

---

### `list_ready_buckets(ticker, from_month) -> list`

Fetch list of ready data buckets for a ticker.

**Arguments:**

- `ticker` (`str`): Ticker symbol.
- `from_month` (`str`): Month in format `YYYY-MM`.

**Returns:**

- `list[dict]` – Buckets where `state == "ready"`.

---

### `fetch_bucket_csv(download_url) -> pd.DataFrame`

Download and load bucket CSV data.

**Arguments:**

- `download_url` (`str`): URL of the CSV file.

**Returns:**

- `pd.DataFrame` – Data from the bucket.

---

### `fetch_ohlcv_data(ticker, from_date: str, max_pages: int = 1000, sleep_sec: float = 0.1) -> pd.DataFrame`

Fetch OHLCV data from the API, handling pagination.

**Arguments:**

- `ticker` (`str`): Ticker symbol.
- `from_date` (`str`): Starting date (`YYYY-MM-DD`).
- `max_pages` (`int`, default=1000): Maximum pages to fetch.
- `sleep_sec` (`float`, default=0.1): Sleep between requests.

**Returns:**

- `pd.DataFrame` – Cleaned OHLCV dataset.

---

### `create_5_min_bars(df: pd.DataFrame, live_mode: bool = False) -> pd.DataFrame`

Resample 1-minute OHLCV data into 5-minute bars.

**Arguments:**

- `df` (`pd.DataFrame`): Input data indexed by datetime.
- `live_mode` (`bool`, default=False): Whether to adjust for incomplete live data.

**Returns:**

- `pd.DataFrame` – 5-minute bar data.

---

### `compute_target(df: pd.DataFrame, hours: int = 24) -> pd.DataFrame`

Compute log return target over a future horizon.

**Arguments:**

- `df` (`pd.DataFrame`): OHLCV data with `close` column.
- `hours` (`int`, default=24): Horizon for target calculation.

**Returns:**

- `pd.DataFrame` – DataFrame with `future_close` and `target` columns.

---

### `extract_rolling_daily_features(data: pd.DataFrame, lookback: int, number_of_candles: int, start_times: list) -> pd.DataFrame`

Extract normalized OHLCV features over rolling windows.

**Arguments:**

- `data` (`pd.DataFrame`): Input OHLCV data with `date` index.
- `lookback` (`int`): Lookback window (in hours).
- `number_of_candles` (`int`): Number of candles to split the window into.
- `start_times` (`list[datetime]`): Anchor times for feature extraction.

**Returns:**

- `pd.DataFrame` – Extracted rolling feature set.

---

### `get_live_features(ticker) -> pd.DataFrame`

Fetch and compute live features for a ticker.

**Arguments:**

- `ticker` (`str`): Ticker symbol.

**Returns:**

- `pd.DataFrame` – Latest extracted features for live inference.

---

### `evaluate_test_data(predictions: pd.Series) -> dict`

Evaluate predictions against stored test targets.

**Arguments:**

- `predictions` (`pd.Series`): Predicted values (index must match test targets).

**Returns:**

- `dict` with keys:
  - `"correlation"` (`float`): Pearson correlation with true targets.
  - `"directional_accuracy"` (`float`): Fraction of correct directional predictions.

---

### `get_full_feature_target_dataframe(from_month="2025-01") -> pd.DataFrame`

Build complete dataset with features and targets for all tickers.

**Arguments:**

- `from_month` (`str`, default="2025-01"): Starting month for bucket retrieval.

**Returns:**

- `pd.DataFrame` – Full dataset indexed by `(date, ticker)`.

---

### `get_train_validation_test_data(from_month="2025-01", validation_months=3, test_months=3, force_redownload=False)`

Prepare train/validation/test datasets with caching.

**Arguments:**

- `from_month` (`str`, default="2025-01\`): Starting month for data retrieval.
- `validation_months` (`int`, default=3): Number of months for validation set.
- `test_months` (`int`, default=3): Number of months for test set.
- `force_redownload` (`bool`, default=False): If `True`, re-download instead of loading cached data.

**Returns:**

- `tuple` – `(X_train, y_train, X_val, y_val, X_test, y_test)` as `pd.DataFrame` / `pd.Series`.


