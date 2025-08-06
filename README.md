[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![GitHub issues](https://img.shields.io/github/issues/jefferythewind/allora-forge-ml-workflow)](https://github.com/jefferythewind/allora-forge-ml-workflow/issues)
[![Last Commit](https://img.shields.io/github/last-commit/jefferythewind/allora-forge-ml-workflow)](https://github.com/jefferythewind/allora-forge-ml-workflow/commits/main)
[![Stars](https://img.shields.io/github/stars/jefferythewind/allora-forge-ml-workflow?style=social)](https://github.com/jefferythewind/allora-forge-ml-workflow/stargazers)

<img width="1536" height="1024" alt="forge_silicon" src="https://github.com/user-attachments/assets/f1444abf-e649-4e48-a9f0-187b78b59ccc" />


# Allora Forge ML Workflow  

Welcome to **Allora Forge**, a cutting-edge machine learning workflow package designed to streamline your ML pipeline. Whether you're a seasoned data scientist or just starting your journey, Allora Forge provides the tools you need to build, evaluate, and deploy ML models with ease.  

---

## Installation  

Allora Forge will soon be available on PyPI. For now, you can install it directly from the GitHub repository:  

```bash  
pip install git+https://github.com/allora-network/allora-forge-ml-workflow.git
```  
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

Explore the full pipeline in action in the included Jupyter notebooks. The first is a bare bone ML workflow to get a feel for how it works.

[Allora Forge ML Workflow Example](https://github.com/jefferythewind/allora-forge-ml-workflow/blob/main/notebooks/Allora%20Forge%20ML%20Workflow.ipynb)

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
from allora_ml_workflow import AlloraMLWorkflow #Allora Forge
import lightgbm as lgb
import pandas as pd

tickers = ["btcusd", "ethusd", "solusd"]
hours_needed = 1*24             # Number of historical hours for feature lookback window
number_of_input_candles = 24    # Number of candles for input features
target_length = 1*24            # Number of hours into the future for target

# Instantiate the workflow
workflow = AlloraMLWorkflow(
    data_api_key=api_key,
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

### Model Deployment for Live Inference on The Allora Network
```python
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

# Call the function and get predictions
tic = time.time()
prediction = predict_fn()
toc = time.time()

print("predict time: ", (toc - tic) )
print("prediction: ", prediction )
```
> predict time:  0.49544739723205566
> prediction:  2025-08-05 17:15:00+00:00    0.002185
---

## Get Started  

Dive into the future of machine learning workflows with Allora Forge. Check out the example notebook to see the magic in action and start building your next ML project today!  

---  

**Stay sharp. Stay cyber. Welcome to the Forge.**  
