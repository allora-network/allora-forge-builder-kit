#!/usr/bin/env python3
"""
Allora Forge Builder Kit v2.0 - Topic 69 Example
=================================================

This script demonstrates how to:
1. Use the v2.0 workflow with Allora data (fast monthly bucket loading)
2. Train a model for 1-day Bitcoin price prediction (Topic 69)
3. Convert log return predictions to actual price predictions
4. Package the predict function for deployment to Allora Network

Topic 69: 24-hour Bitcoin Price Prediction
- Predict BTC price 24 hours into the future
- Train on log returns (standard ML practice)
- Deploy predictions as actual prices (what Topic 69 expects)

Note: This example uses Allora data which loads much faster than Binance
thanks to pre-aggregated monthly buckets. You can also use Binance by changing
`data_source="allora"` to `data_source="binance"` (no API key required).
"""

from allora_forge_builder_kit import AlloraMLWorkflow
from datetime import datetime, timedelta, timezone
import lightgbm as lgb
import pandas as pd
import numpy as np
import cloudpickle

print("="*80)
print("Allora Forge Builder Kit v2.0 - Topic 69: Bitcoin Price Prediction")
print("="*80)

# =============================================================================
# STEP 0: Load Allora API Key
# =============================================================================

from pathlib import Path

api_key_file = Path("notebooks/.allora_api_key")
if api_key_file.exists():
    ALLORA_API_KEY = api_key_file.read_text().strip()
    print(f"✅ Loaded Allora API key from {api_key_file}")
else:
    raise FileNotFoundError(f"API key file not found: {api_key_file}")

# =============================================================================
# STEP 1: Configure Workflow for Topic 69
# =============================================================================

print("\n[1/7] Configuring workflow for Topic 69...")

# Configuration for 1-day Bitcoin price prediction
TICKER = "btcusd"               # Allora format (lowercase, no separator)
HOURS_NEEDED = 7 * 24           # 7 days lookback
NUMBER_OF_CANDLES = 24          # 24 candles in features
TARGET_LENGTH = 1 * 24          # 1 day (24 hours) prediction horizon
INTERVAL = "1h"                 # 1-hour bars

# Create workflow with Allora data source (v2.0) - faster loading!
workflow = AlloraMLWorkflow(
    tickers=[TICKER],
    hours_needed=HOURS_NEEDED,
    number_of_input_candles=NUMBER_OF_CANDLES,
    target_length=TARGET_LENGTH,
    interval=INTERVAL,
    data_source="allora",       # v2.0: Allora Network (faster than Binance)
    api_key=ALLORA_API_KEY
)

print(f"✅ Workflow configured:")
print(f"   - Asset: {TICKER}")
print(f"   - Lookback: {HOURS_NEEDED} hours ({HOURS_NEEDED//24} days)")
print(f"   - Prediction horizon: {TARGET_LENGTH} hours ({TARGET_LENGTH//24} day)")
print(f"   - Interval: {INTERVAL}")
print(f"   - Features: {NUMBER_OF_CANDLES} candles × 5 OHLCV = {NUMBER_OF_CANDLES*5} features")

# =============================================================================
# STEP 2: Backfill Historical Data
# =============================================================================

print("\n[2/7] Backfilling historical data...")

# Backfill data (Allora uses monthly buckets, very fast!)
start_date = datetime.now(timezone.utc) - timedelta(days=180)  # 6 months
print(f"   Fetching data from {start_date.date()} to now...")
print(f"   (Allora loads from monthly buckets - much faster than Binance!)")

workflow.backfill(start=start_date)

print("✅ Backfill complete!")

# =============================================================================
# STEP 3: Extract Features and Targets
# =============================================================================

print("\n[3/7] Extracting features and targets...")

# Get full dataset with features and log return targets
df_all = workflow.get_full_feature_target_dataframe_pandas(start_date=start_date)

# Reset index to make it easier to work with
df_all = df_all.reset_index()

print(f"✅ Extracted {len(df_all):,} samples")
print(f"   Columns: {list(df_all.columns[:5])}...")
print(f"   Target: log returns (will be converted to prices for prediction)")

# =============================================================================
# STEP 4: Split Train/Validation/Test
# =============================================================================

print("\n[4/7] Splitting data...")

# 70% train, 15% validation, 15% test
n = len(df_all)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Get actual feature columns from DataFrame
feature_cols = [col for col in df_all.columns if col.startswith('feature_')]

X_train = df_all.iloc[:train_end][feature_cols]
y_train = df_all.iloc[:train_end]["target"]

X_val = df_all.iloc[train_end:val_end][feature_cols]
y_val = df_all.iloc[train_end:val_end]["target"]

X_test = df_all.iloc[val_end:][feature_cols]
y_test = df_all.iloc[val_end:]["target"]

print(f"✅ Data split:")
print(f"   Features: {len(feature_cols)} columns")
print(f"   Training:   {len(X_train):>6,} samples ({len(X_train)/n*100:.1f}%)")
print(f"   Validation: {len(X_val):>6,} samples ({len(X_val)/n*100:.1f}%)")
print(f"   Test:       {len(X_test):>6,} samples ({len(X_test)/n*100:.1f}%)")

# Show date ranges (using 'date' column from DataFrame)
if 'date' in df_all.columns:
    print(f"\n   Training dates:   {df_all.iloc[0]['date']} to {df_all.iloc[train_end-1]['date']}")
    print(f"   Validation dates: {df_all.iloc[train_end]['date']} to {df_all.iloc[val_end-1]['date']}")
    print(f"   Test dates:       {df_all.iloc[val_end]['date']} to {df_all.iloc[-1]['date']}")

# =============================================================================
# STEP 5: Train Model
# =============================================================================

print("\n[5/7] Training LightGBM model...")

# Train on log returns (standard practice)
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate on validation set
val_preds = model.predict(X_val)
val_correlation = np.corrcoef(y_val, val_preds)[0, 1]
val_directional = np.mean(np.sign(y_val) == np.sign(val_preds))

print(f"✅ Model trained!")
print(f"   Validation correlation: {val_correlation:.4f}")
print(f"   Validation directional accuracy: {val_directional:.4f}")

# =============================================================================
# STEP 6: Test on Hold-Out Set
# =============================================================================

print("\n[6/7] Evaluating on test set...")

test_preds = model.predict(X_test)
test_correlation = np.corrcoef(y_test, test_preds)[0, 1]
test_directional = np.mean(np.sign(y_test) == np.sign(test_preds))

print(f"✅ Test results:")
print(f"   Correlation: {test_correlation:.4f}")
print(f"   Directional accuracy: {test_directional:.4f}")

# =============================================================================
# STEP 7: Create Prediction Function for Topic 69
# =============================================================================

print("\n[7/7] Creating prediction function for Topic 69...")

# Retrain on ALL data for production
model.fit(
    pd.concat([X_train, X_val, X_test]),
    pd.concat([y_train, y_val, y_test])
)
print("✅ Model retrained on full dataset")

# Define prediction function
def predict() -> float:
    """
    Predict Bitcoin price 24 hours into the future (Topic 69).
    
    Returns:
        float: Predicted BTC price in USD
    """
    # Get live features (fetches fresh 1-min data, resamples to 1h, extracts features)
    live_features = workflow.get_live_features(TICKER)
    
    # Predict log return (what the model was trained on)
    predicted_log_return = model.predict(live_features)[0]
    
    # Get current price (last close)
    now = datetime.now(timezone.utc)
    recent_start = now - timedelta(hours=2)  # Get last 2 hours
    raw_data = workflow.load_raw(start=recent_start, end=now)
    current_price = raw_data["close"].iloc[-1]
    
    # Convert log return to predicted price
    # Formula: price_tomorrow = price_today * exp(log_return)
    predicted_price = current_price * np.exp(predicted_log_return)
    
    print(f"\n{'='*60}")
    print(f"Live Prediction for {TICKER}")
    print(f"{'='*60}")
    print(f"Current price:           ${current_price:>12,.2f}")
    print(f"Predicted log return:    {predicted_log_return:>12.6f}")
    print(f"Predicted price (24h):   ${predicted_price:>12,.2f}")
    print(f"Predicted change:        ${predicted_price - current_price:>12,.2f}")
    print(f"Predicted % change:      {(predicted_price/current_price - 1)*100:>12.2f}%")
    print(f"{'='*60}")
    
    return float(predicted_price)

# Test the prediction function
print("\n🧪 Testing prediction function...")
try:
    test_prediction = predict()
    print(f"\n✅ SUCCESS! Prediction function works.")
    print(f"   Predicted BTC price (24h): ${test_prediction:,.2f}")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Save the prediction function
print("\n📦 Saving prediction function...")
with open("predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)
    
print("✅ Prediction function saved to predict.pkl")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✅ Model trained on {len(df_all):,} samples of log returns")
print(f"✅ Test correlation: {test_correlation:.4f}")
print(f"✅ Test directional accuracy: {test_directional:.4f}")
print(f"✅ Prediction function converts log returns to actual prices")
print(f"✅ Ready for deployment to Allora Network (Topic 69)")
print("="*80)

print("\n📝 Next steps:")
print("   1. Test the prediction: python3 -c 'import cloudpickle; f=open(\"predict.pkl\",\"rb\"); predict=cloudpickle.load(f); print(predict())'")
print("   2. Deploy to Allora Network using the worker code below")
print("="*80)

print("""
# Worker code for deployment:

from allora_sdk.worker import AlloraWorker
import cloudpickle
import asyncio

async def main():
    with open("predict.pkl", "rb") as f:
        predict_fn = cloudpickle.load(f)
    
    worker = AlloraWorker(
        predict_fn=predict_fn,
        api_key="your-allora-api-key",
        topic_id=69  # Topic 69: 24-hour Bitcoin Price Prediction
    )
    
    async for result in worker.run():
        if isinstance(result, Exception):
            print(f"Error: {str(result)}")
        else:
            print(f"Prediction submitted: ${result.prediction:,.2f}")

asyncio.run(main())
""")

print("\n🚀 Happy predicting!")

