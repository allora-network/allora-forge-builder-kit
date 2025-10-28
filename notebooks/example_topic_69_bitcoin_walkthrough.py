#!/usr/bin/env python3
"""
================================================================================
Allora Forge Builder Kit v2.0 - Topic 69 Bitcoin Price Prediction Walkthrough
================================================================================

This walkthrough demonstrates 24-hour Bitcoin price prediction using the 
Allora ML Workflow Kit with base features and LightGBM.

================================================================================
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
import cloudpickle
from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Data Configuration
TICKERS = ["btcusd"]
DAYS_OF_HISTORY = 500
INTERVAL = "1h"

# Feature Configuration
NUMBER_OF_INPUT_BARS = 24  # Number of hourly bars for input features
TARGET_BARS = 24           # Predict 24 bars (hours) ahead

# Cross-Validation Configuration
N_SPLITS = 3               # Number of CV folds
MIN_TRAIN_SIZE = 100       # Minimum training samples per fold
MAX_TRAIN_SIZE = 100_000_000  # Maximum training samples per fold

# Model Configuration
N_ESTIMATORS_MAX = 2000    # Train with max trees, evaluate at checkpoints
N_ESTIMATORS_CHECKPOINTS = [10, 100, 300, 500, 700, 1000, 1500, 2000]
LEARNING_RATES = [0.01, 0.05, 0.1, 0.3]
MAX_DEPTHS = [3, 5, 7]
NUM_LEAVES = [15, 31, 63]

# =============================================================================
# SCRIPT START
# =============================================================================

print("="*80)
print("Allora Forge Builder Kit v2.0 - Topic 69 Walkthrough")
print("="*80)

# =============================================================================
# STEP 1: Initialize Workflow
# =============================================================================
print("\n[1/6] Initializing workflow...")

# Read Allora API key
api_key_path = os.path.join(os.path.dirname(__file__), '.allora_api_key')
with open(api_key_path, 'r') as f:
    api_key = f.read().strip()

workflow = AlloraMLWorkflow(
    tickers=TICKERS,
    number_of_input_bars=NUMBER_OF_INPUT_BARS,
    target_bars=TARGET_BARS,
    interval=INTERVAL,
    data_source="allora",
    api_key=api_key
)

print(f"✅ Workflow initialized")
print(f"   Assets: {TICKERS} | Interval: {INTERVAL}")
print(f"   Input: {NUMBER_OF_INPUT_BARS} bars → Features: {NUMBER_OF_INPUT_BARS*5}")
print(f"   Target: {TARGET_BARS} bars ahead")

# =============================================================================
# STEP 2: Backfill Historical Data
# =============================================================================
print(f"\n[2/6] Backfilling {DAYS_OF_HISTORY} days of historical data...")

start_date = datetime.now(timezone.utc) - timedelta(days=DAYS_OF_HISTORY)
workflow.backfill(start=start_date)
print("✅ Backfill complete")

# =============================================================================
# STEP 3: Extract Features & Setup Cross-Validation
# =============================================================================
print("\n[3/6] Extracting features and setting up CV...")

df_all = workflow.get_full_feature_target_dataframe(start_date=start_date).reset_index()
feature_cols = [col for col in df_all.columns if col.startswith('feature_')]
df_all = df_all.dropna(subset=feature_cols + ['target'])

print(f"✅ Dataset: {len(df_all):,} samples ({df_all['open_time'].min().date()} to {df_all['open_time'].max().date()})")
print(f"   Features: {len(feature_cols)} base features")

# Setup time series cross-validation
tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=TARGET_BARS, max_train_size=MAX_TRAIN_SIZE)

print(f"✅ Walk-forward CV: {N_SPLITS} splits, {TARGET_BARS}-bar embargo")
for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df_all)):
    print(f"   Fold {fold_idx+1}: Train={len(train_idx):,}, Test={len(test_idx):,}")

# =============================================================================
# STEP 4: Grid Search with Walk-Forward Cross-Validation
# =============================================================================
print("\n[4/6] Running grid search...")

results = []
evaluator = PerformanceEvaluator()
config_num = 0

for lr in LEARNING_RATES:
    for depth in MAX_DEPTHS:
        for leaves in NUM_LEAVES:
            
            # Train once with max trees, evaluate at checkpoints
            fold_models = []
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df_all)):
                X_train = df_all.iloc[train_idx][feature_cols]
                y_train = df_all.iloc[train_idx]['target']
                
                lgb = LGBMRegressor(
                    n_estimators=N_ESTIMATORS_MAX,
                    learning_rate=lr,
                    max_depth=depth,
                    num_leaves=leaves,
                    random_state=42,
                    verbose=-1
                )
                lgb.fit(X_train, y_train)
                fold_models.append((lgb, test_idx))
            
            # Evaluate at tree count checkpoints
            for n_est in N_ESTIMATORS_CHECKPOINTS:
                config_num += 1
                df_all['pred'] = np.nan
                
                # Generate predictions using first n_est trees
                for lgb, test_idx in fold_models:
                    X_test = df_all.iloc[test_idx][feature_cols]
                    preds = lgb.predict(X_test, num_iteration=n_est)
                    df_all.iloc[test_idx, df_all.columns.get_loc('pred')] = preds
                
                # Evaluate
                valid_mask = ~df_all['pred'].isna()
                metrics = evaluator.evaluate(
                    y_true=df_all.loc[valid_mask, 'target'],
                    y_pred=df_all.loc[valid_mask, 'pred']
                )
                
                # Store results
                results.append({
                    'config_num': config_num,
                    'n_estimators': n_est,
                    'learning_rate': lr,
                    'max_depth': depth,
                    'num_leaves': leaves,
                    'predictions': df_all['pred'].copy(),
                    **metrics
                })
                
                print(f"   [{config_num:2d}] n={n_est:4d}, lr={lr:.2f}, d={depth}, l={leaves:2d} → "
                      f"{metrics['num_passed']}/8 ({metrics['score']:.1%} - {metrics['grade']})")

# Analyze results
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'predictions'} for r in results])
results_df = results_df.sort_values(['num_passed', 'score'], ascending=[False, False])

print(f"\n✅ Tested {len(results)} configurations")
print(f"\n   Top 5 models:")
top5_cols = ['config_num', 'n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'num_passed', 'score']
print(results_df[top5_cols].head().to_string(index=False))

# Select best model
best_result = results[results_df.iloc[0]['config_num'] - 1]
best_params = {k: best_result[k] for k in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves']}

print(f"\n✅ Best: Config #{best_result['config_num']}")
print(f"   {best_result['num_passed']}/8 metrics ({best_result['score']:.1%}) | "
      f"n={best_params['n_estimators']}, lr={best_params['learning_rate']}, d={best_params['max_depth']}, l={best_params['num_leaves']}")

# =============================================================================
# STEP 5: Evaluate Best Model
# =============================================================================
print("\n[5/6] Detailed evaluation...")
print("="*80)
evaluator.print_report(best_result, detailed=False)
print("="*80)

# =============================================================================
# STEP 6: Train Production Model
# =============================================================================
print("\n[6/6] Training production model...")

final_model = LGBMRegressor(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    num_leaves=best_params['num_leaves'],
    random_state=42,
    verbose=-1
)
final_model.fit(df_all[feature_cols], df_all['target'])
print(f"✅ Final model trained on {len(df_all):,} samples")

def predict(nonce: int = None) -> float:
    """
    Predict Bitcoin price 24 hours into the future.
    
    Args:
        nonce: Block nonce from Allora SDK (unused)
    
    Returns:
        float: Predicted BTC price in USD
    """
    # Get live features from workflow
    live_row = workflow.get_live_features(ticker=TICKERS[0])
    
    if live_row is None or len(live_row) == 0:
        raise ValueError("Could not get live features")
    
    # Get current price
    now = datetime.now(timezone.utc)
    recent_start = now - timedelta(hours=1)
    raw_data = workflow.load_raw(start=recent_start, end=now)
    current_price = raw_data["close"].iloc[-1] if len(raw_data) > 0 else 0.0
    
    # Use base features directly
    live_features = live_row[feature_cols]
    
    # Predict log return
    predicted_log_return = final_model.predict(live_features)[0]
    
    # Convert log return to price
    predicted_price = current_price * np.exp(predicted_log_return)
    
    print(f"\nLive Prediction: ${predicted_price:,.2f} ({predicted_log_return:+.4f} log return)")
    
    return float(predicted_price)

# Test and save
print("\n🧪 Testing prediction...")
test_prediction = predict()

with open("predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)

# Summary
print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"✅ {len(feature_cols)} features | {best_result['num_passed']}/8 metrics ({best_result['score']:.1%})")
print(f"✅ Saved to predict.pkl")
print("="*80)
print("\n🚀 Deploy: python deploy_worker.py")

