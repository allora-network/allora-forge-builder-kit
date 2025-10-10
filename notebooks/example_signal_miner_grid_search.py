#!/usr/bin/env python3
"""
Allora Forge Signal Miner v2.0 - Hyperparameter Grid Search
===========================================================

This script demonstrates a systematic model selection approach:
1. Grid search over hyperparameters
2. Cross-validation for robust evaluation
3. Model selection based on validation performance
4. Retraining on full dataset for deployment

Inspired by the ML MCP Server's experiment tracking and systematic optimization.
"""

from allora_forge_builder_kit import AlloraMLWorkflow
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sklearn.model_selection import ParameterSampler
import lightgbm as lgb
import pandas as pd
import numpy as np
import cloudpickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Allora Forge Signal Miner v2.0 - Grid Search")
print("="*80)

# =============================================================================
# STEP 0: Load API Key
# =============================================================================

api_key_file = Path("notebooks/.allora_api_key")
if api_key_file.exists():
    ALLORA_API_KEY = api_key_file.read_text().strip()
    print(f"✅ Loaded Allora API key")
else:
    raise FileNotFoundError(f"API key file not found: {api_key_file}")

# =============================================================================
# STEP 1: Configuration
# =============================================================================

print("\n[1/8] Configuration...")

# Assets and timeframe
TICKERS = ["btcusd"]  # Start with one ticker for speed
HOURS_NEEDED = 7 * 24       # 7 days lookback
NUMBER_OF_CANDLES = 24      # 24 candles
TARGET_LENGTH = 1 * 24      # 1 day prediction
INTERVAL = "1h"             # 1-hour bars

# Grid search settings
N_RANDOM_SAMPLES = 10       # Number of random parameter combinations
N_CV_FOLDS = 3              # Number of cross-validation folds

print(f"✅ Configuration:")
print(f"   Assets: {TICKERS}")
print(f"   Lookback: {HOURS_NEEDED} hours ({HOURS_NEEDED//24} days)")
print(f"   Prediction: {TARGET_LENGTH} hours ({TARGET_LENGTH//24} day)")
print(f"   Grid search: {N_RANDOM_SAMPLES} models × {N_CV_FOLDS} folds = {N_RANDOM_SAMPLES * N_CV_FOLDS} training runs")

# =============================================================================
# STEP 2: Create Workflow
# =============================================================================

print("\n[2/8] Creating workflow...")

workflow = AlloraMLWorkflow(
    tickers=TICKERS,
    hours_needed=HOURS_NEEDED,
    number_of_input_candles=NUMBER_OF_CANDLES,
    target_length=TARGET_LENGTH,
    interval=INTERVAL,
    data_source="allora",
    api_key=ALLORA_API_KEY
)

print("✅ Workflow created")

# =============================================================================
# STEP 3: Backfill Data
# =============================================================================

print("\n[3/8] Backfilling data...")

start_date = datetime.now(timezone.utc) - timedelta(days=180)  # 6 months
workflow.backfill(start=start_date)

print("✅ Backfill complete")

# =============================================================================
# STEP 4: Extract Features
# =============================================================================

print("\n[4/8] Extracting features...")

df_all = workflow.get_full_feature_target_dataframe_pandas(start_date=start_date)
df_all = df_all.reset_index()

feature_cols = [col for col in df_all.columns if col.startswith('feature_')]

print(f"✅ Extracted {len(df_all):,} samples with {len(feature_cols)} features")

# =============================================================================
# STEP 5: Define Parameter Grid
# =============================================================================

print("\n[5/8] Defining parameter grid...")

# LightGBM parameter space
param_space = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [10, 20, 50],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Sample random combinations
param_grid = list(ParameterSampler(param_space, n_iter=N_RANDOM_SAMPLES, random_state=42))

print(f"✅ Generated {len(param_grid)} parameter combinations")
print(f"   Sample params: {param_grid[0]}")

# =============================================================================
# STEP 6: Cross-Validation Grid Search
# =============================================================================

print("\n[6/8] Running grid search with cross-validation...")

# Time series CV splits (70% train, 15% val, 15% test)
n = len(df_all)
fold_size = n // (N_CV_FOLDS + 2)  # +2 for val and test

results = []

for model_idx, params in enumerate(param_grid):
    print(f"\n{'='*60}")
    print(f"Model {model_idx + 1}/{len(param_grid)}")
    print(f"Params: {params}")
    print(f"{'='*60}")
    
    fold_scores = []
    
    # Cross-validation folds
    for fold_idx in range(N_CV_FOLDS):
        # Define fold ranges (expanding window)
        train_end = (fold_idx + 1) * fold_size + fold_size
        val_start = train_end
        val_end = val_start + fold_size // 2
        
        # Split data
        X_train_fold = df_all.iloc[:train_end][feature_cols]
        y_train_fold = df_all.iloc[:train_end]['target']
        
        X_val_fold = df_all.iloc[val_start:val_end][feature_cols]
        y_val_fold = df_all.iloc[val_start:val_end]['target']
        
        # Train model
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate
        val_preds = model.predict(X_val_fold)
        
        # Metrics
        correlation = np.corrcoef(y_val_fold, val_preds)[0, 1] if len(y_val_fold) > 1 else 0
        directional = np.mean(np.sign(y_val_fold) == np.sign(val_preds))
        mse = np.mean((y_val_fold - val_preds) ** 2)
        
        fold_scores.append({
            'fold': fold_idx,
            'correlation': correlation,
            'directional_accuracy': directional,
            'mse': mse,
            'n_train': len(X_train_fold),
            'n_val': len(X_val_fold)
        })
        
        print(f"  Fold {fold_idx + 1}: Corr={correlation:.4f}, DA={directional:.4f}, MSE={mse:.6f}")
    
    # Aggregate fold results
    avg_correlation = np.mean([s['correlation'] for s in fold_scores])
    std_correlation = np.std([s['correlation'] for s in fold_scores])
    avg_directional = np.mean([s['directional_accuracy'] for s in fold_scores])
    avg_mse = np.mean([s['mse'] for s in fold_scores])
    
    results.append({
        'model_idx': model_idx,
        'params': params,
        'avg_correlation': avg_correlation,
        'std_correlation': std_correlation,
        'avg_directional_accuracy': avg_directional,
        'avg_mse': avg_mse,
        'fold_scores': fold_scores
    })
    
    print(f"  Avg: Corr={avg_correlation:.4f} (±{std_correlation:.4f}), DA={avg_directional:.4f}")

print(f"\n{'='*80}")
print("Grid search complete!")
print(f"{'='*80}")

# =============================================================================
# STEP 7: Select Best Model
# =============================================================================

print("\n[7/8] Selecting best model...")

# Sort by average correlation
results_df = pd.DataFrame([{
    'model_idx': r['model_idx'],
    'avg_correlation': r['avg_correlation'],
    'std_correlation': r['std_correlation'],
    'avg_directional_accuracy': r['avg_directional_accuracy'],
    'avg_mse': r['avg_mse'],
    **r['params']
} for r in results])

results_df = results_df.sort_values('avg_correlation', ascending=False)

print("\nTop 5 Models:")
print(results_df.head()[['model_idx', 'avg_correlation', 'std_correlation', 'avg_directional_accuracy', 'avg_mse']])

best_result = results[results_df.iloc[0]['model_idx']]
best_params = best_result['params']

print(f"\n✅ Best Model:")
print(f"   Model Index: {best_result['model_idx']}")
print(f"   Avg Correlation: {best_result['avg_correlation']:.4f} (±{best_result['std_correlation']:.4f})")
print(f"   Avg Directional Accuracy: {best_result['avg_directional_accuracy']:.4f}")
print(f"   Avg MSE: {best_result['avg_mse']:.6f}")
print(f"   Best Params: {best_params}")

# =============================================================================
# STEP 8: Retrain Best Model on Full Data
# =============================================================================

print("\n[8/8] Retraining best model on full dataset...")

# Use 85% for training (save 15% for final test)
train_end = int(len(df_all) * 0.85)

X_train_full = df_all.iloc[:train_end][feature_cols]
y_train_full = df_all.iloc[:train_end]['target']

X_test_final = df_all.iloc[train_end:][feature_cols]
y_test_final = df_all.iloc[train_end:]['target']

# Train best model
best_model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
best_model.fit(X_train_full, y_train_full)

# Final test evaluation
test_preds = best_model.predict(X_test_final)
test_correlation = np.corrcoef(y_test_final, test_preds)[0, 1] if len(y_test_final) > 1 else 0
test_directional = np.mean(np.sign(y_test_final) == np.sign(test_preds))

print(f"✅ Final test performance:")
print(f"   Correlation: {test_correlation:.4f}")
print(f"   Directional Accuracy: {test_directional:.4f}")

# Create prediction function
TICKER = TICKERS[0]
def predict() -> float:
    """Predict Bitcoin price 24 hours into the future."""
    # Get live features
    live_features = workflow.get_live_features(TICKER)
    
    # Predict log return
    predicted_log_return = best_model.predict(live_features)[0]
    
    # Get current price
    now = datetime.now(timezone.utc)
    recent_start = now - timedelta(hours=2)
    raw_data = workflow.load_raw(start=recent_start, end=now)
    current_price = raw_data["close"].iloc[-1]
    
    # Convert to predicted price
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

# Test prediction
test_prediction = predict()

# Save
with open("predict_signal_miner.pkl", "wb") as f:
    cloudpickle.dump(predict, f)

print("\n✅ Prediction function saved to predict_signal_miner.pkl")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✅ Grid search completed: {len(param_grid)} models × {N_CV_FOLDS} folds")
print(f"✅ Best model found with correlation: {best_result['avg_correlation']:.4f}")
print(f"✅ Final test correlation: {test_correlation:.4f}")
print(f"✅ Final test directional accuracy: {test_directional:.4f}")
print(f"✅ Model ready for deployment!")
print("="*80)

# Save results
results_df.to_csv("grid_search_results.csv", index=False)
print("\n📊 Full grid search results saved to: grid_search_results.csv")
print("\n🚀 Ready to deploy!")

