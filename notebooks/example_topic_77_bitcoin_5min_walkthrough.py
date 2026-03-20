#!/usr/bin/env python3
"""
================================================================================
Allora Forge Builder Kit v3.0 - Topic 77 Bitcoin 5-Minute Price Prediction Walkthrough
================================================================================

This walkthrough demonstrates 5-minute Bitcoin price prediction using the 
Allora ML Workflow Kit with base features and LightGBM.

Data is sourced from the Atlas data service (Tiingo 1-min candles).

================================================================================
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import cloudpickle
from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Data Configuration
TICKERS = ["btcusd"]
DAYS_OF_HISTORY = 120
INTERVAL = "5m"

# Feature Configuration
NUMBER_OF_INPUT_BARS = 48  # 4h of 5-minute bars for input features
TARGET_BARS = 1            # Predict 1 bar (5 minutes) ahead

# Cross-Validation Configuration
N_SPLITS = 3               # Number of CV folds
MAX_TRAIN_SIZE = 100_000_000  # Maximum training samples per fold

# Model Configuration
N_ESTIMATORS_MAX = 500    # Train with max trees, evaluate at checkpoints
N_ESTIMATORS_CHECKPOINTS = [100, 300, 500]
LEARNING_RATES = [0.01, 0.05, 0.1]
MAX_DEPTHS = [3, 5, 7]
NUM_LEAVES = [15, 31, 63]

# =============================================================================
# SCRIPT START
# =============================================================================

print("="*80)
print("Allora Forge Builder Kit v3.0 - Topic 77 Walkthrough")
print("="*80)


def _to_serializable(obj):
    """Convert numpy/pandas objects into JSON-serializable Python types."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def save_run_artifacts(df_eval, best_result, best_params, run_dir, feature_cols):
    """Persist config/metrics/predictions and basic diagnostic plots for reproducibility."""
    os.makedirs(run_dir, exist_ok=True)

    # 1) Run config
    config = {
        "tickers": TICKERS,
        "days_of_history": DAYS_OF_HISTORY,
        "interval": INTERVAL,
        "number_of_input_bars": NUMBER_OF_INPUT_BARS,
        "target_bars": TARGET_BARS,
        "n_splits": N_SPLITS,
        "max_train_size": MAX_TRAIN_SIZE,
        "n_estimators_checkpoints": N_ESTIMATORS_CHECKPOINTS,
        "learning_rates": LEARNING_RATES,
        "max_depths": MAX_DEPTHS,
        "num_leaves": NUM_LEAVES,
        "best_params": best_params,
        "feature_count": len(feature_cols),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(_to_serializable(config), f, indent=2)

    # 2) Metrics
    metrics_payload = {
        "score": best_result["score"],
        "grade": best_result["grade"],
        "num_passed": best_result["num_passed"],
        "num_primary_metrics": best_result.get("num_primary_metrics"),
        "thresholds": best_result.get("thresholds", {}),
        "passed": best_result.get("passed", {}),
        "metrics": best_result.get("metrics", {}),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(_to_serializable(metrics_payload), f, indent=2)

    # 3) Predictions table
    export_df = df_eval.copy()
    if "predictions" in best_result:
        export_df["pred"] = best_result["predictions"].values

    export_cols = ["open_time", "target", "pred"]
    export_cols = [c for c in export_cols if c in export_df.columns]
    preds_df = export_df[export_cols].dropna(subset=["pred"]).copy()
    preds_csv_path = os.path.join(run_dir, "predictions.csv")
    preds_df.to_csv(preds_csv_path, index=False)

    # 4) Scatter plot: pred vs target
    plt.figure(figsize=(8, 8))
    plt.scatter(preds_df["target"], preds_df["pred"], s=8, alpha=0.35)
    lim_min = float(min(preds_df["target"].min(), preds_df["pred"].min()))
    lim_max = float(max(preds_df["target"].max(), preds_df["pred"].max()))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1)
    plt.xlabel("Target (log return)")
    plt.ylabel("Prediction (log return)")
    plt.title("Predictions vs Target")
    plt.tight_layout()
    scatter_path = os.path.join(run_dir, "scatter_pred_vs_target.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    # 5) Human-readable report
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write("Allora Topic 77 Run Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Score: {best_result['score']:.1%} ({best_result['num_passed']}/8)\n")
        f.write(f"Grade: {best_result['grade']}\n")
        f.write(f"Best params: {best_params}\n\n")
        f.write("Primary metric pass/fail:\n")
        for metric_name, did_pass in best_result.get("passed", {}).items():
            f.write(f"- {metric_name}: {'PASS' if did_pass else 'FAIL'}\n")

    return {
        "run_dir": run_dir,
        "predictions_csv": preds_csv_path,
        "scatter_png": scatter_path,
    }

# =============================================================================
# STEP 1: Initialize Workflow
# =============================================================================
print("\n[1/6] Initializing workflow...")

# Resolve Allora API key (env var → file → prompt).
# Get a free key at https://developer.allora.network
# Alternatively, set data_source="binance" below to skip the API key entirely.
from allora_forge_builder_kit.utils import get_api_key
api_key = get_api_key(api_key_file=os.path.join(os.path.dirname(__file__), '.allora_api_key'))

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
try:
    workflow.backfill(start=start_date)
    print("✅ Backfill complete")
except Exception as e:
    print(f"⚠️ Backfill failed: {e}")
    print("   Will attempt to use locally cached parquet data...")

# =============================================================================
# STEP 3: Extract Features & Engineer New Features
# =============================================================================
print("\n[3/6] Extracting and engineering features...")

try:
    df_all = workflow.get_full_feature_target_dataframe(start_date=start_date).reset_index()
except Exception as e:
    raise RuntimeError(
        f"No data available: {e}\n\n"
        "This usually means the backfill failed (bad/missing API key) and there is "
        "no locally cached parquet data.\n\n"
        "Fix options:\n"
        "  1. Set a valid ALLORA_API_KEY (free at https://developer.allora.network)\n"
        "  2. Use data_source='binance' in AlloraMLWorkflow() to skip the API key\n"
    ) from e

# Feature Engineering: Add log returns to base features
# For detailed TA indicators and visualizations, see: feature_engineering_example.py

def engineer_returns(row):
    """Add log return features over multiple horizons (no data leakage - same row only)"""
    # NOTE: Base features are already normalized (z-scored) by the workflow
    closes = np.array([row[f'feature_close_{i}'] for i in range(NUMBER_OF_INPUT_BARS)])
    
    # Log returns over different time horizons
    returns = {}
    returns['log_return_5m'] = np.log(closes[-1] + 1e-8) - np.log(closes[-2] + 1e-8) if NUMBER_OF_INPUT_BARS >= 2 else 0
    returns['log_return_15m'] = np.log(closes[-1] + 1e-8) - np.log(closes[-4] + 1e-8) if NUMBER_OF_INPUT_BARS >= 4 else 0
    returns['log_return_30m'] = np.log(closes[-1] + 1e-8) - np.log(closes[-7] + 1e-8) if NUMBER_OF_INPUT_BARS >= 7 else 0
    returns['log_return_60m'] = np.log(closes[-1] + 1e-8) - np.log(closes[-13] + 1e-8) if NUMBER_OF_INPUT_BARS >= 13 else 0
    
    return pd.Series(returns)

# Get base features
base_feature_cols = [col for col in df_all.columns if col.startswith('feature_')]

# Apply feature engineering
print("   Engineering log return features...")
engineered_features = df_all.apply(engineer_returns, axis=1)
df_all = pd.concat([df_all, engineered_features], axis=1)

# Use base features + engineered returns
feature_cols = base_feature_cols + list(engineered_features.columns)
df_all = df_all.dropna(subset=feature_cols + ['target'])

print(f"✅ Dataset: {len(df_all):,} samples ({df_all['open_time'].min().date()} to {df_all['open_time'].max().date()})")
print(f"   Features: {len(base_feature_cols)} base + {len(engineered_features.columns)} returns = {len(feature_cols)} total")
print(f"   📚 See feature_engineering_example.py for more TA indicators")

# Setup time series cross-validation
tscv = TimeSeriesSplit(
    n_splits=N_SPLITS, 
    gap=TARGET_BARS, 
    max_train_size=MAX_TRAIN_SIZE
)

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
                
                print(f"   [{config_num:2d}] n={n_est:4d}, lr={lr:.2f}, d={depth}, l={leaves:2d} -> "
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

print(f"\nBest: Config #{best_result['config_num']}")
print(f"   {best_result['num_passed']}/8 points ({best_result['score']:.1%}) | "
      f"n={best_params['n_estimators']}, lr={best_params['learning_rate']}, d={best_params['max_depth']}, l={best_params['num_leaves']}")

# =============================================================================
# STEP 5: Evaluate Best Model
# =============================================================================
print("\n[5/6] Detailed evaluation...")
print("="*80)
evaluator.print_report(best_result, detailed=False)
print("="*80)

# Save reproducibility artifacts + diagnostic plot
run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join(os.path.dirname(__file__), "runs", run_timestamp)
artifacts = save_run_artifacts(
    df_eval=df_all,
    best_result=best_result,
    best_params=best_params,
    run_dir=run_dir,
    feature_cols=feature_cols,
)

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
    Predict Bitcoin price 5 minutes into the future.
    
    Args:
        nonce: Block nonce from Allora SDK (unused)
    
    Returns:
        float: Predicted BTC price in USD
    """
    # Get live features from workflow
    live_row = workflow.get_live_features(ticker=TICKERS[0])
    
    if live_row is None or len(live_row) == 0:
        raise ValueError("Could not get live features")
    
    # Engineer return features from live data (same as training)
    live_returns = engineer_returns(live_row.iloc[0])
    
    # Combine base features + engineered returns
    live_features = pd.concat([live_row[base_feature_cols].iloc[0], live_returns])
    
    # Get current price from live feature context (remote-only path)
    current_price = float(live_row.attrs.get("current_price", np.nan))
    if not np.isfinite(current_price) or current_price <= 0:
        # Fallback to live snapshot (still remote API; no local parquet)
        snap = workflow._dm.get_live_snapshot(TICKERS)
        if snap is not None and len(snap) > 0 and "close" in snap.columns:
            current_price = float(snap["close"].iloc[-1])

    if not np.isfinite(current_price) or current_price <= 0:
        raise ValueError(f"Invalid current price for inference: {current_price}")
    
    # Predict log return
    predicted_log_return = final_model.predict(live_features[feature_cols].values.reshape(1, -1))[0]
    
    # Convert log return to price
    predicted_price = current_price * np.exp(predicted_log_return)
    
    print(f"\nLive Prediction: ${predicted_price:,.2f} ({predicted_log_return:+.4f} log return)")
    
    return float(predicted_price)

# Test and save
print("\n🧪 Testing prediction...")
test_prediction = predict()

with open("predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"{len(feature_cols)} features | {best_result['num_passed']}/8 points ({best_result['score']:.1%})")
print("Saved to predict.pkl")
print(f"Run artifacts: {artifacts['run_dir']}")
print(f"- Predictions: {artifacts['predictions_csv']}")
print(f"- Scatter plot: {artifacts['scatter_png']}")
print("="*80)
print("\nDeploy: python deploy_worker.py (set TOPIC_ID=77)")

