#!/usr/bin/env python3
"""
================================================================================
Allora Forge Builder Kit v3.0 - Topic 42 BTC/USD Price Prediction Walkthrough
================================================================================

This walkthrough demonstrates 8-hour BTC/USD price prediction using the 
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
DAYS_OF_HISTORY = 1825     # ~5 years
INTERVAL = "1h"            # 1-hour bars

# Feature Configuration
NUMBER_OF_INPUT_BARS = 48   # 2 days of hourly bars (48h lookback)
TARGET_BARS = 8             # Predict 8 bars (8 hours) ahead

# Cross-Validation Configuration
N_SPLITS = 3               # Number of CV folds
MAX_TRAIN_SIZE = 100_000_000  # Maximum training samples per fold

# Model Configuration
N_ESTIMATORS_MAX = 800
N_ESTIMATORS_CHECKPOINTS = [100, 300, 600]
LEARNING_RATES = [0.01, 0.03, 0.07, 0.1]
MAX_DEPTHS = [3, 5, 7]
NUM_LEAVES = [15, 31]
TOP_K_FEATURES_GRID = [5, 10, 25, 50]

# =============================================================================
# SCRIPT START
# =============================================================================

print("="*80)
print("Allora Forge Builder Kit v3.0 - Topic 42 Walkthrough")
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
        f.write("Allora Topic 42 Run Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Score: {best_result['score']:.1%} ({best_result['num_passed']}/7)\n")
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
api_key = get_api_key(api_key_file=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".allora_api_key"))

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
    """Add return, momentum, and volatility features (no data leakage - same row only)"""
    n = NUMBER_OF_INPUT_BARS
    closes = np.array([row[f'feature_close_{i}'] for i in range(n)])
    highs = np.array([row[f'feature_high_{i}'] for i in range(n)])
    lows = np.array([row[f'feature_low_{i}'] for i in range(n)])
    volumes = np.array([row[f'feature_volume_{i}'] for i in range(n)])
    
    log_rets = np.diff(np.log(closes + 1e-12))
    features = {}
    
    # Log returns at multiple horizons
    features['ret_1h'] = log_rets[-1] if len(log_rets) >= 1 else 0
    features['ret_4h'] = np.sum(log_rets[-4:]) if len(log_rets) >= 4 else 0
    features['ret_8h'] = np.sum(log_rets[-8:]) if len(log_rets) >= 8 else 0
    features['ret_24h'] = np.sum(log_rets[-24:]) if len(log_rets) >= 24 else 0
    features['ret_48h'] = np.sum(log_rets) if len(log_rets) >= 2 else 0
    
    # Realised volatility at multiple horizons
    features['vol_8h'] = np.std(log_rets[-8:], ddof=1) if len(log_rets) >= 8 else 0
    features['vol_24h'] = np.std(log_rets[-24:], ddof=1) if len(log_rets) >= 24 else 0
    features['vol_48h'] = np.std(log_rets, ddof=1) if len(log_rets) >= 2 else 0
    
    # Momentum: short vs long return
    features['momentum_ratio'] = features['ret_8h'] / (abs(features['ret_48h']) + 1e-12)
    
    # Mean reversion signal: distance from recent mean
    features['mean_reversion'] = (closes[-1] - np.mean(closes[-24:])) / (np.std(closes[-24:]) + 1e-12) if n >= 24 else 0
    
    # High-low range (proxy for intraday vol)
    hl_range = highs - lows
    features['hl_range_8h'] = np.mean(hl_range[-8:])
    features['hl_range_ratio'] = np.mean(hl_range[-8:]) / (np.mean(hl_range) + 1e-12)
    
    # Volume trend
    features['volume_ratio'] = np.mean(volumes[-8:]) / (np.mean(volumes) + 1e-12)
    
    # Trend strength (efficiency ratio)
    net_move = abs(np.sum(log_rets[-8:]))
    total_path = np.sum(np.abs(log_rets[-8:]))
    features['efficiency_8h'] = net_move / (total_path + 1e-12)
    
    return pd.Series(features)

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

# Track feature importance across all runs
all_feature_importances = {}

total_configs = len(LEARNING_RATES) * len(MAX_DEPTHS) * len(NUM_LEAVES) * len(TOP_K_FEATURES_GRID) * len(N_ESTIMATORS_CHECKPOINTS)
print(f"   Grid: {len(LEARNING_RATES)} lr × {len(MAX_DEPTHS)} depth × {len(NUM_LEAVES)} leaves × {len(TOP_K_FEATURES_GRID)} topK × {len(N_ESTIMATORS_CHECKPOINTS)} checkpoints = {total_configs} configs")

for lr in LEARNING_RATES:
    for depth in MAX_DEPTHS:
        for leaves in NUM_LEAVES:
            
            # Stage 1: Get feature importances ONCE per (lr, depth, leaves) combo
            # (shared across TOP_K values to save compute)
            fold_importances = []
            fold_selectors = []
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df_all)):
                X_train_full = df_all.iloc[train_idx][feature_cols]
                y_train = df_all.iloc[train_idx]['target']
                
                selector = LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=depth,
                    num_leaves=leaves,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    min_child_samples=50,
                    random_state=42,
                    verbose=-1
                )
                selector.fit(X_train_full, y_train)
                fold_importances.append(selector.feature_importances_)
                fold_selectors.append((train_idx, test_idx, y_train))
            
            # Log top features (averaged across folds)
            avg_imp = np.mean(fold_importances, axis=0)
            for feat_idx in np.argsort(avg_imp)[-10:]:
                fname = feature_cols[feat_idx]
                all_feature_importances[fname] = all_feature_importances.get(fname, 0) + avg_imp[feat_idx]
            
            # Stage 2: For each TOP_K, select features and retrain
            for top_k in TOP_K_FEATURES_GRID:
                fold_models = []
                for fold_idx, (train_idx, test_idx, y_train) in enumerate(fold_selectors):
                    importances = fold_importances[fold_idx]
                    top_idx = np.argsort(importances)[-top_k:]
                    selected = [feature_cols[i] for i in top_idx]
                    
                    X_train_sel = df_all.iloc[train_idx][selected]
                    lgb = LGBMRegressor(
                        n_estimators=N_ESTIMATORS_MAX,
                        learning_rate=lr,
                        max_depth=depth,
                        num_leaves=leaves,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        min_child_samples=50,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        random_state=42,
                        verbose=-1
                    )
                    lgb.fit(X_train_sel, y_train)
                    fold_models.append((lgb, test_idx, selected))
                
                # Evaluate at tree count checkpoints
                for n_est in N_ESTIMATORS_CHECKPOINTS:
                    config_num += 1
                    df_all['pred'] = np.nan
                    
                    for lgb, test_idx, selected in fold_models:
                        X_test = df_all.iloc[test_idx][selected]
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
                        'top_k': top_k,
                        'predictions': df_all['pred'].copy(),
                        **metrics
                    })

                    # Track calibration: std(pred) / std(target) — want ~1.0
                    y_t = df_all.loc[valid_mask, 'target'].values
                    y_p = df_all.loc[valid_mask, 'pred'].values
                    cal_ratio = np.std(y_p) / (np.std(y_t) + 1e-12)
                    from scipy.stats import pearsonr as _pr
                    r_val, _ = _pr(y_t, y_p)
                    results[-1]['cal_ratio'] = cal_ratio
                    results[-1]['pearson_r_raw'] = r_val

                    if config_num % 10 == 0 or config_num <= 3:
                        print(f"   [{config_num:3d}/{total_configs}] n={n_est:3d} lr={lr:.2f} d={depth} l={leaves:2d} k={top_k:2d} "
                              f"→ {metrics['num_passed']}/7 r={r_val:+.4f} cal={cal_ratio:.3f}")

# Analyze results — rank by AVERAGE RANK across ALL 7 core metrics + calibration
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'predictions'} for r in results])

# Extract the 7 core metrics from the nested 'metrics' dict
for r_idx, r in enumerate(results):
    m = r.get('metrics', r)
    results_df.loc[r_idx, 'da'] = m.get('directional_accuracy', 0)
    results_df.loc[r_idx, 'da_ci'] = m.get('da_ci_lower', 0)
    results_df.loc[r_idx, 'da_pval'] = m.get('da_pvalue', 1)
    results_df.loc[r_idx, 'pearson'] = m.get('pearson_r', 0)
    results_df.loc[r_idx, 'pearson_pval'] = m.get('pearson_pvalue', 1)
    results_df.loc[r_idx, 'wrmse_imp'] = m.get('wrmse_improvement', 0)
    results_df.loc[r_idx, 'czar_imp'] = m.get('czar_improvement', 0)

# Compute ranks for each metric (lower rank = better)
rank_cols = {}
rank_cols['rk_da'] = results_df['da'].rank(ascending=False)
rank_cols['rk_da_ci'] = results_df['da_ci'].rank(ascending=False)
rank_cols['rk_da_pval'] = results_df['da_pval'].rank(ascending=True)       # lower p = better
rank_cols['rk_pearson'] = results_df['pearson'].rank(ascending=False)
rank_cols['rk_pear_pval'] = results_df['pearson_pval'].rank(ascending=True) # lower p = better
rank_cols['rk_wrmse'] = results_df['wrmse_imp'].rank(ascending=False)
rank_cols['rk_czar'] = results_df['czar_imp'].rank(ascending=False)
rank_cols['rk_cal'] = (1 - results_df['cal_ratio']).abs().rank(ascending=True)  # closer to 1.0 = better

for col, vals in rank_cols.items():
    results_df[col] = vals

# Average rank of core 7 metrics (primary) and calibration (secondary)
core_rank_cols = [c for c in rank_cols if c != 'rk_cal']
results_df['core_avg_rank'] = results_df[core_rank_cols].mean(axis=1)
results_df['avg_rank'] = results_df[list(rank_cols.keys())].mean(axis=1)

# Sort: num_passed DESC (most important), then core_avg_rank ASC (tiebreaker)
results_df = results_df.sort_values(['num_passed', 'core_avg_rank'], ascending=[False, True])

print(f"\n✅ Tested {len(results)} configurations")

# Feature importance report
print(f"\n   Top 20 most important features (aggregated across all configs):")
sorted_feats = sorted(all_feature_importances.items(), key=lambda x: x[1], reverse=True)[:20]
for i, (fname, imp) in enumerate(sorted_feats):
    print(f"   {i+1:2d}. {fname:<35s} {imp:>10.1f}")

print(f"\n   Top 10 models (ranked by avg rank across 7 metrics + calibration):")
print(f"   {'#':>3} {'n':>4} {'lr':>5} {'d':>2} {'l':>3} {'k':>3} │ {'DA':>5} {'CI':>5} {'pval':>5} {'r':>6} {'WRMSE':>6} {'CZAR':>6} {'cal':>5} │ {'pts':>3} {'rk':>5}")
print(f"   {'─'*85}")
for _, row in results_df.head(10).iterrows():
    print(f"   {int(row['config_num']):3d} {int(row['n_estimators']):4d} {row['learning_rate']:5.2f} "
          f"{int(row['max_depth']):2d} {int(row['num_leaves']):3d} {int(row['top_k']):3d} │ "
          f"{row['da']:.3f} {row['da_ci']:.3f} {row['da_pval']:.3f} {row['pearson']:+.4f} "
          f"{row['wrmse_imp']:+.4f} {row['czar_imp']:+.4f} {row['cal_ratio']:.3f} │ "
          f"{int(row['num_passed']):3d} {row['avg_rank']:5.1f}")

# Select best by average rank — look up by config_num in the results list
best_cfg_num = int(results_df.iloc[0]['config_num'])
best_result = next(r for r in results if r['config_num'] == best_cfg_num)
best_params = {k: best_result[k] for k in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves']}

print(f"\n   Best (avg rank): Config #{best_result['config_num']}")
print(f"   r={results_df.iloc[0]['pearson']:+.4f} cal={results_df.iloc[0]['cal_ratio']:.3f} "
      f"DA={results_df.iloc[0]['da']:.3f} WRMSE={results_df.iloc[0]['wrmse_imp']:+.4f} "
      f"({best_result['num_passed']}/7)")

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
# STEP 6: Train Production Models (top 3 diverse configs)
# =============================================================================
print("\n[6/6] Training production models from top cohort...")

# Pick top 3 by average rank for deployment diversity
top_configs = results_df.head(3)
trained_models = []

for rank_idx, (_, row) in enumerate(top_configs.iterrows()):
    params = {
        'n_estimators': int(row['n_estimators']),
        'learning_rate': row['learning_rate'],
        'max_depth': int(row['max_depth']),
        'num_leaves': int(row['num_leaves']),
    }
    
    # Feature selection on full training data (same approach as CV)
    k = int(row['top_k'])
    selector = LGBMRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=int(row['max_depth']), num_leaves=int(row['num_leaves']),
        subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
        random_state=42, verbose=-1,
    )
    selector.fit(df_all[feature_cols], df_all['target'])
    top_idx = np.argsort(selector.feature_importances_)[-k:]
    selected = [feature_cols[i] for i in top_idx]
    
    # Train final model on selected features
    model = LGBMRegressor(
        **params,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(df_all[selected], df_all['target'])
    trained_models.append((int(row['config_num']), params, model, selected, row))
    print(f"   Model {rank_idx+1}: Config #{int(row['config_num'])} "
          f"(k={k}, r={row['pearson']:+.4f} cal={row['cal_ratio']:.3f} pts={int(row['num_passed'])})")

# Use the best (rank 1) as the primary
best_config_num, best_params, final_model, best_selected, best_row = trained_models[0]
best_result = next(r for r in results if r['config_num'] == best_config_num)
print(f"\n✅ Trained {len(trained_models)} models from top cohort")

def predict(nonce: int = None) -> float:
    """
    Predict BTC/USD price 8 hours into the future.
    
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

# Test and save ALL models from the cohort
print("\n🧪 Testing and saving models...")

for rank_idx, (cfg_num, params, model, selected, row) in enumerate(trained_models):
    # Create a predict function that captures this specific model + its selected features
    def _make_predict(m, sel):
        def predict(nonce=None):
            live_row = workflow.get_live_features(ticker=TICKERS[0])
            if live_row is None or len(live_row) == 0:
                raise ValueError("Could not get live features")
            live_returns = engineer_returns(live_row.iloc[0])
            live_features = pd.concat([live_row[base_feature_cols].iloc[0], live_returns])
            current_price = float(live_row.attrs.get("current_price", np.nan))
            if not np.isfinite(current_price) or current_price <= 0:
                snap = workflow._dm.get_live_snapshot(TICKERS)
                if snap is not None and len(snap) > 0 and "close" in snap.columns:
                    current_price = float(snap["close"].iloc[-1])
            if not np.isfinite(current_price) or current_price <= 0:
                raise ValueError(f"Invalid current price: {current_price}")
            predicted_log_return = m.predict(live_features[sel].values.reshape(1, -1))[0]
            predicted_price = current_price * np.exp(predicted_log_return)
            return float(predicted_price)
        return predict
    
    predict_fn = _make_predict(model, selected)
    pkl_name = f"predict_42_rank{rank_idx+1}.pkl"
    
    try:
        price = predict_fn()
        r_val = row.get('pearson_r_raw', row.get('pearson', 0))
        cal = row.get('cal_ratio', 0)
        pts = int(row.get('num_passed', 0))
        print(f"   Model {rank_idx+1} (#{cfg_num}): ${price:,.2f} "
              f"(r={r_val:+.4f} cal={cal:.3f} pts={pts}) → {pkl_name}")
    except Exception as e:
        print(f"   Model {rank_idx+1} (#{cfg_num}): FAILED ({e}) → {pkl_name}")
    
    with open(pkl_name, "wb") as f:
        cloudpickle.dump(predict_fn, f)

# Also save rank1 as the default predict_42.pkl
with open("predict_42.pkl", "wb") as f:
    cloudpickle.dump(_make_predict(trained_models[0][2], trained_models[0][3]), f)

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"{len(feature_cols)} features | {len(trained_models)} models saved")
print(f"Pickles: predict_42.pkl (best), predict_42_rank1/2/3.pkl (cohort)")
print("="*80)

