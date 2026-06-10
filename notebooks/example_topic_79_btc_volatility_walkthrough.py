#!/usr/bin/env python3
"""
================================================================================
Allora Forge Builder Kit v3.0 - Topic 79 BTC/USD 15-Minute Volatility Prediction
================================================================================

This walkthrough demonstrates 15-minute realised volatility prediction for
BTC/USD using the Allora ML Workflow Kit with base features and LightGBM.

Target definition:
    The standard deviation of consecutive 1-minute log returns over the next
    15 minutes.  Formally, for each timestamp t:

        r_i = log(close[t+i] / close[t+i-1])   for i in 1..15
        target[t] = std(r_1, r_2, ..., r_15)

    This matches the ground-truth definition used by the Allora volatility
    reputer (allora-reputer-volatility-prediction).

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
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import cloudpickle
from allora_forge_builder_kit import AlloraMLWorkflow

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Data Configuration
TICKERS = ["btcusd"]
DAYS_OF_HISTORY = 60
INTERVAL = "1m"  # 1-minute base interval for volatility

# Feature Configuration
NUMBER_OF_INPUT_BARS = 15  # 15 minutes of 1-minute bars for input features
TARGET_BARS = 15           # 15-minute volatility horizon

# Target type: volatility (std of 1-min log returns over the horizon)
TARGET_TYPE = "volatility"

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

print("=" * 80)
print("Allora Forge Builder Kit v3.0 - Topic 79 Walkthrough")
print("BTC/USD 15-Minute Volatility Prediction")
print("=" * 80)


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


# =============================================================================
# VOLATILITY-SPECIFIC METRICS
# =============================================================================
def vol_metrics(y_true, y_pred):
    """
    Compute volatility-specific evaluation metrics.

    These replace the standard log-return metrics (DA, CZAR) which are not
    meaningful for volatility prediction.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rel_mae = mae / np.mean(y_true)
    # QLIKE: quasi-likelihood loss (standard for volatility forecasting)
    mask = y_pred > 0
    if mask.sum() > 0:
        ratio = y_true[mask] / y_pred[mask]
        qlike = np.mean(ratio - np.log(ratio) - 1)
    else:
        qlike = float("inf")
    return {
        "pearson_r": r,
        "spearman_rho": rho,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "rel_mae": rel_mae,
        "qlike": qlike,
    }


def print_vol_metrics(metrics, label=""):
    """Pretty-print volatility metrics."""
    print(f"\n  {'─' * 50}")
    if label:
        print(f"  {label}")
        print(f"  {'─' * 50}")
    print(f"  Pearson r:   {metrics['pearson_r']:.4f}")
    print(f"  Spearman ρ:  {metrics['spearman_rho']:.4f}")
    print(f"  R²:          {metrics['r2']:.4f}")
    print(f"  RMSE:        {metrics['rmse']:.6f}")
    print(f"  MAE:         {metrics['mae']:.6f}")
    print(f"  Rel MAE:     {metrics['rel_mae']*100:.2f}%")
    print(f"  QLIKE:       {metrics['qlike']:.6f}")
    print(f"  {'─' * 50}")


def save_run_artifacts(df_eval, best_result, best_params, run_dir, feature_cols):
    """Persist config/metrics/predictions and basic diagnostic plots."""
    os.makedirs(run_dir, exist_ok=True)

    # 1) Run config
    config = {
        "topic_id": 79,
        "target_type": TARGET_TYPE,
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
    metrics_payload = {k: v for k, v in best_result.items() if k != "predictions"}
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
    plt.xlabel("Target (realised volatility)")
    plt.ylabel("Prediction (realised volatility)")
    plt.title("Predictions vs Target — 15-min BTC Volatility")
    plt.tight_layout()
    scatter_path = os.path.join(run_dir, "scatter_pred_vs_target.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    # 5) Human-readable report
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write("Allora Topic 79 Run Report\n")
        f.write("BTC/USD 15-Minute Volatility Prediction\n")
        f.write("=" * 40 + "\n")
        f.write(f"Best params: {best_params}\n\n")
        f.write("Volatility Metrics:\n")
        for key in ["pearson_r", "spearman_rho", "r2", "rmse", "mae", "rel_mae", "qlike"]:
            if key in best_result:
                f.write(f"  {key}: {best_result[key]:.6f}\n")

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

api_key = get_api_key(
    api_key_file=os.path.join(os.path.dirname(__file__), ".allora_api_key")
)

workflow = AlloraMLWorkflow(
    tickers=TICKERS,
    number_of_input_bars=NUMBER_OF_INPUT_BARS,
    target_bars=TARGET_BARS,
    interval=INTERVAL,
    target_type=TARGET_TYPE,
    data_source="allora",
    api_key=api_key,
)

print("✅ Workflow initialized")
print(f"   Assets: {TICKERS} | Interval: {INTERVAL}")
print(f"   Input: {NUMBER_OF_INPUT_BARS} bars → Features: {NUMBER_OF_INPUT_BARS * 5}")
print(f"   Target: {TARGET_TYPE} over {TARGET_BARS}-minute horizon")

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
    df_all = workflow.get_full_feature_target_dataframe(
        start_date=start_date
    ).reset_index()
except Exception as e:
    raise RuntimeError(
        f"No data available: {e}\n\n"
        "This usually means the backfill failed (bad/missing API key) and there is "
        "no locally cached parquet data.\n\n"
        "Fix options:\n"
        "  1. Set a valid ALLORA_API_KEY (free at https://developer.allora.network)\n"
        "  2. Use data_source='binance' in AlloraMLWorkflow() to skip the API key\n"
    ) from e


# Feature Engineering: Add volatility-relevant features from the lookback window
def engineer_vol_features(row):
    """Engineer volatility-predictive features (no data leakage — same row only)."""
    closes = np.array(
        [row[f"feature_close_{i}"] for i in range(NUMBER_OF_INPUT_BARS)]
    )
    highs = np.array(
        [row[f"feature_high_{i}"] for i in range(NUMBER_OF_INPUT_BARS)]
    )
    lows = np.array(
        [row[f"feature_low_{i}"] for i in range(NUMBER_OF_INPUT_BARS)]
    )

    features = {}

    # Realised volatility of the lookback window (std of 1-min log returns)
    log_returns = np.diff(np.log(closes + 1e-12))
    features["hist_vol_full"] = np.std(log_returns, ddof=1) if len(log_returns) > 1 else 0.0

    # Short-term vs long-term vol ratio (regime detection)
    if len(log_returns) >= 5:
        features["hist_vol_5m"] = np.std(log_returns[-5:], ddof=1)
        features["vol_ratio_5_full"] = (
            features["hist_vol_5m"] / (features["hist_vol_full"] + 1e-12)
        )
    else:
        features["hist_vol_5m"] = features["hist_vol_full"]
        features["vol_ratio_5_full"] = 1.0

    # High-low range (Parkinson-style proxy)
    hl_range = highs - lows
    features["hl_range_mean"] = np.mean(hl_range)
    features["hl_range_recent"] = np.mean(hl_range[-3:]) if len(hl_range) >= 3 else hl_range[-1]
    features["hl_range_ratio"] = (
        features["hl_range_recent"] / (features["hl_range_mean"] + 1e-12)
    )

    # Absolute return (magnitude of recent move)
    features["abs_return_1m"] = abs(log_returns[-1]) if len(log_returns) > 0 else 0.0
    features["abs_return_5m"] = abs(np.log(closes[-1] + 1e-12) - np.log(closes[-5] + 1e-12)) if len(closes) >= 5 else 0.0

    return pd.Series(features)


# Get base features
base_feature_cols = [col for col in df_all.columns if col.startswith("feature_")]

# Apply feature engineering
print("   Engineering volatility-predictive features...")
engineered_features = df_all.apply(engineer_vol_features, axis=1)
df_all = pd.concat([df_all, engineered_features], axis=1)

# Use base features + engineered volatility features
feature_cols = base_feature_cols + list(engineered_features.columns)
df_all = df_all.dropna(subset=feature_cols + ["target"])

print(
    f"✅ Dataset: {len(df_all):,} samples "
    f"({df_all['open_time'].min().date()} to {df_all['open_time'].max().date()})"
)
print(
    f"   Features: {len(base_feature_cols)} base + "
    f"{len(engineered_features.columns)} vol = {len(feature_cols)} total"
)

# Setup time series cross-validation
tscv = TimeSeriesSplit(
    n_splits=N_SPLITS,
    gap=TARGET_BARS,
    max_train_size=MAX_TRAIN_SIZE,
)

print(f"✅ Walk-forward CV: {N_SPLITS} splits, {TARGET_BARS}-bar embargo")
for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df_all)):
    print(f"   Fold {fold_idx + 1}: Train={len(train_idx):,}, Test={len(test_idx):,}")

# =============================================================================
# STEP 4: Grid Search with Walk-Forward Cross-Validation
# =============================================================================
print("\n[4/6] Running grid search...")

results = []
config_num = 0

for lr in LEARNING_RATES:
    for depth in MAX_DEPTHS:
        for leaves in NUM_LEAVES:

            # Train once with max trees, evaluate at checkpoints
            fold_models = []
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df_all)):
                X_train = df_all.iloc[train_idx][feature_cols]
                y_train = df_all.iloc[train_idx]["target"]

                lgb = LGBMRegressor(
                    n_estimators=N_ESTIMATORS_MAX,
                    learning_rate=lr,
                    max_depth=depth,
                    num_leaves=leaves,
                    random_state=42,
                    verbose=-1,
                )
                lgb.fit(X_train, y_train)
                fold_models.append((lgb, test_idx))

            # Evaluate at tree count checkpoints
            for n_est in N_ESTIMATORS_CHECKPOINTS:
                config_num += 1
                df_all["pred"] = np.nan

                # Generate predictions using first n_est trees
                for lgb, test_idx in fold_models:
                    X_test = df_all.iloc[test_idx][feature_cols]
                    preds = lgb.predict(X_test, num_iteration=n_est)
                    df_all.iloc[test_idx, df_all.columns.get_loc("pred")] = preds

                # Evaluate with volatility-specific metrics
                valid_mask = ~df_all["pred"].isna()
                y_true_cv = df_all.loc[valid_mask, "target"].values
                y_pred_cv = np.maximum(df_all.loc[valid_mask, "pred"].values, 0)
                metrics = vol_metrics(y_true_cv, y_pred_cv)

                # Store results
                results.append(
                    {
                        "config_num": config_num,
                        "n_estimators": n_est,
                        "learning_rate": lr,
                        "max_depth": depth,
                        "num_leaves": leaves,
                        "predictions": df_all["pred"].copy(),
                        **metrics,
                    }
                )

                print(
                    f"   [{config_num:2d}] n={n_est:4d}, lr={lr:.2f}, "
                    f"d={depth}, l={leaves:2d} -> "
                    f"r={metrics['pearson_r']:.4f} R²={metrics['r2']:.4f} "
                    f"QLIKE={metrics['qlike']:.4f}"
                )

# Analyze results — rank by R² (primary), then QLIKE (secondary, lower=better)
results_df = pd.DataFrame(
    [{k: v for k, v in r.items() if k != "predictions"} for r in results]
)
results_df = results_df.sort_values(["r2", "qlike"], ascending=[False, True])

print(f"\n✅ Tested {len(results)} configurations")
print("\n   Top 5 models:")
top5_cols = [
    "config_num",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "num_leaves",
    "pearson_r",
    "r2",
    "qlike",
]
print(results_df[top5_cols].head().to_string(index=False, float_format="%.4f"))

# Select best model
best_result = results[results_df.iloc[0]["config_num"] - 1]
best_params = {
    k: best_result[k]
    for k in ["n_estimators", "learning_rate", "max_depth", "num_leaves"]
}

print(f"\nBest: Config #{best_result['config_num']}")
print(
    f"   r={best_result['pearson_r']:.4f} R²={best_result['r2']:.4f} "
    f"QLIKE={best_result['qlike']:.4f} | "
    f"n={best_params['n_estimators']}, lr={best_params['learning_rate']}, "
    f"d={best_params['max_depth']}, l={best_params['num_leaves']}"
)

# =============================================================================
# STEP 5: Evaluate Best Model
# =============================================================================
print("\n[5/6] Detailed evaluation...")
print_vol_metrics(best_result, "BEST MODEL — Volatility Metrics")

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
# STEP 6: Train Production Model & Create Predict Function
# =============================================================================
print("\n[6/6] Training production model...")

final_model = LGBMRegressor(
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    num_leaves=best_params["num_leaves"],
    random_state=42,
    verbose=-1,
)
final_model.fit(df_all[feature_cols], df_all["target"])
print(f"✅ Final model trained on {len(df_all):,} samples")


def predict(nonce: int = None) -> float:
    """
    Predict BTC/USD 15-minute realised volatility.

    This is the function submitted to the Allora network for Topic 79.
    It returns the predicted standard deviation of 1-minute log returns
    over the next 15 minutes.

    Args:
        nonce: Block nonce from Allora SDK (unused).

    Returns:
        float: Predicted 15-minute realised volatility.
    """
    # Get live features from workflow (1-minute bars)
    live_row = workflow.get_live_features(ticker=TICKERS[0])

    if live_row is None or len(live_row) == 0:
        raise ValueError("Could not get live features")

    # Engineer volatility features from live data (same as training)
    live_vol_features = engineer_vol_features(live_row.iloc[0])

    # Combine base features + engineered vol features
    live_features = pd.concat([live_row[base_feature_cols].iloc[0], live_vol_features])

    # Predict volatility directly (no price conversion needed)
    predicted_volatility = final_model.predict(
        live_features[feature_cols].values.reshape(1, -1)
    )[0]

    # Volatility must be non-negative
    predicted_volatility = max(0.0, float(predicted_volatility))

    print(f"\nLive Prediction: {predicted_volatility:.6f} (15-min realised vol)")

    return predicted_volatility


# Test and save
print("\n🧪 Testing prediction...")
test_prediction = predict()

with open("predict.pkl", "wb") as f:
    cloudpickle.dump(predict, f)

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(
    f"{len(feature_cols)} features | "
    f"r={best_result['pearson_r']:.4f} | R²={best_result['r2']:.4f} | "
    f"QLIKE={best_result['qlike']:.4f}"
)
print(f"\nTo deploy this worker:")
print(f"  TOPIC_ID=79 python notebooks/deploy_worker_raw.py")
