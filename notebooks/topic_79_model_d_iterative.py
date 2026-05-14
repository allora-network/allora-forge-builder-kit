#!/usr/bin/env python3
"""
Topic 79 — Model D: Iterative Improvement
==========================================

Starting from Model A's baseline (best performer), iteratively adding
features and tuning to push volatility metrics higher.

Baseline (Model A, 60-bar raw features only):
    Pearson r:   0.695
    Spearman ρ:  0.714
    R²:          0.457
    RMSE:        0.000290
    MAE:         0.000194
    Rel MAE:     34.2%
    QLIKE:       0.100
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from scipy.stats import pearsonr, spearmanr
from lightgbm import LGBMRegressor
import cloudpickle
from allora_forge_builder_kit import AlloraMLWorkflow

# =============================================================================
# CONFIGURATION
# =============================================================================
TICKERS = ["btcusd"]
DAYS_OF_HISTORY = 800
INTERVAL = "1m"
NUMBER_OF_INPUT_BARS = 60
TARGET_BARS = 15
TARGET_TYPE = "volatility"

print("=" * 80)
print("Topic 79 — Model D: Iterative Improvement")
print("=" * 80)


# =============================================================================
# METRICS
# =============================================================================
def vol_metrics(y_true, y_pred):
    """Compute volatility-specific metrics."""
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rel_mae = mae / np.mean(y_true)
    # QLIKE (quasi-likelihood — standard vol forecast loss)
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


def print_metrics(metrics, label=""):
    """Pretty-print volatility metrics."""
    print(f"\n  {'─'*50}")
    if label:
        print(f"  {label}")
        print(f"  {'─'*50}")
    print(f"  Pearson r:   {metrics['pearson_r']:.4f}")
    print(f"  Spearman ρ:  {metrics['spearman_rho']:.4f}")
    print(f"  R²:          {metrics['r2']:.4f}")
    print(f"  RMSE:        {metrics['rmse']:.6f}")
    print(f"  MAE:         {metrics['mae']:.6f}")
    print(f"  Rel MAE:     {metrics['rel_mae']*100:.2f}%")
    print(f"  QLIKE:       {metrics['qlike']:.6f}")
    print(f"  {'─'*50}")


def compare_metrics(baseline, current):
    """Show improvement over baseline."""
    print(f"\n  {'Metric':<14} {'Baseline':<10} {'Current':<10} {'Δ':<10}")
    print(f"  {'─'*44}")
    for key in ["pearson_r", "spearman_rho", "r2", "rmse", "mae", "rel_mae", "qlike"]:
        b, c = baseline[key], current[key]
        if key in ["rmse", "mae", "rel_mae", "qlike"]:
            # Lower is better
            delta = (b - c) / b * 100
            arrow = "↓" if c < b else "↑"
        else:
            # Higher is better
            delta = (c - b) / abs(b) * 100 if b != 0 else 0
            arrow = "↑" if c > b else "↓"
        fmt = ".6f" if key in ["rmse", "mae"] else ".4f"
        if key == "rel_mae":
            print(f"  {key:<14} {b*100:<10.2f} {c*100:<10.2f} {arrow}{abs(delta):.1f}%")
        else:
            print(f"  {key:<14} {b:<10{fmt}} {c:<10{fmt}} {arrow}{abs(delta):.1f}%")


# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/4] Loading data...")
from allora_forge_builder_kit.utils import get_api_key

api_key = get_api_key(
    api_key_file=os.path.join(os.path.dirname(__file__), ".allora_api_key")
)

wf = AlloraMLWorkflow(
    tickers=TICKERS,
    number_of_input_bars=NUMBER_OF_INPUT_BARS,
    target_bars=TARGET_BARS,
    interval=INTERVAL,
    target_type=TARGET_TYPE,
    data_source="allora",
    api_key=api_key,
)

start_date = datetime.now(timezone.utc) - timedelta(days=DAYS_OF_HISTORY)
wf.backfill(start=start_date)
df = wf.get_full_feature_target_dataframe(start_date=start_date).reset_index()
base_feature_cols = [c for c in df.columns if c.startswith("feature_")]
df = df.dropna(subset=base_feature_cols + ["target"])

# 80/20 temporal split
split = int(len(df) * 0.8)
df_train = df.iloc[:split].copy()
df_test = df.iloc[split:].copy()
y_test = df_test["target"].values

print(f"✅ {len(df):,} samples | Train: {len(df_train):,} | Test: {len(df_test):,}")
print(f"   Mean vol: {y_test.mean():.6f} | Std vol: {y_test.std():.6f}")

# =============================================================================
# BASELINE: Model A (raw 60-bar features only)
# =============================================================================
print("\n[2/4] Baseline (Model A: 60 raw bars, no engineering)...")
model_baseline = LGBMRegressor(
    n_estimators=200, learning_rate=0.01, max_depth=5, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbose=-1,
)
model_baseline.fit(df_train[base_feature_cols], df_train["target"])
preds_baseline = np.maximum(model_baseline.predict(df_test[base_feature_cols]), 0)
baseline_metrics = vol_metrics(y_test, preds_baseline)
print_metrics(baseline_metrics, "BASELINE (Model A)")


# =============================================================================
# MODEL D: Feature Engineering
# =============================================================================
print("\n[3/4] Engineering Model D features...")


def engineer_model_d_features(row):
    """
    Model D features: combine best of all previous models + new ideas.

    Strategy:
    - Multi-horizon realised vol (from Model A)
    - EWMA vol / vol persistence (from Model B)
    - Parkinson & Garman-Klass estimators (from Model A)
    - NEW: GARCH-inspired features (conditional vol)
    - NEW: Vol regime quantiles
    - NEW: Microstructure features (bid-ask proxy from HL spread)
    - NEW: Return distribution shape (skewness, kurtosis)
    """
    n = NUMBER_OF_INPUT_BARS
    closes = np.array([row[f"feature_close_{i}"] for i in range(n)])
    highs = np.array([row[f"feature_high_{i}"] for i in range(n)])
    lows = np.array([row[f"feature_low_{i}"] for i in range(n)])
    volumes = np.array([row[f"feature_volume_{i}"] for i in range(n)])

    log_rets = np.diff(np.log(closes + 1e-12))
    abs_rets = np.abs(log_rets)
    sq_rets = log_rets ** 2
    features = {}

    # === Multi-horizon realised vol ===
    features["vol_5m"] = np.std(log_rets[-5:], ddof=1)
    features["vol_10m"] = np.std(log_rets[-10:], ddof=1)
    features["vol_15m"] = np.std(log_rets[-15:], ddof=1)
    features["vol_30m"] = np.std(log_rets[-30:], ddof=1)
    features["vol_60m"] = np.std(log_rets, ddof=1)

    # === Vol ratios (regime) ===
    features["vol_ratio_5_15"] = features["vol_5m"] / (features["vol_15m"] + 1e-12)
    features["vol_ratio_5_60"] = features["vol_5m"] / (features["vol_60m"] + 1e-12)
    features["vol_ratio_15_60"] = features["vol_15m"] / (features["vol_60m"] + 1e-12)

    # === GARCH(1,1)-inspired features ===
    # Exponentially weighted variance (RiskMetrics lambda=0.94)
    lam = 0.94
    ewma_var = sq_rets[0]
    for r2 in sq_rets[1:]:
        ewma_var = lam * ewma_var + (1 - lam) * r2
    features["ewma_vol"] = np.sqrt(ewma_var)

    # GARCH persistence: how much does yesterday's vol predict today's?
    # Approximate with ratio of EWMA to realised
    features["garch_persistence"] = features["ewma_vol"] / (features["vol_60m"] + 1e-12)

    # Conditional vol: EWMA computed at different lambdas
    lam_fast = 0.85  # faster decay — more reactive
    ewma_fast = sq_rets[0]
    for r2 in sq_rets[1:]:
        ewma_fast = lam_fast * ewma_fast + (1 - lam_fast) * r2
    features["ewma_vol_fast"] = np.sqrt(ewma_fast)
    features["ewma_fast_slow_ratio"] = features["ewma_vol_fast"] / (features["ewma_vol"] + 1e-12)

    # === Parkinson volatility (high-low based) ===
    hl_log = np.log(highs + 1e-12) - np.log(lows + 1e-12)
    features["parkinson_15m"] = np.sqrt(np.mean(hl_log[-15:] ** 2) / (4 * np.log(2)))
    features["parkinson_60m"] = np.sqrt(np.mean(hl_log ** 2) / (4 * np.log(2)))
    features["parkinson_ratio"] = features["parkinson_15m"] / (features["parkinson_60m"] + 1e-12)

    # === Garman-Klass volatility (uses OHLC) ===
    opens = np.array([row[f"feature_open_{i}"] for i in range(n)])
    gk_terms = 0.5 * hl_log ** 2 - (2 * np.log(2) - 1) * (np.log(closes + 1e-12) - np.log(opens + 1e-12)) ** 2
    features["garman_klass_15m"] = np.sqrt(np.mean(gk_terms[-15:]))
    features["garman_klass_60m"] = np.sqrt(np.mean(gk_terms))

    # === Vol of vol (second-order) ===
    rolling_5m_vols = np.array([
        np.std(log_rets[i:i+5], ddof=1) for i in range(len(log_rets) - 5)
    ])
    if len(rolling_5m_vols) >= 2:
        features["vol_of_vol"] = np.std(rolling_5m_vols, ddof=1)
        features["vol_mean_reversion"] = (features["vol_5m"] - np.mean(rolling_5m_vols)) / (np.std(rolling_5m_vols, ddof=1) + 1e-12)
    else:
        features["vol_of_vol"] = 0.0
        features["vol_mean_reversion"] = 0.0

    # === Vol quantile (where are we in the local distribution?) ===
    if len(rolling_5m_vols) > 0:
        features["vol_percentile"] = np.mean(rolling_5m_vols <= features["vol_5m"])
    else:
        features["vol_percentile"] = 0.5

    # === Return autocorrelation (clustering signal) ===
    features["ret_autocorr_1"] = np.corrcoef(log_rets[1:], log_rets[:-1])[0, 1] if len(log_rets) > 2 else 0.0
    features["absret_autocorr_1"] = np.corrcoef(abs_rets[1:], abs_rets[:-1])[0, 1] if len(abs_rets) > 2 else 0.0
    # Fix NaN
    for k in ["ret_autocorr_1", "absret_autocorr_1"]:
        if not np.isfinite(features[k]):
            features[k] = 0.0

    # === Return distribution shape ===
    if len(log_rets) >= 15:
        recent = log_rets[-15:]
        mean_r = np.mean(recent)
        std_r = np.std(recent, ddof=1)
        if std_r > 1e-12:
            features["skewness_15m"] = np.mean(((recent - mean_r) / std_r) ** 3)
            features["kurtosis_15m"] = np.mean(((recent - mean_r) / std_r) ** 4)
        else:
            features["skewness_15m"] = 0.0
            features["kurtosis_15m"] = 3.0
    else:
        features["skewness_15m"] = 0.0
        features["kurtosis_15m"] = 3.0

    # === Absolute returns (magnitude features) ===
    features["abs_ret_mean_5m"] = np.mean(abs_rets[-5:])
    features["abs_ret_mean_15m"] = np.mean(abs_rets[-15:])
    features["abs_ret_max_15m"] = np.max(abs_rets[-15:])
    features["abs_ret_max_60m"] = np.max(abs_rets)

    # === Volume-volatility interaction ===
    features["volume_ratio_5_60"] = np.mean(volumes[-5:]) / (np.mean(volumes) + 1e-12)
    features["volume_spike"] = np.max(volumes[-5:]) / (np.mean(volumes) + 1e-12)
    # Volume-weighted absolute return
    vol_weights = volumes[1:] / (np.sum(volumes[1:]) + 1e-12)
    features["vwap_absret"] = np.sum(abs_rets * vol_weights)

    # === Trend vs chop (efficiency ratio) ===
    net_move = abs(np.sum(log_rets[-15:]))
    total_path = np.sum(abs_rets[-15:])
    features["efficiency_15m"] = net_move / (total_path + 1e-12)

    # === Recent extreme moves ===
    features["max_abs_ret_5m"] = np.max(abs_rets[-5:])
    features["max_abs_ret_ratio"] = features["max_abs_ret_5m"] / (features["vol_5m"] + 1e-12)

    return pd.Series(features)


print("   Engineering features...")
eng_train = df_train.apply(engineer_model_d_features, axis=1)
eng_test = df_test.apply(engineer_model_d_features, axis=1)

df_train = pd.concat([df_train, eng_train], axis=1)
df_test = pd.concat([df_test, eng_test], axis=1)

eng_cols = list(eng_train.columns)
all_feature_cols = base_feature_cols + eng_cols

# Drop any rows with NaN in engineered features
df_train = df_train.dropna(subset=all_feature_cols)
df_test = df_test.dropna(subset=all_feature_cols)
y_test = df_test["target"].values

print(f"   Features: {len(base_feature_cols)} base + {len(eng_cols)} engineered = {len(all_feature_cols)} total")

# =============================================================================
# MODEL D: Train with Huber loss
# =============================================================================
print("\n[4/4] Training Model D (Huber loss, heavy regularization)...")

model_d = LGBMRegressor(
    objective="huber",          # Robust to vol spikes
    alpha=0.5,                  # Huber delta (transition point)
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_samples=100,      # Conservative splits
    reg_alpha=0.5,              # L1
    reg_lambda=2.0,             # L2
    random_state=42,
    verbose=-1,
)
model_d.fit(df_train[all_feature_cols], df_train["target"])
preds_d = np.maximum(model_d.predict(df_test[all_feature_cols]), 0)

model_d_metrics = vol_metrics(y_test, preds_d)
print_metrics(model_d_metrics, "MODEL D (Huber + GARCH features)")

print("\n📊 Improvement over baseline:")
compare_metrics(baseline_metrics, model_d_metrics)

# =============================================================================
# SAVE
# =============================================================================
print("\n\nSaving Model D...")

# For deployment, we need the workflow and feature engineering in the predict fn
workflow = wf

def predict(nonce=None):
    live_row = workflow.get_live_features(ticker=TICKERS[0])
    if live_row is None or len(live_row) == 0:
        raise ValueError("Could not get live features")
    live_eng = engineer_model_d_features(live_row.iloc[0])
    live_features = pd.concat([live_row[base_feature_cols].iloc[0], live_eng])
    vol = model_d.predict(live_features[all_feature_cols].values.reshape(1, -1))[0]
    vol = max(0.0, float(vol))
    print(f"Model D prediction: {vol:.6f} (15-min vol)")
    return vol

print("🧪 Testing prediction...")
test_pred = predict()

with open("predict_79_model_d.pkl", "wb") as f:
    cloudpickle.dump(predict, f)

print(f"\n✅ Saved predict_79_model_d.pkl")
print(f"   Pearson r: {model_d_metrics['pearson_r']:.4f} | R²: {model_d_metrics['r2']:.4f} | QLIKE: {model_d_metrics['qlike']:.6f}")
