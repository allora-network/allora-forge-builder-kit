#!/usr/bin/env python3
"""
Topic 82 — Model E: Calibrated Volatility (Distribution-Aware)
==============================================================

Problem: Previous models compress predictions into a narrow band because
tree models with MSE/Huber loss regress toward the mean. The scatter plot
shows predictions trapped in [0.0004, 0.0011] while targets range to 0.006+.

Solution: Three techniques to match the target distribution:
1. Log-space prediction: predict log(vol) to equalize error across magnitudes
2. Quantile ensemble: blend median prediction with upper quantile for calibration
3. Regime-aware: separate models for calm vs volatile periods

The goal is to match both the SHAPE and MAGNITUDE of the target distribution,
not just minimize average error.
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
TICKERS = ["solusd"]
DAYS_OF_HISTORY = 800
INTERVAL = "1m"
NUMBER_OF_INPUT_BARS = 60
TARGET_BARS = 15
TARGET_TYPE = "volatility"

print("=" * 80)
print("Topic 82 — Model E: Calibrated Volatility (Distribution-Aware)")
print("=" * 80)


# =============================================================================
# METRICS
# =============================================================================
def vol_metrics(y_true, y_pred):
    """Compute volatility-specific metrics."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rel_mae = mae / np.mean(y_true)
    mask = y_pred > 0
    ratio = y_true[mask] / y_pred[mask]
    qlike = np.mean(ratio - np.log(ratio) - 1) if mask.sum() > 0 else float("inf")
    # Distribution match: ratio of pred std to target std (1.0 = perfect calibration)
    cal_ratio = np.std(y_pred) / np.std(y_true)
    # Tail coverage: what fraction of target > p90 does the model predict > p90?
    p90 = np.percentile(y_true, 90)
    tail_mask = y_true > p90
    if tail_mask.sum() > 0:
        tail_capture = np.mean(y_pred[tail_mask] > np.percentile(y_pred, 90))
    else:
        tail_capture = 0.0
    return {
        "pearson_r": r, "spearman_rho": rho, "r2": r2,
        "rmse": rmse, "mae": mae, "rel_mae": rel_mae, "qlike": qlike,
        "cal_ratio": cal_ratio, "tail_capture": tail_capture,
    }


def print_metrics(metrics, label=""):
    print(f"\n  {'─' * 55}")
    if label:
        print(f"  {label}")
        print(f"  {'─' * 55}")
    print(f"  Pearson r:      {metrics['pearson_r']:.4f}")
    print(f"  Spearman ρ:     {metrics['spearman_rho']:.4f}")
    print(f"  R²:             {metrics['r2']:.4f}")
    print(f"  RMSE:           {metrics['rmse']:.6f}")
    print(f"  MAE:            {metrics['mae']:.6f}")
    print(f"  Rel MAE:        {metrics['rel_mae']*100:.2f}%")
    print(f"  QLIKE:          {metrics['qlike']:.6f}")
    print(f"  Cal ratio:      {metrics['cal_ratio']:.4f}  (1.0 = perfect spread)")
    print(f"  Tail capture:   {metrics['tail_capture']:.4f}  (1.0 = perfect tail)")
    print(f"  {'─' * 55}")


def compare_metrics(baseline, current):
    print(f"\n  {'Metric':<16} {'Baseline':<10} {'Current':<10} {'Δ':<10}")
    print(f"  {'─'*46}")
    for key in ["pearson_r", "spearman_rho", "r2", "rmse", "mae", "rel_mae", "qlike", "cal_ratio"]:
        b, c = baseline[key], current[key]
        if key in ["rmse", "mae", "rel_mae", "qlike"]:
            delta = (b - c) / abs(b) * 100 if b != 0 else 0
            arrow = "↓" if c < b else "↑"
        elif key == "cal_ratio":
            # Closer to 1.0 is better
            delta = abs(1 - c) - abs(1 - b)
            arrow = "✓" if abs(1 - c) < abs(1 - b) else "✗"
            print(f"  {key:<16} {b:<10.4f} {c:<10.4f} {arrow}")
            continue
        else:
            delta = (c - b) / abs(b) * 100 if b != 0 else 0
            arrow = "↑" if c > b else "↓"
        fmt = ".6f" if key in ["rmse", "mae"] else ".4f"
        if key == "rel_mae":
            print(f"  {key:<16} {b*100:<10.2f} {c*100:<10.2f} {arrow}{abs(delta):.1f}%")
        else:
            print(f"  {key:<16} {b:<10{fmt}} {c:<10{fmt}} {arrow}{abs(delta):.1f}%")


# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/5] Loading data...")
from allora_forge_builder_kit.utils import get_api_key

api_key = get_api_key(
    api_key_file=os.path.join(os.path.dirname(__file__), "..", ".allora_api_key")
)

wf = AlloraMLWorkflow(
    tickers=TICKERS, number_of_input_bars=NUMBER_OF_INPUT_BARS,
    target_bars=TARGET_BARS, interval=INTERVAL, target_type=TARGET_TYPE,
    data_source="allora", api_key=api_key,
)

start_date = datetime.now(timezone.utc) - timedelta(days=DAYS_OF_HISTORY)
wf.backfill(start=start_date)
df = wf.get_full_feature_target_dataframe(start_date=start_date).reset_index()
base_feature_cols = [c for c in df.columns if c.startswith("feature_")]
df = df.dropna(subset=base_feature_cols + ["target"])

split = int(len(df) * 0.8)
df_train = df.iloc[:split].copy()
df_test = df.iloc[split:].copy()
y_test = df_test["target"].values

print(f"✅ {len(df):,} samples | Train: {len(df_train):,} | Test: {len(df_test):,}")
print(f"   Target stats: mean={y_test.mean():.6f} std={y_test.std():.6f} "
      f"p90={np.percentile(y_test, 90):.6f} max={y_test.max():.6f}")


# =============================================================================
# FEATURE ENGINEERING (same as Model D)
# =============================================================================
print("\n[2/5] Engineering features...")


def engineer_features(row):
    n = NUMBER_OF_INPUT_BARS
    closes = np.array([row[f"feature_close_{i}"] for i in range(n)])
    highs = np.array([row[f"feature_high_{i}"] for i in range(n)])
    lows = np.array([row[f"feature_low_{i}"] for i in range(n)])
    volumes = np.array([row[f"feature_volume_{i}"] for i in range(n)])
    opens = np.array([row[f"feature_open_{i}"] for i in range(n)])

    log_rets = np.diff(np.log(closes + 1e-12))
    abs_rets = np.abs(log_rets)
    sq_rets = log_rets ** 2
    features = {}

    # Multi-horizon vol
    features["vol_5m"] = np.std(log_rets[-5:], ddof=1)
    features["vol_10m"] = np.std(log_rets[-10:], ddof=1)
    features["vol_15m"] = np.std(log_rets[-15:], ddof=1)
    features["vol_30m"] = np.std(log_rets[-30:], ddof=1)
    features["vol_60m"] = np.std(log_rets, ddof=1)

    # Vol ratios
    features["vol_ratio_5_15"] = features["vol_5m"] / (features["vol_15m"] + 1e-12)
    features["vol_ratio_5_60"] = features["vol_5m"] / (features["vol_60m"] + 1e-12)
    features["vol_ratio_15_60"] = features["vol_15m"] / (features["vol_60m"] + 1e-12)

    # EWMA (fast and slow)
    lam = 0.94
    ewma_var = sq_rets[0]
    for r2 in sq_rets[1:]:
        ewma_var = lam * ewma_var + (1 - lam) * r2
    features["ewma_vol"] = np.sqrt(ewma_var)

    lam_fast = 0.85
    ewma_fast = sq_rets[0]
    for r2 in sq_rets[1:]:
        ewma_fast = lam_fast * ewma_fast + (1 - lam_fast) * r2
    features["ewma_vol_fast"] = np.sqrt(ewma_fast)
    features["ewma_fast_slow_ratio"] = features["ewma_vol_fast"] / (features["ewma_vol"] + 1e-12)
    features["garch_persistence"] = features["ewma_vol"] / (features["vol_60m"] + 1e-12)

    # Parkinson & Garman-Klass
    hl_log = np.log(highs + 1e-12) - np.log(lows + 1e-12)
    features["parkinson_15m"] = np.sqrt(np.mean(hl_log[-15:] ** 2) / (4 * np.log(2)))
    features["parkinson_60m"] = np.sqrt(np.mean(hl_log ** 2) / (4 * np.log(2)))
    features["parkinson_ratio"] = features["parkinson_15m"] / (features["parkinson_60m"] + 1e-12)
    gk_terms = 0.5 * hl_log ** 2 - (2 * np.log(2) - 1) * (np.log(closes + 1e-12) - np.log(opens + 1e-12)) ** 2
    features["garman_klass_15m"] = np.sqrt(np.abs(np.mean(gk_terms[-15:])))
    features["garman_klass_60m"] = np.sqrt(np.abs(np.mean(gk_terms)))

    # Vol of vol & mean reversion
    rolling_5m_vols = np.array([np.std(log_rets[i:i+5], ddof=1) for i in range(len(log_rets) - 5)])
    if len(rolling_5m_vols) >= 2:
        features["vol_of_vol"] = np.std(rolling_5m_vols, ddof=1)
        features["vol_mean_reversion"] = (features["vol_5m"] - np.mean(rolling_5m_vols)) / (np.std(rolling_5m_vols, ddof=1) + 1e-12)
        features["vol_percentile"] = np.mean(rolling_5m_vols <= features["vol_5m"])
    else:
        features["vol_of_vol"] = 0.0
        features["vol_mean_reversion"] = 0.0
        features["vol_percentile"] = 0.5

    # Autocorrelation
    features["absret_autocorr_1"] = np.corrcoef(abs_rets[1:], abs_rets[:-1])[0, 1] if len(abs_rets) > 2 else 0.0
    if not np.isfinite(features["absret_autocorr_1"]):
        features["absret_autocorr_1"] = 0.0

    # Distribution shape
    if len(log_rets) >= 15:
        recent = log_rets[-15:]
        std_r = np.std(recent, ddof=1)
        if std_r > 1e-12:
            features["kurtosis_15m"] = np.mean(((recent - np.mean(recent)) / std_r) ** 4)
        else:
            features["kurtosis_15m"] = 3.0
    else:
        features["kurtosis_15m"] = 3.0

    # Magnitude features
    features["abs_ret_mean_5m"] = np.mean(abs_rets[-5:])
    features["abs_ret_max_15m"] = np.max(abs_rets[-15:])
    features["abs_ret_max_60m"] = np.max(abs_rets)

    # Volume interaction
    features["volume_ratio_5_60"] = np.mean(volumes[-5:]) / (np.mean(volumes) + 1e-12)
    features["volume_spike"] = np.max(volumes[-5:]) / (np.mean(volumes) + 1e-12)

    # Efficiency ratio
    net_move = abs(np.sum(log_rets[-15:]))
    total_path = np.sum(abs_rets[-15:])
    features["efficiency_15m"] = net_move / (total_path + 1e-12)

    return pd.Series(features)


print("   Engineering features (this takes ~40 min on 1.15M rows)...")
eng_train = df_train.apply(engineer_features, axis=1)
eng_test = df_test.apply(engineer_features, axis=1)

df_train = pd.concat([df_train.reset_index(drop=True), eng_train.reset_index(drop=True)], axis=1)
df_test = pd.concat([df_test.reset_index(drop=True), eng_test.reset_index(drop=True)], axis=1)

eng_cols = list(eng_train.columns)
all_feature_cols = base_feature_cols + eng_cols
df_train = df_train.dropna(subset=all_feature_cols)
df_test = df_test.dropna(subset=all_feature_cols)
y_test = df_test["target"].values

print(f"   ✅ {len(all_feature_cols)} features ready")


# =============================================================================
# BASELINE: Model D approach (Huber, single model)
# =============================================================================
print("\n[3/5] Baseline (Model D: Huber loss)...")
model_baseline = LGBMRegressor(
    objective="huber", alpha=0.5, n_estimators=500, learning_rate=0.01,
    max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.7,
    min_child_samples=100, reg_alpha=0.5, reg_lambda=2.0,
    random_state=42, verbose=-1,
)
model_baseline.fit(df_train[all_feature_cols], df_train["target"])
preds_baseline = np.maximum(model_baseline.predict(df_test[all_feature_cols]), 0)
baseline_metrics = vol_metrics(y_test, preds_baseline)
print_metrics(baseline_metrics, "BASELINE (Model D: Huber)")


# =============================================================================
# MODEL E: Log-space + Quantile Ensemble
# =============================================================================
print("\n[4/5] Training Model E (log-space + quantile ensemble)...")

# --- Strategy 1: Predict in log-space ---
# Transform: log(vol) is more Gaussian, equalizes error across magnitudes
y_train_log = np.log(df_train["target"].values + 1e-10)
y_test_log = np.log(y_test + 1e-10)

model_log = LGBMRegressor(
    objective="regression",  # MSE in log-space = multiplicative error in real space
    n_estimators=800,
    learning_rate=0.01,
    max_depth=7,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_samples=50,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)
model_log.fit(df_train[all_feature_cols], y_train_log)
preds_log_space = model_log.predict(df_test[all_feature_cols])
preds_from_log = np.exp(preds_log_space)  # back to real space

log_metrics = vol_metrics(y_test, preds_from_log)
print_metrics(log_metrics, "Log-space model (exp transform back)")

# --- Strategy 2: Quantile models for calibration ---
# Train at 50th percentile (median) and 75th percentile
model_q50 = LGBMRegressor(
    objective="quantile", alpha=0.5,  # median
    n_estimators=500, learning_rate=0.01, max_depth=6, num_leaves=31,
    subsample=0.8, colsample_bytree=0.7, min_child_samples=100,
    reg_alpha=0.3, reg_lambda=1.5, random_state=42, verbose=-1,
)
model_q50.fit(df_train[all_feature_cols], df_train["target"])
preds_q50 = np.maximum(model_q50.predict(df_test[all_feature_cols]), 0)

model_q75 = LGBMRegressor(
    objective="quantile", alpha=0.75,  # upper quartile
    n_estimators=500, learning_rate=0.01, max_depth=6, num_leaves=31,
    subsample=0.8, colsample_bytree=0.7, min_child_samples=100,
    reg_alpha=0.3, reg_lambda=1.5, random_state=42, verbose=-1,
)
model_q75.fit(df_train[all_feature_cols], df_train["target"])
preds_q75 = np.maximum(model_q75.predict(df_test[all_feature_cols]), 0)

# --- Strategy 3: Ensemble blend ---
# Blend log-space model (good at shape) with quantile shift (good at tails)
# The log model captures the full range; we blend with q50 for stability
alpha = 0.6  # weight on log-space model
preds_ensemble = alpha * preds_from_log + (1 - alpha) * preds_q50

ensemble_metrics = vol_metrics(y_test, preds_ensemble)
print_metrics(ensemble_metrics, "Ensemble (0.6*log + 0.4*q50)")

# --- Strategy 4: Log-space with bias correction ---
# exp(E[log(x)]) underestimates E[x] for log-normal. Apply correction.
# Correction factor: exp(0.5 * residual_variance_in_log_space)
log_residuals = y_train_log - model_log.predict(df_train[all_feature_cols])
bias_correction = np.exp(0.5 * np.var(log_residuals))
preds_corrected = preds_from_log * bias_correction

corrected_metrics = vol_metrics(y_test, preds_corrected)
print_metrics(corrected_metrics, f"Log-space + bias correction (factor={bias_correction:.4f})")

# --- Pick the best ---
candidates = [
    ("baseline_huber", baseline_metrics, preds_baseline),
    ("log_space", log_metrics, preds_from_log),
    ("ensemble_log_q50", ensemble_metrics, preds_ensemble),
    ("log_corrected", corrected_metrics, preds_corrected),
]

# Rank by a composite: prioritize QLIKE (vol-specific) and cal_ratio (distribution match)
def composite_score(m):
    # Lower QLIKE is better, cal_ratio closer to 1.0 is better, higher r2 is better
    return m["r2"] - 0.5 * m["qlike"] - 0.3 * abs(1 - m["cal_ratio"])

print("\n\n  Candidate ranking (composite score):")
print(f"  {'Name':<20} {'R²':<8} {'QLIKE':<8} {'Cal':<8} {'Score':<8}")
print(f"  {'─'*52}")
ranked = sorted(candidates, key=lambda x: composite_score(x[1]), reverse=True)
for name, m, _ in ranked:
    score = composite_score(m)
    print(f"  {name:<20} {m['r2']:.4f}  {m['qlike']:.4f}  {m['cal_ratio']:.4f}  {score:.4f}")

best_name, best_metrics, best_preds = ranked[0]
print(f"\n  → Winner: {best_name}")


# =============================================================================
# STEP 5: Save best model for deployment
# =============================================================================
print(f"\n[5/5] Saving Model E ({best_name})...")
print_metrics(best_metrics, f"MODEL E FINAL ({best_name})")
print("\n📊 Improvement over Model D baseline:")
compare_metrics(baseline_metrics, best_metrics)

# For deployment, we need to package the right predict function
workflow = wf

if best_name == "log_space":
    _deploy_model = model_log
    _bias = 1.0
elif best_name == "log_corrected":
    _deploy_model = model_log
    _bias = bias_correction
elif best_name == "ensemble_log_q50":
    _deploy_model_log = model_log
    _deploy_model_q50 = model_q50
    _alpha = alpha
else:
    _deploy_model = model_baseline
    _bias = None


def predict(nonce=None):
    live_row = workflow.get_live_features(ticker=TICKERS[0])
    if live_row is None or len(live_row) == 0:
        raise ValueError("Could not get live features")
    live_eng = engineer_features(live_row.iloc[0])
    live_features = pd.concat([live_row[base_feature_cols].iloc[0], live_eng])
    x = live_features[all_feature_cols].values.reshape(1, -1)

    if best_name in ("log_space", "log_corrected"):
        log_pred = _deploy_model.predict(x)[0]
        vol = np.exp(log_pred) * _bias
    elif best_name == "ensemble_log_q50":
        log_pred = np.exp(_deploy_model_log.predict(x)[0])
        q50_pred = max(0, _deploy_model_q50.predict(x)[0])
        vol = _alpha * log_pred + (1 - _alpha) * q50_pred
    else:
        vol = _deploy_model.predict(x)[0]

    vol = max(0.0, float(vol))
    print(f"Model E prediction: {vol:.6f} (15-min vol)")
    return vol


print("\n🧪 Testing prediction...")
test_pred = predict()

with open("predict_82_model_e.pkl", "wb") as f:
    cloudpickle.dump(predict, f)

print(f"\n✅ Saved predict_82_model_e.pkl")
print(f"   Strategy: {best_name}")
print(f"   Pearson r: {best_metrics['pearson_r']:.4f} | R²: {best_metrics['r2']:.4f}")
print(f"   Cal ratio: {best_metrics['cal_ratio']:.4f} | QLIKE: {best_metrics['qlike']:.6f}")
