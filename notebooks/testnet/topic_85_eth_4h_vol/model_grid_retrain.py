#!/usr/bin/env python3
"""
Topic 85 — ETH/USD 4h Volatility — Grid Search Retrain
========================================================

Grid search over objectives × LightGBM hyperparams.
Saves top 5 diverse models for deployment.

Now with corrected √T scaling in the target (matching the reputer).
"""

import numpy as np
import pandas as pd
import os
import math
from datetime import datetime, timedelta, timezone
from scipy.stats import pearsonr, spearmanr
from lightgbm import LGBMRegressor
import cloudpickle
from allora_forge_builder_kit import AlloraMLWorkflow
from allora_forge_builder_kit.utils import get_api_key

# =============================================================================
# CONFIG
# =============================================================================
TICKERS = ["ethusd"]
TOPIC_ID = 85
DAYS_OF_HISTORY = 365
INTERVAL = "1m"
NUMBER_OF_INPUT_BARS = 240  # 4 hours at 1-min resolution (= TARGET_BARS)
TARGET_BARS = 240
TARGET_TYPE = "volatility"

# Grid
OBJECTIVES = ["regression", "huber"]
LOG_SPACE = [False, True]  # train in log-space?
LEARNING_RATES = [0.005, 0.01, 0.03]
MAX_DEPTHS = [5, 7]
NUM_LEAVES = [31, 63]
N_ESTIMATORS_MAX = 800
N_ESTIMATORS_CHECKPOINTS = [200, 500, 800]

TOP_K_DEPLOY = 5

print("=" * 70)
print(f"Topic {TOPIC_ID} — {TICKERS[0].upper()} 4h Volatility — Grid Retrain")
print(f"Target now includes √{TARGET_BARS} scaling")
print("=" * 70)

# =============================================================================
# METRICS
# =============================================================================
def vol_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mask = y_pred > 0
    ratio = y_true[mask] / y_pred[mask]
    qlike = np.mean(ratio - np.log(ratio) - 1) if mask.sum() > 0 else float("inf")
    cal_ratio = np.std(y_pred) / np.std(y_true)
    return {"pearson_r": r, "spearman_rho": rho, "r2": r2, "rmse": rmse, "qlike": qlike, "cal_ratio": cal_ratio}

def composite_score(m):
    return m["r2"] - 0.5 * m["qlike"] - 0.3 * abs(1 - m["cal_ratio"])

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/4] Loading data...")
api_key = get_api_key(
    api_key_file=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".allora_api_key")
)
os.environ["ALLORA_API_KEY"] = api_key

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

# Sanity check the scaling
target_mean = df["target"].mean()
print(f"  Target mean: {target_mean:.6f} (should be ~0.01-0.04 for ETH 4h vol with √240 scaling)")

split = int(len(df) * 0.8)
df_train = df.iloc[:split - TARGET_BARS].copy()
df_test = df.iloc[split:].copy()
y_test = df_test["target"].values
print(f"  {len(df):,} samples | Train: {len(df_train):,} | Test: {len(df_test):,}")

# =============================================================================
# FEATURE ENGINEERING (vectorized — ~900× faster than row-by-row df.apply)
# =============================================================================
print("\n[2/4] Engineering features (vectorized)...")

def engineer_features_vectorized(df_in, n_input_bars):
    """Vectorized feature engineering for volatility topics.
    Operates on the full DataFrame at once using 2D numpy arrays."""
    n = n_input_bars
    close_cols = [f"feature_close_{i}" for i in range(n)]
    high_cols = [f"feature_high_{i}" for i in range(n)]
    low_cols = [f"feature_low_{i}" for i in range(n)]
    vol_cols = [f"feature_volume_{i}" for i in range(n)]

    closes = df_in[close_cols].values  # (N, n)
    highs = df_in[high_cols].values
    lows = df_in[low_cols].values
    volumes = df_in[vol_cols].values

    log_closes = np.log(closes + 1e-12)
    log_rets = np.diff(log_closes, axis=1)  # (N, n-1)
    abs_rets = np.abs(log_rets)
    sq_rets = log_rets ** 2

    out = {}

    # Multi-horizon vol (std along axis=1 for last K columns)
    out["vol_5m"] = np.std(log_rets[:, -5:], axis=1, ddof=1)
    out["vol_10m"] = np.std(log_rets[:, -10:], axis=1, ddof=1)
    out["vol_15m"] = np.std(log_rets[:, -15:], axis=1, ddof=1)
    out["vol_30m"] = np.std(log_rets[:, -30:], axis=1, ddof=1)
    out["vol_60m"] = np.std(log_rets, axis=1, ddof=1)

    out["vol_ratio_5_15"] = out["vol_5m"] / (out["vol_15m"] + 1e-12)
    out["vol_ratio_5_60"] = out["vol_5m"] / (out["vol_60m"] + 1e-12)
    out["vol_ratio_15_60"] = out["vol_15m"] / (out["vol_60m"] + 1e-12)

    # EWMA (vectorized across rows using cumulative approach)
    lam = 0.94
    ewma = np.zeros(len(df_in))
    ewma[:] = sq_rets[:, 0]
    for t in range(1, sq_rets.shape[1]):
        ewma = lam * ewma + (1 - lam) * sq_rets[:, t]
    out["ewma_vol"] = np.sqrt(ewma)

    lam_fast = 0.85
    ewma_fast = np.zeros(len(df_in))
    ewma_fast[:] = sq_rets[:, 0]
    for t in range(1, sq_rets.shape[1]):
        ewma_fast = lam_fast * ewma_fast + (1 - lam_fast) * sq_rets[:, t]
    out["ewma_vol_fast"] = np.sqrt(ewma_fast)
    out["ewma_fast_slow_ratio"] = out["ewma_vol_fast"] / (out["ewma_vol"] + 1e-12)

    # Parkinson
    hl_log = np.log(highs + 1e-12) - np.log(lows + 1e-12)
    out["parkinson_15m"] = np.sqrt(np.mean(hl_log[:, -15:] ** 2, axis=1) / (4 * np.log(2)))
    out["parkinson_60m"] = np.sqrt(np.mean(hl_log ** 2, axis=1) / (4 * np.log(2)))

    # Vol of vol (vectorized rolling std using stride tricks)
    # Approximate: use std of a few horizon stds
    out["vol_of_vol"] = np.std(
        np.column_stack([out["vol_5m"], out["vol_10m"], out["vol_15m"], out["vol_30m"]]),
        axis=1, ddof=1
    )
    # Vol percentile: fraction of rolling vols <= current vol_5m
    out["vol_percentile"] = (
        (out["vol_5m"] >= out["vol_10m"]).astype(float) +
        (out["vol_5m"] >= out["vol_15m"]).astype(float) +
        (out["vol_5m"] >= out["vol_30m"]).astype(float) +
        (out["vol_5m"] >= out["vol_60m"]).astype(float)
    ) / 4.0

    # Autocorrelation of abs returns (vectorized dot product)
    ar1 = abs_rets[:, 1:]
    ar0 = abs_rets[:, :-1]
    ar1_mean = ar1.mean(axis=1, keepdims=True)
    ar0_mean = ar0.mean(axis=1, keepdims=True)
    num = np.sum((ar1 - ar1_mean) * (ar0 - ar0_mean), axis=1)
    den = np.sqrt(np.sum((ar1 - ar1_mean)**2, axis=1) * np.sum((ar0 - ar0_mean)**2, axis=1))
    out["absret_autocorr_1"] = np.where(den > 1e-12, num / den, 0.0)

    # Magnitude features
    out["abs_ret_mean_5m"] = np.mean(abs_rets[:, -5:], axis=1)
    out["abs_ret_max_15m"] = np.max(abs_rets[:, -15:], axis=1)
    out["volume_ratio_5_60"] = np.mean(volumes[:, -5:], axis=1) / (np.mean(volumes, axis=1) + 1e-12)

    return pd.DataFrame(out, index=df_in.index)

# Row-level version for live inference (single row)
def engineer_features(row):
    """Single-row feature engineering for live prediction."""
    n = NUMBER_OF_INPUT_BARS
    closes = np.array([row[f"feature_close_{i}"] for i in range(n)])
    highs = np.array([row[f"feature_high_{i}"] for i in range(n)])
    lows = np.array([row[f"feature_low_{i}"] for i in range(n)])
    volumes = np.array([row[f"feature_volume_{i}"] for i in range(n)])
    log_rets = np.diff(np.log(closes + 1e-12))
    abs_rets = np.abs(log_rets)
    sq_rets = log_rets ** 2
    f = {}
    f["vol_5m"] = np.std(log_rets[-5:], ddof=1)
    f["vol_10m"] = np.std(log_rets[-10:], ddof=1)
    f["vol_15m"] = np.std(log_rets[-15:], ddof=1)
    f["vol_30m"] = np.std(log_rets[-30:], ddof=1)
    f["vol_60m"] = np.std(log_rets, ddof=1)
    f["vol_ratio_5_15"] = f["vol_5m"] / (f["vol_15m"] + 1e-12)
    f["vol_ratio_5_60"] = f["vol_5m"] / (f["vol_60m"] + 1e-12)
    f["vol_ratio_15_60"] = f["vol_15m"] / (f["vol_60m"] + 1e-12)
    lam = 0.94; ewma = sq_rets[0]
    for r2 in sq_rets[1:]: ewma = lam * ewma + (1 - lam) * r2
    f["ewma_vol"] = np.sqrt(ewma)
    lam_fast = 0.85; ef = sq_rets[0]
    for r2 in sq_rets[1:]: ef = lam_fast * ef + (1 - lam_fast) * r2
    f["ewma_vol_fast"] = np.sqrt(ef)
    f["ewma_fast_slow_ratio"] = f["ewma_vol_fast"] / (f["ewma_vol"] + 1e-12)
    hl_log = np.log(highs + 1e-12) - np.log(lows + 1e-12)
    f["parkinson_15m"] = np.sqrt(np.mean(hl_log[-15:] ** 2) / (4 * np.log(2)))
    f["parkinson_60m"] = np.sqrt(np.mean(hl_log ** 2) / (4 * np.log(2)))
    vols = [f["vol_5m"], f["vol_10m"], f["vol_15m"], f["vol_30m"]]
    f["vol_of_vol"] = np.std(vols, ddof=1)
    f["vol_percentile"] = sum(1 for v in [f["vol_10m"], f["vol_15m"], f["vol_30m"], f["vol_60m"]] if f["vol_5m"] >= v) / 4.0
    if len(abs_rets) > 2:
        c = np.corrcoef(abs_rets[1:], abs_rets[:-1])[0, 1]
        f["absret_autocorr_1"] = c if np.isfinite(c) else 0.0
    else:
        f["absret_autocorr_1"] = 0.0
    f["abs_ret_mean_5m"] = np.mean(abs_rets[-5:])
    f["abs_ret_max_15m"] = np.max(abs_rets[-15:])
    f["volume_ratio_5_60"] = np.mean(volumes[-5:]) / (np.mean(volumes) + 1e-12)
    return pd.Series(f)

import time
t0 = time.time()
eng = engineer_features_vectorized(df, NUMBER_OF_INPUT_BARS)
print(f"   Vectorized features: {time.time() - t0:.1f}s for {len(df):,} rows")

eng_cols = list(eng.columns)
df = pd.concat([df.reset_index(drop=True), eng.reset_index(drop=True)], axis=1)
all_feature_cols = base_feature_cols + eng_cols
df = df.dropna(subset=all_feature_cols)

# Re-split: train / val (model selection) / test (held out for final evaluation)
val_split = int(len(df) * 0.70)
test_split = int(len(df) * 0.80)
df_train = df.iloc[:val_split - TARGET_BARS].copy()
df_val   = df.iloc[val_split:test_split].copy()
df_test  = df.iloc[test_split:].copy()
y_val  = df_val["target"].values
y_test = df_test["target"].values
print(f"   {len(all_feature_cols)} features ready | Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# =============================================================================
# GRID SEARCH
# =============================================================================
print("\n[3/4] Grid search...")

n_configs = len(OBJECTIVES) * len(LOG_SPACE) * len(LEARNING_RATES) * len(MAX_DEPTHS) * len(NUM_LEAVES)
total = n_configs * len(N_ESTIMATORS_CHECKPOINTS)
print(f"   {len(OBJECTIVES)} obj × {len(LOG_SPACE)} log × {len(LEARNING_RATES)} lr × "
      f"{len(MAX_DEPTHS)} d × {len(NUM_LEAVES)} l = {n_configs} models × "
      f"{len(N_ESTIMATORS_CHECKPOINTS)} ckpts = {total} evals")

results = []
model_num = 0

for obj in OBJECTIVES:
    for log_space in LOG_SPACE:
        for lr in LEARNING_RATES:
            for depth in MAX_DEPTHS:
                for leaves in NUM_LEAVES:
                    model_num += 1

                    y_train = df_train["target"].values
                    if log_space:
                        y_train = np.log(y_train + 1e-10)

                    model = LGBMRegressor(
                        objective=obj, n_estimators=N_ESTIMATORS_MAX,
                        learning_rate=lr, max_depth=depth, num_leaves=leaves,
                        subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
                        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
                    )
                    if obj == "huber":
                        model.set_params(alpha=0.5)
                    model.fit(df_train[all_feature_cols], y_train)

                    for n_est in N_ESTIMATORS_CHECKPOINTS:
                        raw_preds = model.predict(df_val[all_feature_cols], num_iteration=n_est)
                        if log_space:
                            preds = np.exp(raw_preds)
                        else:
                            preds = raw_preds
                        preds = np.maximum(preds, 0)

                        m = vol_metrics(y_val, preds)
                        score = composite_score(m)
                        results.append({
                            "model_num": model_num, "obj": obj, "log_space": log_space,
                            "lr": lr, "depth": depth, "leaves": leaves, "n_est": n_est,
                            **m, "score": score,
                        })

                    if model_num % 10 == 0 or model_num <= 2:
                        best_r = max(r["score"] for r in results[-len(N_ESTIMATORS_CHECKPOINTS):])
                        print(f"   [{model_num:3d}/{n_configs}] obj={obj:<10} log={str(log_space):<5} "
                              f"lr={lr:.3f} d={depth} l={leaves:2d} score={best_r:+.4f}")

# =============================================================================
# RANK & DEPLOY TOP 5
# =============================================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("score", ascending=False)

print(f"\n  Top 10:")
print(f"   {'#':>3} {'obj':<10} {'log':>4} {'lr':>5} {'d':>2} {'l':>3} {'n':>4} │ {'r':>6} {'R²':>6} {'QLIKE':>7} {'cal':>5} {'score':>7}")
print(f"   {'─'*70}")
for _, row in results_df.head(10).iterrows():
    print(f"   {int(row['model_num']):3d} {row['obj']:<10} {str(row['log_space']):>4} "
          f"{row['lr']:.3f} {int(row['depth']):2d} {int(row['leaves']):3d} {int(row['n_est']):4d} │ "
          f"{row['pearson_r']:.4f} {row['r2']:.4f} {row['qlike']:.5f} {row['cal_ratio']:.3f} {row['score']:+.4f}")

print(f"\n[4/4] Training & saving top {TOP_K_DEPLOY}...")

# Retrain top 5 on all data and save
top_k = results_df.head(TOP_K_DEPLOY)
for rank, (_, row) in enumerate(top_k.iterrows()):
    y_all = df["target"].values
    log_space = row["log_space"]
    if log_space:
        y_all = np.log(y_all + 1e-10)

    model = LGBMRegressor(
        objective=row["obj"], n_estimators=int(row["n_est"]),
        learning_rate=row["lr"], max_depth=int(row["depth"]),
        num_leaves=int(row["leaves"]),
        subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
    )
    if row["obj"] == "huber":
        model.set_params(alpha=0.5)
    model.fit(df[all_feature_cols], y_all)

    if log_space:
        residuals = y_all - model.predict(df[all_feature_cols])
        _bias_corr = float(np.exp(0.5 * np.var(residuals)))
    else:
        _bias_corr = 1.0

    def _make_predict(m, _log=log_space, _tickers=TICKERS,
                      _n_input=NUMBER_OF_INPUT_BARS, _target_bars=TARGET_BARS,
                      _interval=INTERVAL, _target_type=TARGET_TYPE,
                      _base_cols=base_feature_cols, _all_cols=all_feature_cols,
                      _eng_fn=engineer_features, _bias_correction=_bias_corr):
        _model_str = m.booster_.model_to_string()
        _is_log = _log
        def predict(nonce=None):
            import os
            import lightgbm as lgb
            import numpy as np
            from allora_forge_builder_kit import AlloraMLWorkflow
            _wf = AlloraMLWorkflow(
                tickers=_tickers, number_of_input_bars=_n_input,
                target_bars=_target_bars, interval=_interval,
                target_type=_target_type, data_source="allora",
                api_key=os.environ["ALLORA_API_KEY"],
            )
            booster = lgb.Booster(model_str=_model_str)
            live_row = _wf.get_live_features(ticker=_tickers[0])
            if live_row is None or len(live_row) == 0:
                raise ValueError("No live features")
            live_eng = _eng_fn(live_row.iloc[0])
            live_features = pd.concat([live_row[_base_cols].iloc[0], live_eng])
            x = live_features[_all_cols].values.reshape(1, -1)
            raw = booster.predict(x)[0]
            if _is_log:
                vol = float(np.exp(raw) * _bias_correction)
            else:
                vol = float(raw)
            return max(0.0, vol)
        return predict

    fn = _make_predict(model)
    pkl = f"predict_{TOPIC_ID}_grid_rank{rank+1}.pkl"
    try:
        val = fn()
        print(f"   Rank {rank+1} (#{int(row['model_num'])}): obj={row['obj']} log={row['log_space']} "
              f"score={row['score']:+.4f} → {val:.6f} → {pkl}")
    except Exception as e:
        print(f"   Rank {rank+1}: FAILED ({e}) → {pkl}")
    with open(pkl, "wb") as f:
        cloudpickle.dump(fn, f)

# =============================================================================
# SCATTER PLOT: Best model predictions vs true values
# =============================================================================
print("\n  Generating scatter plot...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

best_row = top_k.iloc[0]
log_space = best_row["log_space"]
y_train_final = df_train["target"].values
if log_space:
    y_train_final = np.log(y_train_final + 1e-10)

best_model = LGBMRegressor(
    objective=best_row["obj"], n_estimators=int(best_row["n_est"]),
    learning_rate=best_row["lr"], max_depth=int(best_row["depth"]),
    num_leaves=int(best_row["leaves"]),
    subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
)
if best_row["obj"] == "huber":
    best_model.set_params(alpha=0.5)
best_model.fit(df_train[all_feature_cols], y_train_final)

raw_preds = best_model.predict(df_test[all_feature_cols])
if log_space:
    scatter_preds = np.exp(raw_preds)
else:
    scatter_preds = raw_preds
scatter_preds = np.maximum(scatter_preds, 0)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, scatter_preds, alpha=0.05, s=4, c="#4A90D9", edgecolors="none")
lims = [0, max(y_test.max(), scatter_preds.max()) * 1.05]
ax.plot(lims, lims, "r--", lw=1.5, alpha=0.7, label="Perfect prediction")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("True Volatility (√T-scaled)", fontsize=13)
ax.set_ylabel("Predicted Volatility", fontsize=13)
ax.set_title(f"Topic {TOPIC_ID} — {TICKERS[0].upper()} 4h Vol\n"
             f"Rank 1: r={best_row['pearson_r']:.4f}, R²={best_row['r2']:.4f}, "
             f"cal={best_row['cal_ratio']:.3f}", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
scatter_path = f"scatter_{TOPIC_ID}_vol.png"
plt.savefig(scatter_path, dpi=150)
plt.close()
print(f"   Saved {scatter_path}")

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
