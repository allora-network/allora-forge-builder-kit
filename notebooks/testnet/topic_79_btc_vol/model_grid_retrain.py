#!/usr/bin/env python3
"""
Topic 79 — BTC/USD 15m Volatility — Grid Search Retrain
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
TICKERS = ["btcusd"]
TOPIC_ID = 79
DAYS_OF_HISTORY = 800
INTERVAL = "1m"
NUMBER_OF_INPUT_BARS = 60
TARGET_BARS = 15
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
print(f"Topic {TOPIC_ID} — {TICKERS[0].upper()} 15m Volatility — Grid Retrain")
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
    api_key_file=os.path.join(os.path.dirname(__file__), "..", "..", ".allora_api_key")
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

# Sanity check the scaling
target_mean = df["target"].mean()
print(f"  Target mean: {target_mean:.6f} (should be ~0.002-0.008 for BTC 15m vol with √15 scaling)")

split = int(len(df) * 0.8)
df_train = df.iloc[:split].copy()
df_test = df.iloc[split:].copy()
y_test = df_test["target"].values
print(f"  {len(df):,} samples | Train: {len(df_train):,} | Test: {len(df_test):,}")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
print("\n[2/4] Engineering features...")

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
    f = {}

    f["vol_5m"] = np.std(log_rets[-5:], ddof=1)
    f["vol_10m"] = np.std(log_rets[-10:], ddof=1)
    f["vol_15m"] = np.std(log_rets[-15:], ddof=1)
    f["vol_30m"] = np.std(log_rets[-30:], ddof=1)
    f["vol_60m"] = np.std(log_rets, ddof=1)
    f["vol_ratio_5_15"] = f["vol_5m"] / (f["vol_15m"] + 1e-12)
    f["vol_ratio_5_60"] = f["vol_5m"] / (f["vol_60m"] + 1e-12)
    f["vol_ratio_15_60"] = f["vol_15m"] / (f["vol_60m"] + 1e-12)

    lam = 0.94
    ewma_var = sq_rets[0]
    for r2 in sq_rets[1:]:
        ewma_var = lam * ewma_var + (1 - lam) * r2
    f["ewma_vol"] = np.sqrt(ewma_var)

    lam_fast = 0.85
    ewma_fast = sq_rets[0]
    for r2 in sq_rets[1:]:
        ewma_fast = lam_fast * ewma_fast + (1 - lam_fast) * r2
    f["ewma_vol_fast"] = np.sqrt(ewma_fast)
    f["ewma_fast_slow_ratio"] = f["ewma_vol_fast"] / (f["ewma_vol"] + 1e-12)

    hl_log = np.log(highs + 1e-12) - np.log(lows + 1e-12)
    f["parkinson_15m"] = np.sqrt(np.mean(hl_log[-15:] ** 2) / (4 * np.log(2)))
    f["parkinson_60m"] = np.sqrt(np.mean(hl_log ** 2) / (4 * np.log(2)))

    rolling_5m_vols = np.array([np.std(log_rets[i:i+5], ddof=1) for i in range(len(log_rets) - 5)])
    if len(rolling_5m_vols) >= 2:
        f["vol_of_vol"] = np.std(rolling_5m_vols, ddof=1)
        f["vol_percentile"] = np.mean(rolling_5m_vols <= f["vol_5m"])
    else:
        f["vol_of_vol"] = 0.0
        f["vol_percentile"] = 0.5

    f["absret_autocorr_1"] = np.corrcoef(abs_rets[1:], abs_rets[:-1])[0, 1] if len(abs_rets) > 2 else 0.0
    if not np.isfinite(f["absret_autocorr_1"]):
        f["absret_autocorr_1"] = 0.0

    f["abs_ret_mean_5m"] = np.mean(abs_rets[-5:])
    f["abs_ret_max_15m"] = np.max(abs_rets[-15:])
    f["volume_ratio_5_60"] = np.mean(volumes[-5:]) / (np.mean(volumes) + 1e-12)

    return pd.Series(f)

print("   Engineering features...")
eng_train = df_train.apply(engineer_features, axis=1)
eng_test = df_test.apply(engineer_features, axis=1)

df_train = pd.concat([df_train.reset_index(drop=True), eng_train.reset_index(drop=True)], axis=1)
df_test = pd.concat([df_test.reset_index(drop=True), eng_test.reset_index(drop=True)], axis=1)

eng_cols = list(eng_train.columns)
all_feature_cols = base_feature_cols + eng_cols
df_train = df_train.dropna(subset=all_feature_cols)
df_test = df_test.dropna(subset=all_feature_cols)
y_test = df_test["target"].values
print(f"   {len(all_feature_cols)} features ready")

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
                        raw_preds = model.predict(df_test[all_feature_cols], num_iteration=n_est)
                        if log_space:
                            preds = np.exp(raw_preds)
                        else:
                            preds = raw_preds
                        preds = np.maximum(preds, 0)

                        m = vol_metrics(y_test, preds)
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

    def _make_predict(m, _log=log_space, _wf=wf, _tickers=TICKERS,
                      _base_cols=base_feature_cols, _all_cols=all_feature_cols,
                      _eng_fn=engineer_features):
        _model_str = m.booster_.model_to_string()
        _is_log = _log
        def predict(nonce=None):
            import lightgbm as lgb
            import numpy as np
            booster = lgb.Booster(model_str=_model_str)
            live_row = _wf.get_live_features(ticker=_tickers[0])
            if live_row is None or len(live_row) == 0:
                raise ValueError("No live features")
            live_eng = _eng_fn(live_row.iloc[0])
            live_features = pd.concat([live_row[_base_cols].iloc[0], live_eng])
            x = live_features[_all_cols].values.reshape(1, -1)
            raw = booster.predict(x)[0]
            if _is_log:
                vol = float(np.exp(raw))
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

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
