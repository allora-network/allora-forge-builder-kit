#!/usr/bin/env python3
"""
Topic 85 — ETH/USD 4h Volatility — Importance-Based Feature Groups
====================================================================

Diverse model search using feature importance to create nested feature groups.
Each CV fold computes its own importance ranking (no lookahead).

Feature groups: top-5, top-10, top-20, top-50, top-100, all features.
Grid: feature_group × lr × depth × n_estimators.
Deploy top model from each feature group for genuine ensemble diversity.
"""

import numpy as np
import pandas as pd
import os
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import TimeSeriesSplit
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

N_SPLITS = 3
FEATURE_GROUP_SIZES = [5, 10, 20, 50, 100, None]  # None = all features

# LightGBM grid (compact — diversity comes from feature groups, not hyperparams)
LEARNING_RATES = [0.001, 0.01, 0.1]
MAX_DEPTHS = [5, 7]
N_ESTIMATORS_LIST = [25, 50, 100, 150]

TOP_K_PER_GROUP = 1  # best model per feature group → 6 diverse models

print("=" * 70)
print(f"Topic {TOPIC_ID} — {TICKERS[0].upper()} 4h Vol — Importance-Based Groups")
print("=" * 70)

# =============================================================================
# METRICS
# =============================================================================
def vol_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mask = y_pred > 0
    ratio = y_true[mask] / y_pred[mask]
    qlike = np.mean(ratio - np.log(ratio) - 1) if mask.sum() > 0 else float("inf")
    cal_ratio = np.std(y_pred) / np.std(y_true)
    return {"pearson_r": r, "spearman_rho": rho, "r2": r2, "qlike": qlike, "cal_ratio": cal_ratio}

def composite_score(m):
    return m["r2"] - 0.5 * m["qlike"] - 0.3 * abs(1 - m["cal_ratio"])

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/5] Loading data...")
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
print(f"  {len(df):,} samples")

# =============================================================================
# VECTORIZED FEATURE ENGINEERING
# =============================================================================
print("\n[2/5] Engineering features (vectorized)...")

def engineer_features_vectorized(df_in, n_input_bars):
    n = n_input_bars
    close_cols = [f"feature_close_{i}" for i in range(n)]
    high_cols = [f"feature_high_{i}" for i in range(n)]
    low_cols = [f"feature_low_{i}" for i in range(n)]
    vol_cols = [f"feature_volume_{i}" for i in range(n)]

    closes = df_in[close_cols].values
    highs = df_in[high_cols].values
    lows = df_in[low_cols].values
    volumes = df_in[vol_cols].values

    log_closes = np.log(closes + 1e-12)
    log_rets = np.diff(log_closes, axis=1)
    abs_rets = np.abs(log_rets)
    sq_rets = log_rets ** 2

    out = {}
    out["vol_5m"] = np.std(log_rets[:, -5:], axis=1, ddof=1)
    out["vol_10m"] = np.std(log_rets[:, -10:], axis=1, ddof=1)
    out["vol_15m"] = np.std(log_rets[:, -15:], axis=1, ddof=1)
    out["vol_30m"] = np.std(log_rets[:, -30:], axis=1, ddof=1)
    out["vol_60m"] = np.std(log_rets, axis=1, ddof=1)
    out["vol_ratio_5_15"] = out["vol_5m"] / (out["vol_15m"] + 1e-12)
    out["vol_ratio_5_60"] = out["vol_5m"] / (out["vol_60m"] + 1e-12)
    out["vol_ratio_15_60"] = out["vol_15m"] / (out["vol_60m"] + 1e-12)

    lam = 0.94
    ewma = np.zeros(len(df_in)); ewma[:] = sq_rets[:, 0]
    for t in range(1, sq_rets.shape[1]):
        ewma = lam * ewma + (1 - lam) * sq_rets[:, t]
    out["ewma_vol"] = np.sqrt(ewma)

    lam_fast = 0.85
    ewma_fast = np.zeros(len(df_in)); ewma_fast[:] = sq_rets[:, 0]
    for t in range(1, sq_rets.shape[1]):
        ewma_fast = lam_fast * ewma_fast + (1 - lam_fast) * sq_rets[:, t]
    out["ewma_vol_fast"] = np.sqrt(ewma_fast)
    out["ewma_fast_slow_ratio"] = out["ewma_vol_fast"] / (out["ewma_vol"] + 1e-12)

    hl_log = np.log(highs + 1e-12) - np.log(lows + 1e-12)
    out["parkinson_15m"] = np.sqrt(np.mean(hl_log[:, -15:] ** 2, axis=1) / (4 * np.log(2)))
    out["parkinson_60m"] = np.sqrt(np.mean(hl_log ** 2, axis=1) / (4 * np.log(2)))

    out["vol_of_vol"] = np.std(
        np.column_stack([out["vol_5m"], out["vol_10m"], out["vol_15m"], out["vol_30m"]]),
        axis=1, ddof=1)
    out["vol_percentile"] = (
        (out["vol_5m"] >= out["vol_10m"]).astype(float) +
        (out["vol_5m"] >= out["vol_15m"]).astype(float) +
        (out["vol_5m"] >= out["vol_30m"]).astype(float) +
        (out["vol_5m"] >= out["vol_60m"]).astype(float)) / 4.0

    ar1 = abs_rets[:, 1:]; ar0 = abs_rets[:, :-1]
    ar1m = ar1.mean(axis=1, keepdims=True); ar0m = ar0.mean(axis=1, keepdims=True)
    num = np.sum((ar1 - ar1m) * (ar0 - ar0m), axis=1)
    den = np.sqrt(np.sum((ar1 - ar1m)**2, axis=1) * np.sum((ar0 - ar0m)**2, axis=1))
    out["absret_autocorr_1"] = np.where(den > 1e-12, num / den, 0.0)

    out["abs_ret_mean_5m"] = np.mean(abs_rets[:, -5:], axis=1)
    out["abs_ret_max_15m"] = np.max(abs_rets[:, -15:], axis=1)
    out["volume_ratio_5_60"] = np.mean(volumes[:, -5:], axis=1) / (np.mean(volumes, axis=1) + 1e-12)

    return pd.DataFrame(out, index=df_in.index)

# Row-level for live inference
def engineer_features(row):
    n = NUMBER_OF_INPUT_BARS
    closes = np.array([row[f"feature_close_{i}"] for i in range(n)])
    highs = np.array([row[f"feature_high_{i}"] for i in range(n)])
    lows = np.array([row[f"feature_low_{i}"] for i in range(n)])
    volumes = np.array([row[f"feature_volume_{i}"] for i in range(n)])
    log_rets = np.diff(np.log(closes + 1e-12))
    abs_rets = np.abs(log_rets); sq_rets = log_rets ** 2
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
    else: f["absret_autocorr_1"] = 0.0
    f["abs_ret_mean_5m"] = np.mean(abs_rets[-5:])
    f["abs_ret_max_15m"] = np.max(abs_rets[-15:])
    f["volume_ratio_5_60"] = np.mean(volumes[-5:]) / (np.mean(volumes) + 1e-12)
    return pd.Series(f)

t0 = _time.time()
eng = engineer_features_vectorized(df, NUMBER_OF_INPUT_BARS)
print(f"  Vectorized: {_time.time() - t0:.1f}s for {len(df):,} rows")

eng_cols = list(eng.columns)
df = pd.concat([df.reset_index(drop=True), eng.reset_index(drop=True)], axis=1)
all_feature_cols = base_feature_cols + eng_cols
df = df.dropna(subset=all_feature_cols)
print(f"  {len(all_feature_cols)} total features ({len(base_feature_cols)} base + {len(eng_cols)} engineered)")

# =============================================================================
# CV WITH PER-FOLD IMPORTANCE RANKING
# =============================================================================
print(f"\n[3/5] Walk-forward CV with per-fold importance ranking...")

tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=TARGET_BARS)

n_lgbm = len(LEARNING_RATES) * len(MAX_DEPTHS) * len(N_ESTIMATORS_LIST)
n_groups = len(FEATURE_GROUP_SIZES)
total = n_groups * n_lgbm
print(f"   {n_groups} feature groups × {n_lgbm} LightGBM configs = {total} configs")
print(f"   Feature groups: {[f'top-{k}' if k else 'all' for k in FEATURE_GROUP_SIZES]}")

results = []
config_num = 0

for fg_size in FEATURE_GROUP_SIZES:
    fg_label = f"top-{fg_size}" if fg_size else "all"

    for lr in LEARNING_RATES:
        for depth in MAX_DEPTHS:
            for n_est in N_ESTIMATORS_LIST:
                config_num += 1
                fold_preds = np.full(len(df), np.nan)

                for train_idx, test_idx in tscv.split(df):
                    X_train_full = df.iloc[train_idx][all_feature_cols]
                    y_train = df.iloc[train_idx]["target"].values
                    X_test_full = df.iloc[test_idx][all_feature_cols]

                    if fg_size is not None and fg_size < len(all_feature_cols):
                        # Compute importance on this fold's training data only
                        imp_model = LGBMRegressor(
                            n_estimators=200, learning_rate=0.05, max_depth=5,
                            num_leaves=31, subsample=0.8, colsample_bytree=0.7,
                            min_child_samples=50, random_state=42, verbose=-1,
                        )
                        imp_model.fit(X_train_full, y_train)
                        importances = imp_model.feature_importances_
                        top_idx = np.argsort(importances)[-fg_size:]
                        selected = [all_feature_cols[i] for i in top_idx]
                    else:
                        selected = all_feature_cols

                    X_train = df.iloc[train_idx][selected]
                    X_test = df.iloc[test_idx][selected]

                    model = LGBMRegressor(
                        objective="regression", n_estimators=n_est,
                        learning_rate=lr, max_depth=depth, num_leaves=31,
                        subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
                        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
                    )
                    model.fit(X_train, y_train)
                    preds = np.maximum(model.predict(X_test), 0)
                    fold_preds[test_idx] = preds

                valid = ~np.isnan(fold_preds)
                y_t = df.loc[valid, "target"].values
                y_p = fold_preds[valid]
                m = vol_metrics(y_t, y_p)
                score = composite_score(m)

                results.append({
                    "config_num": config_num, "fg": fg_label, "fg_size": fg_size,
                    "lr": lr, "depth": depth, "n_est": n_est,
                    **m, "score": score,
                })

                if config_num % 8 == 0 or config_num <= 2:
                    print(f"   [{config_num:3d}/{total}] fg={fg_label:<6} lr={lr:.2f} d={depth} n={n_est:3d} "
                          f"→ r={m['pearson_r']:.4f} R²={m['r2']:.4f} score={score:+.4f}")

# =============================================================================
# RANK & SELECT BEST PER GROUP
# =============================================================================
results_df = pd.DataFrame(results)

print(f"\n[4/5] Results — best per feature group:")
print(f"   {'Group':<8} {'lr':>5} {'d':>2} {'n':>4} │ {'r':>6} {'R²':>6} {'QLIKE':>7} {'cal':>5} {'score':>7}")
print(f"   {'─'*60}")

deploy_configs = []
for fg_label in [f"top-{k}" if k else "all" for k in FEATURE_GROUP_SIZES]:
    fg_df = results_df[results_df["fg"] == fg_label].sort_values("score", ascending=False)
    if len(fg_df) == 0:
        continue
    best = fg_df.iloc[0]
    print(f"   {fg_label:<8} {best['lr']:.2f}  {int(best['depth']):2d} {int(best['n_est']):4d} │ "
          f"{best['pearson_r']:.4f} {best['r2']:.4f} {best['qlike']:.5f} {best['cal_ratio']:.3f} {best['score']:+.4f}")
    deploy_configs.append(best)

# =============================================================================
# TRAIN & SAVE (one per feature group)
# =============================================================================
print(f"\n[5/5] Training & saving {len(deploy_configs)} diverse models...")
n_smoke_failures = 0
for rank, cfg in enumerate(deploy_configs):
    fg_size = cfg["fg_size"]
    fg_label = cfg["fg"]

    # Compute importance on ALL data for final model
    if fg_size is not None and fg_size < len(all_feature_cols):
        imp_model = LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            num_leaves=31, subsample=0.8, colsample_bytree=0.7,
            min_child_samples=50, random_state=42, verbose=-1,
        )
        imp_model.fit(df[all_feature_cols], df["target"])
        importances = imp_model.feature_importances_
        top_idx = np.argsort(importances)[-int(fg_size):]
        selected = [all_feature_cols[i] for i in top_idx]
        print(f"   {fg_label}: top features = {selected[:5]}...")
    else:
        selected = all_feature_cols

    model = LGBMRegressor(
        objective="regression", n_estimators=int(cfg["n_est"]),
        learning_rate=cfg["lr"], max_depth=int(cfg["depth"]), num_leaves=31,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
    )
    model.fit(df[selected], df["target"])

    def _make_predict(m, _tickers=TICKERS, _sel=selected,
                      _n_input=NUMBER_OF_INPUT_BARS, _target_bars=TARGET_BARS,
                      _interval=INTERVAL, _target_type=TARGET_TYPE,
                      _base_cols=base_feature_cols, _eng_fn=engineer_features):
        _model_str = m.booster_.model_to_string()
        _feature_list = _sel[:]
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
            x = live_features[_feature_list].values.reshape(1, -1)
            return max(0.0, float(booster.predict(x)[0]))
        return predict

    fn = _make_predict(model)
    pkl = f"predict_{TOPIC_ID}_imp_{fg_label}.pkl"
    try:
        val = fn()
        print(f"   {fg_label}: score={cfg['score']:+.4f} → {val:.6f} → {pkl}")
    except Exception as e:
        print(f"   {fg_label}: FAILED ({e}) → {pkl}")
        n_smoke_failures += 1
    with open(pkl, "wb") as f:
        cloudpickle.dump(fn, f)

# =============================================================================
# SCATTER PLOT
# =============================================================================
print("\n  Generating scatter plot...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use best overall model for scatter
best_overall = results_df.sort_values("score", ascending=False).iloc[0]
fig, ax = plt.subplots(figsize=(8, 8))

# Re-collect predictions from the best config's CV
fg_size = best_overall["fg_size"]
fold_preds = np.full(len(df), np.nan)
for train_idx, test_idx in tscv.split(df):
    y_train = df.iloc[train_idx]["target"].values
    if fg_size is not None and fg_size < len(all_feature_cols):
        imp_m = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                              num_leaves=31, subsample=0.8, colsample_bytree=0.7,
                              min_child_samples=50, random_state=42, verbose=-1)
        imp_m.fit(df.iloc[train_idx][all_feature_cols], y_train)
        top_idx = np.argsort(imp_m.feature_importances_)[-int(fg_size):]
        sel = [all_feature_cols[i] for i in top_idx]
    else:
        sel = all_feature_cols
    m = LGBMRegressor(objective="regression", n_estimators=int(best_overall["n_est"]),
                      learning_rate=best_overall["lr"], max_depth=int(best_overall["depth"]),
                      num_leaves=31, subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
                      reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)
    m.fit(df.iloc[train_idx][sel], y_train)
    fold_preds[test_idx] = np.maximum(m.predict(df.iloc[test_idx][sel]), 0)

valid = ~np.isnan(fold_preds)
ax.scatter(df.loc[valid, "target"], fold_preds[valid], alpha=0.05, s=4, c="#4A90D9", edgecolors="none")
lims = [0, max(df.loc[valid, "target"].max(), fold_preds[valid].max()) * 1.05]
ax.plot(lims, lims, "r--", lw=1.5, alpha=0.7, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("True Volatility (√T-scaled)", fontsize=13)
ax.set_ylabel("Predicted Volatility", fontsize=13)
ax.set_title(f"Topic {TOPIC_ID} — {TICKERS[0].upper()} 4h Vol (Importance Groups)\n"
             f"Best: {best_overall['fg']}, r={best_overall['pearson_r']:.4f}, "
             f"R²={best_overall['r2']:.4f}", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"scatter_{TOPIC_ID}_imp_groups.png", dpi=150)
plt.close()
print(f"   Saved scatter_{TOPIC_ID}_imp_groups.png")

print("\n" + "=" * 70)
if n_smoke_failures:
    print(f"DONE — {n_smoke_failures} smoke test(s) FAILED (artifacts still saved)")
    print("=" * 70)
    sys.exit(n_smoke_failures)
print("COMPLETE!")
print("=" * 70)
