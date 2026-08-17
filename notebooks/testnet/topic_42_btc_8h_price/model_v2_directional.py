#!/usr/bin/env python3
"""
Topic 42 — BTC/USD 8h Price — v2 (Directional Signals Conditioned on Vol)
=========================================================================

Key insight: Vol features alone predict magnitude, not direction.
BTC is efficient — we need directional signals conditioned on vol regime.

Features:
- Vol-normalized momentum (signed returns / vol)
- Upside vs downside vol asymmetry (skew)
- Return autocorrelation (trending vs mean-reverting regime)
- Vol regime indicator (high/low vol → different dynamics)
- Volume-price divergence (volume up + price down = bearish)
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
import cloudpickle
from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator
from allora_forge_builder_kit.utils import get_api_key

# =============================================================================
# CONFIG
# =============================================================================
TICKERS = ["btcusd"]
DAYS_OF_HISTORY = 1825
INTERVAL = "1h"
NUMBER_OF_INPUT_BARS = 48
TARGET_BARS = 8
H = TARGET_BARS

N_SPLITS = 3
N_ESTIMATORS_MAX = 800
N_ESTIMATORS_CHECKPOINTS = [100, 300, 600]
LEARNING_RATES = [0.01, 0.03, 0.07, 0.1]
MAX_DEPTHS = [3, 5, 7]
NUM_LEAVES = [15, 31]
TOP_K_FEATURES_GRID = [5, 10, 25, 50]

print("=" * 70)
print("Topic 42 — BTC/USD 8h Price — v2 (Directional + Vol-Conditioned)")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/6] Loading data...")
api_key = get_api_key(
    api_key_file=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".allora_api_key")
)
os.environ.setdefault("ALLORA_API_KEY", api_key)

workflow = AlloraMLWorkflow(
    tickers=TICKERS, number_of_input_bars=NUMBER_OF_INPUT_BARS,
    target_bars=TARGET_BARS, interval=INTERVAL,
    data_source="allora", api_key=api_key,
)

start_date = datetime.now(timezone.utc) - timedelta(days=DAYS_OF_HISTORY)
workflow.backfill(start=start_date)
df_all = workflow.get_full_feature_target_dataframe(start_date=start_date).reset_index()
base_feature_cols = [c for c in df_all.columns if c.startswith("feature_")]
df_all = df_all.dropna(subset=base_feature_cols + ["target"])
print(f"✅ {len(df_all):,} samples")

# =============================================================================
# DIRECTIONAL FEATURE ENGINEERING
# =============================================================================
print("\n[2/6] Engineering directional features...")


def engineer_directional_features(row):
    """Directional signals conditioned on volatility regime."""
    n = NUMBER_OF_INPUT_BARS
    closes = np.array([row[f"feature_close_{i}"] for i in range(n)])
    highs = np.array([row[f"feature_high_{i}"] for i in range(n)])
    lows = np.array([row[f"feature_low_{i}"] for i in range(n)])
    volumes = np.array([row[f"feature_volume_{i}"] for i in range(n)])

    log_rets = np.diff(np.log(closes + 1e-12))
    abs_rets = np.abs(log_rets)
    f = {}

    # === RAW RETURNS (for baseline) ===
    f["ret_1h"] = log_rets[-1] if len(log_rets) >= 1 else 0
    f["ret_4h"] = np.sum(log_rets[-4:]) if len(log_rets) >= 4 else 0
    f["ret_8h"] = np.sum(log_rets[-8:]) if len(log_rets) >= 8 else 0
    f["ret_24h"] = np.sum(log_rets[-24:]) if len(log_rets) >= 24 else 0
    f["ret_48h"] = np.sum(log_rets) if len(log_rets) >= 2 else 0

    # === VOLATILITY (for conditioning) ===
    vol_8h = np.std(log_rets[-8:], ddof=1) if len(log_rets) >= 8 else 1e-6
    vol_24h = np.std(log_rets[-24:], ddof=1) if len(log_rets) >= 24 else 1e-6
    vol_48h = np.std(log_rets, ddof=1) if len(log_rets) >= 2 else 1e-6
    f["vol_8h"] = vol_8h
    f["vol_24h"] = vol_24h
    f["vol_48h"] = vol_48h

    # === VOL-NORMALIZED MOMENTUM (the key directional signal) ===
    # "How many sigmas has price moved?" — direction + magnitude in vol context
    f["znorm_ret_1h"] = log_rets[-1] / (vol_8h + 1e-12) if len(log_rets) >= 1 else 0
    f["znorm_ret_4h"] = np.sum(log_rets[-4:]) / (vol_8h * 2 + 1e-12) if len(log_rets) >= 4 else 0
    f["znorm_ret_8h"] = np.sum(log_rets[-8:]) / (vol_8h * np.sqrt(8) + 1e-12) if len(log_rets) >= 8 else 0
    f["znorm_ret_24h"] = np.sum(log_rets[-24:]) / (vol_24h * np.sqrt(24) + 1e-12) if len(log_rets) >= 24 else 0

    # === UPSIDE vs DOWNSIDE VOL (skew — directional asymmetry) ===
    if len(log_rets) >= 24:
        up_rets = log_rets[-24:][log_rets[-24:] > 0]
        dn_rets = log_rets[-24:][log_rets[-24:] < 0]
        up_vol = np.std(up_rets, ddof=1) if len(up_rets) > 1 else 1e-6
        dn_vol = np.std(np.abs(dn_rets), ddof=1) if len(dn_rets) > 1 else 1e-6
        f["vol_skew_24h"] = (up_vol - dn_vol) / (up_vol + dn_vol + 1e-12)
        # Fraction of positive returns (directional bias)
        f["up_fraction_24h"] = np.mean(log_rets[-24:] > 0)
    else:
        f["vol_skew_24h"] = 0
        f["up_fraction_24h"] = 0.5

    if len(log_rets) >= 8:
        f["up_fraction_8h"] = np.mean(log_rets[-8:] > 0)
    else:
        f["up_fraction_8h"] = 0.5

    # === RETURN AUTOCORRELATION (trending vs mean-reverting) ===
    if len(log_rets) >= 10:
        f["ret_autocorr"] = np.corrcoef(log_rets[-9:], log_rets[-10:-1])[0, 1]
        if not np.isfinite(f["ret_autocorr"]):
            f["ret_autocorr"] = 0
    else:
        f["ret_autocorr"] = 0

    # Signed autocorrelation of absolute returns (vol clustering direction)
    if len(abs_rets) >= 10:
        f["absret_autocorr"] = np.corrcoef(abs_rets[-9:], abs_rets[-10:-1])[0, 1]
        if not np.isfinite(f["absret_autocorr"]):
            f["absret_autocorr"] = 0
    else:
        f["absret_autocorr"] = 0

    # === VOL REGIME (high vol vs low vol — different dynamics) ===
    f["vol_ratio_8_48"] = vol_8h / (vol_48h + 1e-12)
    f["vol_expanding"] = 1.0 if vol_8h > vol_24h else 0.0

    # === VOLUME-PRICE DIVERGENCE ===
    # Volume up + price down = bearish divergence
    vol_trend = np.mean(volumes[-4:]) / (np.mean(volumes[-24:]) + 1e-12) if len(volumes) >= 24 else 1
    price_trend = f["ret_4h"]
    f["vol_price_divergence"] = vol_trend * np.sign(-price_trend)  # positive = bearish divergence
    f["volume_ratio"] = vol_trend

    # === MEAN REVERSION SIGNAL ===
    if len(closes) >= 24:
        f["zscore_24h"] = (closes[-1] - np.mean(closes[-24:])) / (np.std(closes[-24:], ddof=1) + 1e-12)
    else:
        f["zscore_24h"] = 0

    # === EFFICIENCY RATIO (trending vs choppy) ===
    if len(log_rets) >= 8:
        net = abs(np.sum(log_rets[-8:]))
        path = np.sum(abs_rets[-8:])
        f["efficiency_8h"] = net / (path + 1e-12)
    else:
        f["efficiency_8h"] = 0

    # === HIGH-LOW RANGE ===
    hl = highs - lows
    f["hl_range_8h"] = np.mean(hl[-8:])
    f["hl_range_ratio"] = np.mean(hl[-8:]) / (np.mean(hl) + 1e-12)

    return pd.Series(f)


engineered = df_all.apply(engineer_directional_features, axis=1)
df_all = pd.concat([df_all, engineered], axis=1)

# Use ONLY engineered features — no raw OHLCV base features
feature_cols = list(engineered.columns)
df_all = df_all.dropna(subset=feature_cols + ["target"])
print(f"✅ {len(feature_cols)} directional features (no raw OHLCV)")

# =============================================================================
# GRID SEARCH
# =============================================================================
print(f"\n[3/6] Grid search...")
tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=TARGET_BARS)
evaluator = PerformanceEvaluator()
results = []
config_num = 0
all_feature_importances = {}

total_configs = len(LEARNING_RATES) * len(MAX_DEPTHS) * len(NUM_LEAVES) * len(TOP_K_FEATURES_GRID) * len(N_ESTIMATORS_CHECKPOINTS)
print(f"   {total_configs} configs")

for lr in LEARNING_RATES:
    for depth in MAX_DEPTHS:
        for leaves in NUM_LEAVES:
            fold_importances = []
            fold_selectors = []
            for train_idx, test_idx in tscv.split(df_all):
                selector = LGBMRegressor(
                    n_estimators=200, learning_rate=0.05,
                    max_depth=depth, num_leaves=leaves,
                    subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
                    random_state=42, verbose=-1,
                )
                selector.fit(df_all.iloc[train_idx][feature_cols], df_all.iloc[train_idx]["target"])
                fold_importances.append(selector.feature_importances_)
                fold_selectors.append((train_idx, test_idx, df_all.iloc[train_idx]["target"]))

            avg_imp = np.mean(fold_importances, axis=0)
            for fi in np.argsort(avg_imp)[-10:]:
                fn = feature_cols[fi]
                all_feature_importances[fn] = all_feature_importances.get(fn, 0) + avg_imp[fi]

            for top_k in TOP_K_FEATURES_GRID:
                fold_models = []
                for fold_idx, (train_idx, test_idx, y_train) in enumerate(fold_selectors):
                    top_idx = np.argsort(fold_importances[fold_idx])[-top_k:]
                    selected = [feature_cols[i] for i in top_idx]
                    lgb = LGBMRegressor(
                        n_estimators=N_ESTIMATORS_MAX, learning_rate=lr,
                        max_depth=depth, num_leaves=leaves,
                        subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
                        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
                    )
                    lgb.fit(df_all.iloc[train_idx][selected], y_train)
                    fold_models.append((lgb, test_idx, selected))

                for n_est in N_ESTIMATORS_CHECKPOINTS:
                    config_num += 1
                    df_all["pred"] = np.nan
                    for lgb, test_idx, selected in fold_models:
                        preds = lgb.predict(df_all.iloc[test_idx][selected], num_iteration=n_est)
                        df_all.iloc[test_idx, df_all.columns.get_loc("pred")] = preds

                    valid = ~df_all["pred"].isna()
                    y_t = df_all.loc[valid, "target"].values
                    y_p = df_all.loc[valid, "pred"].values
                    metrics = evaluator.evaluate(y_true=y_t, y_pred=y_p)
                    r_val, _ = pearsonr(y_t, y_p)
                    cal = np.std(y_p) / (np.std(y_t) + 1e-12)

                    m = metrics.get("metrics", metrics)
                    results.append({
                        "config_num": config_num, "n_estimators": n_est,
                        "learning_rate": lr, "max_depth": depth,
                        "num_leaves": leaves, "top_k": top_k,
                        "da": m.get("directional_accuracy", 0),
                        "da_ci": m.get("da_ci_lower", 0),
                        "da_pval": m.get("da_pvalue", 1),
                        "pearson": r_val,
                        "pearson_pval": m.get("pearson_pvalue", 1),
                        "wrmse_imp": m.get("wrmse_improvement", 0),
                        "czar_imp": m.get("czar_improvement", 0),
                        "cal_ratio": cal,
                        "num_passed": metrics["num_passed"],
                    })

                    if config_num % 20 == 0 or config_num <= 3:
                        print(f"   [{config_num:3d}/{total_configs}] n={n_est:3d} lr={lr:.2f} d={depth} l={leaves:2d} k={top_k:2d} "
                              f"→ {metrics['num_passed']}/7 r={r_val:+.4f} cal={cal:.3f}")

# =============================================================================
# RANK & SELECT
# =============================================================================
results_df = pd.DataFrame(results)

rank_cols = {}
rank_cols["rk_da"] = results_df["da"].rank(ascending=False)
rank_cols["rk_da_ci"] = results_df["da_ci"].rank(ascending=False)
rank_cols["rk_da_pval"] = results_df["da_pval"].rank(ascending=True)
rank_cols["rk_pearson"] = results_df["pearson"].rank(ascending=False)
rank_cols["rk_pear_pval"] = results_df["pearson_pval"].rank(ascending=True)
rank_cols["rk_wrmse"] = results_df["wrmse_imp"].rank(ascending=False)
rank_cols["rk_czar"] = results_df["czar_imp"].rank(ascending=False)
rank_cols["rk_cal"] = (1 - results_df["cal_ratio"]).abs().rank(ascending=True)

for col, vals in rank_cols.items():
    results_df[col] = vals

core_rank_cols = [c for c in rank_cols if c != "rk_cal"]
results_df["core_avg_rank"] = results_df[core_rank_cols].mean(axis=1)
results_df = results_df.sort_values(["num_passed", "core_avg_rank"], ascending=[False, True])

print(f"\n✅ Tested {len(results)} configs")

print(f"\n   Top 20 features:")
for i, (fn, imp) in enumerate(sorted(all_feature_importances.items(), key=lambda x: x[1], reverse=True)[:20]):
    print(f"   {i+1:2d}. {fn:<30s} {imp:>10.1f}")

print(f"\n   Top 10 models:")
print(f"   {'#':>3} {'n':>4} {'lr':>5} {'d':>2} {'l':>3} {'k':>3} │ {'DA':>5} {'CI':>5} {'pval':>5} {'r':>6} {'WRMSE':>6} {'CZAR':>6} {'cal':>5} │ {'pts':>3}")
print(f"   {'─'*80}")
for _, row in results_df.head(10).iterrows():
    print(f"   {int(row['config_num']):3d} {int(row['n_estimators']):4d} {row['learning_rate']:5.2f} "
          f"{int(row['max_depth']):2d} {int(row['num_leaves']):3d} {int(row['top_k']):3d} │ "
          f"{row['da']:.3f} {row['da_ci']:.3f} {row['da_pval']:.3f} {row['pearson']:+.4f} "
          f"{row['wrmse_imp']:+.4f} {row['czar_imp']:+.4f} {row['cal_ratio']:.3f} │ "
          f"{int(row['num_passed']):3d}")

best_cfg = int(results_df.iloc[0]["config_num"])
best = next(r for r in results if r["config_num"] == best_cfg)
print(f"\n   Best: #{best_cfg} r={best['pearson']:+.4f} DA={best['da']:.3f} ({best['num_passed']}/7)")

# =============================================================================
# TRAIN & DEPLOY TOP 3
# =============================================================================
print(f"\n[4/6] Training top 3...")
top3 = results_df.head(3)
trained = []
for rank_idx, (_, row) in enumerate(top3.iterrows()):
    k = int(row["top_k"])
    sel_model = LGBMRegressor(n_estimators=200, learning_rate=0.05,
        max_depth=int(row["max_depth"]), num_leaves=int(row["num_leaves"]),
        subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
        random_state=42, verbose=-1)
    sel_model.fit(df_all[feature_cols], df_all["target"])
    top_idx = np.argsort(sel_model.feature_importances_)[-k:]
    selected = [feature_cols[i] for i in top_idx]

    model = LGBMRegressor(
        n_estimators=int(row["n_estimators"]), learning_rate=row["learning_rate"],
        max_depth=int(row["max_depth"]), num_leaves=int(row["num_leaves"]),
        subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)
    model.fit(df_all[selected], df_all["target"])
    trained.append((int(row["config_num"]), model, selected, row))
    print(f"   Model {rank_idx+1}: #{int(row['config_num'])} (k={k}, r={row['pearson']:+.4f} DA={row['da']:.3f})")

# =============================================================================
# SAVE PICKLES
# =============================================================================
print(f"\n[5/6] Testing & saving...")
for rank_idx, (cfg, model, selected, row) in enumerate(trained):
    def _make_predict(m, sel):
        def predict(nonce=None):
            live_row = workflow.get_live_features(ticker=TICKERS[0])
            if live_row is None or len(live_row) == 0:
                raise ValueError("No live features")
            live_eng = engineer_directional_features(live_row.iloc[0])
            current_price = float(live_row.attrs.get("current_price", np.nan))
            if not np.isfinite(current_price) or current_price <= 0:
                snap = workflow._dm.get_live_snapshot(TICKERS)
                if snap is not None and len(snap) > 0:
                    current_price = float(snap["close"].iloc[-1])
            log_ret = m.predict(live_eng[sel].values.reshape(1, -1))[0]
            return float(current_price * np.exp(log_ret))
        return predict

    fn = _make_predict(model, selected)
    pkl = f"predict_42_rank{rank_idx+1}.pkl"
    try:
        price = fn()
        print(f"   Model {rank_idx+1} (#{cfg}): ${price:,.2f} → {pkl}")
    except Exception as e:
        print(f"   Model {rank_idx+1} (#{cfg}): FAILED ({e}) → {pkl}")
    with open(pkl, "wb") as f:
        cloudpickle.dump(fn, f)

with open("predict_42.pkl", "wb") as fout:
    cloudpickle.dump(_make_predict(trained[0][1], trained[0][2]), fout)

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
