#!/usr/bin/env python3
"""
Topic 38 — SOL/USD 8h Price — v3 (CZAR Loss)
=============================================

Uses CZAR loss instead of MSE/Huber. CZAR penalizes wrong-sign predictions
heavily, softens near-zero returns, and normalizes by local volatility.
This should help SOL where the signal is weak — CZAR won't waste capacity
fitting noise on near-zero returns.

Combined with the directional features from v2.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
import cloudpickle
from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator, make_czar_objective
from allora_forge_builder_kit.utils import get_api_key

# =============================================================================
# CONFIG
# =============================================================================
TICKERS = ["solusd"]
DAYS_OF_HISTORY = 1825
INTERVAL = "1h"
NUMBER_OF_INPUT_BARS = 48
TARGET_BARS = 8

N_SPLITS = 3
# Smaller grid — focus on CZAR-specific params
N_ESTIMATORS_MAX = 600
N_ESTIMATORS_CHECKPOINTS = [100, 300, 600]
LEARNING_RATES = [0.01, 0.03, 0.07]
MAX_DEPTHS = [3, 5]
NUM_LEAVES = [15, 31]
CZAR_ALPHAS = [0.3, 0.5, 0.7, 1.0]  # CZAR alpha param (MSE curvature)

print("=" * 70)
print("Topic 38 — SOL/USD 8h Price — v3 (CZAR Loss)")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/5] Loading data...")
api_key = get_api_key(
    api_key_file=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".allora_api_key")
)
os.environ["ALLORA_API_KEY"] = api_key

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
# DIRECTIONAL FEATURES (same as v2)
# =============================================================================
print("\n[2/5] Engineering directional features...")


def engineer_directional_features(row):
    n = NUMBER_OF_INPUT_BARS
    closes = np.array([row[f"feature_close_{i}"] for i in range(n)])
    highs = np.array([row[f"feature_high_{i}"] for i in range(n)])
    lows = np.array([row[f"feature_low_{i}"] for i in range(n)])
    volumes = np.array([row[f"feature_volume_{i}"] for i in range(n)])

    log_rets = np.diff(np.log(closes + 1e-12))
    abs_rets = np.abs(log_rets)
    f = {}

    f["ret_1h"] = log_rets[-1] if len(log_rets) >= 1 else 0
    f["ret_4h"] = np.sum(log_rets[-4:]) if len(log_rets) >= 4 else 0
    f["ret_8h"] = np.sum(log_rets[-8:]) if len(log_rets) >= 8 else 0
    f["ret_24h"] = np.sum(log_rets[-24:]) if len(log_rets) >= 24 else 0
    f["ret_48h"] = np.sum(log_rets) if len(log_rets) >= 2 else 0

    vol_8h = np.std(log_rets[-8:], ddof=1) if len(log_rets) >= 8 else 1e-6
    vol_24h = np.std(log_rets[-24:], ddof=1) if len(log_rets) >= 24 else 1e-6
    vol_48h = np.std(log_rets, ddof=1) if len(log_rets) >= 2 else 1e-6
    f["vol_8h"] = vol_8h
    f["vol_24h"] = vol_24h
    f["vol_48h"] = vol_48h

    f["znorm_ret_1h"] = log_rets[-1] / (vol_8h + 1e-12) if len(log_rets) >= 1 else 0
    f["znorm_ret_4h"] = np.sum(log_rets[-4:]) / (vol_8h * 2 + 1e-12) if len(log_rets) >= 4 else 0
    f["znorm_ret_8h"] = np.sum(log_rets[-8:]) / (vol_8h * np.sqrt(8) + 1e-12) if len(log_rets) >= 8 else 0

    if len(log_rets) >= 24:
        up_rets = log_rets[-24:][log_rets[-24:] > 0]
        dn_rets = log_rets[-24:][log_rets[-24:] < 0]
        up_vol = np.std(up_rets, ddof=1) if len(up_rets) > 1 else 1e-6
        dn_vol = np.std(np.abs(dn_rets), ddof=1) if len(dn_rets) > 1 else 1e-6
        f["vol_skew_24h"] = (up_vol - dn_vol) / (up_vol + dn_vol + 1e-12)
        f["up_fraction_24h"] = np.mean(log_rets[-24:] > 0)
    else:
        f["vol_skew_24h"] = 0
        f["up_fraction_24h"] = 0.5

    f["up_fraction_8h"] = np.mean(log_rets[-8:] > 0) if len(log_rets) >= 8 else 0.5

    if len(log_rets) >= 10:
        f["ret_autocorr"] = np.corrcoef(log_rets[-9:], log_rets[-10:-1])[0, 1]
        if not np.isfinite(f["ret_autocorr"]):
            f["ret_autocorr"] = 0
    else:
        f["ret_autocorr"] = 0

    f["vol_ratio_8_48"] = vol_8h / (vol_48h + 1e-12)
    f["vol_expanding"] = 1.0 if vol_8h > vol_24h else 0.0

    vol_trend = np.mean(volumes[-4:]) / (np.mean(volumes[-24:]) + 1e-12) if len(volumes) >= 24 else 1
    f["vol_price_divergence"] = vol_trend * np.sign(-f["ret_4h"])
    f["volume_ratio"] = vol_trend

    if len(closes) >= 24:
        f["zscore_24h"] = (closes[-1] - np.mean(closes[-24:])) / (np.std(closes[-24:], ddof=1) + 1e-12)
    else:
        f["zscore_24h"] = 0

    if len(log_rets) >= 8:
        net = abs(np.sum(log_rets[-8:]))
        path = np.sum(abs_rets[-8:])
        f["efficiency_8h"] = net / (path + 1e-12)
    else:
        f["efficiency_8h"] = 0

    hl = highs - lows
    f["hl_range_8h"] = np.mean(hl[-8:])
    f["hl_range_ratio"] = np.mean(hl[-8:]) / (np.mean(hl) + 1e-12)

    return pd.Series(f)


engineered = df_all.apply(engineer_directional_features, axis=1)
df_all = pd.concat([df_all, engineered], axis=1)
feature_cols = list(engineered.columns)
df_all = df_all.dropna(subset=feature_cols + ["target"])
print(f"✅ {len(feature_cols)} features")

# =============================================================================
# COMPUTE ROLLING VOL FOR CZAR (needed for the loss)
# =============================================================================
# Use 8h rolling std of target as the vol normalization for CZAR
targets = df_all["target"].values
rolling_std = pd.Series(targets).rolling(8, min_periods=2).std().fillna(targets.std()).values
df_all["_rolling_std"] = rolling_std

# =============================================================================
# GRID SEARCH WITH CZAR LOSS
# =============================================================================
print(f"\n[3/5] Grid search with CZAR loss...")
tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=TARGET_BARS)
evaluator = PerformanceEvaluator()
results = []
config_num = 0

total = len(LEARNING_RATES) * len(MAX_DEPTHS) * len(NUM_LEAVES) * len(CZAR_ALPHAS) * len(N_ESTIMATORS_CHECKPOINTS)
print(f"   {total} configs (CZAR alpha × LR × depth × leaves × checkpoints)")

for czar_alpha in CZAR_ALPHAS:
    for lr in LEARNING_RATES:
        for depth in MAX_DEPTHS:
            for leaves in NUM_LEAVES:
                    fold_models = []
                    for train_idx, test_idx in tscv.split(df_all):
                        y_train = df_all.iloc[train_idx]["target"].values
                        std_train = df_all.iloc[train_idx]["_rolling_std"].values

                        # Create CZAR objective for this fold
                        czar_obj = make_czar_objective(
                            std=std_train, alpha=czar_alpha
                        )

                        lgb = LGBMRegressor(
                            objective=czar_obj,
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
                            verbose=-1,
                        )
                        lgb.fit(
                            df_all.iloc[train_idx][feature_cols],
                            y_train,
                        )
                        fold_models.append((lgb, test_idx))

                    for n_est in N_ESTIMATORS_CHECKPOINTS:
                        config_num += 1
                        df_all["pred"] = np.nan
                        for lgb, test_idx in fold_models:
                            preds = lgb.predict(
                                df_all.iloc[test_idx][feature_cols],
                                num_iteration=n_est,
                            )
                            df_all.iloc[test_idx, df_all.columns.get_loc("pred")] = preds

                        valid = ~df_all["pred"].isna()
                        y_t = df_all.loc[valid, "target"].values
                        y_p = df_all.loc[valid, "pred"].values
                        metrics = evaluator.evaluate(y_true=y_t, y_pred=y_p)
                        r_val, _ = pearsonr(y_t, y_p)
                        cal = np.std(y_p) / (np.std(y_t) + 1e-12)

                        m = metrics.get("metrics", metrics)
                        results.append({
                            "config_num": config_num,
                            "n_estimators": n_est,
                            "learning_rate": lr,
                            "max_depth": depth,
                            "num_leaves": leaves,
                            "czar_alpha": czar_alpha,
                            "da": m.get("directional_accuracy", 0),
                            "da_ci": m.get("da_ci_lower", 0),
                            "da_pval": m.get("da_pvalue", 1),
                            "pearson": r_val,
                            "wrmse_imp": m.get("wrmse_improvement", 0),
                            "czar_imp": m.get("czar_improvement", 0),
                            "cal_ratio": cal,
                            "num_passed": metrics["num_passed"],
                        })

                        if config_num % 10 == 0 or config_num <= 3:
                            print(
                                f"   [{config_num:3d}/{total}] a={czar_alpha:.1f} "
                                f"lr={lr:.2f} d={depth} l={leaves:2d} n={n_est:3d} "
                                f"→ {metrics['num_passed']}/7 r={r_val:+.4f} DA={m.get('directional_accuracy',0):.3f}"
                            )

# =============================================================================
# RANK & SELECT
# =============================================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["num_passed", "pearson"], ascending=[False, False])

print(f"\n✅ Tested {len(results)} configs")
print(f"\n   Top 10:")
print(f"   {'#':>3} {'a':>3} {'e':>3} {'n':>4} {'lr':>5} {'d':>2} {'l':>3} │ {'DA':>5} {'r':>7} {'WRMSE':>6} {'CZAR':>6} {'cal':>5} │ {'pts':>3}")
print(f"   {'─'*70}")
for _, row in results_df.head(10).iterrows():
    print(
        f"   {int(row['config_num']):3d} {row['czar_alpha']:.1f} "
        f"{int(row['n_estimators']):4d} {row['learning_rate']:5.2f} "
        f"{int(row['max_depth']):2d} {int(row['num_leaves']):3d} │ "
        f"{row['da']:.3f} {row['pearson']:+.4f} "
        f"{row['wrmse_imp']:+.4f} {row['czar_imp']:+.4f} {row['cal_ratio']:.3f} │ "
        f"{int(row['num_passed']):3d}"
    )

best_cfg = int(results_df.iloc[0]["config_num"])
best = next(r for r in results if r["config_num"] == best_cfg)
print(f"\n   Best: #{best_cfg} r={best['pearson']:+.4f} DA={best['da']:.3f} ({best['num_passed']}/7)")

# =============================================================================
# TRAIN & SAVE TOP 3
# =============================================================================
print(f"\n[4/5] Training top 3...")
top3 = results_df.head(3)
trained = []
for rank_idx, (_, row) in enumerate(top3.iterrows()):
    std_all = df_all["_rolling_std"].values
    czar_obj = make_czar_objective(
        std=std_all, alpha=row["czar_alpha"]
    )
    model = LGBMRegressor(
        objective=czar_obj,
        n_estimators=int(row["n_estimators"]),
        learning_rate=row["learning_rate"],
        max_depth=int(row["max_depth"]),
        num_leaves=int(row["num_leaves"]),
        subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
    )
    model.fit(df_all[feature_cols], df_all["target"])
    trained.append((int(row["config_num"]), model, row))
    print(f"   Model {rank_idx+1}: #{int(row['config_num'])} "
          f"(a={row['czar_alpha']:.1f} r={row['pearson']:+.4f} DA={row['da']:.3f})")

print(f"\n[5/5] Saving...")
for rank_idx, (cfg, model, row) in enumerate(trained):
    def _make_predict(m):
        # Serialize booster to string — avoids pickling czar_loss module
        _model_str = m.booster_.model_to_string()
        _feature_cols = feature_cols[:]
        _tickers = TICKERS[:]
        _n_input = NUMBER_OF_INPUT_BARS
        _target_bars = TARGET_BARS
        _interval = INTERVAL
        # Capture feature engineering as a standalone function
        _eng_fn = engineer_directional_features
        def predict(nonce=None):
            import os
            import lightgbm as lgb
            import numpy as np
            from allora_forge_builder_kit import AlloraMLWorkflow
            _wf = AlloraMLWorkflow(
                tickers=_tickers, number_of_input_bars=_n_input,
                target_bars=_target_bars, interval=_interval,
                data_source="allora", api_key=os.environ["ALLORA_API_KEY"],
            )
            booster = lgb.Booster(model_str=_model_str)
            live_row = _wf.get_live_features(ticker=_tickers[0])
            if live_row is None or len(live_row) == 0:
                raise ValueError("No live features")
            live_eng = _eng_fn(live_row.iloc[0])
            current_price = float(live_row.attrs.get("current_price", float("nan")))
            if not np.isfinite(current_price) or current_price <= 0:
                snap = _wf._dm.get_live_snapshot(_tickers)
                if snap is not None and len(snap) > 0:
                    current_price = float(snap["close"].iloc[-1])
            log_ret = booster.predict(live_eng[_feature_cols].values.reshape(1, -1))[0]
            return float(current_price * np.exp(log_ret))
        return predict

    fn = _make_predict(model)
    pkl = f"predict_38_czar_rank{rank_idx+1}.pkl"
    try:
        price = fn()
        print(f"   Model {rank_idx+1} (#{cfg}): ${price:,.2f} → {pkl}")
    except Exception as e:
        print(f"   Model {rank_idx+1} (#{cfg}): FAILED ({e}) → {pkl}")
    with open(pkl, "wb") as f:
        cloudpickle.dump(fn, f)

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
