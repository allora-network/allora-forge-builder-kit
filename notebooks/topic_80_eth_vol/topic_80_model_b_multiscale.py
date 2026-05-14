#!/usr/bin/env python3
"""
Topic 80 — Model B: Multi-Scale Regime Detection
=================================================

Strategy: Use a 30-bar (30-min) lookback with features designed to capture
volatility clustering (GARCH-like persistence), intraday seasonality proxies,
and multi-scale decomposition of price action. Emphasizes regime transitions
and mean-reversion in volatility.

Trained on 2+ years of 1-minute ETH/USD data.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
import cloudpickle
from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator

# =============================================================================
# CONFIGURATION
# =============================================================================
TICKERS = ["ethusd"]
DAYS_OF_HISTORY = 800  # ~2.2 years
INTERVAL = "1m"
NUMBER_OF_INPUT_BARS = 30  # 30 minutes of 1-min bars
TARGET_BARS = 15
TARGET_TYPE = "volatility"

N_SPLITS = 5
N_ESTIMATORS_MAX = 1500
N_ESTIMATORS_CHECKPOINTS = [300, 600, 1000, 1500]
LEARNING_RATES = [0.005, 0.02]
MAX_DEPTHS = [4, 6]
NUM_LEAVES = [15, 31]

print("=" * 80)
print("Topic 80 — Model B: Multi-Scale Regime (30-bar, 2+ years)")
print("=" * 80)

# =============================================================================
# STEP 1: Initialize & Backfill
# =============================================================================
print("\n[1/5] Initializing workflow...")
from allora_forge_builder_kit.utils import get_api_key

api_key = get_api_key(
    api_key_file=os.path.join(os.path.dirname(__file__), "..", ".allora_api_key")
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
print(f"✅ {NUMBER_OF_INPUT_BARS} bars lookback, {TARGET_BARS}-min vol target")

print(f"\n[2/5] Backfilling {DAYS_OF_HISTORY} days...")
start_date = datetime.now(timezone.utc) - timedelta(days=DAYS_OF_HISTORY)
workflow.backfill(start=start_date)
print("✅ Backfill complete")

# =============================================================================
# STEP 2: Features
# =============================================================================
print("\n[3/5] Extracting features...")
df_all = workflow.get_full_feature_target_dataframe(start_date=start_date).reset_index()

base_feature_cols = [col for col in df_all.columns if col.startswith("feature_")]


def engineer_multiscale_features(row):
    """Multi-scale regime features from 30-bar lookback."""
    n = NUMBER_OF_INPUT_BARS
    closes = np.array([row[f"feature_close_{i}"] for i in range(n)])
    highs = np.array([row[f"feature_high_{i}"] for i in range(n)])
    lows = np.array([row[f"feature_low_{i}"] for i in range(n)])
    volumes = np.array([row[f"feature_volume_{i}"] for i in range(n)])

    log_rets = np.diff(np.log(closes + 1e-12))
    abs_rets = np.abs(log_rets)
    features = {}

    # --- Realised vol at multiple scales ---
    features["vol_5m"] = np.std(log_rets[-5:], ddof=1) if len(log_rets) >= 5 else 0.0
    features["vol_10m"] = np.std(log_rets[-10:], ddof=1) if len(log_rets) >= 10 else 0.0
    features["vol_15m"] = np.std(log_rets[-15:], ddof=1) if len(log_rets) >= 15 else 0.0
    features["vol_30m"] = np.std(log_rets, ddof=1) if len(log_rets) >= 2 else 0.0

    # --- Vol persistence (GARCH-like) ---
    # Exponentially weighted vol (lambda=0.94, like RiskMetrics)
    lam = 0.94
    sq_rets = log_rets ** 2
    ewma_var = sq_rets[0]
    for r2 in sq_rets[1:]:
        ewma_var = lam * ewma_var + (1 - lam) * r2
    features["ewma_vol"] = np.sqrt(ewma_var)
    features["ewma_vs_realized"] = features["ewma_vol"] / (features["vol_30m"] + 1e-12)

    # --- Vol of vol (second-order clustering) ---
    if len(abs_rets) >= 10:
        rolling_vols = [np.std(abs_rets[i : i + 5], ddof=1) for i in range(len(abs_rets) - 5)]
        if len(rolling_vols) >= 2:
            features["vol_of_vol"] = np.std(rolling_vols, ddof=1)
        else:
            features["vol_of_vol"] = 0.0
    else:
        features["vol_of_vol"] = 0.0

    # --- Regime indicators ---
    # Vol ratio (short/long) — high = vol expanding, low = vol contracting
    features["vol_ratio_5_30"] = features["vol_5m"] / (features["vol_30m"] + 1e-12)
    features["vol_ratio_10_30"] = features["vol_10m"] / (features["vol_30m"] + 1e-12)

    # Vol percentile within the window (where are we in the local distribution?)
    if len(abs_rets) >= 15:
        recent_vol = features["vol_5m"]
        rolling_5m_vols = [np.std(log_rets[i : i + 5], ddof=1) for i in range(len(log_rets) - 5)]
        if len(rolling_5m_vols) > 0:
            features["vol_percentile"] = np.mean([1 for v in rolling_5m_vols if v <= recent_vol])
        else:
            features["vol_percentile"] = 0.5
    else:
        features["vol_percentile"] = 0.5

    # --- Mean reversion signal ---
    # Distance from "normal" vol (z-score of current vol)
    if len(abs_rets) >= 20:
        rolling_vols = [np.std(log_rets[i : i + 5], ddof=1) for i in range(len(log_rets) - 5)]
        if len(rolling_vols) >= 5:
            vol_mean = np.mean(rolling_vols)
            vol_std = np.std(rolling_vols, ddof=1)
            features["vol_zscore"] = (features["vol_5m"] - vol_mean) / (vol_std + 1e-12)
        else:
            features["vol_zscore"] = 0.0
    else:
        features["vol_zscore"] = 0.0

    # --- Directional features ---
    features["signed_ret_5m"] = np.sum(log_rets[-5:])
    features["signed_ret_15m"] = np.sum(log_rets[-15:]) if len(log_rets) >= 15 else np.sum(log_rets)
    features["abs_ret_5m"] = np.sum(abs_rets[-5:])

    # Efficiency ratio: |net move| / sum(|moves|) — 1 = trending, 0 = choppy
    net_move = abs(features["signed_ret_15m"])
    total_path = np.sum(abs_rets[-15:]) if len(abs_rets) >= 15 else np.sum(abs_rets)
    features["efficiency_ratio"] = net_move / (total_path + 1e-12)

    # --- High-low based estimators ---
    hl_log = np.log(highs + 1e-12) - np.log(lows + 1e-12)
    features["parkinson_5m"] = np.sqrt(np.mean(hl_log[-5:] ** 2) / (4 * np.log(2)))
    features["parkinson_15m"] = np.sqrt(np.mean(hl_log[-15:] ** 2) / (4 * np.log(2)))
    features["parkinson_ratio"] = features["parkinson_5m"] / (features["parkinson_15m"] + 1e-12)

    # --- Volume dynamics ---
    features["volume_trend"] = np.mean(volumes[-5:]) / (np.mean(volumes[-15:]) + 1e-12)
    features["volume_spike_ratio"] = np.max(volumes[-5:]) / (np.mean(volumes) + 1e-12)

    # --- Autocorrelation of absolute returns (persistence) ---
    if len(abs_rets) >= 6:
        features["absret_autocorr"] = np.corrcoef(abs_rets[1:], abs_rets[:-1])[0, 1]
        if not np.isfinite(features["absret_autocorr"]):
            features["absret_autocorr"] = 0.0
    else:
        features["absret_autocorr"] = 0.0

    # --- Recent extreme moves ---
    features["max_abs_ret_5m"] = np.max(abs_rets[-5:])
    features["max_abs_ret_15m"] = np.max(abs_rets[-15:]) if len(abs_rets) >= 15 else np.max(abs_rets)

    return pd.Series(features)


print("   Engineering multi-scale regime features...")
engineered = df_all.apply(engineer_multiscale_features, axis=1)
df_all = pd.concat([df_all, engineered], axis=1)

feature_cols = base_feature_cols + list(engineered.columns)
df_all = df_all.dropna(subset=feature_cols + ["target"])

print(f"✅ Dataset: {len(df_all):,} samples")
print(f"   Features: {len(base_feature_cols)} base + {len(engineered.columns)} engineered = {len(feature_cols)} total")

# =============================================================================
# STEP 3: Grid Search
# =============================================================================
print("\n[4/5] Grid search...")
tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=TARGET_BARS)
evaluator = PerformanceEvaluator()
results = []
config_num = 0

for lr in LEARNING_RATES:
    for depth in MAX_DEPTHS:
        for leaves in NUM_LEAVES:
            fold_models = []
            for train_idx, test_idx in tscv.split(df_all):
                lgb = LGBMRegressor(
                    n_estimators=N_ESTIMATORS_MAX,
                    learning_rate=lr,
                    max_depth=depth,
                    num_leaves=leaves,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    min_child_samples=50,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    random_state=123,
                    verbose=-1,
                )
                lgb.fit(df_all.iloc[train_idx][feature_cols], df_all.iloc[train_idx]["target"])
                fold_models.append((lgb, test_idx))

            for n_est in N_ESTIMATORS_CHECKPOINTS:
                config_num += 1
                df_all["pred"] = np.nan
                for lgb, test_idx in fold_models:
                    preds = lgb.predict(df_all.iloc[test_idx][feature_cols], num_iteration=n_est)
                    df_all.iloc[test_idx, df_all.columns.get_loc("pred")] = preds

                valid_mask = ~df_all["pred"].isna()
                metrics = evaluator.evaluate(
                    y_true=df_all.loc[valid_mask, "target"],
                    y_pred=df_all.loc[valid_mask, "pred"],
                )
                results.append({"config_num": config_num, "n_est": n_est, "lr": lr, "depth": depth, "leaves": leaves, **metrics})
                print(f"   [{config_num:2d}] n={n_est:4d} lr={lr:.3f} d={depth} l={leaves:2d} → {metrics['score']:.1%} ({metrics['grade']})")

results_df = pd.DataFrame(results).sort_values(["num_passed", "score"], ascending=[False, False])
best = results_df.iloc[0]
print(f"\n✅ Best: {best['num_passed']}/7 ({best['score']:.1%}) — n={int(best['n_est'])}, lr={best['lr']}, d={int(best['depth'])}, l={int(best['leaves'])}")

# =============================================================================
# STEP 4: Train Final & Deploy
# =============================================================================
print("\n[5/5] Training final model...")
final_model = LGBMRegressor(
    n_estimators=int(best["n_est"]),
    learning_rate=best["lr"],
    max_depth=int(best["depth"]),
    num_leaves=int(best["leaves"]),
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_samples=50,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=123,
    verbose=-1,
)
final_model.fit(df_all[feature_cols], df_all["target"])
print(f"✅ Trained on {len(df_all):,} samples")


def predict(nonce=None):
    live_row = workflow.get_live_features(ticker=TICKERS[0])
    if live_row is None or len(live_row) == 0:
        raise ValueError("Could not get live features")
    live_eng = engineer_multiscale_features(live_row.iloc[0])
    live_features = pd.concat([live_row[base_feature_cols].iloc[0], live_eng])
    vol = final_model.predict(live_features[feature_cols].values.reshape(1, -1))[0]
    vol = max(0.0, float(vol))
    print(f"\nModel B prediction: {vol:.6f} (15-min vol)")
    return vol


print("\n🧪 Testing...")
test_pred = predict()

with open("predict_80_model_b.pkl", "wb") as f:
    cloudpickle.dump(predict, f)

print(f"\n✅ Saved predict_80_model_b.pkl")
print(f"   Score: {best['score']:.1%} | Features: {len(feature_cols)}")
