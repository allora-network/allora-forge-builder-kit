#!/usr/bin/env python3
"""
Topic 79 — Model A: Deep Lookback + Rich Volatility Features
=============================================================

Strategy: Use a 60-bar (1-hour) lookback window with extensive volatility-
predictive features including multi-horizon realised vol, return autocorrelation,
Parkinson/Garman-Klass estimators, and volume-volatility interaction.

Trained on 2+ years of 1-minute BTC/USD data.
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
TICKERS = ["btcusd"]
DAYS_OF_HISTORY = 800  # ~2.2 years
INTERVAL = "1m"
NUMBER_OF_INPUT_BARS = 60  # 1 hour of 1-min bars
TARGET_BARS = 15
TARGET_TYPE = "volatility"

N_SPLITS = 5
N_ESTIMATORS_MAX = 1000
N_ESTIMATORS_CHECKPOINTS = [200, 500, 800, 1000]
LEARNING_RATES = [0.01, 0.03]
MAX_DEPTHS = [5, 7]
NUM_LEAVES = [31, 63]

print("=" * 80)
print("Topic 79 — Model A: Deep Lookback (60-bar, 2+ years)")
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


def engineer_deep_vol_features(row):
    """Rich volatility features from 60-bar lookback."""
    n = NUMBER_OF_INPUT_BARS
    closes = np.array([row[f"feature_close_{i}"] for i in range(n)])
    highs = np.array([row[f"feature_high_{i}"] for i in range(n)])
    lows = np.array([row[f"feature_low_{i}"] for i in range(n)])
    volumes = np.array([row[f"feature_volume_{i}"] for i in range(n)])

    log_rets = np.diff(np.log(closes + 1e-12))
    features = {}

    # --- Multi-horizon realised volatility ---
    features["vol_5m"] = np.std(log_rets[-5:], ddof=1) if len(log_rets) >= 5 else 0.0
    features["vol_10m"] = np.std(log_rets[-10:], ddof=1) if len(log_rets) >= 10 else 0.0
    features["vol_15m"] = np.std(log_rets[-15:], ddof=1) if len(log_rets) >= 15 else 0.0
    features["vol_30m"] = np.std(log_rets[-30:], ddof=1) if len(log_rets) >= 30 else 0.0
    features["vol_60m"] = np.std(log_rets, ddof=1) if len(log_rets) >= 2 else 0.0

    # --- Vol ratios (regime detection) ---
    features["vol_ratio_5_60"] = features["vol_5m"] / (features["vol_60m"] + 1e-12)
    features["vol_ratio_15_60"] = features["vol_15m"] / (features["vol_60m"] + 1e-12)
    features["vol_ratio_5_30"] = features["vol_5m"] / (features["vol_30m"] + 1e-12)

    # --- Return autocorrelation (vol clustering signal) ---
    if len(log_rets) >= 10:
        features["ret_autocorr_1"] = np.corrcoef(log_rets[1:], log_rets[:-1])[0, 1]
        abs_rets = np.abs(log_rets)
        features["absret_autocorr_1"] = np.corrcoef(abs_rets[1:], abs_rets[:-1])[0, 1]
    else:
        features["ret_autocorr_1"] = 0.0
        features["absret_autocorr_1"] = 0.0

    # Handle NaN from corrcoef
    for k in ["ret_autocorr_1", "absret_autocorr_1"]:
        if not np.isfinite(features[k]):
            features[k] = 0.0

    # --- Parkinson volatility estimator (uses high-low) ---
    hl_log = np.log(highs + 1e-12) - np.log(lows + 1e-12)
    features["parkinson_vol_15m"] = np.sqrt(np.mean(hl_log[-15:] ** 2) / (4 * np.log(2)))
    features["parkinson_vol_60m"] = np.sqrt(np.mean(hl_log ** 2) / (4 * np.log(2)))
    features["parkinson_ratio"] = features["parkinson_vol_15m"] / (features["parkinson_vol_60m"] + 1e-12)

    # --- High-low range features ---
    hl_range = highs - lows
    features["hl_range_mean"] = np.mean(hl_range)
    features["hl_range_5m"] = np.mean(hl_range[-5:])
    features["hl_range_ratio"] = features["hl_range_5m"] / (features["hl_range_mean"] + 1e-12)
    features["hl_range_max"] = np.max(hl_range[-15:])

    # --- Absolute returns (magnitude) ---
    abs_rets = np.abs(log_rets)
    features["abs_ret_mean_5m"] = np.mean(abs_rets[-5:])
    features["abs_ret_mean_15m"] = np.mean(abs_rets[-15:])
    features["abs_ret_max_15m"] = np.max(abs_rets[-15:])
    features["abs_ret_mean_60m"] = np.mean(abs_rets)

    # --- Volume-volatility interaction ---
    features["volume_mean_ratio"] = np.mean(volumes[-5:]) / (np.mean(volumes) + 1e-12)
    features["volume_spike"] = np.max(volumes[-5:]) / (np.mean(volumes) + 1e-12)

    # Volume-weighted volatility
    vol_weights = volumes[1:] / (np.sum(volumes[1:]) + 1e-12)
    features["vol_weighted_absret"] = np.sum(abs_rets * vol_weights)

    # --- Trend strength (directional move vs vol) ---
    net_return = log_rets[-15:].sum() if len(log_rets) >= 15 else 0.0
    features["trend_vs_vol"] = abs(net_return) / (features["vol_15m"] + 1e-12)

    # --- Kurtosis (tail risk) ---
    if len(log_rets) >= 20:
        mean_r = np.mean(log_rets[-30:])
        std_r = np.std(log_rets[-30:], ddof=1)
        if std_r > 1e-12:
            features["kurtosis_30m"] = np.mean(((log_rets[-30:] - mean_r) / std_r) ** 4)
        else:
            features["kurtosis_30m"] = 3.0
    else:
        features["kurtosis_30m"] = 3.0

    return pd.Series(features)


print("   Engineering deep volatility features...")
engineered = df_all.apply(engineer_deep_vol_features, axis=1)
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
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
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
                print(f"   [{config_num:2d}] n={n_est:4d} lr={lr:.2f} d={depth} l={leaves:2d} → {metrics['score']:.1%} ({metrics['grade']})")

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
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)
final_model.fit(df_all[feature_cols], df_all["target"])
print(f"✅ Trained on {len(df_all):,} samples")


def predict(nonce=None):
    live_row = workflow.get_live_features(ticker=TICKERS[0])
    if live_row is None or len(live_row) == 0:
        raise ValueError("Could not get live features")
    live_eng = engineer_deep_vol_features(live_row.iloc[0])
    live_features = pd.concat([live_row[base_feature_cols].iloc[0], live_eng])
    vol = final_model.predict(live_features[feature_cols].values.reshape(1, -1))[0]
    vol = max(0.0, float(vol))
    print(f"\nModel A prediction: {vol:.6f} (15-min vol)")
    return vol


print("\n🧪 Testing...")
test_pred = predict()

with open("predict_79_model_a.pkl", "wb") as f:
    cloudpickle.dump(predict, f)

print(f"\n✅ Saved predict_79_model_a.pkl")
print(f"   Score: {best['score']:.1%} | Features: {len(feature_cols)}")
