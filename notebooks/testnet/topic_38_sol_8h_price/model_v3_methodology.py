#!/usr/bin/env python3
"""
Topic 38 — SOL/USD 8h Price Prediction — v3 (Methodology-Driven)
================================================================

Follows the 9 Principles from allora_research_model_skills:
- Principle 1: Every feature answers "what economic quantity does this estimate?"
- Principle 2: All windows derived from H=8 (horizon-adaptive)
- Principle 3: No lookahead — trailing windows only
- Principle 5: Huber loss (robust to fat tails)
- Principle 6: Purged walk-forward CV with gap >= max_feature_window

Key changes from v1/v2:
- DROP all 240 raw OHLCV base features (overfitting trap)
- Use ONLY ~25 engineered features organized by estimation goal
- Horizon-adaptive windows: [H/4, H/2, H, 2H, 5H, 10H] = [2, 4, 8, 16, 40, 80]
- All features normalized by rolling volatility
- Gap buffer = 80 bars (= max feature window)
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

# =============================================================================
# CONFIGURATION
# =============================================================================
TICKERS = ["solusd"]
DAYS_OF_HISTORY = 1825      # 5 years
INTERVAL = "1h"
NUMBER_OF_INPUT_BARS = 80   # 80h lookback (= 10H, max feature window)
TARGET_BARS = 8             # 8h ahead
H = TARGET_BARS             # horizon shorthand

# CV config
N_SPLITS = 5
GAP_BARS = 80               # purged gap = max feature window

# Grid search (smaller, focused)
N_ESTIMATORS_CHECKPOINTS = [200, 500]
LEARNING_RATES = [0.01, 0.03]
MAX_DEPTHS = [3, 5]
NUM_LEAVES = [15, 31]

print("=" * 70)
print("Topic 38 — SOL/USD 8h Price — v3 (Methodology-Driven)")
print("=" * 70)

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("\n[1/5] Loading data...")
from allora_forge_builder_kit.utils import get_api_key

api_key = get_api_key(
    api_key_file=os.path.join(os.path.dirname(__file__), "..", "..", ".allora_api_key")
)

workflow = AlloraMLWorkflow(
    tickers=TICKERS,
    number_of_input_bars=NUMBER_OF_INPUT_BARS,
    target_bars=TARGET_BARS,
    interval=INTERVAL,
    data_source="allora",
    api_key=api_key,
)

start_date = datetime.now(timezone.utc) - timedelta(days=DAYS_OF_HISTORY)
workflow.backfill(start=start_date)

df_all = workflow.get_full_feature_target_dataframe(start_date=start_date).reset_index()
base_feature_cols = [c for c in df_all.columns if c.startswith("feature_")]
df_all = df_all.dropna(subset=base_feature_cols + ["target"])
print(f"✅ Raw dataset: {len(df_all):,} samples")

# =============================================================================
# STEP 2: Horizon-Adaptive Feature Engineering
# =============================================================================
print("\n[2/5] Engineering horizon-adaptive features (H={H})...")

# Windows derived from H=8: [H/4, H/2, H, 2H, 5H, 10H]
WINDOWS = [2, 4, 8, 16, 40, 80]


def engineer_methodology_features(row):
    """
    ~25 features organized by estimation goal, all horizon-adaptive.
    Uses ONLY the base OHLCV features as raw inputs, then computes
    economically meaningful quantities.
    """
    n = NUMBER_OF_INPUT_BARS
    closes = np.array([row[f"feature_close_{i}"] for i in range(n)])
    highs = np.array([row[f"feature_high_{i}"] for i in range(n)])
    lows = np.array([row[f"feature_low_{i}"] for i in range(n)])
    volumes = np.array([row[f"feature_volume_{i}"] for i in range(n)])

    log_rets = np.diff(np.log(closes + 1e-12))
    abs_rets = np.abs(log_rets)
    f = {}

    # === ESTIMATION GOAL 1: Trend / Momentum ===
    # "What is the recent directional move at each timescale?"
    for w in WINDOWS:
        if len(log_rets) >= w:
            raw_ret = np.sum(log_rets[-w:])
            # Normalize by rolling vol at same window
            vol_w = np.std(log_rets[-w:], ddof=1) if w > 1 else abs_rets[-1]
            f[f"trend_{w}h"] = raw_ret / (vol_w * np.sqrt(w) + 1e-12)

    # === ESTIMATION GOAL 2: Mean Reversion ===
    # "How far is price from its recent average, in vol units?"
    for w in [8, 40, 80]:
        if len(closes) >= w:
            mean_w = np.mean(closes[-w:])
            std_w = np.std(closes[-w:], ddof=1)
            f[f"zscore_{w}h"] = (closes[-1] - mean_w) / (std_w + 1e-12)

    # === ESTIMATION GOAL 3: Volatility Regime ===
    # "What is the current vol level and is it expanding or contracting?"
    for w in [8, 40, 80]:
        if len(log_rets) >= w:
            f[f"vol_{w}h"] = np.std(log_rets[-w:], ddof=1)

    # Vol ratios (regime transitions)
    if f.get("vol_8h") and f.get("vol_40h"):
        f["vol_ratio_8_40"] = f["vol_8h"] / (f["vol_40h"] + 1e-12)
    if f.get("vol_8h") and f.get("vol_80h"):
        f["vol_ratio_8_80"] = f["vol_8h"] / (f["vol_80h"] + 1e-12)

    # === ESTIMATION GOAL 4: Market Microstructure ===
    # "What does the bid-ask spread proxy (HL range) tell us?"
    hl_range = highs - lows
    f["hl_range_8h"] = np.mean(hl_range[-8:])
    f["hl_range_ratio"] = np.mean(hl_range[-8:]) / (np.mean(hl_range[-40:]) + 1e-12) if len(hl_range) >= 40 else 1.0

    # === ESTIMATION GOAL 5: Volume Dynamics ===
    # "Is attention/liquidity increasing or decreasing?"
    f["vol_flow_ratio"] = np.mean(volumes[-8:]) / (np.mean(volumes[-40:]) + 1e-12) if len(volumes) >= 40 else 1.0
    f["vol_spike"] = np.max(volumes[-8:]) / (np.mean(volumes[-40:]) + 1e-12) if len(volumes) >= 40 else 1.0

    # === ESTIMATION GOAL 6: Trend Quality ===
    # "Is the move directional (trending) or choppy (mean-reverting)?"
    if len(log_rets) >= 8:
        net_move = abs(np.sum(log_rets[-8:]))
        total_path = np.sum(abs_rets[-8:])
        f["efficiency_8h"] = net_move / (total_path + 1e-12)

    if len(log_rets) >= 40:
        net_move = abs(np.sum(log_rets[-40:]))
        total_path = np.sum(abs_rets[-40:])
        f["efficiency_40h"] = net_move / (total_path + 1e-12)

    return pd.Series(f)


print("   Computing features...")
engineered = df_all.apply(engineer_methodology_features, axis=1)
df_all = pd.concat([df_all, engineered], axis=1)

# USE ONLY ENGINEERED FEATURES — drop raw base features
eng_cols = list(engineered.columns)
feature_cols = eng_cols  # NOT base_feature_cols + eng_cols
df_all = df_all.dropna(subset=feature_cols + ["target"])

print(f"✅ {len(df_all):,} samples | {len(feature_cols)} engineered features (no raw OHLCV)")
print(f"   Features: {feature_cols}")

# =============================================================================
# STEP 3: Purged Walk-Forward CV with Grid Search
# =============================================================================
print(f"\n[3/5] Grid search (purged CV, gap={GAP_BARS} bars)...")

tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP_BARS)
evaluator = PerformanceEvaluator()
results = []
config_num = 0

for lr in LEARNING_RATES:
    for depth in MAX_DEPTHS:
        for leaves in NUM_LEAVES:
            fold_models = []
            for train_idx, test_idx in tscv.split(df_all):
                lgb = LGBMRegressor(
                    objective="huber",
                    alpha=0.9,
                    n_estimators=500,
                    learning_rate=lr,
                    max_depth=depth,
                    num_leaves=leaves,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=100,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    random_state=42,
                    verbose=-1,
                )
                lgb.fit(
                    df_all.iloc[train_idx][feature_cols],
                    df_all.iloc[train_idx]["target"],
                )
                fold_models.append((lgb, test_idx))

            for n_est in N_ESTIMATORS_CHECKPOINTS:
                config_num += 1
                df_all["pred"] = np.nan
                for lgb, test_idx in fold_models:
                    preds = lgb.predict(
                        df_all.iloc[test_idx][feature_cols], num_iteration=n_est
                    )
                    df_all.iloc[test_idx, df_all.columns.get_loc("pred")] = preds

                valid = ~df_all["pred"].isna()
                y_true = df_all.loc[valid, "target"].values
                y_pred = df_all.loc[valid, "pred"].values

                # Compute our own metrics (Pearson, RMSE)
                r, _ = pearsonr(y_true, y_pred)
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

                # Also get the standard evaluator metrics
                eval_metrics = evaluator.evaluate(y_true=y_true, y_pred=y_pred)

                results.append({
                    "config_num": config_num,
                    "n_est": n_est,
                    "lr": lr,
                    "depth": depth,
                    "leaves": leaves,
                    "pearson_r": r,
                    "rmse": rmse,
                    **eval_metrics,
                })

                print(
                    f"   [{config_num:2d}] n={n_est:3d} lr={lr:.2f} d={depth} l={leaves:2d} "
                    f"→ r={r:+.4f} RMSE={rmse:.6f} "
                    f"DA={eval_metrics.get('metrics', eval_metrics).get('da', 0):.3f} "
                    f"({eval_metrics['num_passed']}/7)"
                )

# Rank by Pearson r (primary for log-return prediction)
results_df = pd.DataFrame(results).sort_values("pearson_r", ascending=False)
best = results_df.iloc[0]

print(f"\n✅ Best: r={best['pearson_r']:+.4f} RMSE={best['rmse']:.6f} "
      f"({best['num_passed']}/7) — n={int(best['n_est'])}, lr={best['lr']}, "
      f"d={int(best['depth'])}, l={int(best['leaves'])}")

# =============================================================================
# STEP 4: Detailed Evaluation
# =============================================================================
print("\n[4/5] Detailed evaluation...")
best_result = results[int(best["config_num"]) - 1]
evaluator.print_report(best_result, detailed=False)

# =============================================================================
# STEP 5: Train Final Model & Save
# =============================================================================
print("\n[5/5] Training final model...")

final_model = LGBMRegressor(
    objective="huber",
    alpha=0.9,
    n_estimators=int(best["n_est"]),
    learning_rate=best["lr"],
    max_depth=int(best["depth"]),
    num_leaves=int(best["leaves"]),
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=100,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    verbose=-1,
)
final_model.fit(df_all[feature_cols], df_all["target"])
print(f"✅ Trained on {len(df_all):,} samples, {len(feature_cols)} features")


def predict(nonce=None):
    """Predict SOL/USD price 8 hours ahead."""
    live_row = workflow.get_live_features(ticker=TICKERS[0])
    if live_row is None or len(live_row) == 0:
        raise ValueError("Could not get live features")

    live_eng = engineer_methodology_features(live_row.iloc[0])
    x = live_eng[feature_cols].values.reshape(1, -1)
    predicted_log_return = final_model.predict(x)[0]

    current_price = float(live_row.attrs.get("current_price", np.nan))
    if not np.isfinite(current_price) or current_price <= 0:
        snap = workflow._dm.get_live_snapshot(TICKERS)
        if snap is not None and len(snap) > 0 and "close" in snap.columns:
            current_price = float(snap["close"].iloc[-1])

    predicted_price = current_price * np.exp(predicted_log_return)
    print(f"\nPrediction: ${predicted_price:,.2f} ({predicted_log_return:+.6f} log return)")
    return float(predicted_price)


print("\n🧪 Testing prediction...")
test_pred = predict()

with open("predict_38.pkl", "wb") as f:
    cloudpickle.dump(predict, f)

print(f"\n✅ Saved predict_38.pkl")
print(f"   Pearson r: {best['pearson_r']:+.4f} | RMSE: {best['rmse']:.6f}")
print(f"   Features: {len(feature_cols)} (engineered only, no raw OHLCV)")
