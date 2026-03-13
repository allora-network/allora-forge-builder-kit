#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator

TICKER = "BTCUSDT"
INTERVAL = "1h"
N_INPUT = 48
TARGET_BARS = 24
HISTORY_DAYS = 900
HOLDOUT_DAYS = 365


def engineer_returns(df: pd.DataFrame, n_input_bars: int) -> pd.DataFrame:
    closes = np.stack([df[f"feature_close_{i}"].to_numpy() for i in range(n_input_bars)], axis=1)
    highs = np.stack([df[f"feature_high_{i}"].to_numpy() for i in range(n_input_bars)], axis=1)
    lows = np.stack([df[f"feature_low_{i}"].to_numpy() for i in range(n_input_bars)], axis=1)
    vols = np.stack([df[f"feature_volume_{i}"].to_numpy() for i in range(n_input_bars)], axis=1)
    eps = 1e-8

    out = pd.DataFrame(index=df.index)
    out["log_return_1h"] = np.log(closes[:, -1] + eps) - np.log(closes[:, -2] + eps)
    out["log_return_6h"] = np.log(closes[:, -1] + eps) - np.log(closes[:, -7] + eps)
    out["log_return_24h"] = np.log(closes[:, -1] + eps) - np.log(closes[:, -25] + eps)
    out["range_mean"] = np.mean(highs - lows, axis=1)
    out["range_std"] = np.std(highs - lows, axis=1)
    out["vol_mean"] = np.mean(vols, axis=1)
    out["vol_std"] = np.std(vols, axis=1)
    out["momentum_12"] = closes[:, -1] - closes[:, -13]
    out["momentum_24"] = closes[:, -1] - closes[:, -25]
    return out


def metrics_summary(metrics: dict) -> dict:
    m = metrics["metrics"]
    return {
        "grade": str(metrics["grade"]),
        "score": float(metrics["score"]),
        "num_passed": int(metrics["num_passed"]),
        "directional_accuracy": float(m["directional_accuracy"]),
        "pearson_r": float(m["pearson_r"]),
        "wrmse_improvement": float(m["wrmse_improvement"]),
        "czar_improvement": float(m["czar_improvement"]),
        "rmse": float(m["rmse"]),
    }


def main() -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    wf = AlloraMLWorkflow(
        tickers=[TICKER],
        number_of_input_bars=N_INPUT,
        target_bars=TARGET_BARS,
        interval=INTERVAL,
        data_source="binance",
        market="futures",
    )

    # Use cached data built earlier; backfill only if missing
    start = datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)
    try:
        df = wf.get_full_feature_target_dataframe(start_date=start).reset_index()
    except Exception:
        wf.backfill(start=start)
        df = wf.get_full_feature_target_dataframe(start_date=start).reset_index()

    df = df.sort_values("open_time").reset_index(drop=True)
    base_cols = [c for c in df.columns if c.startswith("feature_")]
    eng = engineer_returns(df, N_INPUT)
    df = pd.concat([df, eng], axis=1)

    feature_sets = {
        "base": base_cols,
        "base_plus_returns": base_cols + ["log_return_1h", "log_return_6h", "log_return_24h"],
        "base_plus_stats": base_cols + [
            "log_return_1h", "log_return_6h", "log_return_24h", "range_mean", "range_std", "vol_mean", "vol_std", "momentum_12", "momentum_24"
        ],
    }

    df = df.dropna(subset=list(set(sum(feature_sets.values(), [])) | {"target"})).reset_index(drop=True)

    holdout_start = datetime.now(timezone.utc) - timedelta(days=HOLDOUT_DAYS)
    tr = df[df["open_time"] < holdout_start].copy()
    ho = df[df["open_time"] >= holdout_start].copy()

    evaluator = PerformanceEvaluator()
    rows = []

    model_specs = [
        ("linreg", LinearRegression()),
        ("ridge_1", Ridge(alpha=1.0, random_state=42)),
        ("knn_40", Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=40, weights="distance"))])),
        ("rf_shallow", RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=42, n_jobs=-1)),
        ("gbm_shallow", GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42)),
        ("lgbm_small", LGBMRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, num_leaves=15, random_state=42, verbose=-1)),
        ("lgbm_med", LGBMRegressor(n_estimators=250, learning_rate=0.03, max_depth=5, num_leaves=31, random_state=42, verbose=-1)),
        ("svd_lin", Pipeline([
            ("scaler", StandardScaler()),
            ("svd", TruncatedSVD(n_components=40, random_state=42)),
            ("lin", Ridge(alpha=1.0, random_state=42)),
        ])),
    ]

    for fs_name, cols in feature_sets.items():
        Xtr, ytr = tr[cols], tr["target"]
        Xho, yho = ho[cols], ho["target"]

        for model_name, model in model_specs:
            model.fit(Xtr, ytr)
            pred_tr = model.predict(Xtr)
            pred_ho = model.predict(Xho)
            met_ho = evaluator.evaluate(y_true=yho, y_pred=pred_ho)
            rmse_tr = float(np.sqrt(mean_squared_error(ytr, pred_tr)))
            rmse_ho = float(np.sqrt(mean_squared_error(yho, pred_ho)))

            row = {
                "feature_set": fs_name,
                "model": model_name,
                "rmse_train": rmse_tr,
                "rmse_holdout": rmse_ho,
                "generalization_gap": rmse_ho - rmse_tr,
                **metrics_summary(met_ho),
            }
            rows.append(row)
            print(f"{fs_name:18s} | {model_name:12s} | grade={row['grade']:2s} score={row['score']:.3f} passed={row['num_passed']} gap={row['generalization_gap']:.5f}")

    out = pd.DataFrame(rows).sort_values(["score", "num_passed", "pearson_r"], ascending=False)
    out.to_csv(artifacts / "topic69_model_sweep.csv", index=False)

    best = out.iloc[0].to_dict()
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(tr)),
        "n_holdout": int(len(ho)),
        "top5": out.head(5).to_dict(orient="records"),
        "best": best,
    }
    with open(artifacts / "topic69_model_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
