#!/usr/bin/env python3
"""
Train Topic 69 BTC 24h model with a strict 1-year holdout evaluation.

Outputs:
- predict.pkl (callable for deployment)
- artifacts/topic69_year_holdout_report.json
- artifacts/topic69_year_holdout_preds.csv
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cloudpickle
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator


TICKER = "BTCUSDT"
INTERVAL = "1h"
NUMBER_OF_INPUT_BARS = 48
TARGET_BARS = 24
DATA_SOURCE = "binance"
MARKET = "futures"

HISTORY_DAYS = 900
HOLDOUT_DAYS = 365
N_SPLITS = 4

PARAM_GRID = [
    {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 5, "num_leaves": 31},
    {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 7, "num_leaves": 63},
    {"n_estimators": 600, "learning_rate": 0.02, "max_depth": 7, "num_leaves": 63},
    {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 5, "num_leaves": 31},
    {"n_estimators": 800, "learning_rate": 0.015, "max_depth": 8, "num_leaves": 127},
]


@dataclass
class ModelResult:
    params: dict
    cv_score: float
    cv_grade: str
    cv_num_passed: int


def engineer_returns(df: pd.DataFrame, n_input_bars: int) -> pd.DataFrame:
    closes = np.stack([df[f"feature_close_{i}"].to_numpy() for i in range(n_input_bars)], axis=1)
    eps = 1e-8
    out = pd.DataFrame(index=df.index)

    out["log_return_1h"] = np.log(closes[:, -1] + eps) - np.log(closes[:, -2] + eps)
    out["log_return_6h"] = np.where(
        n_input_bars >= 7,
        np.log(closes[:, -1] + eps) - np.log(closes[:, -7] + eps),
        0.0,
    )
    out["log_return_12h"] = np.where(
        n_input_bars >= 13,
        np.log(closes[:, -1] + eps) - np.log(closes[:, -13] + eps),
        0.0,
    )
    out["log_return_24h"] = np.where(
        n_input_bars >= 25,
        np.log(closes[:, -1] + eps) - np.log(closes[:, -25] + eps),
        0.0,
    )
    return out


def evaluate_cv(df_train: pd.DataFrame, feature_cols: list[str], params: dict, evaluator: PerformanceEvaluator) -> ModelResult:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=TARGET_BARS)
    fold_scores = []
    fold_passed = []
    fold_grade = []

    for tr_idx, va_idx in tscv.split(df_train):
        tr = df_train.iloc[tr_idx]
        va = df_train.iloc[va_idx]

        m = LGBMRegressor(random_state=42, verbose=-1, **params)
        m.fit(tr[feature_cols], tr["target"])
        pred = m.predict(va[feature_cols])

        metrics = evaluator.evaluate(y_true=va["target"], y_pred=pred)
        fold_scores.append(float(metrics["score"]))
        fold_passed.append(int(metrics["num_passed"]))
        fold_grade.append(metrics["grade"])

    return ModelResult(
        params=params,
        cv_score=float(np.mean(fold_scores)),
        cv_grade=max(set(fold_grade), key=fold_grade.count),
        cv_num_passed=int(np.round(np.mean(fold_passed))),
    )


def main() -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    workflow = AlloraMLWorkflow(
        tickers=[TICKER],
        number_of_input_bars=NUMBER_OF_INPUT_BARS,
        target_bars=TARGET_BARS,
        interval=INTERVAL,
        data_source=DATA_SOURCE,
        market=MARKET,
    )

    start = datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)
    workflow.backfill(start=start)

    df = workflow.get_full_feature_target_dataframe(start_date=start).reset_index()
    df = df.sort_values("open_time").reset_index(drop=True)

    base_feature_cols = [c for c in df.columns if c.startswith("feature_")]
    eng = engineer_returns(df, NUMBER_OF_INPUT_BARS)
    df = pd.concat([df, eng], axis=1)

    feature_cols = base_feature_cols + list(eng.columns)
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    holdout_start = datetime.now(timezone.utc) - timedelta(days=HOLDOUT_DAYS)
    df_train = df[df["open_time"] < holdout_start].copy()
    df_holdout = df[df["open_time"] >= holdout_start].copy()

    if len(df_train) < 2000 or len(df_holdout) < 500:
        raise RuntimeError(f"Insufficient data: train={len(df_train)}, holdout={len(df_holdout)}")

    evaluator = PerformanceEvaluator()

    cv_results = [evaluate_cv(df_train, feature_cols, params, evaluator) for params in PARAM_GRID]
    cv_results_sorted = sorted(cv_results, key=lambda r: (r.cv_score, r.cv_num_passed), reverse=True)
    best = cv_results_sorted[0]

    model = LGBMRegressor(random_state=42, verbose=-1, **best.params)
    model.fit(df_train[feature_cols], df_train["target"])

    holdout_pred = model.predict(df_holdout[feature_cols])
    holdout_metrics = evaluator.evaluate(y_true=df_holdout["target"], y_pred=holdout_pred)

    model_path = artifacts / "topic69_model.joblib"
    columns_path = artifacts / "topic69_columns.json"
    joblib.dump(model, model_path)
    with open(columns_path, "w") as f:
        json.dump({"base_feature_cols": base_feature_cols, "feature_cols": feature_cols}, f)

    # Persist prediction function for deploy_worker.py (no non-picklable closures)
    _runtime = {"workflow": None, "model": None, "base_feature_cols": None, "feature_cols": None}

    def predict(nonce: int | None = None) -> float:
        if _runtime["workflow"] is None:
            _runtime["workflow"] = AlloraMLWorkflow(
                tickers=[TICKER],
                number_of_input_bars=NUMBER_OF_INPUT_BARS,
                target_bars=TARGET_BARS,
                interval=INTERVAL,
                data_source=DATA_SOURCE,
                market=MARKET,
            )

        if _runtime["model"] is None:
            _runtime["model"] = joblib.load(model_path)
            with open(columns_path, "r") as f:
                cols = json.load(f)
            _runtime["base_feature_cols"] = cols["base_feature_cols"]
            _runtime["feature_cols"] = cols["feature_cols"]

        wf = _runtime["workflow"]
        model_local = _runtime["model"]
        base_cols = _runtime["base_feature_cols"]
        feat_cols = _runtime["feature_cols"]

        live_row = wf.get_live_features(ticker=TICKER)
        if live_row is None or len(live_row) == 0:
            raise ValueError("Could not get live features")

        live_eng = engineer_returns(live_row, NUMBER_OF_INPUT_BARS)
        live_features = pd.concat([live_row[base_cols].reset_index(drop=True), live_eng], axis=1)

        current_price = float(live_row.attrs.get("current_price", np.nan))
        if not np.isfinite(current_price) or current_price <= 0:
            raise ValueError("Could not fetch current price from live feature context")

        predicted_log_return = float(model_local.predict(live_features[feat_cols])[0])
        predicted_price = current_price * np.exp(predicted_log_return)
        return float(predicted_price)

    with open("predict.pkl", "wb") as f:
        cloudpickle.dump(predict, f)

    preds_df = df_holdout[["open_time", "target"]].copy()
    preds_df["prediction"] = holdout_pred
    preds_df.to_csv(artifacts / "topic69_year_holdout_preds.csv", index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "history_days": HISTORY_DAYS,
            "holdout_days": HOLDOUT_DAYS,
            "n_train": len(df_train),
            "n_holdout": len(df_holdout),
            "start": str(df["open_time"].min()),
            "end": str(df["open_time"].max()),
        },
        "best_cv": asdict(best),
        "holdout_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in holdout_metrics.items()},
    }

    with open(artifacts / "topic69_year_holdout_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({
        "best_cv": asdict(best),
        "holdout_grade": holdout_metrics.get("grade"),
        "holdout_score": holdout_metrics.get("score"),
        "holdout_num_passed": holdout_metrics.get("num_passed"),
    }, indent=2))


if __name__ == "__main__":
    main()
