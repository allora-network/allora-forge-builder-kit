#!/usr/bin/env python3
"""Topic 37 (SOLUSD, 5m) multi-model sweep.

Compares LightGBM, RandomForest, Ridge, and ElasticNet with walk-forward CV,
then saves ranked results + best model artifacts for deployment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import TimeSeriesSplit

from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator


@dataclass
class Candidate:
    name: str
    model_type: str
    params: dict


def make_candidates(quick: bool) -> list[Candidate]:
    if quick:
        return [
            Candidate("lgbm_a", "lgbm", {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "num_leaves": 31}),
            Candidate("rf_a", "rf", {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 5, "n_jobs": -1}),
            Candidate("ridge_a", "ridge", {"alpha": 1.0}),
            Candidate("enet_a", "enet", {"alpha": 0.0005, "l1_ratio": 0.2, "max_iter": 5000}),
        ]

    cands = []
    for n, lr, d, leaves in [(300, 0.05, 5, 31), (500, 0.03, 7, 63), (700, 0.02, 8, 127)]:
        cands.append(Candidate(f"lgbm_{n}_{lr}_{d}_{leaves}", "lgbm", {"n_estimators": n, "learning_rate": lr, "max_depth": d, "num_leaves": leaves}))

    for n, d, leaf in [(400, 10, 5), (700, 14, 3), (1000, None, 2)]:
        cands.append(Candidate(f"rf_{n}_{d}_{leaf}", "rf", {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf, "n_jobs": -1}))

    for alpha in [0.1, 1.0, 3.0, 10.0]:
        cands.append(Candidate(f"ridge_{alpha}", "ridge", {"alpha": alpha}))

    for alpha, l1 in [(0.0003, 0.1), (0.0005, 0.2), (0.001, 0.5), (0.003, 0.8)]:
        cands.append(Candidate(f"enet_{alpha}_{l1}", "enet", {"alpha": alpha, "l1_ratio": l1, "max_iter": 8000}))

    return cands


def build_model(c: Candidate):
    if c.model_type == "lgbm":
        return LGBMRegressor(random_state=42, verbose=-1, **c.params)
    if c.model_type == "rf":
        return RandomForestRegressor(random_state=42, **c.params)
    if c.model_type == "ridge":
        return Ridge(random_state=42, **c.params)
    if c.model_type == "enet":
        return ElasticNet(random_state=42, **c.params)
    raise ValueError(f"Unknown model_type={c.model_type}")


def engineer_returns(df: pd.DataFrame, n_input_bars: int) -> pd.DataFrame:
    closes = np.stack([df[f"feature_close_{i}"].to_numpy() for i in range(n_input_bars)], axis=1)
    eps = 1e-8
    out = pd.DataFrame(index=df.index)
    out["log_return_5m"] = np.log(closes[:, -1] + eps) - np.log(closes[:, -2] + eps)
    out["log_return_15m"] = np.where(n_input_bars >= 4, np.log(closes[:, -1] + eps) - np.log(closes[:, -4] + eps), 0.0)
    out["log_return_30m"] = np.where(n_input_bars >= 7, np.log(closes[:, -1] + eps) - np.log(closes[:, -7] + eps), 0.0)
    out["log_return_60m"] = np.where(n_input_bars >= 13, np.log(closes[:, -1] + eps) - np.log(closes[:, -13] + eps), 0.0)
    return out


def best_scale(y_true: np.ndarray, y_pred: np.ndarray, evaluator: PerformanceEvaluator) -> tuple[float, dict]:
    best_k, best_report = 1.0, evaluator.evaluate(y_true=y_true, y_pred=y_pred)
    for k in np.arange(0.6, 1.81, 0.05):
        r = evaluator.evaluate(y_true=y_true, y_pred=y_pred * k)
        if (r["num_passed"], r["score"], r["metrics"]["czar_improvement"]) > (
            best_report["num_passed"],
            best_report["score"],
            best_report["metrics"]["czar_improvement"],
        ):
            best_k, best_report = float(k), r
    return best_k, best_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-backfill", action="store_true")
    args = p.parse_args()

    ticker = "solusd"
    interval = "5m"
    n_input_bars = 48
    target_bars = 1

    workflow = AlloraMLWorkflow(
        tickers=[ticker],
        number_of_input_bars=n_input_bars,
        target_bars=target_bars,
        interval=interval,
        data_source="allora",
        api_key=(Path(__file__).parent / ".allora_api_key").read_text().strip(),
    )

    start = datetime.now(timezone.utc) - timedelta(days=args.days)
    if not args.skip_backfill:
        try:
            workflow.backfill(start=start)
        except Exception as e:
            print(f"⚠️ backfill failed ({e}); proceeding with cached parquet data")
    df = workflow.get_full_feature_target_dataframe(start_date=start).reset_index().sort_values("open_time").reset_index(drop=True)

    base_cols = [c for c in df.columns if c.startswith("feature_")]
    eng = engineer_returns(df, n_input_bars)
    df = pd.concat([df, eng], axis=1)
    feat_cols = base_cols + list(eng.columns)
    df = df.dropna(subset=feat_cols + ["target"]).reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=args.splits, gap=target_bars)
    evaluator = PerformanceEvaluator()

    rows = []
    candidates = make_candidates(args.quick)

    for i, c in enumerate(candidates, start=1):
        preds = pd.Series(np.nan, index=df.index)
        for tr_idx, te_idx in tscv.split(df):
            m = build_model(c)
            m.fit(df.iloc[tr_idx][feat_cols], df.iloc[tr_idx]["target"])
            preds.iloc[te_idx] = m.predict(df.iloc[te_idx][feat_cols])

        mask = ~preds.isna()
        y_true = df.loc[mask, "target"].to_numpy()
        y_pred = preds.loc[mask].to_numpy()
        k, rep = best_scale(y_true, y_pred, evaluator)
        rows.append(
            {
                "candidate": c.name,
                "model_type": c.model_type,
                "params": c.params,
                "k": k,
                "num_passed": rep["num_passed"],
                "score": rep["score"],
                "grade": rep["grade"],
                "metrics": rep["metrics"],
            }
        )
        print(f"[{i:02d}/{len(candidates)}] {c.name:<22} -> {rep['num_passed']}/8 | {rep['score']:.1%} | k={k:.2f}")

    ranked = sorted(rows, key=lambda r: (r["num_passed"], r["score"], r["metrics"]["czar_improvement"]), reverse=True)
    best = ranked[0]
    best_c = next(c for c in candidates if c.name == best["candidate"])

    final_model = build_model(best_c)
    final_model.fit(df[feat_cols], df["target"])

    out_dir = Path(__file__).parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "topic37_best_model.joblib"
    cols_path = out_dir / "topic37_columns.json"
    report_path = out_dir / "topic37_model_sweep_report.json"

    joblib.dump(final_model, model_path)
    with open(cols_path, "w") as f:
        json.dump({"base_feature_cols": base_cols, "feature_cols": feat_cols, "best_k": best["k"]}, f, indent=2)
    with open(report_path, "w") as f:
        json.dump(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": {"days": args.days, "splits": args.splits, "quick": args.quick},
                "data_rows": len(df),
                "best": best,
                "ranked": ranked,
            },
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
        )

    print("\nBest model:")
    summary = {
        "candidate": str(best["candidate"]),
        "num_passed": int(best["num_passed"]),
        "score": float(best["score"]),
        "grade": str(best["grade"]),
        "k": float(best["k"]),
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved: {model_path}")
    print(f"Saved: {cols_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
