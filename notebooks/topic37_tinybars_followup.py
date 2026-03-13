#!/usr/bin/env python3
"""Topic 37 follow-up sweep around tiny-bar sweet spot (2..6 bars).

Models:
- Ridge
- ElasticNet
- Small RandomForest
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import TimeSeriesSplit

from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator


def best_scale(y_true: np.ndarray, y_pred: np.ndarray, evaluator: PerformanceEvaluator):
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


def make_model_specs():
    specs = []
    for a in [0.03, 0.1, 0.3, 1.0, 3.0]:
        specs.append((f"ridge_a{a}", lambda a=a: Ridge(alpha=a, random_state=42), {"alpha": a}))

    for a in [0.0005, 0.001, 0.003, 0.01]:
        for l1 in [0.2, 0.5, 0.8]:
            specs.append((
                f"enet_a{a}_l1{l1}",
                lambda a=a, l1=l1: ElasticNet(alpha=a, l1_ratio=l1, random_state=42, max_iter=20000),
                {"alpha": a, "l1_ratio": l1},
            ))

    rf_params = [
        {"n_estimators": 80, "max_depth": 8, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 120, "max_depth": 10, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 160, "max_depth": 12, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    ]
    for p in rf_params:
        name = f"rf_n{p['n_estimators']}_d{p['max_depth']}_leaf{p['min_samples_leaf']}"
        specs.append((
            name,
            lambda p=p: RandomForestRegressor(random_state=42, n_jobs=-1, **p),
            p,
        ))

    return specs


def run_for_bars(api_key: str, days: int, splits: int, n_bars: int, skip_backfill: bool):
    wf = AlloraMLWorkflow(
        tickers=["solusd"],
        number_of_input_bars=n_bars,
        target_bars=1,
        interval="5m",
        data_source="allora",
        api_key=api_key,
    )

    start = datetime.now(timezone.utc) - timedelta(days=days)
    if not skip_backfill:
        try:
            wf.backfill(start=start)
        except Exception as e:
            print(f"⚠️ backfill failed for bars={n_bars}: {e}")

    df = (
        wf.get_full_feature_target_dataframe(start_date=start)
        .reset_index()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    df = df.dropna(subset=feat_cols + ["target"]).reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=splits, gap=1)
    evaluator = PerformanceEvaluator()

    ranked = []
    specs = make_model_specs()

    for i, (name, build_model, params) in enumerate(specs, start=1):
        preds = pd.Series(np.nan, index=df.index)
        for tr_idx, te_idx in tscv.split(df):
            m = build_model()
            m.fit(df.iloc[tr_idx][feat_cols], df.iloc[tr_idx]["target"])
            preds.iloc[te_idx] = m.predict(df.iloc[te_idx][feat_cols])

        mask = ~preds.isna()
        y_true = df.loc[mask, "target"].to_numpy()
        y_pred = preds.loc[mask].to_numpy()
        k, rep = best_scale(y_true, y_pred, evaluator)

        ranked.append(
            {
                "model": name,
                "params": params,
                "k": k,
                "num_passed": rep["num_passed"],
                "score": rep["score"],
                "grade": rep["grade"],
                "metrics": rep["metrics"],
            }
        )
        print(
            f"[{n_bars} bars {i:02d}/{len(specs)}] {name} "
            f"pass={rep['num_passed']}/8 score={rep['score']:.1%} "
            f"DA={rep['metrics']['directional_accuracy']:.4f}"
        )

    ranked.sort(key=lambda r: (r["num_passed"], r["score"], r["metrics"]["czar_improvement"]), reverse=True)

    return {
        "interval": "5m",
        "n_bars": n_bars,
        "n_features": len(feat_cols),
        "rows": len(df),
        "best": ranked[0],
        "ranked": ranked,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--skip-backfill", action="store_true")
    args = p.parse_args()

    bars_grid = [2, 3, 4, 5, 6]
    api_key = (Path(__file__).parent / ".allora_api_key").read_text().strip()

    by_bars = []
    for n_bars in bars_grid:
        print(f"\n=== bars={n_bars} ===")
        by_bars.append(run_for_bars(api_key, args.days, args.splits, n_bars, args.skip_backfill))

    overall_ranked = sorted(
        [
            {
                "interval": x["interval"],
                "n_bars": x["n_bars"],
                "n_features": x["n_features"],
                "rows": x["rows"],
                **x["best"],
            }
            for x in by_bars
        ],
        key=lambda r: (r["num_passed"], r["score"], r["metrics"]["czar_improvement"]),
        reverse=True,
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "days": args.days,
            "splits": args.splits,
            "bars_grid": bars_grid,
            "model_families": ["ridge", "elasticnet", "random_forest_small"],
        },
        "best_overall": overall_ranked[0],
        "overall_ranked": overall_ranked,
        "by_bars": by_bars,
    }

    out_path = Path(__file__).parent / "artifacts" / "topic37_tinybars_followup_report.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    print("\n=== best overall ===")
    print(json.dumps(payload["best_overall"], indent=2, default=str))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
