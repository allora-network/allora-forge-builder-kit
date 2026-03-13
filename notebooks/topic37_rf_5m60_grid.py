#!/usr/bin/env python3
"""Topic 37 RF grid on best base geometry (5m, 60 bars)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator


@dataclass
class RFCandidate:
    name: str
    params: dict


def make_candidates(quick: bool) -> list[RFCandidate]:
    if quick:
        return [
            RFCandidate("rf_80_10_2_auto", {"n_estimators": 80, "max_depth": 10, "min_samples_leaf": 2, "max_features": "sqrt", "n_jobs": -1}),
            RFCandidate("rf_120_12_2_auto", {"n_estimators": 120, "max_depth": 12, "min_samples_leaf": 2, "max_features": "sqrt", "n_jobs": -1}),
            RFCandidate("rf_160_16_1_auto", {"n_estimators": 160, "max_depth": 16, "min_samples_leaf": 1, "max_features": "sqrt", "n_jobs": -1}),
            RFCandidate("rf_180_20_1_all", {"n_estimators": 180, "max_depth": 20, "min_samples_leaf": 1, "max_features": 1.0, "n_jobs": -1}),
            RFCandidate("rf_192_none_1_all", {"n_estimators": 192, "max_depth": None, "min_samples_leaf": 1, "max_features": 1.0, "n_jobs": -1}),
        ]

    out = []
    for n, d, leaf, mf in [
        (400, 12, 2, "sqrt"),
        (700, 16, 2, "sqrt"),
        (1000, None, 1, "sqrt"),
        (700, 20, 1, 1.0),
        (1200, None, 1, 1.0),
        (800, 24, 1, "log2"),
        (1000, 28, 1, "sqrt"),
        (1200, 32, 1, "sqrt"),
        (1000, None, 2, 1.0),
        (1400, None, 1, "sqrt"),
        (1400, None, 1, 1.0),
        (1600, None, 1, "sqrt"),
    ]:
        out.append(
            RFCandidate(
                f"rf_{n}_{d}_{leaf}_{mf}",
                {
                    "n_estimators": n,
                    "max_depth": d,
                    "min_samples_leaf": leaf,
                    "max_features": mf,
                    "n_jobs": -1,
                },
            )
        )
    return out


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=360)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--skip-backfill", action="store_true")
    p.add_argument("--full", action="store_true", help="Run larger RF parameter grid")
    p.add_argument("--max-candidates", type=int, default=0, help="Limit number of RF candidates (0 = all)")
    p.add_argument("--max-estimators", type=int, default=0, help="Keep only candidates with n_estimators < this value (0 = no filter)")
    args = p.parse_args()

    api_key = (Path(__file__).parent / ".allora_api_key").read_text().strip()

    wf = AlloraMLWorkflow(
        tickers=["solusd"],
        number_of_input_bars=60,
        target_bars=1,
        interval="5m",
        data_source="allora",
        api_key=api_key,
    )

    start = datetime.now(timezone.utc) - timedelta(days=args.days)
    if not args.skip_backfill:
        try:
            wf.backfill(start=start)
        except Exception as e:
            print(f"⚠️ backfill failed ({e}); using cached data")

    df = wf.get_full_feature_target_dataframe(start_date=start).reset_index().sort_values("open_time").reset_index(drop=True)
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    df = df.dropna(subset=feat_cols + ["target"]).reset_index(drop=True)

    print(f"rows={len(df):,} features={len(feat_cols)}")

    tscv = TimeSeriesSplit(n_splits=args.splits, gap=1)
    evaluator = PerformanceEvaluator()
    candidates = make_candidates(quick=not args.full)
    if args.max_estimators and args.max_estimators > 0:
        candidates = [c for c in candidates if int(c.params.get("n_estimators", 0)) < args.max_estimators]
    if args.max_candidates and args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    if not candidates:
        raise ValueError("No RF candidates left after filters; relax --max-estimators/--max-candidates")

    ranked = []
    for i, cand in enumerate(candidates, start=1):
        preds = pd.Series(np.nan, index=df.index)
        for tr_idx, te_idx in tscv.split(df):
            m = RandomForestRegressor(random_state=42, **cand.params)
            m.fit(df.iloc[tr_idx][feat_cols], df.iloc[tr_idx]["target"])
            preds.iloc[te_idx] = m.predict(df.iloc[te_idx][feat_cols])

        mask = ~preds.isna()
        y_true = df.loc[mask, "target"].to_numpy()
        y_pred = preds.loc[mask].to_numpy()
        k, rep = best_scale(y_true, y_pred, evaluator)

        row = {
            "candidate": cand.name,
            "params": cand.params,
            "k": k,
            "num_passed": rep["num_passed"],
            "score": rep["score"],
            "grade": rep["grade"],
            "metrics": rep["metrics"],
        }
        ranked.append(row)
        print(f"[{i:02d}/{len(candidates)}] {cand.name} pass={rep['num_passed']}/8 score={rep['score']:.1%} DA={rep['metrics']['directional_accuracy']:.4f}")

    ranked.sort(key=lambda r: (r["num_passed"], r["score"], r["metrics"]["czar_improvement"]), reverse=True)
    best = ranked[0]

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "days": args.days,
            "splits": args.splits,
            "interval": "5m",
            "number_of_input_bars": 60,
            "quick": not args.full,
            "n_candidates": len(candidates),
        },
        "rows": len(df),
        "n_features": len(feat_cols),
        "best": best,
        "ranked": ranked,
    }

    out_path = Path(__file__).parent / "artifacts" / "topic37_rf_5m60_grid_report.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    print("\n=== best RF ===")
    print(json.dumps(best, indent=2, default=str))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
