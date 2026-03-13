#!/usr/bin/env python3
"""Topic 37 tiny-bars ensemble search.

Build OOF predictions from top tiny-bar models, then test weighted blends.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
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


def model_registry():
    return [
        # strongest ridge variants from prior sweep
        ("b3_ridge_a1.0", 3, lambda: Ridge(alpha=1.0, random_state=42)),
        ("b4_ridge_a3.0", 4, lambda: Ridge(alpha=3.0, random_state=42)),
        ("b5_ridge_a3.0", 5, lambda: Ridge(alpha=3.0, random_state=42)),
        ("b6_ridge_a3.0", 6, lambda: Ridge(alpha=3.0, random_state=42)),
        ("b2_ridge_a0.1", 2, lambda: Ridge(alpha=0.1, random_state=42)),
        # diversity model
        ("b6_rf_n160_d12", 6, lambda: RandomForestRegressor(n_estimators=160, max_depth=12, min_samples_leaf=1, max_features="sqrt", n_jobs=-1, random_state=42)),
    ]


def get_df_for_bars(api_key: str, n_bars: int, days: int, skip_backfill: bool) -> pd.DataFrame:
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
            print(f"⚠️ backfill failed bars={n_bars}: {e}")

    df = wf.get_full_feature_target_dataframe(start_date=start).reset_index().sort_values("open_time").reset_index(drop=True)
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    keep = ["open_time", "target"] + feat_cols
    return df[keep].dropna().reset_index(drop=True)


def oof_predictions(df: pd.DataFrame, model_builder, splits: int) -> pd.DataFrame:
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    tscv = TimeSeriesSplit(n_splits=splits, gap=1)
    preds = pd.Series(np.nan, index=df.index)

    for tr_idx, te_idx in tscv.split(df):
        m = model_builder()
        m.fit(df.iloc[tr_idx][feat_cols], df.iloc[tr_idx]["target"])
        preds.iloc[te_idx] = m.predict(df.iloc[te_idx][feat_cols])

    out = df[["open_time", "target"]].copy()
    out["pred"] = preds
    return out.dropna().reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--skip-backfill", action="store_true")
    args = p.parse_args()

    evaluator = PerformanceEvaluator()
    api_key = (Path(__file__).parent / ".allora_api_key").read_text().strip()

    # Build OOF prediction matrix
    reg = model_registry()
    by_bars = {}
    for _, bars, _ in reg:
        if bars not in by_bars:
            print(f"Loading bars={bars}")
            by_bars[bars] = get_df_for_bars(api_key, bars, args.days, args.skip_backfill)

    mat = None
    base_reports = []
    for name, bars, builder in reg:
        print(f"OOF: {name}")
        oof = oof_predictions(by_bars[bars], builder, args.splits)
        oof = oof.rename(columns={"pred": name})
        if mat is None:
            mat = oof
        else:
            mat = mat.merge(oof[["open_time", name]], on="open_time", how="inner")

    model_cols = [c for c in mat.columns if c not in ["open_time", "target"]]
    y = mat["target"].to_numpy()

    for c in model_cols:
        k, rep = best_scale(y, mat[c].to_numpy(), evaluator)
        base_reports.append({
            "model": c,
            "k": k,
            "num_passed": rep["num_passed"],
            "score": rep["score"],
            "grade": rep["grade"],
            "metrics": rep["metrics"],
        })

    base_reports.sort(key=lambda r: (r["num_passed"], r["score"], r["metrics"]["czar_improvement"]), reverse=True)

    # Ensemble search: choose 2 or 3 models, nonnegative weights summing to 1, step=0.1
    ensemble_reports = []

    def weight_grid(n):
        vals = np.arange(0, 1.01, 0.1)
        for tup in itertools.product(vals, repeat=n):
            if abs(sum(tup) - 1.0) < 1e-9 and max(tup) < 1.0 and min(tup) > 0.0:
                yield tup

    for n_models in [2, 3]:
        for combo in itertools.combinations(model_cols, n_models):
            X = mat[list(combo)].to_numpy()
            for w in weight_grid(n_models):
                pred = np.dot(X, np.array(w))
                k, rep = best_scale(y, pred, evaluator)
                ensemble_reports.append({
                    "models": list(combo),
                    "weights": list(map(float, w)),
                    "k": k,
                    "num_passed": rep["num_passed"],
                    "score": rep["score"],
                    "grade": rep["grade"],
                    "metrics": rep["metrics"],
                })

    ensemble_reports.sort(key=lambda r: (r["num_passed"], r["score"], r["metrics"]["czar_improvement"]), reverse=True)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "days": args.days,
            "splits": args.splits,
            "n_rows_common": len(mat),
            "base_models": model_cols,
            "ensemble_sizes": [2, 3],
            "weight_step": 0.1,
        },
        "best_base": base_reports[0],
        "top5_base": base_reports[:5],
        "best_ensemble": ensemble_reports[0],
        "top10_ensembles": ensemble_reports[:10],
    }

    out_path = Path(__file__).parent / "artifacts" / "topic37_tinybars_ensemble_report.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    print("\n=== best base ===")
    print(json.dumps(payload["best_base"], indent=2, default=str))
    print("\n=== best ensemble ===")
    print(json.dumps(payload["best_ensemble"], indent=2, default=str))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
