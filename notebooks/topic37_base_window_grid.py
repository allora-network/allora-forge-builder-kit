#!/usr/bin/env python3
"""Topic 37 base-feature window geometry grid.

Sweeps interval + number_of_input_bars using base normalized OHLCV features only,
with a compact Ridge grid. Writes a single report artifact (overwritten each run).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator


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


def run_combo(api_key: str, days: int, splits: int, interval: str, n_bars: int, alphas: list[float], skip_backfill: bool) -> dict:
    workflow = AlloraMLWorkflow(
        tickers=["solusd"],
        number_of_input_bars=n_bars,
        target_bars=1,
        interval=interval,
        data_source="allora",
        api_key=api_key,
    )

    start = datetime.now(timezone.utc) - timedelta(days=days)
    if not skip_backfill:
        try:
            workflow.backfill(start=start)
        except Exception as e:
            print(f"⚠️ backfill failed for {interval}/{n_bars}: {e}")

    df = workflow.get_full_feature_target_dataframe(start_date=start).reset_index().sort_values("open_time").reset_index(drop=True)
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    df = df.dropna(subset=feat_cols + ["target"]).reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=splits, gap=1)
    evaluator = PerformanceEvaluator()

    ranked = []
    for alpha in alphas:
        preds = pd.Series(np.nan, index=df.index)
        for tr_idx, te_idx in tscv.split(df):
            m = Ridge(alpha=alpha, random_state=42)
            m.fit(df.iloc[tr_idx][feat_cols], df.iloc[tr_idx]["target"])
            preds.iloc[te_idx] = m.predict(df.iloc[te_idx][feat_cols])

        mask = ~preds.isna()
        y_true = df.loc[mask, "target"].to_numpy()
        y_pred = preds.loc[mask].to_numpy()
        k, rep = best_scale(y_true, y_pred, evaluator)
        ranked.append({
            "alpha": alpha,
            "k": k,
            "num_passed": rep["num_passed"],
            "score": rep["score"],
            "grade": rep["grade"],
            "metrics": rep["metrics"],
        })

    ranked.sort(key=lambda r: (r["num_passed"], r["score"], r["metrics"]["czar_improvement"]), reverse=True)
    best = ranked[0]
    return {
        "interval": interval,
        "n_bars": n_bars,
        "n_features": len(feat_cols),
        "rows": len(df),
        "best": best,
        "ranked": ranked,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=360)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--skip-backfill", action="store_true")
    p.add_argument("--quick", action="store_true", help="Run smaller geometry grid")
    args = p.parse_args()

    if args.quick:
        geometry = {
            "5m": [24, 48],
            "15m": [16, 32],
            "1h": [8, 16],
        }
    else:
        geometry = {
            "5m": [24, 48, 96],
            "15m": [16, 32, 64],
            "30m": [12, 24, 48],
            "1h": [8, 16, 32],
        }

    alphas = [0.1, 0.3, 1.0, 3.0]
    api_key = (Path(__file__).parent / ".allora_api_key").read_text().strip()

    results = []
    for interval, bars_list in geometry.items():
        for n_bars in bars_list:
            print(f"\n=== interval={interval} bars={n_bars} ===")
            out = run_combo(
                api_key=api_key,
                days=args.days,
                splits=args.splits,
                interval=interval,
                n_bars=n_bars,
                alphas=alphas,
                skip_backfill=args.skip_backfill,
            )
            b = out["best"]
            print(
                f"best alpha={b['alpha']} pass={b['num_passed']}/8 score={b['score']:.1%} "
                f"DA={b['metrics']['directional_accuracy']:.4f} rows={out['rows']:,}"
            )
            results.append(out)

    overall = sorted(
        [
            {
                "interval": r["interval"],
                "n_bars": r["n_bars"],
                "n_features": r["n_features"],
                "rows": r["rows"],
                **r["best"],
            }
            for r in results
        ],
        key=lambda r: (r["num_passed"], r["score"], r["metrics"]["czar_improvement"]),
        reverse=True,
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "days": args.days,
            "splits": args.splits,
            "geometry": geometry,
            "alphas": alphas,
        },
        "best_overall": overall[0] if overall else None,
        "overall_ranked": overall,
        "by_combo": results,
    }

    out_path = Path(__file__).parent / "artifacts" / "topic37_base_window_grid_report.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    print("\n=== best overall ===")
    print(json.dumps(payload["best_overall"], indent=2, default=str))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
