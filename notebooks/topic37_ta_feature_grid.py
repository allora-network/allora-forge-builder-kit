#!/usr/bin/env python3
"""Topic 37 (SOLUSD, 5m) TA-feature prototype grid.

Lightweight first-pass experiment:
- keeps builder-kit base normalized OHLCV features
- adds derived TA-style features on top
- compares a small Ridge/ElasticNet grid across feature sets

Use this to quickly identify promising feature families before running a larger search.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
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
            Candidate("ridge_0.3", "ridge", {"alpha": 0.3}),
            Candidate("ridge_1.0", "ridge", {"alpha": 1.0}),
            Candidate("ridge_3.0", "ridge", {"alpha": 3.0}),
            Candidate("enet_0.0003_0.1", "enet", {"alpha": 0.0003, "l1_ratio": 0.1, "max_iter": 8000}),
            Candidate("enet_0.001_0.3", "enet", {"alpha": 0.001, "l1_ratio": 0.3, "max_iter": 8000}),
        ]

    out = [Candidate(f"ridge_{a}", "ridge", {"alpha": a}) for a in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]]
    out += [
        Candidate(f"enet_{a}_{l1}", "enet", {"alpha": a, "l1_ratio": l1, "max_iter": 10000})
        for a in [0.0001, 0.0003, 0.001, 0.003]
        for l1 in [0.05, 0.1, 0.2, 0.35]
    ]
    return out


def build_model(c: Candidate):
    if c.model_type == "ridge":
        return Ridge(random_state=42, **c.params)
    if c.model_type == "enet":
        return ElasticNet(random_state=42, **c.params)
    raise ValueError(c.model_type)


def _mat(df: pd.DataFrame, key: str, n: int) -> np.ndarray:
    return np.stack([df[f"feature_{key}_{i}"].to_numpy(dtype=float) for i in range(n)], axis=1)


def engineer_ta_features(df: pd.DataFrame, n_input_bars: int) -> pd.DataFrame:
    """Build TA-style derived features from normalized base OHLCV features."""
    eps = 1e-12
    o = _mat(df, "open", n_input_bars)
    h = _mat(df, "high", n_input_bars)
    l = _mat(df, "low", n_input_bars)
    c = _mat(df, "close", n_input_bars)
    v = _mat(df, "volume", n_input_bars)

    def ema_last(arr: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1.0)
        e = arr[:, 0].copy()
        for i in range(1, arr.shape[1]):
            e = alpha * arr[:, i] + (1.0 - alpha) * e
        return e

    out = pd.DataFrame(index=df.index)

    # Basic returns/momentum in normalized price space
    out["ret_1"] = np.log(c[:, -1] + eps) - np.log(c[:, -2] + eps)
    out["ret_3"] = np.log(c[:, -1] + eps) - np.log(c[:, -4] + eps)
    out["ret_6"] = np.log(c[:, -1] + eps) - np.log(c[:, -7] + eps)
    out["ret_12"] = np.log(c[:, -1] + eps) - np.log(c[:, -13] + eps)

    # SMA spreads (proxy for MA crossover pressure)
    sma5 = c[:, -5:].mean(axis=1)
    sma10 = c[:, -10:].mean(axis=1)
    sma20 = c[:, -20:].mean(axis=1)
    out["sma5_sma10_spread"] = sma5 / (sma10 + eps) - 1.0
    out["sma10_sma20_spread"] = sma10 / (sma20 + eps) - 1.0
    out["close_sma20_spread"] = c[:, -1] / (sma20 + eps) - 1.0

    # Crossover flags (localized strategy signals)
    prev_sma5 = c[:, -6:-1].mean(axis=1)
    prev_sma20 = c[:, -21:-1].mean(axis=1)
    out["cross_up_5_20"] = ((sma5 > sma20) & (prev_sma5 <= prev_sma20)).astype(float)
    out["cross_down_5_20"] = ((sma5 < sma20) & (prev_sma5 >= prev_sma20)).astype(float)

    # RSI-like momentum (Wilder-lite using simple means over recent deltas)
    d = np.diff(c, axis=1)
    d14 = d[:, -14:]
    gain = np.maximum(d14, 0.0).mean(axis=1)
    loss = (-np.minimum(d14, 0.0)).mean(axis=1)
    rs = gain / (loss + eps)
    out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    out["rsi14_centered"] = (out["rsi14"] - 50.0) / 50.0

    # Volatility / range structure
    tr1 = h[:, -1] - l[:, -1]
    tr2 = np.abs(h[:, -1] - c[:, -2])
    tr3 = np.abs(l[:, -1] - c[:, -2])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    out["tr_last"] = tr
    out["range_5_mean"] = (h[:, -5:] - l[:, -5:]).mean(axis=1)
    out["range_20_mean"] = (h[:, -20:] - l[:, -20:]).mean(axis=1)
    out["range_ratio_5_20"] = out["range_5_mean"] / (out["range_20_mean"] + eps) - 1.0

    # Bollinger-style position
    mu20 = sma20
    sd20 = c[:, -20:].std(axis=1)
    out["boll_z_20"] = (c[:, -1] - mu20) / (sd20 + eps)

    # MACD family (12, 26, 9)
    ema12 = ema_last(c, 12)
    ema26 = ema_last(c, 26)
    macd_line = ema12 - ema26
    # lightweight signal approximation from last 9 closes transformed similarly
    c9 = c[:, -9:]
    ema12_9 = ema_last(c9, 12)
    ema26_9 = ema_last(c9, 26)
    macd_recent = ema12_9 - ema26_9
    macd_signal = 0.5 * macd_line + 0.5 * macd_recent
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal

    # Stochastic oscillator (%K, %D)
    ll14 = l[:, -14:].min(axis=1)
    hh14 = h[:, -14:].max(axis=1)
    stoch_k = (c[:, -1] - ll14) / (hh14 - ll14 + eps)
    out["stoch_k_14"] = stoch_k
    out["stoch_d_3"] = np.stack([
        (c[:, -i] - l[:, -(13+i):-(i-1) if i > 1 else None].min(axis=1)) /
        (h[:, -(13+i):-(i-1) if i > 1 else None].max(axis=1) - l[:, -(13+i):-(i-1) if i > 1 else None].min(axis=1) + eps)
        for i in [1, 2, 3]
    ], axis=1).mean(axis=1)

    # Trend slope over last 10 closes
    x = np.arange(10, dtype=float)
    x_centered = x - x.mean()
    y = c[:, -10:]
    y_centered = y - y.mean(axis=1, keepdims=True)
    slope = (y_centered * x_centered).sum(axis=1) / ((x_centered ** 2).sum() + eps)
    out["trend_slope_10"] = slope

    # Volume behavior (still normalized by last volume)
    out["vol_recent_mean"] = v[:, -5:].mean(axis=1)
    out["vol_rel_5_20"] = v[:, -5:].mean(axis=1) / (v[:, -20:].mean(axis=1) + eps) - 1.0

    # Simple divergence proxy: price momentum minus RSI momentum
    out["divergence_proxy"] = out["ret_6"] - (out["rsi14_centered"] * 0.01)

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


def run_experiment(df: pd.DataFrame, feature_cols: list[str], candidates: list[Candidate], splits: int):
    tscv = TimeSeriesSplit(n_splits=splits, gap=1)
    evaluator = PerformanceEvaluator()
    rows = []

    for c in candidates:
        preds = pd.Series(np.nan, index=df.index)
        for tr_idx, te_idx in tscv.split(df):
            m = build_model(c)
            m.fit(df.iloc[tr_idx][feature_cols], df.iloc[tr_idx]["target"])
            preds.iloc[te_idx] = m.predict(df.iloc[te_idx][feature_cols])

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
    ranked = sorted(rows, key=lambda r: (r["num_passed"], r["score"], r["metrics"]["czar_improvement"]), reverse=True)
    return ranked


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--splits", type=int, default=3)
    p.add_argument("--full", action="store_true", help="Run larger model grid (default is lightweight prototype)")
    p.add_argument("--skip-backfill", action="store_true")
    p.add_argument(
        "--feature-sets",
        type=str,
        default="base,base_plus_ta,ta_only,base_plus_ta_flags",
        help="Comma-separated feature sets to run",
    )
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
            print(f"⚠️ backfill failed ({e}); proceeding with cached data")

    df = workflow.get_full_feature_target_dataframe(start_date=start).reset_index().sort_values("open_time").reset_index(drop=True)
    base_cols = [c for c in df.columns if c.startswith("feature_")]
    ta_df = engineer_ta_features(df, n_input_bars=n_input_bars)
    df = pd.concat([df, ta_df], axis=1)

    ta_cols = list(ta_df.columns)
    ta_flag_cols = [
        "cross_up_5_20",
        "cross_down_5_20",
        "rsi14_centered",
        "boll_z_20",
        "range_ratio_5_20",
        "divergence_proxy",
    ]

    feature_sets = {
        "base": base_cols,
        "base_plus_ta": base_cols + ta_cols,
        "ta_only": ta_cols,
        "base_plus_ta_flags": base_cols + ta_flag_cols,
    }

    selected_sets = [s.strip() for s in args.feature_sets.split(",") if s.strip()]
    for s in selected_sets:
        if s not in feature_sets:
            raise ValueError(f"Unknown feature set: {s}. Available: {list(feature_sets.keys())}")

    needed_cols = set(["target"])
    for s in selected_sets:
        needed_cols.update(feature_sets[s])
    df = df.dropna(subset=list(needed_cols)).reset_index(drop=True)

    quick_mode = not args.full
    candidates = make_candidates(quick=quick_mode)

    all_results = []
    print(f"rows={len(df):,} candidates={len(candidates)} feature_sets={selected_sets}")
    for fs_name in selected_sets:
        fcols = feature_sets[fs_name]
        print(f"\n=== feature_set={fs_name} n_features={len(fcols)} ===")
        ranked = run_experiment(df=df, feature_cols=fcols, candidates=candidates, splits=args.splits)
        best = ranked[0]
        print(
            f"best={best['candidate']} pass={best['num_passed']}/8 score={best['score']:.1%} "
            f"DA={best['metrics']['directional_accuracy']:.4f}"
        )
        all_results.append(
            {
                "feature_set": fs_name,
                "n_features": len(fcols),
                "best": best,
                "ranked": ranked,
            }
        )

    overall = sorted(
        [
            {
                "feature_set": r["feature_set"],
                "n_features": r["n_features"],
                **r["best"],
            }
            for r in all_results
        ],
        key=lambda r: (r["num_passed"], r["score"], r["metrics"]["czar_improvement"]),
        reverse=True,
    )

    out_dir = Path(__file__).parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "topic37_ta_feature_grid_report.json"

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "days": args.days,
            "splits": args.splits,
            "quick": bool(quick_mode),
            "feature_sets": selected_sets,
            "n_candidates": len(candidates),
        },
        "data_rows": len(df),
        "best_overall": overall[0] if overall else None,
        "by_feature_set": all_results,
        "feature_columns": {
            "base": base_cols,
            "ta": ta_cols,
            "ta_flags": ta_flag_cols,
        },
    }

    with open(report_path, "w") as f:
        json.dump(
            payload,
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
        )

    print("\n=== best overall ===")
    print(json.dumps(payload["best_overall"], indent=2, default=str))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
