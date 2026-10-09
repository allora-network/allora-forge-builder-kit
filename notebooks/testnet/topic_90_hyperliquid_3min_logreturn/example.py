#!/usr/bin/env python3
"""Topic 90: pooled Hyperliquid perp log-return walkthrough.

Train and inspect interactively, or pass --worker to launch the testnet SDK worker.
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
from allora_forge_builder_kit import AlloraMLWorkflow, AtlasDataManager, PerformanceEvaluator
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import requests
from sklearn.model_selection import ParameterSampler
import gc
import json
import joblib
import time
import asyncio
import os
import sys

TOPIC_ID = 90
INTERVAL = "3m"
TARGET_BARS = 1
LOOKBACK = 10  # Number of recent candles; five base OHLCV features per candle.
HISTORY_MONTHS = 6
SEARCH_TRIALS = 12
EXPORT_TOP_MODELS = 3
ESTIMATOR_CHECKPOINTS = [1, 5, 10, 20, 30, 50, 100, 200, 400, 800, 1600, 3200]
VALIDATION_FRACTION = 0.20
RANDOM_STATE = 42

# Randomly sample SEARCH_TRIALS combinations; score every tree checkpoint per fit.
PARAMETER_GRID = {
    "max_bin": [5, 9, 13, 17],
    "learning_rate": [0.0001, 0.0005, 0.001, 0.002, 0.005],
    "max_depth": [3, 4, 5,6,7,8],
    "num_leaves": [15, 31, 63, 128],
    "min_child_samples": [500, 1000, 2000, 5000, 10000],
    "colsample_bytree": [0.5, 0.75],
    "subsample": [0.8, 1.0],
    "reg_alpha": [5.0, 10.0, 20.0],
    "reg_lambda": [5.0, 10.0, 20.0],
}

# 1. Configure and discover the training universe
api_key = os.environ.get("ALLORA_API_KEY", "").strip()
if not api_key:
    key_file = Path(".allora_api_key")
    api_key = key_file.read_text().strip() if key_file.is_file() else ""
if not api_key:
    raise ValueError("Set ALLORA_API_KEY or put a non-empty API key in .allora_api_key")
base_dir = Path(__file__).parent
data_dir = base_dir / "data"
data_dir.mkdir(exist_ok=True)
atlas = AtlasDataManager(api_key=api_key, base_dir=data_dir, interval=INTERVAL)
hl_universe = atlas.discover_hl_universe()
# Fixed bounds for this run; exclude incomplete 3m candles from training.
end_date = pd.Timestamp.now(tz="UTC").floor("3min")
start_date = end_date - pd.DateOffset(months=HISTORY_MONTHS)

print(
    f"Discovered {len(hl_universe)} Hyperliquid perp datasets. \nUsing history from {start_date}.\nData Directory: {data_dir}"
)

# 2. Reuse cached history; pass --backfill explicitly to fetch missing data.
workflow = AlloraMLWorkflow(
    tickers=hl_universe, target_type="log_return", interval=INTERVAL,
    target_bars=TARGET_BARS, number_of_input_bars=LOOKBACK, data_manager=atlas,
)
if "--backfill" in sys.argv or "--backfill-only" in sys.argv:
    workflow.backfill(start=start_date)
    if "--backfill-only" in sys.argv:
        print("Historical backfill complete.")
        raise SystemExit(0)
else:
    print("Reusing cached minute data; historical backfill skipped.")

# Build one asset at a time to avoid materializing all history as float64.
# Track the actual next observed bar's close time, including gaps, for the split.
parts = []
for ticker in hl_universe:
    workflow.tickers = [ticker]
    frame = workflow.get_full_feature_target_dataframe(start_date=start_date, end_date=end_date)
    feature_columns = [c for c in frame.columns if c.startswith("feature_")]
    target_end = pd.Series(frame.index.get_level_values("open_time"), index=frame.index).shift(-TARGET_BARS) + pd.Timedelta(INTERVAL)
    frame = frame[feature_columns + ["target"]].astype("float32")
    valid = np.isfinite(frame.to_numpy()).all(axis=1) & (target_end <= end_date)
    frame = frame.loc[valid].copy()
    frame["target_end"] = target_end.loc[valid]
    if not frame.empty:
        parts.append(frame)
workflow.tickers = hl_universe
if not parts:
    raise RuntimeError("No complete training rows after backfill")
data = pd.concat(parts)
del parts, frame
feature_columns = [c for c in data.columns if c.startswith("feature_")]
print(f"Training dataset: {len(data):,} rows across {data.index.get_level_values('ticker').nunique()} assets")

# 3. One chronological holdout; sklearn random sampling, shared tree checkpoints.
# RandomizedSearchCV would refit every n_estimators candidate. ParameterSampler
# samples the other parameters; a callback scores/saves prefixes of ONE fit.
def search_tree_checkpoints(X_train, y_train, X_valid, y_valid, candidates, checkpoints, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(set(checkpoints))
    if not checkpoints or checkpoints[0] < 1:
        raise ValueError("Tree checkpoints must be positive")
    records = []
    for trial, params in enumerate(candidates, start=1):
        def save_checkpoint(env):
            trees = env.iteration + 1
            if trees not in checkpoints:
                return
            predicted = env.model.predict(X_valid, num_iteration=trees, num_threads=params.get("n_jobs", 1))
            correlation = float(np.corrcoef(y_valid, predicted)[0, 1]) if np.std(predicted) > 0 and np.std(y_valid) > 0 else None
            mse = float(np.mean((np.asarray(y_valid, dtype=float) - predicted) ** 2))
            rmse = float(np.sqrt(mse))
            checkpoint = directory / f"trial_{trial:03d}_trees_{trees:04d}.txt"
            env.model.save_model(str(checkpoint), num_iteration=trees)
            records.append(dict(trial=trial, n_estimators=trees, fitted_trees=env.model.current_iteration(),
                                correlation=correlation, mse=mse, rmse=rmse, validation_loss=mse, checkpoint=str(checkpoint), params=dict(params)))
            (directory / "checkpoints.json").write_text(json.dumps(records, indent=2, allow_nan=False))
            print(f"Trial {trial}: trees={trees}, MSE={mse:.8g}, correlation={correlation}, RMSE={rmse:.6g}", flush=True)
        save_checkpoint.order = 30
        save_checkpoint.before_iteration = False
        candidate = lgb.LGBMRegressor(**params, n_estimators=max(checkpoints))
        candidate.fit(X_train, y_train, callbacks=[save_checkpoint])  # No early stopping.
        del candidate
        gc.collect()
    scored = [r for r in records if np.isfinite(r["validation_loss"])]
    if not scored:
        raise RuntimeError("No checkpoint has a finite validation loss")
    best = min(scored, key=lambda r: (r["validation_loss"], r["n_estimators"], r["trial"]))
    (directory / "best_validation.json").write_text(json.dumps(best, indent=2))
    return best


all_dates = data.index.get_level_values("open_time").unique().sort_values()
validation_start = all_dates[int(len(all_dates) * (1 - VALIDATION_FRACTION))]
# Purge labels that resolve after the validation boundary (even across gaps).
train_mask = data["target_end"] <= validation_start
valid_mask = data.index.get_level_values("open_time") >= validation_start
if not train_mask.any() or not valid_mask.any():
    raise RuntimeError("Insufficient history for the chronological holdout")
y = data.pop("target")
data.pop("target_end")
search_dir = base_dir / "results" / "model_search" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
search_dir.mkdir(parents=True, exist_ok=True)

candidates = [dict(params, objective="regression", random_state=RANDOM_STATE,
                   subsample_freq=1, n_jobs=min(8, os.cpu_count() or 1), verbosity=-1)
              for params in ParameterSampler(PARAMETER_GRID, n_iter=SEARCH_TRIALS, random_state=RANDOM_STATE)]
metadata = dict(start_date=str(start_date), end_date=str(end_date), validation_start=str(validation_start),
                training_rows=int(train_mask.sum()), validation_rows=int(valid_mask.sum()),
                all_rows=len(data), training_universe=sorted(set(data.index.get_level_values("ticker"))),
                feature_columns=feature_columns, interval=INTERVAL, lookback=LOOKBACK,
                target_bars=TARGET_BARS, selection_metric="lowest validation MSE", worker_objective="mse",
                estimator_checkpoints=ESTIMATOR_CHECKPOINTS, candidates=candidates)
(search_dir / "run.json").write_text(json.dumps(metadata, indent=2))
print(f"Holdout starts {validation_start}; {train_mask.sum():,} train / {valid_mask.sum():,} validation rows. Search: {search_dir}")
best = search_tree_checkpoints(
    data.loc[train_mask], y.loc[train_mask], data.loc[valid_mask], y.loc[valid_mask],
    candidates, ESTIMATOR_CHECKPOINTS, search_dir)

def fit_positive_wasserstein_scale(prediction, target):
    """Fit the exact positive scale minimizing empirical 1-D Wasserstein-1."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(prediction) & np.isfinite(target)
    prediction = np.sort(prediction[finite])
    target = np.sort(target[finite])
    if len(prediction) == 0:
        return 1.0, float('nan'), float('nan')

    # For equal-size samples, W1 is the mean distance between sorted values.
    # |s*p_i-y_i| = |p_i|*|s-y_i/p_i|, so the minimizer is the weighted
    # median of the ratios, constrained here to be strictly positive.
    usable = np.abs(prediction) > np.finfo(np.float64).eps
    if not usable.any():
        raw_w1 = float(np.mean(np.abs(target)))
        return 1.0, raw_w1, raw_w1
    ratios = target[usable] / prediction[usable]
    weights = np.abs(prediction[usable])
    order = np.argsort(ratios)
    ratios, weights = ratios[order], weights[order]
    midpoint = weights.sum() / 2
    scale = float(ratios[np.searchsorted(np.cumsum(weights), midpoint)])
    scale = max(scale, 1e-8)
    raw_w1 = float(np.mean(np.abs(prediction - target)))
    scaled_w1 = float(np.mean(np.abs(scale * prediction - target)))
    return scale, raw_w1, scaled_w1

def report_per_asset(evaluator, truth, predictions, directory, label):
    """Equal-weight descriptive summaries; no inferred aggregate eligibility or CI."""
    frame = pd.DataFrame({"target": truth, "prediction": predictions}, index=truth.index)
    metrics_rows, criterion_rows = [], []
    for ticker, group in frame.groupby(level="ticker", sort=True):
        group = group.sort_index(level="open_time")
        result = evaluator.evaluate(group.target.to_numpy(), group.prediction.to_numpy(), epoch_length_minutes=3)
        metrics_rows.append(dict(ticker=ticker, rows=len(group), **{
            key: value for key, value in result["metrics"].items()
            if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))}))
        criterion_rows.extend(dict(ticker=ticker, **criterion) for criterion in result["criteria"])
    per_asset = pd.DataFrame(metrics_rows).set_index("ticker")
    numeric = per_asset.drop(columns="rows").apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    summary = numeric.agg(["mean", "median", "count"]).T.rename(columns={"count": "assets_contributing"})
    summary["assets_total"] = len(per_asset)
    criteria = pd.DataFrame(criterion_rows)
    pass_rates = criteria.groupby("key")["passed"].agg(["mean", "sum", "count"]).rename(
        columns={"mean": "fraction_assets_passing", "sum": "assets_passing", "count": "assets_total"})
    per_asset.to_csv(directory / f"validation_per_asset_{label}.csv")
    summary.to_csv(directory / f"validation_asset_summary_{label}.csv")
    criteria.to_csv(directory / f"validation_asset_criteria_{label}.csv", index=False)
    pass_rates.to_csv(directory / f"validation_asset_pass_rates_{label}.csv")
    print(f"Equal-weight asset metrics ({label}); descriptive averages, no aggregate eligibility verdict")
    print("Averaged CI bounds in the CSV are descriptive only, not confidence intervals for the mean.")
    print(summary.loc[["directional_accuracy", "pearson_r", "wrmse_improvement", "czar_improvement", "log_aspect_ratio", "mse", "rmse"]].to_string())
    print("Per-asset criterion pass rates (offline participation assumed):")
    print(pass_rates.to_string())
    return per_asset, summary, pass_rates


def select_top_trials(records, count=3):
    """Take each trial's lowest-MSE checkpoint, then rank those trial winners."""
    winners = {}
    for record in records:
        if not np.isfinite(record["mse"]):
            continue
        prior = winners.get(record["trial"])
        if prior is None or (record["mse"], record["n_estimators"]) < (prior["mse"], prior["n_estimators"]):
            winners[record["trial"]] = record
    return sorted(winners.values(), key=lambda r: (r["mse"], r["n_estimators"], r["trial"]))[:count]


# Export three separately calibrated models; each trial contributes at most one.
selected_models = select_top_trials(json.loads((search_dir / "checkpoints.json").read_text()), EXPORT_TOP_MODELS)
exports = []
for rank, best in enumerate(selected_models, 1):
    model_dir = search_dir / f"model_{rank}"
    model_dir.mkdir(exist_ok=True)
    print(f"Exporting rank {rank}: trial {best['trial']}, {best['n_estimators']} trees")
    # Official metrics on the selected holdout checkpoint, before full-data refitting.
    # This is validation-selected performance, not an independent test or portfolio return.
    validation_model = lgb.Booster(model_file=best["checkpoint"])
    validation_predictions = validation_model.predict(data.loc[valid_mask], num_threads=best["params"]["n_jobs"])
    training_predictions = validation_model.predict(data.loc[train_mask], num_threads=best["params"]["n_jobs"])
    prediction_scale, training_w1_raw, training_w1_scaled = fit_positive_wasserstein_scale(
        training_predictions, y.loc[train_mask].to_numpy())
    calibration = dict(method="positive_wasserstein_scale", fitted_on="training partition",
                       scale=prediction_scale, training_w1_raw=training_w1_raw, training_w1_scaled=training_w1_scaled)
    (model_dir / "return_calibration.json").write_text(json.dumps(calibration, indent=2))
    print(f"Training-fitted return scale: {prediction_scale:.6g}")
    del training_predictions
    evaluator = PerformanceEvaluator(target_type="log_return")
    report = evaluator.evaluate(y.loc[valid_mask].to_numpy(), validation_predictions, epoch_length_minutes=3)
    print("Official builder-kit metrics: pooled assets, selected validation checkpoint")
    evaluator.print_report(report)
    (model_dir / "validation_performance.json").write_text(json.dumps(
        dict(split="validation",selection="minimum validation MSE",scope="pooled assets",report=report),
        indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()))
    scaled_report = evaluator.evaluate(y.loc[valid_mask].to_numpy(), validation_predictions * prediction_scale, epoch_length_minutes=3)
    print("Official builder-kit metrics: validation with frozen training-fitted return scale")
    evaluator.print_report(scaled_report)
    (model_dir / "validation_performance_scaled.json").write_text(json.dumps(
        dict(split="validation",scope="pooled assets",calibration=calibration,report=scaled_report), indent=2))
    report_per_asset(evaluator, y.loc[valid_mask], validation_predictions, model_dir, "raw")
    report_per_asset(evaluator, y.loc[valid_mask], validation_predictions * prediction_scale, model_dir, "scaled")
    del validation_model, validation_predictions

    # Refit the selected settings on ALL eligible data. The live worker uses this model.
    # The validation score belongs to the saved validation checkpoint, not this refit.
    gc.collect()
    model = lgb.LGBMRegressor(**best["params"], n_estimators=best["n_estimators"])
    model.fit(data, y)  # No early stopping and no historical test fold; live evaluation follows.
    joblib.dump(dict(model=model, metadata=metadata, best_validation=best, calibration=calibration), model_dir / "model.joblib")
    model.booster_.save_model(str(model_dir / "model.txt"))
    print(f"Refit complete: {best['n_estimators']} trees on {len(data):,} rows; saved {model_dir / 'model.joblib'}")
    exports.append(dict(rank=rank, trial=best["trial"], validation_mse=best["mse"],
                        directory=model_dir.name, model=f"{model_dir.name}/model.joblib"))
    del model
    gc.collect()
(search_dir / "top_models.json").write_text(json.dumps(exports, indent=2))
# Keep rank one in memory for interactive inference and the optional single worker.
primary = joblib.load(search_dir / exports[0]["model"])
model, calibration, best = primary["model"], primary["calibration"], primary["best_validation"]
prediction_scale = calibration["scale"]
del primary


# 3b. Rank live perps by approximate 30-day dollar volume (training unchanged).
# discover_hl_universe() selects status="consumable", market_group="perp".
# One bulk daily request at startup, never inside the inference callback.
def rank_hl_volume(atlas, symbols):
    if not symbols or len(symbols) * 30 > 10_000:
        raise ValueError("Daily ranking must fit one Atlas bulk request (30 rows per asset)")
    response = requests.get(
        f"{atlas.base_url}/rows/", headers=atlas.headers,
        params={"dataset_names": ",".join(symbols), "resolution": "1d",
                "limit": 30, "ordering": "-timestamp"}, timeout=60,
    )
    response.raise_for_status()
    groups = response.json()["datasets"]
    if len(groups) != len(symbols):
        raise ValueError("Incomplete Atlas daily response")
    records = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    for symbol, group in zip(symbols, groups):
        bars = pd.DataFrame(group["rows"])
        if bars.empty:
            continue
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars = bars[bars["timestamp"] >= cutoff].drop_duplicates("timestamp")
        # Candle volume is in base units. Daily close gives approximate USD notional,
        # not exact trade-level turnover; missing days are not extrapolated.
        dollars = pd.to_numeric(bars["volume"], errors="coerce") * pd.to_numeric(bars["close"], errors="coerce")
        valid = np.isfinite(dollars) & (dollars >= 0)
        if not valid.any():
            continue
        records.append(dict(symbol=symbol, label=symbol.removeprefix("hl_").removesuffix("_1min").lower(),
                            volume_usd=dollars[valid].sum(), daily_bars=int(valid.sum()),
                            first_day=bars.loc[valid, "timestamp"].min(), last_day=bars.loc[valid, "timestamp"].max()))
    if not records:
        raise RuntimeError("No usable daily bars for volume ranking")
    ranked = pd.DataFrame(records).sort_values(["volume_usd", "symbol"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["above_20m"] = ranked["volume_usd"] >= 20_000_000
    return ranked


volume_ranking = rank_hl_volume(atlas, atlas.discover_hl_universe())
volume_ranked_labels = volume_ranking["label"].tolist()
print(f"Volume ranking: {len(volume_ranking)} assets; {volume_ranking.above_20m.sum()} at or above $20M (reference only).")
print(f"Daily coverage: {volume_ranking.first_day.min()} to {volume_ranking.last_day.max()}; "
      f"{volume_ranking.daily_bars.min()}–{volume_ranking.daily_bars.max()} bars per asset (up to 30 requested).")

# Export a readable ranked chart and its data, including incomplete coverage.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

output_dir = base_dir / "results"
output_dir.mkdir(exist_ok=True)
volume_ranking.to_csv(output_dir / "hl_30day_volume_ranking.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(16, 24), sharex=True)
midpoint = (len(volume_ranking) + 1) // 2
for ax, part in zip(axes, [volume_ranking.iloc[:midpoint], volume_ranking.iloc[midpoint:]]):
    ax.barh(np.arange(len(part)), part.volume_usd.clip(lower=1),
            color=np.where(part["rank"] <= 100, "#2878b5", "#adb5bd"))
    ax.set_yticks(np.arange(len(part)), [f"{r.rank:3}. {r.label.upper()}" for r in part.itertuples()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.axvline(20_000_000, color="#c65b13", linestyle="--", linewidth=1.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1e9:g}B" if v >= 1e9 else f"${v/1e6:g}M" if v >= 1e6 else f"${v/1e3:g}K"))
    ax.set_xlabel("Approximate dollar volume (log scale)")
    ax.grid(axis="x", alpha=0.2)
fig.suptitle("Hyperliquid native perps — approximate 30-day volume ranking", fontsize=18, y=0.995)
fig.text(0.5, 0.974, f"Available daily bars: {volume_ranking.first_day.min():%Y-%m-%d} to {volume_ranking.last_day.max():%Y-%m-%d} | "
         f"{volume_ranking.daily_bars.min()}–{volume_ranking.daily_bars.max()} bars/asset, up to 30 requested\n"
         "Sum of daily base volume × daily close; incomplete days/history are not extrapolated. $20M is a reference, not a submission filter.",
         ha="center", fontsize=10)
fig.legend(handles=[Patch(color="#2878b5", label="Top 100 by volume"), Patch(color="#adb5bd", label="Rank >100"),
                    Line2D([0], [0], color="#c65b13", linestyle="--", label="$20M reference cutoff")],
           loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.945))
fig.savefig(output_dir / "hl_30day_volume_ranking.png", dpi=160)
fig.savefig(output_dir / "hl_30day_volume_ranking.pdf")
plt.close(fig)
print("Volume chart:", output_dir / "hl_30day_volume_ranking.png")


# 4. Prepare in-memory minute history and runtime startup

class LiveMinuteBuffer:
    """Process-local raw candles; no live-data disk reads or writes.

    start() bootstraps before scheduling the failsafe. Inference calls refresh()
    and then snapshot(). A successful empty response counts as a refresh: no
    trades is different from an HTTP failure. Missing minutes stay missing.
    """

    def __init__(self, atlas, symbols, lookback=80, interval_minutes=3,
                 margin_minutes=20, overlap_minutes=4, check_seconds=300):
        from threading import Event, Lock

        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("Provide a non-empty, unique list of Atlas dataset names")
        if min(lookback, interval_minutes, margin_minutes, overlap_minutes, check_seconds) <= 0:
            raise ValueError("Buffer durations and lookback must be positive")
        self.atlas = atlas
        self.symbols = list(symbols)
        self.history_minutes = lookback * interval_minutes + margin_minutes
        self.overlap_minutes = overlap_minutes
        self.check_seconds = check_seconds
        self._state_lock = Lock()
        self._refresh_lock = Lock()
        self._stop = Event()
        self._thread = None
        index = pd.MultiIndex.from_arrays(
            [[], pd.DatetimeIndex([], tz="UTC")], names=["symbol", "open_time"]
        )
        self._candles = pd.DataFrame(
            index=index, columns=["open", "high", "low", "close", "volume"], dtype=float
        )
        self.last_successful_refresh = {}  # Request coverage, not last trade time.
        self.last_errors = {}

    def _merge(self, incoming, cutoff, end):
        """Overwrite overlapping timestamps and retain only the bounded window."""
        with self._state_lock:
            if not incoming.empty:
                times = incoming.index.get_level_values("open_time")
                incoming = incoming.loc[(times >= cutoff) & (times <= end)]
                self._candles = pd.concat([self._candles, incoming])
                self._candles = self._candles.loc[
                    ~self._candles.index.duplicated(keep="last")
                ]
            times = self._candles.index.get_level_values("open_time")
            self._candles = self._candles.loc[times >= cutoff].sort_index()

    def refresh(self, *, only_if_needed=False, timeout=10.0):
        """Catch up from the last successful query with overlapping 20m chunks.

        Normally one request for the whole universe; bootstrap/long outages use
        multiple bounded requests. The lock serializes inference and watchdog
        refreshes. This is not yet an inference-deadline scheduler.
        """
        from time import monotonic

        started = monotonic()
        with self._refresh_lock:
            now = datetime.now(timezone.utc)
            end = pd.Timestamp(now).floor("min").to_pydatetime()
            cutoff = end - timedelta(minutes=self.history_minutes)
            with self._state_lock:
                due = [s for s in self.symbols if not only_if_needed or
                       s not in self.last_successful_refresh or
                       (now - self.last_successful_refresh[s]).total_seconds() >= self.check_seconds]
                checked = dict(self.last_successful_refresh)
            self._merge(pd.DataFrame(), cutoff, end)
            if not due:
                return {"requested": 0, "rows_received": 0, "errors": {}, "seconds": monotonic() - started}

            # 21 inclusive minute openings per 20m chunk; each request <=10k rows.
            group_size = min(1000, 10_000 // 21)
            received = 0
            errors = {}
            succeeded = []

            def fetch(group, start, chunk_end, limit):
                try:
                    return self.atlas.get_bulk_1min_candles(
                        group, limit=limit, start=start, end=chunk_end, timeout=timeout
                    )
                except requests.RequestException as exc:
                    status = getattr(exc.response, "status_code", None)
                    if status in (403, 404) and len(group) > 1:
                        # One inaccessible dataset must not block the others.
                        middle = len(group) // 2
                        parts = [fetch(group[:middle], start, chunk_end, limit),
                                 fetch(group[middle:], start, chunk_end, limit)]
                        return pd.concat(parts)
                    for symbol in group:
                        errors[symbol] = str(exc)
                    return pd.DataFrame()

            for offset in range(0, len(due), group_size):
                group = due[offset:offset + group_size]
                start = min(max(cutoff, checked.get(s, cutoff) -
                                timedelta(minutes=self.overlap_minutes)) for s in group)
                start = pd.Timestamp(start).floor("min").to_pydatetime()
                while start <= end:
                    chunk_end = min(start + timedelta(minutes=20), end)
                    limit = int((chunk_end - start).total_seconds() // 60) + 1
                    rows = fetch(group, start, chunk_end, limit)
                    self._merge(rows, cutoff, end)
                    received += len(rows)
                    group = [symbol for symbol in group if symbol not in errors]
                    if not group:
                        break
                    if chunk_end == end:
                        break
                    start = chunk_end  # Inclusive overlap is overwritten by _merge.
                succeeded.extend(group)
            with self._state_lock:
                for symbol in succeeded:
                    self.last_successful_refresh[symbol] = now
                    self.last_errors.pop(symbol, None)
                self.last_errors.update(errors)
            return {"requested": len(due), "rows_received": received,
                    "errors": errors, "seconds": monotonic() - started}

    def snapshot(self):
        """Return an independent consistent frame; callers cannot mutate the buffer."""
        with self._state_lock:
            return self._candles.copy(deep=True)

    def _watch(self):
        import logging

        while not self._stop.wait(self.check_seconds):
            try:
                report = self.refresh(only_if_needed=True)
                if report["errors"]:
                    logging.warning("Live candle refresh failed for %d assets", len(report["errors"]))
            except Exception:
                logging.exception("Live candle updater failed; will retry at the next check")

    def start(self):
        """Top up synchronously, then start one background failsafe thread."""
        from threading import Thread

        if self._thread is not None and self._thread.is_alive():
            return None
        report = self.refresh()
        if report["errors"] and len(report["errors"]) == report["requested"]:
            raise RuntimeError("Initial Atlas refresh failed for every asset; retry before listening")
        self._stop.clear()
        self._thread = Thread(target=self._watch, name="topic90-candle-updater", daemon=True)
        self._thread.start()
        return report

    def stop(self):
        """Stop the scheduler and wait for any in-progress refresh to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None


# Use the assets that contributed training rows, intersected with live discovery.
training_universe = set(data.index.get_level_values("ticker"))
live_symbols = sorted(training_universe.intersection(atlas.discover_hl_universe()))
live_buffer = LiveMinuteBuffer(atlas, live_symbols, lookback=LOOKBACK)
startup_report = live_buffer.start()  # Complete before starting any listener.
print("Live minute buffer ready:", startup_report)
print("In-memory rows:", len(live_buffer.snapshot()))

# Warm the already-loaded model without fetching more data or retraining.
model.predict(data.iloc[:1][feature_columns])
# Feature extraction already ran in step 2 in this same process.
# Interactive use: live_buffer.refresh(); minute_data = live_buffer.snapshot()
# Stop before replacing/re-running this object: live_buffer.stop(). A plain
# script exits after its final statement; a listener or `python -i` keeps it alive.

# 5. Define the live inference callback
# The SDK callback below resolves T from the nonce block timestamp.
# Use current live universe intersected with the saved training universe.
# a. One get_bulk_1min_candles() request for overlapping raw minute rows.
# b. Overwrite matching local rows and append new rows; update the cache.
# c. Select complete input minutes ending at T (last opening time T - 1m).
# d. Find each asset's T-labeled partial independently; absent means zero drift.
# e. Resample using resample_ohlcv_polars(): final bar [T - 3m, T).
# f. Build the final feature row using extract_features_polars().
# g. Batch predict and reanchor: submitted_return = r + log(partial_close / C),
#    where C is the last input close and partial_close defaults to C if absent.
# h. Return successful lowercase labels, e.g. {"btc": ..., "eth": ...}.
# Record stage timings and omissions; attempt all eligible assets next round.

# Live inference workflow
def predict_live(T):
    """Use the prepared model and buffer for the explicit target start T."""
    tic = time.time()
    live_buffer.refresh()
    minute_data = live_buffer.snapshot()
    # Completed input minutes only; gaps and zero-volume candles remain as supplied.
    input_data = minute_data[minute_data.index.get_level_values("open_time") < T]
    if input_data.empty:
        return {}

    # Shift onto the fixed 3m grid so every ticker's final bar ends at the same T.
    offset = timedelta(seconds=int(T.timestamp()) % 180)
    input_bars = pl.from_pandas(input_data.reset_index()).with_columns(pl.col("open_time").cast(pl.Datetime("us", "UTC")) - offset)
    resampled = workflow.resample_ohlcv_polars(input_bars, freq="3m", groupby=["symbol"], live_mode=False).with_columns(pl.col("open_time") + offset)

    # Extract the final feature row independently for each ticker.
    features = pl.concat([
        workflow.extract_features_polars(bars, LOOKBACK, [T - timedelta(minutes=3)]).with_columns(pl.lit(symbol[0]).alias("symbol"))
        for symbol, bars in resampled.partition_by("symbol", as_dict=True).items()
    ])
    live_features = features.filter(pl.col("open_time") == T - timedelta(minutes=3)).to_pandas().set_index("symbol")
    live_features = live_features.loc[np.isfinite(live_features[feature_columns]).all(axis=1)]

    # Predict all eligible tickers in one batch.
    returns = pd.Series(np.asarray(model.predict(live_features[feature_columns])) * prediction_scale, index=live_features.index) if len(live_features) else pd.Series(dtype=float)

    # Reanchor using each ticker's T-labeled partial, or zero drift when absent.
    closes = resampled.filter(pl.col("open_time") == T - timedelta(minutes=3)).to_pandas().set_index("symbol")["close"]
    partials = minute_data[minute_data.index.get_level_values("open_time") == T].reset_index().set_index("symbol")["close"]
    returns += np.log(partials.reindex(returns.index).fillna(closes)) - np.log(closes.reindex(returns.index))
    predictions = {symbol.removeprefix("hl_").removesuffix("_1min").lower(): float(value) for symbol, value in returns.items() if np.isfinite(value)}
    toc = time.time()
    print(f"Live inference workflow completed in {toc - tic:.2f} seconds. Predicted {len(predictions)} assets.")
    return predictions


predictions = predict_live(datetime.now(timezone.utc).replace(second=0, microsecond=0))


# 6. Launch the SDK worker in this same process (no pickle or second process).
# Terminal: python notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py --worker
# Already inside python -i: asyncio.run(run_worker())
# Notebook: await run_worker()
# Ctrl-C stops the listener and the buffer updater.
async def run_inference(context):
    from allora_sdk import get_block_time

    opened_at = await get_block_time(context.client, context.nonce)
    T = opened_at.astimezone(timezone.utc).replace(second=0, microsecond=0)
    # Let Atlas publish newer candles before refreshing the buffer. Keep T fixed.
    await asyncio.sleep(10)
    predictions = await asyncio.to_thread(predict_live, T)
    if not predictions:
        raise RuntimeError("No eligible assets this round")
    return predictions


async def run_worker(timeout=None, max_attempts=None):
    from allora_sdk import AlloraWorker, AlloraWalletConfig
    from contextlib import aclosing
    from cosmpy.mnemonic import generate_mnemonic

    attempts = 0

    async def submit_predictions(context):
        values = await run_inference(context)
        values = {label: values[label] for label in volume_ranked_labels if label in values}
        values = dict(list(values.items())[:100])  # Highest-volume available predictions; $20M is reference only.
        print(f"Attempting {len(values)} labels for nonce {context.nonce}")
        return values

    try:
        key_file = Path("worker_keys") / f"topic_{TOPIC_ID}.key"
        key_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not key_file.exists():
            with os.fdopen(os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
                f.write(generate_mnemonic())
        live_buffer.start()  # Idempotent; also allows restarting after Ctrl-C.
        worker = AlloraWorker.inferer(
            run=submit_predictions,
            topic_id=TOPIC_ID,
            wallet=AlloraWalletConfig(mnemonic_file=str(key_file)),
            api_key=api_key,
            polling_interval=5,
            max_unfulfilled_nonces=1,
        )
        print(f"Topic {TOPIC_ID} testnet worker: {worker.address}")
        async with aclosing(worker.run(timeout=timeout)) as results:
            async for result in results:
                attempts += 1
                if isinstance(result, Exception):
                    print(f"Submission failed: {result}")
                else:
                    print(f"Submitted {len(result.submission)} assets: {result.tx_result.txhash}")
                if max_attempts is not None and attempts >= max_attempts:
                    break
    finally:
        live_buffer.stop()


if __name__ == "__main__" and "--worker" in sys.argv:
    asyncio.run(run_worker())
