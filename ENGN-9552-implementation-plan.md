# ENGN-9552 — implementation checklist

Reviewed implementation plan; no implementation yet. Track this checklist on the feature branch and update it as work proceeds.
Branch: `tim/engn-9552-triple-barrier-topic-example-code-in-builder-kit-and-blog`.

## Scope and fixed decisions

Add triple barrier through the existing workflow → example → local WorkerManager flow. Preserve features, normalization, resampling, bar-count semantics, timing, existing scalar behavior, and existing tests. No timing redesign, hosted export work, or changes to the running worker fleet.

| Topic | Asset | Atlas dataset name |
| --- | --- | --- |
| 87 | Gold | `hl_xyzgold_1min` |
| 88 | Silver | `hl_xyzsilver_1min` |
| 89 | WTI | `hl_xyzcl_1min` |

Resolve dataset IDs using existing Atlas lookup; no Binance fallback.

- Targets: one-hot `target_up`, `target_neutral`, `target_down`; unresolved rows remain null.
- Predictions: raw probabilities in `pred_up`, `pred_neutral`, `pred_down`; submit `{"down": p_down, "neutral": p_neutral, "up": p_up}`.
- Explicit array order: `[down, neutral, up]`. Argmax is internal to label-based metrics/diagnostics; earliest tied class wins. Any optional hard-output presentation does not replace stored probabilities or default submissions.
- All target code stays in `workflow.py`; all evaluation code in `evaluation.py`; diagnostics in the example. No new library modules.
- All new tests go in **one file**, `tests/test_triple_barrier.py`. Existing tests and shared fixtures remain untouched.
- SDK target: `allora-sdk==1.4.0rc4`. Its published wheel supports labeled dictionaries and existing scalar callbacks (label `y`).

## Setup

- [x] Create the feature branch and commit this plan before implementation.
- [ ] Create `notebooks/.venv`; install `-e ".[dev]"` and the exact SDK prerelease. Do not alter other workers' environments.
- [ ] Copy the authorized key from `../atlas_map/.allora_api_key` to an ignored key location and export it without printing or embedding it in artifacts.
- [ ] Align `pyproject.toml` and `notebooks/requirements_worker_runtime.txt` with the SDK version; verify generated-file ignore coverage.
- [ ] Read `Open_Trader_V1.pdf` as reference material. The supplied topic specification and review decisions govern implementation.

## Code changes

### `allora_forge_builder_kit/workflow.py`

- [ ] Accept `target_type="triple_barrier"` and add dispatch alongside existing targets.
- [ ] Add `compute_triple_barrier_target_polars` immediately after `compute_volatility_target_polars` (currently near line 537). Append the three target columns.
- [ ] Preserve existing resampling and `target_bars` conventions: e.g. 1h rows, 24 target bars, overlapping target per row. Keep original minute data available only for target calculation; do not change feature methods.
- [ ] Implement a general triple-barrier method using existing `target_bars` and resampling interval: horizon H = `target_bars × bar duration`. Compute rolling high–low log ranges over `target_bars` already-resampled bars, then average over `100 × target_bars` bars: ATR lookback is ALWAYS 100 target horizons. Use the existing native/resampled candles for ATR, with the established half-open/bar-count convention and enough additional warmup for the rolling ranges. Do not hard-code 24h or 2,400h in the builder. The topic example uses 1h bars, `target_bars=24`, and k=0.25, yielding the specified 24h horizon and 2,400h lookback; retain minute high/low first-touch testing for these topics.
- [ ] For the 24h topic example, apply supplied base/barrier formulas (general method derives window durations from H and the 100-horizon rule): base=`close(T-1m)`, upper/lower=`base * exp(±0.25 * ATR)`; historical averaging samples `[T-2401h,T-1h)`; test interval `[T-1m,T+24h-1m)`. Map each row to T using the existing candle-end convention. ATR means averaged log range, not conventional true range.
- [ ] Inclusive touches; first lower→down, first upper→up, neither→neutral, same-minute both→down. Missing historical/future coverage→unresolved, never fabricated neutral or forward-filled prices.
- [ ] Keep necessary plot metadata outside `feature_*`; use efficient rolling/indexed calculations.

### `allora_forge_builder_kit/evaluation.py`

- [ ] Add classification dispatch/methods to `PerformanceEvaluator`; preserve scalar defaults. Read one-hot/probability columns in explicit class order.
- [ ] Baseline: previous 100 resolved targets available at prediction time, per topic; fewer if necessary, uniform if none. Reuse established timing rather than adding workflow scheduling.
- [ ] Compute accuracy improvement, ordinal quadratic weighted kappa, Brier skill, gamma=2 unweighted focal skill, and participation. Focal clips only p_true to `[1e-15,1]` before log.
- [ ] Paired circular bootstrap: length 10, 1,000 replicates, identical sampled worker/baseline/truth rows; report 5th/95th percentiles. Exclude non-finite replicates per statistic; none finite→null estimate/interval and failed associated criterion.
- [ ] Exactly six strict gates: accuracy improvement >0.02; its lower bound >0; kappa lower bound >0; Brier skill lower bound >0; focal skill lower bound >0; participation >0.90.
- [ ] Report raw/baseline metrics, intervals, `nvalid`, documented `n_eff`, participation, `{key,value,threshold,passed}` criteria, and `eligible=all passed`. No grade or minimum-sample gate; optional provisional label below 100 observations does not affect eligibility. Do not present offline coverage as measured network participation.
- [ ] Optional payoff diagnostic: +1 correct directional, -1 opposite directional, zero otherwise, minus declared c for directional predictions. Separate from eligibility.

### NEW `notebooks/example_triple_barrier_walkthrough.py`

- [ ] One example configurable for the three Atlas presets; follow existing initialization/backfill/dataset/train/evaluate/save flow and live-feature usage.
- [ ] Immediately after loading data, before model training, render and save `triple_barrier_example.png` as the first walkthrough illustration. Select a historical row with complete context and a resolved directional touch so the figure demonstrates the learning problem.
- [ ] Plot 1h OHLC candlesticks and an aligned volume panel. Row timestamps are candle OPEN times; mark current time as selected row open time + candle width. Show the selected candle and preceding candles as observed history, visually distinguish future candles, and label the history/future split clearly.
- [ ] Draw the base price and upper/lower barriers across the future prediction period, ending at current time + configured horizon (24h in this example). Show the future candles throughout this barrier box, including the remaining horizon after a touch, and draw the vertical time boundary.
- [ ] Annotate the first touched boundary with an arrow/marker, direction, and hit time. Locate the touch using the same underlying minute high/low target evidence, even though the displayed candles are hourly; do not infer touch order from an hourly candle that crosses both barriers. Apply the target's down-first rule for minute ties. Label the resulting target and retain the existing exact target-window convention.
- [ ] Keep this figure's code in the example, show it when the execution environment supports display, and always save it with the run artifacts. Include the image among test 2's expected outputs and retain ordinary example-run images for the later blog.

- [ ] Train a modest `LGBMClassifier`; explicitly map estimator classes to probability columns. Use established chronological validation, keep final evaluation separate from model selection, and backfill enough warmup/training history.
- [ ] Select the best model/configuration using only the earlier walk-forward CV folds. Reserve the last fold as the final OOS holdout; use its already-in-memory predictions/data for the trading example, without selecting the model on that fold or rerunning inference with a later full-data refit.
- [ ] Produce `predict.pkl`, `config.json`, `metrics.json`, `predictions.csv`, `report.txt`, and printed evaluation/deploy guidance, consistent with current examples.
- [ ] Put all diagnostics and plotting directly in the example: introductory barrier illustration, confusion matrix, optional classification-payoff diagnostic, and the final-fold trading example below. Keep plotting out of the inference artifact.
- [ ] Trading signals: derive -1/0/+1 from final-fold probability argmax in declared class order. Every row is a possible trade: down opens a short, up opens a long, neutral opens none. Retain probabilities unchanged. Use each row's base/entry price and its own target barriers/horizon.
- [ ] Replay underlying minute candles for each trade. Exit at whichever boundary is touched first and book the EXACT boundary price, not the hit candle close; use the established same-minute down-first rule. If neither barrier is hit, exit at the close of the last candle inside that row's evaluation interval. Include those realized expiry gains/losses; neutral-as-zero belongs only to the separate optional classification-payoff metric.
- [ ] Record a trade ledger with signal, entry/exit timestamps and prices, barriers, exit reason, size, gross PnL, declared costs, and net PnL. Compute signed PnL as direction × (exit price - entry price) × size. State a fixed-size convention explicitly; proposed default is one asset unit per directional row. Allow overlapping trades, as required by one possible trade per row; do not silently suppress signals while another trade is open.
- [ ] Save `example_trades.png`: show a series of final-fold OOS trades on a candlestick chart, with each trade's upper/lower/time barriers and annotated entry/exit points, direction, and exit reason. Use panels or a readable subset if overlapping boxes obscure the chart.
- [ ] Save `cumulative_trade_pnl.png`: cumulative realized PnL over ALL final-fold directional trades, ordered by exit time and combining simultaneous exits. State sizing, costs, PnL units, and overlapping exposure; do not imply a capital-normalized portfolio return. Save `trades.csv` alongside the figure. Preserve enough future candles to resolve final-fold rows; incomplete intervals stay unresolved rather than inventing an exit.
- [ ] Include the trade ledger and both trading plots among test 2's expected artifacts. Use the in-memory holdout results for this demonstration before any production refit; no separate backtesting module or new test file.
- [ ] Save artifacts in a uniquely named run directory beside the example for inspection/blog use. Test runs clean theirs up; ordinary example runs retain theirs.

### Existing submission/monitoring files only

- [ ] `worker_runtime.py`: accept scalar OR labeled numeric dictionary, validate finite values, and pass through to SDK. Keep scalar price checks; zero class probabilities are valid. Probability constraints belong to the classification path, not all multi-output transport.
- [ ] `worker_manager.py`: verify existing artifact/lifecycle/interpreter handling; change only if necessary. Use isolated new-worker state; never bulk start/stop/redeploy existing workers.
- [ ] `notebooks/deploy_worker.py`: reuse `TOPIC_ID` and `PREDICT_PKL`; add new-example guidance, no duplicate deploy script.
- [ ] `worker_monitor.py`: inspect rc4 event schema and retain labeled values rather than scalar-coercing them. `web_dashboard.py` and, if needed, `workerctl.py`: display these values while preserving scalar behavior.
- [ ] Leave hosted `export.py` and its example unchanged.

## Tests and completion — three integration steps

Only `tests/test_triple_barrier.py`. Use real Atlas Gold data and the workflow's normal cache, reused across the run. Helpers/fixtures live in this file; no existing test changes. These three steps are the completion checks.

- [ ] **1. Minimal workflow + independent target verification.** Build a non-empty dataset with `target_type="triple_barrier"`. Verify target columns exist; resolved rows are 0/1, non-negative, and sum to 1; unresolved rows remain null.
  Independently verify targets on a deterministic sample of resolved rows using raw Atlas minute OHLC (normal cached source data is acceptable). Include each available class, and both-hit cases if present. Starting from the candle end represented by each sampled row, derive T using the existing convention. In a simple test-only reference loop, independently calculate hourly ranges/lookback/barriers and replay the exact minute testing interval in chronological order. Compare the expected one-hot result with the workflow row; report timestamp, expected/actual label, barriers, and first touch on mismatch. Do not call the production target method or reuse its computed barriers/intermediates. Use raw timestamps to check complete coverage and inspect available unresolved rows too. This deliberately duplicates the mathematical definition, not the optimized implementation; keep it small and readable. Do not claim coverage for edge cases absent from the real data.
- [ ] **2. Full example on the same dataset.** Run the whole script; require successful completion and the listed artifact family. Check CSV target/probability columns, finite non-negative unit-sum probabilities, agreed report structure, and plots. Reload `predict.pkl` and verify labeled probability output. Functional completion does not require the model to pass all six quality criteria.
- [ ] **3. Deploy with `deploy_worker.py`.** Use test 2's artifact and topic 87 in isolated new-worker state. Require a successful labeled on-chain submission and monitoring evidence, not just a running process.

**Isolation/cleanup:** one unique run ID owns all produced files: cache, example outputs, copied artifacts, worker keys/state/database, logs, and temporary evidence. Standard filenames are allowed inside unique directories. Share prerequisites without depending on pytest execution order. Guaranteed teardown stops only the test-created worker and removes only that run's local files, including after failure. Leave existing caches, fleet, source key, venv, code, and this plan untouched. On-chain records naturally remain.

| Step | Status / concise evidence |
| --- | --- |
| 1. Dataset + independent minute replay | Pending |
| 2. Example outputs + reloadable artifact | Pending |
| 3. Successful deployment + monitoring | Pending |
| Run ID / cleanup outcome | Pending |

## Documentation after implementation

- [ ] `README.md`: topic/Atlas mapping, target selection, labeled predictions, six metrics, example/deploy/test commands, warmup and artifacts.
- [ ] `AGENTS.md`, `SKILLS.md`, relevant repository skills: update only affected target/output guidance; keep setup safeguards. Inspect relevant skills before implementation.
- [ ] Allora docs repo: inspect structure, then add exact target/evaluation reference and navigation. Paths TBD; link actual pages, distinguish validation criteria from on-chain loss.
- [ ] After code/example finalization, write **local untracked** `ENGN-9552-blog.local.md` using real example artifacts. No publication now.

References: https://github.com/allora-network/allora-sdk-py · https://pypi.org/project/allora_sdk/1.4.0rc4/ · https://github.com/allora-network/docs

Latest review (2026-10-01): plan reviewed for consistency and prepared as the first feature-branch commit. Implementation has not started. The reference PDF remains local; the later blog draft remains untracked as requested.
