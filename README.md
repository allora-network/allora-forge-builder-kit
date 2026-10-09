<img width="100%" alt="forge_silicon" src="https://github.com/user-attachments/assets/f1444abf-e649-4e48-a9f0-187b78b59ccc" />

# Allora Forge Builder Kit

Build, evaluate, and deploy ML inference workers on the [Allora Network](https://allora.network).

> [!IMPORTANT]
> **SDK v10 — testnet and mainnet.** Both networks have completed the emissions v9 → v10 chain upgrade. This repository pins `allora-sdk==1.4.0rc4` for scalar and labeled multi-output submissions.
>
> ```bash
> pip install "allora-forge-builder-kit @ git+https://github.com/allora-network/allora-forge-builder-kit.git@main"
> ```

## Contents

- [What is Allora?](#what-is-allora)
- [What is the Allora Forge?](#what-is-the-allora-forge)
- [What you get](#what-you-get)
- [Topic types](#topic-types)
- [Topics](#topics)
- [Quick start](#quick-start)
- [The learning problem](#the-learning-problem)
- [Evaluation metrics](#evaluation-metrics)
- [Worker operations](#worker-operations)
- [Development and references](#development-and-references)

## What is Allora?

Allora is a decentralized AI network that coordinates predictions across many independent ML models. Rather than relying on a single model, the network aggregates inferences from competing workers and weights them by historical accuracy — producing a combined output that outperforms any individual contributor.

The network is organized into **topics**. Each topic defines a prediction task (e.g. "8-hour BTC/USD log return") and runs a continuous lifecycle:

1. **Submission window opens** — the network pings all registered workers for their inference
2. **Workers respond** with a prediction value
3. **Evaluation window** runs for the topic's time horizon (e.g. 8 hours)
4. **Scores are revealed** — workers are ranked by loss against the ground truth, and rewards are distributed

```
time ──────────────────────────────────────────────────────────────────►

  ◄── submission ──►◄─────────────── evaluation period (e.g. 8h) ──────►
  │                 │                                                    │
open             close                                               scores
workers          predictions                                        revealed
polled           locked                                           + rewarded
```

Live topics include crypto market predictions and commodity triple-barrier classification. New topics are added over time.

## What is the Allora Forge?

The [Allora Model Forge](https://forge.allora.network) is the hub for ML practitioners to compete, earn rewards, and build reputation on the network. Workers start on testnet to establish a track record, then graduate to mainnet where top performers earn ALLO token rewards.

This toolkit handles everything between your model and the network: data, feature engineering, evaluation, wallet management, and worker deployment.

## What you get

- **Workflow API** — backfill historical data → engineer features → build training datasets
- **Evaluation** — grade your model against Allora's scoring methodology before deploying
- **Deployment tooling** — wallet creation, faucet funding, worker lifecycle management
- **Monitoring dashboard** — web UI showing submission history, on-chain scores, and live logs
- **Topic discovery** — query all live topics on testnet and mainnet

---

## Topic types

The workflow supports three target types. Price topics use the log-return workflow
and convert its prediction to an absolute price at submission.

| Topic family | Workflow `target_type` | Worker output |
|---|---|---|
| Price / log returns | `log_return` (default) | Absolute price or log return, as required by the topic |
| Volatility | `volatility` | Nonnegative horizon-scaled realized volatility |
| Triple barrier | `triple_barrier` | Labeled probabilities: `down`, `neutral`, `up` |

## Topics

### Price / log returns

Predict future price or `log(future price / current price)`. Start with the
[topic 69 price walkthrough](notebooks/example_topic_69_bitcoin_walkthrough.py),
[topic 77 fast price walkthrough](notebooks/example_topic_77_bitcoin_5min_walkthrough.py),
or [topic 83 log-return workflow](notebooks/testnet/topic_83_btc_8h_logreturn/example.py).

Playground topics:

| Testnet ID | Name | Target type | Notes |
|-----------|------|-------------|-------|
| **69** | BTC/USD - 1 Day Price Prediction | Price | Example walkthroughs use this |
| **77** | BTC/USD - 5 Min Price Prediction | Price | Playground Fast |
| **90** | Hyperliquid perps — 3-minute log returns | Labeled log returns | Pooled LightGBM; up to 100 asset labels per worker |

Mainnet topics and testnet equivalents:

| Mainnet ID | Mainnet Name | Testnet ID | Testnet Name |
|-----------|-------------|-----------|-------------|
| 1  | BTC/USD - Log Returns - 8h  | 83 | BTC/USD - 8h Log-Return Prediction |
| 2  | ETH/USD - Log Returns - 8h  | 84 | ETH/USD - 8h Log-Return Prediction |
| 3  | SOL/USD - Log Returns - 8h  | 58 | 8h SOL/USD Log-Return Prediction |
| 9  | ETH/USD - Price Prediction - 8h | 41 | ETH/USD - 8h Price Prediction |
| 10 | SOL/USD - Price Prediction - 8h | 38 | SOL/USD - 8h Price Prediction |
| 14 | BTC/USD - Price Prediction - 8h | 42 | BTC/USD - 8h Price Prediction |
| 15 | BTC/USD - Log Returns - 24h | 61 | 1 day BTC/USD Log-Return Prediction |
| 16 | ETH/USD - Log Returns - 24h | 63 | 1 day ETH/USD Log-Return Prediction |
| 17 | SOL/USD - Log Returns - 24h | 62 | 1 day SOL/USD Log-Return Prediction |
| 18 | BTC/USD - Log Returns - 20m | — | Missing |
| 19 | NEAR/USD - Log Returns - 8h | 71 | 8h NEAR/USD Log-Return Prediction |

### Volatility

Predict `sample_std(r₁, …, r_H) × √H` for consecutive one-minute log returns,
using `ddof=1`. See the [BTC grid-search workflow](notebooks/testnet/topic_79_btc_vol/model_grid_retrain.py).
Equivalent workflows are available for
[ETH](notebooks/testnet/topic_80_eth_vol/model_grid_retrain.py),
[XRP](notebooks/testnet/topic_81_xrp_vol/model_grid_retrain.py),
[SOL](notebooks/testnet/topic_82_sol_vol/model_grid_retrain.py), and
[4-hour ETH](notebooks/testnet/topic_85_eth_4h_vol/model_grid_retrain.py).

| Testnet ID | Name | Target type | Notes |
|-----------|------|-------------|-------|
| **79** | BTC/USD - 15 Min Volatility Prediction | Volatility | Sample std of 1-min log returns × √15 |
| **80** | ETH/USD - 15 Min Volatility Prediction | Volatility | Same definition as 79, ETH pair |
| **81** | XRP/USD - 15 Min Volatility Prediction | Volatility | Same definition as 79, XRP pair |
| **82** | SOL/USD - 15 Min Volatility Prediction | Volatility | Same definition as 79, SOL pair |
| **85** | ETH/USD - 4h Volatility Prediction | Volatility | Sample std of 1-min log returns × √240 |

### Triple barrier

Predict which price barrier is touched first, or neutral if neither is touched
before expiry. Follow the [triple-barrier walkthrough](notebooks/example_triple_barrier_walkthrough.py).
These commodity datasets require Atlas; Binance is not an alternative source.

| Testnet ID | Asset | Atlas dataset | Horizon |
|---|---|---|---|
| 87 | Gold | `hl_xyzgold_1min` | 24h |
| 88 | Silver | `hl_xyzsilver_1min` | 24h |
| 89 | WTI oil | `hl_xyzcl_1min` | 24h |

The tables are a repository reference. For current network metadata, see
[topic discovery](allora_forge_builder_kit/topic_discovery.py) and the
[Forge](https://forge.allora.network).

## Quick start

### 1. Set up the environment

Use Python 3.10 or newer. The repository pins `allora-sdk==1.4.0rc4`, including
support for labeled multi-output submissions.

```bash
git clone https://github.com/allora-network/allora-forge-builder-kit.git
cd allora-forge-builder-kit
python3.11 -m venv notebooks/.venv
source notebooks/.venv/bin/activate
pip install -e ".[dev,wallet-link]"
export REPO_ROOT="$PWD"
```

Get an API key from [developer.allora.network](https://developer.allora.network),
save it locally as `.allora_api_key`, then load it without displaying it:

```bash
export ALLORA_API_KEY="$(cat "$REPO_ROOT/.allora_api_key")"
export ALLORA_NETWORK=testnet
```

Keep the same shell for the commands below. Binance can be used for supported
crypto datasets when adapting a workflow; the examples below use Atlas.

### 2. Run one example and deploy its artifact

Choose one block. Each uses its own working directory for artifacts and worker
state. Training can take substantial time, especially the volatility grid search.
The scripts print their evaluation results and output locations.

**Hyperliquid perps — topic 90 (three workers):**

Run from the repository root with the activated environment and Atlas API key
configured above. The full parameter grid, lookback, six-month history, tree
checkpoints, and number of exported models are at the top of
[example.py](notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py).

```bash
cd "$REPO_ROOT"
# Fetch available history; existing minute parquet stays under the example's data/.
python notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py --backfill-only
# Train from that cache, evaluate, and export the three best distinct trials.
python notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py
# Replace the path below with the results/model_search/<timestamp> printed above.
python notebooks/testnet/topic_90_hyperliquid_3min_logreturn/deploy_managed_example.py \
  --model-run notebooks/testnet/topic_90_hyperliquid_3min_logreturn/results/model_search/RUN_TIMESTAMP \
  --deploy
```

Each exported `model_1` / `model_2` / `model_3` directory contains a full-data-refitted
`model.joblib`, training-fitted return calibration, and raw/scaled validation
reports in both pooled and equal-weight per-asset views. A chronological holdout
reserves the latest 20% of observed timestamps for validation; training labels
that resolve after the validation boundary are excluded. Selection uses the lowest
validation MSE per trial. Those reports are validation-selected, not independent
test results. `top_models.json` tells the launcher which models to package.
Omit `--deploy` to create the callable artifacts without allocating workers.

WorkerManager allocates a separate local wallet for each model and records it in
that model's `managed_worker.json`. Keep those files to reuse the same addresses.
Workers run in independent process sessions: **no tmux or open terminal is required**.
They survive launcher exit, but are not automatically restarted after a host reboot.
Existing workers outside this deployment are not stopped.

Each artifact tops up its own in-memory minute buffer before listening. At inference
it anchors to the nonce's minute, waits ten seconds, refreshes Atlas data in bulk,
resamples history to that exact boundary, and applies each asset's partial-candle
offset. It submits up to 100 available predictions by cached volume rank; unavailable
assets are omitted. Runtime state and API keys are not embedded in the artifacts.

For example, a window opening at **12:01:27 UTC** targets **12:01–12:04**, even
though submission closes around 12:01:57. The final completed input bar is labeled
11:58 and ends at 12:01. The worker separately checks each asset's 12:01-labeled
partial candle and adds its observed log price drift to the model prediction;
if that candle is missing, the asset gets a zero-drift adjustment. The ten-second
wait does not move the target boundary.

For direct Atlas access, using a configured `AtlasDataManager` instance:

```python
symbols = atlas.discover_hl_universe()
minutes = atlas.get_bulk_1min_candles(symbols, limit=10)
```

Discovery returns exact Atlas dataset names for fresh, consumable native perps.
The bulk call uses one HTTP request; `limit` is **per asset**, with at most 10,000
requested rows across the batch. It returns raw one-minute OHLCV indexed by
`(symbol, open_time)`, preserving partial candles and gaps. It does not resample
or write to the local cache.

Logs go to `worker_logs/worker_90_<address>.log` in the launch directory. Inspect workers with:

```bash
python -m allora_forge_builder_kit.workerctl dashboard
```

**Price — topic 69:**

```bash
mkdir -p "$REPO_ROOT/notebooks/runs/price_example"
cd "$REPO_ROOT/notebooks/runs/price_example"
python "$REPO_ROOT/notebooks/example_topic_69_bitcoin_walkthrough.py"
TOPIC_ID=69 PREDICT_PKL="$PWD/predict.pkl" python "$REPO_ROOT/notebooks/deploy_worker.py"
```

**Volatility — topic 79:** runs the default 800-day grid search and saves ranked
artifacts and a scatter plot, and prints evaluation metrics. Deploy the highest-ranked artifact:

```bash
mkdir -p "$REPO_ROOT/notebooks/runs/volatility_example"
cd "$REPO_ROOT/notebooks/runs/volatility_example"
python "$REPO_ROOT/notebooks/testnet/topic_79_btc_vol/model_grid_retrain.py"
TOPIC_ID=79 PREDICT_PKL="$PWD/predict_79_grid_rank1.pkl" python "$REPO_ROOT/notebooks/deploy_worker.py"
```

**Triple barrier — topic 87:**

```bash
cd "$REPO_ROOT"
python notebooks/example_triple_barrier_walkthrough.py --topic 87
mkdir -p notebooks/triple_barrier_example_output/worker
cd notebooks/triple_barrier_example_output/worker
TOPIC_ID=87 PREDICT_PKL="$REPO_ROOT/notebooks/triple_barrier_example_output/predict.pkl" python "$REPO_ROOT/notebooks/deploy_worker.py"
```

Triple-barrier outputs are in `notebooks/triple_barrier_example_output/`:
`predict.pkl`, `config.json`, `metrics.json`, `predictions.csv`, `report.txt`,
`trades.csv`, and six charts. Repeated runs replace these outputs; use
`--output-dir` to retain another run. Data is cached inside the output directory
unless `--cache-dir` is supplied. To use Silver or WTI, change both the example's
`--topic` and deployment's `TOPIC_ID` to 88 or 89.

Deployment creates a wallet, requests testnet funding, and starts a worker.
Inspect `worker_logs/` for funding, registration, or submission failures.
A started process alone does not confirm successful submissions.

### 3. Monitor

From the same worker directory:

```bash
python -m allora_forge_builder_kit.web_dashboard
```

Open **http://localhost:8787** to inspect submissions, scores, and logs.
For a terminal summary, use `python -m allora_forge_builder_kit.workerctl dashboard`.

## The learning problem

### Framing forecasting as supervised learning

At any point in time $t$, the model observes a window of $N$ past bars as input features $\mathbf{x} \in \mathbb{R}^d$ and predicts a future outcome $y$ over the next $H$ bars. The target $y$ depends on the topic type:

- **Price / log-return topics** — $y = \log(p_{t+H} / p_t)$ or the absolute price $p_{t+H}$
- **Volatility topics** — $y = \text{sample\_std}(r_1, \ldots, r_H)\sqrt{H}$, using `ddof=1`, where $r_i = \log(p_{t+i} / p_{t+i-1})$ are consecutive 1-minute log returns over the horizon

- **Triple-barrier topics** — a one-hot label for the first upper/lower barrier touched, or neutral at expiry. Models submit probabilities labeled `down`, `neutral`, and `up`.

By sliding this window across the full history, a single time series becomes thousands of labeled examples $(\mathbf{x}_i, y_i)$, turning forecasting into a standard supervised learning problem.

The `AlloraMLWorkflow` handles this construction: `backfill()` fetches historical data, `get_full_feature_target_dataframe()` builds the feature matrix and target vector, ready for any scikit-learn compatible model.

### Triple-barrier targets

The horizon is `target_bars × interval`. The example uses 100 hourly input bars,
24 target bars, and a barrier multiplier of 0.25. These settings define Forge
topics 87–89; changing the native interval or horizon defines a different target.
The general builder accepts other configurations for research. Historical source
candles must be one-minute data (Atlas supplies these regardless of feature interval). ATR is the mean high–low log
range over 100 target horizons, computed on native/resampled candles.

For hourly bars and prediction time T, average the 2,400 samples opening in
`[T − 2401h, T − 1h)`. Each range includes both endpoints of `[t − 24h, t]`
(25 hourly openings). Match the reputer's SQL by restricting history to the
averaging interval first, so the initial ranges are partial.

With base price P equal to the minute close at `T − 1m`, upper and lower
barriers are `P × exp(0.25 × ATR)` and `P × exp(−0.25 × ATR)`.
Test minute high/low candles in `[T − 1m, T + 24h − 1m)` chronologically:
first lower touch → down; first upper touch → up; no touch → neutral.
A same-minute tie resolves down. Missing required coverage leaves targets null.
Allow 100 horizons plus one native bar of history before usable targets,
plus sufficient training and evaluation data.

The walkthrough illustrates the learning problem, fold periods, confusion
matrix, example trades, and cumulative PnL. Trades use overlapping one-unit
positions, exact barrier fills, and the last in-window close at expiry.
`--trade-cost-bps` sets per-side trading costs (default zero); these plots are
not capital-normalized portfolio returns.

[Target implementation](allora_forge_builder_kit/workflow.py#L543).

### Empirical risk minimization

The standard recipe is to pick a model $f$ by minimizing empirical (in-sample) loss:

$$f^* = \arg\min_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^{n} \ell(y_i,\, f(\mathbf{x}_i))$$

The ERM assumption is that training and deployment data share the same distribution — so a model that fits well in-sample will generalize out-of-sample. This is a reasonable working assumption in many domains.

### Why finance makes this hard

Financial markets violate the ERM assumption routinely:

- **Regime changes** — volatility regimes, macro shocks, and structural breaks mean the distribution of returns today can look nothing like last year's.
- **Non-stationarity** — correlations, volatility, and return distributions all drift over time.
- **Low signal-to-noise** — crypto returns are heavily noise-dominated, making it easy to fit noise rather than signal.

The practical consequence is that **overfitting is the default failure mode**. A model can lower in-sample loss while out-of-sample loss increases — more model complexity captures noise instead of signal. Traditional remedies (early stopping, depth limits, regularization, conservative learning rates) are especially important here.

### Walk-forward validation

To measure true out-of-sample performance the toolkit uses **walk-forward cross-validation**: train on data up to time $t$, evaluate on data strictly after $t$, advance the window, repeat. This respects temporal ordering (no lookahead leakage) and produces a realistic sample of out-of-sample predictions. The evaluation metrics in the next section are computed entirely on these held-out predictions.

### The model builder's job

The example notebooks use **LightGBM** (gradient boosting over decision trees) with conservative defaults as a starting point. Gradient boosting is a strong tabular baseline — it handles non-linearity and feature interactions well and is relatively robust to scale.

From here, improving your score comes down to three levers:

1. **Feature engineering** — what information goes into $\mathbf{x}$. The base features are normalized OHLCV ratios (last-close normalized to 1.0). Adding technical indicators (RSI, MACD, realized volatility), log-return series, or cross-asset signals is where most alpha lives.
2. **Model and regularization** — early stopping, tree depth, learning rate, and subsampling to keep variance in check.
3. **Out-of-sample evaluation** — use the metrics appropriate to the topic family below. The triple-barrier example defaults to five folds: the first three select the lowest mean log loss, and the last two supply combined OOS evaluation. The selected configuration stays fixed; each OOS fold refits using labels available at its cutoff, including earlier OOS outcomes once resolved. Production refitting follows evaluation. `--folds` and `--holdout-folds` control the split; `LGBM_SEARCH_GRID` and `LGBM_FIXED_PARAMS` near the top of the script expose the model search. Just below them, edit `ENGINEERED_SPECS` and `engineer_features()` to add derived features; the same function runs during training and is captured in the inference artifact.

For structured methodology guidance on each of these levers, see the [Model creation skills](#model-creation-skills) section.

## Evaluation metrics

Evaluation depends on the topic family. Offline reports help assess a model;
network participation and live scores must be checked after deployment.

### Price / log returns

The price examples evaluate predicted log returns before converting to prices
for submission. [PerformanceEvaluator](allora_forge_builder_kit/evaluation.py#L697)
reports seven primary criteria and a letter grade:

| # | Criterion | Requirement |
|---|-----------|-------------|
| 1 | Effective samples | ≥ 20, before horizon adjustment |
| 2 | Directional accuracy | One-sided 95% lower confidence bound > 50% |
| 3 | Pearson correlation | Two-sided 95% lower confidence bound > 0 |
| 4 | WRMSE improvement | Horizon-adjusted 95% lower bound > 0 |
| 5 | WCZAR improvement | Horizon-adjusted 95% lower bound > 0 |
| 6 | Log aspect ratio | Confidence interval overlaps [-0.5, +0.5] |
| 7 | Participation | > 90% |

All seven must pass for `report["eligible"]`. Estimators and criteria are ported
from worker-metrics revision `bd01f9d2bdb6351e351ecddeb68d0fb101ebc0ee`.
DA and improvement bounds use `sqrt(max(1, horizon_minutes / 20))` as the
effective-sample multiplier; Pearson and aspect ratio are unscaled.

Existing `evaluate(y_true, y_pred, epoch_length_minutes=3)` and `print_report(report)`
calls still work. Without submission counts, the report explicitly assumes full
offline participation. Supply `n_expected_epochs` and optionally `n_submitted` for
observed participation; optional `lags`, `gt_ratio`, and `horizon_seconds` describe
irregular/overlapping samples. Defaults assume one horizon per row. Offline
pooled-asset results do not establish live participation or cross-asset independence.

**Grading:**

| Points (out of 7) | Grade |
|-------------------|-------|
| 7 | A+ |
| 6 | A |
| 5 | B+ |
| 4 | B |
| 3 | C |
| 2 | D |
| ≤ 1 | F |

### Volatility

The volatility workflows calculate their own metrics in
[`vol_metrics`](notebooks/testnet/topic_79_btc_vol/model_grid_retrain.py#L54).
These are example diagnostics, not the return evaluator's seven-point grade.

| Metric | Purpose |
|---|---|
| Pearson r / Spearman rho | Linear / rank association |
| R² / RMSE | Explained variation / prediction error |
| QLIKE | Relative volatility prediction loss |
| Calibration ratio | Predicted standard deviation divided by actual standard deviation |

The example ranks models using `R² − 0.5 × QLIKE − 0.3 × abs(1 − calibration ratio)`;
see [`composite_score`](notebooks/testnet/topic_79_btc_vol/model_grid_retrain.py#L68).

### Triple barrier

[`evaluate_classification`](allora_forge_builder_kit/evaluation.py#L603) consumes
probabilities in `[down, neutral, up]` order. Hard classes for accuracy and
confusion matrices use deterministic argmax; submissions retain probabilities.
The causal baseline uses the previous 100 resolved targets, with uniform
probabilities when no prior targets exist.

| Criterion | Strict threshold |
|---|---|
| Accuracy improvement over baseline | > 0.02 |
| Accuracy-improvement lower bound | > 0 |
| Quadratic weighted kappa lower bound | > 0 |
| Brier skill lower bound | > 0 |
| Focal skill lower bound | > 0 |
| Participation | > 0.90 |

Brier loss is the mean sum of squared probability errors. Focal loss is
`mean(−(1 − p_true)² × ln(p_true))`, clipping only `p_true` to `[1e-15, 1]`.
Both skills are `1 − worker loss / baseline loss`. Kappa treats classes as ordinal.

Bounds use paired circular bootstrap blocks of 10 rows, 1,000 replicates, and
5th/95th percentiles. Reports include raw metrics, intervals, individual criteria,
and `eligible` when all six pass; there is no letter grade. Offline participation
is coverage of the evaluation sample, not observed network participation.

Optional directional payoff awards +1 for a correct directional class, −1 for
an opposite class, and zero when either class is neutral, minus the declared
cost for a directional prediction. `--diagnostic-cost` sets this cost in barrier
units, separately from trading costs. It is not a seventh eligibility criterion.

## Worker operations

### Local workers

Use the [deployment script](notebooks/deploy_worker.py) and
[WorkerManager](allora_forge_builder_kit/worker_manager.py) for lifecycle operations.
Worker state, keys, and logs belong to the working directory used at deployment;
run monitoring and wallet-linking commands there. Keep generated state and secrets
out of git.

The dashboard defaults to localhost. `--host 0.0.0.0` exposes it on all interfaces;
use the authentication token printed to stderr as the URL's `?token=...` parameter.

> **Managed-custody security:** `FORGE_API_KEY` is a managed-wallet signing
> credential, not a transaction-scoped permission. The remote signer can sign
> arbitrary SignDoc bytes and 32-byte digests. Disabling the optional `/transfer`
> route does not constrain `/sign`. Protect and revoke this key like a private key.

<details>
<summary>Wallet linking: setup, options, and troubleshooting</summary>

### Wallet linking

When you deploy a **local-custody** worker the signing key lives in `worker_keys/` on your machine, but Forge doesn't know which `allo1...` addresses belong to your account. **Wallet linking** proves ownership: the CLI signs an ADR-036 challenge with each local worker key and a browser-authenticated Forge user approves the link.

> **Managed-custody workers** (deployed with `custody="managed"`) are linked automatically by the backend — no `workerctl link` step needed.

#### Custody modes at a glance

| Mode | Key lives | Linking |
|------|-----------|---------|
| **Local** (default) | `worker_keys/` on your machine | Run `workerctl link` once per address |
| **Managed** | Forge backend (Privy wallet) | Automatic — no CLI step |

#### Quick start

```bash
# Requires the wallet-link extra (cosmpy for ADR-036 signing)
pip install -e ".[wallet-link]"

# Link all local worker wallets to your Forge account
workerctl link
```

The CLI:
1. Reads your `worker_secrets.json` to find local key files
2. Opens a device-flow session with the Forge API
3. Signs each ADR-036 challenge locally — the mnemonic never leaves your machine
4. Opens your browser; you approve with your logged-in Forge account
5. Polls until approved and prints which addresses were linked

```
$ workerctl link

Linking 2 worker address(es) to Allora Forge at https://forge.allora.network

  First copy your one-time code: ABCD-1234
  Then approve the link at: https://forge.allora.network/link?code=ABCD-1234

Opened your browser. Waiting for approval...

Linked 2 verified worker(s):
  + allo1abc...
  + allo1def...
```

#### Link specific addresses

```bash
# Link a single address
workerctl link --address allo1abc...

# Link two specific addresses
workerctl link --address allo1abc... --address allo1def...
```

#### Headless / CI environments

```bash
# Print the URL and code without opening a browser
workerctl link --no-browser
```

Output the one-time code and URL to stdout so you can open them on a separate device or paste them into a CI log.

#### Non-default secrets file

```bash
workerctl link --secrets-path /path/to/worker_secrets.json
```

#### CLI reference

`workerctl link` accepts the following flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--secrets-path PATH` | `worker_secrets.json` | Path to the WorkerManager secrets file that maps addresses to local key files. |
| `--address ADDR` | all local keys | Limit to a specific `allo1...` address. Repeatable — pass once per address. |
| `--no-browser` | off | Print the approval URL and code without auto-opening a browser. |

#### Troubleshooting

**`cosmpy` not found** — install the wallet-link extra: `pip install -e ".[wallet-link]"` or `pip install cosmpy==0.11.1`.

**`No worker keys found`** — the secrets file is missing or empty. Deploy a local-custody worker first (`WorkerManager.deploy_worker(...)` or `python deploy_worker.py`).

**`No local key for: allo1...`** — the address is a managed-custody worker (linked automatically) or the secrets file is stale. Managed workers do not need manual linking.

**Link request denied** — the browser approval was rejected. Re-run `workerctl link` to start a fresh session.

**Link request expired** — the 30-minute approval window closed before the browser was used. Re-run to start a new session.

</details>

### Hosting export

For hosted deployment, follow the runnable
[export walkthrough](notebooks/export_to_hosting.py) and
[export implementation](allora_forge_builder_kit/export.py).
The package contains worker code, dependencies, a manifest, and optional weights.
Choose training on the platform or bundling locally trained weights.
The triple-barrier walkthrough currently demonstrates local WorkerManager deployment.

<details>
<summary>Hosting export: modes and deployment configuration</summary>

Or from the command line, which can also produce the upload-ready zip:

```bash
workerctl export-payload --config model.json --out build/my_lgbm_package --zip
# then upload build/my_lgbm_package.zip to forge (POST /api/v1/models)
```

`--zip` writes the package **contents** at the archive root, so forge finds `manifest.json` at the extraction root. The generated worker code is **generic over pair/timeframe**; code-only (train-on-platform) packages can be deployed against many pairs/timeframes/topics. Bundled-weight packages must use parameters matching how the weights were trained.

#### Two deployment modes

Exactly **one** of these must hold (forge rejects the package otherwise; `export_payload_for_hosting` enforces it and fails loudly):

| Mode | Set | Weights | Who trains |
|------|-----|---------|-----------|
| **Train-on-platform** | `supports_training=True` (default) | none | the platform |
| **Train-locally** | `supports_training=False` (or `--no-training`) + `--weights <dir>` | bundled | you, before export |

- **Train-on-platform.** The platform runs training as an `allora-worker train` job on a schedule the operator configures (it is not an in-process timer). Each run skips retraining if the current artifact is younger than 12h (unless `FORCE_RETRAIN=true`). Training only runs while `supports_training` is true. **Before the first successful training run there is no artifact**, and the generated worker's inference raises `model artifact not found` until one exists — expect the first inferences to fail until training completes and writes weights.
- **Train-locally.** `supports_training=False` means the platform never retrains; it serves the weights you bundled (imported into storage by the platform's import step). Updating those weights means re-exporting/re-importing — the generated worker does not hot-reload weights in this mode (the SDK's model watcher only runs for models that report a watchable artifact, which the generated model ties to `supports_training`).

#### Deployment env vars

`pair`/`timeframe`/`topic` are **deploy-time** parameters the operator injects as env vars, never baked into the package. The hosted worker reads:

| Env var | Purpose | Default |
|---------|---------|---------|
| `PAIR` | Trading pair, e.g. `BTCUSD` | required |
| `TIMEFRAME` | Bar interval, e.g. `5m`, `1h` | required |
| `ALLORA_TOPIC_ID` | Target topic | `69` |
| `ALLORA_API_KEY` | Required only for the `allora` data source | — |
| `SUBMIT_RETURNS` | `true`/`false` to force log-return vs price output; unset/`auto` derives it from the topic's on-chain loss method | `auto` |
| `DATA_BASE_PATH` | Where the worker reads/writes model artifacts | `./data` |

</details>

## Development and references

### Testing

Run `pytest tests/test_data_managers.py -v -m "not integration"` for the data-manager
unit checks. Network integration tests require `RUN_INTEGRATION_TESTS=1` and an
exported `ALLORA_API_KEY`.

For triple barrier, run `RUN_INTEGRATION_TESTS=1 python -m pytest tests/test_triple_barrier.py -v -s`.
Its three tests build a real Atlas dataset and independently replay minute targets,
run the complete example and reload its artifact, then deploy an isolated testnet
worker and verify a labeled submission. They clean up their own files and worker.
Add `-k 'not deploy_worker'` for data and example verification only.

### Model creation skills

The [`allora_research_model_skills/`](allora_research_model_skills/README.md) bundle contains three Claude Code skills for building financial prediction models. Each enters model design from a different angle:

| Skill | Entry point |
|-------|-------------|
| `forge-hypothesis-driven` | Start from a theory about what moves markets (deductive) |
| `forge-signal-discovery` | Start from interesting data, discover what is predictable (inductive) |
| `forge-robustness-first` | Start from validation gates, work backwards to a design that survives them (adversarial) |

All three produce a complete, runnable pipeline and satisfy the same nine methodology principles. See [`allora_research_model_skills/README.md`](allora_research_model_skills/README.md) for selection guidance.

### Module map

| Module | Purpose |
|---|---|
| [workflow.py](allora_forge_builder_kit/workflow.py) | Data, features, and all three target builders |
| [evaluation.py](allora_forge_builder_kit/evaluation.py) | Return and triple-barrier evaluation |
| [engineered_features.py](allora_forge_builder_kit/engineered_features.py) | Shared training/inference feature transformations |
| [topic_discovery.py](allora_forge_builder_kit/topic_discovery.py) | Network topic metadata |
| [worker_manager.py](allora_forge_builder_kit/worker_manager.py) | Wallets and worker lifecycle |
| [worker_runtime.py](allora_forge_builder_kit/worker_runtime.py) | Scalar and labeled inference submission |
| [workerctl.py](allora_forge_builder_kit/workerctl.py) | CLI operations |
| [web_dashboard.py](allora_forge_builder_kit/web_dashboard.py) | Monitoring UI |
| [Feature example](notebooks/feature_engineering_example.py) | Feature engineering walkthrough |

### Links

- [Allora Network](https://allora.network)
- [Allora Explorer](https://explorer.allora.network)
- [Developer Portal](https://developer.allora.network)
- [Testnet Faucet](https://faucet.testnet.allora.run)
- [Discord](https://discord.gg/allora)

## License

MIT
