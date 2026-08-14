<img width="100%" alt="forge_silicon" src="https://github.com/user-attachments/assets/f1444abf-e649-4e48-a9f0-187b78b59ccc" />

# Allora Forge Builder Kit

Build, evaluate, and deploy ML inference workers on the [Allora Network](https://allora.network).

> [!IMPORTANT]
> **SDK v10 — Testnet only.** This version of the builder kit targets the Allora testnet after its emissions v9 → v10 chain upgrade. It requires `allora-sdk>=1.3.0`, now available on PyPI.
>
> **Testnet / v10 install:**
> ```bash
> pip install "allora-forge-builder-kit @ git+https://github.com/allora-network/allora-forge-builder-kit.git@main"
> ```
>
> **Mainnet users** — the network is still on emissions v9. Use the last stable builder kit release:
> ```bash
> pip install "allora-forge-builder-kit @ git+https://github.com/allora-network/allora-forge-builder-kit.git@8ef3200"
> ```

## Contents

- [What is Allora?](#what-is-allora)
- [What is the Allora Forge?](#what-is-the-allora-forge)
- [What you get](#what-you-get)
- [Zero to deploy](#zero-to-deploy)
- [Topic reference](#topic-reference)
- [Wallet linking](#wallet-linking)
- [Deploy to the hosting platform (export)](#deploy-to-the-hosting-platform-export)
- [Python API (quick reference)](#python-api-quick-reference)
- [The learning problem](#the-learning-problem)
- [Evaluation metrics](#evaluation-metrics)
- [Model creation skills](#model-creation-skills)
- [File map](#file-map)
- [Testing](#testing)
- [Links](#links)

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

All live topics today are crypto market predictions across assets like BTC, ETH, SOL, and NEAR. New topics are added over time.

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

## Zero to deploy

### Step 1 — Clone and install

```bash
git clone https://github.com/allora-network/allora-forge-builder-kit.git
cd allora-forge-builder-kit

python3.11 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,wallet-link]"
```

Get a free API key from [developer.allora.network](https://developer.allora.network) and save it:

```bash
echo "UP-..." > .allora_api_key

# Load into env without displaying the value
export ALLORA_API_KEY=$(cat .allora_api_key)
```

To persist across terminal sessions, add to your shell profile:

```bash
echo 'export ALLORA_API_KEY=$(cat /path/to/allora-forge-builder-kit/.allora_api_key)' >> ~/.bashrc
```

> **No API key?** Use `data_source="binance"` in `AlloraMLWorkflow()` to pull data from Binance instead.

### Step 2 — Train a model

```bash
cd notebooks

# Topic 69 — 1-day BTC/USD price prediction (1h bars, ~3 min)
python example_topic_69_bitcoin_walkthrough.py

# Topic 77 — 5-min BTC/USD price prediction (5m bars, ~2 min)
python example_topic_77_bitcoin_5min_walkthrough.py
```

Each script backfills historical data, engineers features, trains and evaluates a model, and saves a `predict.pkl` artifact.

### Step 3 — Deploy a worker

```bash
# Still in notebooks/
python deploy_worker.py
```

On first run, `WorkerManager` creates a wallet, writes the key file to `worker_keys/`, and requests testnet ALLO from the faucet automatically. The worker process starts and begins polling the chain for open submission windows.

> **Faucet activity is logged, not printed.** If a worker fails to start, check `worker_logs/` for the subprocess output — faucet requests, balance checks, and on-chain errors all appear there.

### Step 4 — Monitor and manage workers

```bash
# Web dashboard (recommended)
python -m allora_forge_builder_kit.web_dashboard
```

Open **http://localhost:8787** — auto-refreshes every 5 seconds, shows all workers with submission timelines, on-chain scores, and live log tails.

> Pass `--host 0.0.0.0` to expose on all interfaces. An auth token is printed to stderr; append it as `?token=...` in the URL.

```bash
# CLI dashboard — text summary of all workers
python -m allora_forge_builder_kit.workerctl dashboard
```

**Worker management via the Python API:**

```python
from allora_forge_builder_kit import WorkerManager

wm = WorkerManager(reconcile_on_start=False)

# See all workers and their status
for w in wm.status_all():
    print(w['topic_id'], w['address'], w['status'])

# Stop a worker (keeps it registered, can be restarted)
wm.stop_worker(topic_id=69, address="allo1...")

# Start a stopped worker
wm.start_worker(topic_id=69, address="allo1...")

# Remove a worker entirely (stops it and deletes the record)
wm.remove_worker(topic_id=69, address="allo1...", force=True)

# Stop all running workers
wm.stop_all()

# Restart all enabled workers (e.g. after a reboot)
wm.start_all()

# Tail a worker's log
lines = wm.get_worker_log_tail(topic_id=69, address="allo1...", lines=50)
print("\n".join(lines))
```

> **Managed-custody security:** when `WorkerManager` uses `FORGE_API_KEY`, that key is a
> managed-wallet signing credential, not a transaction-scoped permission. The underlying
> remote signer can sign arbitrary SignDoc bytes and 32-byte digests, so possession of the key
> authorizes any transaction the managed wallet can sign. Disabling Forge's optional
> `/transfer` convenience route does not constrain `/sign`. Protect and revoke the API key as
> carefully as a private wallet key.

### Step 5 — Deploy other topics

```bash
TOPIC_ID=42 python deploy_worker.py   # deploy topic 42
TOPIC_ID=77 python deploy_worker.py   # deploy topic 77
```

Discover available topics:

```python
from allora_forge_builder_kit import AlloraTopicDiscovery

d = AlloraTopicDiscovery(api_key="UP-...", network="testnet")
for t in d.get_all_topics():
    print(t.topic_id, t.raw.get("topic_name"), t.epoch_length, t.loss_method)
```

See [Topic reference](#topic-reference) for all available topics and their prediction types.

---

## Topic reference

Every Allora topic defines a prediction task with a specific **target type** — what the model must output and what the reputer scores against.

**Log-return topics** — predict `log(price[t+H] / price[t])` over a fixed horizon `H`. The output is a dimensionless ratio; positive means "price goes up." Most mainnet topics are log-return.

**Price topics** — predict the absolute price `price[t+H]`. The playground topics (69, 77) use this format and are the recommended starting point.

**Volatility topics** — predict the realized volatility of 1-minute log returns over the horizon: `std(r₁, …, r_H)` where `rᵢ = log(p[t+i] / p[t+i-1])`. The output is a non-negative float. Use `target_type="volatility"` in `AlloraMLWorkflow`.

### Playground topics

No whitelist required — the recommended starting point.

| Testnet ID | Name | Target type | Notes |
|-----------|------|-------------|-------|
| **69** | BTC/USD - 1 Day Price Prediction | Price | Example walkthroughs use this |
| **77** | BTC/USD - 5 Min Price Prediction | Price | Playground Fast |

### Volatility topics

Testnet only; may require whitelist.

| Testnet ID | Name | Target type | Notes |
|-----------|------|-------------|-------|
| **79** | BTC/USD - 15 Min Volatility Prediction | Volatility | Std of 1-min log returns over 15-min horizon |
| **80** | ETH/USD - 15 Min Volatility Prediction | Volatility | Same definition as 79, ETH pair |
| **81** | XRP/USD - 15 Min Volatility Prediction | Volatility | Same definition as 79, XRP pair |
| **82** | SOL/USD - 15 Min Volatility Prediction | Volatility | Same definition as 79, SOL pair |
| **85** | ETH/USD - 4h Volatility Prediction | Volatility | Std of 1-min log returns over 4-hour horizon |

### Mainnet topics and testnet equivalents

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

---

## Wallet linking

When you deploy a **local-custody** worker the signing key lives in `worker_keys/` on your machine, but Forge doesn't know which `allo1...` addresses belong to your account. **Wallet linking** proves ownership: the CLI signs an ADR-036 challenge with each local worker key and a browser-authenticated Forge user approves the link.

> **Managed-custody workers** (deployed with `custody="managed"`) are linked automatically by the backend — no `workerctl link` step needed.

### Custody modes at a glance

| Mode | Key lives | Linking |
|------|-----------|---------|
| **Local** (default) | `worker_keys/` on your machine | Run `workerctl link` once per address |
| **Managed** | Forge backend (Privy wallet) | Automatic — no CLI step |

### Quick start

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

### Link specific addresses

```bash
# Link a single address
workerctl link --address allo1abc...

# Link two specific addresses
workerctl link --address allo1abc... --address allo1def...
```

### Headless / CI environments

```bash
# Print the URL and code without opening a browser
workerctl link --no-browser
```

Output the one-time code and URL to stdout so you can open them on a separate device or paste them into a CI log.

### Non-default secrets file

```bash
workerctl link --secrets-path /path/to/worker_secrets.json
```

### CLI reference

`workerctl link` accepts the following flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--secrets-path PATH` | `worker_secrets.json` | Path to the WorkerManager secrets file that maps addresses to local key files. |
| `--address ADDR` | all local keys | Limit to a specific `allo1...` address. Repeatable — pass once per address. |
| `--no-browser` | off | Print the approval URL and code without auto-opening a browser. |

### Python API

```python
from allora_forge_builder_kit.wallet_link import run_link

rc = run_link(
    secrets_path="worker_secrets.json",
    addresses=None,       # None = link all local keys; pass a list to limit
    open_browser=True,
)
# rc is 0 on success, 1 on any error
```

### Troubleshooting

**`cosmpy` not found** — install the wallet-link extra: `pip install -e ".[wallet-link]"` or `pip install cosmpy==0.11.1`.

**`No worker keys found`** — the secrets file is missing or empty. Deploy a local-custody worker first (`WorkerManager.deploy_worker(...)` or `python deploy_worker.py`).

**`No local key for: allo1...`** — the address is a managed-custody worker (linked automatically) or the secrets file is stale. Managed workers do not need manual linking.

**Link request denied** — the browser approval was rejected. Re-run `workerctl link` to start a fresh session.

**Link request expired** — the 30-minute approval window closed before the browser was used. Re-run to start a new session.

---

## Deploy to the hosting platform (export)

The [Zero to deploy](#zero-to-deploy) flow runs a worker **locally** with `WorkerManager`. The other path is to let the Allora **hosting platform** run the worker for you in a container. Instead of a running process, you produce a *package* — worker code + `pyproject.toml` + `manifest.json` (+ an optional `weights/` dir) — and upload it to forge.

See [`notebooks/export_to_hosting.py`](notebooks/export_to_hosting.py) for a runnable walkthrough. The essentials:

```python
from allora_forge_builder_kit import WorkerManager, ModelSpec

# Model-INTRINSIC config (baked into the package's config.json). Pair/timeframe/
# topic are NOT here — they are chosen per deployment (see env vars below).
spec = ModelSpec(
    model_type="my_lgbm",                          # entry-point name; [a-z0-9][a-z0-9_-]*
    engineered_specs=[{"kind": "log_return", "window_bars": 6}],
    number_of_input_bars=24,
    target_bars=24,
    hyperparameters={"n_estimators": 500},
    data_source="binance",                          # "binance" | "allora"
    supports_training=True,                         # train-on-platform (no weights)
)
wm = WorkerManager(reconcile_on_start=False)
wm.export_payload_for_hosting(spec, out_dir="build/my_lgbm_package")
```

Or from the command line, which can also produce the upload-ready zip:

```bash
workerctl export-payload --config model.json --out build/my_lgbm_package --zip
# then upload build/my_lgbm_package.zip to forge (POST /api/v1/models)
```

`--zip` writes the package **contents** at the archive root, so forge finds `manifest.json` at the extraction root. The generated worker code is **generic over pair/timeframe**; code-only (train-on-platform) packages can be deployed against many pairs/timeframes/topics. Bundled-weight packages must use parameters matching how the weights were trained.

### Two deployment modes

Exactly **one** of these must hold (forge rejects the package otherwise; `export_payload_for_hosting` enforces it and fails loudly):

| Mode | Set | Weights | Who trains |
|------|-----|---------|-----------|
| **Train-on-platform** | `supports_training=True` (default) | none | the platform |
| **Train-locally** | `supports_training=False` (or `--no-training`) + `--weights <dir>` | bundled | you, before export |

- **Train-on-platform.** The platform runs training as an `allora-worker train` job on a schedule the operator configures (it is not an in-process timer). Each run skips retraining if the current artifact is younger than 12h (unless `FORCE_RETRAIN=true`). Training only runs while `supports_training` is true. **Before the first successful training run there is no artifact**, and the generated worker's inference raises `model artifact not found` until one exists — expect the first inferences to fail until training completes and writes weights.
- **Train-locally.** `supports_training=False` means the platform never retrains; it serves the weights you bundled (imported into storage by the platform's import step). Updating those weights means re-exporting/re-importing — the generated worker does not hot-reload weights in this mode (the SDK's model watcher only runs for models that report a watchable artifact, which the generated model ties to `supports_training`).

### Deployment env vars

`pair`/`timeframe`/`topic` are **deploy-time** parameters the operator injects as env vars, never baked into the package. The hosted worker reads:

| Env var | Purpose | Default |
|---------|---------|---------|
| `PAIR` | Trading pair, e.g. `BTCUSD` | required |
| `TIMEFRAME` | Bar interval, e.g. `5m`, `1h` | required |
| `ALLORA_TOPIC_ID` | Target topic | `69` |
| `ALLORA_API_KEY` | Required only for the `allora` data source | — |
| `SUBMIT_RETURNS` | `true`/`false` to force log-return vs price output; unset/`auto` derives it from the topic's on-chain loss method | `auto` |
| `DATA_BASE_PATH` | Where the worker reads/writes model artifacts | `./data` |

---

## Python API (quick reference)

```python
from allora_forge_builder_kit import AlloraMLWorkflow

# Build a training dataset (log-return target — default)
workflow = AlloraMLWorkflow(
    tickers=["btcusd"],
    number_of_input_bars=48,
    target_bars=24,
    interval="1h",
    data_source="allora",
    api_key="UP-...",
)
workflow.backfill(days=500)
df = workflow.get_full_feature_target_dataframe()

# Volatility target (std of 1-min log returns over the horizon)
vol_workflow = AlloraMLWorkflow(
    tickers=["btcusd"],
    number_of_input_bars=15,
    target_bars=15,           # 15-minute volatility window
    interval="1m",
    target_type="volatility", # NEW: "log_return" (default) or "volatility"
    data_source="allora",
    api_key="UP-...",
)

# Evaluate a predict function
from allora_forge_builder_kit import PerformanceEvaluator
evaluator = PerformanceEvaluator()
report = evaluator.evaluate(y_true, y_pred)

# Shared engineered features (identical at train and serve — the anti-skew guard)
from allora_forge_builder_kit import apply_engineered_features, engineered_feature_names
specs = [{"kind": "log_return", "window_bars": 6}]
df, added_cols = apply_engineered_features(df, specs, number_of_input_bars=48)

# Package a model for the hosting platform (see "Deploy to the hosting platform")
from allora_forge_builder_kit import WorkerManager, ModelSpec
wm = WorkerManager(reconcile_on_start=False)
wm.export_payload_for_hosting(ModelSpec(model_type="my_lgbm", engineered_specs=specs,
                                        number_of_input_bars=48, target_bars=24), out_dir="build/pkg")
```

---

## The learning problem

### Framing forecasting as supervised learning

At any point in time $t$, the model observes a window of $N$ past bars as input features $\mathbf{x} \in \mathbb{R}^d$ and predicts a future outcome $y$ over the next $H$ bars. The target $y$ depends on the topic type:

- **Price / log-return topics** — $y = \log(p_{t+H} / p_t)$ or the absolute price $p_{t+H}$
- **Volatility topics** — $y = \text{std}(r_1, \ldots, r_H)$ where $r_i = \log(p_{t+i} / p_{t+i-1})$ are consecutive 1-minute log returns over the horizon

By sliding this window across the full history, a single time series becomes thousands of labeled examples $(\mathbf{x}_i, y_i)$, turning forecasting into a standard supervised learning problem.

The `AlloraMLWorkflow` handles this construction: `backfill()` fetches historical data, `get_full_feature_target_dataframe()` builds the feature matrix and target vector, ready for any scikit-learn compatible model.

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
3. **Maximizing out-of-sample metrics** — the evaluation suite (DA, Pearson $r$, WRMSE, CZAR) is the scorecard, not in-sample loss. A higher grade means better generalization and a higher expected score on the Allora network.

For structured methodology guidance on each of these levers, see the [Model creation skills](#model-creation-skills) section.

---

## Evaluation metrics

`PerformanceEvaluator` scores your model on 7 primary metrics before you deploy. Each has a pass/fail threshold. The composite score (out of 7) maps to a letter grade.

| # | Metric | Threshold | What it measures |
|---|--------|-----------|-----------------|
| 1 | **Directional Accuracy (DA)** | ≥ 52% | Fraction of predictions where the sign (up/down) matches the actual return |
| 2 | **DA CI Lower Bound** | ≥ 0.50 | Lower bound of the 95% Wilson confidence interval for DA, adjusted for autocorrelation — ensures the edge isn't a statistical fluke |
| 3 | **DA p-value** | < 0.05 | One-tailed z-test (H₀: DA = 50%) with continuity correction and autocorrelation-aware effective sample size |
| 4 | **Pearson r** | ≥ 0.05 | Linear correlation between predicted and actual returns |
| 5 | **Pearson p-value** | < 0.05 | Statistical significance of the Pearson correlation |
| 6 | **WRMSE Improvement** | ≥ 5% | Weighted RMSE vs. a zero-prediction baseline, where errors are weighted by the magnitude of actual returns — bigger moves count more |
| 7 | **CZAR Improvement** | ≥ 10% | Cumulative Z-scored Absolute Return: the fraction of z-scored directional return captured vs. a perfect oracle. 0 = random guessing, 1 = perfect |

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

---

## Model creation skills

The [`allora_research_model_skills/`](allora_research_model_skills/README.md) bundle contains three Claude Code skills for building financial prediction models. Each enters model design from a different angle:

| Skill | Entry point |
|-------|-------------|
| `forge-hypothesis-driven` | Start from a theory about what moves markets (deductive) |
| `forge-signal-discovery` | Start from interesting data, discover what is predictable (inductive) |
| `forge-robustness-first` | Start from validation gates, work backwards to a design that survives them (adversarial) |

All three produce a complete, runnable pipeline and satisfy the same nine methodology principles. See [`allora_research_model_skills/README.md`](allora_research_model_skills/README.md) for selection guidance.

---

## File map

| Path | Purpose |
|------|---------|
| `notebooks/example_topic_69_bitcoin_walkthrough.py` | End-to-end example for topic 69: data → features → model → artifact |
| `notebooks/example_topic_77_bitcoin_5min_walkthrough.py` | End-to-end example for topic 77: 5-min BTC prediction |
| `notebooks/testnet/topic_38_sol_8h_price/` | Topic 38 SOL/USD 8h price: example + CZAR model |
| `notebooks/testnet/topic_41_eth_8h_price/` | Topic 41 ETH/USD 8h price: example + CZAR model |
| `notebooks/testnet/topic_42_btc_8h_price/` | Topic 42 BTC/USD 8h price: example + directional + CZAR models |
| `notebooks/testnet/topic_57_sol_8h_logreturn/` | Topic 57 SOL/USD 8h log-return example |
| `notebooks/testnet/topic_61_btc_24h_logreturn/` | Topic 61 BTC/USD 24h log-return example |
| `notebooks/testnet/topic_62_sol_24h_logreturn/` | Topic 62 SOL/USD 24h log-return example |
| `notebooks/testnet/topic_63_eth_24h_logreturn/` | Topic 63 ETH/USD 24h log-return example |
| `notebooks/testnet/topic_71_near_8h_logreturn/` | Topic 71 NEAR/USD 8h log-return example |
| `notebooks/testnet/topic_79_btc_vol/` | Topic 79 BTC/USD 15m volatility: grid-retrain model |
| `notebooks/testnet/topic_80_eth_vol/` | Topic 80 ETH/USD 15m volatility: grid-retrain model |
| `notebooks/testnet/topic_81_xrp_vol/` | Topic 81 XRP/USD 15m volatility: grid-retrain model |
| `notebooks/testnet/topic_82_sol_vol/` | Topic 82 SOL/USD 15m volatility: grid-retrain model |
| `notebooks/testnet/topic_83_btc_8h_logreturn/` | Topic 83 BTC/USD 8h log-return example |
| `notebooks/testnet/topic_84_eth_8h_logreturn/` | Topic 84 ETH/USD 8h log-return example |
| `notebooks/testnet/topic_85_eth_4h_vol/` | Topic 85 ETH/USD 4h volatility: grid-retrain + importance-groups |
| `notebooks/deploy_worker.py` | Deploy any topic with WorkerManager (`TOPIC_ID=N python deploy_worker.py`) |
| `notebooks/deploy_worker_raw.py` | Minimal SDK-only deployment reference (no WorkerManager) |
| `notebooks/feature_engineering_example.py` | Standalone feature engineering reference |
| `notebooks/export_to_hosting.py` | Export a model into a hosting-deployable package (`ModelSpec` → `WorkerManager.export_payload_for_hosting`) |
| `allora_forge_builder_kit/workflow.py` | Data + feature pipeline (`target_type="log_return"` or `"volatility"`) |
| `allora_forge_builder_kit/czar_loss.py` | CZAR directional loss with gradient/hessian for custom LightGBM objectives |
| `allora_forge_builder_kit/engineered_features.py` | Shared engineered-feature computation (train == serve; the guard against skew) |
| `allora_forge_builder_kit/export.py` | Package a model for the hosting platform (`ModelSpec`, internal packaging logic) |
| `allora_forge_builder_kit/evaluation.py` | Model scoring (7 primary metrics + grading) |
| `allora_forge_builder_kit/topic_discovery.py` | Query live topics on testnet/mainnet |
| `allora_forge_builder_kit/worker_manager.py` | Wallet creation, key management, process lifecycle (local + managed custody) |
| `allora_forge_builder_kit/wallet_link.py` | Device-flow wallet linking CLI — ADR-036 signing, invoked via `workerctl link` |
| `allora_forge_builder_kit/workerctl.py` | `workerctl` CLI entry point (dashboard, link, export-payload subcommands) |
| `allora_forge_builder_kit/worker_monitor.py` | On-chain event tracking |
| `allora_forge_builder_kit/web_dashboard.py` | Web monitoring UI |
| `allora_research_model_skills/` | Methodology skills for building generalizable financial models (hypothesis-driven, signal-discovery, robustness-first) |

---

## Testing

```bash
pytest tests/test_data_managers.py -v -m "not integration"

# Full suite (requires network)
export RUN_INTEGRATION_TESTS=1
pytest -v
```

---

## Links

- [Allora Network](https://allora.network)
- [Allora Explorer](https://explorer.allora.network)
- [Developer Portal](https://developer.allora.network)
- [Testnet Faucet](https://faucet.testnet.allora.run)
- [Discord](https://discord.gg/allora)

## License

MIT
