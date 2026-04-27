<img width="100%" alt="forge_silicon" src="https://github.com/user-attachments/assets/f1444abf-e649-4e48-a9f0-187b78b59ccc" />

# Allora Forge Builder Kit

Build, evaluate, and deploy ML inference workers on the [Allora Network](https://allora.network).

## What is Allora?

Allora is a decentralized AI network that coordinates predictions across many independent ML models. Rather than relying on a single model, the network aggregates inferences from competing workers and weights them by historical accuracy — producing a combined output that outperforms any individual contributor.

The network is organized into **topics**. Each topic defines a prediction task (e.g. "8-hour BTC/USD log return") and runs a continuous lifecycle:

1. **Submission window opens** — the network pings all registered workers for their inference
2. **Workers respond** with a prediction value
3. **Evaluation window** runs for the topic's time horizon (e.g. 8 hours)
4. **Scores are revealed** — workers are ranked by loss against the ground truth, and rewards are distributed

All live topics today are crypto market predictions across assets like BTC, ETH, SOL, and NEAR. New topics are added over time.

## What is the Allora Forge?

The [Allora Model Forge](https://allora.network/forge) is the hub for ML practitioners to compete, earn rewards, and build reputation on the network. Workers start on testnet to establish a track record, then graduate to mainnet where top performers earn ALLO token rewards.

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

python -m pip install .
python -m pip install -r requirements.txt
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

### Topic reference

Playground topics (testnet only) are the recommended starting point — no whitelist required.

| Testnet ID | Name | Notes |
|-----------|------|-------|
| **69** | BTC/USD - 1 Day Price Prediction | Playground — example walkthroughs use this |
| **77** | BTC/USD - 5 Min Price Prediction | Playground Fast |

Mainnet topics and their testnet equivalents:

| Mainnet ID | Mainnet Name | Testnet ID | Testnet Name |
|-----------|-------------|-----------|-------------|
| 1  | BTC/USD - Log Returns - 8h  | 64 | 8h BTC/USD Log-Return (5min updates) |
| 2  | ETH/USD - Log Returns - 8h  | — | Missing |
| 3  | SOL/USD - Log Returns - 8h  | 57 | 8h SOL/USD Log-Return *(inactive)* |
| 9  | ETH/USD - Price Prediction - 8h | 41 | ETH/USD - 8h Price Prediction |
| 10 | SOL/USD - Price Prediction - 8h | 38 | SOL/USD - 8h Price Prediction |
| 14 | BTC/USD - Price Prediction - 8h | 42 | BTC/USD - 8h Price Prediction |
| 15 | BTC/USD - Log Returns - 24h | 61 | 1 day BTC/USD Log-Return Prediction |
| 16 | ETH/USD - Log Returns - 24h | 63 | 1 day ETH/USD Log-Return Prediction |
| 17 | SOL/USD - Log Returns - 24h | 62 | 1 day SOL/USD Log-Return Prediction |
| 18 | BTC/USD - Log Returns - 20m | — | Missing |
| 19 | NEAR/USD - Log Returns - 8h | 71 | 8h NEAR/USD Log-Return Prediction |

---

## Python API (quick reference)

```python
from allora_forge_builder_kit import AlloraMLWorkflow

# Build a training dataset
workflow = AlloraMLWorkflow(
    tickers=["btcusd"],
    topic_id=69,
    interval="1h",
    n_input_bars=48,
    n_target_bars=24,
)
workflow.backfill(days=500)
df = workflow.get_full_feature_target_dataframe()

# Evaluate a predict function
from allora_forge_builder_kit import PerformanceEvaluator
evaluator = PerformanceEvaluator(workflow)
grade = evaluator.evaluate(predict_fn)
```

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

## File map

| Path | Purpose |
|------|---------|
| `notebooks/example_topic_69_bitcoin_walkthrough.py` | End-to-end example for topic 69: data → features → model → artifact |
| `notebooks/example_topic_77_bitcoin_5min_walkthrough.py` | End-to-end example for topic 77: 5-min BTC prediction |
| `notebooks/deploy_worker.py` | Deploy any topic with WorkerManager (`TOPIC_ID=N python deploy_worker.py`) |
| `notebooks/deploy_worker_raw.py` | Minimal SDK-only deployment reference (no WorkerManager) |
| `notebooks/feature_engineering_example.py` | Standalone feature engineering reference |
| `allora_forge_builder_kit/workflow.py` | Data + feature pipeline |
| `allora_forge_builder_kit/evaluation.py` | Model scoring (7 primary metrics + grading) |
| `allora_forge_builder_kit/topic_discovery.py` | Query live topics on testnet/mainnet |
| `allora_forge_builder_kit/worker_manager.py` | Wallet creation, key management, process lifecycle |
| `allora_forge_builder_kit/worker_monitor.py` | On-chain event tracking |
| `allora_forge_builder_kit/web_dashboard.py` | Web monitoring UI |

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
