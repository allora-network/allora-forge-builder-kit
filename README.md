# Allora Forge Builder Kit

Train, evaluate, and deploy Allora prediction workers from one Python toolkit.

## What you get
- **Workflow API** for data → features/targets → training dataset
- **Evaluation** aligned with Allora scoring expectations
- **Deployment tooling** for local worker runtime + monitoring dashboard
- **Out-of-box examples for whitelist-free topics:** **69** and **77**

---

## Zero to deploy (complete walkthrough)

Follow these steps in order to go from a fresh clone to live workers with a
monitoring dashboard. The whole process runs in one terminal session.

### Step 1 — Clone and install

```bash
git clone https://github.com/allora-network/allora-forge-builder-kit.git
cd allora-forge-builder-kit

# Requires Python 3.10+. On macOS, use python3.11 or python3.12 explicitly
# if `python3 --version` shows 3.9.
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

export ALLORA_API_KEY="UP-..."   # Free key from https://developer.allora.network
```

> **No API key?** You can use `data_source="binance"` in the workflow to pull data directly from Binance instead.

### Step 2 — Train a model

Run one (or both) of the example walkthroughs. Each backfills historical data,
engineers features, grid-searches hyperparameters, evaluates the model, and
saves a `predict.pkl` artifact.

```bash
cd notebooks

# Topic 69 — 1-day BTC/USD price prediction (1h bars, ~3 min)
python example_topic_69_bitcoin_walkthrough.py

# Topic 77 — 5-min BTC/USD price prediction (5m bars, ~2 min)
python example_topic_77_bitcoin_5min_walkthrough.py
```

When each script finishes you will see a summary with the evaluation grade and
a `predict.pkl` file in the current directory.

### Step 3 — Deploy workers

The deploy scripts use `WorkerManager` to handle wallet creation, key files,
faucet funding, and process lifecycle automatically — no interactive prompts.

```bash
# Still in the notebooks/ directory.
# Deploy topic 69 (uses the predict.pkl from step 2)
python deploy_worker.py

# Deploy topic 77
python deploy_worker_topic_77.py
```

Each script prints the assigned wallet address and confirms the worker is
running. Workers poll the chain for open nonces and submit predictions
automatically.

### Step 4 — Monitor with the web dashboard

```bash
# From the notebooks/ directory:
python -m allora_forge_builder_kit.web_dashboard
```

Then open **http://localhost:8787** in your browser. You will see all deployed
workers, their submission timelines, on-chain scores, and live log tails.
The page auto-refreshes every 5 seconds.

> **CLI alternative:** `python -m allora_forge_builder_kit.workerctl dashboard`
> prints a text summary to the terminal.

> **Remote access:** Pass `--host 0.0.0.0` to bind to all interfaces. An auth
> token is auto-generated and printed to stderr — append it as `?token=...` in
> the URL.

### Step 5 — Deploy more topics (optional)

```bash
# Deploy any topic by ID
TOPIC_ID=42 python deploy_worker.py
```

---

## Minimal path map

### Notebook/script path (fastest)
- `notebooks/example_topic_69_bitcoin_walkthrough.py`
- `notebooks/example_topic_77_bitcoin_5min_walkthrough.py`
- `notebooks/deploy_worker.py`

### Python API path (modular)
- `allora_forge_builder_kit/workflow.py`
- `allora_forge_builder_kit/evaluation.py`
- `allora_forge_builder_kit/topic_discovery.py`

### Worker operations path (optional)
- `allora_forge_builder_kit/worker_manager.py`
- `allora_forge_builder_kit/worker_monitor.py`
- `allora_forge_builder_kit/web_dashboard.py`

Use one path or combine them.

---

## Agent docs
- `AGENTS.md` – exact runbook for coding agents
- `SKILLS.md` – task routing for model/data/deploy work

If you are an agent, start with `AGENTS.md`.

---

## Testing

```bash
# fast tests
pytest tests/test_data_managers.py -v -m "not integration"

# full suite (integration)
export RUN_INTEGRATION_TESTS=1
pytest -v
```

---

## License
MIT
