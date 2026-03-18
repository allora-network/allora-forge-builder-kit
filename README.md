# Allora Forge Builder Kit

Train, evaluate, and deploy Allora prediction workers from one Python toolkit.

## What you get
- **Workflow API** for data → features/targets → training dataset
- **Evaluation** aligned with Allora scoring expectations
- **Deployment tooling** for local worker runtime + monitoring dashboard
- **Out-of-box examples for whitelist-free topics:** **69** and **77**

---

## Quick start (one session)

```bash
git clone https://github.com/allora-network/allora-forge-builder-kit.git
cd allora-forge-builder-kit

# Requires Python 3.10+. On macOS, use python3.11 or python3.12 explicitly
# if `python3 --version` shows 3.9.
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

export ALLORA_API_KEY="UP-..."
```

Run a full example:

```bash
# topic 69 example
python notebooks/example_topic_69_bitcoin_walkthrough.py

# topic 77 example
python notebooks/example_topic_77_bitcoin_5min_walkthrough.py
```

---

## Deploy + monitor locally

```bash
# generic deploy flow
python notebooks/deploy_worker.py

# topic-specific deploy helpers
python notebooks/deploy_worker_topic_77.py

# multi-worker dashboard
python -m allora_forge_builder_kit.workerctl dashboard
# optional web dashboard
python -m allora_forge_builder_kit.web_dashboard
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
