# Allora Forge Builder Kit

Build, evaluate, and deploy Allora worker models with a **modular** workflow.

Use only what you need:
- **Notebook path**: train + evaluate + export prediction artifact.
- **Python API path**: integrate into your own scripts/pipelines.
- **Worker manager path**: run multiple workers locally with monitoring/dashboard.

No part of the kit is required to use the others.

---

## 1) Quick start (10 minutes)

```bash
# Clone + enter
git clone https://github.com/allora-network/allora-forge-builder-kit.git
cd allora-forge-builder-kit

# Python env
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"
```

Set API key (either env var or file):

```bash
export ALLORA_API_KEY="UP-..."
# OR write to notebooks/.allora_api_key
```

Run one example end-to-end:

```bash
python notebooks/example_topic_69_bitcoin_walkthrough.py
```

---

## 2) Choose your path

### A) Notebook-first (fastest for humans/agents)

Use notebooks/scripts in `notebooks/` to:
1. fetch/backfill data,
2. build features/targets,
3. train + evaluate,
4. export a `predict` artifact,
5. deploy worker.

Start with:
- `notebooks/example_topic_69_bitcoin_walkthrough.py`
- `notebooks/deploy_worker.py`

### B) Python API (embed in your own system)

```python
from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator

workflow = AlloraMLWorkflow(
    tickers=["btcusd"],
    number_of_input_bars=48,
    target_bars=24,
    interval="1h",
    data_source="allora",  # atlas-backed
    api_key="UP-...",
)

workflow.backfill(start="2025-01-01")
df = workflow.get_full_feature_target_dataframe()

# train model -> y_pred
# report = PerformanceEvaluator().evaluate(y_true, y_pred)
```

### C) Local worker operations (multi-worker + dashboard)

Use manager/runtime/monitor modules for local deployment lifecycle.

```bash
# CLI dashboard (from module)
python -m allora_forge_builder_kit.workerctl dashboard

# Include stopped workers
python -m allora_forge_builder_kit.workerctl dashboard --all
```

Web dashboard (if enabled in your flow):

```bash
python -m allora_forge_builder_kit.web_dashboard
# opens local status server/UI
```

---

## 3) Repository map

```text
allora_forge_builder_kit/
  workflow.py            # Data -> features/targets orchestration
  evaluation.py          # Allora-aligned scoring + grading
  topic_discovery.py     # Discover active topics + metadata
  worker_manager.py      # Local worker lifecycle manager
  worker_runtime.py      # Managed worker runtime entrypoint
  worker_monitor.py      # Submission/inference monitoring
  web_dashboard.py       # Local dashboard server

notebooks/
  example_*              # End-to-end training examples
  deploy_worker*.py      # Deployment scripts
  *_sweep.py             # Model/feature experiments

skills/
  allora-model-builder/
  allora-data-exploration/
  allora-worker-manager/
```

---

## 4) Agent-first docs

- `AGENTS.md` → exact operating guide for coding agents.
- `SKILLS.md` → which skill to use for each task.
- `skills/*/SKILL.md` → task-specific runbooks.

If you are an agent: start with `AGENTS.md`.

---

## 5) Data sources

- `allora` / `atlas` (default): Atlas-backed candles (API key required)
- `binance`: REST/WebSocket market data (no key required for public data)

---

## 6) Testing

```bash
# Fast unit tests
pytest tests/test_data_managers.py -v -m "not integration"

# Full tests (network/integration)
export RUN_INTEGRATION_TESTS=1
pytest -v
```

---

## 7) Professional release checklist (suggested)

Before merging to `main`:
1. Remove local runtime artifacts (`.venv*`, `worker_state.db`, secrets files, temp artifacts).
2. Ensure `.gitignore` covers generated files.
3. Run tests and capture result summary in PR.
4. Add a short "How to launch in one session" section in PR description.
5. Verify notebook path and API path both work independently.

---

## License

MIT — see `LICENSE`.
