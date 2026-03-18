# AGENTS.md

Agent operating guide for this repository.

## Goal
In one working session, get a user from clone to live worker submissions with visible monitoring.

## First-run checklist (mandatory)
1. Read `README.md`.
2. Read `SKILLS.md` (task routing and skills package map).
3. Confirm environment:
   - Python 3.10+ (`python3 --version`). On macOS, use `python3.11` or `python3.12` explicitly if system `python3` is 3.9.
   - virtualenv active
   - `pip install -e ".[dev]"`
4. Confirm API key exists:
   - `ALLORA_API_KEY` env var, or
   - `.allora_api_key` / `notebooks/.allora_api_key`
5. Ask which execution path they want:
   - Notebook/script
   - Python API
   - Worker operations

## Canonical starter flows

### Flow A — Notebook/script (recommended default)
Run one of the whitelist-free examples:

```bash
python notebooks/example_topic_69_bitcoin_walkthrough.py
python notebooks/example_topic_77_bitcoin_5min_walkthrough.py
```

Then deploy with the script matching the topic walkthrough:

```bash
# topic 69 walkthrough
python notebooks/deploy_worker.py

# topic 77 walkthrough
python notebooks/deploy_worker_topic_77.py
```

### Flow B — Python API (modular integration)
Use these modules directly:
- `workflow.py`
- `evaluation.py`
- `topic_discovery.py`

Do not require notebooks when API path is requested.

### Flow C — Worker operations (optional)
Use manager/monitor/dashboard:

```bash
python -m allora_forge_builder_kit.workerctl dashboard
python -m allora_forge_builder_kit.web_dashboard
```

## Critical correctness rule
Before deployment, verify topic prediction format:
- **Price topic** → absolute price prediction
- **Log-return topic** → `log(future/current)` prediction

## Repo hygiene rules
- Never commit secrets or keys.
- Keep generated runtime state out of git.
- Keep docs short and command-oriented.
- Prefer explicit, reproducible commands over prose.

## Done criteria
A task is done when:
1. model path runs end-to-end,
2. worker submits successfully,
3. monitoring/dashboard confirms live flow,
4. docs still let a new agent repeat this without ambiguity.
