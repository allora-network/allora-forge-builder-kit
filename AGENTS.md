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
4. **Allora API key — STOP AND ASK THE USER. Do not silently proceed.**
   Treat the API key as **human-confirmed input**, not implicit context.
   Never auto-use a discovered key without the user's explicit approval in this session.

   **If a key file or env var already exists**, tell the user what you found and ask:
   > I found an Allora API key at `.allora_api_key`. Should I use this, or do you want to provide a different one?

   **If no key is found anywhere**, stop and prompt with clear options:
   > I need an Allora API key to fetch data. You have three options:
   > 1. **Sign up free** at https://developer.allora.network and paste your key here.
   > 2. **Set it as an env var**: `export ALLORA_API_KEY="UP-..."`
   > 3. **Skip the API key** — I'll use Binance as the data source instead (no key required).

   **Wait for the user to respond before continuing.** Do not proceed to any data, model,
   or deploy step without either a confirmed key or explicit approval to use Binance.
   Once confirmed, write the key to `.allora_api_key` and export it as `ALLORA_API_KEY`.
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
