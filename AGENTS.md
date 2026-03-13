# AGENTS.md

Agent operating guide for **Allora Forge Builder Kit**.

## Mission

Get a fresh user from clone → trained/evaluated model → deployed worker → live monitoring in one short session, with minimal ambiguity.

## First-session protocol (mandatory)

1. Read `README.md` (quickstart + path selection).
2. Read `SKILLS.md` (task routing).
3. Confirm environment setup:
   - Python 3.10+
   - virtualenv active
   - `pip install -e ".[dev]"`
4. Confirm API key is available (`ALLORA_API_KEY` or `.allora_api_key` file).
5. Ask user which path they want:
   - Notebook path
   - Python API path
   - Local worker manager path

Do **not** force one path when another is sufficient.

## Modular architecture rule

The toolkit is intentionally modular:
- Training/evaluation can run without worker manager.
- Worker manager can run with prebuilt artifacts.
- Topic discovery can be used standalone.

When editing docs/code, preserve this separation.

## Canonical execution paths

### Path A — Notebook/script operator

Primary files:
- `notebooks/example_topic_69_bitcoin_walkthrough.py`
- `notebooks/deploy_worker.py`

Success criteria:
- Feature/target dataframe generated
- Evaluation report produced
- Deployable predict artifact created
- Worker deployment command succeeds

### Path B — Python API integrator

Primary modules:
- `allora_forge_builder_kit/workflow.py`
- `allora_forge_builder_kit/evaluation.py`
- `allora_forge_builder_kit/topic_discovery.py`

Success criteria:
- API-only script runs without notebook dependency
- Inputs/outputs are explicit and typed where possible

### Path C — Local worker operations

Primary modules:
- `allora_forge_builder_kit/worker_manager.py`
- `allora_forge_builder_kit/worker_runtime.py`
- `allora_forge_builder_kit/worker_monitor.py`
- `allora_forge_builder_kit/web_dashboard.py`

Success criteria:
- Local workers can be started/stopped/reconciled
- Monitoring shows inference/submission flow
- Dashboard view works for active workers

## Repo hygiene rules

Before merge/PR, agents should check:

1. `git status --short` and identify accidental runtime artifacts.
2. Never commit secrets (`*.key`, `worker_secrets.json`, tokens).
3. Keep generated state out of git (`worker_state.db`, local artifacts, env dirs).
4. Keep README concise; put deep details in skill docs.
5. Keep docs consistent with actual module names and commands.

## Validation checklist

Run at least:

```bash
pytest tests/test_data_managers.py -v -m "not integration"
```

If integration scope is requested:

```bash
export RUN_INTEGRATION_TESTS=1
pytest -v
```

## Topic prediction correctness

Always verify topic type before deployment:
- **Price topic** → predict absolute price
- **Log-return topic** → predict `log(future/current)`

Use topic metadata/discovery to decide format.

## Definition of done (release prep)

- README is short, accurate, and action-oriented.
- `AGENTS.md` + `SKILLS.md` are present and up to date.
- A first-time agent can complete setup and first deployment from docs only.
- Untracked/dirty local runtime noise is either intentionally tracked or ignored.
- User can monitor live submissions with local dashboard/monitor tools.
