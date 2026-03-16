---
name: allora-worker-manager
description: "Manage multiple Allora SDK workers by topic/address with lightweight local state. Use for add/deploy/start/stop/replace and health checks."
---

# Allora Worker Manager

Use `WorkerManager` when running multiple topics and ensuring one worker per `(topic_id, address)`.

## Core Rules

- Unique worker key is `(topic_id, address)`.
- Deploy with only `topic_id + artifact` will:
  1) reuse an existing managed address not used for that topic,
  2) otherwise create a new wallet and address automatically.
- Deploying to an existing `(topic_id, address)` requires `replace=True`.

## Quick Start

```python
from pathlib import Path
from allora_forge_builder_kit import WorkerManager

wm = WorkerManager(db_path="worker_state.db", secrets_path="worker_secrets.json")

result = wm.deploy_worker(topic_id=69, artifact_path=Path("predict.pkl"))
print(result.message)

wm.start_all()
print(wm.health_all())
```

## Key management

- Wallet mnemonics are stored as individual files in `worker_keys/` (chmod 600).
- `worker_secrets.json` maps identity aliases to `{address, key_file}` — never contains raw mnemonics.
- The runtime reads the key file directly via `AlloraWalletConfig(mnemonic_file=...)`.
- Users can inspect their mnemonic by reading the key file: `cat worker_keys/<alias>.key`

## Monitoring

```python
from allora_forge_builder_kit import WorkerMonitor, AlloraSDKEventFetcher

monitor = WorkerMonitor(db_path="worker_state.db", event_fetcher=AlloraSDKEventFetcher())
wm.attach_monitor(monitor, sync_now=True)

# CLI dashboard
# python -m allora_forge_builder_kit.workerctl dashboard

# Web dashboard
# python -m allora_forge_builder_kit.web_dashboard --host 0.0.0.0 --port 8787
```
