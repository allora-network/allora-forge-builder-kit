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
  2) otherwise create a new identity/address.
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

## Notes

- Secrets are stored in `worker_secrets.json` (chmod 600 when possible).
- Prefer environment/keychain integration for production secret storage.
