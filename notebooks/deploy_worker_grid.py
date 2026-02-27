#!/usr/bin/env python3
"""Minimal multi-topic worker deployment example.

This script demonstrates the WorkerManager API for smart address assignment:
- one worker per (topic, address)
- auto-reuse existing addresses when possible
- auto-create new address when needed
"""

from pathlib import Path

from allora_forge_builder_kit import WorkerManager


def main() -> None:
    manager = WorkerManager(
        db_path="worker_state.db",
        secrets_path="worker_secrets.json",
    )

    # Example: deploy with automatic address assignment
    result = manager.deploy_worker(
        topic_id=69,
        topic_desc="BTC 5m price prediction",
        artifact_path=Path("predict.pkl"),
    )
    print(f"[{result.action}] {result.message}")

    # Start and view statuses
    manager.start_all()
    for row in manager.status_all(include_desc=True):
        print(
            f"Topic {row['topic_id']} ({row.get('topic_desc','')}) | "
            f"{row['address']} | {row['status']} | {row['artifact_path']}"
        )


if __name__ == "__main__":
    main()
