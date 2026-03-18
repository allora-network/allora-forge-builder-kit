#!/usr/bin/env python3
"""
Deploy trained Topic 77 model via WorkerManager.

Usage:
    python notebooks/deploy_worker_topic_77.py

Expects predict.pkl in the current directory (output of the walkthrough).
The WorkerManager handles wallet creation, key management, and process
lifecycle automatically — no interactive prompts required.
"""

from pathlib import Path
from allora_forge_builder_kit import WorkerManager, WorkerMonitor, AlloraSDKEventFetcher

TOPIC_ID = 77
PREDICT_PKL = "predict.pkl"

artifact = Path(PREDICT_PKL)
if not artifact.exists():
    raise FileNotFoundError(
        f"{PREDICT_PKL} not found. Run the Topic 77 walkthrough first:\n"
        "  python notebooks/example_topic_77_bitcoin_5min_walkthrough.py"
    )

wm = WorkerManager()

print(f"Deploying worker for Topic {TOPIC_ID}...")
result = wm.deploy_worker(topic_id=TOPIC_ID, artifact_path=artifact)
print(f"  {result.message}")
print(f"  Address: {result.address_assigned}")

monitor = WorkerMonitor(event_fetcher=AlloraSDKEventFetcher())
wm.attach_monitor(monitor)

print("Starting worker...")
wm.start_worker(TOPIC_ID, result.address_assigned)

status = wm.status_worker(TOPIC_ID, result.address_assigned)
print(f"  Status: {status['status']}")
print(f"  PID: {status.get('last_pid')}")
print(f"  Log: worker_logs/worker_{TOPIC_ID}_{result.address_assigned}.log")

print(f"\nWorker running. Monitor with:")
print(f"  python -m allora_forge_builder_kit.workerctl dashboard")
print(f"  python -m allora_forge_builder_kit.web_dashboard")
