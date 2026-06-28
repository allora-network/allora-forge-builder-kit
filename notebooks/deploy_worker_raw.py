#!/usr/bin/env python3
"""
Bare-bones worker deployment using the Allora SDK directly.

This shows the minimal path from a saved predict.pkl to a running worker,
with no WorkerManager, no wallet lifecycle management, and no monitoring.
Useful as a reference or for quick one-off deployments.

For production use, see deploy_worker.py which uses WorkerManager for
wallet creation, faucet funding, process management, and the web dashboard.

Targets the branch allora_sdk worker API (AlloraWorker.inferer + a RunContext-based
run fn), matching worker_runtime.py.
"""

import os
import asyncio
import math
import traceback
import cloudpickle
from allora_sdk.worker import AlloraWorker

# Configuration
TOPIC_ID = 69
PREDICT_PKL = "predict.pkl"
API_KEY_FILE = ".allora_api_key"
DEBUG_MODE = True

# SECURITY NOTE: cloudpickle.load executes arbitrary code. Only load pickle
# files that you created yourself. Never load untrusted pickle files.
print(f"Loading model from {PREDICT_PKL}...")
with open(PREDICT_PKL, "rb") as f:
    predict_fn = cloudpickle.load(f)
print("Model loaded")

# Read API key — env var takes priority, then checks key file in notebooks/ then repo root
api_key = os.environ.get("ALLORA_API_KEY", "").strip()
if not api_key:
    _search_paths = [
        os.path.join(os.path.dirname(__file__), API_KEY_FILE),
        os.path.join(os.path.dirname(__file__), "..", API_KEY_FILE),
    ]
    for _path in _search_paths:
        if os.path.exists(_path):
            with open(_path) as f:
                api_key = f.read().strip()
            if api_key:
                break
if not api_key:
    raise RuntimeError(
        "ALLORA_API_KEY not found. Set the env var or create a .allora_api_key file."
    )


def _run_fn(ctx):
    # The branch SDK invokes the inferer callback with a RunContext; the pickled model fn
    # takes the integer nonce, so adapt via ctx.nonce.
    value = predict_fn(ctx.nonce)
    # Validate before returning so a NaN/Inf/non-numeric prediction fails loudly here instead of
    # being silently submitted to the network (mirrors worker_runtime's production validation).
    try:
        v = float(value)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Invalid inference output type: {value!r}") from e
    if not math.isfinite(v):
        raise RuntimeError(f"Invalid inference output (non-finite): {v}")
    return v


async def main():
    """Run the Allora worker with the trained model."""
    print(f"\nStarting Allora worker for Topic {TOPIC_ID}...")

    worker = AlloraWorker.inferer(
        run=_run_fn,
        topic_id=TOPIC_ID,
        api_key=api_key,
        debug=DEBUG_MODE,
    )

    print("Worker initialized. Submitting predictions...")

    async for result in worker.run():
        if isinstance(result, Exception):
            print(f"Error: {result!r} ({type(result).__name__})")
            tb = "".join(traceback.format_exception(type(result), result, result.__traceback__))
            print("--- exception traceback start ---")
            print(tb)
            print("--- exception traceback end ---")
        else:
            print(f"Prediction submitted: {result.prediction}")


if __name__ == "__main__":
    asyncio.run(main())
