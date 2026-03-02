#!/usr/bin/env python3
"""
Deploy trained model to Allora Network as a worker.

Compatible with allora_sdk >= 1.0.6.
"""

import os
import asyncio
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

# Read API key (prefer ALLORA_API_KEY env var over plaintext file)
api_key = os.environ.get("ALLORA_API_KEY")
if not api_key:
    api_key_path = os.path.join(os.path.dirname(__file__), API_KEY_FILE)
    with open(api_key_path, "r") as f:
        api_key = f.read().strip()


async def main():
    """Run the Allora worker with the trained model."""
    print(f"\nStarting Allora worker for Topic {TOPIC_ID}...")

    worker = AlloraWorker(
        run=predict_fn,
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

