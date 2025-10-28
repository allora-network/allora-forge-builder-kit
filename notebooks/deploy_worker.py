#!/usr/bin/env python3
"""
Deploy trained model to Allora Network as a worker
"""

import os
import asyncio
from allora_sdk.worker import AlloraWorker

# Configuration
TOPIC_ID = 69
PREDICT_PKL = "predict.pkl"
API_KEY_FILE = ".allora_api_key"

# Read API key
api_key_path = os.path.join(os.path.dirname(__file__), API_KEY_FILE)
with open(api_key_path, 'r') as f:
    api_key = f.read().strip()

async def main():
    """Run the Allora worker with the trained model"""
    print(f"Starting Allora worker for Topic {TOPIC_ID}...")
    print(f"Using model: {PREDICT_PKL}")
    
    worker = AlloraWorker(
        topic_id=TOPIC_ID,
        predict_pkl=PREDICT_PKL,
        api_key=api_key,
        debug=True  # Enable debug logging
    )
    
    print("\n✅ Worker initialized. Submitting predictions...")
    
    async for result in worker.run():
        if isinstance(result, Exception):
            print(f"❌ Error: {result}")
        else:
            print(f"✅ Prediction submitted: {result.prediction}")

if __name__ == "__main__":
    asyncio.run(main())

