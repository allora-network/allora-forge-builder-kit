#!/usr/bin/env python3
"""Build a robust predict.pkl for topic 69 worker deployment.

The function avoids notebook/workflow object closures and only depends on
requests + numpy, which are present in deployment images.
"""

from __future__ import annotations

import cloudpickle
import numpy as np
import requests


def predict(nonce: int | None = None) -> float:
    """Predict BTCUSDT price 24h ahead using a simple momentum baseline.

    Strategy:
    - pull recent 1h candles from Binance futures
    - compute 24h log return from closes
    - apply capped momentum continuation to current price
    """
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 80}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    rows = r.json()

    if not rows or len(rows) < 30:
        raise ValueError("Insufficient candle data from Binance")

    closes = np.array([float(k[4]) for k in rows], dtype=float)
    if not np.isfinite(closes).all():
        raise ValueError("Non-finite close values in candle data")

    current = float(closes[-1])
    prev_24h = float(closes[-25])
    if current <= 0 or prev_24h <= 0:
        raise ValueError("Invalid price values for inference")

    log_ret_24h = float(np.log(current / prev_24h))

    # conservative continuation with clipping to avoid extreme outputs
    projected_log_ret = float(np.clip(0.35 * log_ret_24h, -0.08, 0.08))
    predicted_price = float(current * np.exp(projected_log_ret))

    if not np.isfinite(predicted_price) or predicted_price <= 0:
        raise ValueError("Predicted price is invalid")

    return predicted_price


if __name__ == "__main__":
    with open("predict.pkl", "wb") as f:
        cloudpickle.dump(predict, f)
    print("✅ Wrote predict.pkl with safe Binance-based predictor")
