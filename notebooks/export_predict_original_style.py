#!/usr/bin/env python3
"""Export topic 69 predict.pkl using original example-style callable (`run`)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cloudpickle
import joblib
import numpy as np
import pandas as pd

from allora_forge_builder_kit import AlloraMLWorkflow

TICKERS = ["BTCUSDT"]
INTERVAL = "1h"
NUMBER_OF_INPUT_BARS = 48
TARGET_BARS = 24
DATA_SOURCE = "binance"
MARKET = "futures"

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "topic69_model.joblib"
COLS_PATH = ROOT / "artifacts" / "topic69_columns.json"

model = joblib.load(MODEL_PATH)
with open(COLS_PATH, "r") as f:
    cols = json.load(f)
base_feature_cols = cols["base_feature_cols"]
feature_cols = cols["feature_cols"]

_WORKFLOW = None


def _get_workflow() -> AlloraMLWorkflow:
    global _WORKFLOW
    if _WORKFLOW is None:
        _WORKFLOW = AlloraMLWorkflow(
            tickers=TICKERS,
            number_of_input_bars=NUMBER_OF_INPUT_BARS,
            target_bars=TARGET_BARS,
            interval=INTERVAL,
            data_source=DATA_SOURCE,
            market=MARKET,
        )
    return _WORKFLOW


def engineer_returns(live_row_df: pd.DataFrame) -> pd.Series:
    closes = np.array([float(live_row_df[f"feature_close_{i}"].iloc[0]) for i in range(NUMBER_OF_INPUT_BARS)], dtype=float)
    eps = 1e-8
    return pd.Series(
        {
            "log_return_1h": float(np.log(closes[-1] + eps) - np.log(closes[-2] + eps)),
            "log_return_6h": float(np.log(closes[-1] + eps) - np.log(closes[-7] + eps)) if NUMBER_OF_INPUT_BARS >= 7 else 0.0,
            "log_return_12h": float(np.log(closes[-1] + eps) - np.log(closes[-13] + eps)) if NUMBER_OF_INPUT_BARS >= 13 else 0.0,
            "log_return_24h": float(np.log(closes[-1] + eps) - np.log(closes[-25] + eps)) if NUMBER_OF_INPUT_BARS >= 25 else 0.0,
        }
    )


def run(nonce: int | None = None) -> float:
    workflow = _get_workflow()
    live_row = workflow.get_live_features(ticker=TICKERS[0])
    if live_row is None or len(live_row) == 0:
        raise ValueError("Could not get live features")

    live_returns = engineer_returns(live_row)
    live_features = pd.concat([live_row[base_feature_cols].iloc[0], live_returns])

    now = datetime.now(timezone.utc)
    raw_data = workflow.load_raw(start=now - timedelta(hours=2), end=now)
    if raw_data is None or len(raw_data) == 0:
        raise ValueError("Could not get current price from raw data; aborting inference")

    current_price = float(raw_data["close"].iloc[-1])
    if not np.isfinite(current_price) or current_price <= 0:
        raise ValueError(f"Invalid current price for inference: {current_price}")

    predicted_log_return = float(model.predict(live_features[feature_cols].values.reshape(1, -1))[0])
    predicted_price = float(current_price * np.exp(predicted_log_return))

    if not np.isfinite(predicted_price) or predicted_price <= 0:
        raise ValueError(f"Invalid predicted price: {predicted_price}")

    return predicted_price


if __name__ == "__main__":
    out_notebook = Path("predict.pkl")
    out_root = ROOT / "predict.pkl"

    with open(out_notebook, "wb") as f:
        cloudpickle.dump(run, f)
    with open(out_root, "wb") as f:
        cloudpickle.dump(run, f)

    print("✅ Saved original-style run callable to:")
    print(f" - {out_notebook}")
    print(f" - {out_root}")
