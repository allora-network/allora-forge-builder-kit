#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import cloudpickle
import joblib
import numpy as np
import pandas as pd

from allora_forge_builder_kit import AlloraMLWorkflow

TICKERS = ["btcusd"]
INTERVAL = "1h"
NUMBER_OF_INPUT_BARS = 48
TARGET_BARS = 24

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
MODEL = joblib.load(ART / "topic69_model.joblib")
with open(ART / "topic69_columns.json", "r") as f:
    COLS = json.load(f)
BASE_FEATURE_COLS = COLS["base_feature_cols"]
FEATURE_COLS = COLS["feature_cols"]

api_key_path = Path(__file__).resolve().parent / ".allora_api_key"
API_KEY = api_key_path.read_text().strip()

WORKFLOW = AlloraMLWorkflow(
    tickers=TICKERS,
    number_of_input_bars=NUMBER_OF_INPUT_BARS,
    target_bars=TARGET_BARS,
    interval=INTERVAL,
    data_source="allora",
    api_key=API_KEY,
)


def engineer_returns(row: pd.Series) -> pd.Series:
    closes = np.array([row[f"feature_close_{i}"] for i in range(NUMBER_OF_INPUT_BARS)])
    eps = 1e-8
    return pd.Series(
        {
            "log_return_1h": float(np.log(closes[-1] + eps) - np.log(closes[-2] + eps)) if NUMBER_OF_INPUT_BARS >= 2 else 0.0,
            "log_return_6h": float(np.log(closes[-1] + eps) - np.log(closes[-7] + eps)) if NUMBER_OF_INPUT_BARS >= 7 else 0.0,
            "log_return_12h": float(np.log(closes[-1] + eps) - np.log(closes[-13] + eps)) if NUMBER_OF_INPUT_BARS >= 13 else 0.0,
            "log_return_24h": float(np.log(closes[-1] + eps) - np.log(closes[-25] + eps)) if NUMBER_OF_INPUT_BARS >= 25 else 0.0,
        }
    )


def predict(nonce: int | None = None) -> float:
    live_row = WORKFLOW.get_live_features(ticker=TICKERS[0])
    if live_row is None or len(live_row) == 0:
        raise ValueError("Could not get live features")

    live_returns = engineer_returns(live_row.iloc[0])
    live_features = pd.concat([live_row[BASE_FEATURE_COLS].iloc[0], live_returns])

    current_price = float(live_row.attrs.get("current_price", np.nan))
    if not np.isfinite(current_price) or current_price <= 0:
        snap = WORKFLOW._dm.get_live_snapshot(TICKERS)
        if snap is not None and len(snap) > 0 and "close" in snap.columns:
            current_price = float(snap["close"].iloc[-1])

    if not np.isfinite(current_price) or current_price <= 0:
        raise ValueError("Invalid current_price from live sources")

    predicted_log_return = float(MODEL.predict(live_features[FEATURE_COLS].values.reshape(1, -1))[0])
    predicted_price = float(current_price * np.exp(predicted_log_return))
    if not np.isfinite(predicted_price) or predicted_price <= 0:
        raise ValueError("Invalid predicted price")
    return predicted_price


if __name__ == "__main__":
    out = ROOT / "predict.pkl"
    with open(out, "wb") as f:
        cloudpickle.dump(predict, f)
    print(f"Wrote {out}")
