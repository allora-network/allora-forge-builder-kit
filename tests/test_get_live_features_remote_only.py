from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from allora_forge_builder_kit.base_data_manager import BaseDataManager
from allora_forge_builder_kit.workflow import AlloraMLWorkflow


class DummyRemoteOnlyDM(BaseDataManager):
    def __init__(self, interval: str = "1h"):
        super().__init__(base_dir="unused", interval=interval, symbols=["BTCUSDT"])
        self.requested_hours_back = []

    # ---- not used in these tests ----
    def backfill_symbol(self, symbol, start=None, end=None):
        return None

    def backfill_realtime(self, symbols, start=None, end=None):
        return None

    def backfill_missing(self, symbols, start=None):
        return None

    def get_live_snapshot(self, symbols):
        raise NotImplementedError

    # ---- used by workflow.get_live_features ----
    def get_live_1min_data(self, symbol: str, hours_back: int = 2) -> pd.DataFrame:
        self.requested_hours_back.append(hours_back)

        n = hours_back * 60
        end = pd.Timestamp.now(tz="UTC").floor("min")
        idx = pd.date_range(end=end, periods=n, freq="1min", tz="UTC")

        base = 50000.0
        close = pd.Series([base + i * 0.1 for i in range(n)], index=idx)
        out = pd.DataFrame(
            {
                "open_time": idx,
                "open": close.values,
                "high": (close + 5).values,
                "low": (close - 5).values,
                "close": close.values,
                "volume": [100.0] * n,
                "symbol": [symbol] * n,
            }
        ).set_index("open_time")
        return out

    def load_pandas(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("get_live_features must not call load_pandas/local parquet")


def test_get_live_features_uses_remote_data_only_and_returns_price_context():
    dm = DummyRemoteOnlyDM()
    workflow = AlloraMLWorkflow(
        tickers=["BTCUSDT"],
        number_of_input_bars=48,
        target_bars=24,
        interval="1h",
        data_manager=dm,
    )

    features = workflow.get_live_features("BTCUSDT")

    assert len(dm.requested_hours_back) == 1
    assert features.shape[0] == 1
    assert features.shape[1] == 48 * 5
    assert "current_price" in features.attrs
    assert features.attrs["current_price"] > 0


def test_get_live_features_requests_enough_history_for_interval_and_lookback():
    dm = DummyRemoteOnlyDM()
    number_of_input_bars = 48
    interval = "1h"

    workflow = AlloraMLWorkflow(
        tickers=["BTCUSDT"],
        number_of_input_bars=number_of_input_bars,
        target_bars=24,
        interval=interval,
        data_manager=dm,
    )

    workflow.get_live_features("BTCUSDT")
    requested_hours = dm.requested_hours_back[-1]

    # Must request at least enough wall-clock history to cover feature bars.
    # (48 bars * 1h each = 48h minimum, plus buffer for live alignment)
    assert requested_hours >= 48
