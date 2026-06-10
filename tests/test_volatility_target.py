"""
Tests for the volatility target computation.

Verifies that target_type="volatility" correctly computes the standard
deviation of consecutive 1-minute log returns over the target horizon.
"""

import numpy as np
import polars as pl
import pytest
from datetime import datetime, timezone, timedelta

from allora_forge_builder_kit.workflow import AlloraMLWorkflow


@pytest.fixture
def synthetic_ohlcv():
    """Create synthetic 1-minute OHLCV data with known properties."""
    np.random.seed(42)
    n = 50
    times = [
        datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)
        for i in range(n)
    ]
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.001, n)))
    return pl.DataFrame(
        {
            "open_time": times,
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": np.ones(n) * 1000.0,
        }
    ), prices


def _make_workflow(target_bars=15):
    """Create a bare workflow instance for calling target methods."""
    wf = AlloraMLWorkflow.__new__(AlloraMLWorkflow)
    wf.target_bars = target_bars
    return wf


class TestVolatilityTargetComputation:
    """Tests for compute_volatility_target_polars."""

    def test_matches_manual_calculation(self, synthetic_ohlcv):
        """Verify each row matches manually computed std of forward log returns."""
        df, prices = synthetic_ohlcv
        target_bars = 15
        wf = _make_workflow(target_bars)

        result = wf.compute_volatility_target_polars(df, target_bars=target_bars)

        for row_idx in range(len(prices) - target_bars):
            window_prices = prices[row_idx : row_idx + target_bars + 1]
            log_rets = np.diff(np.log(window_prices))
            expected = np.std(log_rets, ddof=1)
            computed = result["target"][row_idx]
            assert np.isclose(expected, computed, rtol=1e-6), (
                f"Row {row_idx}: expected={expected:.10f}, got={computed:.10f}"
            )

    def test_trailing_rows_are_null(self, synthetic_ohlcv):
        """Rows without a full forward window should have null targets."""
        df, _ = synthetic_ohlcv
        target_bars = 15
        wf = _make_workflow(target_bars)

        result = wf.compute_volatility_target_polars(df, target_bars=target_bars)
        null_count = result["target"].null_count()
        assert null_count == target_bars

    def test_all_valid_targets_are_non_negative(self, synthetic_ohlcv):
        """Volatility (std) must be non-negative."""
        df, _ = synthetic_ohlcv
        wf = _make_workflow(15)

        result = wf.compute_volatility_target_polars(df, target_bars=15)
        valid = result["target"].drop_nulls()
        assert (valid >= 0).all()

    def test_different_horizon_sizes(self, synthetic_ohlcv):
        """Verify correctness with different target_bars values."""
        df, prices = synthetic_ohlcv

        for target_bars in [3, 5, 10, 20]:
            wf = _make_workflow(target_bars)
            result = wf.compute_volatility_target_polars(df, target_bars=target_bars)

            # Check first valid row
            if len(prices) > target_bars:
                window_prices = prices[0 : target_bars + 1]
                log_rets = np.diff(np.log(window_prices))
                expected = np.std(log_rets, ddof=1)
                computed = result["target"][0]
                assert np.isclose(expected, computed, rtol=1e-6), (
                    f"target_bars={target_bars}: expected={expected}, got={computed}"
                )


class TestTargetTypeParameter:
    """Tests for the target_type parameter on AlloraMLWorkflow."""

    def test_default_is_log_return(self):
        """Default target_type should be 'log_return'."""
        wf = AlloraMLWorkflow(
            tickers=["btcusd"],
            number_of_input_bars=15,
            target_bars=15,
            data_source="binance",
        )
        assert wf.target_type == "log_return"

    def test_volatility_accepted(self):
        """target_type='volatility' should be accepted."""
        wf = AlloraMLWorkflow(
            tickers=["btcusd"],
            number_of_input_bars=15,
            target_bars=15,
            target_type="volatility",
            data_source="binance",
        )
        assert wf.target_type == "volatility"

    def test_invalid_target_type_raises(self):
        """Invalid target_type should raise ValueError."""
        with pytest.raises(ValueError, match="target_type must be one of"):
            AlloraMLWorkflow(
                tickers=["btcusd"],
                number_of_input_bars=15,
                target_bars=15,
                target_type="invalid",
                data_source="binance",
            )

    def test_log_return_target_unchanged(self, synthetic_ohlcv):
        """Existing log-return target should still work identically."""
        df, prices = synthetic_ohlcv
        wf = _make_workflow(15)

        result = wf.compute_target_polars(df, target_bars=15)
        expected = np.log(prices[15] / prices[0])
        assert np.isclose(result["target"][0], expected, rtol=1e-6)
