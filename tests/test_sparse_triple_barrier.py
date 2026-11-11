"""Sparse no-trade minutes must not invalidate nonempty native bars."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from allora_forge_builder_kit import AlloraMLWorkflow

TARGETS = ['target_down', 'target_neutral', 'target_up']


@pytest.fixture(scope='module')
def candles():
    times = pd.date_range('2026-01-01', periods=54 * 1440, freq='min', tz='UTC')
    return pl.from_pandas(pd.DataFrame(dict(open_time=times, open=100., high=101.,
                                           low=99., close=100., volume=1.)))


def labelled(raw, cls=AlloraMLWorkflow):
    workflow = object.__new__(cls)
    workflow.interval = '3h'
    native = workflow.resample_ohlcv_polars(raw, freq='3h')
    return workflow.compute_triple_barrier_target_polars(native, 4, raw).to_pandas()


def test_complete_data_targets(candles):
    result = labelled(candles).dropna(subset=TARGETS)
    assert len(result) > 0
    np.testing.assert_allclose(result.tb_atr, np.log(101 / 99))
    np.testing.assert_allclose(result.tb_upper, 100 * np.exp(.25 * np.log(101 / 99)))
    np.testing.assert_allclose(result.tb_lower, 100 * np.exp(-.25 * np.log(101 / 99)))
    assert (result.target_down == 1).all()
    assert (result.tb_exit_price == result.tb_lower).all()
    assert (result.tb_exit_time == result.tb_prediction_time).all()
    assert ((result.tb_resolution_time - result.tb_prediction_time)
            == pd.Timedelta(hours=12) - pd.Timedelta(minutes=1)).all()


def test_sparse_history_future_and_base_minute(candles):
    original = labelled(candles)
    opening = pd.Timestamp('2026-02-21T00:00Z')
    # Sparse history, absent final input minute and sparse future remain usable.
    missing = [pd.Timestamp('2026-01-20T03:11Z'), opening + pd.Timedelta(hours=3) - pd.Timedelta(minutes=1),
               opening + pd.Timedelta(hours=4)]
    sparse = candles.filter(~pl.col('open_time').is_in(missing))
    actual = labelled(sparse)
    row = actual.loc[actual.open_time == opening].iloc[0]
    assert row[TARGETS].notna().all()
    assert row.tb_base == 100.
    assert row.target_down == 1  # Observed simultaneous touches still choose DOWN.
    assert actual[TARGETS].notna().all(axis=1).sum() == original[TARGETS].notna().all(axis=1).sum()


def test_whole_missing_native_bar_is_unresolved(candles):
    start = pd.Timestamp('2026-02-21T06:00Z')
    missing = candles.filter(~((pl.col('open_time') >= start) &
                              (pl.col('open_time') < start + pd.Timedelta(hours=3))))
    result = labelled(missing)
    row = result.loc[result.open_time == start - pd.Timedelta(hours=6)].iloc[0]
    assert row[TARGETS].isna().all()


def test_corrupt_observed_prices_are_not_sparse_minutes(candles):
    broken = candles.with_columns(pl.when(pl.col('open_time') == pd.Timestamp('2026-01-20T03:11Z'))
                                 .then(-1.).otherwise(pl.col('low')).alias('low'))
    result = labelled(broken)
    row = result.loc[result.open_time == pd.Timestamp('2026-02-21T00:00Z')].iloc[0]
    assert row[TARGETS].isna().all()
