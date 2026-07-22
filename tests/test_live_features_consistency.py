"""Golden parity test for engineered features (train path == serve path).

Base-feature correctness is already covered by ``test_feature_integrity.py``.
This test owns the part that was previously an untested hazard: the engineered
log-return features, which used to be copy-pasted inline in both the training
frame step and the serving ``predict()`` closure of each walkthrough notebook.

Now that both paths go through ``apply_engineered_features``, parity is
guaranteed by construction — this test proves it, needs no network / API key,
and would fail loudly if the two paths ever diverged again.
"""

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from allora_forge_builder_kit import (
    apply_engineered_features,
    engineered_feature_names,
)

N = 24  # number_of_input_bars
SPECS = [
    {"kind": "log_return", "window_bars": 1},
    {"kind": "log_return", "window_bars": 6},
    {"kind": "log_return", "window_bars": 12},
    {"kind": "log_return", "window_bars": 24},  # does not fit (needs 25 bars) -> 0.0
]
_EPS = 1e-8


def _base_frame(rows: int = 50, seed: int = 7) -> pd.DataFrame:
    """Synthetic base-feature frame with feature_close_{i} columns."""
    rng = np.random.default_rng(seed)
    data = {f"feature_close_{i}": rng.uniform(0.5, 1.5, size=rows) for i in range(N)}
    return pd.DataFrame(data)


def test_names_are_stable_and_ordered():
    assert engineered_feature_names(SPECS) == [
        "log_return_1",
        "log_return_6",
        "log_return_12",
        "log_return_24",
    ]


def test_train_and_serve_paths_are_identical():
    """A row computed inside a full frame (train) == the same row alone (serve)."""
    frame = _base_frame()

    trained, train_cols = apply_engineered_features(frame, SPECS, N)

    for idx in (0, 1, len(frame) - 1):
        single = frame.iloc[[idx]].reset_index(drop=True)
        served, serve_cols = apply_engineered_features(single, SPECS, N)
        assert serve_cols == train_cols
        for col in train_cols:
            np.testing.assert_allclose(
                served[col].to_numpy(),
                trained[col].to_numpy()[idx : idx + 1],
                rtol=1e-6,
                atol=1e-8,
            )


def test_matches_reference_computation():
    frame = _base_frame()
    out, cols = apply_engineered_features(frame, SPECS, N)

    last = N - 1
    recent = frame[f"feature_close_{last}"].to_numpy(dtype=float)
    for spec in SPECS:
        w = spec["window_bars"]
        col = f"log_return_{w}"
        if N < w + 1:
            expected = np.zeros(len(frame))
        else:
            past = frame[f"feature_close_{last - w}"].to_numpy(dtype=float)
            expected = np.log(recent + _EPS) - np.log(past + _EPS)
        np.testing.assert_allclose(out[col].to_numpy(), expected, rtol=1e-9, atol=1e-12)


def test_window_that_does_not_fit_is_zero():
    frame = _base_frame()
    out, _ = apply_engineered_features(frame, SPECS, N)
    # window_bars=24 needs 25 bars; N=24 -> filled with 0.0
    assert (out["log_return_24"] == 0.0).all()


def test_input_frame_is_not_mutated():
    frame = _base_frame()
    before = frame.copy()
    apply_engineered_features(frame, SPECS, N)
    pd.testing.assert_frame_equal(frame, before)


def test_unsupported_kind_raises():
    frame = _base_frame()
    with pytest.raises(ValueError, match="unsupported"):
        apply_engineered_features(frame, [{"kind": "ema", "window_bars": 5}], N)


def test_engineered_feature_names_rejects_unsupported():
    with pytest.raises(ValueError, match="unsupported"):
        engineered_feature_names([{"kind": "ema", "window_bars": 5}])


@pytest.mark.parametrize(
    "bad_window",
    [
        1.5,
        np.float32(1.5),
        np.float64(2.5),
        Decimal("1.5"),
        Fraction(3, 2),
        True,
        0,
        -1,
        complex(6, 0),
    ],
)
def test_window_bars_rejects_non_positive_integer(bad_window):
    with pytest.raises(ValueError, match="positive integer"):
        engineered_feature_names([{"kind": "log_return", "window_bars": bad_window}])


@pytest.mark.parametrize(
    "good_window",
    [6, 6.0, np.int64(6), np.float32(6.0), Decimal("6"), Fraction(6, 1)],
)
def test_window_bars_accepts_integral_numeric_types(good_window):
    assert engineered_feature_names(
        [{"kind": "log_return", "window_bars": good_window}]
    ) == ["log_return_6"]
