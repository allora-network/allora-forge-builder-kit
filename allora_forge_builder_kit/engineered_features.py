"""Shared engineered-feature computation.

Single source of truth for derived features built on top of the base
normalized-OHLCV lookback window (``feature_close_{i}`` columns produced by
``AlloraMLWorkflow``). Both the training path and the serving/inference path MUST
call this module so the two produce byte-identical feature vectors — this is the
guard against train/serve skew.

The recipe contract (see forge-v2 ``docs/hosting/01-recipe-schema.md``) expresses
each derived feature as a spec dict, e.g.::

    {"kind": "log_return", "window_bars": 6}

Only ``log_return`` is supported at v1, matching the ``engineer_returns`` logic
previously duplicated inline in the walkthrough notebooks.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd

_EPS = 1e-8


def _log_return_column_name(window_bars: int) -> str:
    return f"log_return_{window_bars}"


def _window_bars(spec: dict) -> int:
    """Validate and return ``spec['window_bars']`` as a positive integer.

    Rejects non-integral values (e.g. ``1.5``) rather than silently truncating,
    which would train/serve a different feature than the recipe requested.
    """
    raw = spec["window_bars"]
    if isinstance(raw, bool) or not isinstance(raw, numbers.Number):
        raise ValueError(f"window_bars must be a positive integer, got {raw!r}")
    try:
        window = int(raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"window_bars must be a positive integer, got {raw!r}")
    if window != raw or window < 1:
        raise ValueError(f"window_bars must be a positive integer, got {raw!r}")
    return window


def engineered_feature_names(specs: list[dict]) -> list[str]:
    """Return the column names ``apply_engineered_features`` will add, in order."""
    names: list[str] = []
    for spec in specs:
        kind = spec["kind"]
        if kind != "log_return":
            raise ValueError(f"unsupported engineered feature kind: {kind!r}")
        name = _log_return_column_name(_window_bars(spec))
        if name in names:
            raise ValueError(f"duplicate engineered feature: {name!r}")
        names.append(name)
    return names


def apply_engineered_features(
    df: pd.DataFrame,
    specs: list[dict],
    number_of_input_bars: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Add engineered feature columns to ``df`` and return ``(df, added_names)``.

    ``df`` must contain the base ``feature_close_{i}`` columns for
    ``i in [0, number_of_input_bars)``. A copy is returned; the input is not
    mutated. The computation is vectorized and identical whether ``df`` has one
    row (serving) or many (training), which is what guarantees parity.

    A ``log_return`` with ``window_bars = w`` compares the most recent bar's
    normalized close against the bar ``w`` steps earlier::

        log(feature_close_{N-1} + eps) - log(feature_close_{N-1-w} + eps)

    If the window does not fit the lookback (``number_of_input_bars < w + 1``)
    the column is filled with ``0.0``, matching the notebooks' guard.
    """
    out = df.copy()
    added: list[str] = []
    last = number_of_input_bars - 1

    for spec in specs:
        kind = spec["kind"]
        if kind != "log_return":
            raise ValueError(f"unsupported engineered feature kind: {kind!r}")
        window = _window_bars(spec)
        name = _log_return_column_name(window)
        if name in added:
            raise ValueError(f"duplicate engineered feature: {name!r}")

        if number_of_input_bars < window + 1:
            # Window doesn't fit the lookback → 0.0 (matches the notebooks' guard).
            # pandas broadcasts the scalar to every row; on a 0-row frame this
            # yields an empty float64 column, which is the intended dtype here.
            out[name] = 0.0
        else:
            recent = out[f"feature_close_{last}"].to_numpy(dtype=float)
            past = out[f"feature_close_{last - window}"].to_numpy(dtype=float)
            out[name] = np.log(recent + _EPS) - np.log(past + _EPS)
        added.append(name)

    return out, added
