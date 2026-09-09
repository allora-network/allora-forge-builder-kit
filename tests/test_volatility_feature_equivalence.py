"""Check train-time and live volatility feature implementations for skew."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


N_INPUT_BARS = 60
SCRIPT_PATHS = [
    Path(f"notebooks/testnet/topic_{topic}/model_grid_retrain.py")
    for topic in (
        "79_btc_vol",
        "80_eth_vol",
        "81_xrp_vol",
        "82_sol_vol",
        "85_eth_4h_vol",
    )
]


def _load_feature_functions(path: Path):
    """Load only feature function definitions, without executing the script."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {"engineer_features_vectorized", "engineer_features"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    assert {node.name for node in functions} == names

    namespace = {"np": np, "pd": pd, "NUMBER_OF_INPUT_BARS": N_INPUT_BARS}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["engineer_features_vectorized"], namespace["engineer_features"]


def _synthetic_feature_rows() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(4):
        closes = np.exp(np.cumsum(rng.normal(0.0, 0.002, N_INPUT_BARS)))
        closes /= closes[-1]
        spreads = rng.uniform(0.0001, 0.003, N_INPUT_BARS)
        volumes = rng.lognormal(0.0, 0.4, N_INPUT_BARS)
        volumes /= volumes[-1]
        row = {}
        for i in range(N_INPUT_BARS):
            row[f"feature_close_{i}"] = closes[i]
            row[f"feature_high_{i}"] = closes[i] * (1.0 + spreads[i])
            row[f"feature_low_{i}"] = closes[i] * (1.0 - spreads[i])
            row[f"feature_volume_{i}"] = volumes[i]
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("script_path", SCRIPT_PATHS, ids=lambda path: path.parent.name)
def test_vectorized_features_match_live_row_features(script_path: Path):
    vectorized, row_wise = _load_feature_functions(script_path)
    source = _synthetic_feature_rows()

    train_features = vectorized(source, N_INPUT_BARS)
    live_features = pd.DataFrame(
        [row_wise(source.iloc[i]) for i in range(len(source))],
        index=source.index,
    )

    assert list(train_features.columns) == list(live_features.columns)
    np.testing.assert_allclose(
        train_features.to_numpy(),
        live_features.to_numpy(),
        rtol=1e-12,
        atol=1e-15,
    )
