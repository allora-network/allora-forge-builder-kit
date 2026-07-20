"""Tests for the hosting-package `export` (single-package format + genericity)."""

import json
import py_compile
from pathlib import Path

import pytest

from allora_forge_builder_kit.export import ModelSpec, export_package


def _spec(**over):
    base = dict(
        model_type="my_lgbm",
        engineered_specs=[{"kind": "log_return", "window_bars": 6}],
        number_of_input_bars=24,
        target_bars=24,
        hyperparameters={"n_estimators": 500},
    )
    base.update(over)
    return ModelSpec(**base)


def test_export_file_presence(tmp_path):
    export_package(_spec(), tmp_path)
    for f in ("pyproject.toml", ".dockerignore", "manifest.json",
              "forge_model/__init__.py", "forge_model/model.py", "forge_model/config.json"):
        assert (tmp_path / f).exists(), f"missing {f}"


def test_manifest_is_model_only(tmp_path):
    export_package(_spec(), tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["schema_version"] == 1
    assert m["model_type"] == "my_lgbm"
    assert m["supports_training"] is True
    assert m["has_weights"] is False
    assert m["code_hash"].startswith("sha256:")
    # pair/timeframe/topic are deploy-time — never in the manifest.
    assert "pair" not in m and "timeframe" not in m and "topic_id" not in m


def test_model_is_generic_over_pair_and_timeframe(tmp_path):
    export_package(_spec(), tmp_path)
    src = (tmp_path / "forge_model" / "model.py").read_text()
    # pair/timeframe come from the environment, not baked in.
    assert 'os.environ.get("PAIR")' in src
    assert 'os.environ.get("TIMEFRAME")' in src
    assert "BTCUSD" not in src  # no baked ticker
    # the generated model must be importable Python.
    py_compile.compile(str(tmp_path / "forge_model" / "model.py"), doraise=True)


def test_api_key_not_passed_unconditionally(tmp_path):
    # Regression: the binance data-manager factory rejects unknown kwargs, so the
    # generated model must NOT pass api_key= unconditionally (it crashed binance).
    export_package(_spec(data_source="binance"), tmp_path)
    src = (tmp_path / "forge_model" / "model.py").read_text()
    assert "api_key=os.environ" not in src  # never unconditional
    assert 'if SOURCE in ("allora", "atlas")' in src  # gated by source


def test_pyproject_contract(tmp_path):
    export_package(_spec(), tmp_path, builder_kit_ref="some-branch")
    pp = (tmp_path / "pyproject.toml").read_text()
    assert '"allora-worker-sdk"' in pp  # unpinned
    assert "allora-worker-sdk==" not in pp
    assert "@some-branch" in pp  # builder-kit ref threaded through
    assert 'my_lgbm = "forge_model.model:ForgeModel"' in pp  # entry-point keyed on model_type


def test_config_baked_but_not_pair_timeframe(tmp_path):
    export_package(_spec(), tmp_path)
    cfg = json.loads((tmp_path / "forge_model" / "config.json").read_text())
    assert cfg["engineered_specs"] == [{"kind": "log_return", "window_bars": 6}]
    assert cfg["hyperparameters"] == {"n_estimators": 500}
    assert "pair" not in cfg and "timeframe" not in cfg


def test_supports_training_flag(tmp_path):
    # Default: trains on-platform.
    export_package(_spec(), tmp_path / "a")
    assert json.loads((tmp_path / "a" / "manifest.json").read_text())["supports_training"] is True
    a_model = (tmp_path / "a" / "forge_model" / "model.py").read_text()
    assert "SUPPORTS_TRAINING = _CONFIG.get(" in a_model  # method reads config

    # Inference-only: manifest + config both reflect it.
    export_package(_spec(supports_training=False), tmp_path / "b")
    assert json.loads((tmp_path / "b" / "manifest.json").read_text())["supports_training"] is False
    cfg = json.loads((tmp_path / "b" / "forge_model" / "config.json").read_text())
    assert cfg["supports_training"] is False


def test_bundled_weights(tmp_path):
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "model.joblib").write_bytes(b"x")
    out = tmp_path / "pkg"
    export_package(_spec(), out, weights_dir=weights)
    assert (out / "weights" / "model.joblib").exists()
    m = json.loads((out / "manifest.json").read_text())
    assert m["has_weights"] is True


def test_reexport_without_weights_clears_stale_weights(tmp_path):
    # Regression: exporting WITH weights then re-exporting the SAME out_dir
    # WITHOUT weights must not leave a stale weights/ dir behind manifest's
    # has_weights=false (weights/ is excluded from code_hash, so it would drift
    # silently).
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "model.joblib").write_bytes(b"x")
    out = tmp_path / "pkg"

    export_package(_spec(), out, weights_dir=weights)
    assert (out / "weights" / "model.joblib").exists()

    export_package(_spec(), out)  # no weights this time
    assert not (out / "weights").exists()
    assert json.loads((out / "manifest.json").read_text())["has_weights"] is False


def test_empty_weights_dir_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        export_package(_spec(), tmp_path / "pkg", weights_dir=empty)


@pytest.mark.parametrize("over", [
    dict(model_type="Bad Name"),
    dict(model_type="ok", model_family="xgboost"),
    dict(model_type="ok", data_source="coinbase"),
    dict(model_type="ok", number_of_input_bars=0),
])
def test_validation_rejects(over, tmp_path):
    with pytest.raises(ValueError):
        _spec(**over).validate()


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        ModelSpec.from_dict({
            "model_type": "ok", "engineered_specs": [], "number_of_input_bars": 1,
            "target_bars": 1, "surprise": True,
        })
