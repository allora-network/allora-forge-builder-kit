"""Tests for the hosting-package `export` (single-package format + genericity)."""

import json
import py_compile
import zipfile
from pathlib import Path

import pytest

from allora_forge_builder_kit.export import ModelSpec, export_package, main


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


def test_output_kind_driven_by_submit_returns(tmp_path):
    export_package(_spec(), tmp_path)
    src = (tmp_path / "forge_model" / "model.py").read_text()
    # Output kind is the SDK-derived per-topic signal, not a bespoke env: the
    # config exposes submit_returns (so the SDK's hasattr gate passes) and reads
    # SUBMIT_RETURNS for the explicit-override path.
    assert "submit_returns" in src
    assert 'os.environ.get("SUBMIT_RETURNS"' in src
    # the abandoned bespoke mechanism must be gone.
    assert "PREDICTION_KIND" not in src
    py_compile.compile(str(tmp_path / "forge_model" / "model.py"), doraise=True)


def _load_generated_model(tmp_path):
    """Import the generated forge_model/model.py as a standalone module."""
    import importlib.util

    export_package(_spec(), tmp_path)
    mod_path = tmp_path / "forge_model" / "model.py"
    spec = importlib.util.spec_from_file_location("generated_forge_model", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_inference(mod, submit_returns, current_price, monkeypatch):
    import asyncio

    import numpy as np
    import pandas as pd

    monkeypatch.setenv("PAIR", "BTCUSD")
    monkeypatch.setenv("TIMEFRAME", "5min")

    # Passthrough features + a fake workflow/model so get_inference exercises only
    # the output-kind branch, not data fetching or feature engineering.
    monkeypatch.setattr(mod, "apply_engineered_features", lambda df, specs, n: (df, []))

    live_row = pd.DataFrame({"feature_close_0": [1.0]})
    live_row.attrs["current_price"] = current_price

    class _WF:
        def get_live_features(self, ticker):
            return live_row

    monkeypatch.setattr(mod, "_workflow", lambda: _WF())

    class _M:
        def predict(self, x):
            return np.array([0.02])  # predicted log-return

    model = mod.ForgeModel(mod.ForgeModelConfig(submit_returns=submit_returns))
    model._bundle = {"model": _M(), "feature_cols": []}
    return asyncio.run(model.get_inference("BTCUSD"))


def test_submit_returns_true_publishes_raw_log_return(tmp_path, monkeypatch):
    mod = _load_generated_model(tmp_path)
    out = _run_inference(mod, submit_returns=True, current_price=100.0, monkeypatch=monkeypatch)
    assert out["prediction"] == pytest.approx(0.02)  # native output, no exp()


def test_submit_returns_false_publishes_absolute_price(tmp_path, monkeypatch):
    import math

    mod = _load_generated_model(tmp_path)
    out = _run_inference(mod, submit_returns=False, current_price=100.0, monkeypatch=monkeypatch)
    assert out["prediction"] == pytest.approx(100.0 * math.exp(0.02))


def test_submit_returns_false_requires_positive_current_price(tmp_path, monkeypatch):
    mod = _load_generated_model(tmp_path)
    with pytest.raises(ValueError, match="current_price"):
        _run_inference(mod, submit_returns=False, current_price=float("nan"), monkeypatch=monkeypatch)


def test_default_config_exposes_submit_returns_for_sdk_gate(tmp_path):
    # The SDK only auto-resolves SUBMIT_RETURNS when hasattr(config, "submit_returns").
    mod = _load_generated_model(tmp_path)
    cfg = mod.ForgeModel.default_config(timeframe="5min")
    assert hasattr(cfg, "submit_returns")


def test_submit_returns_defaults_to_log_return(tmp_path, monkeypatch):
    monkeypatch.delenv("SUBMIT_RETURNS", raising=False)
    mod = _load_generated_model(tmp_path)
    assert mod.ForgeModel.default_config().submit_returns is True


def test_explicit_submit_returns_false_env_is_honored(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBMIT_RETURNS", "false")
    mod = _load_generated_model(tmp_path)
    assert mod.ForgeModel.default_config().submit_returns is False


def test_api_key_not_passed_unconditionally(tmp_path):
    # Regression: the binance data-manager factory rejects unknown kwargs, so the
    # generated model must NOT pass api_key= unconditionally (it crashed binance).
    export_package(_spec(data_source="binance"), tmp_path)
    src = (tmp_path / "forge_model" / "model.py").read_text()
    assert "api_key=os.environ" not in src  # never unconditional
    assert 'if SOURCE == "allora"' in src  # gated by source


def test_hyperparameters_can_override_defaults_without_duplicate_kwargs(tmp_path):
    # Regression: the generated train_model builds a params dict so user-supplied
    # random_state/verbose win instead of raising "multiple values for keyword
    # argument". The template must not pass them positionally to LGBMRegressor.
    export_package(_spec(hyperparameters={"random_state": 7, "verbose": 1}), tmp_path)
    src = (tmp_path / "forge_model" / "model.py").read_text()
    assert 'params = {"random_state": 42, "verbose": -1, **HYPERPARAMETERS}' in src
    assert "LGBMRegressor(**params)" in src
    assert "LGBMRegressor(random_state=42, verbose=-1, **HYPERPARAMETERS)" not in src


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

    # Inference-only: manifest + config both reflect it. Must bundle weights, since
    # forge requires exactly one of supports_training/has_weights.
    export_package(_spec(supports_training=False), tmp_path / "b", weights_dir=_weights(tmp_path / "w"))
    assert json.loads((tmp_path / "b" / "manifest.json").read_text())["supports_training"] is False
    cfg = json.loads((tmp_path / "b" / "forge_model" / "config.json").read_text())
    assert cfg["supports_training"] is False


def _weights(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.joblib").write_bytes(b"x")
    return path


def test_bundled_weights(tmp_path):
    out = tmp_path / "pkg"
    # weights ⇒ inference-only (XOR with supports_training).
    export_package(_spec(supports_training=False), out, weights_dir=_weights(tmp_path / "w"))
    assert (out / "weights" / "model.joblib").exists()
    m = json.loads((out / "manifest.json").read_text())
    assert m["has_weights"] is True


@pytest.mark.parametrize("supports_training, with_weights", [(True, True), (False, False)])
def test_manifest_xor_enforced(tmp_path, supports_training, with_weights):
    # Forge (hosting/manifest.go) requires exactly one of the two true.
    weights = _weights(tmp_path / "w") if with_weights else None
    with pytest.raises(ValueError, match="exactly one"):
        export_package(_spec(supports_training=supports_training), tmp_path / "pkg", weights_dir=weights)


def test_reexport_without_weights_clears_stale_weights(tmp_path):
    # Regression: exporting WITH weights then re-exporting the SAME out_dir
    # WITHOUT weights must not leave a stale weights/ dir behind manifest's
    # has_weights=false (weights/ is excluded from code_hash, so it would drift
    # silently).
    out = tmp_path / "pkg"

    export_package(_spec(supports_training=False), out, weights_dir=_weights(tmp_path / "w"))
    assert (out / "weights" / "model.joblib").exists()

    export_package(_spec(), out)  # train-only, no weights this time
    assert not (out / "weights").exists()
    assert json.loads((out / "manifest.json").read_text())["has_weights"] is False


def test_empty_weights_dir_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="missing or empty"):
        export_package(_spec(supports_training=False), tmp_path / "pkg", weights_dir=empty)


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


def test_from_dict_missing_required_key_is_clean_value_error():
    # Missing model_type would surface as a TypeError from the dataclass ctor; it
    # must be re-raised as ValueError so main() prints the clean "export failed:".
    with pytest.raises(ValueError):
        ModelSpec.from_dict({"engineered_specs": [], "number_of_input_bars": 1, "target_bars": 1})


@pytest.mark.parametrize("bad_specs", ["notalist", [42], ["log_return"]])
def test_validate_rejects_malformed_engineered_specs(bad_specs):
    # Non-list / non-object specs used to raise AttributeError; now a clean ValueError.
    with pytest.raises(ValueError):
        _spec(engineered_specs=bad_specs).validate()


def _write_config(tmp_path, **over):
    base = dict(
        model_type="my_lgbm",
        engineered_specs=[{"kind": "log_return", "window_bars": 6}],
        number_of_input_bars=24,
        target_bars=24,
    )
    base.update(over)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(base))
    return cfg


def test_main_malformed_config_returns_clean_error(tmp_path, capsys):
    cfg = _write_config(tmp_path, engineered_specs="notalist")
    rc = main(["--config", str(cfg), "--out", str(tmp_path / "pkg")])
    assert rc == 1
    assert "export failed:" in capsys.readouterr().err


def test_main_zip_writes_flat_archive(tmp_path):
    cfg = _write_config(tmp_path)
    out = tmp_path / "pkg"
    rc = main(["--config", str(cfg), "--out", str(out), "--zip"])
    assert rc == 0
    archive = Path(str(out) + ".zip")
    assert archive.exists()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    # manifest.json must sit at the archive root (forge extracts flat), not nested.
    assert "manifest.json" in names
    assert "forge_model/model.py" in names
