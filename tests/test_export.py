"""Tests for the hosting-package `export` (single-package format + genericity)."""

import json
import py_compile
import zipfile
from pathlib import Path

import pytest

from allora_forge_builder_kit import WorkerManager
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


def _model_src(tmp_path) -> str:
    export_package(_spec(), tmp_path)
    return (tmp_path / "forge_model" / "model.py").read_text()


def test_submit_returns_true_publishes_raw_log_return(tmp_path):
    # Prove branch wiring: raw log-return is in the else clause, not the is-False
    # clause. An inverted condition or swapped branch bodies changes relative order.
    src = _model_src(tmp_path)
    price_pos = src.index("prediction = current_price * float(np.exp(predicted_log_return))")
    else_raw = src.index("else:\n            prediction = predicted_log_return")
    assert price_pos < else_raw


def test_submit_returns_false_publishes_absolute_price(tmp_path):
    # Prove branch wiring: exp() price conversion is inside the is-False clause
    # (between the condition and the else).
    src = _model_src(tmp_path)
    cond_pos = src.index('getattr(self.config, "submit_returns", True) is False:')
    price_pos = src.index("prediction = current_price * float(np.exp(predicted_log_return))")
    else_pos = src.index("else:\n            prediction = predicted_log_return")
    assert cond_pos < price_pos < else_pos


def test_submit_returns_false_requires_positive_current_price(tmp_path):
    # Prove guard wiring: inside the is-False clause, the price guard + ValueError
    # appear before the exp() conversion — a bad current_price must be rejected
    # before any price is computed.
    src = _model_src(tmp_path)
    cond_pos = src.index('getattr(self.config, "submit_returns", True) is False:')
    guard_pos = src.index("if not (np.isfinite(current_price) and current_price > 0):")
    raise_pos = src.index("raise ValueError", guard_pos)
    price_pos = src.index("prediction = current_price * float(np.exp(predicted_log_return))")
    assert cond_pos < guard_pos < raise_pos < price_pos


def test_default_config_exposes_submit_returns_for_sdk_gate(tmp_path):
    # The SDK gates SUBMIT_RETURNS auto-resolution on hasattr(config, "submit_returns").
    # ForgeModelConfig must declare the field so the attribute always exists.
    src = _model_src(tmp_path)
    assert "submit_returns: bool" in src
    assert "dataclasses.field(default_factory=_env_submit_returns)" in src


def test_submit_returns_defaults_to_log_return(tmp_path):
    # When SUBMIT_RETURNS is absent, the default must be True (publish log-return).
    # The sentinel is != "false": anything not explicitly "false" is True.
    src = _model_src(tmp_path)
    assert '!= "false"' in src


def test_explicit_submit_returns_false_env_is_honored(tmp_path):
    # SUBMIT_RETURNS=false must be parsed case-insensitively from the environment.
    src = _model_src(tmp_path)
    assert 'os.environ.get("SUBMIT_RETURNS", "")' in src
    assert '.strip().lower() != "false"' in src


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
    dict(model_type="ok", supports_training="true"),  # non-bool would bypass the XOR check
    dict(model_type="ok", supports_training=1),
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


def test_export_payload_malformed_spec_raises(tmp_path):
    # Malformed engineered_specs must raise a clean ValueError before writing anything.
    bad = _spec(engineered_specs="notalist")
    wm = WorkerManager(db_path=tmp_path / "state.db", reconcile_on_start=False)
    with pytest.raises(ValueError):
        wm.export_payload_for_hosting(bad, out_dir=tmp_path / "pkg")


def test_export_payload_zip_writes_flat_archive(tmp_path):
    wm = WorkerManager(db_path=tmp_path / "state.db", reconcile_on_start=False)
    archive = wm.export_payload_for_hosting(_spec(), out_dir=tmp_path / "pkg", zip_output=True)
    assert archive.suffix == ".zip"
    assert archive.exists()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    # manifest.json must sit at the archive root (forge extracts flat), not nested.
    assert "manifest.json" in names
    assert "forge_model/model.py" in names
