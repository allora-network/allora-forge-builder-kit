"""Export a builder-kit model into a hosting-deployable package.

The Allora hosting platform deploys a *package*: worker code + ``pyproject.toml``
+ ``manifest.json`` + an optional ``weights/`` dir. This module turns a model the
user built with the builder kit into that package, ready to upload
(forge-v2 ``POST /api/v1/models``).

Two modes, both supported:
  * **code-only** (train-on-platform): no ``weights/``; the platform trains on
    first run and on schedule. The package is GENERIC — ``PAIR``/``TIMEFRAME`` are
    read from the environment at runtime (set per deployment), so one package can
    be deployed against many pairs/timeframes.
  * **with weights** (train-locally): pass ``weights_dir`` to bundle a pre-trained
    artifact. Bundled weights are naturally specific to the pair/timeframe they
    were trained on; deploy with matching params.

The generated model computes engineered features via the shared
``allora_forge_builder_kit.apply_engineered_features`` module — the same code the
user trains with — so train and serve are identical by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The python package dir inside every generated package.
_PKG_NAME = "forge_model"
_MANIFEST_SCHEMA_VERSION = 1
# Model-type names become an entry-point name and an OCI path segment; keep them
# to a safe, filesystem/identifier-friendly charset.
_MODEL_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass
class ModelSpec:
    """The model-intrinsic description baked into the package (NOT pair/timeframe/
    topic — those are chosen per deployment)."""

    model_type: str
    engineered_specs: list[dict[str, Any]]
    number_of_input_bars: int
    target_bars: int
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    model_family: str = "lightgbm"
    data_source: str = "binance"
    days_of_history: int = 180
    # Whether the model retrains on-platform. False = train-locally / inference-only:
    # the platform never retrains; it just serves weights (bundled via weights_dir or
    # imported separately). See the SDK's TrainingRunner (gates on supports_training).
    supports_training: bool = True

    def validate(self) -> None:
        if not _MODEL_TYPE_RE.match(self.model_type):
            raise ValueError(
                f"model_type {self.model_type!r} must match [a-z0-9][a-z0-9_-]*"
            )
        if self.model_family != "lightgbm":
            raise ValueError(f"model_family {self.model_family!r} unsupported (v1: lightgbm)")
        if self.data_source not in ("binance", "allora"):
            raise ValueError(f"data_source {self.data_source!r} must be binance|allora")
        if self.number_of_input_bars < 1:
            raise ValueError("number_of_input_bars must be >= 1")
        if self.target_bars < 1:
            raise ValueError("target_bars must be >= 1")
        if self.days_of_history < 1:
            raise ValueError("days_of_history must be >= 1")
        for i, spec in enumerate(self.engineered_specs):
            if spec.get("kind") != "log_return":
                raise ValueError(f"engineered_specs[{i}].kind must be 'log_return' (v1)")
            if int(spec.get("window_bars", 0)) < 1:
                raise ValueError(f"engineered_specs[{i}].window_bars must be >= 1")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def config_json(self) -> dict[str, Any]:
        """The subset baked into forge_model/config.json (read by the model at runtime)."""
        return {
            "model_type": self.model_type,
            "model_family": self.model_family,
            "data_source": self.data_source,
            "days_of_history": self.days_of_history,
            "number_of_input_bars": self.number_of_input_bars,
            "target_bars": self.target_bars,
            "engineered_specs": self.engineered_specs,
            "hyperparameters": self.hyperparameters,
            "supports_training": self.supports_training,
        }


def export_package(
    spec: ModelSpec,
    out_dir: str | os.PathLike,
    *,
    weights_dir: str | os.PathLike | None = None,
    builder_kit_ref: str = "main",
) -> Path:
    """Write a deployable package for ``spec`` into ``out_dir`` and return its path.

    ``weights_dir`` (optional): a directory whose contents are copied into the
    package's ``weights/`` (train-locally mode). ``builder_kit_ref``: the git ref
    the generated package installs allora-forge-builder-kit from.
    """
    spec.validate()
    out = Path(out_dir)
    pkg_dir = out / _PKG_NAME
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Package code (static model.py + config.json + __init__).
    (pkg_dir / "model.py").write_text(_MODEL_PY)
    (pkg_dir / "config.json").write_text(json.dumps(spec.config_json(), indent=2, sort_keys=True) + "\n")
    (pkg_dir / "__init__.py").write_text(_INIT_PY)

    # 2. pyproject + dockerignore.
    (out / "pyproject.toml").write_text(
        _PYPROJECT_TMPL.format(
            model_type=spec.model_type,
            pkg=_PKG_NAME,
            builder_kit_ref=builder_kit_ref,
        )
    )
    (out / ".dockerignore").write_text(_DOCKERIGNORE)

    # 3. Optional bundled weights.
    has_weights = weights_dir is not None
    if has_weights:
        src = Path(weights_dir)
        if not src.is_dir() or not any(src.iterdir()):
            raise ValueError(f"weights_dir {src} is missing or empty")
        dst = out / "weights"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # 4. Manifest (model-only; code_hash excludes weights/, matching .dockerignore).
    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "model_type": spec.model_type,
        "supports_training": spec.supports_training,
        "has_weights": has_weights,
        "code_hash": _code_hash(out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def _code_hash(out: Path) -> str:
    """sha256 over the package's code files (deterministic; excludes weights/)."""
    h = hashlib.sha256()
    files = sorted(
        p for p in out.rglob("*")
        if p.is_file() and "weights" not in p.relative_to(out).parts and p.name != "manifest.json"
    )
    for p in files:
        h.update(str(p.relative_to(out)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return "sha256:" + h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="allora-forge-export",
        description="Export a builder-kit model into a hosting-deployable package (.zip contents).",
    )
    ap.add_argument("--config", required=True, help="Path to a model config JSON (see ModelSpec fields).")
    ap.add_argument("--out", required=True, help="Output directory for the package.")
    ap.add_argument("--weights", default=None, help="Optional directory of pre-trained weights to bundle.")
    ap.add_argument("--builder-kit-ref", default="main", help="git ref the package installs builder-kit from.")
    ap.add_argument(
        "--no-training",
        action="store_true",
        help="Inference-only: the platform never retrains; it serves bundled/imported weights.",
    )
    args = ap.parse_args(argv)

    try:
        spec = ModelSpec.from_dict(json.loads(Path(args.config).read_text()))
        if args.no_training:
            spec.supports_training = False
        if not spec.supports_training and args.weights is None:
            print(
                "warning: inference-only model (supports_training=false) has no bundled "
                "--weights; it will serve nothing until weights are imported into storage.",
                file=sys.stderr,
            )
        out = export_package(spec, args.out, weights_dir=args.weights, builder_kit_ref=args.builder_kit_ref)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"export failed: {e}", file=sys.stderr)
        return 1
    print(f"exported package to {out} (zip its contents to upload)")
    return 0


# The generated model. Static (no per-model templating) — model-intrinsic values
# live in config.json next to it; pair/timeframe come from the environment. Its
# docstrings use \"\"\" so this module embeds it with ''' safely.
_MODEL_PY = '''"""Generated by allora-forge-builder-kit `export`. Do not edit by hand.

Generic across pair/timeframe: PAIR and TIMEFRAME are read from the environment
at runtime (the hosting platform sets them per deployment), so one package can be
deployed against many pairs/timeframes. Model-intrinsic config (features, target,
hyperparameters) is baked into config.json next to this file. Training and
inference both compute features via the shared
allora_forge_builder_kit.apply_engineered_features module, so the two paths are
identical by construction (no train/serve skew).
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import joblib
from lightgbm import LGBMRegressor

from allora_worker import BaseModel, ModelConfig
from allora_forge_builder_kit import AlloraMLWorkflow, apply_engineered_features

logger = logging.getLogger(__name__)

_CONFIG = json.loads((Path(__file__).parent / "config.json").read_text())
ENGINEERED_SPECS = _CONFIG["engineered_specs"]
HYPERPARAMETERS = _CONFIG["hyperparameters"]
SOURCE = _CONFIG["data_source"]
NUMBER_OF_INPUT_BARS = _CONFIG["number_of_input_bars"]
TARGET_BARS = _CONFIG["target_bars"]
DAYS_OF_HISTORY = _CONFIG["days_of_history"]
SUPPORTS_TRAINING = _CONFIG.get("supports_training", True)
MODEL_FILENAME = "model.joblib"


def _pair() -> str:
    pair = os.environ.get("PAIR")
    if not pair:
        raise RuntimeError("PAIR env var is required")
    return pair.upper()


def _timeframe() -> str:
    tf = os.environ.get("TIMEFRAME")
    if not tf:
        raise RuntimeError("TIMEFRAME env var is required")
    return tf


def _workflow() -> AlloraMLWorkflow:
    # api_key is only valid for the allora/atlas data source; the binance data
    # manager rejects unknown kwargs, so pass it only when it applies.
    kwargs = {}
    if SOURCE in ("allora", "atlas"):
        kwargs["api_key"] = os.environ.get("ALLORA_API_KEY")
    return AlloraMLWorkflow(
        tickers=[_pair()],
        number_of_input_bars=NUMBER_OF_INPUT_BARS,
        target_bars=TARGET_BARS,
        interval=_timeframe(),
        data_source=SOURCE,
        **kwargs,
    )


def _model_dir() -> Path:
    # Matches the SDK on-disk layout {model_type}-{pair}_{timeframe} under
    # DATA_BASE_PATH, where the serving pod loads artifacts at startup.
    base = os.environ.get("DATA_BASE_PATH", "./data")
    model_type = os.environ.get("MODEL_TYPE", _CONFIG["model_type"])
    return Path(base) / f"{model_type}-{_pair()}_{_timeframe()}"


def _feature_columns(df):
    base_cols = [c for c in df.columns if c.startswith("feature_")]
    df, eng_cols = apply_engineered_features(df, ENGINEERED_SPECS, NUMBER_OF_INPUT_BARS)
    return df, base_cols + eng_cols


class ForgeModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._bundle = None

    def supports_training(self) -> bool:
        return SUPPORTS_TRAINING

    def train_model(self, pair: str = None, **kwargs):
        wf = _workflow()
        start = datetime.now(timezone.utc) - timedelta(days=DAYS_OF_HISTORY)
        wf.backfill(start=start)
        df = wf.get_full_feature_target_dataframe(start_date=start).reset_index()
        df, feature_cols = _feature_columns(df)
        df = df.dropna(subset=feature_cols + ["target"])

        model = LGBMRegressor(random_state=42, verbose=-1, **HYPERPARAMETERS)
        model.fit(df[feature_cols], df["target"])
        logger.info("trained on %d samples, %d features", len(df), len(feature_cols))

        out_dir = _model_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_cols": feature_cols}, out_dir / MODEL_FILENAME)
        self._bundle = None  # force reload from disk on next inference
        return {
            "newly_trained": True,
            "training_stats": {"n_samples": int(len(df)), "n_features": len(feature_cols)},
            "model_time": time.time(),
            "model_path": str(out_dir),
        }

    def _load(self):
        if self._bundle is None:
            path = _model_dir() / MODEL_FILENAME
            if not path.exists():
                raise FileNotFoundError(f"model artifact not found at {path}")
            self._bundle = joblib.load(path)
        return self._bundle

    async def get_inference(self, pair: str, **kwargs):
        bundle = self._load()
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]

        wf = _workflow()
        live_row = wf.get_live_features(ticker=_pair())
        if live_row is None or len(live_row) == 0:
            raise ValueError("could not get live features")

        live_row, _ = apply_engineered_features(live_row, ENGINEERED_SPECS, NUMBER_OF_INPUT_BARS)
        x = live_row[feature_cols].iloc[0].values.reshape(1, -1)
        predicted_log_return = float(model.predict(x)[0])

        current_price = float(live_row.attrs.get("current_price", np.nan))
        if np.isfinite(current_price) and current_price > 0:
            prediction = current_price * float(np.exp(predicted_log_return))
        else:
            prediction = predicted_log_return

        return {
            "prediction": float(prediction),
            "input_data": {
                "log_return": predicted_log_return,
                "current_price": current_price if np.isfinite(current_price) else None,
            },
            "timestamp": time.time(),
        }
'''

_INIT_PY = '''from .model import ForgeModel

__all__ = ["ForgeModel"]
'''

_DOCKERIGNORE = "weights/\n__pycache__/\n*.pyc\n*.pyo\n.git/\n"

# allora-worker-sdk is declared UNPINNED on purpose: the base image already has
# it installed, so an unpinned requirement no-ops on-image and never shadows the
# image's SDK (it is not on PyPI, so pinning would break the build).
_PYPROJECT_TMPL = '''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "forge-model-{model_type}"
version = "0.1.0"
description = "Forge model package: {model_type}"
requires-python = ">=3.10"
dependencies = [
    "allora-worker-sdk",
    "allora-forge-builder-kit @ git+https://github.com/allora-network/allora-forge-builder-kit.git@{builder_kit_ref}",
    "lightgbm",
    "joblib",
]

[project.entry-points."allora_worker.models"]
{model_type} = "{pkg}.model:ForgeModel"

[tool.hatch.build.targets.wheel]
packages = ["{pkg}"]
'''


if __name__ == "__main__":
    raise SystemExit(main())
