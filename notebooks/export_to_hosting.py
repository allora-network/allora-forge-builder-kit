#!/usr/bin/env python3
"""
Export a model to the Allora hosting platform.

The walkthroughs (`example_topic_69_*`, `example_topic_77_*`) train and deploy a
worker *locally* via `WorkerManager`. This example shows the other deployment
path: packaging a model so the Allora **hosting platform** runs the worker for
you in a container.

The unit the platform deploys is a *package*: worker code + `pyproject.toml` +
`manifest.json` (+ an optional `weights/` dir). `WorkerManager.export_payload_for_hosting`
builds that package from a `ModelSpec`. The generated worker is generic over pair/timeframe —
`PAIR`/`TIMEFRAME` are supplied per deployment as env vars, so one package can be
deployed against many pairs/timeframes/topics.

Two modes:
  * train-on-platform (default): no bundled weights; the platform runs training
    (an `allora-worker train` job the operator schedules) and the worker serves the
    trained artifact. Set on the spec with `supports_training=True`.
  * train-locally: bundle pre-trained weights with `weights_dir=`; the platform
    only serves them and never retrains. Set `supports_training=False`.

The platform requires **exactly one** of `supports_training` / bundled weights —
a trainable package must NOT bundle weights, and a weights package must set
`supports_training=False`. `export_payload_for_hosting` enforces this and fails loudly.

Run:
    python notebooks/export_to_hosting.py

Or via workerctl CLI:
    workerctl export-payload --config model.json --out build/my_lgbm_package [--zip]
"""

import json
from pathlib import Path

from allora_forge_builder_kit import WorkerManager, ModelSpec

print("=" * 80)
print("Export to hosting — build a deployable package")
print("=" * 80)

# 1. Describe the model. These fields are model-INTRINSIC — baked into the
#    package's config.json and read by the worker at runtime. Pair/timeframe/topic
#    are NOT here: they are chosen per deployment.
spec = ModelSpec(
    model_type="my_lgbm",                 # entry-point name + OCI path segment; [a-z0-9_-]
    engineered_specs=[                     # derived features, computed identically at train & serve
        {"kind": "log_return", "window_bars": 6},
        {"kind": "log_return", "window_bars": 12},
    ],
    number_of_input_bars=24,               # lookback window the model consumes
    target_bars=24,                        # forecast horizon (bars ahead)
    hyperparameters={"n_estimators": 500}, # passed to the LightGBM model at train time
    data_source="binance",                 # "binance" or "allora"
    days_of_history=180,                   # history the platform backfills before training
    supports_training=True,                # train-on-platform (no bundled weights)
)

# `validate()` runs inside export_payload_for_hosting too; calling it here
# surfaces config errors early with a clean message.
spec.validate()

manager = WorkerManager(reconcile_on_start=False)
out = Path("build/my_lgbm_package")

# 2a. Train-on-platform: no weights. The platform trains via a scheduled
#     `allora-worker train` job; the worker serves the artifact once it exists.
pkg = manager.export_payload_for_hosting(spec, out_dir=out)

# What the package contains:
print()
for p in sorted(pkg.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(pkg)}")

manifest = json.loads((pkg / "manifest.json").read_text())
print(f"\nmanifest.json: {json.dumps(manifest, indent=2)}")
# Note: manifest carries NO pair/timeframe/topic (those are deploy-time), and
# code_hash is a deterministic sha256 over the code (weights/ excluded).

# 2b. Train-locally: bundle pre-trained weights instead. The platform serves them
#     and never retrains, so supports_training MUST be False (exactly-one rule).
#
#     weights_out = Path("build/my_lgbm_weights_package")
#     manager.export_payload_for_hosting(
#         ModelSpec(**{**spec.__dict__, "supports_training": False}),
#         out_dir=weights_out,
#         weights_dir="path/to/trained/weights",
#     )

# 3. Upload: zip the package and POST to the Allora hosting platform.
#
#    Python — pass zip_output=True to get a ready-to-upload archive in one call:
#
#       archive = manager.export_payload_for_hosting(spec, out_dir=out, zip_output=True)
#
#    CLI — write the spec as JSON first, then use workerctl:
#
#       Path("model.json").write_text(json.dumps(spec.__dict__))
#       workerctl export-payload --config model.json --out build/my_lgbm_package --zip
#
#    then POST the resulting .zip to the Allora hosting platform (POST /api/v1/models).
print()
print("To produce an upload-ready zip from Python:")
print("  manager.export_payload_for_hosting(spec, out_dir=out, zip_output=True)")
