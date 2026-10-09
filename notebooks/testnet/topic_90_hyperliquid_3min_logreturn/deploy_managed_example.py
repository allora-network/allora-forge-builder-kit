#!/usr/bin/env python3
"""Package the exported topic-90 models for the unchanged WorkerManager runtime.

Run from the repository root:
  .venv/bin/python notebooks/testnet/topic_90_hyperliquid_3min_logreturn/deploy_managed_example.py --model-run PATH
Add --deploy to allocate/start a local-custody worker per exported model. No training or backfill.
Loading this trusted artifact bootstraps Atlas history before the listener starts.
"""
from pathlib import Path
import argparse
import ast
import json
import os
import cloudpickle
import joblib

BASE_DIR = Path(__file__).resolve().parent


def make_artifact(bundle, ranking, inference_source, rpc_url):
    # Local class is serialized by value: the deployed artifact needs neither this
    # script nor example.py, and never imports/runs the training walkthrough.
    class Topic90Prediction:
        def __init__(self):
            self.bundle = bundle
            self.ranking = list(ranking)
            self.source = inference_source
            self.rpc_url = rpc_url
            self._predict = None
            self._buffer = None

        def __getstate__(self):
            return {k: v for k, v in self.__dict__.items() if k not in ('_predict', '_buffer')}

        def __setstate__(self, state):
            self.__dict__.update(state)
            self._predict = self._buffer = None
            self.initialize()

        def initialize(self):
            if self._predict is not None:
                return
            import atexit
            import time
            from datetime import datetime, timedelta, timezone
            import numpy as np
            import pandas as pd
            import polars as pl
            import requests
            from allora_forge_builder_kit import AtlasDataManager, AlloraMLWorkflow
            metadata = self.bundle['metadata']
            atlas = AtlasDataManager(api_key=os.environ['ALLORA_API_KEY'], interval='3m')
            symbols = sorted(set(metadata['training_universe']) & set(atlas.discover_hl_universe()))
            workflow = object.__new__(AlloraMLWorkflow)
            namespace = dict(np=np, pd=pd, pl=pl, requests=requests, time=time, datetime=datetime,
                             timedelta=timedelta, timezone=timezone, model=self.bundle['model'],
                             workflow=workflow, LOOKBACK=metadata['lookback'],
                             feature_columns=metadata['feature_columns'],
                             prediction_scale=self.bundle.get('calibration', {}).get('scale', 1.0))
            exec(self.source, namespace)
            buffer = namespace['LiveMinuteBuffer'](atlas, symbols, lookback=metadata['lookback'])
            namespace['live_buffer'] = buffer
            try:
                print('Managed artifact startup:', buffer.start(), flush=True)
                namespace['predict_live'](datetime.now(timezone.utc).replace(second=0, microsecond=0))
            except Exception:
                buffer.stop()
                raise
            self._buffer = buffer
            self._predict = namespace['predict_live']
            atexit.register(buffer.stop)

        def __call__(self, context: 'RunContext'):
            import time
            import pandas as pd
            import requests
            # A synchronous public query avoids crossing the SDK client's event loop.
            response = requests.get(self.rpc_url.rstrip('/') + '/block',
                                    params={'height': context.nonce}, timeout=10)
            response.raise_for_status()
            block = response.json()['result']['block']
            if int(block['header']['height']) != context.nonce:
                raise RuntimeError('RPC returned a different block height')
            T = pd.to_datetime(block['header']['time'], utc=True).floor('min').to_pydatetime()
            time.sleep(10)
            values = self._predict(T)
            selected = {label: values[label] for label in self.ranking if label in values}
            selected = dict(list(selected.items())[:100])
            if not selected:
                raise RuntimeError('No eligible assets this round')
            return selected

    return Topic90Prediction()


def package(model_run, output, rpc_url):
    bundle = joblib.load(model_run / 'model.joblib')
    if bundle['metadata']['interval'] != '3m':
        raise ValueError('This adapter expects a 3-minute topic-90 model')
    import pandas as pd
    ranking = pd.read_csv(BASE_DIR / 'results/hl_30day_volume_ranking.csv')
    # Preserve the same startup ranking used by the walkthrough.
    ranking = ranking.sort_values('volume_usd', ascending=False)
    labels = ranking['symbol'].str.removeprefix('hl_').str.removesuffix('_1min').str.lower().tolist()
    tree = ast.parse((BASE_DIR / 'example.py').read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in ('LiveMinuteBuffer', 'predict_live')]
    if len(nodes) != 2:
        raise ValueError('Expected the example buffer and inference definitions')
    source = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    artifact = make_artifact(bundle, labels, source, rpc_url)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(cloudpickle.dumps(artifact))
    print('Packaged:', output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run', type=Path, required=True)
    parser.add_argument('--rpc-url', default='https://allora-rpc.testnet.allora.network')
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    run = args.model_run.resolve()
    manifest = run / 'top_models.json'
    model_dirs = [run / row['directory'] for row in json.loads(manifest.read_text())] if manifest.exists() else [run]
    artifacts = [(directory, package(directory, directory / 'predict_worker.pkl', args.rpc_url))
                 for directory in model_dirs]
    if not args.deploy:
        return
    from allora_forge_builder_kit import WorkerManager
    manager = WorkerManager(network='testnet', reconcile_on_start=False)
    for directory, artifact in artifacts:
        mapping = directory / 'managed_worker.json'
        address = json.loads(mapping.read_text())['address'] if mapping.exists() else None
        result = manager.deploy_worker(topic_id=90, artifact_path=artifact, address=address, replace=bool(address))
        mapping.write_text(json.dumps(dict(topic_id=90, address=result.address_assigned, artifact=str(artifact)), indent=2))
        manager.start_worker(90, result.address_assigned)
        print(manager.status_worker(90, result.address_assigned))



if __name__ == '__main__':
    main()
