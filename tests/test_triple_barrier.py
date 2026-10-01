"""Three real-Atlas integration tests. Opt in with RUN_INTEGRATION_TESTS=1.

All generated state belongs to one unique directory, removed in fixture teardown.
Existing test fixtures and worker state are never modified.
"""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from uuid import uuid4

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from allora_forge_builder_kit import AlloraMLWorkflow, WorkerManager, WorkerMonitor, AlloraSDKEventFetcher

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'notebooks/example_triple_barrier_walkthrough.py'
TARGETS = ['target_down', 'target_neutral', 'target_up']
PREDS = ['pred_down', 'pred_neutral', 'pred_up']
pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def integration_run():
    if os.environ.get('RUN_INTEGRATION_TESTS') != '1':
        pytest.skip('Set RUN_INTEGRATION_TESTS=1 for real Atlas and on-chain deployment')
    if not os.environ.get('ALLORA_API_KEY'):
        pytest.fail('ALLORA_API_KEY required')
    run_id = 'triple-barrier-test-' + uuid4().hex
    root = ROOT / 'notebooks/runs' / run_id
    root.mkdir(parents=True)
    print(f'Integration run: {run_id}', flush=True)
    state = dict(root=root, run_id=run_id, example=None)
    try:
        workflow = AlloraMLWorkflow(
            tickers=['hl_xyzgold_1min'], number_of_input_bars=24, target_bars=24,
            interval='1h', target_type='triple_barrier', data_source='allora',
            api_key=os.environ['ALLORA_API_KEY'], base_dir=str(root / 'cache'))
        workflow.backfill(start=pd.Timestamp.now(tz='UTC').to_pydatetime() - pd.Timedelta(days=220))
        state['workflow'] = workflow
        state['data'] = workflow.get_full_feature_target_dataframe().reset_index()
        yield state
    finally:
        # A failed deploy can still have launched a worker. Inspect only this run's
        # isolated registry, never the caller's working directory or fleet DB.
        worker_dir = root / 'worker'
        if (worker_dir / 'worker_state.db').exists():
            previous = Path.cwd()
            try:
                os.chdir(worker_dir)
                manager = WorkerManager(reconcile_on_start=False, auto_monitor_sync=False)
                for worker in manager.status_all():
                    manager.stop_worker(worker['topic_id'], worker['address'])
            finally:
                os.chdir(previous)
        shutil.rmtree(root)
        print(f'Cleaned {run_id}; only this run\'s worker/files removed', flush=True)


def _reference_target(raw, row):
    """Readable independent replay: raw minutes → hourly windows → first touch.

    No production target method or precomputed barriers are used here.
    """
    t = row.open_time + pd.Timedelta(hours=1)
    historical_end = row.open_time
    historical_start = historical_end - pd.Timedelta(hours=2400)
    history = raw.loc[(raw.index >= historical_start) & (raw.index < historical_end)]
    expected = pd.date_range(historical_start, historical_end, freq='min', inclusive='left')
    if not history.index.equals(expected):
        return None
    hourly = history.resample('1h').agg({'high': 'max', 'low': 'min'})
    ranges = []
    # Reproduce SQL WHERE first, then inclusive RANGE through CURRENT ROW.
    for sample in pd.date_range(historical_end-pd.Timedelta(hours=2400), historical_end, freq='h', inclusive='left'):
        window = hourly[(hourly.index >= sample-pd.Timedelta(hours=24)) & (hourly.index <= sample)]
        assert len(window) == min(25, int((sample-historical_start)/pd.Timedelta(hours=1)) + 1)
        ranges.append(np.log(window.high.max()) - np.log(window.low.min()))
    atr = sum(ranges)/len(ranges)
    start, end = t-pd.Timedelta(minutes=1), t+pd.Timedelta(hours=24)-pd.Timedelta(minutes=1)
    future = raw[(raw.index >= start) & (raw.index < end)]
    if not future.index.equals(pd.date_range(start, end, freq='min', inclusive='left')):
        return None
    base = float(raw.loc[start, 'close'])
    lower, upper = base*np.exp(-.25*atr), base*np.exp(.25*atr)
    for minute in future.itertuples():
        if minute.low <= lower:
            return np.array([1, 0, 0]), lower, upper, minute.Index+pd.Timedelta(minutes=1), lower
        if minute.high >= upper:
            return np.array([0, 0, 1]), lower, upper, minute.Index+pd.Timedelta(minutes=1), upper
    return np.array([0, 1, 0]), lower, upper, end, float(future.close.iloc[-1])


def test_workflow_dataset(integration_run):
    data = integration_run['data']
    assert set(TARGETS) <= set(data)
    resolved = data.dropna(subset=TARGETS)
    assert len(resolved)
    assert np.isin(resolved[TARGETS], [0, 1]).all()
    np.testing.assert_array_equal(resolved[TARGETS].sum(axis=1), 1)
    assert data.loc[data[TARGETS].isna().any(axis=1), TARGETS].isna().all().all()
    raw = integration_run['workflow']._dm.load_polars(['hl_xyzgold_1min']).to_pandas().set_index('open_time').sort_index()
    samples = resolved.sample(min(9, len(resolved)), random_state=42)
    for c in TARGETS:
        candidates = resolved[resolved[c] == 1]
        if len(candidates):
            samples = pd.concat([samples, candidates.iloc[[len(candidates)//2]]])
    for _, row in samples.drop_duplicates('open_time').iterrows():
        result = _reference_target(raw, row)
        assert result is not None, f'Incomplete raw reference at {row.open_time}'
        label, lower, upper, exit_time, exit_price = result
        context = f'{row.open_time}: expected={label}, actual={row[TARGETS].tolist()}, barriers={lower,upper}, hit={exit_time}'
        np.testing.assert_array_equal(row[TARGETS].to_numpy(dtype=float), label, err_msg=context)
        np.testing.assert_allclose([row.tb_lower, row.tb_upper, row.tb_exit_price], [lower, upper, exit_price], rtol=1e-10)
        assert row.tb_exit_time == exit_time, context
    # Verify real unresolved leading/trailing rows rather than fabricating gaps.
    unresolved = data[data[TARGETS].isna().all(axis=1)]
    if len(unresolved):
        for _, row in unresolved.iloc[[0, -1]].iterrows():
            assert _reference_target(raw, row) is None
    print(f'Independent replay matched {len(samples.drop_duplicates("open_time"))} rows; {len(resolved)} resolved total', flush=True)


def _run_example(state):
    if state['example'] is not None:
        return state['example']
    output = state['root'] / 'example'
    log = state['root'] / 'example.log'
    with log.open('w') as handle:
        result = subprocess.run([sys.executable, str(SCRIPT), '--cache-dir', str(state['root']/'cache'),
                                 '--output-dir', str(output), '--skip-backfill'],
                                cwd=state['root'], stdout=handle, stderr=subprocess.STDOUT, timeout=600,
                                env={**os.environ, 'MPLBACKEND': 'Agg', 'NUMBA_NUM_THREADS': '2'})
    assert result.returncode == 0, log.read_text()[-6000:]
    state['example'] = output
    return output


def test_example_outputs(integration_run):
    output = _run_example(integration_run)
    expected = ['predict.pkl', 'config.json', 'metrics.json', 'predictions.csv', 'report.txt',
                'triple_barrier_example.png', 'confusion_matrix.png', 'directional_payoff.png',
                'example_trades.png', 'cumulative_trade_pnl.png', 'trades.csv', 'walk_forward_folds.png']
    assert all((output/name).stat().st_size > 0 for name in expected)
    config = json.loads((output/'config.json').read_text())
    assert config['fold_periods'][-1]['role'] == 'final_holdout'
    assert all(p['role'] == 'model_selection' for p in config['fold_periods'][:-1])
    frame = pd.read_csv(output/'predictions.csv')
    p = frame[PREDS].to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_array_equal(frame[TARGETS].sum(axis=1), 1)
    report = json.loads((output/'metrics.json').read_text())
    assert len(report['criteria']) == 6 and report['nvalid'] == len(frame)
    assert report['participation_kind'] == 'offline'
    assert report['eligible'] == all(c['passed'] for c in report['criteria'])
    trades = pd.read_csv(output/'trades.csv')
    np.testing.assert_allclose(trades.gross_pnl, trades.direction*(trades.exit_price-trades.entry_price)*trades['size'])
    with (output/'predict.pkl').open('rb') as handle:
        predict = cloudpickle.load(handle)
    previous = Path.cwd()
    try:
        os.chdir(integration_run['root'])  # any runtime cache is run-owned too
        live = predict()
    finally:
        os.chdir(previous)
    assert set(live) == {'down', 'neutral', 'up'}
    assert all(np.isfinite(v) and v >= 0 for v in live.values())
    assert sum(live.values()) == pytest.approx(1)
    print('Example outputs and live reloaded artifact verified', flush=True)


def test_deploy_worker(integration_run):
    output = _run_example(integration_run)
    worker_dir = integration_run['root']/'worker'
    worker_dir.mkdir()
    env = {**os.environ, 'TOPIC_ID': '87', 'PREDICT_PKL': str(output/'predict.pkl'), 'ALLORA_NETWORK': 'testnet'}
    # Avoid inheriting any unrelated managed-wallet configuration.
    for name in ('FORGE_API_KEY', 'FORGE_SIGNING_WALLET_ID', 'PRIVATE_KEY', 'MNEMONIC', 'MNEMONIC_FILE'):
        env.pop(name, None)
    with (worker_dir/'deploy.log').open('w') as log:
        result = subprocess.run([sys.executable, str(ROOT/'notebooks/deploy_worker.py')], cwd=worker_dir,
                                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
    assert result.returncode == 0, (worker_dir/'deploy.log').read_text()[-4000:]
    db = worker_dir/'worker_state.db'
    manager = WorkerManager(db_path=db, secrets_path=worker_dir/'worker_secrets.json',
                            reconcile_on_start=False, auto_monitor_sync=False)
    workers = manager.status_all()
    assert len(workers) == 1
    address = workers[0]['address']
    monitor = WorkerMonitor(db_path=db, event_fetcher=AlloraSDKEventFetcher(network='testnet'))
    deadline = time.monotonic() + int(os.environ.get('TRIPLE_BARRIER_DEPLOY_TIMEOUT', '1200'))
    while time.monotonic() < deadline:
        sync = monitor.sync_once()
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT value_text,tx_hash FROM monitor_events WHERE topic_id=87 AND address=? AND event_type='submission' AND status='success'", (address,)).fetchall()
        for value, tx in rows:
            try:
                vector = json.loads(value)
            except (ValueError, TypeError):
                continue
            if isinstance(vector, dict) and set(vector) == {'down', 'neutral', 'up'}:
                assert sum(float(v) for v in vector.values()) == pytest.approx(1)
                assert tx
                print(f'Labeled submission verified: topic=87 address={address} tx={tx}', flush=True)
                return
        logs = '\n'.join(p.read_text(errors='replace')[-6000:] for p in (worker_dir/'worker_logs').glob('*.log'))
        if 'is not whitelisted on topic' in logs:
            pytest.fail('External topic permission blocker: ' + logs[-6000:])
        if 'Traceback (most recent call last)' in logs and manager.status_worker(87, address)['status'] != 'running':
            pytest.fail(logs[-6000:])
        print(f'Waiting for topic 87 labeled submission: {address}; monitor errors={sync["errors"]}', flush=True)
        time.sleep(15)
    logs = '\n'.join(p.read_text(errors='replace')[-4000:] for p in (worker_dir/'worker_logs').glob('*.log'))
    pytest.fail('No confirmed labeled submission within timeout\n'+logs)
