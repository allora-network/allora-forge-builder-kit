"""Three real-Atlas integration tests. Opt in with RUN_INTEGRATION_TESTS=1.

All generated state belongs to one unique directory, removed in fixture teardown.
Existing test fixtures and worker state are never modified.
"""
import json
import copy
import asyncio
import runpy
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


def _verify_target_edges(workflow):
    """Small deterministic checks within the dataset integration test."""
    import polars as pl
    for interval in ('7m', '50m', '1h'):
        probe = copy.copy(workflow)
        probe.interval = interval
        width = pd.Timedelta(interval)
        times = pd.date_range('2026-01-02', periods=int(width / pd.Timedelta(minutes=1)) * 210,
                              freq='min', tz='UTC')
        raw = pl.from_pandas(pd.DataFrame(dict(open_time=times, open=100., high=101.,
                                               low=99., close=100., volume=1.)))
        native = probe.resample_ohlcv_polars(raw, freq=interval)
        result = probe.compute_triple_barrier_target_polars(native, 2, raw).to_pandas()
        assert [c for c in result if c.startswith('target_')] == TARGETS
        resolved = result.dropna(subset=TARGETS)
        assert len(resolved) > 0, interval
        # Every minute touches both fixed barriers: tie must resolve down.
        assert (resolved.target_down == 1).all()
        np.testing.assert_allclose(resolved.tb_upper, 100 * np.exp(.25 * np.log(101/99)))
        np.testing.assert_allclose(resolved.tb_lower, 100 * np.exp(-.25 * np.log(101/99)))
        for bad in (0, -1, True):
            with pytest.raises(ValueError):
                probe.compute_triple_barrier_target_polars(native, bad, raw)
        if interval == '1h':
            # Removing a minute keeps required history unresolved; no silent neutral.
            missing = raw.filter(pl.col('open_time') != times[60 * 100].to_pydatetime())
            gaps = probe.compute_triple_barrier_target_polars(native, 2, missing).to_pandas()
            assert gaps[TARGETS].isna().all(axis=1).sum() > result[TARGETS].isna().all(axis=1).sum()

    # Trace the established live path without network or wall-clock heuristics.
    # A partial final hour is handled identically for old/new target types.
    from types import SimpleNamespace
    times = pd.date_range('2026-01-01', periods=185, freq='min', tz='UTC')
    prices = 100 + np.arange(len(times)) / 100
    raw = pd.DataFrame(dict(open=prices, high=prices+1, low=prices-1,
                            close=prices, volume=np.ones(len(times))), index=times)
    raw.index.name = 'open_time'
    probe = copy.copy(workflow)
    probe.number_of_input_bars = 2
    probe._dm = SimpleNamespace(get_live_1min_data=lambda *a, **kw: raw)
    triple = probe.get_live_features('test')
    probe.target_type = 'log_return'
    pd.testing.assert_frame_equal(triple, probe.get_live_features('test'))
    assert triple.index[-1] == times[-1].floor('h')
    # The final row is included (right-inclusive search), contrary to R04's claim.
    assert triple.feature_close_0.iloc[0] == pytest.approx(prices[179] / prices[184])


def _verify_review_evaluation_and_monitoring():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from allora_forge_builder_kit import PerformanceEvaluator
    from allora_forge_builder_kit.worker_monitor import _labeled_value_text
    from allora_sdk.rpc_client.protos.emissions.v10 import (
        GetWorkerLatestInputInferenceByTopicIdResponse, InputInference, InputLabeledValue)

    history = pd.date_range('2026-01-01', periods=105, freq='h', tz='UTC')
    labels = np.eye(3)[np.arange(105) % 3]
    queries = history[[0, 50, 104]]
    expected = np.array([labels[:1].mean(0), labels[:51].mean(0), labels[5:105].mean(0)])
    for hu, qu in [('us', 'ns'), ('ns', 'us')]:
        actual = PerformanceEvaluator.causal_class_baseline(
            queries.as_unit(qu), history[::-1].as_unit(hu), labels[::-1])
        np.testing.assert_allclose(actual, expected)
    ev = PerformanceEvaluator(target_type='triple_barrier')
    truth = np.eye(3)[np.arange(30) % 3]
    wrong = np.roll(truth, 1, axis=1)
    baseline = np.full(truth.shape, 1/3)
    fail = ev.evaluate(truth, wrong, baseline_probabilities=baseline, n_expected_epochs=30)
    assert not fail['eligible']
    assert not fail['criteria'][0]['passed']
    good = ev.evaluate(truth, truth, baseline_probabilities=baseline, n_expected_epochs=30)
    assert good['eligible'] and good['provisional']
    for bad in (np.zeros((2, 3)), np.full((2, 3), np.nan), np.ones((2, 2))):
        with pytest.raises(ValueError):
            ev.validate_probabilities(bad)

    pairs = [{'label': 'down', 'value': '.2'}, {'label': 'neutral', 'value': '.3'},
             {'label': 'up', 'value': '.5'}]
    expected_text = {'down': '.2', 'neutral': '.3', 'up': '.5'}
    for values in (json.dumps(pairs), json.dumps(json.dumps(pairs)), np.array(pairs, dtype=object)):
        assert json.loads(_labeled_value_text(values)) == expected_text
    for empty in ('', '  ', None, [], {}, iter([]), np.array([], dtype=object)):
        assert _labeled_value_text(empty, '42') == '42'
    assert _labeled_value_text('not-json') == 'not-json'
    entries = [InputLabeledValue(**p) for p in pairs]
    response = GetWorkerLatestInputInferenceByTopicIdResponse(
        latest_input_inference=InputInference(block_height=123, values=entries))
    query = SimpleNamespace(get_worker_latest_input_inference_by_topic_id=AsyncMock(return_value=response))
    client = SimpleNamespace(emissions=SimpleNamespace(query=query))
    # No tx pages: the snapshot must independently populate inference events.
    events = asyncio.run(AlloraSDKEventFetcher(max_pages=0)._fetch(client, 87, 'test', None))
    snapshots = [e for e in events if e['event_type'] == 'inference']
    assert len(snapshots) == 1
    assert json.loads(snapshots[0]['value_text']) == expected_text
    assert snapshots[0]['value_num'] is None  # labeled JSON has no scalar projection


    # Snapshots supply the latest value but are not additional submissions.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix='triple-barrier-monitor-') as directory:
        fetched = [snapshots[0]]
        monitor = WorkerMonitor(
            db_path=Path(directory) / 'monitor.db',
            event_fetcher=lambda *_: fetched,
        )
        monitor.register_target(87, 'test')
        assert monitor.sync_once()['inserted'] == 1
        summary = monitor.get_summary(87, 'test')
        assert summary['inference_count'] == 0
        assert summary['last_inference']['value_text'] == snapshots[0]['value_text']
        confirmed = dict(snapshots[0], event_id='inference:tx:123',
                         status='success', tx_hash='tx')
        submission = dict(confirmed, event_id='submit:tx:123', event_type='submission')
        fetched.extend([confirmed, submission])
        assert monitor.sync_once()['inserted'] == 2
        assert monitor.sync_once()['inserted'] == 0
        summary = monitor.get_summary(87, 'test')
        assert summary['inference_count'] == 1
        assert summary['submission_success'] == 1
        assert summary['events_total'] == 3  # all stored records, including snapshots
        assert summary['last_inference']['tx_hash'] == 'tx'
        for period in summary['period_metrics'].values():
            assert period['inference_count'] == 1
        # Preserve the previous counting behavior for legacy rows without status.
        fetched.append(dict(confirmed, event_id='legacy-inference', status=None))
        assert monitor.sync_once()['inserted'] == 1
        summary = monitor.get_summary(87, 'test')
        assert summary['inference_count'] == 2
        assert all(p['inference_count'] == 2 for p in summary['period_metrics'].values())


def test_workflow_dataset(integration_run):
    _verify_target_edges(integration_run['workflow'])
    # Reject coarse source candles before loading them; preserve the general
    # native-bar builder rather than forcing all research horizons to 1h/24.
    from types import SimpleNamespace
    probe = copy.copy(integration_run['workflow'])
    probe._dm = SimpleNamespace(minute_candles_available=False)
    with pytest.raises(ValueError, match='one-minute source candles'):
        probe.get_full_feature_target_dataframe()
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


def _verify_editable_feature_artifact():
    import allora_forge_builder_kit as kit
    example = runpy.run_path(str(SCRIPT))
    recipe = [{'kind': 'log_return', 'window_bars': 1}]
    base = pd.DataFrame({'feature_close_0': [.9, .8], 'feature_close_1': [.95, .9],
                         'feature_close_2': [1., 1.], 'feature_high_2': [1.02, 1.03],
                         'feature_low_2': [.98, .97]})
    default_engineer = example['engineer_features']

    def custom_engineer(frame, specs, input_bars):
        frame, added = default_engineer(frame, specs, input_bars)
        frame['custom_range'] = frame[f'feature_high_{input_bars-1}'] - frame[f'feature_low_{input_bars-1}']
        return frame, added + ['custom_range']

    trained, added = custom_engineer(base, recipe, 3)
    features = list(base) + added
    expected = trained.iloc[[-1]][features]

    class CheckingModel:
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, frame):
            pd.testing.assert_frame_equal(frame, expected)
            return np.array([[.2, .3, .5]])

    predict = example['make_predict'](CheckingModel(), features, 'test', 3,
                                      feature_fn=custom_engineer, engineered_specs=recipe)
    recipe[0]['window_bars'] = 2  # artifact must retain its training recipe
    loaded = cloudpickle.loads(cloudpickle.dumps(predict))
    from types import SimpleNamespace
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('ALLORA_API_KEY', 'test-feature-key')
        patch.setattr(kit, 'AlloraMLWorkflow', lambda **kw: SimpleNamespace(
            get_live_features=lambda ticker: base.iloc[[-1]].copy()))
        assert loaded() == {'down': .2, 'neutral': .3, 'up': .5}


def test_example_outputs(integration_run):
    _verify_editable_feature_artifact()
    _verify_review_evaluation_and_monitoring()
    from allora_forge_builder_kit import PerformanceEvaluator
    from allora_forge_builder_kit.worker_monitor import _labeled_value_text
    from allora_sdk.rpc_client.protos.emissions.v10 import InputLabeledValue

    queries = pd.date_range('2026-01-01', periods=2, tz='UTC')
    for empty in ([], np.empty((0, 3))):
        np.testing.assert_allclose(
            PerformanceEvaluator.causal_class_baseline(queries, [], empty),
            np.full((2, 3), 1/3))
    with pytest.raises(ValueError):
        PerformanceEvaluator.causal_class_baseline(queries, [], np.empty((0, 2)))
    with pytest.raises(ValueError):
        PerformanceEvaluator.causal_class_baseline(queries, queries[:1], np.empty((0, 3)))

    entries = tuple(InputLabeledValue(label=k, value=str(v))
                    for k, v in zip(('down', 'neutral', 'up'), (.2, .3, .5)))
    for values in (entries, iter(entries), list(entries)):
        assert json.loads(_labeled_value_text(values)) == {'down': '0.2', 'neutral': '0.3', 'up': '0.5'}
    assert _labeled_value_text((InputLabeledValue(label='y', value='42'),)) == '42'
    assert _labeled_value_text([], scalar='42') == '42'

    example = runpy.run_path(str(SCRIPT))
    key_globals = example['api_key'].__globals__
    key_root = integration_run['root'] / 'credentials'
    (key_root / 'notebooks').mkdir(parents=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(key_root / 'notebooks')
        patch.setitem(key_globals, '__file__', str(key_root / 'notebooks' / 'example.py'))
        patch.setenv('ALLORA_API_KEY', '   ')
        (key_root / 'notebooks' / '.allora_api_key').write_text('  ')
        with pytest.raises(RuntimeError, match='Set ALLORA_API_KEY'):
            example['api_key']()
        sentinel = 'test-only-review-credential'
        (key_root / '.allora_api_key').write_text('  ' + sentinel + '\n')
        assert example['api_key']() == sentinel
        patch.setitem(key_globals, 'AlloraMLWorkflow', lambda **kwargs: kwargs)
        config = example['make_workflow'](87, key_root / 'cache')
        assert config['api_key'] == sentinel
        assert os.environ['ALLORA_API_KEY'] == sentinel
        artifact = example['make_predict'](None, [], 'hl_xyzgold_1min', 100)
        assert sentinel.encode() not in cloudpickle.dumps(artifact)
        patch.setenv('ALLORA_API_KEY', '  env-key  ')
        assert example['api_key']() == 'env-key'
        patch.setenv('ALLORA_API_KEY', '   ')
        with pytest.raises(RuntimeError, match='required for live Atlas'):
            artifact()

    output = _run_example(integration_run)
    expected = ['predict.pkl', 'config.json', 'metrics.json', 'predictions.csv', 'report.txt',
                'triple_barrier_example.png', 'confusion_matrix.png', 'directional_payoff.png',
                'example_trades.png', 'cumulative_trade_pnl.png', 'trades.csv', 'walk_forward_folds.png']
    assert all((output/name).stat().st_size > 0 for name in expected)
    config = json.loads((output/'config.json').read_text())
    count = config['holdout_folds']
    assert count == 2
    assert all(p['role'] == 'oos_holdout' for p in config['fold_periods'][-count:])
    assert all(p['role'] == 'model_selection' for p in config['fold_periods'][:-count])
    assert all(len(c['fold_losses']) == config['folds'] - count for c in config['cv'])
    assert config['best_params'] == min(config['cv'], key=lambda c: c['mean_log_loss'])['params']
    assert config['search_grid'] and config['fixed_params']
    from sklearn.model_selection import ParameterGrid
    assert [c['params'] for c in config['cv']] == list(ParameterGrid(config['search_grid']))
    frame = pd.read_csv(output/'predictions.csv')
    expected_folds = config['fold_periods'][-count:]
    assert set(frame.oos_fold) == {p['fold'] for p in expected_folds}
    assert not frame.open_time.duplicated().any()
    for period in expected_folds:
        rows = frame[frame.oos_fold == period['fold']]
        assert len(rows) == period['validation_rows']
        assert pd.Timestamp(rows.tb_prediction_time.min()) == pd.Timestamp(period['validation_start'])
        assert pd.Timestamp(period['train_end_exclusive']) <= pd.Timestamp(period['validation_start'])
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
