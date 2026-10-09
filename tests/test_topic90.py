"""Topic 90: Atlas data, log-return evaluation, training, and managed workers.

Uses synthetic data and mocked services; no live submissions or backfill.
"""
import ast
import asyncio
import gc
import importlib.util
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cloudpickle
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pytest
import requests

from allora_forge_builder_kit import AlloraMLWorkflow, PerformanceEvaluator
from allora_forge_builder_kit.atlas_data_manager import AtlasDataManager
from allora_forge_builder_kit.worker_runtime import _ArtifactCaller


# Atlas bulk candles and universe discovery

@pytest.fixture
def manager(tmp_path):
    return AtlasDataManager(api_key="test", base_dir=str(tmp_path), auto_acquire_tag=False)


def test_bulk_preserves_symbols_partials_gaps_and_zero_volume(manager, monkeypatch):
    def row(timestamp, volume):
        return {"timestamp": timestamp, "values": dict(open=1, high=2, low=1, close=2, volume=volume)}
    response = Mock()
    response.json.return_value = {"datasets": [
        {"rows": [row("2026-10-06T12:01:00Z", 0), row("2026-10-06T11:58:00Z", 3)]},
        {"rows": []},
    ]}
    get = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", get)
    start = datetime(2026, 10, 6, 11, 41, tzinfo=timezone.utc)
    end = start + timedelta(minutes=20)
    result = manager.get_bulk_1min_candles(["hl_btc_1min", "hl_eth_1min"], 21, start=start, end=end, timeout=4)
    get.assert_called_once()
    params = get.call_args.kwargs["params"]
    assert params == {"dataset_names": "hl_btc_1min,hl_eth_1min", "limit": 21,
                      "ordering": "-timestamp", "start": start.isoformat(), "end": end.isoformat()}
    assert get.call_args.kwargs["timeout"] == 4
    assert result.index.names == ["symbol", "open_time"]
    assert len(result) == 2
    assert result.index.is_monotonic_increasing
    assert result.iloc[-1].volume == 0
    assert result.index[-1] == ("hl_btc_1min", pd.Timestamp(end))


def test_empty_and_single_name_with_comma(manager, monkeypatch):
    get = Mock(return_value=Mock(json=lambda: {"datasets": [{"rows": []}]}))
    monkeypatch.setattr(requests, "get", get)
    result = manager.get_bulk_1min_candles(["a,b"])
    assert get.call_args.kwargs["params"]["dataset_names"] == '"a,b"'
    assert result.empty
    assert result.index.names == ["symbol", "open_time"]
    assert str(result.index.levels[1].dtype) == "datetime64[ns, UTC]"
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]


@pytest.mark.parametrize("symbols, options", [
    ("hl_btc_1min", {}), ([], {}), (["a", "a"], {}), ([""], {}), (["a"], {"limit": 0}),
    (["a"], {"limit": True}), (["a", "b"], {"limit": 5001}),
    ([str(i) for i in range(1001)], {}), (["a"], {"timeout": 0}),
    (["a"], {"start": datetime(2026, 1, 1)}),
    (["a"], {"start": datetime(2026, 1, 2, tzinfo=timezone.utc),
               "end": datetime(2026, 1, 1, tzinfo=timezone.utc)}),
])
def test_invalid_request_never_calls_atlas(manager, monkeypatch, symbols, options):
    get = Mock()
    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(ValueError):
        manager.get_bulk_1min_candles(symbols, **options)
    get.assert_not_called()


def test_http_failure_and_missing_group_are_not_empty_success(manager, monkeypatch):
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("403")
    monkeypatch.setattr(requests, "get", Mock(return_value=response))
    with pytest.raises(requests.HTTPError):
        manager.get_bulk_1min_candles(["a"])
    response.raise_for_status.side_effect = None
    response.json.return_value = {"datasets": []}
    with pytest.raises(ValueError, match="every requested"):
        manager.get_bulk_1min_candles(["a"])


def test_multiple_assets_keep_identity_and_convert_timezone(manager, monkeypatch):
    row = {"timestamp": "2026-10-06T12:00:00Z", "values":
           {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}}
    response = Mock()
    response.json.return_value = {"datasets": [{"rows": [row]}, {"rows": [row]}]}
    get = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", get)
    local_time = datetime(2026, 10, 6, 14, tzinfo=timezone(timedelta(hours=2)))
    result = manager.get_bulk_1min_candles(["hl_btc_1min", "hl_eth_1min"], end=local_time)
    assert get.call_args.kwargs["params"]["end"] == "2026-10-06T12:00:00+00:00"
    assert list(result.index.get_level_values("symbol")) == ["hl_btc_1min", "hl_eth_1min"]
    assert not list(Path(manager.base_dir).rglob("*.parquet"))


def test_discover_universe_filters_and_uses_manager_credentials(manager, monkeypatch):
    now = datetime.now(timezone.utc)
    def row(name, **overrides):
        values = dict(candle_dataset_name=name, market_group="perp",
                      status="consumable", updated_at=now.isoformat())
        return {"values": {**values, **overrides}}
    response = Mock()
    response.json.return_value = {"next": None, "results": [
        row("hl_eth_1min"), row("hl_btc_1min"), row("hl_btc_1min"),
        row("hl_xyzgold_1min", market_group="xyz"),
        row("retired", status="retired"),
        row("stale", updated_at=(now - timedelta(minutes=6)).isoformat()),
        row("bad", updated_at="invalid"),
    ]}
    get = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", get)
    assert manager.discover_hl_universe() == ["hl_btc_1min", "hl_eth_1min"]
    get.assert_called_once()
    assert get.call_args.args[0] == manager.base_url + "/rows/"
    assert get.call_args.kwargs["headers"] == manager.headers
    response.raise_for_status.assert_called_once()


def test_discover_universe_rejects_truncation(manager, monkeypatch):
    response = Mock()
    response.json.return_value = {"next": "next-page", "results": []}
    monkeypatch.setattr(requests, "get", Mock(return_value=response))
    with pytest.raises(RuntimeError, match="partial registry"):
        manager.discover_hl_universe()


# Log-return evaluation and reference parity

def test_offline_defaults_and_reporting(capsys):
    y=np.random.default_rng(42).normal(size=300)
    evaluator=PerformanceEvaluator()
    report=evaluator.evaluate(y,y*.8,epoch_length_minutes=3)
    assert report['eligible']
    assert report['metrics']['neff_scale']==1
    assert report['participation_basis']=='assumed full offline coverage'
    assert len(report['criteria'])==7
    json.dumps(report,allow_nan=False)
    evaluator.print_report(report)
    assert 'worker-metrics' in capsys.readouterr().out


def test_participation_is_strict_and_aspect_is_scored():
    y=np.random.default_rng(42).normal(size=300)
    evaluator=PerformanceEvaluator()
    report=evaluator.evaluate(y,y*.001,n_expected_epochs=100,n_submitted=90)
    assert not report['passed']['participation']
    assert not report['passed']['log_aspect_ratio']
    assert not report['eligible']


# Golden outputs generated from worker-metrics commit
# bd01f9d2bdb6351e351ecddeb68d0fb101ebc0ee, using the deterministic inputs below.
# Kept inline so CI needs neither the sibling checkout nor another fixture file.
_REFERENCE_METRICS = ['n_eff', 'neff_scale', 'nvalid', 'dir_acc', 'dir_acc_pval', 'dir_acc_ci', 'pearson_r', 'pearson_pval', 'pearson_ci', 'wrmse_imp', 'wrmse_ci', 'wczar_imp', 'wczar_ci', 'log_aspect_ratio', 'log_aspect_ratio_ci', 'participation']
_REFERENCE_CASES = [
    ('signal', 3, 1,
     (300.0, 1.0, 300, 0.56, 0.021654071405395815, (0.5108617041688603, 1.0), 0.14570195467808306,
      0.011517469677559188, (0.03300569001164882, 0.25473973801831823), -0.05264303055066488,
      (-0.13463242194998573, 0.029346360848655967), -0.027897994052618946,
      (-0.11200475510830206, 0.05620876700306417), -0.11372590105297616,
      (-0.13735397969598517, -0.09009782240996712), 0.95),
     (True, True, True, False, False, True, True),
     (300.0, 0.5108617041688603, 0.03300569001164882, -13.463242194998573, -11.200475510830206,
      -0.11372590105297616, 0.95)),
    ('signal', 60, 4,
     (300.0, 1.7320508075688772, 300, 0.56, 0.003556155534467786, (0.522996778913665, 1.0),
      0.14570195467808306, 0.011517469677559188, (0.03300569001164882, 0.25473973801831823),
      -0.05264303055066488, (-0.11494149598072463, 0.009655434879394867), -0.027897994052618946,
      (-0.09180531250729858, 0.036009324402060686), -0.11372590105297616,
      (-0.13735397969598517, -0.09009782240996712), 0.95),
     (True, True, True, False, False, True, True),
     (300.0, 0.522996778913665, 0.03300569001164882, -11.494149598072463, -9.180531250729858,
      -0.11372590105297616, 0.95)),
    ('signal', 240, 8,
     (300.0, 3.4641016151377544, 300, 0.56, 6.216392621392983e-05, (0.5340663444167336, 1.0),
      0.14570195467808306, 0.011517469677559188, (0.03300569001164882, 0.25473973801831823),
      -0.05264303055066488, (-0.09669469791377583, -0.008591363187553935), -0.027897994052618946,
      (-0.07308729229937111, 0.017291304194133217), -0.11372590105297616,
      (-0.13735397969598517, -0.09009782240996712), 0.95),
     (True, True, True, False, False, True, True),
     (300.0, 0.5340663444167336, 0.03300569001164882, -9.669469791377583, -7.308729229937111,
      -0.11372590105297616, 0.95)),
    ('zero', 3, 1,
     (225.0, 1.0, 225, 0.5688888888888889, 0.022750131948179195, (0.5118536223130554, 1.0),
      0.15563941553966593, 0.0194995713482365, (0.025365108486176215, 0.2807157589547017),
      -0.002801186983738546, (-0.11366521232102846, 0.10806283835355136), 0.0028585369755459444,
      (-0.11477148948731938, 0.12048856343841127), -0.05248042127066643,
      (-0.08302969747239394, -0.02193114506893892), 0.95),
     (True, True, True, False, False, True, True),
     (225.0, 0.5118536223130554, 0.025365108486176215, -11.366521232102846, -11.477148948731939,
      -0.05248042127066643, 0.95)),
    ('zero', 60, 4,
     (225.0, 1.7320508075688772, 225, 0.5688888888888889, 0.0038012619240167436,
      (0.5260016425833672, 1.0), 0.15563941553966593, 0.0194995713482365,
      (0.025365108486176215, 0.2807157589547017), -0.002801186983738546,
      (-0.08703962968999375, 0.08143725572251666), 0.0028585369755459444,
      (-0.08652095483508031, 0.0922380287861722), -0.05248042127066643,
      (-0.08302969747239394, -0.02193114506893892), 0.95),
     (True, True, True, False, False, True, True),
     (225.0, 0.5260016425833672, 0.025365108486176215, -8.703962968999376, -8.652095483508031,
      -0.05248042127066643, 0.95)),
    ('zero', 240, 8,
     (225.0, 3.4641016151377544, 225, 0.5688888888888889, 6.929222668244914e-05,
      (0.538878072765551, 1.0), 0.15563941553966593, 0.0194995713482365,
      (0.025365108486176215, 0.2807157589547017), -0.002801186983738546,
      (-0.062366761057926066, 0.056764387090448974), 0.0028585369755459444,
      (-0.06034230778275537, 0.06605938173384726), -0.05248042127066643,
      (-0.08302969747239394, -0.02193114506893892), 0.95),
     (True, True, True, False, False, True, True),
     (225.0, 0.538878072765551, 0.025365108486176215, -6.236676105792607, -6.034230778275537,
      -0.05248042127066643, 0.95)),
    ('constant', 3, 1,
     (300.0, 1.0, 300, 0.5233333333333333, 0.2264601505518622, (0.4742468578215006, 1.0), None, None,
      (None, None), -0.004409238232431445, (-0.01449445789005111, 0.00567598142518822),
      -0.0017435823478277879, (-0.014441872128907143, 0.010954707433251568), -4.0050008209900785,
      (-4.022515160644818, -3.987486481335339), 0.95),
     (True, False, False, False, False, False, True),
     (300.0, 0.4742468578215006, None, -1.449445789005111, -1.4441872128907143, -4.0050008209900785,
      0.95)),
    ('constant', 60, 4,
     (300.0, 1.7320508075688772, 300, 0.5233333333333333, 0.15388774885152556,
      (0.4863054661181078, 1.0), None, None, (None, None), -0.004409238232431445,
      (-0.012072348025925802, 0.0032538715610629116), -0.0017435823478277879,
      (-0.01139219607023683, 0.007905031374581255), -4.0050008209900785,
      (-4.022515160644818, -3.987486481335339), 0.95),
     (True, False, False, False, False, False, True),
     (300.0, 0.4863054661181078, None, -1.20723480259258, -1.139219607023683, -4.0050008209900785,
      0.95)),
    ('constant', 240, 8,
     (300.0, 3.4641016151377544, 300, 0.5233333333333333, 0.07032460420614982,
      (0.4973407559666067, 1.0), None, None, (None, None), -0.004409238232431445,
      (-0.009827875132388348, 0.0010093986675254584), -0.0017435823478277879,
      (-0.008566182539992799, 0.005079017844337223), -4.0050008209900785,
      (-4.022515160644818, -3.987486481335339), 0.95),
     (True, False, False, False, False, False, True),
     (300.0, 0.4973407559666067, None, -0.9827875132388348, -0.8566182539992799, -4.0050008209900785,
      0.95)),
    ('nonfinite', 3, 1,
     (299.0, 1.0, 299, 0.5585284280936454, 0.024633670771412763, (0.5093022486183906, 1.0),
      0.1462074437578298, 0.011506850233745237, (0.03313716824124963, 0.25558241426738765),
      -0.05257886047357441, (-0.13468648082588472, 0.029528759878735905), -0.028009672936473384,
      (-0.11246955772218094, 0.05645021184923417), -0.11277192394799396,
      (-0.13643707701923166, -0.08910677087675625), 0.95),
     (True, True, True, False, False, True, True),
     (299.0, 0.5093022486183906, 0.03313716824124963, -13.468648082588473, -11.246955772218094,
      -0.11277192394799396, 0.95)),
    ('nonfinite', 60, 4,
     (299.0, 1.7320508075688772, 299, 0.5585284280936454, 0.004397443049343399,
      (0.5214568170196014, 1.0), 0.1462074437578298, 0.011506850233745237,
      (0.03313716824124963, 0.25558241426738765), -0.05257886047357441,
      (-0.11496716048119278, 0.009809439534043965), -0.028009672936473384,
      (-0.092185307402676, 0.03616596152972923), -0.11277192394799396,
      (-0.13643707701923166, -0.08910677087675625), 0.95),
     (True, True, True, False, False, True, True),
     (299.0, 0.5214568170196014, 0.03313716824124963, -11.496716048119278, -9.2185307402676,
      -0.11277192394799396, 0.95)),
    ('nonfinite', 240, 8,
     (299.0, 3.4641016151377544, 299, 0.5585284280936454, 9.340852046675804e-05,
      (0.5325454468066875, 1.0), 0.1462074437578298, 0.011506850233745237,
      (0.03313716824124963, 0.25558241426738765), -0.05257886047357441,
      (-0.0966940504756621, -0.008463670471486717), -0.028009672936473384,
      (-0.07338869925447437, 0.017369353381527605), -0.11277192394799396,
      (-0.13643707701923166, -0.08910677087675625), 0.95),
     (True, True, True, False, False, True, True),
     (299.0, 0.5325454468066875, 0.03313716824124963, -9.66940504756621, -7.338869925447437,
      -0.11277192394799396, 0.95)),
]


@pytest.mark.parametrize("kind,horizon,ratio,expected,passed,criterion_values", _REFERENCE_CASES)
def test_matches_reference_worker_metrics(kind, horizon, ratio, expected, passed, criterion_values):
    rng = np.random.default_rng(43)
    y = rng.normal(0, .01, 300)
    p = y * .1 + rng.normal(0, .008, 300)
    if kind == 'zero':
        y[::4] = 0
    if kind == 'constant':
        p[:] = .001
    if kind == 'nonfinite':
        y[3] = np.nan
        p[10] = np.nan
    lags = rng.integers(1, 4, len(y) - 1)
    actual = PerformanceEvaluator().evaluate(
        y, p, epoch_length_minutes=horizon / ratio, lags=lags, gt_ratio=ratio,
        horizon_seconds=horizon * 60, n_expected_epochs=100, n_submitted=95,
    )
    for key, value in zip(_REFERENCE_METRICS, expected):
        observed = actual['metrics'][key]
        if value is None:
            assert observed is None
        elif isinstance(value, tuple) and any(v is None for v in value):
            assert observed == list(value)
        else:
            assert observed == pytest.approx(value, rel=1e-10, abs=1e-12), key
    assert tuple(c['passed'] for c in actual['criteria']) == passed
    for criterion, value in zip(actual['criteria'], criterion_values):
        if value is None:
            assert criterion['value'] is None
        else:
            assert criterion['value'] == pytest.approx(value, rel=1e-10, abs=1e-12)
    assert actual['eligible'] == all(passed)
    assert actual['num_passed'] == sum(passed)
    json.dumps(actual, allow_nan=False)


# Equal-weight asset reporting

def test_asset_summary_equal_weights_and_missing_correlations(tmp_path):
    path=Path(__file__).parents[1]/'notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='report_per_asset')
    ns=dict(pd=pd,np=np)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    rng=np.random.default_rng(42)
    y=rng.normal(size=120)
    index=pd.MultiIndex.from_arrays([['a']*80+['b']*30+['c']*10,pd.date_range('2026-01-01',periods=120,freq='3min',tz='UTC')],names=['ticker','open_time'])
    truth=pd.Series(y,index=index)
    pred=np.r_[y[:80],-y[80:110],np.zeros(10)]
    per_asset,summary,rates=ns['report_per_asset'](PerformanceEvaluator(),truth,pred,tmp_path,'raw')
    assert summary.loc['pearson_r','assets_contributing']==2
    assert summary.loc['pearson_r','mean']==pytest.approx(0,abs=1e-12)
    assert summary.loc['rmse','mean']==pytest.approx(per_asset.rmse.mean())
    assert rates.loc['correlation_ci','fraction_assets_passing']==pytest.approx(1/3)
    assert len(list(tmp_path.glob('*.csv')))==4


# In-memory minute buffer

@pytest.fixture
def example():
    path = Path(__file__).parents[1] / 'notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LiveMinuteBuffer')
    class Clock(datetime):
        current = datetime(2026, 10, 6, 12, 1, 27, tzinfo=timezone.utc)
        @classmethod
        def now(cls, tz=None):
            return cls.current
    namespace = dict(pd=pd, requests=requests, datetime=Clock, timedelta=timedelta, timezone=timezone)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(Buffer=namespace['LiveMinuteBuffer'], clock=Clock)


class FakeAtlas:
    def __init__(self):
        self.calls = []
        self.price = 100.
        self.bad = set()
        self.empty = set()

    def get_bulk_1min_candles(self, symbols, *, limit, start, end, timeout):
        self.calls.append((list(symbols), limit, start, end))
        assert len(symbols) * limit <= 10_000
        assert len(pd.date_range(start, end, freq='min')) <= limit
        if self.bad.intersection(symbols):
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError('missing dataset', response=response)
        records = [dict(symbol=s, open_time=t, open=self.price, high=self.price,
                        low=self.price, close=self.price, volume=0.)
                   for s in symbols if s not in self.empty
                   for t in pd.date_range(start, end, freq='min')]
        df = pd.DataFrame(records, columns=['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume'])
        df['open_time'] = pd.to_datetime(df.open_time, utc=True)
        return df.set_index(['symbol', 'open_time'])


def test_bootstrap_overlap_pruning_and_snapshot_isolation(example):
    atlas = FakeAtlas()
    buffer = example.Buffer(atlas, ['btc', 'eth'])
    result = buffer.refresh()
    assert not result['errors']
    assert len(atlas.calls) == 13  # 260-minute history in inclusive 20m spans.
    assert len(buffer.snapshot()) == 2 * 261
    atlas.price = 200.
    example.clock.current += timedelta(minutes=3)
    result = buffer.refresh()
    assert len(atlas.calls) == 14  # Routine refresh is one request.
    assert atlas.calls[-1][1] == 8  # 3 elapsed minutes + 4 overlap + inclusive endpoint.
    snapshot = buffer.snapshot()
    assert len(snapshot) == 522
    assert snapshot.index.is_unique
    assert (snapshot.groupby(level='symbol').tail(1).close == 200).all()
    assert (snapshot.volume == 0).all()
    snapshot.iloc[0, 0] = -1
    assert buffer.snapshot().iloc[0, 0] != -1


def test_failsafe_uses_successful_query_time_even_when_empty(example):
    atlas = FakeAtlas()
    atlas.empty.add('quiet')
    buffer = example.Buffer(atlas, ['quiet'])
    buffer.refresh()
    calls = len(atlas.calls)
    assert buffer.snapshot().empty
    assert buffer.refresh(only_if_needed=True)['requested'] == 0
    assert len(atlas.calls) == calls
    example.clock.current += timedelta(minutes=5)
    assert buffer.refresh(only_if_needed=True)['requested'] == 1


def test_missing_asset_isolated_and_retried(example):
    atlas = FakeAtlas()
    atlas.bad.add('bad')
    buffer = example.Buffer(atlas, ['btc', 'bad'], lookback=1, margin_minutes=1)
    report = buffer.refresh()
    assert set(report['errors']) == {'bad'}
    assert list(buffer.last_successful_refresh) == ['btc']
    assert set(buffer.snapshot().index.get_level_values('symbol')) == {'btc'}
    atlas.bad.clear()
    assert not buffer.refresh()['errors']
    assert not buffer.last_errors
    assert set(buffer.snapshot().index.get_level_values('symbol')) == {'btc', 'bad'}


def test_start_stop_and_all_failed_startup(example):
    atlas = FakeAtlas()
    buffer = example.Buffer(atlas, ['btc'], lookback=1, margin_minutes=1)
    try:
        buffer.start()
        assert buffer._thread.is_alive()
        assert buffer.start() is None
    finally:
        buffer.stop()
    assert buffer._thread is None
    atlas.bad.add('btc')
    with pytest.raises(RuntimeError, match='every asset'):
        buffer.start()
    assert buffer._thread is None


def test_large_universe_is_chunked(example):
    atlas = FakeAtlas()
    buffer = example.Buffer(atlas, [str(i) for i in range(500)], lookback=1, margin_minutes=1)
    buffer.refresh()
    assert len(atlas.calls) == 2


# Inference alignment and partial candles

@pytest.mark.parametrize("scale", [1., 7.])
@pytest.mark.parametrize("minute", [0, 1, 2])
def test_linear_inference_alignment_and_partial_fallback(minute, scale):
    T = datetime(2026, 10, 6, 12, minute, tzinfo=timezone.utc)
    class Clock:
        @staticmethod
        def now(tz):
            return T
    rows = []
    for symbol in ["hl_btc_1min", "hl_eth_1min"]:
        for i in range(6):
            rows.append(dict(symbol=symbol, open_time=T-timedelta(minutes=6-i),
                             open=100., high=100., low=100., close=100., volume=1.))
    rows.append(dict(symbol="hl_btc_1min", open_time=T,
                     open=100., high=110., low=100., close=110., volume=200.))
    snapshot = pd.DataFrame(rows).set_index(["symbol", "open_time"])
    buffer = Mock()
    buffer.snapshot.return_value = snapshot
    model = Mock()
    model.predict.return_value = [0.01, -0.02]
    workflow = object.__new__(AlloraMLWorkflow)
    cols = [f"feature_{f}_{i}" for i in range(2) for f in ["open", "high", "low", "close", "volume"]]
    path = Path(__file__).parents[1] / "notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py"
    snippet = ast.Module(body=[node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "predict_live"], type_ignores=[])
    ns = dict(datetime=Clock, timezone=timezone, timedelta=timedelta, time=time,
              np=np, pd=pd, pl=pl, live_buffer=buffer, model=model,
              workflow=workflow, prediction_scale=scale, LOOKBACK=2, feature_columns=cols)
    exec(compile(snippet, str(path), "exec"), ns)
    predictions = ns["predict_live"](T)
    assert predictions == pytest.approx({"btc": scale*0.01+np.log(1.1), "eth": scale*-0.02})
    assert np.allclose(model.predict.call_args.args[0][cols], 1.)
    buffer.refresh.assert_called_once()
    model.predict.assert_called_once()


def test_sdk_callback_waits_ten_seconds_without_local_deadline_rejection(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import allora_sdk

    opened = datetime(2026, 10, 6, 12, 1, 27, tzinfo=timezone.utc)
    now = [opened + timedelta(seconds=2)]
    class Clock:
        @staticmethod
        def now(tz):
            return now[0]
    path = Path(__file__).parents[1] / "notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_inference")
    predict = Mock(return_value={"btc": 0.01})
    ns = dict(datetime=Clock, timezone=timezone, asyncio=asyncio, predict_live=predict)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    get_time = AsyncMock(return_value=opened)
    monkeypatch.setattr(allora_sdk, "get_block_time", get_time)
    async def advance_clock(seconds):
        now[0] += timedelta(seconds=seconds)
    sleep = AsyncMock(side_effect=advance_clock)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    context = SimpleNamespace(client=object(), nonce=123)
    assert asyncio.run(ns["run_inference"](context)) == {"btc": 0.01}
    get_time.assert_awaited_once_with(context.client, 123)
    predict.assert_called_once_with(opened.replace(second=0))
    sleep.assert_awaited_once_with(10)
    assert now[0] == opened + timedelta(seconds=12)

    # Old or slow rounds still attempt inference; submission validity belongs to the chain.
    now[0] = opened + timedelta(seconds=26)
    predict.reset_mock()
    assert asyncio.run(ns["run_inference"](context)) == {"btc": 0.01}
    predict.assert_called_once_with(opened.replace(second=0))
    assert sleep.await_count == 2

    def slow_prediction(T):
        now[0] = opened + timedelta(seconds=60)
        return {"btc": 0.01}
    predict.side_effect = slow_prediction
    assert asyncio.run(ns["run_inference"](context)) == {"btc": 0.01}
    assert sleep.await_count == 3


def test_worker_submits_highest_volume_100_predictions_each_round(monkeypatch, tmp_path):
    import asyncio
    import os
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import allora_sdk
    from allora_sdk.rpc_client.tx_manager import TxError

    monkeypatch.chdir(tmp_path)
    (tmp_path / "worker_keys").mkdir()
    (tmp_path / "worker_keys/topic_90.key").write_text("unused by fake worker")
    attempted = []
    closed = []
    def inferer(**kwargs):
        async def run(timeout=None):
            try:
                attempted.append(await kwargs["run"](SimpleNamespace(nonce=1)))
                yield TxError("emissions", 1, "too many labels")
                attempted.append(await kwargs["run"](SimpleNamespace(nonce=2)))
                yield SimpleNamespace(submission=attempted[-1], tx_result=SimpleNamespace(txhash="test"))
            finally:
                closed.append(True)
        return SimpleNamespace(run=run, address="test-wallet")
    monkeypatch.setattr(allora_sdk.AlloraWorker, "inferer", inferer)
    path = Path(__file__).parents[1] / "notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_worker")
    ns = dict(Path=Path, os=os, TOPIC_ID=90, api_key="test", live_buffer=Mock(), volume_ranked_labels=["unavailable"] + [f"asset{i:03}" for i in range(178)],
              run_inference=AsyncMock(return_value={f"asset{i:03}": i / 1000 for i in reversed(range(178))}))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    asyncio.run(ns["run_worker"](max_attempts=2))
    assert [len(values) for values in attempted] == [100, 100]
    expected = {label: ns["run_inference"].return_value[label] for label in ns["volume_ranked_labels"][1:101]}
    assert attempted == [expected, expected]
    assert closed == [True]
    ns["live_buffer"].stop.assert_called_once()


@pytest.mark.parametrize("days", [16, 30])
def test_volume_ranking_uses_available_daily_bars(days):
    from types import SimpleNamespace
    path = Path(__file__).parents[1] / "notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "rank_hl_volume")
    now = datetime.now(timezone.utc)
    rows = lambda close: [{"timestamp": (now-timedelta(days=i)).isoformat(), "volume": 10, "close": close} for i in range(days)]
    response = Mock()
    response.json.return_value = {"datasets": [{"rows": rows(1)}, {"rows": rows(100)}]}
    request = Mock(return_value=response)
    ns = dict(pd=pd, np=np, datetime=datetime, timezone=timezone, timedelta=timedelta, requests=SimpleNamespace(get=request))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    result = ns["rank_hl_volume"](SimpleNamespace(base_url="atlas", headers={}), ["hl_low_1min", "hl_high_1min"])
    assert result.label.tolist() == ["high", "low"]
    assert result.volume_usd.tolist() == [days * 1000, days * 10]
    assert result.daily_bars.tolist() == [days, days]
    assert request.call_args.kwargs["params"]["limit"] == 30
    request.assert_called_once()


def test_partial_detection_is_exact_per_asset_and_recomputed_each_refresh():
    T=datetime(2026,10,6,12,1,tzinfo=timezone.utc)
    path=Path(__file__).parents[1]/'notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='predict_live')
    def candle(symbol,when,close):
        return dict(symbol=symbol,open_time=when,open=100.,high=max(100.,close),low=min(100.,close),close=close,volume=1.)
    history=[candle(symbol,T-timedelta(minutes=i),100.) for symbol in ['hl_btc_1min','hl_eth_1min','hl_sol_1min'] for i in range(1,7)]
    # A newer row must not stand in for a missing exact-T partial.
    future=[candle('hl_eth_1min',T+timedelta(minutes=1),900.)]
    first=history+future+[candle('hl_btc_1min',T,101.),candle('hl_sol_1min',T,98.)]
    second=history+future+[candle('hl_eth_1min',T,103.),candle('hl_sol_1min',T,100.)]
    buffer=Mock();buffer.snapshot.side_effect=[pd.DataFrame(rows).set_index(['symbol','open_time']).sort_index() for rows in [first,second]]
    model=Mock();model.predict.side_effect=lambda frame:np.full(len(frame),.01)
    cols=[f'feature_{f}_{i}' for i in range(2) for f in ['open','high','low','close','volume']]
    ns=dict(time=time,datetime=datetime,timezone=timezone,timedelta=timedelta,np=np,pd=pd,pl=pl,
            live_buffer=buffer,model=model,workflow=object.__new__(AlloraMLWorkflow),prediction_scale=1.0, LOOKBACK=2,feature_columns=cols)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    assert ns['predict_live'](T)==pytest.approx({'btc':.01+np.log(1.01),'eth':.01,'sol':.01+np.log(.98)})
    assert ns['predict_live'](T)==pytest.approx({'btc':.01,'eth':.01+np.log(1.03),'sol':.01})
    pd.testing.assert_frame_equal(model.predict.call_args_list[0].args[0],model.predict.call_args_list[1].args[0])
    assert np.allclose(model.predict.call_args.args[0],1.)


def test_training_wasserstein_scale_matches_known_magnitude():
    path=Path(__file__).parents[1]/'notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='fit_positive_wasserstein_scale')
    ns=dict(np=np)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    pred=np.array([-.02,-.01,.01,.02])
    scale,before,after=ns['fit_positive_wasserstein_scale'](pred,pred*5)
    assert scale==pytest.approx(5)
    assert after==pytest.approx(0) and before>0
    assert ns['fit_positive_wasserstein_scale'](np.zeros(4),pred)[0]==1


# Managed deployment artifacts

def test_packaged_sync_artifact_initializes_and_runs_in_existing_runtime(monkeypatch):
    path=Path(__file__).parents[1]/'notebooks/testnet/topic_90_hyperliquid_3min_logreturn/deploy_managed_example.py'
    spec=importlib.util.spec_from_file_location('managed_example',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    import allora_forge_builder_kit as kit
    import requests,time
    monkeypatch.setenv('ALLORA_API_KEY','test-runtime-key')
    class FakeAtlas:
        def __init__(self,**kwargs):assert kwargs['api_key']=='test-runtime-key'
        def discover_hl_universe(self):return ['hl_btc_1min']
    monkeypatch.setattr(kit,'AtlasDataManager',FakeAtlas)
    source='''
class LiveMinuteBuffer:
    def __init__(self,atlas,symbols,lookback): self.symbols=symbols
    def start(self): return {'ready':True}
    def stop(self): pass

def predict_live(T):
    return {str(i): prediction_scale*i for i in range(120)}
'''
    bundle=dict(model=None,metadata=dict(training_universe=['hl_btc_1min'],lookback=3,feature_columns=[]),calibration=dict(scale=2))
    artifact=mod.make_artifact(bundle,[str(i) for i in reversed(range(120))],source,'https://example.test')
    payload=cloudpickle.dumps(artifact)
    assert b'test-runtime-key' not in payload
    loaded=cloudpickle.loads(payload)
    assert loaded._buffer.symbols==['hl_btc_1min']
    sleeps=[];monkeypatch.setattr(time,'sleep',sleeps.append)
    response=SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'result':{'block':{'header':{'height':'123','time':'2026-10-08T12:01:27Z'}}}})
    monkeypatch.setattr(requests,'get',lambda *a,**kw:response)
    async def inside_sdk_loop():return _ArtifactCaller(loaded)(SimpleNamespace(nonce=123))
    result=asyncio.run(inside_sdk_loop())
    assert len(result)==100 and list(result)[0]=='119' and result['119']==238
    assert sleeps==[10]


def test_deployer_uses_manifest_and_reuses_addresses(tmp_path,monkeypatch):
    import sys,json
    import allora_forge_builder_kit as kit
    path=Path(__file__).parents[1]/'notebooks/testnet/topic_90_hyperliquid_3min_logreturn/deploy_managed_example.py'
    spec=importlib.util.spec_from_file_location('managed_example',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rows=[dict(directory=f'model_{i}') for i in range(1,4)]
    (tmp_path/'top_models.json').write_text(json.dumps(rows))
    for row in rows:(tmp_path/row['directory']).mkdir()
    monkeypatch.setattr(mod,'package',lambda directory,output,rpc:output)
    calls=[];starts=[]
    class Manager:
        def __init__(self,**kwargs):pass
        def deploy_worker(self,**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(address_assigned=kwargs['address'] or f'address{len(calls)}')
        def start_worker(self,*args):starts.append(args)
        def status_worker(self,*args):return dict(status='running')
    monkeypatch.setattr(kit,'WorkerManager',Manager)
    monkeypatch.setattr(sys,'argv',['deployer','--model-run',str(tmp_path),'--deploy'])
    mod.main();mod.main()
    assert len(starts)==6
    assert [c['address'] for c in calls[:3]]==[None]*3
    assert [c['address'] for c in calls[3:]]==['address1','address2','address3']
    assert all(c['replace'] for c in calls[3:])


# Model search and checkpoint selection

def test_checkpoint_search_fits_once_per_candidate_and_saves_prefixes(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / 'notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py'
    function = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'search_tree_checkpoints')
    ns = dict(Path=Path, lgb=lgb, np=np, gc=gc, json=json)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), ns)
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(240, 4)).astype('float32'), columns=list('abcd'))
    y = 2 * X.a - X.b + rng.normal(scale=.1, size=len(X))
    candidates = [dict(objective="regression", learning_rate=rate, num_leaves=7, min_child_samples=5, n_jobs=1, verbosity=-1, random_state=42) for rate in [.01, .1]]
    fits = []
    original_fit = lgb.LGBMRegressor.fit
    def tracked_fit(self, *args, **kwargs):
        fits.append(self.n_estimators)
        return original_fit(self, *args, **kwargs)
    monkeypatch.setattr(lgb.LGBMRegressor, 'fit', tracked_fit)
    best = ns['search_tree_checkpoints'](X.iloc[:180], y.iloc[:180], X.iloc[180:], y.iloc[180:], candidates, [3, 7], tmp_path)
    assert fits == [7, 7]
    records = json.loads((tmp_path / 'checkpoints.json').read_text())
    assert len(records) == 4
    for record in records:
        saved = lgb.Booster(model_file=record['checkpoint'])
        assert saved.current_iteration() == record['n_estimators']
        predictions = saved.predict(X.iloc[180:], num_threads=1)
        assert np.corrcoef(y.iloc[180:], predictions)[0, 1] == pytest.approx(record['correlation'])
        assert np.sqrt(np.mean((y.iloc[180:] - predictions)**2)) == pytest.approx(record['rmse'])
    assert best['validation_loss'] == min(r['validation_loss'] for r in records)
    assert best['validation_loss'] == best['mse']
    assert best['mse'] == pytest.approx(best['rmse']**2)
    assert json.loads((tmp_path / 'best_validation.json').read_text()) == best


def test_top_models_take_one_checkpoint_per_trial():
    path=Path(__file__).parents[1]/'notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='select_top_trials')
    ns=dict(np=np)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    rows=[dict(trial=t,n_estimators=n,mse=loss) for t,n,loss in
          [(1,10,.1),(1,20,.09),(1,30,.08),(2,10,.2),(3,10,.3),(4,10,.4)]]
    selected=ns['select_top_trials'](rows,3)
    assert [r['trial'] for r in selected]==[1,2,3]
    assert selected[0]['n_estimators']==30


# Detached worker lifecycle
@pytest.mark.skipif(__import__('os').name != 'posix', reason='POSIX session lifecycle')
def test_worker_survives_launcher_session_hangup(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    worker_script = tmp_path / 'worker.py'
    worker_script.write_text('import time\ntime.sleep(60)\n')
    launcher = tmp_path / 'launch.py'
    launcher.write_text('''
import os,sys,time
from pathlib import Path
from allora_forge_builder_kit.worker_manager import WorkerManager
root=Path(sys.argv[1])
m=WorkerManager(db_path=root/'state.db',secrets_path=root/'secrets.json',
                topic_desc_resolver=lambda _:None,reconcile_on_start=False,
                runtime_log_dir=root/'logs')
m.status_worker=lambda *args: {'last_pid':None}
m._build_run_command=lambda *args: ([sys.executable,str(root/'worker.py')],os.environ.copy())
m.start_worker(90,'test-address')
(root/'pid').write_text(str(m._runners[(90,'test-address')]['proc'].pid))
time.sleep(60)
''')
    parent=subprocess.Popen([sys.executable,str(launcher),str(tmp_path)],start_new_session=True)
    child=None
    try:
        deadline=time.monotonic()+10
        while not (tmp_path/'pid').exists() and time.monotonic()<deadline:
            if parent.poll() is not None:pytest.fail('Launcher exited before starting worker')
            time.sleep(.05)
        child=int((tmp_path/'pid').read_text())
        assert os.getsid(child)==child
        assert os.getpgid(child)!=os.getpgid(parent.pid)
        os.killpg(parent.pid,signal.SIGHUP)
        parent.wait(timeout=5)
        time.sleep(.1)
        os.kill(child,0)  # Worker remains alive after its launch session is gone.
    finally:
        if child is not None:
            try:os.kill(child,signal.SIGTERM)
            except ProcessLookupError:pass
        if parent.poll() is None:parent.kill()
        parent.wait(timeout=5)


# Review regressions: credentials, callback failures, and real managed-buffer errors.

@pytest.mark.parametrize('env,file_value,expected', [
    (' env-key ', 'file-key', 'env-key'),
    ('env-key', None, 'env-key'),
    (None, ' file-key\n', 'file-key'),
    (' \t', 'file-key', 'file-key'),
    (None, None, None),
    ('', '', None),
    (' ', ' \n', None),
])
def test_example_api_key_resolution(tmp_path, monkeypatch, env, file_value, expected):
    import os
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('ALLORA_API_KEY', raising=False)
    if env is not None:
        monkeypatch.setenv('ALLORA_API_KEY', env)
    if file_value is not None:
        (tmp_path / '.allora_api_key').write_text(file_value)
    path = Path(__file__).parents[1] / 'notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py'
    source = path.read_text()
    # Execute only credential resolution, stopping before any data/API work.
    snippet = source[source.index('api_key = '):source.index('base_dir = ')]
    namespace = dict(os=os, Path=Path)
    if expected is None:
        with pytest.raises(ValueError, match='Set ALLORA_API_KEY'):
            exec(compile(snippet, str(path), 'exec'), namespace)
    else:
        exec(compile(snippet, str(path), 'exec'), namespace)
        assert namespace['api_key'] == expected


@pytest.mark.parametrize('submitted,passed', [(None, False), (90, False), (91, True)])
def test_log_return_participation_counts(submitted, passed):
    y = np.random.default_rng(20).normal(size=70)
    options = {} if submitted is None else {'n_submitted': submitted}
    report = PerformanceEvaluator().evaluate(y, .8*y, n_expected_epochs=100, **options)
    assert report['metrics']['participation'] == (70 if submitted is None else submitted) / 100
    assert report['passed']['participation'] is passed
    assert report['temporal_coverage_pass'] is passed
    assert report['eligible'] is passed
    assert report['num_passed'] == (7 if passed else 6)


@pytest.mark.parametrize('options', [
    {'n_expected_epochs': 0}, {'n_expected_epochs': 69},
    {'n_expected_epochs': 100, 'n_submitted': 101}, {'n_submitted': 70},
])
def test_log_return_invalid_participation_counts(options):
    y = np.linspace(-1, 1, 70)
    with pytest.raises(ValueError):
        PerformanceEvaluator().evaluate(y, y, **options)


def test_standalone_max_attempts_counts_callback_errors(tmp_path, monkeypatch):
    import os
    import allora_sdk
    monkeypatch.chdir(tmp_path)
    key = tmp_path / 'worker_keys/topic_90.key'
    key.parent.mkdir()
    key.write_text('test-only-existing-key')
    monkeypatch.setattr(allora_sdk, 'AlloraWalletConfig', lambda **kw: None)
    yielded = []
    class Worker:
        address = 'test-worker'
        async def run(self, timeout=None):
            for i in range(4):
                yielded.append(i)
                yield RuntimeError('synthetic callback failure')
    monkeypatch.setattr(allora_sdk, 'AlloraWorker', SimpleNamespace(inferer=lambda **kw: Worker()))
    path = Path(__file__).parents[1] / 'notebooks/testnet/topic_90_hyperliquid_3min_logreturn/example.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run_worker')
    buffer = Mock()
    namespace = dict(Path=Path, os=os, TOPIC_ID=90, api_key='test', live_buffer=buffer)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    asyncio.run(namespace['run_worker'](max_attempts=2))
    assert yielded == [0, 1]
    buffer.stop.assert_called_once()


@pytest.mark.parametrize('failure', ['timeout', 403, 404])
def test_managed_artifact_real_buffer_handles_atlas_errors(monkeypatch, failure):
    import allora_forge_builder_kit as kit
    base = Path(__file__).parents[1] / 'notebooks/testnet/topic_90_hyperliquid_3min_logreturn'
    spec = importlib.util.spec_from_file_location('managed_failure_example', base / 'deploy_managed_example.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('ALLORA_API_KEY', 'test-only-key')
    calls = []
    class Atlas:
        def __init__(self, **kw):
            pass
        def discover_hl_universe(self):
            return ['hl_btc_1min', 'hl_eth_1min']
        def get_bulk_1min_candles(self, symbols, **kw):
            calls.append(list(symbols))
            if failure == 'timeout':
                raise requests.Timeout('synthetic timeout')
            if 'hl_btc_1min' in symbols:
                response = requests.Response()
                response.status_code = failure
                raise requests.HTTPError('unavailable asset', response=response)
            return pd.DataFrame()
    monkeypatch.setattr(kit, 'AtlasDataManager', Atlas)
    node = next(n for n in ast.parse((base / 'example.py').read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'LiveMinuteBuffer')
    source = ast.unparse(node) + '\n\ndef predict_live(T):\n    return {}\n'
    bundle = dict(model=None, metadata=dict(training_universe=['hl_btc_1min', 'hl_eth_1min'],
                                          lookback=2, feature_columns=[]))
    payload = cloudpickle.dumps(module.make_artifact(bundle, [], source, 'https://example.test'))
    if failure == 'timeout':
        with pytest.raises(RuntimeError, match='Initial Atlas refresh failed'):
            cloudpickle.loads(payload)
    else:
        loaded = cloudpickle.loads(payload)
        try:
            assert 'hl_btc_1min' in loaded._buffer.last_errors
            assert 'hl_eth_1min' in loaded._buffer.last_successful_refresh
            assert ['hl_eth_1min'] in calls
        finally:
            loaded._buffer.stop()
