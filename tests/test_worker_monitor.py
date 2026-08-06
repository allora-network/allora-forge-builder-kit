from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from allora_forge_builder_kit.worker_monitor import WorkerMonitor, AlloraSDKEventFetcher


def test_register_and_list_targets(tmp_path: Path):
    monitor = WorkerMonitor(db_path=tmp_path / "state.db", event_fetcher=lambda *_: [])
    monitor.register_target(topic_id=77, address="allo1abc", deployed_at="2026-02-27T00:00:00Z")

    targets = monitor.list_targets()
    assert len(targets) == 1
    assert targets[0].topic_id == 77
    assert targets[0].address == "allo1abc"


def test_sync_inserts_events_and_is_idempotent(tmp_path: Path):
    calls = {"n": 0}

    def fetcher(topic_id, address, since):
        calls["n"] += 1
        return [
            {
                "event_id": "e1",
                "event_type": "submission",
                "status": "success",
                "tx_hash": "tx1",
                "observed_at": "2026-02-27T10:00:00Z",
            },
            {
                "event_id": "e2",
                "event_type": "inference",
                "value_num": 65500.12,
                "value_text": "65500.12",
                "tx_hash": "tx1",
                "observed_at": "2026-02-27T10:00:00Z",
            },
            {
                "event_id": "e3",
                "event_type": "reward",
                "value_num": 1.25,
                "value_text": "1.25 uallo",
                "tx_hash": "tx2",
                "observed_at": "2026-02-27T10:05:00Z",
            },
        ]

    monitor = WorkerMonitor(db_path=tmp_path / "state.db", event_fetcher=fetcher)
    monitor.register_target(topic_id=77, address="allo1abc", deployed_at="2026-02-27T00:00:00Z")

    out1 = monitor.sync_once()
    out2 = monitor.sync_once()

    assert out1["inserted"] == 3
    assert out2["inserted"] == 0
    assert calls["n"] == 2


def test_summary_rollups(tmp_path: Path):
    def fetcher(topic_id, address, since):
        return [
            {
                "event_id": "s-ok",
                "event_type": "submission",
                "status": "success",
                "tx_hash": "txs",
                "observed_at": "2026-02-27T10:00:00Z",
            },
            {
                "event_id": "s-err",
                "event_type": "submission",
                "status": "error",
                "tx_hash": "txe",
                "observed_at": "2026-02-27T10:01:00Z",
            },
            {
                "event_id": "inf",
                "event_type": "inference",
                "value_num": 0.001,
                "value_text": "0.001",
                "tx_hash": "txs",
                "observed_at": "2026-02-27T10:00:00Z",
            },
            {
                "event_id": "rew",
                "event_type": "reward",
                "value_num": 2.5,
                "value_text": "2.5 uallo",
                "tx_hash": "txr",
                "observed_at": "2026-02-27T10:02:00Z",
            },
        ]

    monitor = WorkerMonitor(db_path=tmp_path / "state.db", event_fetcher=fetcher)
    monitor.register_target(topic_id=69, address="allo1xyz", deployed_at="2026-02-27T00:00:00Z")
    monitor.sync_once()

    summary = monitor.get_summary(topic_id=69, address="allo1xyz")
    assert summary["events_total"] == 4
    assert summary["submission_success"] == 1
    assert summary["submission_error"] == 1
    assert summary["inference_count"] == 1
    assert summary["reward_count"] == 1
    assert summary["rewards_total"] == 2.5
    assert summary["last_inference"]["value_text"] == "0.001"


def test_sync_accepts_async_fetcher(tmp_path: Path):
    """WorkerMonitor.sync_once must transparently handle an async event fetcher."""
    import inspect

    async def async_fetcher(topic_id, address, since):
        return [
            {
                "event_id": "af1",
                "event_type": "submission",
                "status": "success",
                "observed_at": "2026-07-22T00:00:00Z",
            }
        ]

    assert inspect.iscoroutinefunction(async_fetcher)

    monitor = WorkerMonitor(db_path=tmp_path / "state.db", event_fetcher=async_fetcher)
    monitor.register_target(topic_id=42, address="allo1test", deployed_at="2026-07-22T00:00:00Z")

    result = monitor.sync_once()
    assert result["inserted"] == 1


def test_allora_sdk_event_fetcher_call_is_async():
    """AlloraSDKEventFetcher.__call__ must be a coroutine function so WorkerMonitor can await it."""
    import inspect
    assert inspect.iscoroutinefunction(AlloraSDKEventFetcher.__call__)


def _make_mock_client(_network):
    """Return a mock AlloraRPCClient with async stubs for all queried methods."""
    mock = MagicMock()
    mock.close = AsyncMock()
    mock.tx.query.get_txs_event = AsyncMock(return_value=MagicMock(tx_responses=[]))
    mock.emissions.query.get_worker_latest_input_inference_by_topic_id = AsyncMock(
        return_value=MagicMock(latest_inference=None)
    )
    mock.emissions.query.get_inferer_score_ema = AsyncMock(side_effect=Exception("no score"))
    mock.emissions.query.get_previous_inference_reward_fraction = AsyncMock(side_effect=Exception("no reward"))
    mock.emissions.query.is_whitelisted_topic_worker = AsyncMock(side_effect=Exception())
    mock.emissions.query.can_submit_worker_payload = AsyncMock(side_effect=Exception())
    return mock


def test_event_fetcher_creates_fresh_client_per_asyncio_run():
    """FIND-001: AlloraSDKEventFetcher must create a fresh AlloraRPCClient on each __call__
    invocation. A shared client (created in __init__) captures grpclib's self._loop outside
    any asyncio.run() context. Every subsequent asyncio.run() runs a different loop, so
    _create_connection registers I/O on the wrong loop and raises
    'Future attached to a different loop' — silently swallowed, blacking out all monitoring.
    """
    import asyncio

    clients_created: list = []

    def tracking_make_client(network):
        client = _make_mock_client(network)
        clients_created.append(client)
        return client

    with patch("allora_sdk.rpc_client.client.AlloraRPCClient", side_effect=tracking_make_client):
        fetcher = AlloraSDKEventFetcher()
        asyncio.run(fetcher(topic_id=1, address="allo1test", since=None))
        asyncio.run(fetcher(topic_id=1, address="allo1test", since=None))

    assert len(clients_created) >= 2, (
        f"Expected a fresh AlloraRPCClient per asyncio.run() call, got {len(clients_created)}. "
        "A shared client causes 'Future attached to a different loop' RuntimeError, silently "
        "swallowed by the bare except blocks, permanently blacking out monitoring."
    )
