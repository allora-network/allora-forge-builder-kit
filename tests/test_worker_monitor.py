from pathlib import Path

from allora_forge_builder_kit.worker_monitor import WorkerMonitor


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
