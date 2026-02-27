from pathlib import Path

import pytest

from allora_forge_builder_kit.worker_manager import WorkerManager, WorkerSpec


def _new_manager(tmp_path: Path) -> WorkerManager:
    counter = {"i": 0}

    def create_identity() -> tuple[str, str, str]:
        counter["i"] += 1
        i = counter["i"]
        return (f"identity_{i:03d}", f"addr_{i:03d}", f"mnemonic-{i}")

    return WorkerManager(
        db_path=tmp_path / "state.db",
        secrets_path=tmp_path / "secrets.json",
        identity_creator=create_identity,
    )


def test_add_worker_enforces_unique_topic_address(tmp_path: Path):
    manager = _new_manager(tmp_path)
    spec = WorkerSpec(
        topic_id=69,
        topic_desc="BTC",
        address="addr_001",
        artifact_path=tmp_path / "predict.pkl",
        identity_ref="identity_001",
    )
    spec.artifact_path.write_text("x")

    manager.add_worker(spec)

    with pytest.raises(ValueError):
        manager.add_worker(spec)


def test_deploy_without_address_reuses_existing_free_identity(tmp_path: Path):
    manager = _new_manager(tmp_path)

    # Seed two identities
    a = manager.ensure_identity(alias="alpha")
    b = manager.ensure_identity(alias="beta")

    (tmp_path / "a.pkl").write_text("a")
    (tmp_path / "b.pkl").write_text("b")

    # Topic 1 takes alpha
    manager.deploy_worker(topic_id=1, artifact_path=tmp_path / "a.pkl", address=a.address)

    # Topic 2 with no address should reuse alpha (first free for topic 2)
    result = manager.deploy_worker(topic_id=2, artifact_path=tmp_path / "b.pkl")

    assert result.action in {"created", "reused"}
    assert result.address_assigned == a.address
    assert b.address != result.address_assigned


def test_deploy_without_address_creates_new_when_all_used_for_topic(tmp_path: Path):
    manager = _new_manager(tmp_path)

    a = manager.ensure_identity(alias="alpha")
    b = manager.ensure_identity(alias="beta")

    (tmp_path / "a.pkl").write_text("a")
    (tmp_path / "b.pkl").write_text("b")
    (tmp_path / "c.pkl").write_text("c")

    manager.deploy_worker(topic_id=7, artifact_path=tmp_path / "a.pkl", address=a.address)
    manager.deploy_worker(topic_id=7, artifact_path=tmp_path / "b.pkl", address=b.address)

    result = manager.deploy_worker(topic_id=7, artifact_path=tmp_path / "c.pkl")

    assert result.action == "created"
    assert result.address_assigned not in {a.address, b.address}


def test_replace_existing_worker_updates_artifact(tmp_path: Path):
    manager = _new_manager(tmp_path)
    ident = manager.ensure_identity(alias="alpha")

    old_artifact = tmp_path / "old.pkl"
    new_artifact = tmp_path / "new.pkl"
    old_artifact.write_text("old")
    new_artifact.write_text("new")

    manager.deploy_worker(topic_id=9, artifact_path=old_artifact, address=ident.address)
    result = manager.deploy_worker(
        topic_id=9,
        artifact_path=new_artifact,
        address=ident.address,
        replace=True,
    )

    assert result.action == "replaced"
    status = manager.status_worker(topic_id=9, address=ident.address)
    assert "managed_artifacts" in status["artifact_path"]
    assert status["artifact_path"].endswith(".pkl")


def test_conflict_without_replace_auto_assigns_alternate_address(tmp_path: Path):
    manager = _new_manager(tmp_path)
    ident = manager.ensure_identity(alias="alpha")

    p1 = tmp_path / "1.pkl"
    p2 = tmp_path / "2.pkl"
    p1.write_text("1")
    p2.write_text("2")

    manager.deploy_worker(topic_id=11, artifact_path=p1, address=ident.address)
    result = manager.deploy_worker(topic_id=11, artifact_path=p2, address=ident.address, replace=False)

    assert result.action == "created"
    assert result.address_assigned != ident.address


def test_attach_monitor_bootstraps_existing_workers(tmp_path: Path):
    manager = _new_manager(tmp_path)
    ident = manager.ensure_identity(alias="alpha")
    p1 = tmp_path / "1.pkl"
    p1.write_text("1")
    manager.deploy_worker(topic_id=69, artifact_path=p1, address=ident.address)

    class DummyMonitor:
        def __init__(self):
            self.targets = []
            self.synced = 0

        def register_target(self, topic_id, address, deployed_at=None, deployment_id=None):
            self.targets.append((topic_id, address, deployed_at, deployment_id))

        def sync_once(self):
            self.synced += 1

        def backfill_target(self, topic_id, address, since=None):
            self.synced += 1

    mon = DummyMonitor()
    out = manager.attach_monitor(mon, sync_now=True)

    assert out["registered"] == 1
    assert mon.targets[0][0] == 69
    assert mon.targets[0][1] == ident.address
    assert mon.synced >= 1
