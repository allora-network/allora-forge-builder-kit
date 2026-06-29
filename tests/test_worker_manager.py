from pathlib import Path
from types import SimpleNamespace
import sqlite3

import pytest

from allora_forge_builder_kit.worker_manager import ForgeClientProtocol, WorkerManager, WorkerSpec


class _FakeForgeClient:
    """Stand-in for allora_sdk's ForgeBackendClient. Idempotent get-or-create by topic:
    the same topic always yields the same address/id, mirroring the backend's one-wallet-
    per-(user, topic) contract."""

    def __init__(self):
        self.provisioned: list[tuple[int, str | None]] = []
        self.cleared: list[str] = []

    def provision_wallet(self, topic_id: int, label: str | None = None):
        self.provisioned.append((topic_id, label))
        return SimpleNamespace(
            id=f"wallet-{topic_id}",
            address=f"allo1managed{topic_id:04d}",
            pubkey="ab" * 33,
        )

    def clear_association(self, wallet_id: str) -> None:
        self.cleared.append(wallet_id)


def _managed_manager(tmp_path: Path, client: ForgeClientProtocol, **kwargs) -> WorkerManager:
    return WorkerManager(
        db_path=tmp_path / "state.db",
        secrets_path=tmp_path / "secrets.json",
        identity_creator=lambda: ("unused", "unused", "unused"),
        forge_client=client,
        reconcile_on_start=False,
        **kwargs,
    )


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


def test_redeploy_advances_monitor_target_deployment_without_attached_monitor(tmp_path: Path):
    manager = _new_manager(tmp_path)
    ident = manager.ensure_identity(alias="alpha")

    p1 = tmp_path / "v1.pkl"
    p2 = tmp_path / "v2.pkl"
    p1.write_text("v1")
    p2.write_text("v2")

    manager.deploy_worker(topic_id=69, artifact_path=p1, address=ident.address)

    with sqlite3.connect(tmp_path / "state.db") as conn:
        dep1 = conn.execute(
            "SELECT deployment_id FROM monitor_targets WHERE topic_id=? AND address=?",
            (69, ident.address),
        ).fetchone()[0]

    manager.deploy_worker(topic_id=69, artifact_path=p2, address=ident.address, replace=True, mode="strict")

    with sqlite3.connect(tmp_path / "state.db") as conn:
        dep2 = conn.execute(
            "SELECT deployment_id FROM monitor_targets WHERE topic_id=? AND address=?",
            (69, ident.address),
        ).fetchone()[0]

    assert dep1
    assert dep2
    assert dep1 != dep2


def test_status_all_with_logs_returns_tail_and_artifact_path(tmp_path: Path):
    manager = _new_manager(tmp_path)
    ident = manager.ensure_identity(alias="alpha")
    p1 = tmp_path / "1.pkl"
    p1.write_text("1")
    manager.deploy_worker(topic_id=77, artifact_path=p1, address=ident.address)

    log_path = manager.runtime_log_dir / f"worker_77_{ident.address}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("l1\nl2\nl3\nl4\n")

    rows = manager.status_all_with_logs(tail_lines=2)
    row = [r for r in rows if r["topic_id"] == 77 and r["address"] == ident.address][0]

    assert row["artifact_path"].endswith(".pkl")
    assert row["log_tail"] == ["l3", "l4"]


# ----------------------------
# Managed custody (ENGN-8646)
# ----------------------------
def test_deploy_managed_provisions_and_registers(tmp_path: Path):
    client = _FakeForgeClient()
    manager = _managed_manager(tmp_path, client)
    artifact = tmp_path / "m.pkl"
    artifact.write_text("m")

    result = manager.deploy_worker(topic_id=42, artifact_path=artifact, custody="managed")

    assert result.action == "created"
    assert result.address_assigned == "allo1managed0042"
    # Provisioned exactly once, get-or-create by topic with a display label.
    assert client.provisioned == [(42, "worker-topic-42")]

    status = manager.status_worker(topic_id=42, address="allo1managed0042")
    assert status["custody"] == "managed"
    assert status["signing_wallet_id"] == "wallet-42"
    assert status["artifact_path"].endswith(".pkl")
    # identity_ref is a sentinel for managed workers (no identities row); the canonical wallet id
    # lives in signing_wallet_id, not overloaded into identity_ref.
    assert status["identity_ref"] == "managed"


def test_deploy_managed_redeploy_different_artifact_with_replace_rotates(tmp_path: Path):
    """synth-009: a genuinely different artifact with replace=True rotates the deployment on the
    one-per-topic wallet (no second worker row) and reports 'replaced'."""
    client = _FakeForgeClient()
    manager = _managed_manager(tmp_path, client)
    v1 = tmp_path / "v1.pkl"
    v2 = tmp_path / "v2.pkl"
    v1.write_text("v1")
    v2.write_text("v2")

    manager.deploy_worker(topic_id=8, artifact_path=v1, custody="managed")
    dep1 = manager._get_active_deployment_id(8, "allo1managed0008")
    result = manager.deploy_worker(topic_id=8, artifact_path=v2, custody="managed", replace=True)

    assert result.action == "replaced"
    assert result.address_assigned == "allo1managed0008"
    # One worker per topic — no second worker row was created.
    assert len([w for w in manager.status_all() if w["topic_id"] == 8]) == 1
    # The artifact was rotated: a new active deployment supersedes the first.
    dep2 = manager._get_active_deployment_id(8, "allo1managed0008")
    assert dep1 and dep2 and dep1 != dep2


def test_deploy_managed_redeploy_identical_artifact_is_reused(tmp_path: Path):
    """synth-009: re-deploying byte-identical artifact (e.g. an idempotent CI re-run) is a no-op
    'reused' — the running deployment is left untouched rather than churned."""
    client = _FakeForgeClient()
    manager = _managed_manager(tmp_path, client)
    v1 = tmp_path / "v1.pkl"
    v1.write_text("same-bytes")

    first = manager.deploy_worker(topic_id=8, artifact_path=v1, custody="managed")
    assert first.action == "created"
    dep1 = manager._get_active_deployment_id(8, "allo1managed0008")
    artifact_before = manager.status_worker(8, "allo1managed0008")["artifact_path"]

    # Identical bytes (a fresh file with the same content) re-deployed without replace.
    v1_again = tmp_path / "v1_again.pkl"
    v1_again.write_text("same-bytes")
    result = manager.deploy_worker(topic_id=8, artifact_path=v1_again, custody="managed")

    assert result.action == "reused"
    assert result.address_assigned == "allo1managed0008"
    assert len([w for w in manager.status_all() if w["topic_id"] == 8]) == 1
    # No rotation: the active deployment and the on-disk managed artifact are unchanged.
    assert manager._get_active_deployment_id(8, "allo1managed0008") == dep1
    assert manager.status_worker(8, "allo1managed0008")["artifact_path"] == artifact_before


def test_deploy_managed_redeploy_identical_artifact_resyncs_metadata(tmp_path: Path):
    """The hash-identical 'reused' path must still re-sync the worker-row metadata (reject_zero +
    the freshly-provisioned signing_wallet_id) so an idempotent re-run cannot leave the row
    pointing at a stale flag or wallet binding — while leaving the artifact unrotated."""
    client = _FakeForgeClient()
    manager = _managed_manager(tmp_path, client)
    v1 = tmp_path / "v1.pkl"
    v1.write_text("same-bytes")

    manager.deploy_worker(topic_id=8, artifact_path=v1, custody="managed", reject_zero=False)
    dep1 = manager._get_active_deployment_id(8, "allo1managed0008")
    artifact_before = manager.status_worker(8, "allo1managed0008")["artifact_path"]

    # Drift the row: simulate a signing wallet id that has fallen behind the backend.
    with sqlite3.connect(tmp_path / "state.db") as conn:
        conn.execute(
            "UPDATE workers SET signing_wallet_id='stale-wallet' WHERE topic_id=? AND address=?",
            (8, "allo1managed0008"),
        )
        conn.commit()

    # Identical artifact, no replace, but flip reject_zero.
    v1_again = tmp_path / "v1_again.pkl"
    v1_again.write_text("same-bytes")
    result = manager.deploy_worker(topic_id=8, artifact_path=v1_again, custody="managed", reject_zero=True)

    assert result.action == "reused"
    row = manager.status_worker(8, "allo1managed0008")
    # Metadata re-synced from the fresh provision...
    assert row["reject_zero"] is True
    assert row["signing_wallet_id"] == "wallet-8"
    # ...but the artifact was NOT rotated.
    assert manager._get_active_deployment_id(8, "allo1managed0008") == dep1
    assert row["artifact_path"] == artifact_before


def test_deploy_managed_redeploy_identical_artifact_with_replace_rotates(tmp_path: Path):
    """An explicit replace=True is honored even when the artifact is byte-identical: the caller
    asked to rotate, so the deployment is rotated and reported 'replaced' rather than
    short-circuited to 'reused'."""
    client = _FakeForgeClient()
    manager = _managed_manager(tmp_path, client)
    v1 = tmp_path / "v1.pkl"
    v1.write_text("same-bytes")

    manager.deploy_worker(topic_id=8, artifact_path=v1, custody="managed")
    dep1 = manager._get_active_deployment_id(8, "allo1managed0008")

    v1_again = tmp_path / "v1_again.pkl"
    v1_again.write_text("same-bytes")
    result = manager.deploy_worker(topic_id=8, artifact_path=v1_again, custody="managed", replace=True)

    assert result.action == "replaced"
    assert len([w for w in manager.status_all() if w["topic_id"] == 8]) == 1
    # Explicit replace rotated the deployment even though the bytes were identical.
    dep2 = manager._get_active_deployment_id(8, "allo1managed0008")
    assert dep1 and dep2 and dep1 != dep2


def test_deploy_managed_redeploy_legacy_null_hash_without_replace_raises(tmp_path: Path):
    """A legacy active deployment with no recorded hash is treated conservatively as 'unknown':
    a redeploy without replace=True refuses rather than risk overwriting a different deployment,
    even when the bytes happen to match."""
    client = _FakeForgeClient()
    manager = _managed_manager(tmp_path, client)
    v1 = tmp_path / "v1.pkl"
    v1.write_text("legacy")
    manager.deploy_worker(topic_id=8, artifact_path=v1, custody="managed")

    # Simulate a pre-hash-tracking deployment row.
    with sqlite3.connect(tmp_path / "state.db") as conn:
        conn.execute(
            "UPDATE worker_deployments SET artifact_hash=NULL WHERE topic_id=? AND address=? AND is_active=1",
            (8, "allo1managed0008"),
        )
        conn.commit()

    with pytest.raises(ValueError, match="replace=True"):
        manager.deploy_worker(topic_id=8, artifact_path=v1, custody="managed")

    # With replace=True the unknown-hash row is rotated normally.
    result = manager.deploy_worker(topic_id=8, artifact_path=v1, custody="managed", replace=True)
    assert result.action == "replaced"


def test_managed_redeploy_syncs_reject_zero_into_db_and_command(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ALLORA_API_KEY", "test-allora-key")
    client = _FakeForgeClient()
    manager = _managed_manager(
        tmp_path,
        client,
        forge_api_key="forge_sk_test",
        forge_backend_url="http://localhost:8080",
    )
    v1 = tmp_path / "v1.pkl"
    v2 = tmp_path / "v2.pkl"
    v1.write_text("v1")
    v2.write_text("v2")

    manager.deploy_worker(topic_id=8, artifact_path=v1, custody="managed", reject_zero=False)
    manager.deploy_worker(topic_id=8, artifact_path=v2, custody="managed", replace=True, reject_zero=True)

    row = [w for w in manager.status_all() if w["topic_id"] == 8][0]
    assert row["reject_zero"] is True

    status = manager.status_worker(topic_id=8, address="allo1managed0008")
    cmd, _ = manager._build_run_command(8, "allo1managed0008", status)
    assert "--reject-zero" in cmd


def test_managed_redeploy_different_artifact_without_replace_raises(tmp_path: Path):
    """synth-009 core fix: a different artifact for an existing managed worker is NOT silently
    overwritten in auto mode — it raises unless replace=True is passed, leaving the running
    deployment intact."""
    client = _FakeForgeClient()
    manager = _managed_manager(tmp_path, client)
    v1 = tmp_path / "v1.pkl"
    v2 = tmp_path / "v2.pkl"
    v1.write_text("v1")
    v2.write_text("v2")

    manager.deploy_worker(topic_id=8, artifact_path=v1, custody="managed")
    dep1 = manager._get_active_deployment_id(8, "allo1managed0008")

    # Auto mode (no replace=True) with a different artifact must refuse rather than overwrite.
    with pytest.raises(ValueError, match="replace=True"):
        manager.deploy_worker(topic_id=8, artifact_path=v2, custody="managed")

    # The original deployment is untouched: same active deployment, still one worker row.
    assert manager._get_active_deployment_id(8, "allo1managed0008") == dep1
    assert len([w for w in manager.status_all() if w["topic_id"] == 8]) == 1


def test_build_run_command_managed_injects_forge_env_and_no_keyfile(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ALLORA_API_KEY", "test-allora-key")
    client = _FakeForgeClient()
    manager = _managed_manager(
        tmp_path,
        client,
        forge_api_key="forge_sk_test",
        forge_backend_url="http://localhost:8080",
    )
    artifact = tmp_path / "m.pkl"
    artifact.write_text("m")
    manager.deploy_worker(topic_id=7, artifact_path=artifact, custody="managed")

    status = manager.status_worker(topic_id=7, address="allo1managed0007")
    cmd, env = manager._build_run_command(7, "allo1managed0007", status)

    assert "--custody" in cmd
    assert cmd[cmd.index("--custody") + 1] == "managed"
    assert "--mnemonic-file" not in cmd
    assert env is not None
    assert env["FORGE_API_KEY"] == "forge_sk_test"
    assert env["FORGE_BACKEND_URL"] == "http://localhost:8080"
    # The DB-stored signing wallet id is pinned into the env so the worker signs with the exact
    # provisioned wallet (deterministic) rather than re-deriving via topic get-or-create.
    assert env["FORGE_SIGNING_WALLET_ID"] == "wallet-7"


def test_build_run_command_managed_without_signing_wallet_id_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ALLORA_API_KEY", "test-allora-key")
    client = _FakeForgeClient()
    manager = _managed_manager(
        tmp_path,
        client,
        forge_api_key="forge_sk_test",
        forge_backend_url="http://localhost:8080",
    )
    artifact = tmp_path / "m.pkl"
    artifact.write_text("m")
    manager.deploy_worker(topic_id=7, artifact_path=artifact, custody="managed")
    status = manager.status_worker(topic_id=7, address="allo1managed0007")
    status["signing_wallet_id"] = None

    # Fails loudly before spawning rather than letting the subprocess exit post-'running'.
    with pytest.raises(ValueError, match="signing_wallet_id"):
        manager._build_run_command(7, "allo1managed0007", status)


def test_allora_api_key_present_treats_empty_file_as_absent(tmp_path: Path, monkeypatch):
    """An existing-but-empty .allora_api_key must not satisfy the precheck: the subprocess would
    resolve it to an empty key and fail at runtime, defeating the fail-before-running guarantee."""
    monkeypatch.delenv("ALLORA_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    key_file = tmp_path / ".allora_api_key"

    key_file.write_text("   \n")
    assert WorkerManager._allora_api_key_present() is False

    key_file.write_text("real-key")
    assert WorkerManager._allora_api_key_present() is True


def test_start_worker_validation_failure_leaves_no_log_file(tmp_path: Path, monkeypatch):
    """A managed worker that fails the _build_run_command precheck must not create an empty log:
    the file open is deferred until after validation so a persistently-misconfigured worker does
    not re-touch a silent empty log on every reconcile."""
    monkeypatch.setenv("ALLORA_API_KEY", "test-allora-key")  # clear the api-key precheck
    monkeypatch.delenv("FORGE_API_KEY", raising=False)
    monkeypatch.delenv("FORGE_BACKEND_URL", raising=False)
    client = _FakeForgeClient()
    manager = _managed_manager(tmp_path, client)  # no Forge creds on the manager -> precheck fails
    artifact = tmp_path / "m.pkl"
    artifact.write_text("m")
    manager.deploy_worker(topic_id=4, artifact_path=artifact, custody="managed")

    log_path = manager.runtime_log_dir / "worker_4_allo1managed0004.log"
    with pytest.raises(ValueError):
        manager.start_worker(topic_id=4, address="allo1managed0004")
    assert not log_path.exists()


def test_remove_managed_worker_clears_association(tmp_path: Path):
    client = _FakeForgeClient()
    manager = _managed_manager(tmp_path, client)
    artifact = tmp_path / "m.pkl"
    artifact.write_text("m")
    manager.deploy_worker(topic_id=9, artifact_path=artifact, custody="managed")

    manager.remove_worker(topic_id=9, address="allo1managed0009")

    assert client.cleared == ["wallet-9"]
    with pytest.raises(KeyError):
        manager.status_worker(topic_id=9, address="allo1managed0009")


def test_remove_managed_worker_tolerates_clear_failure(tmp_path: Path):
    class _FailingClient(_FakeForgeClient):
        def clear_association(self, wallet_id: str) -> None:
            raise RuntimeError("backend down")

    client = _FailingClient()
    manager = _managed_manager(tmp_path, client)
    artifact = tmp_path / "m.pkl"
    artifact.write_text("m")
    manager.deploy_worker(topic_id=5, artifact_path=artifact, custody="managed")

    # Decommission cleanup must never raise — the worker is still removed locally.
    manager.remove_worker(topic_id=5, address="allo1managed0005")
    with pytest.raises(KeyError):
        manager.status_worker(topic_id=5, address="allo1managed0005")


def test_remove_local_worker_does_not_clear(tmp_path: Path):
    client = _FakeForgeClient()
    manager = _managed_manager(tmp_path, client)
    ident = manager.ensure_identity(alias="alpha")
    artifact = tmp_path / "l.pkl"
    artifact.write_text("l")
    manager.deploy_worker(topic_id=3, artifact_path=artifact, address=ident.address)

    manager.remove_worker(topic_id=3, address=ident.address)

    assert client.cleared == []


def test_release_managed_binding_bounds_slow_backend(tmp_path: Path):
    """A degraded backend must not stall teardown: _release_managed_binding returns within its
    timeout even when clear_association blocks, and never raises."""
    import threading
    import time

    entered = threading.Event()
    release = threading.Event()

    class _HangingClient(_FakeForgeClient):
        def clear_association(self, wallet_id: str) -> None:
            entered.set()
            release.wait(30)  # would block the caller for the full SDK timeout without the bound

    manager = _managed_manager(tmp_path, _HangingClient())

    start = time.monotonic()
    manager._release_managed_binding("wallet-x", topic_id=1, timeout=0.5)
    elapsed = time.monotonic() - start
    release.set()  # let the daemon thread unwind

    assert entered.wait(1.0)  # the clear actually started on the background thread
    assert elapsed < 5.0  # returned at ~0.5s, not blocked on the 30s backend call


def test_deploy_managed_rejects_malformed_backend_wallet(tmp_path: Path):
    class _BadClient(_FakeForgeClient):
        def provision_wallet(self, topic_id: int, label: str | None = None):
            return SimpleNamespace(id="", address="", pubkey="")

    client = _BadClient()
    manager = _managed_manager(tmp_path, client)
    artifact = tmp_path / "m.pkl"
    artifact.write_text("m")

    with pytest.raises(RuntimeError, match="malformed wallet"):
        manager.deploy_worker(topic_id=1, artifact_path=artifact, custody="managed")


def test_deploy_managed_rejects_local_only_inputs(tmp_path: Path):
    client = _FakeForgeClient()
    manager = _managed_manager(tmp_path, client)
    artifact = tmp_path / "m.pkl"
    artifact.write_text("m")

    for kwargs in ({"address": "allo1xxx"}, {"mnemonic": "abandon abandon"}, {"identity_alias": "alias"}):
        with pytest.raises(ValueError, match="local-custody inputs"):
            manager.deploy_worker(topic_id=1, artifact_path=artifact, custody="managed", **kwargs)


def test_deploy_managed_requires_forge_credentials(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FORGE_API_KEY", raising=False)
    monkeypatch.delenv("FORGE_BACKEND_URL", raising=False)
    manager = WorkerManager(
        db_path=tmp_path / "state.db",
        secrets_path=tmp_path / "secrets.json",
        identity_creator=lambda: ("unused", "unused", "unused"),
        reconcile_on_start=False,
    )
    artifact = tmp_path / "m.pkl"
    artifact.write_text("m")

    with pytest.raises(ValueError, match="managed custody requires"):
        manager.deploy_worker(topic_id=1, artifact_path=artifact, custody="managed")


# ----------------------------
# Real-SDK contract tests (synth-001 / synth-003): exercise the actual allora_sdk surface
# the managed lifecycle depends on, so a cross-repo contract break cannot hide behind the
# _FakeForgeClient stub or an argv-only assertion.
# ----------------------------
def test_real_forge_backend_client_exposes_managed_custody_methods():
    """The real ForgeBackendClient WorkerManager imports must expose the two methods the
    lifecycle calls: provision_wallet (deploy) and clear_association (remove)."""
    from allora_sdk.rpc_client.remote_signer import ForgeBackendClient

    assert hasattr(ForgeBackendClient, "provision_wallet")
    assert hasattr(ForgeBackendClient, "clear_association")


def test_real_sdk_wallet_config_defers_for_managed_worker_without_crashing(monkeypatch):
    """Refutes 'every managed worker crashes with No wallet credentials provided': with only
    FORGE_API_KEY set (the deferred managed contract), the real AlloraWalletConfig.from_env()
    returns a managed config instead of raising."""
    from allora_sdk.rpc_client.config import AlloraWalletConfig

    monkeypatch.setenv("FORGE_API_KEY", "forge_sk_test")
    monkeypatch.setenv("FORGE_BACKEND_URL", "http://localhost:8080")
    for key in ("FORGE_SIGNING_WALLET_ID", "PRIVATE_KEY", "MNEMONIC", "MNEMONIC_FILE"):
        monkeypatch.delenv(key, raising=False)

    cfg = AlloraWalletConfig.from_env()
    assert cfg.forge_api_key == "forge_sk_test"


def test_managed_env_from_build_run_command_constructs_wallet_config(tmp_path: Path, monkeypatch):
    """The exact env _build_run_command injects for a managed worker drives the real
    AlloraWalletConfig.from_env() to a wallet-backed config without raising (HTTP mocked)."""
    import allora_sdk.rpc_client.remote_signer as rs
    from allora_sdk.rpc_client.config import AlloraWalletConfig

    monkeypatch.setenv("ALLORA_API_KEY", "test-allora-key")
    client = _FakeForgeClient()
    manager = _managed_manager(
        tmp_path,
        client,
        forge_api_key="forge_sk_test",
        forge_backend_url="http://localhost:8080",
    )
    artifact = tmp_path / "m.pkl"
    artifact.write_text("m")
    manager.deploy_worker(topic_id=7, artifact_path=artifact, custody="managed")
    status = manager.status_worker(topic_id=7, address="allo1managed0007")
    _, env = manager._build_run_command(7, "allo1managed0007", status)

    fake_wallet = SimpleNamespace(address=lambda: "allo1managed0007")
    monkeypatch.setattr(rs, "make_remote_wallet", lambda *a, **k: fake_wallet)
    for key in ("FORGE_API_KEY", "FORGE_BACKEND_URL", "FORGE_SIGNING_WALLET_ID"):
        monkeypatch.setenv(key, env[key])

    cfg = AlloraWalletConfig.from_env()
    assert cfg.wallet is fake_wallet


def test_status_all_includes_custody_and_signing_wallet_id(tmp_path: Path):
    """status_all() exposes the same custody/signing_wallet_id contract as status_worker(), so
    a dashboard iterating status_all() can tell managed from local without an N+1 round-trip."""
    client = _FakeForgeClient()
    manager = _managed_manager(tmp_path, client)
    artifact = tmp_path / "m.pkl"
    artifact.write_text("m")
    manager.deploy_worker(topic_id=42, artifact_path=artifact, custody="managed")

    row = [w for w in manager.status_all() if w["topic_id"] == 42][0]
    assert row["custody"] == "managed"
    assert row["signing_wallet_id"] == "wallet-42"

    worker = manager.status_worker(topic_id=42, address="allo1managed0042")
    assert {"custody", "signing_wallet_id"} <= (set(row) & set(worker))
