from __future__ import annotations

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Any


@dataclass(frozen=True)
class Identity:
    alias: str
    address: str


@dataclass
class WorkerSpec:
    topic_id: int
    topic_desc: Optional[str]
    address: str
    artifact_path: Path
    identity_ref: str
    enabled: bool = True


@dataclass
class DeployResult:
    topic_id: int
    address_assigned: str
    artifact_path: str
    action: str  # created|reused|replaced
    message: str


class WorkerManager:
    """Lightweight local worker registry + lifecycle manager.

    Designed for one worker per (topic_id, address) with simple auto-assignment logic.
    """

    def __init__(
        self,
        db_path: str | Path = "worker_state.db",
        secrets_path: str | Path = "worker_secrets.json",
        identity_creator: Optional[Callable[[], tuple[str, str, str]]] = None,
        monitor: Optional[Any] = None,
        auto_monitor_sync: bool = True,
        topic_desc_resolver: Optional[Callable[[int], Optional[str]]] = None,
        runtime_log_dir: str | Path = "worker_logs",
        artifact_dir: str | Path = "managed_artifacts",
        reconcile_on_start: bool = True,
    ):
        self.db_path = Path(db_path)
        self.secrets_path = Path(secrets_path)
        self._identity_creator = identity_creator or self._default_identity_creator
        self._monitor = monitor
        self._auto_monitor_sync = auto_monitor_sync
        self._topic_desc_resolver = topic_desc_resolver or self._build_default_topic_desc_resolver()
        self.runtime_log_dir = Path(runtime_log_dir)
        self.runtime_log_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._runners: dict[tuple[int, str], dict] = {}
        self._init_db()
        if reconcile_on_start:
            self.reconcile()

    # ----------------------------
    # Identity handling
    # ----------------------------
    def ensure_identity(self, alias: str | None = None, address: str | None = None, mnemonic: str | None = None) -> Identity:
        with self._lock:
            if address:
                existing = self._get_identity_by_address(address)
                if existing:
                    return Identity(alias=existing["alias"], address=existing["address"])
                if not mnemonic:
                    raise ValueError("Mnemonic is required when importing a new address")
                final_alias = alias or f"imported_{int(time.time())}"
                self._insert_identity(final_alias, address, mnemonic)
                return Identity(alias=final_alias, address=address)

            created_alias, created_address, created_mnemonic = self._identity_creator()
            final_alias = alias or created_alias
            self._insert_identity(final_alias, created_address, created_mnemonic)
            return Identity(alias=final_alias, address=created_address)

    def list_identities(self) -> list[Identity]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT alias, address FROM identities ORDER BY alias").fetchall()
        return [Identity(alias=r[0], address=r[1]) for r in rows]

    # ----------------------------
    # Worker CRUD
    # ----------------------------
    def add_worker(self, spec: WorkerSpec) -> None:
        if not spec.artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {spec.artifact_path}")
        resolved_desc = self._resolve_topic_desc(spec.topic_id, spec.topic_desc)
        managed_artifact = self._materialize_artifact(spec.topic_id, spec.address, spec.artifact_path)
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO workers(topic_id, topic_desc, address, artifact_path, identity_ref, enabled, status, deployed_at, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, 'stopped', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        spec.topic_id,
                        resolved_desc,
                        spec.address,
                        str(managed_artifact),
                        spec.identity_ref,
                        1 if spec.enabled else 0,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"Worker already exists for topic={spec.topic_id}, address={spec.address}") from e

        deployment_id = self._create_deployment_record(spec.topic_id, spec.address, managed_artifact)
        self._monitor_register(spec.topic_id, spec.address, deployment_id=deployment_id)

    def remove_worker(self, topic_id: int, address: str, force: bool = False) -> None:
        if force:
            self.stop_worker(topic_id, address)
        self._archive_active_deployment(topic_id, address)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM workers WHERE topic_id=? AND address=?", (topic_id, address))
            conn.commit()
        self._monitor_disable(topic_id, address)

    def status_worker(self, topic_id: int, address: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT topic_id, COALESCE(topic_desc, ''), address, artifact_path, identity_ref, enabled, status,
                       COALESCE(last_error, ''), deployed_at, updated_at, last_pid, last_started_at, last_stopped_at, last_exit_code
                FROM workers WHERE topic_id=? AND address=?
                """,
                (topic_id, address),
            ).fetchone()
        if not row:
            raise KeyError(f"Worker not found: topic={topic_id} address={address}")
        return {
            "topic_id": row[0],
            "topic_desc": row[1],
            "address": row[2],
            "artifact_path": row[3],
            "identity_ref": row[4],
            "enabled": bool(row[5]),
            "status": row[6],
            "last_error": row[7] or None,
            "deployed_at": row[8],
            "updated_at": row[9],
            "last_pid": row[10],
            "last_started_at": row[11],
            "last_stopped_at": row[12],
            "last_exit_code": row[13],
        }

    def status_all(self, include_desc: bool = True) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT topic_id, COALESCE(topic_desc, ''), address, artifact_path, identity_ref, enabled, status,
                       COALESCE(last_error, ''), deployed_at, updated_at, last_pid, last_started_at, last_stopped_at, last_exit_code
                FROM workers ORDER BY topic_id, address
                """
            ).fetchall()
        out = []
        for row in rows:
            item = {
                "topic_id": row[0],
                "address": row[2],
                "artifact_path": row[3],
                "identity_ref": row[4],
                "enabled": bool(row[5]),
                "status": row[6],
                "last_error": row[7] or None,
                "deployed_at": row[8],
                "updated_at": row[9],
                "last_pid": row[10],
                "last_started_at": row[11],
                "last_stopped_at": row[12],
                "last_exit_code": row[13],
            }
            if include_desc:
                item["topic_desc"] = row[1]
            out.append(item)
        return out

    def get_worker_log_tail(self, topic_id: int, address: str, lines: int = 20) -> list[str]:
        """Return last N stdout log lines for a worker slot."""
        lines = max(1, min(int(lines), 500))
        log_path = self.runtime_log_dir / f"worker_{topic_id}_{address}.log"
        if not log_path.exists():
            return []
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                data = f.readlines()
            return [ln.rstrip("\n") for ln in data[-lines:]]
        except Exception:
            return []

    def status_all_with_logs(self, include_desc: bool = True, tail_lines: int = 20) -> list[dict]:
        rows = self.status_all(include_desc=include_desc)
        for r in rows:
            r["log_tail"] = self.get_worker_log_tail(r["topic_id"], r["address"], lines=tail_lines)
        return rows

    # ----------------------------
    # Deploy logic (smart assignment)
    # ----------------------------
    def deploy_worker(
        self,
        topic_id: int,
        artifact_path: str | Path,
        address: str | None = None,
        mnemonic: str | None = None,
        identity_alias: str | None = None,
        topic_desc: str | None = None,
        replace: bool = False,
        mode: str = "auto",
    ) -> DeployResult:
        artifact = Path(artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact}")

        if mode not in {"auto", "strict"}:
            raise ValueError("mode must be 'auto' or 'strict'")

        # Explicit address path
        if address:
            existing = self._worker_exists(topic_id, address)
            if existing and replace:
                self._update_worker(topic_id, address, artifact, topic_desc)
                current_artifact = Path(self.status_worker(topic_id, address)["artifact_path"])
                deployment_id = self._rotate_deployment(topic_id, address, current_artifact)
                self._monitor_register(topic_id, address, deployment_id=deployment_id)
                return DeployResult(
                    topic_id=topic_id,
                    address_assigned=address,
                    artifact_path=str(artifact),
                    action="replaced",
                    message=f"Replaced worker artifact for topic {topic_id} and address {address}",
                )
            if existing and not replace:
                if mode == "strict":
                    raise ValueError(f"Worker already exists for topic={topic_id} address={address}")
                # auto mode: allocate alternate identity/address
                ident, _ = self._pick_or_create_identity_for_topic(topic_id)
                spec = WorkerSpec(topic_id, topic_desc, ident.address, artifact, ident.alias)
                self.add_worker(spec)
                return DeployResult(
                    topic_id=topic_id,
                    address_assigned=ident.address,
                    artifact_path=str(artifact),
                    action="created",
                    message=(
                        f"Address {address} already used for topic {topic_id}; "
                        f"created new worker with address {ident.address}"
                    ),
                )

            ident = self.ensure_identity(alias=identity_alias, address=address, mnemonic=mnemonic)
            action = "reused" if self._address_has_other_topics(ident.address) else "created"
            spec = WorkerSpec(topic_id, topic_desc, ident.address, artifact, ident.alias)
            self.add_worker(spec)
            return DeployResult(
                topic_id=topic_id,
                address_assigned=ident.address,
                artifact_path=str(artifact),
                action=action,
                message=f"Deployed worker for topic {topic_id} with address {ident.address}",
            )

        # Auto address path: reuse free identity first, else create new
        ident, created = self._pick_or_create_identity_for_topic(topic_id)
        action = "created" if created else "reused"
        spec = WorkerSpec(topic_id, topic_desc, ident.address, artifact, ident.alias)
        self.add_worker(spec)
        return DeployResult(
            topic_id=topic_id,
            address_assigned=ident.address,
            artifact_path=str(artifact),
            action=action,
            message=f"Deployed worker for topic {topic_id} with address {ident.address}",
        )

    # ----------------------------
    # Lifecycle (persistent managed process runner)
    # ----------------------------
    def start_worker(self, topic_id: int, address: str) -> None:
        status = self.status_worker(topic_id, address)
        pid = status.get("last_pid")
        if pid and self._is_pid_alive(pid):
            self._set_worker_status(topic_id, address, status="running", last_error=None)
            return

        log_path = self.runtime_log_dir / f"worker_{topic_id}_{address}.log"
        log_f = open(log_path, "ab")
        cmd = [
            sys.executable,
            "-m",
            "allora_forge_builder_kit.worker_runtime",
            "--topic",
            str(topic_id),
            "--artifact",
            str(status["artifact_path"]),
        ]
        if "price" in str(status.get("topic_desc", "")).lower():
            cmd.append("--reject-zero")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(Path.cwd()))
        key = (topic_id, address)
        self._runners[key] = {"proc": proc, "log": log_f}

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE workers
                SET status='running', last_error=NULL, last_pid=?, last_started_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
                WHERE topic_id=? AND address=?
                """,
                (proc.pid, topic_id, address),
            )
            conn.commit()

    def stop_worker(self, topic_id: int, address: str, timeout_sec: int = 20) -> None:
        key = (topic_id, address)
        runner = self._runners.get(key)
        pid = None

        if runner and runner.get("proc"):
            proc = runner["proc"]
            pid = proc.pid
            proc.terminate()
            try:
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
            if runner.get("log"):
                runner["log"].close()
            self._runners.pop(key, None)
        else:
            s = self.status_worker(topic_id, address)
            pid = s.get("last_pid")
            if pid and self._is_pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE workers
                SET status='stopped', last_error=NULL, last_stopped_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
                WHERE topic_id=? AND address=?
                """,
                (topic_id, address),
            )
            conn.commit()

    def start_all(self, only_enabled: bool = True) -> dict:
        started = 0
        for row in self.status_all():
            if only_enabled and not row["enabled"]:
                continue
            self.start_worker(row["topic_id"], row["address"])
            started += 1
        return {"started": started}

    def stop_all(self, timeout_sec: int = 20) -> dict:
        count = 0
        for row in self.status_all():
            if row["status"] == "running":
                self.stop_worker(row["topic_id"], row["address"], timeout_sec=timeout_sec)
                count += 1
        return {"stopped": count}

    def health_worker(self, topic_id: int, address: str) -> dict:
        status = self.status_worker(topic_id, address)
        good = status["status"] == "running" and not status["last_error"]
        status["health"] = "good" if good else ("bad" if status["status"] == "crashed" else "degraded")
        return status

    def health_all(self) -> list[dict]:
        return [self.health_worker(r["topic_id"], r["address"]) for r in self.status_all()]

    def reconcile(self) -> dict:
        restarted = 0
        running = 0
        for row in self.status_all():
            pid = row.get("last_pid")
            alive = bool(pid and self._is_pid_alive(pid))
            if row["enabled"] and (row["status"] == "running" or row.get("last_pid")):
                if alive:
                    running += 1
                else:
                    self.start_worker(row["topic_id"], row["address"])
                    restarted += 1
            elif row["enabled"] and row["status"] == "stopped":
                # autostart enabled workers on manager boot
                self.start_worker(row["topic_id"], row["address"])
                restarted += 1
        return {"running": running, "restarted": restarted}

    def refresh_topic_descriptions(self) -> dict:
        """Refresh all worker topic descriptions from resolver, when configured."""
        if not self._topic_desc_resolver:
            return {"updated": 0}
        updated = 0
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT DISTINCT topic_id FROM workers").fetchall()
            for (topic_id,) in rows:
                desc = self._resolve_topic_desc(topic_id, None)
                if desc:
                    conn.execute("UPDATE workers SET topic_desc=? WHERE topic_id=?", (desc, topic_id))
                    updated += 1
            conn.commit()
        return {"updated": updated}

    def attach_monitor(self, monitor: Any, backfill_since: Optional[str] = None, sync_now: bool = True) -> dict:
        """Attach monitor and optionally bootstrap existing workers into monitoring.

        Args:
            monitor: WorkerMonitor-compatible instance.
            backfill_since: Optional ISO timestamp for backfill cursor.
            sync_now: Whether to trigger monitor sync/backfill immediately.
        """
        self._monitor = monitor
        registered = 0
        for row in self.status_all():
            try:
                deployed_at = row.get("deployed_at")
                dep_id = self._get_active_deployment_id(row["topic_id"], row["address"])
                self._monitor.register_target(row["topic_id"], row["address"], deployed_at=deployed_at, deployment_id=dep_id)
                if sync_now:
                    if backfill_since:
                        self._monitor.backfill_target(row["topic_id"], row["address"], since=backfill_since)
                    else:
                        self._monitor.sync_once()
                registered += 1
            except Exception:
                pass
        return {"registered": registered}

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.secrets_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS identities (
                    alias TEXT PRIMARY KEY,
                    address TEXT NOT NULL UNIQUE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id INTEGER NOT NULL,
                    topic_desc TEXT,
                    address TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    identity_ref TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'stopped',
                    last_error TEXT,
                    deployed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_pid INTEGER,
                    last_started_at TEXT,
                    last_stopped_at TEXT,
                    last_exit_code INTEGER,
                    UNIQUE(topic_id, address)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_deployments (
                    deployment_id TEXT PRIMARY KEY,
                    topic_id INTEGER NOT NULL,
                    address TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    artifact_hash TEXT,
                    deployed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    ended_at TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            # lightweight migration for existing installs
            cols = [r[1] for r in conn.execute("PRAGMA table_info(workers)").fetchall()]
            if "deployed_at" not in cols:
                conn.execute("ALTER TABLE workers ADD COLUMN deployed_at TEXT")
                conn.execute("UPDATE workers SET deployed_at = COALESCE(updated_at, CURRENT_TIMESTAMP) WHERE deployed_at IS NULL")
            if "last_pid" not in cols:
                conn.execute("ALTER TABLE workers ADD COLUMN last_pid INTEGER")
            if "last_started_at" not in cols:
                conn.execute("ALTER TABLE workers ADD COLUMN last_started_at TEXT")
            if "last_stopped_at" not in cols:
                conn.execute("ALTER TABLE workers ADD COLUMN last_stopped_at TEXT")
            if "last_exit_code" not in cols:
                conn.execute("ALTER TABLE workers ADD COLUMN last_exit_code INTEGER")
            conn.commit()
        if not self.secrets_path.exists():
            self.secrets_path.write_text("{}")
            try:
                self.secrets_path.chmod(0o600)
            except PermissionError:
                pass

    def _insert_identity(self, alias: str, address: str, mnemonic: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO identities(alias, address) VALUES(?, ?)", (alias, address))
            conn.commit()
        data = self._load_secrets()
        data[alias] = {"address": address, "mnemonic": mnemonic}
        self._save_secrets(data)

    def _load_secrets(self) -> dict:
        try:
            return json.loads(self.secrets_path.read_text())
        except Exception:
            return {}

    def _save_secrets(self, data: dict) -> None:
        self.secrets_path.write_text(json.dumps(data, indent=2))
        try:
            self.secrets_path.chmod(0o600)
        except PermissionError:
            pass

    def _default_identity_creator(self) -> tuple[str, str, str]:
        # Placeholder creator for local development; callers should inject SDK creator.
        ts = int(time.time())
        return (f"identity_{ts}", f"address_{ts}", f"mnemonic_{ts}")

    def _get_identity_by_address(self, address: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT alias, address FROM identities WHERE address=?", (address,)).fetchone()
        if not row:
            return None
        return {"alias": row[0], "address": row[1]}

    def _identity_existed(self, alias: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT 1 FROM identities WHERE alias=?", (alias,)).fetchone()
        return row is not None

    def _worker_exists(self, topic_id: int, address: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT 1 FROM workers WHERE topic_id=? AND address=?", (topic_id, address)).fetchone()
        return row is not None

    def _address_has_other_topics(self, address: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(1) FROM workers WHERE address=?", (address,)).fetchone()
        return bool(row and row[0] > 0)

    def _pick_or_create_identity_for_topic(self, topic_id: int) -> tuple[Identity, bool]:
        identities = self.list_identities()
        for ident in identities:
            if not self._worker_exists(topic_id, ident.address):
                return ident, False
        return self.ensure_identity(), True

    def _materialize_artifact(self, topic_id: int, address: str, source_artifact: Path) -> Path:
        target_dir = self.artifact_dir / f"topic_{topic_id}" / address
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"predict_{uuid.uuid4().hex}.pkl"
        shutil.copy2(source_artifact, target_path)
        return target_path

    def _update_worker(self, topic_id: int, address: str, artifact_path: Path, topic_desc: str | None) -> None:
        resolved_desc = self._resolve_topic_desc(topic_id, topic_desc)
        managed_artifact = self._materialize_artifact(topic_id, address, artifact_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE workers
                   SET artifact_path=?,
                       topic_desc=COALESCE(?, topic_desc),
                       updated_at=CURRENT_TIMESTAMP
                 WHERE topic_id=? AND address=?
                """,
                (str(managed_artifact), resolved_desc, topic_id, address),
            )
            conn.commit()

    def _build_default_topic_desc_resolver(self) -> Optional[Callable[[int], Optional[str]]]:
        api_key = os.environ.get("ALLORA_API_KEY")
        if not api_key:
            for candidate in (Path("notebooks/.allora_api_key"), Path(".allora_api_key")):
                try:
                    if candidate.exists():
                        api_key = candidate.read_text().strip()
                        if api_key:
                            break
                except Exception:
                    pass
        try:
            return build_topic_desc_resolver(api_key=api_key, network="testnet")
        except Exception:
            return None

    def _resolve_topic_desc(self, topic_id: int, fallback: Optional[str]) -> Optional[str]:
        if self._topic_desc_resolver:
            try:
                resolved = self._topic_desc_resolver(topic_id)
                if resolved:
                    return resolved
            except Exception:
                pass
        return fallback

    def _set_worker_status(self, topic_id: int, address: str, status: str, last_error: str | None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE workers
                   SET status=?, last_error=?, updated_at=CURRENT_TIMESTAMP
                 WHERE topic_id=? AND address=?
                """,
                (status, last_error, topic_id, address),
            )
            conn.commit()

    def _is_pid_alive(self, pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False

    def _get_worker_deployed_at(self, topic_id: int, address: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT deployed_at FROM workers WHERE topic_id=? AND address=?",
                (topic_id, address),
            ).fetchone()
        if not row:
            return None
        return row[0]

    def _create_deployment_record(self, topic_id: int, address: str, artifact_path: Path) -> str:
        deployment_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE worker_deployments SET is_active=0, ended_at=CURRENT_TIMESTAMP WHERE topic_id=? AND address=? AND is_active=1",
                (topic_id, address),
            )
            conn.execute(
                """
                INSERT INTO worker_deployments(deployment_id, topic_id, address, artifact_path, artifact_hash, deployed_at, is_active)
                VALUES(?, ?, ?, ?, NULL, CURRENT_TIMESTAMP, 1)
                """,
                (deployment_id, topic_id, address, str(artifact_path)),
            )
            conn.commit()

        # API-first safety: always advance monitor target deployment pointer at redeploy,
        # even when WorkerManager is instantiated without a monitor object.
        self._set_monitor_target_deployment_db(topic_id=topic_id, address=address, deployment_id=deployment_id)
        return deployment_id

    def _get_active_deployment_id(self, topic_id: int, address: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT deployment_id FROM worker_deployments WHERE topic_id=? AND address=? AND is_active=1 ORDER BY deployed_at DESC LIMIT 1",
                (topic_id, address),
            ).fetchone()
        return row[0] if row else None

    def _archive_active_deployment(self, topic_id: int, address: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE worker_deployments SET is_active=0, ended_at=CURRENT_TIMESTAMP WHERE topic_id=? AND address=? AND is_active=1",
                (topic_id, address),
            )
            conn.commit()

    def _rotate_deployment(self, topic_id: int, address: str, artifact_path: Path) -> str:
        self._archive_active_deployment(topic_id, address)
        return self._create_deployment_record(topic_id, address, artifact_path)

    def _monitor_register(self, topic_id: int, address: str, deployment_id: Optional[str] = None) -> None:
        if not self._monitor:
            return
        deployed_at = self._get_worker_deployed_at(topic_id, address)
        dep_id = deployment_id or self._get_active_deployment_id(topic_id, address)
        try:
            self._monitor.register_target(topic_id=topic_id, address=address, deployed_at=deployed_at, deployment_id=dep_id)
            if dep_id and hasattr(self._monitor, "set_target_deployment"):
                self._monitor.set_target_deployment(topic_id=topic_id, address=address, deployment_id=dep_id, deployed_at=deployed_at)
            if self._auto_monitor_sync:
                self._monitor.sync_once()
        except Exception:
            # keep manager resilient if monitor backend is unavailable
            pass

    def _monitor_disable(self, topic_id: int, address: str) -> None:
        if not self._monitor:
            return
        try:
            self._monitor.set_target_enabled(topic_id=topic_id, address=address, enabled=False)
        except Exception:
            pass

    def _set_monitor_target_deployment_db(self, topic_id: int, address: str, deployment_id: str) -> None:
        """Advance monitor target pointer directly in DB for deployment-scoped stats.

        This keeps monitoring history segmented per deployed artifact even when no
        WorkerMonitor instance is attached to the active WorkerManager process.
        """
        deployed_at = self._get_worker_deployed_at(topic_id, address)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS monitor_targets (
                    topic_id INTEGER NOT NULL,
                    address TEXT NOT NULL,
                    deployed_at TEXT NOT NULL,
                    deployment_id TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_sync_at TEXT,
                    PRIMARY KEY(topic_id, address)
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO monitor_targets(topic_id, address, deployed_at, deployment_id, enabled, last_sync_at)
                VALUES(?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?, 1, NULL)
                """,
                (topic_id, address, deployed_at, deployment_id),
            )
            conn.execute(
                """
                UPDATE monitor_targets
                SET deployment_id=?, deployed_at=COALESCE(?, deployed_at), last_sync_at=NULL
                WHERE topic_id=? AND address=?
                """,
                (deployment_id, deployed_at, topic_id, address),
            )
            conn.commit()


def build_topic_desc_resolver(api_key: Optional[str] = None, network: str = "testnet") -> Callable[[int], Optional[str]]:
    """Build a topic description resolver backed by Allora topic discovery."""
    from .topic_discovery import AlloraTopicDiscovery

    discovery = AlloraTopicDiscovery(api_key=api_key, network=network)
    cache = {
        t.topic_id: (t.raw.get("topic_name") or t.description or "")
        for t in discovery.get_all_topics()
    }

    def _resolve(topic_id: int) -> Optional[str]:
        return cache.get(topic_id)

    return _resolve
