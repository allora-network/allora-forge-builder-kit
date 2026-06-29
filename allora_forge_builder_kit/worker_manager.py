from __future__ import annotations

import hashlib
import json
import logging
import os
import re
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
from typing import Any, Callable, Literal, Optional, Protocol, runtime_checkable

from .worker_monitor import MONITOR_TARGETS_DDL

logger = logging.getLogger(__name__)

# Wallet custody discriminant: 'local' (self-custodial key file on disk) or 'managed' (Privy
# wallet provisioned by the Forge backend, keyed by signing_wallet_id). A Literal gives the
# fixed two-value set type-checker coverage and IDE completion while staying a plain str on the
# wire and in SQLite.
CustodyMode = Literal["local", "managed"]

# Outcome of a deploy_worker call. The domain is fixed and small, so a Literal (like CustodyMode)
# gives the type-checker coverage that catches a typo such as "replace" and documents the contract
# callers branch on (e.g. firing downstream jobs only on "created"/"replaced").
DeployAction = Literal["created", "reused", "replaced"]


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
    reject_zero: bool = False
    # custody: "local" (self-custodial key file on disk) or "managed" (Privy-managed
    # wallet provisioned by the Forge backend; signing_wallet_id is the backend wallet id).
    custody: CustodyMode = "local"
    signing_wallet_id: Optional[str] = None


@dataclass
class DeployResult:
    topic_id: int
    address_assigned: str
    artifact_path: str
    action: DeployAction
    message: str


class ProvisionedWallet(Protocol):
    """Structural view of a Forge-provisioned wallet (the SDK's ``SigningWalletInfo``): the
    non-secret id + address the managed lifecycle reads back after provisioning."""

    id: str
    address: str


@runtime_checkable
class ForgeClientProtocol(Protocol):
    """Contract for the Forge backend client used by managed custody.

    Implemented by ``allora_sdk``'s ``ForgeBackendClient`` and stubbed in tests. Captures only the
    two calls :class:`WorkerManager` makes so local-custody installs need not import the SDK and the
    injected client is checked at the boundary instead of being typed as ``Any``. ``@runtime_checkable``
    lets the lazy build assert the SDK client satisfies this contract at the injection boundary.
    """

    def provision_wallet(self, topic_id: int, label: Optional[str] = None) -> ProvisionedWallet:
        """Idempotently get-or-create the managed wallet bound to ``topic_id``."""
        ...

    def clear_association(self, wallet_id: str) -> None:
        """Release the wallet's (user, topic) binding on the backend."""
        ...


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
        key_dir: str | Path = "worker_keys",
        network: str = "testnet",
        no_faucet: bool = False,
        reconcile_on_start: bool = True,
        forge_api_key: Optional[str] = None,
        forge_backend_url: Optional[str] = None,
        forge_client: Optional[ForgeClientProtocol] = None,
    ):
        """Initialise the worker manager.

        Args:
            db_path: Path to the SQLite state database.
            secrets_path: Path to the JSON file storing worker identities.
            identity_creator: Callable that returns ``(alias, address, mnemonic)``
                for new worker identities.  Defaults to an internal creator that
                generates a fresh mnemonic and derives the Allora address.
            monitor: Optional :class:`WorkerMonitor` instance for on-chain
                event tracking.
            auto_monitor_sync: When *True*, automatically sync monitor targets
                after worker state changes (deploy, remove, etc.).
            topic_desc_resolver: Optional callable mapping a topic ID to a
                human-readable description string.
            runtime_log_dir: Directory for worker process stdout/stderr logs.
            artifact_dir: Directory for managed worker artifacts
                (Dockerfiles, configs, etc.).
            key_dir: Directory for per-worker key files.
            network: Allora network name (``'testnet'`` or ``'mainnet'``).
            no_faucet: Skip the testnet faucet drip when creating identities.
            reconcile_on_start: When *True* (the default), call
                :meth:`reconcile` during construction, which spawns
                subprocesses for every enabled worker.  Set to *False* for
                unit tests or when deferred startup is desired.
            forge_api_key: Forge API key (``forge_sk_…``) used to provision /
                release managed (Privy) wallets. Defaults to ``$FORGE_API_KEY``.
            forge_backend_url: Forge backend base URL for managed custody.
                Defaults to ``$FORGE_BACKEND_URL``.
            forge_client: Pre-built Forge backend client (must expose
                ``provision_wallet`` and ``clear_association``). Injected in tests;
                in production it is built lazily from the api key + url.
        """
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
        self.key_dir = Path(key_dir)
        self.key_dir.mkdir(parents=True, exist_ok=True)
        self._network = network
        self._no_faucet = no_faucet
        self._forge_api_key = forge_api_key or os.environ.get("FORGE_API_KEY")
        self._forge_backend_url = forge_backend_url or os.environ.get("FORGE_BACKEND_URL")
        self._forge_client_cache = forge_client
        self._lock = threading.RLock()
        self._runners: dict[tuple[int, str], dict] = {}
        self._init_db()
        if reconcile_on_start:
            self.reconcile()

    # ----------------------------
    # Managed custody (Privy via Forge backend)
    # ----------------------------
    def _forge_client(self) -> ForgeClientProtocol:
        """Return the Forge backend client for managed custody, building it lazily.

        Raises ``ValueError`` if the api key / backend url are missing, so a misconfigured
        managed deploy fails loudly instead of silently falling back to local custody.
        """
        # Fast path: an already-built (or test-injected) client needs no lock.
        if self._forge_client_cache is not None:
            return self._forge_client_cache
        if not self._forge_api_key or not self._forge_backend_url:
            raise ValueError(
                "managed custody requires a Forge API key and backend URL; set "
                "$FORGE_API_KEY and $FORGE_BACKEND_URL or pass forge_api_key/forge_backend_url"
            )
        # Imported lazily: local-custody installs need not import the SDK signing client.
        try:
            from allora_sdk.rpc_client.remote_signer import ForgeBackendClient
        except ImportError as e:
            raise ValueError(
                "managed custody requires the 'allora-sdk' package "
                "(allora_sdk.rpc_client.remote_signer.ForgeBackendClient); install it to deploy "
                "managed workers"
            ) from e

        # Build the client (SDK import + requests.Session/TLS setup) outside the lock so it does
        # not serialize unrelated manager operations that share this RLock. Two threads may race
        # to build; the double-checked assignment under the lock keeps the first and drops the
        # loser (its Session is released on GC).
        client = ForgeBackendClient(self._forge_backend_url, self._forge_api_key)
        # Enforce the structural contract at the injection boundary, not only in tests: if the SDK
        # renames or drops provision_wallet / clear_association, fail loudly here instead of at the
        # first managed deploy/teardown call.
        if not isinstance(client, ForgeClientProtocol):
            raise TypeError(
                "allora_sdk ForgeBackendClient does not satisfy ForgeClientProtocol "
                "(expected provision_wallet + clear_association); SDK contract drift"
            )
        with self._lock:
            if self._forge_client_cache is None:
                self._forge_client_cache = client
            return self._forge_client_cache

    # ----------------------------
    # Identity handling
    # ----------------------------
    def ensure_identity(self, alias: str | None = None, address: str | None = None, mnemonic: str | None = None, key_file: str | Path | None = None) -> Identity:
        with self._lock:
            if address:
                existing = self._get_identity_by_address(address)
                if existing:
                    return Identity(alias=existing["alias"], address=existing["address"])
                if not mnemonic and not key_file:
                    raise ValueError("A mnemonic or key_file is required when importing a new address")
                final_alias = alias or f"imported_{int(time.time())}"
                kf = self._persist_key_file(final_alias, mnemonic=mnemonic, source_file=key_file)
                self._insert_identity(final_alias, address, kf)
                return Identity(alias=final_alias, address=address)

            created_alias, created_address, created_mnemonic = self._identity_creator()
            final_alias = alias or created_alias
            kf = self._persist_key_file(final_alias, mnemonic=created_mnemonic)
            self._insert_identity(final_alias, created_address, kf)
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
                    INSERT INTO workers(topic_id, topic_desc, address, artifact_path, identity_ref, enabled, status, reject_zero, custody, signing_wallet_id, deployed_at, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, 'stopped', ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        spec.topic_id,
                        resolved_desc,
                        spec.address,
                        str(managed_artifact),
                        spec.identity_ref,
                        1 if spec.enabled else 0,
                        1 if spec.reject_zero else 0,
                        spec.custody,
                        spec.signing_wallet_id,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"Worker already exists for topic={spec.topic_id}, address={spec.address}") from e

        deployment_id = self._create_deployment_record(spec.topic_id, spec.address, managed_artifact)
        self._monitor_register(spec.topic_id, spec.address, deployment_id=deployment_id)

    def remove_worker(self, topic_id: int, address: str, force: bool = False) -> None:
        custody, signing_wallet_id = self._get_custody(topic_id, address)
        # Stop the running process before tearing down local state and the backend binding.
        # Managed custody always stops first (best-effort): a still-running subprocess would
        # otherwise keep submitting with a wallet we are about to unbind server-side, and a later
        # redeploy could provision a second wallet for the same topic — two active workers. Local
        # custody keeps its force-gated stop. The stop is best-effort so a dead/unknown process
        # never blocks decommission.
        if force or custody == "managed":
            try:
                self.stop_worker(topic_id, address)
            except Exception as e:  # noqa: BLE001 - a dead/unknown process must not block removal
                logger.warning(
                    "stop before remove failed for topic=%s address=%s: %s; removing anyway",
                    topic_id,
                    address,
                    e,
                )
        self._archive_active_deployment(topic_id, address)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM workers WHERE topic_id=? AND address=?", (topic_id, address))
            conn.commit()
        self._monitor_disable(topic_id, address)
        # Managed custody: release the (user, topic) binding on the backend so the topic slot is
        # freed. Best-effort — the worker is already gone locally, and the backend get-or-create is
        # idempotent, so a stale binding is simply reused on the next deploy rather than leaking.
        if custody == "managed" and signing_wallet_id:
            self._release_managed_binding(signing_wallet_id, topic_id)

    def _release_managed_binding(self, signing_wallet_id: str, topic_id: int, timeout: float = 5.0) -> None:
        """Best-effort release of a managed wallet's (user, topic) binding on the Forge backend.

        Runs ``clear_association`` on a bounded daemon thread and waits at most ``timeout`` seconds.
        The SDK call takes no per-request timeout, so without this bound a degraded backend would
        stall the caller for the full SDK timeout and serialize batch teardowns one RTT at a time —
        even though the local state is already gone. The backend get-or-create is idempotent, so a
        binding left unreleased here is reused on the next deploy rather than leaking. Never raises:
        decommission cleanup must neither block on nor be aborted by the backend.
        """
        result: dict[str, BaseException] = {}

        def _clear() -> None:
            try:
                self._forge_client().clear_association(signing_wallet_id)
            except BaseException as e:  # noqa: BLE001 - surfaced via result; must not escape the thread
                result["error"] = e

        worker = threading.Thread(
            target=_clear, name=f"clear-association-{signing_wallet_id}", daemon=True
        )
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            logger.warning(
                "clear-association for managed wallet %s (topic %s) did not finish within %.0fs; "
                "removed locally anyway (stale binding is reused on the next deploy)",
                signing_wallet_id, topic_id, timeout,
            )
            return
        error = result.get("error")
        if error is not None:
            logger.warning(
                "clear-association failed for managed wallet %s (topic %s): %s; removed locally anyway",
                signing_wallet_id, topic_id, error,
            )
            return
        logger.info("released managed wallet %s topic binding (topic %s)", signing_wallet_id, topic_id)

    def status_worker(self, topic_id: int, address: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT topic_id, COALESCE(topic_desc, ''), address, artifact_path, identity_ref, enabled, status,
                       COALESCE(last_error, ''), deployed_at, updated_at, last_pid, last_started_at, last_stopped_at, last_exit_code,
                       COALESCE(custody, 'local'), signing_wallet_id, reject_zero
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
            "custody": row[14],
            "signing_wallet_id": row[15],
            "reject_zero": bool(row[16]) if row[16] is not None else False,
        }

    def status_all(self, include_desc: bool = True) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT topic_id, COALESCE(topic_desc, ''), address, artifact_path, identity_ref, enabled, status,
                       COALESCE(last_error, ''), deployed_at, updated_at, last_pid, last_started_at, last_stopped_at, last_exit_code,
                       reject_zero, COALESCE(custody, 'local'), signing_wallet_id
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
                "reject_zero": bool(row[14]) if row[14] is not None else False,
                "custody": row[15],
                "signing_wallet_id": row[16],
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
        # None = preserve the row's existing flag on redeploy; False/True = set it explicitly.
        # New workers default to False (coerced at the WorkerSpec create sites below).
        reject_zero: bool | None = None,
        custody: CustodyMode = "local",
    ) -> DeployResult:
        artifact = Path(artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact}")
        self._validate_artifact_for_deploy(artifact)

        if mode not in {"auto", "strict"}:
            raise ValueError("mode must be 'auto' or 'strict'")
        if custody not in {"local", "managed"}:
            raise ValueError("custody must be 'local' or 'managed'")

        # Managed custody: the address is not chosen locally — the backend get-or-creates a
        # Privy wallet bound to (user, topic) and returns its address (ENGN-8646 / one-worker =
        # one-topic). address/mnemonic/identity_alias are local-custody inputs; reject them
        # loudly rather than silently dropping them — a silently-orphaned address or a leaked
        # mnemonic is an operator footgun.
        if custody == "managed":
            local_only = [
                name
                for name, value in (("address", address), ("mnemonic", mnemonic), ("identity_alias", identity_alias))
                if value is not None
            ]
            if local_only:
                raise ValueError(
                    f"custody='managed' does not accept local-custody inputs {local_only}; the "
                    "backend provisions a topic-bound wallet and assigns the address"
                )
            return self._deploy_managed_worker(topic_id, artifact, topic_desc, replace, reject_zero)

        # Explicit address path
        if address:
            existing = self._worker_exists(topic_id, address)
            if existing and replace:
                current_artifact = self._update_worker(topic_id, address, artifact, topic_desc, reject_zero=reject_zero)
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
                spec = WorkerSpec(topic_id, topic_desc, ident.address, artifact, ident.alias, reject_zero=bool(reject_zero))
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
            action: DeployAction = "reused" if self._address_has_other_topics(ident.address) else "created"
            spec = WorkerSpec(topic_id, topic_desc, ident.address, artifact, ident.alias, reject_zero=bool(reject_zero))
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
        action: DeployAction = "created" if created else "reused"
        spec = WorkerSpec(topic_id, topic_desc, ident.address, artifact, ident.alias, reject_zero=bool(reject_zero))
        self.add_worker(spec)
        return DeployResult(
            topic_id=topic_id,
            address_assigned=ident.address,
            artifact_path=str(artifact),
            action=action,
            message=f"Deployed worker for topic {topic_id} with address {ident.address}",
        )

    def _deploy_managed_worker(
        self,
        topic_id: int,
        artifact: Path,
        topic_desc: str | None,
        replace: bool,
        reject_zero: bool | None,
    ) -> DeployResult:
        """Provision (idempotent get-or-create) a managed Privy wallet bound to ``topic_id`` and
        register a managed worker against its backend-assigned address. One wallet per topic, so a
        re-deploy targets the same wallet rather than allocating a new one.

        Redeploys are hash-aware (synth-009): the new artifact's SHA-256 is compared against the
        active deployment's recorded hash.

        * identical hash, ``replace=False`` -> ``reused``: the running deployment already serves
          byte-identical artifact, so the artifact is left in place (no rotation). The worker-row
          metadata (reject_zero + the freshly-provisioned signing_wallet_id, and topic_desc) is
          still re-synced so an idempotent re-run cannot leave the row pointing at a stale flag or
          wallet binding.
        * ``replace=True`` -> ``replaced``: the caller explicitly asked to rotate, so rotate the
          artifact on the one-per-topic wallet even when it is byte-identical.
        * different (or unknown) hash with ``replace=False`` -> ``ValueError``: a different
          artifact is never silently swapped onto a running deployment, even in auto mode.

        A legacy active deployment with no recorded hash is treated conservatively as "unknown":
        without ``replace=True`` we cannot prove the artifact is unchanged, so we refuse rather
        than risk overwriting a different running deployment.
        """
        client = self._forge_client()
        # Prefer a resolved topic name (same source the rest of the registry uses) over the bare
        # topic_id fallback so the backend wallet label is human-meaningful.
        label = self._resolve_topic_desc(topic_id, topic_desc) or f"worker-topic-{topic_id}"
        info = client.provision_wallet(topic_id, label=label)
        if not getattr(info, "address", None) or not getattr(info, "id", None):
            raise RuntimeError(
                f"Forge backend returned a malformed wallet for topic {topic_id}: {info!r}"
            )
        address = info.address

        if self._worker_exists(topic_id, address):
            new_hash = self._artifact_sha256(artifact)
            active_hash = self._get_active_deployment_hash(topic_id, address)
            identical = active_hash is not None and active_hash == new_hash

            # Idempotent re-run: identical artifact AND no explicit replace. Leave the running
            # artifact in place (no rotation) but still re-sync the worker-row metadata
            # (reject_zero + the freshly-provisioned signing_wallet_id, and topic_desc) so the row
            # never drifts from the fresh provision. An explicit replace=True is honored instead
            # and falls through to the rotate path below — the caller asked to rotate.
            if identical and not replace:
                self._sync_worker_metadata(
                    topic_id, address, topic_desc, reject_zero=reject_zero, signing_wallet_id=info.id
                )
                return DeployResult(
                    topic_id=topic_id,
                    address_assigned=address,
                    artifact_path=str(artifact),
                    action="reused",
                    message=f"Reused managed worker for topic {topic_id} (wallet {address}); artifact unchanged, metadata re-synced",
                )

            # synth-009: a different artifact — or a legacy active deployment with no recorded hash,
            # treated conservatively as unknown — must not silently overwrite the running deployment.
            # Require an explicit replace=True, even in auto mode.
            if not replace:
                raise ValueError(
                    f"Managed worker for topic {topic_id} (wallet {address}) already has an active "
                    "deployment with a different (or unknown) artifact; pass replace=True to rotate it"
                )

            # Explicit replace (or a genuinely different artifact): rotate the artifact on the
            # one-per-topic wallet. Re-sync reject_zero and the freshly-provisioned wallet id so a
            # redeploy cannot leave the row pointing at a stale flag or wallet binding.
            current_artifact = self._update_worker(
                topic_id, address, artifact, topic_desc, reject_zero=reject_zero, signing_wallet_id=info.id
            )
            deployment_id = self._rotate_deployment(topic_id, address, current_artifact)
            self._monitor_register(topic_id, address, deployment_id=deployment_id)
            return DeployResult(
                topic_id=topic_id,
                address_assigned=address,
                artifact_path=str(artifact),
                action="replaced",
                message=f"Replaced managed worker artifact for topic {topic_id} (wallet {address})",
            )

        spec = WorkerSpec(
            topic_id,
            topic_desc,
            address,
            artifact,
            # identity_ref is a local-identity alias keying the identities table for local custody;
            # managed workers have no identities row, so use a sentinel rather than overloading it
            # with the Privy wallet UUID. signing_wallet_id stays the canonical wallet identifier.
            "managed",
            reject_zero=bool(reject_zero),
            custody="managed",
            signing_wallet_id=info.id,
        )
        self.add_worker(spec)
        return DeployResult(
            topic_id=topic_id,
            address_assigned=address,
            artifact_path=str(artifact),
            action="created",
            message=f"Provisioned managed wallet {address} for topic {topic_id}",
        )

    # ----------------------------
    # Lifecycle (persistent managed process runner)
    # ----------------------------
    @staticmethod
    def _allora_api_key_present() -> bool:
        """True when a non-empty ALLORA_API_KEY is resolvable, mirroring worker_runtime._load_api_key.

        Checks the environment first, then the ``.allora_api_key`` file fallbacks the runtime reads.
        A file that exists but is empty or whitespace-only is treated as absent: the subprocess
        would resolve it to an empty key and fail at runtime, so accepting it here would defeat the
        fail-before-running guarantee this precheck exists to provide.
        """
        if os.environ.get("ALLORA_API_KEY", "").strip():
            return True
        for p in ("notebooks/.allora_api_key", ".allora_api_key"):
            try:
                if Path(p).read_text().strip():
                    return True
            except OSError:
                continue
        return False

    def _build_run_command(self, topic_id: int, address: str, status: dict[str, Any]) -> tuple[list[str], Optional[dict[str, str]]]:
        """Build the ``worker_runtime`` argv (and subprocess env) for a worker slot.

        Local custody passes the on-disk key file via ``--mnemonic-file``. Managed custody passes
        ``--custody managed`` and injects the Forge credentials into the env so the SDK provisions
        and signs against the backend with no local key. Returns ``(argv, env)``; ``env`` is None
        for local custody (inherit the parent environment unchanged).
        """
        cmd = [
            sys.executable,
            "-m",
            "allora_forge_builder_kit.worker_runtime",
            "--topic",
            str(topic_id),
            "--artifact",
            str(status["artifact_path"]),
        ]
        env: Optional[dict[str, str]] = None
        # ALLORA_API_KEY is required by the worker_runtime subprocess (faucet drips + topic
        # queries). Pre-check it for both custody modes so a missing key fails loudly here —
        # surfaced by start_worker/reconcile — instead of the subprocess raising
        # "ALLORA_API_KEY not found" right after the DB row is marked 'running', defeating the
        # same fail-before-running guarantee the Forge-credential prechecks provide.
        if not self._allora_api_key_present():
            raise ValueError(
                f"worker topic={topic_id} address={address} requires ALLORA_API_KEY; set it in the "
                "environment or create a .allora_api_key file before starting the worker"
            )
        if status.get("custody") == "managed":
            # Precheck the managed credentials before spawning so a misconfigured worker fails
            # loudly here (start_worker/reconcile surface this) instead of the subprocess exiting
            # right after the DB row is marked 'running'.
            if not self._forge_api_key or not self._forge_backend_url:
                raise ValueError(
                    f"managed worker topic={topic_id} address={address} requires a Forge API key "
                    "and backend URL; set $FORGE_API_KEY and $FORGE_BACKEND_URL"
                )
            signing_wallet_id = status.get("signing_wallet_id")
            if not signing_wallet_id:
                raise ValueError(
                    f"managed worker topic={topic_id} address={address} is missing signing_wallet_id; "
                    "re-deploy with custody='managed' to provision the backend wallet"
                )
            cmd.extend(["--custody", "managed"])
            env = os.environ.copy()
            # Strip inherited local-key env vars before injecting the Forge credentials: the
            # sibling SDK's AlloraWalletConfig.from_env() hard-raises when FORGE_API_KEY +
            # FORGE_SIGNING_WALLET_ID coexist with any of these, so a parent shell / systemd unit
            # that still exports a local key would otherwise crash every managed worker at startup.
            for _local_key_var in ("PRIVATE_KEY", "MNEMONIC", "MNEMONIC_FILE"):
                env.pop(_local_key_var, None)
            env["FORGE_API_KEY"] = self._forge_api_key
            env["FORGE_BACKEND_URL"] = self._forge_backend_url
            # Pin the exact provisioned wallet (the DB row already stores it) so the worker signs
            # with that wallet deterministically instead of re-deriving one via topic get-or-create.
            env["FORGE_SIGNING_WALLET_ID"] = signing_wallet_id
        else:
            key_file = self._get_key_file_for_address(address)
            if not key_file:
                raise FileNotFoundError(
                    f"No key file found for address {address}. "
                    f"Check worker_secrets.json and worker_keys/ directory."
                )
            cmd.extend(["--mnemonic-file", str(key_file)])
        cmd.extend(["--network", self._network])
        if self._no_faucet:
            cmd.append("--no-faucet")
        if status.get("reject_zero"):
            cmd.append("--reject-zero")
        return cmd, env

    def start_worker(self, topic_id: int, address: str) -> None:
        status = self.status_worker(topic_id, address)
        pid = status.get("last_pid")
        if pid and self._is_pid_alive(pid):
            self._set_worker_status(topic_id, address, status="running", last_error=None)
            return

        # Build (and validate) the launch command before opening the log file. A managed worker
        # that fails the credential/wallet precheck must not leave an empty
        # worker_<topic>_<addr>.log behind — otherwise every reconcile over a persistent
        # misconfiguration re-touches a silent empty file with no forensic value.
        cmd, env = self._build_run_command(topic_id, address, status)

        log_path = self.runtime_log_dir / f"worker_{topic_id}_{address}.log"
        # Managed-custody workers carry FORGE_API_KEY in their env, so if the worker (or a
        # transitive dependency) ever echoes it the secret lands in this log file. Create the
        # log owner-only (0600) so it is not world-readable on a shared host, matching the
        # secrets-file permissions. os.open's mode only applies on creation, so also chmod an
        # already-existing log (best-effort) to tighten logs written before this change.
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        log_f = os.fdopen(log_fd, "ab")
        try:
            os.chmod(log_path, 0o600)
        except OSError:
            pass  # tightening a pre-existing log must not block worker start
        try:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(Path.cwd()), env=env)
        except Exception:
            log_f.close()
            raise
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
        failed: list[dict] = []
        for row in self.status_all():
            pid = row.get("last_pid")
            alive = bool(pid and self._is_pid_alive(pid))
            need_start = False
            if row["enabled"] and (row["status"] == "running" or row.get("last_pid")):
                if alive:
                    running += 1
                else:
                    need_start = True
            elif row["enabled"] and row["status"] == "stopped":
                need_start = True
            if need_start:
                try:
                    self.start_worker(row["topic_id"], row["address"])
                    restarted += 1
                except Exception as exc:
                    logger.error(
                        "reconcile: failed to start worker topic=%s addr=%s: %s",
                        row["topic_id"], row["address"], exc,
                    )
                    failed.append({
                        "topic_id": row["topic_id"],
                        "address": row["address"],
                        "error": str(exc),
                    })
        return {"running": running, "restarted": restarted, "failed": failed}

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
            conn.execute("PRAGMA journal_mode=WAL")
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
                    reject_zero INTEGER NOT NULL DEFAULT 0,
                    custody TEXT NOT NULL DEFAULT 'local',
                    signing_wallet_id TEXT,
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
            if "reject_zero" not in cols:
                conn.execute("ALTER TABLE workers ADD COLUMN reject_zero INTEGER NOT NULL DEFAULT 0")
            if "custody" not in cols:
                conn.execute("ALTER TABLE workers ADD COLUMN custody TEXT NOT NULL DEFAULT 'local'")
            if "signing_wallet_id" not in cols:
                conn.execute("ALTER TABLE workers ADD COLUMN signing_wallet_id TEXT")
            conn.commit()
        if not self.secrets_path.exists():
            self.secrets_path.write_text("{}")
            try:
                self.secrets_path.chmod(0o600)
            except PermissionError:
                pass

    def _insert_identity(self, alias: str, address: str, key_file: Path) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO identities(alias, address) VALUES(?, ?)", (alias, address))
            conn.commit()
        with self._secrets_file_lock():
            data = self._load_secrets()
            data[alias] = {"address": address, "key_file": str(key_file)}
            self._save_secrets(data)

    @staticmethod
    def _sanitize_alias(alias: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", alias)
        return safe[:128] or "unnamed"

    def _persist_key_file(self, alias: str, mnemonic: str | None = None, source_file: str | Path | None = None) -> Path:
        """Write or copy a mnemonic into worker_keys/<alias>.key with mode 0o600."""
        safe = self._sanitize_alias(alias)
        dest = self.key_dir / f"{safe}.key"
        if dest.exists():
            dest = self.key_dir / f"{safe}_{uuid.uuid4().hex[:8]}.key"
        if source_file:
            shutil.copy2(str(source_file), str(dest))
            try:
                dest.chmod(0o600)
            except OSError as exc:
                logger.warning(
                    "Could not set permissions on %s: %s", dest, exc
                )
        elif mnemonic:
            fd = os.open(str(dest), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            try:
                os.write(fd, mnemonic.encode())
            finally:
                os.close(fd)
        else:
            raise ValueError("Either mnemonic or source_file must be provided")
        return dest

    def _get_key_file_for_address(self, address: str) -> Optional[Path]:
        secrets = self._load_secrets()
        for entry in secrets.values():
            if isinstance(entry, dict) and entry.get("address") == address:
                kf = entry.get("key_file")
                if kf:
                    p = Path(kf)
                    if p.exists():
                        return p
        return None

    def _secrets_file_lock(self):
        """Cross-process advisory lock for worker_secrets.json read-modify-write."""
        import contextlib
        import fcntl

        @contextlib.contextmanager
        def _lock():
            lock_path = str(self.secrets_path) + ".lock"
            lf = open(lock_path, "w")
            try:
                fcntl.flock(lf, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
                lf.close()

        return _lock()

    def _load_secrets(self) -> dict:
        try:
            return json.loads(self.secrets_path.read_text())
        except Exception:
            return {}

    def _save_secrets(self, data: dict) -> None:
        content = json.dumps(data, indent=2).encode()
        fd = os.open(str(self.secrets_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            os.write(fd, content)
        finally:
            os.close(fd)

    @staticmethod
    def _default_identity_creator() -> tuple[str, str, str]:
        from cosmpy.mnemonic import generate_mnemonic
        from cosmpy.aerial.wallet import LocalWallet

        mnemonic = generate_mnemonic()
        wallet = LocalWallet.from_mnemonic(mnemonic, "allo")
        address = str(wallet.address())
        alias = f"identity_{uuid.uuid4().hex[:12]}"
        return (alias, address, mnemonic)

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

    def _get_custody(self, topic_id: int, address: str) -> tuple[CustodyMode, str | None]:
        """Return ``(custody, signing_wallet_id)`` for a worker; ``("local", None)`` if absent."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT custody, signing_wallet_id FROM workers WHERE topic_id=? AND address=?",
                (topic_id, address),
            ).fetchone()
        if not row:
            return ("local", None)
        return (row[0] or "local", row[1])

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

    def _update_worker(
        self,
        topic_id: int,
        address: str,
        artifact_path: Path,
        topic_desc: str | None,
        reject_zero: Optional[bool] = None,
        signing_wallet_id: Optional[str] = None,
    ) -> Path:
        """Re-materialize the artifact, update the worker row, and return the new artifact path.

        ``reject_zero`` and ``signing_wallet_id`` are written only when provided (non-None), so a
        redeploy can re-sync flags that would otherwise drift while leaving them untouched when the
        caller does not supply them. Returning the materialized path lets callers skip a redundant
        status round-trip before rotating the deployment.
        """
        resolved_desc = self._resolve_topic_desc(topic_id, topic_desc)
        managed_artifact = self._materialize_artifact(topic_id, address, artifact_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE workers
                   SET artifact_path=?,
                       topic_desc=COALESCE(?, topic_desc),
                       reject_zero=COALESCE(?, reject_zero),
                       signing_wallet_id=COALESCE(?, signing_wallet_id),
                       updated_at=CURRENT_TIMESTAMP
                 WHERE topic_id=? AND address=?
                """,
                (
                    str(managed_artifact),
                    resolved_desc,
                    None if reject_zero is None else (1 if reject_zero else 0),
                    signing_wallet_id,
                    topic_id,
                    address,
                ),
            )
            conn.commit()
        return managed_artifact

    def _sync_worker_metadata(
        self,
        topic_id: int,
        address: str,
        topic_desc: str | None = None,
        reject_zero: Optional[bool] = None,
        signing_wallet_id: Optional[str] = None,
    ) -> None:
        """Re-sync a worker row's mutable metadata in place, without rotating its artifact.

        The hash-identical managed redeploy path uses this: the running artifact is byte-identical
        so it is left untouched, but reject_zero / signing_wallet_id (and topic_desc) are still
        refreshed from the fresh provision so an idempotent re-run cannot leave the row pointing at
        a stale flag or wallet binding. Each field is written only when provided (non-None), via
        COALESCE — mirroring the metadata half of :meth:`_update_worker` minus the artifact swap.
        """
        resolved_desc = self._resolve_topic_desc(topic_id, topic_desc)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE workers
                   SET topic_desc=COALESCE(?, topic_desc),
                       reject_zero=COALESCE(?, reject_zero),
                       signing_wallet_id=COALESCE(?, signing_wallet_id),
                       updated_at=CURRENT_TIMESTAMP
                 WHERE topic_id=? AND address=?
                """,
                (
                    resolved_desc,
                    None if reject_zero is None else (1 if reject_zero else 0),
                    signing_wallet_id,
                    topic_id,
                    address,
                ),
            )
            conn.commit()

    def _validate_artifact_for_deploy(self, artifact_path: Path) -> None:
        """Block known-bad artifact variants from deployment.

        Guardrail: old pickles that embed `load_raw` for live price lookup are
        not deploy-safe in managed worker runtime.
        """
        try:
            blob = artifact_path.read_bytes()
        except Exception:
            return

        if b"load_raw" in blob and b"Could not get current price from raw data" in blob:
            raise ValueError(
                f"Refusing to deploy artifact with raw-data inference path: {artifact_path}. "
                "Use export_predict_self_contained.py."
            )

    @staticmethod
    def _artifact_sha256(artifact_path: Path) -> str:
        """Return the SHA-256 hex digest of an artifact's bytes.

        Recorded on every deployment so a managed redeploy can tell an idempotent re-run (identical
        bytes) from a genuinely different model, and so that comparison survives a process restart.
        """
        return hashlib.sha256(artifact_path.read_bytes()).hexdigest()

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
        artifact_hash = self._artifact_sha256(artifact_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE worker_deployments SET is_active=0, ended_at=CURRENT_TIMESTAMP WHERE topic_id=? AND address=? AND is_active=1",
                (topic_id, address),
            )
            conn.execute(
                """
                INSERT INTO worker_deployments(deployment_id, topic_id, address, artifact_path, artifact_hash, deployed_at, is_active)
                VALUES(?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 1)
                """,
                (deployment_id, topic_id, address, str(artifact_path), artifact_hash),
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

    def _get_active_deployment_hash(self, topic_id: int, address: str) -> Optional[str]:
        """Return the active deployment's recorded artifact SHA-256, or None when there is no
        active deployment or it predates hash tracking (legacy NULL row)."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT artifact_hash FROM worker_deployments WHERE topic_id=? AND address=? AND is_active=1 ORDER BY deployed_at DESC LIMIT 1",
                (topic_id, address),
            ).fetchone()
        return row[0] if row and row[0] else None

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
            conn.execute(MONITOR_TARGETS_DDL)
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
