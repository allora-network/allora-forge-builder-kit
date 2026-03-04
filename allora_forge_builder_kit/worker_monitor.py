from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class MonitorTarget:
    topic_id: int
    address: str
    deployed_at: str
    deployment_id: Optional[str] = None
    enabled: bool = True


class WorkerMonitor:
    """Lightweight on-chain monitoring registry for (topic_id, address) workers.

    The monitor is intentionally modular and fetcher-driven:
    - storage + aggregation is local (SQLite)
    - network retrieval is pluggable via `event_fetcher`
    """

    def __init__(
        self,
        db_path: str | Path = "worker_state.db",
        event_fetcher: Optional[Callable[[int, str, Optional[str]], list[dict]]] = None,
    ):
        self.db_path = Path(db_path)
        self._event_fetcher = event_fetcher or self._default_event_fetcher
        self._init_db()

    # ----------------------------
    # Target management
    # ----------------------------
    def register_target(self, topic_id: int, address: str, deployed_at: Optional[str] = None, deployment_id: Optional[str] = None) -> MonitorTarget:
        deployed_at = deployed_at or datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO monitor_targets(topic_id, address, deployed_at, deployment_id, enabled, last_sync_at)
                VALUES(?, ?, ?, ?, 1, NULL)
                """,
                (topic_id, address, deployed_at, deployment_id),
            )
            conn.commit()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE monitor_targets SET deployment_id=COALESCE(?, deployment_id) WHERE topic_id=? AND address=?",
                (deployment_id, topic_id, address),
            )
            conn.commit()
        return MonitorTarget(topic_id=topic_id, address=address, deployed_at=deployed_at, deployment_id=deployment_id)

    def list_targets(self, enabled_only: bool = True) -> list[MonitorTarget]:
        query = "SELECT topic_id, address, deployed_at, deployment_id, enabled FROM monitor_targets"
        params: tuple = ()
        if enabled_only:
            query += " WHERE enabled=1"
        query += " ORDER BY topic_id, address"

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [MonitorTarget(topic_id=r[0], address=r[1], deployed_at=r[2], deployment_id=r[3], enabled=bool(r[4])) for r in rows]

    def set_target_enabled(self, topic_id: int, address: str, enabled: bool) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE monitor_targets SET enabled=? WHERE topic_id=? AND address=?",
                (1 if enabled else 0, topic_id, address),
            )
            conn.commit()

    def set_target_deployment(self, topic_id: int, address: str, deployment_id: str, deployed_at: Optional[str] = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            if deployed_at:
                conn.execute(
                    "UPDATE monitor_targets SET deployment_id=?, deployed_at=? WHERE topic_id=? AND address=?",
                    (deployment_id, deployed_at, topic_id, address),
                )
            else:
                conn.execute(
                    "UPDATE monitor_targets SET deployment_id=? WHERE topic_id=? AND address=?",
                    (deployment_id, topic_id, address),
                )
            conn.commit()

    # ----------------------------
    # Sync / backfill
    # ----------------------------
    def backfill_target(self, topic_id: int, address: str, since: Optional[str] = None) -> dict:
        return self._sync_target(topic_id=topic_id, address=address, since=since)

    def sync_once(self) -> dict:
        targets = self.list_targets(enabled_only=True)
        inserted = 0
        errors: list[dict] = []
        for t in targets:
            try:
                out = self._sync_target(topic_id=t.topic_id, address=t.address, since=None)
                inserted += out["inserted"]
            except Exception as e:
                errors.append({
                    "topic_id": t.topic_id,
                    "address": t.address,
                    "error": str(e),
                })
        return {"targets": len(targets), "inserted": inserted, "errors": errors}

    # ----------------------------
    # Read APIs
    # ----------------------------
    def get_summary(self, topic_id: int, address: str, deployment_id: Optional[str] = None) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            target = conn.execute(
                """
                SELECT topic_id, address, deployed_at, deployment_id, enabled, last_sync_at
                FROM monitor_targets
                WHERE topic_id=? AND address=?
                """,
                (topic_id, address),
            ).fetchone()
            if not target:
                raise KeyError(f"Monitor target not found: topic={topic_id} address={address}")

            where = "topic_id=? AND address=?"
            params: list = [topic_id, address]
            dep = deployment_id or target[3]
            if dep:
                where += " AND deployment_id=?"
                params.append(dep)

            counts = conn.execute(
                f"""
                SELECT
                    COUNT(1),
                    SUM(CASE WHEN event_type='inference' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN event_type='reward' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN event_type='score' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN event_type='submission' AND status='success' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN event_type='submission' AND status='error' THEN 1 ELSE 0 END),
                    COALESCE(SUM(CASE WHEN event_type='reward' THEN value_num ELSE 0 END), 0.0)
                FROM monitor_events
                WHERE {where}
                """,
                tuple(params),
            ).fetchone()

            # Prefer confirmed inference events over snapshot-only rows (some chain
            # snapshot endpoints can transiently return 0.0 and mask a recent success).
            last_inference = conn.execute(
                f"""
                SELECT observed_at, value_text, value_num, tx_hash
                FROM monitor_events
                WHERE {where} AND event_type='inference' AND (
                    status='success' OR tx_hash IS NOT NULL OR (value_num IS NOT NULL AND value_num != 0)
                )
                ORDER BY observed_at DESC
                LIMIT 1
                """,
                tuple(params),
            ).fetchone()
            if not last_inference:
                last_inference = conn.execute(
                    f"""
                    SELECT observed_at, value_text, value_num, tx_hash
                    FROM monitor_events
                    WHERE {where} AND event_type='inference'
                    ORDER BY observed_at DESC
                    LIMIT 1
                    """,
                    tuple(params),
                ).fetchone()

        return {
            "topic_id": target[0],
            "address": target[1],
            "deployed_at": target[2],
            "deployment_id": deployment_id or target[3],
            "enabled": bool(target[4]),
            "last_sync_at": target[5],
            "events_total": counts[0] or 0,
            "inference_count": counts[1] or 0,
            "reward_count": counts[2] or 0,
            "score_count": counts[3] or 0,
            "submission_success": counts[4] or 0,
            "submission_error": counts[5] or 0,
            "rewards_total": float(counts[6] or 0.0),
            "last_inference": None
            if not last_inference
            else {
                "observed_at": last_inference[0],
                "value_text": last_inference[1],
                "value_num": last_inference[2],
                "tx_hash": last_inference[3],
            },
        }

    def get_timeseries(self, topic_id: int, address: str, event_type: Optional[str] = None, deployment_id: Optional[str] = None, limit: int = 200) -> list[dict]:
        query = """
            SELECT observed_at, event_type, status, value_num, value_text, tx_hash, details_json, deployment_id
            FROM monitor_events
            WHERE topic_id=? AND address=?
        """
        params: list = [topic_id, address]
        if deployment_id:
            query += " AND deployment_id=?"
            params.append(deployment_id)
        if event_type:
            query += " AND event_type=?"
            params.append(event_type)
        query += " ORDER BY observed_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()

        return [
            {
                "observed_at": r[0],
                "event_type": r[1],
                "status": r[2],
                "value_num": r[3],
                "value_text": r[4],
                "tx_hash": r[5],
                "details_json": r[6],
                "deployment_id": r[7],
            }
            for r in rows
        ]

    # ----------------------------
    # Internals
    # ----------------------------
    def _sync_target(self, topic_id: int, address: str, since: Optional[str]) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT deployed_at, deployment_id, last_sync_at FROM monitor_targets WHERE topic_id=? AND address=?",
                (topic_id, address),
            ).fetchone()
        if not row:
            raise KeyError(f"Monitor target not registered: topic={topic_id} address={address}")

        deployed_at, deployment_id, last_sync_at = row
        cursor_since = since or last_sync_at or deployed_at

        events = self._event_fetcher(topic_id, address, cursor_since)
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            for ev in events:
                unique_key = ev.get("event_id") or self._event_fingerprint(topic_id, address, ev)
                try:
                    conn.execute(
                        """
                        INSERT INTO monitor_events(
                            topic_id, address, deployment_id, event_id, event_type, status,
                            value_num, value_text, tx_hash, observed_at, details_json
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            topic_id,
                            address,
                            deployment_id,
                            unique_key,
                            ev.get("event_type", "unknown"),
                            ev.get("status"),
                            ev.get("value_num"),
                            ev.get("value_text"),
                            ev.get("tx_hash"),
                            ev.get("observed_at") or datetime.now(timezone.utc).isoformat(),
                            ev.get("details_json"),
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass

            conn.execute(
                """
                UPDATE monitor_targets
                SET last_sync_at=?
                WHERE topic_id=? AND address=?
                """,
                (datetime.now(timezone.utc).isoformat(), topic_id, address),
            )
            conn.commit()

        return {"topic_id": topic_id, "address": address, "fetched": len(events), "inserted": inserted}

    @staticmethod
    def _event_fingerprint(topic_id: int, address: str, ev: dict) -> str:
        return "|".join(
            [
                str(topic_id),
                address,
                str(ev.get("event_type", "unknown")),
                str(ev.get("tx_hash", "")),
                str(ev.get("observed_at", "")),
                str(ev.get("value_text", ev.get("value_num", ""))),
            ]
        )

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS monitor_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id INTEGER NOT NULL,
                    address TEXT NOT NULL,
                    deployment_id TEXT,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT,
                    value_num REAL,
                    value_text TEXT,
                    tx_hash TEXT,
                    observed_at TEXT NOT NULL,
                    details_json TEXT,
                    UNIQUE(topic_id, address, event_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_monitor_events_lookup ON monitor_events(topic_id, address, observed_at)")
            # lightweight migrations
            tcols = [r[1] for r in conn.execute("PRAGMA table_info(monitor_targets)").fetchall()]
            if "deployment_id" not in tcols:
                conn.execute("ALTER TABLE monitor_targets ADD COLUMN deployment_id TEXT")
            ecols = [r[1] for r in conn.execute("PRAGMA table_info(monitor_events)").fetchall()]
            if "deployment_id" not in ecols:
                conn.execute("ALTER TABLE monitor_events ADD COLUMN deployment_id TEXT")
            conn.commit()

    @staticmethod
    def _default_event_fetcher(topic_id: int, address: str, since: Optional[str]) -> list[dict]:
        # SDK/plumbing integration should be injected from caller to keep this module lightweight.
        return []


class AlloraSDKEventFetcher:
    """SDK-backed event fetcher for WorkerMonitor.

    Captures a practical lightweight set of signals for a (topic, address):
    - submission/inference tx events (recent + paginated)
    - latest inference value
    - inferer score EMA
    - previous inference reward fraction
    - whitelist/can-submit checks
    """

    def __init__(self, network: str = "testnet", max_pages: int = 5, page_limit: int = 50):
        from allora_sdk.rpc_client.client import AlloraNetworkConfig, AlloraRPCClient

        cfg = AlloraNetworkConfig.mainnet() if network.lower() == "mainnet" else AlloraNetworkConfig.testnet()
        self.client = AlloraRPCClient(network=cfg)
        self.max_pages = max_pages
        self.page_limit = page_limit

    def __call__(self, topic_id: int, address: str, since: Optional[str]) -> list[dict]:
        from allora_sdk.protos.cosmos.tx.v1beta1 import GetTxsEventRequest, OrderBy
        from allora_sdk.protos.emissions.v9 import (
            CanSubmitWorkerPayloadRequest,
            GetInfererScoreEmaRequest,
            GetPreviousInferenceRewardFractionRequest,
            GetWorkerLatestInferenceByTopicIdRequest,
            IsWhitelistedTopicWorkerRequest,
        )

        since_dt = _parse_dt(since)
        out: list[dict] = []

        # 1) Historical tx events for this sender (submission/inference)
        query = f"message.sender='{address}'"
        for page in range(1, self.max_pages + 1):
            txs = self.client.tx.get_txs_event(
                GetTxsEventRequest(
                    query=query,
                    order_by=OrderBy.DESC,
                    page=page,
                    limit=self.page_limit,
                )
            )
            tx_responses = list(getattr(txs, "tx_responses", []) or [])
            if not tx_responses:
                break

            stop_paging = False
            for tx in tx_responses:
                observed = _parse_dt(getattr(tx, "timestamp", None))
                if since_dt and observed and observed < since_dt:
                    stop_paging = True
                    continue

                for ev in getattr(tx, "events", []):
                    if ev.type != "emissions.v9.EventInsertInfererPayload":
                        continue
                    attrs = {a.key: a.value for a in ev.attributes}
                    ev_topic = str(attrs.get("topic_id", "")).strip('"')
                    if ev_topic != str(topic_id):
                        continue

                    nonce = str(attrs.get("nonce", "")).strip('"')
                    value_text = str(attrs.get("value", "")).strip('"')
                    out.append(
                        {
                            "event_id": f"submit:{tx.txhash}:{nonce}",
                            "event_type": "submission",
                            "status": "success" if not getattr(tx, "code", 0) else "error",
                            "tx_hash": tx.txhash,
                            "observed_at": getattr(tx, "timestamp", None),
                            "value_text": value_text,
                            "details_json": f'{{"nonce":"{nonce}","code":{getattr(tx, "code", 0)}}}',
                        }
                    )
                    out.append(
                        {
                            "event_id": f"inference:{tx.txhash}:{nonce}",
                            "event_type": "inference",
                            "status": "success" if not getattr(tx, "code", 0) else "error",
                            "tx_hash": tx.txhash,
                            "observed_at": getattr(tx, "timestamp", None),
                            "value_text": value_text,
                            "value_num": _to_float(value_text),
                            "details_json": f'{{"nonce":"{nonce}"}}',
                        }
                    )
            if stop_paging:
                break

        # 2) Latest point-in-time signals
        try:
            latest = self.client.emissions.query.get_worker_latest_inference_by_topic_id(
                GetWorkerLatestInferenceByTopicIdRequest(topic_id=topic_id, worker_address=address)
            )
            li = getattr(latest, "latest_inference", None)
            if li:
                out.append(
                    {
                        "event_id": f"latest_inference:{getattr(li, 'block_height', '')}:{address}",
                        "event_type": "inference",
                        "status": "snapshot",
                        "value_text": str(getattr(li, "value", "")),
                        "value_num": _to_float(getattr(li, "value", None)),
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                        "details_json": f'{{"block_height":{getattr(li, "block_height", 0)}}}',
                    }
                )
        except Exception:
            pass

        try:
            s = self.client.emissions.query.get_inferer_score_ema(
                GetInfererScoreEmaRequest(topic_id=topic_id, inferer=address)
            )
            score = getattr(getattr(s, "score", None), "score", None)
            if score is not None:
                out.append(
                    {
                        "event_id": f"score_ema:{topic_id}:{address}",
                        "event_type": "score",
                        "status": "snapshot",
                        "value_text": str(score),
                        "value_num": _to_float(score),
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
        except Exception:
            pass

        try:
            r = self.client.emissions.query.get_previous_inference_reward_fraction(
                GetPreviousInferenceRewardFractionRequest(topic_id=topic_id, worker=address)
            )
            rf = getattr(r, "reward_fraction", None)
            if rf is not None:
                out.append(
                    {
                        "event_id": f"reward_fraction:{topic_id}:{address}",
                        "event_type": "reward",
                        "status": "snapshot",
                        "value_text": str(rf),
                        "value_num": _to_float(rf),
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                        "details_json": f'{{"not_found":{str(getattr(r, "not_found", False)).lower()}}}',
                    }
                )
        except Exception:
            pass

        try:
            w = self.client.emissions.query.is_whitelisted_topic_worker(
                IsWhitelistedTopicWorkerRequest(topic_id=topic_id, address=address)
            )
            out.append(
                {
                    "event_id": f"whitelist:{topic_id}:{address}",
                    "event_type": "whitelist",
                    "status": "snapshot",
                    "value_text": str(bool(getattr(w, "is_whitelisted", False))).lower(),
                    "value_num": 1.0 if bool(getattr(w, "is_whitelisted", False)) else 0.0,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception:
            pass

        try:
            c = self.client.emissions.query.can_submit_worker_payload(
                CanSubmitWorkerPayloadRequest(topic_id=topic_id, address=address)
            )
            out.append(
                {
                    "event_id": f"can_submit:{topic_id}:{address}",
                    "event_type": "eligibility",
                    "status": "snapshot",
                    "value_text": str(bool(getattr(c, "can_submit", False))).lower(),
                    "value_num": 1.0 if bool(getattr(c, "can_submit", False)) else 0.0,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception:
            pass

        return out


def _to_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None
