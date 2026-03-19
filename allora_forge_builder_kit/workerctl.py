from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .worker_manager import WorkerManager
from .worker_monitor import WorkerMonitor, AlloraSDKEventFetcher


def _make_manager(
    with_monitor: bool,
    db_path: str = "worker_state.db",
    secrets_path: str = "worker_secrets.json",
    network: str = "testnet",
) -> tuple[WorkerManager, WorkerMonitor | None]:
    monitor = None
    if with_monitor:
        monitor = WorkerMonitor(
            db_path=db_path,
            event_fetcher=AlloraSDKEventFetcher(network=network, max_pages=2, page_limit=25),
        )
    wm = WorkerManager(
        db_path=db_path,
        secrets_path=secrets_path,
        network=network,
        monitor=monitor,
        reconcile_on_start=False,
    )
    wm.reconcile()
    return wm, monitor


def cmd_dashboard(with_monitor: bool, running_only: bool, **mgr_kwargs) -> None:
    wm, monitor = _make_manager(with_monitor=with_monitor, **mgr_kwargs)
    if monitor:
        monitor.sync_once()

    rows = wm.status_all()
    if running_only:
        rows = [r for r in rows if r["status"] == "running"]

    print(f"WORKER DASHBOARD @ {datetime.now(timezone.utc).isoformat()}")
    print(f"workers={len(rows)}")
    for r in rows:
        print("---")
        print(f"topic={r['topic_id']} | {r.get('topic_desc','')}")
        print(f"address={r['address']}")
        print(f"status={r['status']} pid={r.get('last_pid')} deployed={r.get('deployed_at')}")
        if monitor:
            try:
                s = monitor.get_summary(r["topic_id"], r["address"])
            except (KeyError, Exception):
                s = {"events_total": 0, "submission_success": 0, "submission_error": 0,
                     "inference_count": 0, "rewards_total": 0, "last_inference": None}
            li = s.get("last_inference") or {}
            print(
                f"events={s['events_total']} subs_ok={s['submission_success']} subs_err={s['submission_error']} "
                f"inferences={s['inference_count']} rewards={s['rewards_total']}"
            )
            print(f"last_inference={li.get('value_text')} at {li.get('observed_at')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker control CLI")
    parser.add_argument("--db-path", default="worker_state.db")
    parser.add_argument("--secrets-path", default="worker_secrets.json")
    parser.add_argument("--network", default="testnet")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dash = sub.add_parser("dashboard", help="Show worker dashboard")
    p_dash.add_argument("--all", action="store_true", help="Include non-running workers")
    p_dash.add_argument("--no-monitor", action="store_true", help="Skip live monitor sync")

    sub.add_parser("reconcile", help="Reconcile runtime with DB state")
    sub.add_parser("start-all", help="Start all enabled workers")
    sub.add_parser("stop-all", help="Stop running workers")

    args = parser.parse_args()
    mgr_kwargs = dict(db_path=args.db_path, secrets_path=args.secrets_path, network=args.network)

    if args.cmd == "dashboard":
        cmd_dashboard(with_monitor=not args.no_monitor, running_only=not args.all, **mgr_kwargs)
        return

    wm, _ = _make_manager(with_monitor=False, **mgr_kwargs)
    if args.cmd == "reconcile":
        print(wm.reconcile())
    elif args.cmd == "start-all":
        print(wm.start_all())
    elif args.cmd == "stop-all":
        print(wm.stop_all())


if __name__ == "__main__":
    main()
