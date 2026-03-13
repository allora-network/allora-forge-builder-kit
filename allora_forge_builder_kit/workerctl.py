from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .worker_manager import WorkerManager
from .worker_monitor import WorkerMonitor, AlloraSDKEventFetcher


def _make_manager(with_monitor: bool) -> tuple[WorkerManager, WorkerMonitor | None]:
    monitor = None
    if with_monitor:
        monitor = WorkerMonitor(
            db_path="worker_state.db",
            event_fetcher=AlloraSDKEventFetcher(network="testnet", max_pages=2, page_limit=25),
        )
    wm = WorkerManager(
        db_path="worker_state.db",
        secrets_path="worker_secrets.json",
        monitor=monitor,
        reconcile_on_start=True,
    )
    return wm, monitor


def cmd_dashboard(with_monitor: bool, running_only: bool) -> None:
    wm, monitor = _make_manager(with_monitor=with_monitor)
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
            s = monitor.get_summary(r["topic_id"], r["address"])
            li = s.get("last_inference") or {}
            print(
                f"events={s['events_total']} subs_ok={s['submission_success']} subs_err={s['submission_error']} "
                f"inferences={s['inference_count']} rewards={s['rewards_total']}"
            )
            print(f"last_inference={li.get('value_text')} at {li.get('observed_at')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker control CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dash = sub.add_parser("dashboard", help="Show worker dashboard")
    p_dash.add_argument("--all", action="store_true", help="Include non-running workers")
    p_dash.add_argument("--no-monitor", action="store_true", help="Skip live monitor sync")

    sub.add_parser("reconcile", help="Reconcile runtime with DB state")
    sub.add_parser("start-all", help="Start all enabled workers")
    sub.add_parser("stop-all", help="Stop running workers")

    args = parser.parse_args()

    if args.cmd == "dashboard":
        cmd_dashboard(with_monitor=not args.no_monitor, running_only=not args.all)
        return

    wm, _ = _make_manager(with_monitor=False)
    if args.cmd == "reconcile":
        print(wm.reconcile())
    elif args.cmd == "start-all":
        print(wm.start_all())
    elif args.cmd == "stop-all":
        print(wm.stop_all())


if __name__ == "__main__":
    main()
