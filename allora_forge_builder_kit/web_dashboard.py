from __future__ import annotations

import argparse
import json
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .worker_manager import WorkerManager
from .worker_monitor import WorkerMonitor, AlloraSDKEventFetcher


HTML = """<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <title>Allora Worker Dashboard</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; margin: 20px; }
    .muted { color: #666; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; margin: 10px 0; }
    .addr { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
    .path-row td, .logs-row td { background: #fcfcfd; }
    .path-label { font-size: 12px; font-weight: 600; color: #666; margin-right: 6px; }
    .path-short { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #444; }
    .logbox { margin-top: 2px; background: #fafafa; border: 1px solid #eee; border-radius: 8px; padding: 6px; }
    .logbox summary { cursor: pointer; font-size: 12px; color: #444; }
    .logpre { margin: 6px 0 0 0; max-height: 220px; overflow: auto; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .timeline { --slots: 25; display: grid; grid-template-columns: repeat(var(--slots), minmax(10px, 1fr)); gap: 3px; width: 100%; padding: 4px 0; }
    .timeline-axis { font-size: 11px; color: #666; margin-top: 2px; }
    .cell { width: 100%; height: clamp(12px, 1.8vw, 22px); border-radius: 3px; background: #ddd; cursor: help; }
    .cell.success { background: #1f9d55; }
    .cell.error { background: #d64545; }
    .cell.missed { background: #cfd4da; }
    .tooltip { position: fixed; z-index: 9999; display: none; max-width: 420px; background: #111; color: #fff; border-radius: 8px; padding: 8px 10px; font-size: 12px; line-height: 1.35; box-shadow: 0 6px 20px rgba(0,0,0,0.25); white-space: pre-wrap; pointer-events: none; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th, td { text-align: left; border-bottom: 1px solid #eee; padding: 6px; font-size: 14px; vertical-align: top; }
    .ok { color: #0a7; }
    .warn { color: #c75; }
  </style>
</head>
<body>
  <h2>Allora Worker Dashboard</h2>
  <div id='meta' class='muted'>Loading...</div>
  <div id='root'></div>
  <div id='tip' class='tooltip'></div>
  <script>
    const openLogPanels = new Set();

    function workerKey(grp, w) {
      return `${grp.address}::${w.topic_id}`;
    }

    function esc(s) {
      const v = (s === null || s === undefined) ? '' : String(s);
      return v.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function shortPath(p) {
      if (!p) return '';
      const parts = p.split('/');
      if (parts.length <= 3) return p;
      return '.../' + parts.slice(-3).join('/');
    }

    function fmtSig(v, sig = 5) {
      if (v === null || v === undefined || v === '') return '—';
      const n = Number(v);
      if (!Number.isFinite(n)) return String(v);
      return Number.parseFloat(n.toPrecision(sig)).toString();
    }

    function timelineHtml(tl) {
      const slots = (tl && tl.slots) ? tl.slots : [];
      const cells = slots.map((s) => {
        const sub = s.submission || {};
        const tt = [
          `nonce: ${(s.nonce === null || s.nonce === undefined) ? '—' : s.nonce}`,
          `window time: ${s.ts || '—'}`,
          `slot status: ${s.status || 'missed'}`,
          `submission at: ${sub.observed_at || '—'}`,
          `submission status: ${sub.status || '—'}`,
          `inference value: ${(sub.inference_value === null || sub.inference_value === undefined) ? '—' : sub.inference_value}`,
          `tx hash: ${sub.tx_hash || '—'}`,
          `tx code: ${(sub.code === null || sub.code === undefined) ? '—' : sub.code}`,
          `reward fraction (latest): ${(sub.reward_fraction === null || sub.reward_fraction === undefined) ? '—' : sub.reward_fraction}`,
          `score EMA (latest): ${(sub.score_ema === null || sub.score_ema === undefined) ? '—' : sub.score_ema}`
        ].join('\\n');
        return `<span class='cell ${esc(s.status || 'missed')}' data-tip='${esc(tt)}'></span>`;
      }).join('');
      const first = slots[0] || {};
      const last = slots[slots.length - 1] || {};
      const axisText = `Axis: oldest nonce ${(first.nonce === null || first.nonce === undefined) ? '—' : first.nonce} (${first.ts || '—'}) → newest nonce ${(last.nonce === null || last.nonce === undefined) ? '—' : last.nonce} (${last.ts || '—'})`;
      const axis = `<div class='timeline-axis'>${esc(axisText)}</div>`;
      const slotCount = Math.max(slots.length, 1);
      return `${axis}<div class='timeline' style='--slots:${slotCount}'>${cells}</div>`;
    }

    function placeTooltip(tip, e) {
      const pad = 12;
      const margin = 8;
      const vw = window.innerWidth || document.documentElement.clientWidth;
      const vh = window.innerHeight || document.documentElement.clientHeight;
      const tw = tip.offsetWidth || 260;
      const th = tip.offsetHeight || 80;

      let left = e.clientX + pad;
      let top = e.clientY + pad;

      if (left + tw + margin > vw) left = e.clientX - tw - pad;
      if (top + th + margin > vh) top = e.clientY - th - pad;

      left = Math.max(margin, Math.min(left, vw - tw - margin));
      top = Math.max(margin, Math.min(top, vh - th - margin));

      tip.style.left = left + 'px';
      tip.style.top = top + 'px';
    }

    function wireTooltips(root) {
      const tip = document.getElementById('tip');
      root.querySelectorAll('.cell[data-tip]').forEach((el) => {
        el.addEventListener('mouseenter', (e) => {
          tip.textContent = el.getAttribute('data-tip') || '';
          tip.style.display = 'block';
          placeTooltip(tip, e);
        });
        el.addEventListener('mousemove', (e) => {
          placeTooltip(tip, e);
        });
        el.addEventListener('mouseleave', () => {
          tip.style.display = 'none';
        });
      });
    }

    async function load() {
      try {
        const bust = Date.now();
        const r = await fetch(`/api/dashboard?tail=30&_=${bust}`, {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache'
          }
        });
        const d = await r.json();
        const sync = d.sync || {};
        const syncErrs = (sync.errors || []).length;
        document.getElementById('meta').textContent = `Updated: ${d.updated_at} | workers=${d.worker_count} | cadence=5s | sync_ok=${sync.ok} inserted=${sync.inserted || 0} errs=${syncErrs}`;
        const root = document.getElementById('root');
        root.innerHTML = '';
        for (const grp of d.by_address) {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `<div><b>Address</b> <span class='addr'>${grp.address}</span> (${grp.running}/${grp.total} running)</div>`;
        const tbl = document.createElement('table');
        tbl.innerHTML = `<thead><tr>
            <th>Topic</th><th>Status</th><th>Submissions</th><th>Inferences</th><th>Score EMA</th><th>Reward Frac</th><th>24h</th><th>7d</th><th>Last Inference</th><th>At</th>
          </tr></thead>`;
        const tb = document.createElement('tbody');
        for (const w of grp.workers) {
          const logs = (w.log_tail || []).join('\\\\n');
          const key = workerKey(grp, w);

          const trMain = document.createElement('tr');
          const p24 = (w.period_metrics || {})['24h'] || {};
          const p7 = (w.period_metrics || {})['7d'] || {};
          const scoreEma = (w.last_score || {}).value_text;
          const rewardFrac = (w.last_reward_fraction || {}).value_text;

          trMain.innerHTML = `<td>${esc(w.topic_id)} — ${esc(w.topic_desc || '')}</td>
            <td>${esc(w.status)}</td>
            <td class='ok'>${esc(w.submission_success)} ok / ${esc(w.submission_error)} err</td>
            <td>${esc(w.inference_count)}</td>
            <td>${esc(fmtSig(scoreEma))}</td>
            <td>${esc(fmtSig(rewardFrac))}</td>
            <td>${esc((p24.submission_success || 0))} ok · avgS ${esc(fmtSig(p24.score_avg))}</td>
            <td>${esc((p7.submission_success || 0))} ok · avgS ${esc(fmtSig(p7.score_avg))}</td>
            <td>${esc(fmtSig(w.last_inference_value))}</td>
            <td>${esc((w.last_inference_at === null || w.last_inference_at === undefined) ? '—' : w.last_inference_at)}</td>`;
          tb.appendChild(trMain);

          const trPath = document.createElement('tr');
          trPath.className = 'path-row';
          trPath.innerHTML = `<td colspan='10'><span class='path-label'>Artifact:</span><span class='path-short' title='${esc(w.artifact_path || '')}'>${esc(shortPath(w.artifact_path || ''))}</span></td>`;
          tb.appendChild(trPath);

          const trTimeline = document.createElement('tr');
          trTimeline.className = 'path-row';
          trTimeline.innerHTML = `<td colspan='10'><span class='path-label'>Submissions (last 25 slots):</span>${timelineHtml(w.timeline)}</td>`;
          tb.appendChild(trTimeline);

          const trLogs = document.createElement('tr');
          trLogs.className = 'logs-row';
          trLogs.innerHTML = `<td colspan='10'>
              <div class='logbox'>
                <details data-worker-key='${esc(key)}'>
                  <summary>stdout tail (${(w.log_tail || []).length} lines)</summary>
                  <pre class='logpre'>${esc(logs || '— no log output yet —')}</pre>
                </details>
              </div>
            </td>`;
          tb.appendChild(trLogs);

          const details = trLogs.querySelector('details');
          if (openLogPanels.has(key)) details.open = true;
          details.addEventListener('toggle', () => {
            if (details.open) openLogPanels.add(key);
            else openLogPanels.delete(key);
          });
        }
        tbl.appendChild(tb);
        card.appendChild(tbl);
        root.appendChild(card);
      }
        wireTooltips(root);
      } catch (e) {
        const msg = (e && e.message) ? e.message : String(e);
        document.getElementById('meta').textContent = `UI error: ${msg}`;
      }
    }
    window.addEventListener('error', (e) => {
      const msg = (e && e.message) ? e.message : 'unknown frontend error';
      document.getElementById('meta').textContent = `UI error: ${msg}`;
    });
    load();
    setInterval(load, 5000);
  </script>
</body>
</html>
"""


class DashboardApp:
    def __init__(
        self,
        db_path: str = "worker_state.db",
        secrets_path: str = "worker_secrets.json",
        network: str = "testnet",
    ):
        self.monitor = WorkerMonitor(
            db_path=db_path,
            event_fetcher=AlloraSDKEventFetcher(network=network, max_pages=2, page_limit=25),
        )
        self.wm = WorkerManager(
            db_path=db_path,
            secrets_path=secrets_path,
            network=network,
            monitor=self.monitor,
            reconcile_on_start=False,
        )
        self.wm.reconcile()
        self._last_sync = {"ok": None, "inserted": 0, "targets": 0, "errors": [], "at": None}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()

    def _sync_loop(self):
        while not self._stop.is_set():
            try:
                out = self.monitor.sync_once()
                self._last_sync = {
                    "ok": True,
                    "inserted": out.get("inserted", 0),
                    "targets": out.get("targets", 0),
                    "errors": out.get("errors", []),
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                self._last_sync = {
                    "ok": False,
                    "inserted": 0,
                    "targets": 0,
                    "errors": [{"error": str(e)}],
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            time.sleep(5)

    def stop(self):
        self._stop.set()

    def snapshot(self, tail_lines: int = 20) -> dict:
        rows = self.wm.status_all_with_logs(tail_lines=tail_lines)
        grouped: dict[str, list[dict]] = {}
        for r in rows:
            try:
                s = self.monitor.get_summary(r["topic_id"], r["address"])
            except (KeyError, Exception):
                s = {"events_total": 0, "submission_success": 0, "submission_error": 0,
                     "inference_count": 0, "rewards_total": 0, "last_inference": None,
                     "period_metrics": {}, "last_score": None, "last_reward_fraction": None,
                     "deployment_id": None}
            li = s.get("last_inference") or {}
            item = {
                "topic_id": r["topic_id"],
                "topic_desc": r.get("topic_desc"),
                "status": r.get("status"),
                "artifact_path": r.get("artifact_path"),
                "deployed_at": r.get("deployed_at"),
                "log_tail": r.get("log_tail", []),
                "submission_success": s.get("submission_success", 0),
                "submission_error": s.get("submission_error", 0),
                "inference_count": s.get("inference_count", 0),
                "period_metrics": s.get("period_metrics", {}),
                "last_score": s.get("last_score"),
                "last_reward_fraction": s.get("last_reward_fraction"),
                "timeline": self.monitor.get_submission_timeline(r["topic_id"], r["address"], deployment_id=s.get("deployment_id"), hours=24),
                "last_inference_value": li.get("value_text"),
                "last_inference_at": li.get("observed_at"),
            }
            grouped.setdefault(r["address"], []).append(item)

        by_addr = []
        for addr, workers in grouped.items():
            running = sum(1 for w in workers if w["status"] == "running")
            workers = sorted(workers, key=lambda x: x["topic_id"])
            by_addr.append({"address": addr, "running": running, "total": len(workers), "workers": workers})

        by_addr = sorted(by_addr, key=lambda x: x["address"])
        return {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "worker_count": len(rows),
            "sync": self._last_sync,
            "by_address": by_addr,
        }


def make_handler(app: DashboardApp, auth_token: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if auth_token:
                header = self.headers.get("Authorization", "")
                qs_check = parse_qs(urlparse(self.path).query)
                token_ok = (
                    header == f"Bearer {auth_token}"
                    or qs_check.get("token", [None])[0] == auth_token
                )
                if not token_ok:
                    self.send_response(401)
                    self.end_headers()
                    self.wfile.write(b"Unauthorized")
                    return

            parsed = urlparse(self.path)
            route = parsed.path
            qs = parse_qs(parsed.query)
            try:
                tail = max(1, min(int(qs.get("tail", ["20"])[0]), 500))
            except (ValueError, IndexError):
                tail = 20

            if route == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(HTML.encode("utf-8"))
                return

            if route == "/api/dashboard":
                data = app.snapshot(tail_lines=tail)
                payload = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return

            if route == "/api/workers":
                data = {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "workers": app.wm.status_all_with_logs(include_desc=True, tail_lines=tail),
                }
                payload = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Run local web dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--db-path", default="worker_state.db")
    parser.add_argument("--secrets-path", default="worker_secrets.json")
    parser.add_argument("--network", default="testnet")
    parser.add_argument("--token", default=None, help="Bearer token for auth (auto-generated if host is non-loopback)")
    args = parser.parse_args()

    auth_token = args.token
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        if not auth_token:
            auth_token = secrets.token_urlsafe(32)
        print(
            "WARNING: Dashboard bound to non-loopback interface "
            f"({args.host}). Auth token required for all requests.",
            file=sys.stderr,
        )
        print(f"Auth token: {auth_token}", file=sys.stderr)

    app = DashboardApp(
        db_path=args.db_path,
        secrets_path=args.secrets_path,
        network=args.network,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app, auth_token=auth_token))
    url = f"http://{args.host}:{args.port}"
    if auth_token:
        url += f"?token={auth_token}"
    print(f"Dashboard: {url}")
    try:
        server.serve_forever()
    finally:
        app.stop()
        server.server_close()


if __name__ == "__main__":
    main()
