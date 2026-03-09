from __future__ import annotations

import argparse
import json
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

    async function load() {
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
            <th>Topic</th><th>Status</th><th>Submissions</th><th>Inferences</th><th>Last Inference</th><th>At</th>
          </tr></thead>`;
        const tb = document.createElement('tbody');
        for (const w of grp.workers) {
          const logs = (w.log_tail || []).join('\\\\n');
          const key = workerKey(grp, w);

          const trMain = document.createElement('tr');
          trMain.innerHTML = `<td>${esc(w.topic_id)} — ${esc(w.topic_desc || '')}</td>
            <td>${esc(w.status)}</td>
            <td class='ok'>${esc(w.submission_success)} ok</td>
            <td>${esc(w.inference_count)}</td>
            <td>${esc((w.last_inference_value === null || w.last_inference_value === undefined) ? '—' : w.last_inference_value)}</td>
            <td>${esc((w.last_inference_at === null || w.last_inference_at === undefined) ? '—' : w.last_inference_at)}</td>`;
          tb.appendChild(trMain);

          const trPath = document.createElement('tr');
          trPath.className = 'path-row';
          trPath.innerHTML = `<td colspan='6'><span class='path-label'>Artifact:</span><span class='path-short' title='${esc(w.artifact_path || '')}'>${esc(shortPath(w.artifact_path || ''))}</span></td>`;
          tb.appendChild(trPath);

          const trLogs = document.createElement('tr');
          trLogs.className = 'logs-row';
          trLogs.innerHTML = `<td colspan='6'>
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
    }
    load();
    setInterval(load, 5000);
  </script>
</body>
</html>
"""


class DashboardApp:
    def __init__(self):
        self.monitor = WorkerMonitor(
            db_path="worker_state.db",
            event_fetcher=AlloraSDKEventFetcher(network="testnet", max_pages=2, page_limit=25),
        )
        self.wm = WorkerManager(
            db_path="worker_state.db",
            secrets_path="worker_secrets.json",
            monitor=self.monitor,
            reconcile_on_start=True,
        )
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
            s = self.monitor.get_summary(r["topic_id"], r["address"])
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


def make_handler(app: DashboardApp):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            route = parsed.path
            qs = parse_qs(parsed.query)
            tail = int(qs.get("tail", ["20"])[0])

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
    args = parser.parse_args()

    app = DashboardApp()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Dashboard: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        app.stop()
        server.server_close()


if __name__ == "__main__":
    main()
