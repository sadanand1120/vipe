#!/usr/bin/env python3

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def _fmt_time(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else None


def _run_rows(experiment_dir: Path) -> list[dict]:
    runs_dir = experiment_dir / "runs"
    if not runs_dir.exists():
        return []
    rows = []
    baseline_scene_metrics = {}
    baseline_fps = None
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = _load_json(run_dir / "meta.json") or {}
        summary = _load_json(run_dir / "summary.json") or {}
        scenes = meta.get("scenes") or []
        scene_metrics = {}
        scene_status = {}
        for scene in scenes:
            payload = _load_json(run_dir / "scenes" / f"{scene}.json") or {}
            scene_status[scene] = payload.get("status", "pending")
            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                scene_metrics[scene] = metrics
        row = {
            "run_id": run_dir.name,
            "description": meta.get("description", ""),
            "status": summary.get("status") or meta.get("status", "pending"),
            "complete": int(summary.get("complete", 0)),
            "total": int(summary.get("total", len(scenes))),
            "failed": int(summary.get("failed", 0)),
            "mean_auc30": summary.get("mean_auc30"),
            "mean_auc3": summary.get("mean_auc03"),
            "build_fps": summary.get("build_fps"),
            "build_seconds": summary.get("build_seconds"),
            "delta_auc30": None,
            "fps_ratio": None,
            "overrides": meta.get("overrides", {}),
            "scene_metrics": scene_metrics,
            "scene_status": scene_status,
            "updated_at": _fmt_time(summary.get("updated_at") or meta.get("updated_at")),
        }
        if run_dir.name == "baseline":
            baseline_scene_metrics = scene_metrics
            baseline_fps = row["build_fps"]
        rows.append(row)

    for row in rows:
        done_scenes = [
            scene
            for scene, metrics in row["scene_metrics"].items()
            if isinstance(metrics, dict) and scene in baseline_scene_metrics
        ]
        baseline_subset = [
            float(baseline_scene_metrics[scene]["auc30"])
            for scene in done_scenes
            if "auc30" in baseline_scene_metrics[scene]
        ]
        if row["mean_auc30"] is not None and baseline_subset:
            row["baseline_subset_auc30"] = sum(baseline_subset) / len(baseline_subset)
            row["delta_auc30"] = row["mean_auc30"] - row["baseline_subset_auc30"]
        else:
            row["baseline_subset_auc30"] = None
        if row["build_fps"] is not None and baseline_fps:
            row["fps_ratio"] = row["build_fps"] / baseline_fps
    return rows


def build_payload(experiment_dir: Path):
    rows = _run_rows(experiment_dir)
    scenes = []
    for row in rows:
        for scene in row["scene_status"]:
            if scene not in scenes:
                scenes.append(scene)
    return {
        "generated_at": _fmt_time(time.time()),
        "experiment_dir": str(experiment_dir),
        "total_runs": len(rows),
        "scenes": scenes,
        "rows": rows,
    }


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ScanNet Minibench Search</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #090d12;
      --panel: rgba(235, 241, 255, 0.08);
      --line: rgba(235, 241, 255, 0.16);
      --text: #eef4ff;
      --muted: #9eadc4;
      --good: #7ee0a1;
      --bad: #ff8277;
      --warn: #ffd36b;
      --blue: #84c7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 14% -8%, rgba(126, 224, 161, 0.20), transparent 32rem),
        radial-gradient(circle at 90% 8%, rgba(132, 199, 255, 0.18), transparent 35rem),
        linear-gradient(135deg, #090d12, #121722 52%, #19120f);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 4;
      background: rgba(9, 13, 18, 0.78);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(18px);
      padding: 18px 22px;
    }
    h1 { margin: 0 0 8px; font-size: 24px; letter-spacing: -0.03em; }
    .sub { color: var(--muted); display: flex; gap: 14px; flex-wrap: wrap; font-size: 13px; }
    main { padding: 18px 22px 34px; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
    .card { border: 1px solid var(--line); background: var(--panel); border-radius: 18px; padding: 14px; box-shadow: 0 18px 60px rgba(0,0,0,0.24); }
    .label { color: var(--muted); text-transform: uppercase; letter-spacing: 0.09em; font-size: 12px; }
    .value { margin-top: 8px; font-weight: 800; font-size: 24px; }
    .small { color: var(--muted); font-size: 12px; margin-top: 6px; overflow-wrap: anywhere; }
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }
    input, select { background: rgba(235,241,255,0.08); border: 1px solid var(--line); color: var(--text); border-radius: 12px; padding: 10px 12px; outline: none; }
    input { min-width: 300px; }
    .tablewrap { border: 1px solid var(--line); border-radius: 18px; overflow: auto; background: rgba(9,13,18,0.58); box-shadow: 0 24px 70px rgba(0,0,0,0.24); }
    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    th, td { padding: 10px 12px; border-bottom: 1px solid rgba(235,241,255,0.09); white-space: nowrap; vertical-align: top; }
    th { position: sticky; top: 0; z-index: 2; background: #121923; color: var(--muted); text-align: left; font-size: 12px; cursor: pointer; }
    td { font-size: 13px; }
    tr:hover td { background: rgba(235,241,255,0.045); }
    tr.row-good td { background: rgba(69, 214, 119, 0.34); border-bottom-color: rgba(126,224,161,0.35); }
    tr.row-bad td { background: rgba(255, 82, 70, 0.34); border-bottom-color: rgba(255,130,119,0.35); }
    tr.row-slow td { background: rgba(255, 198, 56, 0.28); border-bottom-color: rgba(255,211,107,0.35); }
    .desc {
      max-width: 260px;
      white-space: normal;
      font-size: 12px;
      line-height: 1.25;
      color: var(--text);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .override {
      max-width: 260px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; font-size: 12px; color: var(--muted); background: rgba(235,241,255,0.06); }
    .done { color: var(--good); border-color: rgba(126,224,161,0.4); }
    .running { color: var(--blue); border-color: rgba(132,199,255,0.4); }
    .failed { color: var(--bad); border-color: rgba(255,130,119,0.4); }
    .pending { color: var(--warn); border-color: rgba(255,211,107,0.4); }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .na { color: #6f7d91; }
    .scenegrid { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 6px; min-width: 760px; }
    .scene { border: 1px solid rgba(235,241,255,0.10); border-radius: 9px; padding: 5px 7px; background: rgba(235,241,255,0.045); }
    .scene .name { color: var(--muted); font-size: 11px; }
    .scene .metric { margin-top: 2px; font-weight: 750; }
    @media (max-width: 850px) {
      header, main { padding-left: 12px; padding-right: 12px; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      th, td { padding: 9px 10px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>ScanNet minibench search</h1>
    <div class="sub"><span id="updated">loading...</span><span>Auto-refresh: 30s</span><span id="exp"></span></div>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div class="label">Runs</div><div class="value" id="runs">NA</div><div class="small">baseline plus hypothesis runs</div></div>
      <div class="card"><div class="label">Best AUC30</div><div class="value" id="bestAuc">NA</div><div class="small" id="bestRun">NA</div></div>
      <div class="card"><div class="label">Best Delta</div><div class="value" id="bestDelta">NA</div><div class="small">vs matching completed baseline scenes; &lt;0.02 is noise</div></div>
      <div class="card"><div class="label">FPS Guardrail</div><div class="value" id="guard">0.50x</div><div class="small">runs below this are marked slow</div></div>
    </section>
    <div class="toolbar">
      <input id="search" placeholder="Filter runs or descriptions">
      <select id="status"><option value="">all statuses</option><option>pending</option><option>running</option><option>done</option><option>failed</option></select>
      <select id="sort"><option value="run_id">sort: run id</option><option value="mean_auc30">sort: AUC30</option><option value="delta_auc30">sort: delta AUC30</option><option value="build_fps">sort: build FPS</option></select>
    </div>
    <div class="tablewrap">
      <table>
        <thead><tr><th>Run</th><th>Status</th><th>Description</th><th>Complete</th><th>Mean AUC30</th><th>Delta</th><th>Build FPS</th><th>FPS Ratio</th><th>Scenes</th></tr></thead>
        <tbody id="tbody"><tr><td colspan="9">Loading...</td></tr></tbody>
      </table>
    </div>
  </main>
  <script>
    let payload = null;
    const fmt = v => (v === null || v === undefined || Number.isNaN(v)) ? '<span class="na">NA</span>' : Number(v).toFixed(4);
    const fmt2 = v => (v === null || v === undefined || Number.isNaN(v)) ? '<span class="na">NA</span>' : Number(v).toFixed(2);
    const delta = v => {
      if (v === null || v === undefined || Number.isNaN(v)) return '<span class="na">NA</span>';
      const cls = v >= 0 ? 'good' : 'bad';
      return `<span class="${cls}">${v >= 0 ? '+' : ''}${Number(v).toFixed(4)}</span>`;
    };
    const pill = s => `<span class="pill ${s}">${s}</span>`;
    const rowClass = r => {
      if (r.fps_ratio !== null && r.fps_ratio !== undefined && r.fps_ratio < 0.5) return 'row-slow';
      if (r.delta_auc30 !== null && r.delta_auc30 !== undefined && r.delta_auc30 >= 0.01) return 'row-good';
      if (r.delta_auc30 !== null && r.delta_auc30 !== undefined && r.delta_auc30 <= -0.01) return 'row-bad';
      return '';
    };
    const sceneCell = r => {
      const scenes = payload.scenes || [];
      return `<div class="scenegrid">` + scenes.map(s => {
        const m = r.scene_metrics[s];
        const st = r.scene_status[s] || 'pending';
        const val = m && m.auc30 !== undefined ? Number(m.auc30).toFixed(3) : st;
        return `<div class="scene"><div class="name">${s}</div><div class="metric">${val}</div></div>`;
      }).join('') + `</div>`;
    };
    async function load() {
      const res = await fetch('/data?ts=' + Date.now());
      payload = await res.json();
      render();
    }
    function render() {
      if (!payload) return;
      document.getElementById('updated').textContent = 'Updated: ' + payload.generated_at;
      document.getElementById('exp').textContent = payload.experiment_dir;
      document.getElementById('runs').textContent = payload.total_runs;
      const done = payload.rows.filter(r => r.status === 'done' && r.mean_auc30 !== null && r.mean_auc30 !== undefined);
      const best = done.slice().sort((a,b) => b.mean_auc30 - a.mean_auc30)[0];
      document.getElementById('bestAuc').innerHTML = best ? fmt(best.mean_auc30) : '<span class="na">NA</span>';
      document.getElementById('bestRun').textContent = best ? best.run_id : 'NA';
      document.getElementById('bestDelta').innerHTML = best ? delta(best.delta_auc30) : '<span class="na">NA</span>';
      const q = document.getElementById('search').value.trim().toLowerCase();
      const st = document.getElementById('status').value;
      const sortKey = document.getElementById('sort').value;
      let rows = payload.rows.filter(r => (!q || r.run_id.toLowerCase().includes(q) || (r.description || '').toLowerCase().includes(q)) && (!st || r.status === st));
      rows.sort((a,b) => {
        if (sortKey === 'run_id') return a.run_id.localeCompare(b.run_id);
        const av = a[sortKey], bv = b[sortKey];
        if (av === null || av === undefined) return 1;
        if (bv === null || bv === undefined) return -1;
        return bv - av;
      });
      document.getElementById('tbody').innerHTML = rows.map(r => `
        <tr class="${rowClass(r)}">
          <td><b>${r.run_id}</b><div class="small">${r.updated_at || ''}</div></td>
          <td>${pill(r.status)}</td>
          <td><div class="desc" title="${r.description || ''}">${r.description || ''}</div><div class="small override" title="${JSON.stringify(r.overrides || {})}">${JSON.stringify(r.overrides || {})}</div></td>
          <td>${r.complete}/${r.total}${r.failed ? `<div class="bad small">${r.failed} failed</div>` : ''}</td>
          <td>${fmt(r.mean_auc30)}</td>
          <td>${delta(r.delta_auc30)}</td>
          <td>${fmt2(r.build_fps)}</td>
          <td>${fmt2(r.fps_ratio)}</td>
          <td>${sceneCell(r)}</td>
        </tr>`).join('');
    }
    document.getElementById('search').addEventListener('input', render);
    document.getElementById('status').addEventListener('change', render);
    document.getElementById('sort').addEventListener('change', render);
    load();
    setInterval(load, 30000);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    experiment_dir: Path

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/data":
            body = json.dumps(build_payload(self.experiment_dir)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()

    Handler.experiment_dir = args.experiment_dir.resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving http://{args.host}:{args.port}", flush=True)
    print(f"pid {os.getpid()}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
