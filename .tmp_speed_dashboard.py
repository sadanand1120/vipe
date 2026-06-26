#!/usr/bin/env python3

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


STAGE_COLUMNS = [
    ("pass1", "slam.pass1_s"),
    ("backend", "slam.backend_s"),
    ("pass2", "slam.pass2_s"),
    ("artifacts", "artifacts_s"),
    ("load", "artifacts.frame_load_attach_s"),
    ("tsdf", "artifacts.tsdf_prepare_s", "artifacts.tsdf_integrate_s"),
    ("extract+write", "artifacts.tsdf_extract_s", "artifacts.tsdf_ply_write_s"),
]


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def _fmt_time(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else None


def _stage_value(summary: dict, keys: tuple[str, ...]) -> float | None:
    means = summary.get("build_stage_summary", {}).get("mean_seconds", {})
    values = [means.get(key) for key in keys]
    if any(value is None for value in values):
        return None
    return float(sum(values))


def _nested_get(value: dict, dotted: str) -> float | None:
    cur = value
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return float(cur) if isinstance(cur, (int, float)) else None


def _stage_means_from_scenes(scene_payloads: dict[str, dict], summary: dict) -> dict[str, float | None]:
    stages = {}
    for label, *keys in STAGE_COLUMNS:
        scene_values = []
        for payload in scene_payloads.values():
            build_timing = payload.get("timing", {}).get("build", {})
            stage_timing = build_timing.get("stages", {})
            values = [_nested_get(stage_timing, key) for key in keys]
            if all(value is not None for value in values):
                scene_values.append(sum(values))
        if scene_values:
            stages[label] = sum(scene_values) / len(scene_values)
        else:
            stages[label] = _stage_value(summary, tuple(keys))
    return stages


def _run_rows(experiment_dir: Path) -> list[dict]:
    runs_dir = experiment_dir / "runs"
    if not runs_dir.exists():
        return []

    rows = []
    baseline_auc_by_scene = {}
    baseline_fps = None
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = _load_json(run_dir / "meta.json") or {}
        summary = _load_json(run_dir / "summary.json") or {}
        scenes = meta.get("scenes") or []
        scene_payloads = {}
        for scene in scenes:
            scene_payloads[scene] = _load_json(run_dir / "scenes" / f"{scene}.json") or {}

        scene_status = {scene: payload.get("status", "pending") for scene, payload in scene_payloads.items()}
        scene_auc30 = {
            scene: payload.get("metrics", {}).get("auc30")
            for scene, payload in scene_payloads.items()
            if isinstance(payload.get("metrics"), dict)
        }
        stages = _stage_means_from_scenes(scene_payloads, summary)
        row = {
            "run_id": run_dir.name,
            "description": meta.get("description", ""),
            "status": summary.get("status") or meta.get("status", "pending"),
            "complete": int(summary.get("complete", 0)),
            "total": int(summary.get("total", len(scenes))),
            "failed": int(summary.get("failed", 0)),
            "mean_auc30": summary.get("mean_auc30"),
            "build_fps": summary.get("build_fps"),
            "build_seconds": summary.get("build_seconds"),
            "mean_build_seconds": (
                float(summary["build_seconds"]) / int(summary["complete"])
                if summary.get("build_seconds") is not None and int(summary.get("complete", 0)) > 0
                else None
            ),
            "fps_ratio": None,
            "auc30_delta_done": None,
            "overrides": meta.get("overrides", {}),
            "scene_status": scene_status,
            "scene_auc30": scene_auc30,
            "stages": stages,
            "updated_at": _fmt_time(summary.get("updated_at") or meta.get("updated_at")),
        }
        if run_dir.name == "baseline":
            baseline_auc_by_scene = {
                scene: float(auc)
                for scene, auc in scene_auc30.items()
                if auc is not None
            }
            baseline_fps = row["build_fps"]
        rows.append(row)

    for row in rows:
        done_scenes = [scene for scene, auc in row["scene_auc30"].items() if auc is not None and scene in baseline_auc_by_scene]
        if done_scenes and row["mean_auc30"] is not None:
            baseline_subset = sum(baseline_auc_by_scene[scene] for scene in done_scenes) / len(done_scenes)
            row["auc30_delta_done"] = float(row["mean_auc30"]) - baseline_subset
        if row["build_fps"] is not None and baseline_fps:
            row["fps_ratio"] = float(row["build_fps"]) / float(baseline_fps)
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
  <title>ViPE Speed Minibench</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #070a0d;
      --panel: rgba(234, 244, 255, 0.075);
      --line: rgba(234, 244, 255, 0.14);
      --text: #eef6ff;
      --muted: #9fb0c5;
      --good: #7af0a5;
      --bad: #ff7b6d;
      --warn: #ffd66e;
      --blue: #7dc7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 12% -10%, rgba(122, 240, 165, 0.18), transparent 32rem),
        radial-gradient(circle at 90% 0%, rgba(125, 199, 255, 0.16), transparent 34rem),
        linear-gradient(135deg, #070a0d, #121820 56%, #17110c);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(7, 10, 13, 0.78);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(18px);
      padding: 16px 20px;
    }
    h1 { margin: 0 0 8px; font-size: 24px; letter-spacing: -0.035em; }
    .sub { color: var(--muted); display: flex; gap: 14px; flex-wrap: wrap; font-size: 13px; }
    main { padding: 16px 20px 34px; }
    .cards { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
    .card { border: 1px solid var(--line); background: var(--panel); border-radius: 18px; padding: 13px; box-shadow: 0 18px 60px rgba(0,0,0,0.25); }
    .label { color: var(--muted); text-transform: uppercase; letter-spacing: 0.09em; font-size: 11px; }
    .value { margin-top: 7px; font-weight: 850; font-size: 24px; }
    .small { color: var(--muted); font-size: 12px; margin-top: 5px; overflow-wrap: anywhere; }
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }
    input, select { background: rgba(234,244,255,0.08); border: 1px solid var(--line); color: var(--text); border-radius: 12px; padding: 10px 12px; outline: none; }
    input { min-width: 300px; }
    .tablewrap { border: 1px solid var(--line); border-radius: 18px; overflow: auto; background: rgba(7,10,13,0.60); box-shadow: 0 24px 70px rgba(0,0,0,0.24); }
    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    th, td { padding: 9px 10px; border-bottom: 1px solid rgba(234,244,255,0.09); white-space: nowrap; vertical-align: top; }
    th { position: sticky; top: 0; z-index: 2; background: #111923; color: var(--muted); text-align: left; font-size: 12px; }
    td { font-size: 13px; }
    tr:hover td { background: rgba(234,244,255,0.045); }
    tr.row-fast td { background: rgba(69, 214, 119, 0.29); border-bottom-color: rgba(122,240,165,0.32); }
    tr.row-slow td { background: rgba(255, 198, 56, 0.24); border-bottom-color: rgba(255,214,110,0.32); }
    tr.row-bad td { background: rgba(255, 82, 70, 0.28); border-bottom-color: rgba(255,123,109,0.32); }
    .desc {
      max-width: 250px;
      white-space: normal;
      font-size: 12px;
      line-height: 1.25;
      color: var(--text);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .override { max-width: 250px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; font-size: 12px; color: var(--muted); background: rgba(234,244,255,0.06); }
    .done { color: var(--good); border-color: rgba(122,240,165,0.42); }
    .running { color: var(--blue); border-color: rgba(125,199,255,0.42); }
    .failed { color: var(--bad); border-color: rgba(255,123,109,0.42); }
    .pending { color: var(--warn); border-color: rgba(255,214,110,0.42); }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .na { color: #6f7d91; }
    .scenegrid { display: grid; grid-template-columns: repeat(4, minmax(105px, 1fr)); gap: 6px; min-width: 480px; }
    .scene { border: 1px solid rgba(234,244,255,0.10); border-radius: 9px; padding: 5px 7px; background: rgba(234,244,255,0.045); }
    .scene .name { color: var(--muted); font-size: 11px; }
    .scene .metric { margin-top: 2px; font-weight: 750; }
    @media (max-width: 950px) {
      header, main { padding-left: 12px; padding-right: 12px; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      th, td { padding: 8px 9px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>ViPE speed minibench</h1>
    <div class="sub"><span id="updated">loading...</span><span>Auto-refresh: 30s</span><span id="exp"></span></div>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div class="label">Runs</div><div class="value" id="runs">NA</div><div class="small">one-at-a-time speed experiments</div></div>
      <div class="card"><div class="label">Best FPS</div><div class="value" id="bestFps">NA</div><div class="small" id="bestFpsRun">NA</div></div>
      <div class="card"><div class="label">Best Speedup</div><div class="value" id="bestSpeed">NA</div><div class="small">vs baseline FPS</div></div>
      <div class="card"><div class="label">Mean AUC30</div><div class="value" id="bestAuc">NA</div><div class="small">accuracy guardrail</div></div>
    </section>
    <div class="toolbar">
      <input id="search" placeholder="Filter runs or descriptions">
      <select id="status"><option value="">all statuses</option><option>pending</option><option>running</option><option>done</option><option>failed</option></select>
      <select id="sort"><option value="run_id">sort: run id</option><option value="build_fps">sort: build FPS</option><option value="fps_ratio">sort: speedup</option><option value="mean_auc30">sort: mean AUC30</option><option value="build_seconds">sort: build seconds</option></select>
    </div>
    <div class="tablewrap">
      <table>
        <thead>
          <tr>
            <th>Run</th><th>Status</th><th>Complete</th><th>Mean AUC30</th><th>FPS</th><th>Speedup</th><th>Mean Build s</th>
            <th>Pass1 s</th><th>Backend s</th><th>Pass2 s</th><th>Artifacts s</th><th>Load s</th><th>TSDF s</th><th>Extract+Write s</th>
            <th>Description</th><th>Scenes</th>
          </tr>
        </thead>
        <tbody id="tbody"><tr><td colspan="16">Loading...</td></tr></tbody>
      </table>
    </div>
  </main>
  <script>
    let payload = null;
    const fmt = v => (v === null || v === undefined || Number.isNaN(v)) ? '<span class="na">NA</span>' : Number(v).toFixed(4);
    const fmt2 = v => (v === null || v === undefined || Number.isNaN(v)) ? '<span class="na">NA</span>' : Number(v).toFixed(2);
    const sec = v => (v === null || v === undefined || Number.isNaN(v)) ? '<span class="na">NA</span>' : Number(v).toFixed(1);
    const ratio = v => (v === null || v === undefined || Number.isNaN(v)) ? '<span class="na">NA</span>' : `${Number(v).toFixed(2)}x`;
    const pill = s => `<span class="pill ${s}">${s}</span>`;
    const rowClass = r => {
      if (r.status === 'failed') return 'row-bad';
      if (r.auc30_delta_done !== null && r.auc30_delta_done !== undefined && r.auc30_delta_done < -0.01) return 'row-bad';
      if (r.fps_ratio !== null && r.fps_ratio !== undefined && r.fps_ratio > 1.03) return 'row-fast';
      return '';
    };
    const sceneCell = r => {
      const scenes = payload.scenes || [];
      return `<div class="scenegrid">` + scenes.map(s => {
        const st = r.scene_status[s] || 'pending';
        const auc = r.scene_auc30[s];
        const val = auc !== null && auc !== undefined ? Number(auc).toFixed(3) : st;
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
      const done = payload.rows.filter(r => r.status === 'done' && r.build_fps !== null && r.build_fps !== undefined);
      const best = done.slice().sort((a,b) => b.build_fps - a.build_fps)[0];
      document.getElementById('bestFps').innerHTML = best ? fmt2(best.build_fps) : '<span class="na">NA</span>';
      document.getElementById('bestFpsRun').textContent = best ? best.run_id : 'NA';
      document.getElementById('bestSpeed').innerHTML = best ? ratio(best.fps_ratio) : '<span class="na">NA</span>';
      document.getElementById('bestAuc').innerHTML = best ? fmt(best.mean_auc30) : '<span class="na">NA</span>';
      const q = document.getElementById('search').value.trim().toLowerCase();
      const st = document.getElementById('status').value;
      const sortKey = document.getElementById('sort').value;
      let rows = payload.rows.filter(r => (!q || r.run_id.toLowerCase().includes(q) || (r.description || '').toLowerCase().includes(q)) && (!st || r.status === st));
      rows.sort((a,b) => {
        if (sortKey === 'run_id') return a.run_id.localeCompare(b.run_id);
        const av = a[sortKey], bv = b[sortKey];
        if (av === null || av === undefined) return 1;
        if (bv === null || bv === undefined) return -1;
        if (sortKey === 'build_seconds') return av - bv;
        return bv - av;
      });
      document.getElementById('tbody').innerHTML = rows.map(r => `
        <tr class="${rowClass(r)}">
          <td><b>${r.run_id}</b><div class="small">${r.updated_at || ''}</div></td>
          <td>${pill(r.status)}</td>
          <td>${r.complete}/${r.total}${r.failed ? `<div class="bad small">${r.failed} failed</div>` : ''}</td>
          <td>${fmt(r.mean_auc30)}</td>
          <td>${fmt2(r.build_fps)}</td>
          <td>${ratio(r.fps_ratio)}</td>
          <td>${sec(r.mean_build_seconds)}</td>
          <td>${sec(r.stages.pass1)}</td>
          <td>${sec(r.stages.backend)}</td>
          <td>${sec(r.stages.pass2)}</td>
          <td>${sec(r.stages.artifacts)}</td>
          <td>${sec(r.stages.load)}</td>
          <td>${sec(r.stages.tsdf)}</td>
          <td>${sec(r.stages['extract+write'])}</td>
          <td><div class="desc" title="${r.description || ''}">${r.description || ''}</div><div class="small override" title="${JSON.stringify(r.overrides || {})}">${JSON.stringify(r.overrides || {})}</div></td>
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
    parser.add_argument("--port", type=int, default=8778)
    args = parser.parse_args()

    Handler.experiment_dir = args.experiment_dir.resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving http://{args.host}:{args.port}", flush=True)
    print(f"pid {os.getpid()}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
