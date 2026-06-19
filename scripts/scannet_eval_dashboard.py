#!/usr/bin/env python3
#
# Optional public URL:
#   wget -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
#   chmod +x cloudflared
#   ./cloudflared tunnel --url http://127.0.0.1:<port> --protocol http2 --no-autoupdate

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
        with path.open() as f:
            return json.load(f)
    except Exception as exc:
        return {"_error": str(exc)}


def _mtime(path: Path):
    return path.stat().st_mtime if path.exists() else None


def _fmt_time(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else None


def _scene_keys(*dicts):
    scenes = set()
    for data in dicts:
        if isinstance(data, dict):
            scenes.update(k for k in data.keys() if k != "mean" and not k.startswith("_"))
    return sorted(scenes)


def _unavailable_scenes(after_root: Path):
    paths = [
        after_root / "metric_results" / "skipped_gt_pose_scenes.json",
        Path("tmpouts/scannet_gt_unavailable.json"),
    ]
    for path in paths:
        data = _load_json(path)
        if isinstance(data, dict):
            return {str(scene): str(reason) for scene, reason in data.items()}
    return {}


def _mean_pose_metrics(scene_metrics: dict[str, dict]) -> dict[str, float]:
    metrics = [item for item in scene_metrics.values() if isinstance(item, dict)]
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: sum(float(item[key]) for item in metrics) / len(metrics) for key in keys}


def _mean_rows(rows, keys):
    usable = [row for row in rows if all(row.get(key) is not None for key in keys)]
    if not usable:
        return {}
    return {key: sum(float(row[key]) for row in usable) / len(usable) for key in keys}


def _load_pose_metrics(root: Path):
    final_path = root / "metric_results" / "scannet_pose.json"
    final_data = _load_json(final_path)
    if final_data is not None:
        return final_data, _mtime(final_path), "final"

    inc_dir = root / "metric_results" / "incremental_pose" / "scannet"
    if not inc_dir.exists():
        return None, None, "missing"

    data = {}
    latest = None
    for path in sorted(inc_dir.glob("*.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict) or "_error" in payload:
            continue
        scene = payload.get("scene") or path.stem
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            data[scene] = {key: float(value) for key, value in metrics.items()}
            latest = max(latest or 0.0, _mtime(path) or 0.0)

    if data:
        data["mean"] = _mean_pose_metrics({key: val for key, val in data.items() if key != "mean"})
        return data, latest, "incremental"
    return None, latest, "incremental"


def _metric(data, scene, key):
    if not isinstance(data, dict):
        return None
    value = data.get(scene, {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _scene_status(after_root: Path, input_root: Path, scene: str, has_metric: bool, unavailable: dict[str, str]):
    if has_metric:
        return "done"
    if scene in unavailable:
        return "unavailable"
    if (after_root / "metric_results" / "scannet_pose.json").exists():
        return "unavailable"
    if not (input_root / scene).exists():
        return "unavailable"
    if (after_root / "vipe_outputs" / scene).exists():
        return "running"
    if (after_root / "model_results" / "scannet" / scene).exists():
        return "running"
    return "pending"


def build_payload(before_root: Path, after_root: Path, input_root: Path):
    before_pose_path = before_root / "metric_results" / "scannet_pose.json"
    before_pose = _load_json(before_pose_path)
    after_pose, after_pose_mtime, after_pose_source = _load_pose_metrics(after_root)
    unavailable = _unavailable_scenes(after_root)

    scenes = _scene_keys(before_pose, after_pose)
    rows = []
    for scene in scenes:
        after_auc3 = _metric(after_pose, scene, "auc03")
        after_auc30 = _metric(after_pose, scene, "auc30")
        status = _scene_status(after_root, input_root, scene, after_auc30 is not None, unavailable)
        rows.append(
            {
                "scene": scene,
                "before_auc3": _metric(before_pose, scene, "auc03"),
                "before_auc30": _metric(before_pose, scene, "auc30"),
                "after_auc3": after_auc3,
                "after_auc30": after_auc30,
                "delta_auc3": None
                if after_auc3 is None
                else after_auc3 - (_metric(before_pose, scene, "auc03") or 0.0),
                "delta_auc30": None
                if after_auc30 is None
                else after_auc30 - (_metric(before_pose, scene, "auc30") or 0.0),
                "status": status,
                "unavailable_reason": unavailable.get(scene),
            }
        )

    available_rows = [row for row in rows if row["status"] != "unavailable"]
    complete = sum(1 for row in available_rows if row["after_auc30"] is not None)
    completed_rows = [
        row
        for row in available_rows
        if row["before_auc3"] is not None
        and row["before_auc30"] is not None
        and row["after_auc3"] is not None
        and row["after_auc30"] is not None
    ]
    if completed_rows:
        mean_row = {
            "scene": f"MEAN ({len(completed_rows)} comparable)",
            "before_auc3": sum(row["before_auc3"] for row in completed_rows) / len(completed_rows),
            "before_auc30": sum(row["before_auc30"] for row in completed_rows) / len(completed_rows),
            "after_auc3": sum(row["after_auc3"] for row in completed_rows) / len(completed_rows),
            "after_auc30": sum(row["after_auc30"] for row in completed_rows) / len(completed_rows),
            "status": "done",
        }
        mean_row["delta_auc3"] = mean_row["after_auc3"] - mean_row["before_auc3"]
        mean_row["delta_auc30"] = mean_row["after_auc30"] - mean_row["before_auc30"]
    else:
        before_mean = _mean_rows(available_rows, ["before_auc3", "before_auc30"])
        mean_row = {
            "scene": "MEAN",
            "before_auc3": before_mean.get("before_auc3"),
            "before_auc30": before_mean.get("before_auc30"),
            "after_auc3": None,
            "after_auc30": None,
            "delta_auc3": None,
            "delta_auc30": None,
            "status": "pending",
        }
    return {
        "generated_at": _fmt_time(time.time()),
        "before_root": str(before_root),
        "after_root": str(after_root),
        "input_root": str(input_root),
        "before_pose_mtime": _fmt_time(_mtime(before_pose_path)),
        "after_pose_mtime": _fmt_time(after_pose_mtime),
        "after_pose_source": after_pose_source,
        "before_mean": before_pose.get("mean") if isinstance(before_pose, dict) else None,
        "after_mean": after_pose.get("mean") if isinstance(after_pose, dict) else None,
        "before_error": before_pose.get("_error") if isinstance(before_pose, dict) else None,
        "after_error": after_pose.get("_error") if isinstance(after_pose, dict) else None,
        "total": len(available_rows),
        "unavailable": len(rows) - len(available_rows),
        "complete": complete,
        "mean_row": mean_row,
        "rows": rows,
    }


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ScanNet AUC Monitor</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #09100d;
      --panel: rgba(246, 239, 218, 0.08);
      --line: rgba(246, 239, 218, 0.18);
      --text: #f6efda;
      --muted: #acb7a7;
      --good: #81d39c;
      --bad: #f07f6f;
      --warn: #e7c66a;
      --blue: #8cc7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 20% -10%, rgba(129, 211, 156, 0.22), transparent 35rem),
        radial-gradient(circle at 80% 10%, rgba(140, 199, 255, 0.16), transparent 34rem),
        linear-gradient(135deg, #09100d, #121814 45%, #17120d);
      color: var(--text);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      min-height: 100vh;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 3;
      backdrop-filter: blur(18px);
      background: rgba(9, 16, 13, 0.78);
      border-bottom: 1px solid var(--line);
      padding: 18px 22px;
    }
    h1 { margin: 0 0 10px; font-size: 24px; letter-spacing: -0.03em; }
    .sub { color: var(--muted); font-size: 13px; display: flex; flex-wrap: wrap; gap: 14px; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; padding: 18px 22px 0; }
    .card {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 16px 50px rgba(0,0,0,0.22);
    }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.09em; }
    .value { margin-top: 7px; font-size: 24px; font-weight: 750; }
    .small { font-size: 12px; color: var(--muted); margin-top: 5px; overflow-wrap: anywhere; }
    main { padding: 18px 22px 32px; }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 12px;
    }
    input, select {
      background: rgba(246,239,218,0.09);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      outline: none;
    }
    input { min-width: 230px; }
    .tablewrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(9, 16, 13, 0.58);
      box-shadow: 0 24px 70px rgba(0,0,0,0.22);
    }
    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    th, td { padding: 10px 12px; border-bottom: 1px solid rgba(246,239,218,0.09); white-space: nowrap; }
    th {
      position: sticky;
      top: 0;
      background: #131a15;
      color: var(--muted);
      text-align: left;
      font-size: 12px;
      cursor: pointer;
      user-select: none;
    }
    td { font-size: 13px; }
    tr:hover td { background: rgba(246,239,218,0.045); }
    tr.row-good td { background: rgba(129, 211, 156, 0.105); }
    tr.row-good:hover td { background: rgba(129, 211, 156, 0.155); }
    tr.row-bad td { background: rgba(240, 127, 111, 0.115); }
    tr.row-bad:hover td { background: rgba(240, 127, 111, 0.17); }
    tr.row-mixed td { background: rgba(231, 198, 106, 0.105); }
    tr.row-mixed:hover td { background: rgba(231, 198, 106, 0.155); }
    tr.mean-row td {
      position: sticky;
      bottom: 0;
      background: #181f18;
      border-top: 1px solid var(--line);
      font-weight: 800;
    }
    .na { color: #74806f; }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .pill {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: var(--muted);
      background: rgba(246,239,218,0.06);
    }
    .done { color: var(--good); border-color: rgba(129,211,156,0.38); }
    .running { color: var(--blue); border-color: rgba(140,199,255,0.35); }
    .pending { color: var(--warn); border-color: rgba(231,198,106,0.35); }
    .unavailable { color: #858a82; border-color: rgba(133,138,130,0.28); }
    .low-before {
      color: var(--warn);
      font-weight: 900;
      margin-left: 5px;
    }
    tr.row-unavailable td {
      background: rgba(0, 0, 0, 0.30);
      color: #858a82;
    }
    tr.row-unavailable:hover td { background: rgba(0, 0, 0, 0.42); }
    @media (max-width: 850px) {
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); padding: 12px; }
      header, main { padding-left: 12px; padding-right: 12px; }
      th, td { padding: 9px 10px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>ScanNet full eval monitor</h1>
    <div class="sub">
      <span id="generated">loading...</span>
      <span>Auto-refresh: 30s</span>
      <span>Before: <span id="beforeRoot">loading...</span></span>
      <span>After: <span id="afterRoot">loading...</span></span>
    </div>
  </header>
  <section class="cards">
    <div class="card"><div class="label">Scenes complete</div><div class="value" id="complete">NA</div><div class="small" id="completeSmall">after pose metrics available; unavailable excluded</div></div>
    <div class="card"><div class="label">Before mean AUC30</div><div class="value" id="beforeMean30">NA</div><div class="small" id="beforeRootSmall">baseline run</div></div>
    <div class="card"><div class="label">After mean AUC30</div><div class="value" id="afterMean30">NA</div><div class="small" id="afterRootSmall">new run</div></div>
    <div class="card"><div class="label">After pose JSON</div><div class="value" id="afterMtime">NA</div><div class="small">last write time</div></div>
  </section>
  <main>
    <div class="toolbar">
      <input id="search" placeholder="Filter scenes, e.g. scene0012">
      <select id="status"><option value="">all statuses</option><option>pending</option><option>running</option><option>done</option><option>unavailable</option></select>
      <select id="sort"><option value="scene">sort: scene</option><option value="before_auc30">sort: before AUC30</option><option value="after_auc30">sort: after AUC30</option><option value="delta_auc30">sort: delta AUC30</option></select>
      <span class="small"><span class="low-before">*</span> baseline AUC30 &lt; 0.6</span>
    </div>
    <div class="tablewrap">
      <table>
        <thead><tr>
          <th>Scene</th><th>Status</th><th>Before AUC3</th><th>After AUC3</th><th>Delta AUC3</th><th>Before AUC30</th><th>After AUC30</th><th>Delta AUC30</th>
        </tr></thead>
        <tbody id="tbody"><tr><td colspan="8">Loading...</td></tr></tbody>
      </table>
    </div>
  </main>
  <script>
    let payload = null;
    const fmt = v => (v === null || v === undefined || Number.isNaN(v)) ? '<span class="na">NA</span>' : Number(v).toFixed(4);
    const delta = v => {
      if (v === null || v === undefined || Number.isNaN(v)) return '<span class="na">NA</span>';
      const cls = v >= 0 ? 'good' : 'bad';
      const sign = v >= 0 ? '+' : '';
      return `<span class="${cls}">${sign}${Number(v).toFixed(4)}</span>`;
    };
    const rowClass = r => {
      if (r.after_auc3 === null || r.after_auc3 === undefined || r.after_auc30 === null || r.after_auc30 === undefined) return '';
      if (r.status === 'unavailable') return 'row-unavailable';
      if (r.delta_auc3 >= 0 && r.delta_auc30 >= 0) return 'row-good';
      if (r.delta_auc3 < 0 && r.delta_auc30 < 0) return 'row-bad';
      return 'row-mixed';
    };
    const pill = s => `<span class="pill ${s}">${s}</span>`;
    const sceneName = r => `${r.scene}${r.before_auc30 !== null && r.before_auc30 !== undefined && r.before_auc30 < 0.6 ? '<span class="low-before">*</span>' : ''}`;
    const baseName = p => (p || '').split('/').filter(Boolean).pop() || 'NA';
    async function load() {
      const res = await fetch('/data?ts=' + Date.now());
      payload = await res.json();
      render();
    }
    function render() {
      if (!payload) return;
      document.getElementById('generated').textContent = 'Updated: ' + payload.generated_at;
      const beforeRoot = baseName(payload.before_root);
      const afterRoot = baseName(payload.after_root);
      document.getElementById('beforeRoot').textContent = beforeRoot;
      document.getElementById('afterRoot').textContent = afterRoot;
      document.getElementById('beforeRootSmall').textContent = beforeRoot;
      document.getElementById('afterRootSmall').textContent = afterRoot;
      document.getElementById('complete').textContent = `${payload.complete}/${payload.total}`;
      document.getElementById('completeSmall').textContent = `unavailable excluded: ${payload.unavailable || 0}`;
      const mean = payload.mean_row || {};
      document.getElementById('beforeMean30').innerHTML = fmt(mean.before_auc30);
      document.getElementById('afterMean30').innerHTML = fmt(mean.after_auc30);
      document.getElementById('afterMtime').textContent = payload.after_pose_mtime || 'NA';
      const q = document.getElementById('search').value.trim().toLowerCase();
      const st = document.getElementById('status').value;
      const sortKey = document.getElementById('sort').value;
      let rows = payload.rows.filter(r => (!q || r.scene.toLowerCase().includes(q)) && (!st || r.status === st));
      rows.sort((a,b) => {
        if (sortKey === 'scene') return a.scene.localeCompare(b.scene);
        const av = a[sortKey], bv = b[sortKey];
        if (av === null || av === undefined) return 1;
        if (bv === null || bv === undefined) return -1;
        return bv - av;
      });
      const tableRows = rows.map(r => `
        <tr class="${r.status === 'unavailable' ? 'row-unavailable' : rowClass(r)}">
          <td>${sceneName(r)}</td><td>${pill(r.status)}</td>
          <td>${fmt(r.before_auc3)}</td><td>${fmt(r.after_auc3)}</td><td>${delta(r.delta_auc3)}</td>
          <td>${fmt(r.before_auc30)}</td><td>${fmt(r.after_auc30)}</td><td>${delta(r.delta_auc30)}</td>
        </tr>`).join('');
      const meanRow = `
        <tr class="mean-row ${rowClass(mean)}">
          <td>${mean.scene || 'MEAN'}</td><td>${pill(mean.status || 'pending')}</td>
          <td>${fmt(mean.before_auc3)}</td><td>${fmt(mean.after_auc3)}</td><td>${delta(mean.delta_auc3)}</td>
          <td>${fmt(mean.before_auc30)}</td><td>${fmt(mean.after_auc30)}</td><td>${delta(mean.delta_auc30)}</td>
        </tr>`;
      document.getElementById('tbody').innerHTML = tableRows + meanRow;
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
    before_root: Path
    after_root: Path
    input_root: Path

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/data":
            data = build_payload(self.before_root, self.after_root, self.input_root)
            body = json.dumps(data).encode("utf-8")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", default="workspace/evaluation_scannet_default_full")
    parser.add_argument("--after-root", default="workspace/evaluation_scannet_default_full2")
    parser.add_argument("--input-root", default="data/scannet")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    Handler.before_root = Path(args.before_root).resolve()
    Handler.after_root = Path(args.after_root).resolve()
    Handler.input_root = Path(args.input_root).resolve()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving http://{args.host}:{args.port}", flush=True)
    print(f"pid {os.getpid()}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
