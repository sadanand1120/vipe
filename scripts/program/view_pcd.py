from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.parse

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


DEFAULT_PCD_DIR = Path("tmpgt/pcd")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8088
DEFAULT_POINT_SIZE = 0.02


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PCD Viewer</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #111; color: #eee; font-family: ui-sans-serif, system-ui, sans-serif; }
    #bar { position: fixed; left: 14px; top: 14px; z-index: 10; display: flex; gap: 10px; align-items: center; padding: 10px 12px; background: rgba(20,20,20,.88); border: 1px solid #444; border-radius: 10px; }
    select, input, label { color: #eee; background: #222; border: 1px solid #555; border-radius: 6px; padding: 5px 7px; }
    label { border: 0; background: transparent; padding: 0; }
    #status { position: fixed; left: 14px; bottom: 14px; z-index: 10; padding: 8px 10px; background: rgba(20,20,20,.82); border: 1px solid #444; border-radius: 8px; font-size: 13px; }
    canvas { display: block; }
  </style>
  <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
      }
    }
  </script>
</head>
<body>
  <div id="bar">
    <label>View <select id="view"></select></label>
    <label>Point size <input id="pointSize" type="number" min="0.001" step="0.001" value="0.02" /></label>
  </div>
  <div id="status">Loading...</div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

    const statusEl = document.getElementById('status');
    const viewEl = document.getElementById('view');
    const pointSizeEl = document.getElementById('pointSize');

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.001, 10000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    let currentObject = null;
    let loadedViews = new Map();
    let orientPayload = null;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function fitCamera(geometry) {
      geometry.computeBoundingSphere();
      const sphere = geometry.boundingSphere;
      const center = sphere.center;
      const radius = Math.max(sphere.radius, 1e-3);
      controls.target.copy(center);
      camera.near = Math.max(radius / 1000, 0.001);
      camera.far = radius * 1000;
      camera.position.set(center.x + radius * 1.5, center.y - radius * 2.0, center.z + radius * 1.2);
      camera.updateProjectionMatrix();
      controls.update();
    }

    function setVisibleView(file) {
      if (currentObject) currentObject.visible = false;
      currentObject = loadedViews.get(file);
      if (currentObject) {
        currentObject.visible = true;
        setStatus(currentObject.userData.status);
      }
    }

    function loadPlyView(file) {
      setStatus(`Loading ${file}...`);
      const loader = new PLYLoader();
      return new Promise((resolve, reject) => loader.load(
        `/ply/${encodeURIComponent(file)}`,
        (geometry) => {
          geometry.computeVertexNormals();
          const material = new THREE.PointsMaterial({
            size: Number(pointSizeEl.value),
            vertexColors: Boolean(geometry.attributes.color),
            color: geometry.attributes.color ? 0xffffff : 0xd8d8d8,
            sizeAttenuation: true,
          });
          const points = new THREE.Points(geometry, material);
          points.visible = false;
          points.userData.status = `${file}: ${geometry.attributes.position.count.toLocaleString()} points`;
          scene.add(points);
          loadedViews.set(file, points);
          resolve(points);
        },
        (xhr) => {
          if (xhr.lengthComputable) {
            setStatus(`Loading ${file}: ${Math.round(xhr.loaded / xhr.total * 100)}%`);
          }
        },
        (err) => {
          console.error(err);
          setStatus(`Failed to load ${file}`);
          reject(err);
        },
      ));
    }

    async function loadOrient() {
      const response = await fetch('/orient');
      orientPayload = await response.json();
    }

    function buildOrientLines() {
      const axisLength = orientPayload.axis_length || 0.4;
      const positions = [];
      const colors = [];
      const axisColors = [[1, 0, 0], [0, 1, 0], [0.25, 0.45, 1]];
      for (let i = 0; i < orientPayload.centroids.length; i++) {
        const c = orientPayload.centroids[i];
        const axes = orientPayload.axes[i];
        for (let a = 0; a < 3; a++) {
          positions.push(c[0], c[1], c[2]);
          positions.push(c[0] + axes[a][0] * axisLength, c[1] + axes[a][1] * axisLength, c[2] + axes[a][2] * axisLength);
          colors.push(...axisColors[a], ...axisColors[a]);
        }
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      const material = new THREE.LineBasicMaterial({ vertexColors: true });
      const lines = new THREE.LineSegments(geometry, material);
      lines.userData.status = `orient: ${orientPayload.instance_ids.length.toLocaleString()} orientation axes`;
      return lines;
    }

    function buildOrientOverlay() {
      const rgb = loadedViews.get('rgb.ply');
      if (!rgb) throw new Error('rgb.ply is required for orient overlay');
      const rgbClone = new THREE.Points(rgb.geometry, rgb.material);
      const axes = buildOrientLines();
      const overlay = new THREE.Group();
      overlay.add(rgbClone);
      overlay.add(axes);
      overlay.visible = false;
      overlay.userData.status = `orient: rgb + ${orientPayload.instance_ids.length.toLocaleString()} orientation axes`;
      scene.add(overlay);
      loadedViews.set('orient', overlay);
    }

    async function init() {
      const response = await fetch('/views');
      const payload = await response.json();
      pointSizeEl.value = String(payload.point_size);
      viewEl.disabled = true;
      for (const view of payload.views) {
        const option = document.createElement('option');
        option.value = view.file;
        option.textContent = `${view.name} (${view.points.toLocaleString()})`;
        viewEl.appendChild(option);
      }
      const orientOption = document.createElement('option');
      orientOption.value = 'orient';
      orientOption.textContent = `orient (${payload.orient_instances.toLocaleString()} instances)`;
      viewEl.appendChild(orientOption);
      await loadOrient();
      for (const view of payload.views) {
        await loadPlyView(view.file);
      }
      buildOrientOverlay();
      const initialFile = payload.views.length > 0 ? payload.views[0].file : 'orient';
      fitCamera(loadedViews.get(initialFile).geometry);
      setVisibleView(initialFile);
      viewEl.disabled = false;
    }

    viewEl.addEventListener('change', () => setVisibleView(viewEl.value));
    pointSizeEl.addEventListener('change', () => {
      for (const object of loadedViews.values()) {
        if (object.isPoints) object.material.size = Number(pointSizeEl.value);
      }
    });

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }

    init().catch((err) => {
      console.error(err);
      setStatus(`Viewer init failed: ${err.message || err}`);
      viewEl.disabled = false;
    });
    animate();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a lightweight browser viewer for ViPE/GT PCD bundles.")
    parser.add_argument("pcd_dir", nargs="?", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    return parser.parse_args()


def read_ply_header(path: Path) -> tuple[int, list[str]]:
    vertex_count = 0
    properties: list[str] = []
    with path.open("rb") as f:
        first = f.readline().decode("ascii").strip()
        fmt = f.readline().decode("ascii").strip()
        if first != "ply" or fmt != "format binary_little_endian 1.0":
            return 0, []
        while True:
            raw = f.readline()
            if not raw:
                return 0, []
            line = raw.decode("ascii").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property "):
                properties.append(line)
            elif line == "end_header":
                break
    return vertex_count, properties


def discover_views(pcd_dir: Path) -> list[dict[str, object]]:
    preferred = ["rgb.ply", "normals_viz.ply", "instance_viz.ply", "clip_viz.ply"]
    paths = [pcd_dir / name for name in preferred if (pcd_dir / name).exists()]
    paths.extend(sorted(path for path in pcd_dir.glob("*.ply") if path.name not in preferred))

    views = []
    seen = set()
    for path in paths:
        if path.name in seen:
            continue
        seen.add(path.name)
        vertex_count, properties = read_ply_header(path)
        has_color = all(prop in properties for prop in ["property uchar red", "property uchar green", "property uchar blue"])
        if vertex_count > 0 and has_color:
            views.append({"name": path.stem, "file": path.name, "points": vertex_count})
    return views


def orient_instance_count(pcd_dir: Path) -> int:
    orient_path = pcd_dir / "orient.npz"
    if not orient_path.exists():
        raise FileNotFoundError(f"Missing required orientation file: {orient_path}")
    data = np.load(orient_path)
    return int(len(data["instance_ids"]))


def make_handler(pcd_dir: Path, point_size: float):
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

        def send_json(self, payload: dict[str, object]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                data = HTML.replace('value="0.02"', f'value="{point_size:g}"').encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if parsed.path == "/views":
                self.send_json(
                    {
                        "views": discover_views(pcd_dir),
                        "point_size": point_size,
                        "orient_instances": orient_instance_count(pcd_dir),
                    }
                )
                return

            if parsed.path == "/orient":
                orient_path = pcd_dir / "orient.npz"
                if not orient_path.exists():
                    self.send_error(500, f"Missing required orientation file: {orient_path}")
                    return
                data = np.load(orient_path)
                self.send_json(
                    {
                        "axis_length": float(data["axis_length"]),
                        "instance_ids": data["instance_ids"].astype(int).tolist(),
                        "centroids": data["centroids"].astype(float).tolist(),
                        "axes": data["axes"].astype(float).tolist(),
                    }
                )
                return

            if parsed.path.startswith("/ply/"):
                name = urllib.parse.unquote(parsed.path.removeprefix("/ply/"))
                path = (pcd_dir / name).resolve()
                if path.parent != pcd_dir or not path.exists() or path.suffix != ".ply":
                    self.send_error(404, "PLY not found")
                    return
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(path.stat().st_size))
                self.end_headers()
                with path.open("rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                return

            self.send_error(404)

    return Handler


def main() -> None:
    args = parse_args()
    pcd_dir = args.pcd_dir.expanduser().resolve()
    if not pcd_dir.exists():
        raise FileNotFoundError(f"Missing PCD dir: {pcd_dir}")
    orient_instance_count(pcd_dir)

    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(pcd_dir, float(args.point_size)))
    print(f"Serving {pcd_dir}")
    print(f"Open: http://127.0.0.1:{args.port}/")
    print("Stop with Ctrl-C")
    server.serve_forever()


if __name__ == "__main__":
    main()
