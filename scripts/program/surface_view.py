from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.parse

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PROGRAM_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = PROGRAM_DIR.parent
for import_dir in (PROGRAM_DIR, SCRIPT_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from orient_annotate import INSTANCE_VERTEX_DTYPE, read_binary_ply_header
from view_pcd import DEFAULT_HOST, DEFAULT_PCD_DIR, DEFAULT_POINT_SIZE, read_ply_header


DEFAULT_PORT = 8094
RGB_FILE = "rgb.ply"
INSTANCE_FILE = "instance.ply"
NORMALS_FILE = "normals.ply"
ORIENT_FILE = "orient.npz"
DEFAULT_ALIGNMENT_THRESHOLD = 0.5
SURFACE_FNS = ("frontside", "backside")
NORMAL_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
    ]
)


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Surface Side Viewer</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #111; color: #eee; font-family: ui-sans-serif, system-ui, sans-serif; }
    #bar { position: fixed; left: 14px; top: 14px; z-index: 10; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 10px 12px; background: rgba(20,20,20,.9); border: 1px solid #444; border-radius: 10px; }
    input, select, button { color: #eee; background: #222; border: 1px solid #555; border-radius: 6px; padding: 6px 8px; }
    button { cursor: pointer; }
    button:disabled { opacity: .45; cursor: wait; }
    #status { position: fixed; left: 14px; bottom: 14px; z-index: 10; max-width: min(900px, calc(100vw - 28px)); padding: 8px 10px; background: rgba(20,20,20,.84); border: 1px solid #444; border-radius: 8px; font-size: 13px; }
    #legend { position: fixed; right: 18px; bottom: 18px; z-index: 10; padding: 10px; background: rgba(20,20,20,.84); border: 1px solid #444; border-radius: 10px; font-size: 13px; }
    .swatch { display: inline-block; width: 12px; height: 12px; margin-right: 6px; border: 1px solid #777; vertical-align: -1px; }
    .green { background: #00ff00; }
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
    <label>Instance <input id="instanceId" type="number" min="0" step="1" value="0" /></label>
    <label>Function
      <select id="surfaceFn">
        <option value="frontside">frontside</option>
        <option value="backside">backside</option>
      </select>
    </label>
    <label>Normal dot thresh <input id="threshold" type="number" min="-1" max="1" step="0.05" value="__THRESHOLD__" /></label>
    <button id="submit">Submit</button>
    <button id="reset">Reset RGB</button>
    <label>Point size <input id="pointSize" type="number" min="0.001" step="0.001" value="__POINT_SIZE__" /></label>
  </div>
  <div id="status">Loading RGB pointcloud...</div>
  <div id="legend"><span class="swatch green"></span>selected side points</div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

    const statusEl = document.getElementById('status');
    const instanceIdEl = document.getElementById('instanceId');
    const surfaceFnEl = document.getElementById('surfaceFn');
    const thresholdEl = document.getElementById('threshold');
    const submitEl = document.getElementById('submit');
    const resetEl = document.getElementById('reset');
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

    let pcd = null;
    let originalColors = null;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function setBusy(isBusy) {
      submitEl.disabled = isBusy;
      resetEl.disabled = isBusy;
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

    function applyMask(mask) {
      const colorAttr = pcd.geometry.attributes.color;
      const colors = colorAttr.array;
      colors.set(originalColors);
      if (mask.length * 3 !== colors.length) {
        throw new Error(`Mask length mismatch: got ${mask.length}, expected ${colors.length / 3}`);
      }
      for (let i = 0; i < mask.length; i++) {
        if (!mask[i]) continue;
        const j = i * 3;
        colors[j] = 0.0;
        colors[j + 1] = 1.0;
        colors[j + 2] = 0.0;
      }
      colorAttr.needsUpdate = true;
    }

    function resetRgb() {
      if (!pcd || !originalColors) return;
      pcd.geometry.attributes.color.array.set(originalColors);
      pcd.geometry.attributes.color.needsUpdate = true;
      setStatus(`RGB view: ${pcd.geometry.attributes.position.count.toLocaleString()} points`);
    }

    async function submitQuery() {
      const instanceId = Number(instanceIdEl.value);
      const surfaceFn = surfaceFnEl.value;
      const threshold = Number(thresholdEl.value);
      if (!Number.isInteger(instanceId) || instanceId < 0) {
        setStatus('Instance id must be a non-negative integer.');
        return;
      }
      if (threshold < -1.0 || threshold > 1.0) {
        setStatus('Normal dot threshold must be in [-1, 1].');
        return;
      }
      setBusy(true);
      setStatus(`${surfaceFn}(instance ${instanceId})...`);
      try {
        const response = await fetch('/query', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({instance_id: instanceId, surface_fn: surfaceFn, threshold}),
        });
        if (!response.ok) throw new Error(await response.text());
        const meta = JSON.parse(decodeURIComponent(response.headers.get('X-Surface-Result')));
        const mask = new Uint8Array(await response.arrayBuffer());
        applyMask(mask);
        setStatus(
          `${surfaceFn}(instance ${instanceId}): ${meta.selected_point_count.toLocaleString()} / ` +
          `${meta.instance_point_count.toLocaleString()} points selected. ` +
          `dot min/mean/max=${meta.dot_min.toFixed(3)}/${meta.dot_mean.toFixed(3)}/${meta.dot_max.toFixed(3)}, ` +
          `threshold=${meta.threshold.toFixed(3)}`
        );
      } catch (err) {
        console.error(err);
        setStatus(`Query failed: ${err.message || err}`);
      } finally {
        setBusy(false);
      }
    }

    function loadRgb() {
      const loader = new PLYLoader();
      loader.load(
        '/ply/rgb.ply',
        (geometry) => {
          if (!geometry.attributes.color) throw new Error('rgb.ply has no vertex colors');
          originalColors = geometry.attributes.color.array.slice();
          const material = new THREE.PointsMaterial({
            size: Number(pointSizeEl.value),
            vertexColors: true,
            sizeAttenuation: true,
          });
          pcd = new THREE.Points(geometry, material);
          scene.add(pcd);
          fitCamera(geometry);
          resetRgb();
        },
        (xhr) => {
          if (xhr.lengthComputable) setStatus(`Loading RGB: ${Math.round(xhr.loaded / xhr.total * 100)}%`);
        },
        (err) => {
          console.error(err);
          setStatus(`Failed to load RGB: ${err.message || err}`);
        },
      );
    }

    submitEl.addEventListener('click', submitQuery);
    resetEl.addEventListener('click', resetRgb);
    instanceIdEl.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') submitQuery();
    });
    pointSizeEl.addEventListener('change', () => {
      if (pcd) pcd.material.size = Number(pointSizeEl.value);
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

    loadRgb();
    animate();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a surface-side pointcloud viewer.")
    parser.add_argument("pcd_dir", nargs="?", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_ALIGNMENT_THRESHOLD)
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("Cannot normalize near-zero vector")
    return (vector / norm).astype(np.float32)


def load_instance_vertices(pcd_dir: Path) -> np.ndarray:
    path = pcd_dir / INSTANCE_FILE
    vertex_count, properties, body_offset = read_binary_ply_header(path)
    expected = [
        "property float x",
        "property float y",
        "property float z",
        "property int instance_id",
    ]
    if properties != expected:
        raise ValueError(f"Unexpected instance PLY schema in {path}: {properties}")
    with path.open("rb") as f:
        f.seek(body_offset)
        vertices = np.fromfile(f, dtype=INSTANCE_VERTEX_DTYPE, count=vertex_count)
    if len(vertices) != vertex_count:
        raise ValueError(f"Instance PLY ended early: expected {vertex_count}, found {len(vertices)}")
    return vertices


def load_normals(pcd_dir: Path) -> np.ndarray:
    path = pcd_dir / NORMALS_FILE
    vertex_count, properties, body_offset = read_binary_ply_header(path)
    expected = [
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
    ]
    if properties != expected:
        raise ValueError(f"Unexpected normals PLY schema in {path}: {properties}")
    with path.open("rb") as f:
        f.seek(body_offset)
        vertices = np.fromfile(f, dtype=NORMAL_VERTEX_DTYPE, count=vertex_count)
    if len(vertices) != vertex_count:
        raise ValueError(f"Normals PLY ended early: expected {vertex_count}, found {len(vertices)}")
    normals = np.column_stack([vertices["nx"], vertices["ny"], vertices["nz"]]).astype(np.float32, copy=False)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-8)


class SurfaceEngine:
    def __init__(self, pcd_dir: Path) -> None:
        self.pcd_dir = pcd_dir
        instance_vertices = load_instance_vertices(pcd_dir)
        self.instance_ids = instance_vertices["instance_id"].astype(np.int32, copy=False)
        self.normals = load_normals(pcd_dir)
        if len(self.instance_ids) != len(self.normals):
            raise ValueError(f"point count mismatch: instance={len(self.instance_ids):,} normals={len(self.normals):,}")
        unique_ids = np.unique(self.instance_ids[self.instance_ids >= 0])
        self.indices_by_instance = {
            int(instance_id): np.flatnonzero(self.instance_ids == int(instance_id))
            for instance_id in unique_ids.tolist()
        }
        self.orient_axes = self._load_orient_axes(pcd_dir)

    def _load_orient_axes(self, pcd_dir: Path) -> dict[int, np.ndarray]:
        path = pcd_dir / ORIENT_FILE
        if not path.exists():
            raise FileNotFoundError(f"Missing required orientation file: {path}")
        data = np.load(path)
        instance_ids = data["instance_ids"].astype(np.int32, copy=False)
        axes = data["axes"].astype(np.float32, copy=False)
        return {int(instance_id): axes[pos] for pos, instance_id in enumerate(instance_ids.tolist())}

    def query(self, instance_id: int, surface_fn: str, threshold: float) -> tuple[np.ndarray, dict[str, object]]:
        if surface_fn not in SURFACE_FNS:
            raise ValueError(f"Unknown surface function {surface_fn}; expected one of {SURFACE_FNS}")
        if instance_id not in self.indices_by_instance:
            raise ValueError(f"Unknown instance id: {instance_id}")
        if instance_id not in self.orient_axes:
            raise ValueError(f"No orientation annotation for instance id: {instance_id}")
        if threshold < -1.0 or threshold > 1.0:
            raise ValueError(f"threshold must be in [-1, 1], got {threshold}")

        direction = normalize(self.orient_axes[instance_id][0])
        if surface_fn == "backside":
            direction = -direction
        indices = self.indices_by_instance[instance_id]
        dots = self.normals[indices] @ direction
        selected_indices = indices[dots >= threshold]
        mask = np.zeros(len(self.instance_ids), dtype=np.uint8)
        mask[selected_indices] = 1
        meta = {
            "instance_id": int(instance_id),
            "surface_fn": surface_fn,
            "threshold": float(threshold),
            "instance_point_count": int(len(indices)),
            "selected_point_count": int(len(selected_indices)),
            "dot_min": float(np.min(dots)),
            "dot_mean": float(np.mean(dots)),
            "dot_max": float(np.max(dots)),
        }
        return mask, meta


def validate_inputs(pcd_dir: Path) -> tuple[int, SurfaceEngine]:
    rgb_path = pcd_dir / RGB_FILE
    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing required RGB PLY: {rgb_path}")
    vertex_count, properties = read_ply_header(rgb_path)
    required = ["property uchar red", "property uchar green", "property uchar blue"]
    if vertex_count <= 0 or not all(prop in properties for prop in required):
        raise ValueError(f"{rgb_path} must be a colored binary PLY")
    engine = SurfaceEngine(pcd_dir)
    if len(engine.instance_ids) != vertex_count:
        raise ValueError(f"point count mismatch: rgb={vertex_count:,} instance={len(engine.instance_ids):,}")
    return vertex_count, engine


def make_handler(pcd_dir: Path, point_size: float, threshold: float, vertex_count: int, engine: SurfaceEngine):
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

        def send_plain_error(self, status: int, message: str) -> None:
            data = message.encode("utf-8", errors="replace")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                data = (
                    HTML.replace("__POINT_SIZE__", f"{point_size:g}")
                    .replace("__THRESHOLD__", f"{threshold:g}")
                    .encode("utf-8")
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if parsed.path == "/ply/rgb.ply":
                path = pcd_dir / RGB_FILE
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

            if parsed.path == "/meta":
                payload = {
                    "points": vertex_count,
                    "instances": sorted(engine.indices_by_instance.keys()),
                    "surface_fns": list(SURFACE_FNS),
                }
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/query":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                mask, meta = engine.query(
                    int(request["instance_id"]),
                    str(request["surface_fn"]),
                    float(request["threshold"]),
                )
            except Exception as exc:
                self.send_plain_error(500, f"Query failed: {exc}")
                return

            payload = mask.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Surface-Result", urllib.parse.quote(json.dumps(meta)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main() -> None:
    args = parse_args()
    pcd_dir = args.pcd_dir.expanduser().resolve()
    if not pcd_dir.exists():
        raise FileNotFoundError(f"Missing PCD dir: {pcd_dir}")
    vertex_count, engine = validate_inputs(pcd_dir)
    server = ThreadingHTTPServer(
        (args.host, int(args.port)),
        make_handler(pcd_dir, float(args.point_size), float(args.threshold), vertex_count, engine),
    )
    print(f"Serving {pcd_dir}")
    print(f"Open: http://127.0.0.1:{args.port}/")
    print("Stop with Ctrl-C")
    server.serve_forever()


if __name__ == "__main__":
    main()
