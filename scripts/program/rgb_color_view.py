from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.parse

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

PROGRAM_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = PROGRAM_DIR.parent
PROMPT_DIR = PROGRAM_DIR / "prompts"
for import_dir in (PROGRAM_DIR, SCRIPT_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from openai_utils import DEFAULT_LLM_MODEL, llm_json_call
from orient_annotate import read_binary_ply_header
from view_pcd import DEFAULT_HOST, DEFAULT_PCD_DIR, DEFAULT_POINT_SIZE, read_ply_header


DEFAULT_PORT = 8093
RGB_FILE = "rgb.ply"
DEFAULT_THRESHOLD = 0.0
DELTA_E_SCALE = 100.0
RGB_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)
RGB_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "red": {"type": "integer", "minimum": 0, "maximum": 255},
        "green": {"type": "integer", "minimum": 0, "maximum": 255},
        "blue": {"type": "integer", "minimum": 0, "maximum": 255},
    },
    "required": ["red", "green", "blue"],
}
COLOR_INSTRUCTIONS = (PROMPT_DIR / "rgb_color_instructions.txt").read_text().strip()
COLOR_PROMPT_TEMPLATE = (PROMPT_DIR / "rgb_color_prompt.txt").read_text()
HEATMAP_STOPS = np.asarray(
    [
        [49, 54, 149],
        [69, 117, 180],
        [116, 173, 209],
        [171, 217, 233],
        [224, 243, 248],
        [254, 224, 144],
        [253, 174, 97],
        [244, 109, 67],
        [215, 48, 39],
        [165, 0, 38],
    ],
    dtype=np.float32,
)
HEATMAP_CSS = ", ".join(f"rgb({int(r)}, {int(g)}, {int(b)})" for r, g, b in HEATMAP_STOPS)


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RGB Color Similarity PCD Viewer</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #111; color: #eee; font-family: ui-sans-serif, system-ui, sans-serif; }
    #bar { position: fixed; left: 14px; top: 14px; z-index: 10; display: flex; gap: 10px; align-items: center; padding: 10px 12px; background: rgba(20,20,20,.9); border: 1px solid #444; border-radius: 10px; }
    input, button { color: #eee; background: #222; border: 1px solid #555; border-radius: 6px; padding: 6px 8px; }
    button { cursor: pointer; }
    button:disabled { opacity: .45; cursor: wait; }
    #query { width: 220px; }
    #status { position: fixed; left: 14px; bottom: 14px; z-index: 10; max-width: min(900px, calc(100vw - 28px)); padding: 8px 10px; background: rgba(20,20,20,.84); border: 1px solid #444; border-radius: 8px; font-size: 13px; }
    #colorbar { display: none; position: fixed; right: 18px; bottom: 18px; z-index: 10; width: 320px; padding: 10px; background: rgba(20,20,20,.84); border: 1px solid #444; border-radius: 10px; }
    #gradient { height: 18px; border: 1px solid #222; border-radius: 4px; background: linear-gradient(to right, __HEATMAP_CSS__); }
    #ticks { display: flex; justify-content: space-between; margin-top: 6px; font-size: 12px; color: #ddd; }
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
    <input id="query" type="text" placeholder="red, dark blue, beige, ..." />
    <button id="submit">Submit</button>
    <button id="reset">Reset RGB</button>
    <label>Point size <input id="pointSize" type="number" min="0.001" step="0.001" value="__POINT_SIZE__" /></label>
    <label>Thresh <input id="threshold" type="number" min="0" max="1" step="0.01" value="__THRESHOLD__" /></label>
  </div>
  <div id="status">Loading RGB pointcloud...</div>
  <div id="colorbar">
    <div id="gradient"></div>
    <div id="ticks"><span>0.000</span><span>0.500</span><span>1.000</span></div>
  </div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

    const statusEl = document.getElementById('status');
    const queryEl = document.getElementById('query');
    const submitEl = document.getElementById('submit');
    const resetEl = document.getElementById('reset');
    const pointSizeEl = document.getElementById('pointSize');
    const thresholdEl = document.getElementById('threshold');
    const colorbarEl = document.getElementById('colorbar');

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
      queryEl.disabled = isBusy;
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

    function heatmapColor(value) {
      const stops = [
        [49, 54, 149], [69, 117, 180], [116, 173, 209], [171, 217, 233], [224, 243, 248],
        [254, 224, 144], [253, 174, 97], [244, 109, 67], [215, 48, 39], [165, 0, 38],
      ];
      const t = Math.max(0, Math.min(1, value));
      const scaled = t * (stops.length - 1);
      const low = Math.floor(scaled);
      const high = Math.min(low + 1, stops.length - 1);
      const frac = scaled - low;
      return [
        Math.round(stops[low][0] * (1 - frac) + stops[high][0] * frac) / 255.0,
        Math.round(stops[low][1] * (1 - frac) + stops[high][1] * frac) / 255.0,
        Math.round(stops[low][2] * (1 - frac) + stops[high][2] * frac) / 255.0,
      ];
    }

    function applyScores(scores) {
      const colorAttr = pcd.geometry.attributes.color;
      const colors = colorAttr.array;
      if (scores.length * 3 !== colors.length) {
        throw new Error(`Score length mismatch: got ${scores.length}, expected ${colors.length / 3}`);
      }
      for (let i = 0; i < scores.length; i++) {
        const j = i * 3;
        const color = heatmapColor(Number.isFinite(scores[i]) ? scores[i] : 0.0);
        colors[j] = color[0];
        colors[j + 1] = color[1];
        colors[j + 2] = color[2];
      }
      colorAttr.needsUpdate = true;
    }

    function resetRgb() {
      if (!pcd || !originalColors) return;
      pcd.geometry.attributes.color.array.set(originalColors);
      pcd.geometry.attributes.color.needsUpdate = true;
      colorbarEl.style.display = 'none';
      setStatus(`RGB view: ${pcd.geometry.attributes.position.count.toLocaleString()} points`);
    }

    async function submitQuery() {
      const text = queryEl.value.trim();
      if (!text) {
        setStatus('Enter a color first.');
        return;
      }
      const threshold = Number(thresholdEl.value || 0.0);
      setBusy(true);
      setStatus(`Parsing "${text}" to RGB and computing CIEDE2000 similarity...`);
      try {
        const response = await fetch('/query', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({text, threshold}),
        });
        if (!response.ok) throw new Error(await response.text());
        const meta = JSON.parse(decodeURIComponent(response.headers.get('X-Color-Result')));
        const scores = new Float32Array(await response.arrayBuffer());
        applyScores(scores);
        colorbarEl.style.display = 'block';
        setStatus(
          `"${text}" -> RGB(${meta.rgb.join(', ')}). ` +
          `Similarity=1-clamp(deltaE/100); threshold=${meta.threshold.toFixed(3)}; ` +
          `${meta.passing_point_count.toLocaleString()} / ${meta.point_count.toLocaleString()} points pass. ` +
          `deltaE min/mean/max=${meta.delta_e_min.toFixed(2)}/${meta.delta_e_mean.toFixed(2)}/${meta.delta_e_max.toFixed(2)}`
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
    queryEl.addEventListener('keydown', (event) => {
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
    parser = argparse.ArgumentParser(description="Serve an RGB pointcloud viewer with LLM-parsed color similarity heatmaps.")
    parser.add_argument("pcd_dir", nargs="?", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    return parser.parse_args()


def load_rgb_vertices(pcd_dir: Path) -> np.ndarray:
    path = pcd_dir / RGB_FILE
    vertex_count, properties, body_offset = read_binary_ply_header(path)
    expected = [
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if properties != expected:
        raise ValueError(f"Unexpected RGB PLY schema in {path}: {properties}")
    with path.open("rb") as f:
        f.seek(body_offset)
        vertices = np.fromfile(f, dtype=RGB_VERTEX_DTYPE, count=vertex_count)
    if len(vertices) != vertex_count:
        raise ValueError(f"RGB PLY ended early: expected {vertex_count}, found {len(vertices)}")
    return vertices


class ColorScorer:
    def __init__(self, pcd_dir: Path, model: str) -> None:
        self.pcd_dir = pcd_dir
        self.model = model
        vertices = load_rgb_vertices(pcd_dir)
        rgb = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]).astype(np.float32) / 255.0
        print(f"Precomputing Lab colors for {len(rgb):,} RGB points")
        self.point_lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float32)

    def parse_color(self, text: str) -> tuple[int, int, int]:
        result = llm_json_call(
            COLOR_PROMPT_TEMPLATE.replace("{{color_text}}", text),
            schema=RGB_SCHEMA,
            schema_name="rgb_color",
            model=self.model,
            instructions=COLOR_INSTRUCTIONS,
            max_output_tokens=256,
        )
        return int(result["red"]), int(result["green"]), int(result["blue"])

    def score(self, text: str, threshold: float) -> tuple[np.ndarray, dict[str, object]]:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        rgb = self.parse_color(text)
        query = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
        query_lab = rgb2lab(query).reshape(1, 3).astype(np.float32)
        delta_e = deltaE_ciede2000(self.point_lab, query_lab).astype(np.float32)
        scores = 1.0 - np.clip(delta_e / DELTA_E_SCALE, 0.0, 1.0)
        scores[scores < threshold] = 0.0
        meta = {
            "rgb": list(rgb),
            "threshold": float(threshold),
            "point_count": int(len(scores)),
            "passing_point_count": int(np.count_nonzero(scores > 0.0)),
            "delta_e_min": float(np.min(delta_e)),
            "delta_e_mean": float(np.mean(delta_e)),
            "delta_e_max": float(np.max(delta_e)),
            "score_min": 0.0,
            "score_max": 1.0,
        }
        return scores.astype(np.float32, copy=False), meta


def validate_inputs(pcd_dir: Path, model: str) -> tuple[int, ColorScorer]:
    rgb_path = pcd_dir / RGB_FILE
    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing required RGB PLY: {rgb_path}")
    vertex_count, properties = read_ply_header(rgb_path)
    required = ["property uchar red", "property uchar green", "property uchar blue"]
    if vertex_count <= 0 or not all(prop in properties for prop in required):
        raise ValueError(f"{rgb_path} must be a colored binary PLY")
    return vertex_count, ColorScorer(pcd_dir, model)


def make_handler(pcd_dir: Path, point_size: float, vertex_count: int, scorer: ColorScorer):
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
                    .replace("__THRESHOLD__", f"{DEFAULT_THRESHOLD:g}")
                    .replace("__HEATMAP_CSS__", HEATMAP_CSS)
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
                payload = {"points": vertex_count, "point_size": point_size}
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
                text = str(request["text"]).strip()
                if not text:
                    raise ValueError("empty color text")
                scores, meta = scorer.score(text, float(request.get("threshold", DEFAULT_THRESHOLD)))
            except Exception as exc:
                self.send_plain_error(500, f"Query failed: {exc}")
                return

            payload = scores.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Color-Result", urllib.parse.quote(json.dumps(meta)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main() -> None:
    args = parse_args()
    pcd_dir = args.pcd_dir.expanduser().resolve()
    if not pcd_dir.exists():
        raise FileNotFoundError(f"Missing PCD dir: {pcd_dir}")
    vertex_count, scorer = validate_inputs(pcd_dir, str(args.model))
    server = ThreadingHTTPServer(
        (args.host, int(args.port)),
        make_handler(pcd_dir, float(args.point_size), vertex_count, scorer),
    )
    print(f"Serving {pcd_dir}")
    print(f"Open: http://127.0.0.1:{args.port}/")
    print("Stop with Ctrl-C")
    server.serve_forever()


if __name__ == "__main__":
    main()
