from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
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

from open_vocab_clip_view import ClipScorer, DEFAULT_TEMPERATURE, VlmRejector
from orient_annotate import INSTANCE_VERTEX_DTYPE, read_binary_ply_header
from view_pcd import DEFAULT_HOST, DEFAULT_PCD_DIR, DEFAULT_POINT_SIZE, read_ply_header


DEFAULT_PORT = 8090
RGB_FILE = "rgb.ply"
RELATIONS = ("close_to", "in_front", "on_the_side", "behind", "on_top", "below")


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Spatial Relation PCD Viewer</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #111; color: #eee; font-family: ui-sans-serif, system-ui, sans-serif; }
    #bar { position: fixed; left: 14px; top: 14px; z-index: 10; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 10px 12px; background: rgba(20,20,20,.9); border: 1px solid #444; border-radius: 10px; }
    input, select, button { color: #eee; background: #222; border: 1px solid #555; border-radius: 6px; padding: 6px 8px; }
    button { cursor: pointer; }
    button:disabled { opacity: .45; cursor: wait; }
    #objClass { width: 160px; }
    #status { position: fixed; left: 14px; bottom: 14px; z-index: 10; max-width: min(980px, calc(100vw - 28px)); padding: 8px 10px; background: rgba(20,20,20,.84); border: 1px solid #444; border-radius: 8px; font-size: 13px; }
    #legend { position: fixed; right: 18px; bottom: 18px; z-index: 10; padding: 10px; background: rgba(20,20,20,.84); border: 1px solid #444; border-radius: 10px; font-size: 13px; }
    .swatch { display: inline-block; width: 12px; height: 12px; margin-right: 6px; border: 1px solid #777; vertical-align: -1px; }
    .green { background: #00ff00; }
    .blue { background: #0077ff; }
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
    <label>Relation
      <select id="relation">
        <option value="close_to">close_to</option>
        <option value="in_front">in_front</option>
        <option value="on_the_side">on_the_side</option>
        <option value="behind">behind</option>
        <option value="on_top">on_top</option>
        <option value="below">below</option>
      </select>
    </label>
    <label>Object <input id="objClass" type="text" placeholder="chair, helmet, ..." /></label>
    <label>Dist <input id="dist" type="number" min="0" step="0.05" value="1.0" /></label>
    <label>CLIP thresh <input id="threshold" type="number" min="0" max="1" step="0.01" value="0.95" /></label>
    <label><input id="wholeObj" type="checkbox" checked /> wholeobj</label>
    <button id="submit">Submit</button>
    <button id="reset">Reset RGB</button>
    <label>Point size <input id="pointSize" type="number" min="0.001" step="0.001" value="__POINT_SIZE__" /></label>
  </div>
  <div id="status">Loading RGB pointcloud...</div>
  <div id="legend">
    <div><span class="swatch green"></span>focus instance</div>
    <div><span class="swatch blue"></span>matched object instances</div>
  </div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

    const statusEl = document.getElementById('status');
    const instanceIdEl = document.getElementById('instanceId');
    const relationEl = document.getElementById('relation');
    const objClassEl = document.getElementById('objClass');
    const distEl = document.getElementById('dist');
    const thresholdEl = document.getElementById('threshold');
    const wholeObjEl = document.getElementById('wholeObj');
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
        const j = i * 3;
        if (mask[i] === 1) {
          colors[j] = 0.0;
          colors[j + 1] = 1.0;
          colors[j + 2] = 0.0;
        } else if (mask[i] === 2) {
          colors[j] = 0.0;
          colors[j + 1] = 0.45;
          colors[j + 2] = 1.0;
        }
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
      const relation = relationEl.value;
      const objClass = objClassEl.value.trim();
      const dist = Number(distEl.value);
      const threshold = Number(thresholdEl.value);
      if (!Number.isInteger(instanceId) || instanceId < 0) {
        setStatus('Instance id must be a non-negative integer.');
        return;
      }
      if (!objClass) {
        setStatus('Enter an object class.');
        return;
      }
      if (!(dist > 0.0)) {
        setStatus('Dist must be positive.');
        return;
      }
      setBusy(true);
      const wholeObj = wholeObjEl.checked;
      setStatus(`${relation}(instance ${instanceId}, "${objClass}", dist=${dist}, wholeobj=${wholeObj})...`);
      try {
        const response = await fetch('/query', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({instance_id: instanceId, relation, obj_class: objClass, dist, threshold, whole_obj: wholeObj}),
        });
        if (!response.ok) throw new Error(await response.text());
        const meta = JSON.parse(decodeURIComponent(response.headers.get('X-Relation-Result')));
        const mask = new Uint8Array(await response.arrayBuffer());
        applyMask(mask);
        const ids = meta.matched_instance_ids.length ? meta.matched_instance_ids.join(', ') : 'none';
        setStatus(
          `${relation}(instance ${instanceId}, "${objClass}", dist=${dist}, wholeobj=${meta.whole_obj}): ` +
          `${meta.matched_instance_ids.length} match(es) [${ids}], ${meta.selected_target_point_count.toLocaleString()} blue points. ` +
          `VLM checked ${meta.vlm.checked_count}, rejected ${meta.vlm.rejected_count}. ` +
          `Candidates=${meta.candidate_count}. Top labels: ${meta.top_labels.map((x) => `${x.label}=${Number(x.score).toFixed(3)}`).join(', ')}`
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
    objClassEl.addEventListener('keydown', (event) => {
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
    parser = argparse.ArgumentParser(description="Serve a spatial relation pointcloud viewer.")
    parser.add_argument("pcd_dir", nargs="?", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("Cannot normalize near-zero orientation vector")
    return (vector / norm).astype(np.float32)


def load_instance_vertices(pcd_dir: Path) -> np.ndarray:
    path = pcd_dir / "instance.ply"
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


class RelationEngine:
    def __init__(self, pcd_dir: Path, temperature: float) -> None:
        self.pcd_dir = pcd_dir
        vertices = load_instance_vertices(pcd_dir)
        self.points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32, copy=False)
        self.instance_ids = vertices["instance_id"].astype(np.int32, copy=False)
        self.unique_instance_ids = np.unique(self.instance_ids[self.instance_ids >= 0])
        self.indices_by_instance = {
            int(instance_id): np.flatnonzero(self.instance_ids == int(instance_id))
            for instance_id in self.unique_instance_ids.tolist()
        }
        self.scorer = ClipScorer(pcd_dir, temperature)
        self.rejector = VlmRejector(pcd_dir, self.unique_instance_ids)
        self.orient_axes = self._load_orient_axes(pcd_dir)

    def _load_orient_axes(self, pcd_dir: Path) -> dict[int, np.ndarray]:
        path = pcd_dir / "orient.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing required orientation file: {path}")
        data = np.load(path)
        instance_ids = data["instance_ids"].astype(np.int32, copy=False)
        axes = data["axes"].astype(np.float32, copy=False)
        return {int(instance_id): axes[pos] for pos, instance_id in enumerate(instance_ids.tolist())}

    def orientation_for_indices(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids, counts = np.unique(self.instance_ids[indices], return_counts=True)
        fronts = []
        tops = []
        weights = []
        for instance_id, count in zip(ids.tolist(), counts.tolist(), strict=True):
            if int(instance_id) not in self.orient_axes:
                continue
            axes = self.orient_axes[int(instance_id)]
            fronts.append(normalize(axes[0]))
            tops.append(normalize(axes[2]))
            weights.append(float(count))
        if not weights:
            raise ValueError("No orientation annotation found for focus points")
        weights_arr = np.asarray(weights, dtype=np.float32)
        front = normalize(np.average(np.stack(fronts), axis=0, weights=weights_arr))
        top = np.average(np.stack(tops), axis=0, weights=weights_arr).astype(np.float32)
        top = normalize(top - float(np.dot(top, front)) * front)
        side = normalize(np.cross(top, front))
        return front, side, top

    def point_relation_mask(
        self,
        relation: str,
        nearest_distances: np.ndarray,
        target_points: np.ndarray,
        focus_centroid: np.ndarray,
        front_axis: np.ndarray,
        side_axis: np.ndarray,
        top_axis: np.ndarray,
        dist: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        deltas = target_points - focus_centroid[None, :]
        front = deltas @ front_axis
        side = deltas @ side_axis
        top = deltas @ top_axis
        abs_side = np.abs(side)

        if relation == "close_to":
            horizontal = np.linalg.norm(deltas[:, :2], axis=1)
            point_mask = horizontal <= dist
        else:
            within = nearest_distances <= dist
            if relation == "in_front":
                point_mask = within & (front > 0.0) & (front <= dist)
            elif relation == "behind":
                point_mask = within & (front < 0.0) & (front >= -dist)
            elif relation == "on_the_side":
                point_mask = within & (abs_side > 0.0) & (abs_side <= dist)
            elif relation == "on_top":
                point_mask = within & (top > 0.0) & (top <= dist)
            elif relation == "below":
                point_mask = within & (top < 0.0) & (top >= -dist)
            else:
                raise ValueError(f"Unknown relation: {relation}")

        if point_mask.any():
            selected = point_mask
        else:
            selected = slice(None)
        stats = {
            "nearest_distance": float(np.min(nearest_distances)),
            "horizontal_distance": float(np.min(np.linalg.norm(deltas[:, :2], axis=1))),
            "front": float(np.mean(front[selected])),
            "side": float(np.mean(side[selected])),
            "top": float(np.mean(top[selected])),
        }
        return point_mask, stats

    def query(
        self,
        *,
        focus_instance_id: int,
        relation: str,
        obj_class: str,
        dist: float,
        threshold: float,
        whole_obj: bool,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if relation not in RELATIONS:
            raise ValueError(f"Unknown relation {relation}; expected one of {RELATIONS}")
        if focus_instance_id not in self.indices_by_instance:
            raise ValueError(f"Unknown focus instance id: {focus_instance_id}")
        if dist <= 0.0:
            raise ValueError(f"dist must be positive, got {dist}")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        _, instance_scores, _, _, top_labels = self.scorer.score(obj_class)
        vlm_results = asyncio.run(self.rejector.reject(obj_class, instance_scores, threshold))
        rejected_ids = set(int(instance_id) for instance_id in vlm_results["rejected_instance_ids"])
        candidate_ids = [
            int(instance_id)
            for instance_id, score in instance_scores.items()
            if int(instance_id) != focus_instance_id and float(score) >= threshold and int(instance_id) not in rejected_ids
        ]
        candidate_ids.sort(key=lambda instance_id: (-float(instance_scores[instance_id]), instance_id))

        focus_indices = self.indices_by_instance[focus_instance_id]
        focus_points = self.points[focus_indices]
        focus_centroid = focus_points.mean(axis=0)
        front, side, top = self.orientation_for_indices(focus_indices)
        from scipy.spatial import cKDTree

        workers = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
        focus_tree = cKDTree(focus_points)

        mask = np.zeros(len(self.points), dtype=np.uint8)
        mask[focus_indices] = 1
        matched = []
        selected_target_point_count = 0
        for candidate_id in candidate_ids:
            target_indices = self.indices_by_instance[candidate_id]
            target_points = self.points[target_indices]
            nearest_distances, _ = focus_tree.query(target_points, k=1, workers=workers)
            point_mask, stats = self.point_relation_mask(
                relation,
                nearest_distances,
                target_points,
                focus_centroid,
                front,
                side,
                top,
                dist,
            )
            pass_count = int(np.count_nonzero(point_mask))
            if pass_count:
                selected_indices = target_indices if whole_obj else target_indices[point_mask]
                mask[selected_indices] = 2
                selected_target_point_count += int(len(selected_indices))
                matched.append(
                    {
                        "instance_id": int(candidate_id),
                        "score": float(instance_scores[candidate_id]),
                        "pass_point_count": pass_count,
                        "selected_point_count": int(len(selected_indices)),
                        **stats,
                    }
                )

        meta = {
            "focus_instance_id": int(focus_instance_id),
            "focus_point_count": int(len(focus_indices)),
            "relation": relation,
            "obj_class": obj_class,
            "dist": float(dist),
            "threshold": float(threshold),
            "whole_obj": bool(whole_obj),
            "candidate_count": len(candidate_ids),
            "vlm": {
                "checked_count": int(vlm_results["checked_count"]),
                "rejected_count": int(vlm_results["rejected_count"]),
                "rejected_instance_ids": vlm_results["rejected_instance_ids"],
            },
            "matched_instance_ids": [item["instance_id"] for item in matched],
            "selected_target_point_count": int(selected_target_point_count),
            "matched": matched,
            "top_labels": top_labels,
        }
        return mask, meta


def validate_inputs(pcd_dir: Path, temperature: float) -> tuple[int, RelationEngine]:
    rgb_path = pcd_dir / RGB_FILE
    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing required RGB PLY: {rgb_path}")
    vertex_count, properties = read_ply_header(rgb_path)
    required = ["property uchar red", "property uchar green", "property uchar blue"]
    if vertex_count <= 0 or not all(prop in properties for prop in required):
        raise ValueError(f"{rgb_path} must be a colored binary PLY")
    engine = RelationEngine(pcd_dir, temperature)
    if len(engine.points) != vertex_count:
        raise ValueError(f"point count mismatch: rgb={vertex_count:,} instance={len(engine.points):,}")
    return vertex_count, engine


def make_handler(pcd_dir: Path, point_size: float, vertex_count: int, engine: RelationEngine):
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
                data = HTML.replace("__POINT_SIZE__", f"{point_size:g}").encode("utf-8")
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
                    "instances": engine.unique_instance_ids.astype(int).tolist(),
                    "relations": list(RELATIONS),
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
                    focus_instance_id=int(request["instance_id"]),
                    relation=str(request["relation"]),
                    obj_class=str(request["obj_class"]).strip(),
                    dist=float(request["dist"]),
                    threshold=float(request["threshold"]),
                    whole_obj=bool(request.get("whole_obj", True)),
                )
            except Exception as exc:
                self.send_plain_error(500, f"Query failed: {exc}")
                return

            payload = mask.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Relation-Result", urllib.parse.quote(json.dumps(meta)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main() -> None:
    args = parse_args()
    pcd_dir = args.pcd_dir.expanduser().resolve()
    if not pcd_dir.exists():
        raise FileNotFoundError(f"Missing PCD dir: {pcd_dir}")
    vertex_count, engine = validate_inputs(pcd_dir, float(args.temperature))
    server = ThreadingHTTPServer(
        (args.host, int(args.port)),
        make_handler(pcd_dir, float(args.point_size), vertex_count, engine),
    )
    print(f"Serving {pcd_dir}")
    print(f"Open: http://127.0.0.1:{args.port}/")
    print("Stop with Ctrl-C")
    server.serve_forever()


if __name__ == "__main__":
    main()
