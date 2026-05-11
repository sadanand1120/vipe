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
from relation_judge import (
    DEFAULT_POINT_SELECTION_DIST_M,
    RELATION_JUDGE_INSTRUCTIONS,
    RELATION_JUDGE_SCHEMA,
    RELATION_LLM_CONCURRENCY,
    make_relation_judge_prompt,
)
from openai_utils import async_llm_json_call, make_async_client
from orient_annotate import INSTANCE_VERTEX_DTYPE, read_binary_ply_header
from view_pcd import DEFAULT_HOST, DEFAULT_PCD_DIR, DEFAULT_POINT_SIZE, read_ply_header


DEFAULT_PORT = 8090
RGB_FILE = "rgb.ply"
RELATIONS = ("close_to", "in_front", "on_the_side", "behind", "on_top", "below")
FLOOR_CLASS = "floor"
FLOOR_UNOCCUPIED_HALF_WIDTH_M = 0.1
FLOOR_UNOCCUPIED_HEIGHT_M = 2.0
FLOOR_UNOCCUPIED_MIN_Z_M = 0.05
FLOOR_UNOCCUPIED_GRID_M = 0.05
FLOOR_SELECTION_BAND_M = 0.6
ORTHOGONAL_SURFACE_LIMIT_M = 0.3
HEATMAP_DISTANCE_TAU_M = 0.25


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
    .red { background: #ff2222; }
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
    <label>CLIP thresh <input id="threshold" type="number" min="0" max="1" step="0.01" value="0.95" /></label>
    <label><input id="wholeObj" type="checkbox" checked /> wholeobj</label>
    <label><input id="showHeatmap" type="checkbox" /> show as heatmap</label>
    <label><input id="showFloorUnoccupied" type="checkbox" /> show floor unoccupied</label>
    <button id="submit">Submit</button>
    <button id="reset">Reset RGB</button>
    <label>Point size <input id="pointSize" type="number" min="0.001" step="0.001" value="__POINT_SIZE__" /></label>
  </div>
  <div id="status">Loading RGB pointcloud...</div>
  <div id="legend">
    <div><span class="swatch green"></span>focus instance</div>
    <div><span class="swatch blue"></span>matched object instances</div>
    <div><span class="swatch red"></span>occupied floor</div>
  </div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

    const statusEl = document.getElementById('status');
    const instanceIdEl = document.getElementById('instanceId');
    const relationEl = document.getElementById('relation');
    const objClassEl = document.getElementById('objClass');
    const thresholdEl = document.getElementById('threshold');
    const wholeObjEl = document.getElementById('wholeObj');
    const showHeatmapEl = document.getElementById('showHeatmap');
    const showFloorUnoccupiedEl = document.getElementById('showFloorUnoccupied');
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
        } else if (mask[i] === 3) {
          colors[j] = 1.0;
          colors[j + 1] = 0.05;
          colors[j + 2] = 0.05;
        }
      }
      colorAttr.needsUpdate = true;
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

    function applyHeatmap(scores) {
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
      showFloorUnoccupiedEl.checked = false;
      showHeatmapEl.checked = false;
      pcd.geometry.attributes.color.array.set(originalColors);
      pcd.geometry.attributes.color.needsUpdate = true;
      setStatus(`RGB view: ${pcd.geometry.attributes.position.count.toLocaleString()} points`);
    }

    async function showFloorUnoccupied() {
      if (!showFloorUnoccupiedEl.checked) {
        resetRgb();
        return;
      }
      setBusy(true);
      showHeatmapEl.checked = false;
      setStatus('Rendering floor occupancy mask...');
      try {
        const response = await fetch('/floor_unoccupied_mask');
        if (!response.ok) throw new Error(await response.text());
        const meta = JSON.parse(decodeURIComponent(response.headers.get('X-Floor-Unoccupied-Result')));
        const mask = new Uint8Array(await response.arrayBuffer());
        applyMask(mask);
        setStatus(
          `Floor occupancy: ${meta.unoccupied_floor_point_count.toLocaleString()} unoccupied blue, ` +
          `${meta.occupied_floor_point_count.toLocaleString()} occupied red, ` +
          `${meta.floor_point_count.toLocaleString()} total floor points.`
        );
      } catch (err) {
        console.error(err);
        setStatus(`Floor occupancy failed: ${err.message || err}`);
        showFloorUnoccupiedEl.checked = false;
      } finally {
        setBusy(false);
      }
    }

    async function submitQuery() {
      const instanceId = Number(instanceIdEl.value);
      const relation = relationEl.value;
      const objClass = objClassEl.value.trim();
      const threshold = Number(thresholdEl.value);
      if (!Number.isInteger(instanceId) || instanceId < 0) {
        setStatus('Instance id must be a non-negative integer.');
        return;
      }
      if (!objClass) {
        setStatus('Enter an object class.');
        return;
      }
      setBusy(true);
      showFloorUnoccupiedEl.checked = false;
      const wholeObj = wholeObjEl.checked;
      setStatus(`${relation}(instance ${instanceId}, "${objClass}", wholeobj=${wholeObj})...`);
      try {
        const response = await fetch('/query', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            instance_id: instanceId,
            relation,
            obj_class: objClass,
            threshold,
            whole_obj: wholeObj,
            heatmap: showHeatmapEl.checked,
          }),
        });
        if (!response.ok) throw new Error(await response.text());
        const meta = JSON.parse(decodeURIComponent(response.headers.get('X-Relation-Result')));
        const payloadMode = response.headers.get('X-Relation-Payload') || 'mask';
        const buffer = await response.arrayBuffer();
        if (payloadMode === 'heatmap') {
          applyHeatmap(new Float32Array(buffer));
        } else {
          applyMask(new Uint8Array(buffer));
        }
        const ids = meta.matched_instance_ids.length ? meta.matched_instance_ids.join(', ') : 'none';
        const rejected = meta.llm.rejected.map((x) =>
          `${x.instance_id}@${Number(x.measurements.horizontal_surface_distance_m).toFixed(2)}m: ${x.reason}`
        ).join(' | ');
        const kept = meta.llm.kept.map((x) =>
          `${x.instance_id}@${Number(x.measurements.horizontal_surface_distance_m).toFixed(2)}m`
        ).join(', ');
        setStatus(
          `${relation}(instance ${instanceId}, "${objClass}", wholeobj=${meta.whole_obj}): ` +
          `${meta.matched_instance_ids.length} match(es) [${ids}], ${meta.selected_target_point_count.toLocaleString()} blue points. ` +
          `Floor-unoccupied filtered ${meta.floor_unoccupied_filtered_point_count.toLocaleString()} point(s). ` +
          `${payloadMode === 'heatmap' ? `Heatmap tau=${Number(meta.heatmap.tau_m).toFixed(2)}m. ` : ''}` +
          `LLM kept ${meta.llm.kept.length} [${kept || 'none'}], rejected ${meta.llm.rejected.length}${rejected ? ` [${rejected}]` : ''}. ` +
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
    showFloorUnoccupiedEl.addEventListener('change', showFloorUnoccupied);
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
        self.floor_point_mask = self._load_floor_point_mask()
        self.floor_unoccupied_mask = self._compute_floor_unoccupied_mask()

    def _load_orient_axes(self, pcd_dir: Path) -> dict[int, np.ndarray]:
        path = pcd_dir / "orient.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing required orientation file: {path}")
        data = np.load(path)
        instance_ids = data["instance_ids"].astype(np.int32, copy=False)
        axes = data["axes"].astype(np.float32, copy=False)
        return {int(instance_id): axes[pos] for pos, instance_id in enumerate(instance_ids.tolist())}

    def _load_floor_point_mask(self) -> np.ndarray:
        labels = [label.strip().lower() for label in self.scorer.label_texts]
        if FLOOR_CLASS not in labels:
            raise ValueError(f'clip.npz has no "{FLOOR_CLASS}" semantic label')
        floor_label_id = labels.index(FLOOR_CLASS)
        mask = self.scorer.point_label_ids == floor_label_id
        if len(mask) != len(self.points):
            raise ValueError(f"floor label point count mismatch: labels={len(mask):,} points={len(self.points):,}")
        if not np.any(mask):
            raise ValueError("No floor points found in clip labels")
        return mask

    def _compute_floor_unoccupied_mask(self) -> np.ndarray:
        floor_indices = np.flatnonzero(self.floor_point_mask)
        obstacle_indices = np.flatnonzero(~self.floor_point_mask)
        floor_unoccupied = np.zeros(len(self.points), dtype=bool)
        if not len(obstacle_indices):
            floor_unoccupied[floor_indices] = True
            return floor_unoccupied

        cell_size = float(FLOOR_UNOCCUPIED_GRID_M)
        xy_min = self.points[:, :2].min(axis=0) - cell_size
        xy_max = self.points[:, :2].max(axis=0) + cell_size
        grid_size_xy = np.ceil((xy_max - xy_min) / cell_size).astype(np.int64) + 1
        grid_w = int(grid_size_xy[0])
        grid_h = int(grid_size_xy[1])
        grid_size = grid_w * grid_h
        grid_xy = np.floor((self.points[:, :2] - xy_min[None, :]) / cell_size).astype(np.int64)
        cell_ids = grid_xy[:, 0] + grid_xy[:, 1] * grid_w
        floor_cell_ids = cell_ids[floor_indices]

        floor_counts = np.bincount(floor_cell_ids, minlength=grid_size)
        floor_z_sums = np.bincount(floor_cell_ids, weights=self.points[floor_indices, 2], minlength=grid_size)
        has_floor = floor_counts > 0
        floor_z = np.full(grid_size, np.nan, dtype=np.float32)
        floor_z[has_floor] = (floor_z_sums[has_floor] / floor_counts[has_floor]).astype(np.float32)

        print(
            "Computing floor unoccupied mask: "
            f"{len(floor_indices):,} floor points, {len(obstacle_indices):,} non-floor obstacle points, "
            f"{FLOOR_UNOCCUPIED_HALF_WIDTH_M * 2:g}m x "
            f"{FLOOR_UNOCCUPIED_HALF_WIDTH_M * 2:g}m x {FLOOR_UNOCCUPIED_HEIGHT_M:g}m clearance, "
            f"{cell_size:g}m grid"
        )
        occupied_cells = np.zeros(grid_size, dtype=bool)
        obstacle_grid = grid_xy[obstacle_indices]
        obstacle_z = self.points[obstacle_indices, 2]
        offset_radius = int(np.ceil(FLOOR_UNOCCUPIED_HALF_WIDTH_M / cell_size))
        for dx in range(-offset_radius, offset_radius + 1):
            for dy in range(-offset_radius, offset_radius + 1):
                candidate_x = obstacle_grid[:, 0] + dx
                candidate_y = obstacle_grid[:, 1] + dy
                valid = (candidate_x >= 0) & (candidate_x < grid_w) & (candidate_y >= 0) & (candidate_y < grid_h)
                candidate_cells = candidate_x[valid] + candidate_y[valid] * grid_w
                valid_cells = has_floor[candidate_cells]
                if not np.any(valid_cells):
                    continue
                cells = candidate_cells[valid_cells]
                dz = obstacle_z[valid][valid_cells] - floor_z[cells]
                blocks = (dz > FLOOR_UNOCCUPIED_MIN_Z_M) & (dz <= FLOOR_UNOCCUPIED_HEIGHT_M)
                occupied_cells[cells[blocks]] = True

        floor_unoccupied[floor_indices] = ~occupied_cells[floor_cell_ids]
        print(
            "Floor unoccupied mask ready: "
            f"{int(np.count_nonzero(floor_unoccupied)):,} unoccupied / {len(floor_indices):,} floor points"
        )
        return floor_unoccupied

    def is_floor_query(self, obj_class: str) -> bool:
        return obj_class.strip().lower() == FLOOR_CLASS

    def unoccupied(self, point_indices: np.ndarray) -> np.ndarray:
        return point_indices[self.floor_unoccupied_mask[point_indices]]

    def filter_floor_unoccupied(self, point_indices: np.ndarray, obj_class: str) -> tuple[np.ndarray, int]:
        if not self.is_floor_query(obj_class):
            return point_indices, 0
        filtered = self.unoccupied(point_indices)
        return filtered, int(len(point_indices) - len(filtered))

    def local_surface_components(
        self,
        target_points: np.ndarray,
        focus_points: np.ndarray,
        focus_centroid: np.ndarray,
        front_axis: np.ndarray,
        side_axis: np.ndarray,
        top_axis: np.ndarray,
    ) -> dict[str, np.ndarray]:
        focus_deltas = focus_points - focus_centroid[None, :]
        target_deltas = target_points - focus_centroid[None, :]
        focus_front = focus_deltas @ front_axis
        focus_side = focus_deltas @ side_axis
        focus_top = focus_deltas @ top_axis
        front = target_deltas @ front_axis
        side = target_deltas @ side_axis
        top = target_deltas @ top_axis

        def signed_surface(values: np.ndarray, ref_values: np.ndarray) -> np.ndarray:
            low = float(np.min(ref_values))
            high = float(np.max(ref_values))
            return np.where(values > high, values - high, np.where(values < low, values - low, 0.0))

        front_surface = signed_surface(front, focus_front)
        side_surface = signed_surface(side, focus_side)
        top_surface = signed_surface(top, focus_top)
        return {
            "front": front,
            "side": side,
            "top": top,
            "front_surface": front_surface,
            "side_surface": side_surface,
            "top_surface": top_surface,
            "front_surface_abs": np.abs(front_surface),
            "side_surface_abs": np.abs(side_surface),
            "top_surface_abs": np.abs(top_surface),
        }

    def floor_point_band_selection(
        self,
        relation: str,
        target_indices: np.ndarray,
        focus_points: np.ndarray,
        focus_centroid: np.ndarray,
        front_axis: np.ndarray,
        side_axis: np.ndarray,
        top_axis: np.ndarray,
        workers: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        from scipy.spatial import cKDTree

        unoccupied_indices = self.unoccupied(target_indices)
        if not len(unoccupied_indices):
            return unoccupied_indices, {
                "nearest_distance": float("nan"),
                "horizontal_distance": float("nan"),
                "front": float("nan"),
                "side": float("nan"),
                "top": float("nan"),
                "floor_band_min": float("nan"),
                "floor_band_max": float("nan"),
                "unoccupied_floor_point_count": 0,
                "directional_unoccupied_point_count": 0,
            }

        unoccupied_points = self.points[unoccupied_indices]
        components = self.local_surface_components(
            unoccupied_points,
            focus_points,
            focus_centroid,
            front_axis,
            side_axis,
            top_axis,
        )
        front = components["front"]
        side = components["side"]
        top = components["top"]
        front_surface = components["front_surface"]
        side_surface = components["side_surface"]
        top_surface = components["top_surface"]
        front_surface_abs = components["front_surface_abs"]
        side_surface_abs = components["side_surface_abs"]

        if relation == "close_to":
            direction_mask = np.ones(len(unoccupied_indices), dtype=bool)
        elif relation == "in_front":
            direction_mask = (front_surface > 0.0) & (side_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
        elif relation == "behind":
            direction_mask = (front_surface < 0.0) & (side_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
        elif relation == "on_the_side":
            direction_mask = (np.abs(side_surface) > 0.0) & (front_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
        elif relation == "on_top":
            direction_mask = (
                (top_surface > 0.0)
                & (front_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
                & (side_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
            )
        elif relation == "below":
            direction_mask = (
                (top_surface < 0.0)
                & (front_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
                & (side_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
            )
        else:
            raise ValueError(f"Unknown relation: {relation}")

        if not np.any(direction_mask):
            selected = slice(None)
            return unoccupied_indices[:0], {
                "nearest_distance": float("nan"),
                "horizontal_distance": float("nan"),
                "front": float(np.mean(front[selected])),
                "side": float(np.mean(side[selected])),
                "top": float(np.mean(top[selected])),
                "floor_band_min": float("nan"),
                "floor_band_max": float("nan"),
                "unoccupied_floor_point_count": int(len(unoccupied_indices)),
                "directional_unoccupied_point_count": 0,
                "orthogonal_surface_limit": float(ORTHOGONAL_SURFACE_LIMIT_M),
            }

        xy_distances, _ = cKDTree(focus_points[:, :2]).query(unoccupied_points[:, :2], k=1, workers=workers)
        directional_distances = xy_distances[direction_mask]
        band_min = float(np.min(directional_distances))
        band_max = band_min + FLOOR_SELECTION_BAND_M
        band_mask = direction_mask & (xy_distances >= band_min) & (xy_distances <= band_max)
        selected_indices = unoccupied_indices[band_mask]
        selected = band_mask if np.any(band_mask) else direction_mask
        stats = {
            "nearest_distance": band_min,
            "horizontal_distance": band_min,
            "front": float(np.mean(front[selected])),
            "side": float(np.mean(side[selected])),
            "top": float(np.mean(top[selected])),
            "floor_band_min": band_min,
            "floor_band_max": band_max,
            "unoccupied_floor_point_count": int(len(unoccupied_indices)),
            "directional_unoccupied_point_count": int(np.count_nonzero(direction_mask)),
            "front_surface": float(np.mean(front_surface[selected])),
            "side_surface": float(np.mean(side_surface[selected])),
            "top_surface": float(np.mean(top_surface[selected])),
            "orthogonal_surface_limit": float(ORTHOGONAL_SURFACE_LIMIT_M),
        }
        return selected_indices, stats

    def floor_unoccupied_visual_mask(self) -> tuple[np.ndarray, dict[str, int]]:
        mask = np.zeros(len(self.points), dtype=np.uint8)
        floor_occupied = self.floor_point_mask & ~self.floor_unoccupied_mask
        mask[self.floor_unoccupied_mask] = 2
        mask[floor_occupied] = 3
        meta = {
            "floor_point_count": int(np.count_nonzero(self.floor_point_mask)),
            "unoccupied_floor_point_count": int(np.count_nonzero(self.floor_unoccupied_mask)),
            "occupied_floor_point_count": int(np.count_nonzero(floor_occupied)),
        }
        return mask, meta

    def distance_heatmap_from_mask(self, mask: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
        from scipy.spatial import cKDTree

        selected_indices = np.flatnonzero(mask == 2)
        if not len(selected_indices):
            raise ValueError("Cannot build heatmap: query selected zero blue points")
        workers = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
        distances, _ = cKDTree(self.points[selected_indices]).query(self.points, k=1, workers=workers)
        scores = np.exp(-distances.astype(np.float32) / HEATMAP_DISTANCE_TAU_M).astype(np.float32)
        return scores, {
            "selected_point_count": int(len(selected_indices)),
            "tau_m": float(HEATMAP_DISTANCE_TAU_M),
            "distance_min_m": float(np.min(distances)),
            "distance_mean_m": float(np.mean(distances)),
            "distance_max_m": float(np.max(distances)),
        }

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

    def relation_measurements(
        self,
        source_instance_id: int,
        target_id: int,
        target_points: np.ndarray,
        focus_points: np.ndarray,
        focus_centroid: np.ndarray,
        front_axis: np.ndarray,
        side_axis: np.ndarray,
        top_axis: np.ndarray,
        nearest_distances: np.ndarray,
    ) -> dict[str, float]:
        from scipy.spatial import cKDTree

        target_centroid = target_points.mean(axis=0)
        delta = target_centroid - focus_centroid
        horizontal_surface, _ = cKDTree(focus_points[:, :2]).query(target_points[:, :2], k=1)
        return {
            "source_instance_id": int(source_instance_id),
            "target_instance_id": int(target_id),
            "horizontal_surface_distance_m": float(np.min(horizontal_surface)),
            "front_m": float(np.dot(delta, front_axis)),
            "lateral_m": float(np.dot(delta, side_axis)),
            "vertical_m": float(np.dot(delta, top_axis)),
        }

    async def judge_whole_object_relations(self, cases: list[dict[str, object]]) -> list[dict[str, object]]:
        if not cases:
            return []
        client = make_async_client()
        semaphore = asyncio.Semaphore(RELATION_LLM_CONCURRENCY)

        async def judge(case: dict[str, object]) -> dict[str, object]:
            async with semaphore:
                result = await async_llm_json_call(
                    str(case["prompt"]),
                    schema=RELATION_JUDGE_SCHEMA,
                    schema_name="spatial_relation_judge",
                    model="gpt-5.4-mini",
                    instructions=RELATION_JUDGE_INSTRUCTIONS,
                    max_output_tokens=1024,
                    client=client,
                )
                keep = result.get("keep")
                reason = result.get("reason")
                if not isinstance(keep, bool) or not isinstance(reason, str):
                    raise ValueError(f"Invalid relation judge output: {result}")
                case["keep"] = bool(keep)
                case["reason"] = reason[:220]
                return case

        try:
            return await asyncio.gather(*(judge(case) for case in cases))
        finally:
            await client.close()

    def point_relation_mask(
        self,
        relation: str,
        nearest_distances: np.ndarray,
        target_points: np.ndarray,
        focus_points: np.ndarray,
        focus_centroid: np.ndarray,
        front_axis: np.ndarray,
        side_axis: np.ndarray,
        top_axis: np.ndarray,
        dist: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        components = self.local_surface_components(
            target_points,
            focus_points,
            focus_centroid,
            front_axis,
            side_axis,
            top_axis,
        )
        front = components["front"]
        side = components["side"]
        top = components["top"]
        front_surface = components["front_surface"]
        side_surface = components["side_surface"]
        top_surface = components["top_surface"]
        front_surface_abs = components["front_surface_abs"]
        side_surface_abs = components["side_surface_abs"]

        if relation == "close_to":
            point_mask = nearest_distances <= dist
        else:
            within = nearest_distances <= dist
            if relation == "in_front":
                point_mask = (
                    within
                    & (front_surface > 0.0)
                    & (front_surface <= dist)
                    & (side_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
                )
            elif relation == "behind":
                point_mask = (
                    within
                    & (front_surface < 0.0)
                    & (front_surface >= -dist)
                    & (side_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
                )
            elif relation == "on_the_side":
                point_mask = (
                    within
                    & (np.abs(side_surface) > 0.0)
                    & (np.abs(side_surface) <= dist)
                    & (front_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
                )
            elif relation == "on_top":
                point_mask = (
                    within
                    & (top_surface > 0.0)
                    & (top_surface <= dist)
                    & (front_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
                    & (side_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
                )
            elif relation == "below":
                point_mask = (
                    within
                    & (top_surface < 0.0)
                    & (top_surface >= -dist)
                    & (front_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
                    & (side_surface_abs <= ORTHOGONAL_SURFACE_LIMIT_M)
                )
            else:
                raise ValueError(f"Unknown relation: {relation}")

        if point_mask.any():
            selected = point_mask
        else:
            selected = slice(None)
        stats = {
            "nearest_distance": float(np.min(nearest_distances)),
            "horizontal_distance": float(np.min(np.linalg.norm((target_points - focus_centroid[None, :])[:, :2], axis=1))),
            "front": float(np.mean(front[selected])),
            "side": float(np.mean(side[selected])),
            "top": float(np.mean(top[selected])),
            "front_surface": float(np.mean(front_surface[selected])),
            "side_surface": float(np.mean(side_surface[selected])),
            "top_surface": float(np.mean(top_surface[selected])),
            "orthogonal_surface_limit": float(ORTHOGONAL_SURFACE_LIMIT_M),
        }
        return point_mask, stats

    def query(
        self,
        *,
        focus_instance_id: int,
        relation: str,
        obj_class: str,
        threshold: float,
        whole_obj: bool,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if relation not in RELATIONS:
            raise ValueError(f"Unknown relation {relation}; expected one of {RELATIONS}")
        if focus_instance_id not in self.indices_by_instance:
            raise ValueError(f"Unknown focus instance id: {focus_instance_id}")
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
        llm_kept = []
        llm_rejected = []
        llm_cases = []
        selected_target_point_count = 0
        floor_unoccupied_filtered_point_count = 0
        for candidate_id in candidate_ids:
            target_indices = self.indices_by_instance[candidate_id]
            target_points = self.points[target_indices]
            score = float(instance_scores[candidate_id])
            if whole_obj or not self.is_floor_query(obj_class):
                nearest_distances, _ = focus_tree.query(target_points, k=1, workers=workers)
            if whole_obj:
                measurements = self.relation_measurements(
                    focus_instance_id,
                    candidate_id,
                    target_points,
                    focus_points,
                    focus_centroid,
                    front,
                    side,
                    top,
                    nearest_distances,
                )
                llm_cases.append(
                    {
                        "target_id": int(candidate_id),
                        "target_indices": target_indices,
                        "score": score,
                        "measurements": measurements,
                        "prompt": make_relation_judge_prompt(
                            task="(manual spatial relation query)",
                            relation=relation,
                            source_class=f"instance {focus_instance_id}",
                            target_class=obj_class,
                            target_score=score,
                            measurements=measurements,
                        ),
                    }
                )
                continue

            if self.is_floor_query(obj_class):
                selected_indices, stats = self.floor_point_band_selection(
                    relation,
                    target_indices,
                    focus_points,
                    focus_centroid,
                    front,
                    side,
                    top,
                    workers,
                )
                filtered_count = int(len(target_indices) - stats["unoccupied_floor_point_count"])
                floor_unoccupied_filtered_point_count += filtered_count
                pass_count = int(stats["directional_unoccupied_point_count"])
            else:
                point_mask, stats = self.point_relation_mask(
                    relation,
                    nearest_distances,
                    target_points,
                    focus_points,
                    focus_centroid,
                    front,
                    side,
                    top,
                    DEFAULT_POINT_SELECTION_DIST_M,
                )
                pass_count = int(np.count_nonzero(point_mask))
                selected_indices = target_indices[point_mask]
                filtered_count = 0

            if not len(selected_indices):
                continue
            mask[selected_indices] = 2
            selected_target_point_count += int(len(selected_indices))
            matched.append(
                {
                    "instance_id": int(candidate_id),
                    "score": score,
                    "pass_point_count": pass_count,
                    "selected_point_count": int(len(selected_indices)),
                    "floor_unoccupied_filtered_point_count": int(filtered_count),
                    **stats,
                }
            )

        if llm_cases:
            judged_cases = asyncio.run(self.judge_whole_object_relations(llm_cases))
            for case in judged_cases:
                target_id = int(case["target_id"])
                target_indices = case["target_indices"]
                measurements = case["measurements"]
                result = {
                    "instance_id": target_id,
                    "score": float(case["score"]),
                    "measurements": measurements,
                    "reason": str(case["reason"]),
                }
                if bool(case["keep"]):
                    mask[target_indices] = 2
                    selected_target_point_count += int(len(target_indices))
                    matched.append(
                        {
                            "selected_point_count": int(len(target_indices)),
                            "mode": "llm-object",
                            **result,
                        }
                    )
                    llm_kept.append(result)
                else:
                    llm_rejected.append(result)

        meta = {
            "focus_instance_id": int(focus_instance_id),
            "focus_point_count": int(len(focus_indices)),
            "relation": relation,
            "obj_class": obj_class,
            "point_selection_dist": float(DEFAULT_POINT_SELECTION_DIST_M),
            "floor_selection_band": float(FLOOR_SELECTION_BAND_M),
            "threshold": float(threshold),
            "whole_obj": bool(whole_obj),
            "candidate_count": len(candidate_ids),
            "vlm": {
                "checked_count": int(vlm_results["checked_count"]),
                "rejected_count": int(vlm_results["rejected_count"]),
                "rejected_instance_ids": vlm_results["rejected_instance_ids"],
            },
            "llm": {
                "checked_count": len(llm_cases),
                "kept": llm_kept,
                "rejected": llm_rejected,
            },
            "matched_instance_ids": [item["instance_id"] for item in matched],
            "selected_target_point_count": int(selected_target_point_count),
            "floor_unoccupied_filtered_point_count": int(floor_unoccupied_filtered_point_count),
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

            if parsed.path == "/floor_unoccupied_mask":
                mask, meta = engine.floor_unoccupied_visual_mask()
                payload = mask.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Floor-Unoccupied-Result", urllib.parse.quote(json.dumps(meta)))
                self.end_headers()
                self.wfile.write(payload)
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
                    threshold=float(request["threshold"]),
                    whole_obj=bool(request.get("whole_obj", True)),
                )
                heatmap = bool(request.get("heatmap", False))
                if heatmap:
                    scores, heatmap_meta = engine.distance_heatmap_from_mask(mask)
                    meta["heatmap"] = heatmap_meta
                    payload = scores.tobytes()
                    payload_mode = "heatmap"
                else:
                    payload = mask.tobytes()
                    payload_mode = "mask"
            except Exception as exc:
                self.send_plain_error(500, f"Query failed: {exc}")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Relation-Result", urllib.parse.quote(json.dumps(meta)))
            self.send_header("X-Relation-Payload", payload_mode)
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
