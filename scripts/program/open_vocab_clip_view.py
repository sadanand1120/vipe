from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import sys
import urllib.parse

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PROGRAM_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = PROGRAM_DIR.parent
PROMPT_DIR = PROGRAM_DIR / "prompts"
for import_dir in (PROGRAM_DIR, SCRIPT_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from view_pcd import DEFAULT_HOST, DEFAULT_PCD_DIR, DEFAULT_POINT_SIZE, read_ply_header
from openai_utils import DEFAULT_VLM_MODEL, async_vlm_json_call, make_async_client
from orient_annotate import INSTANCE_VERTEX_DTYPE, read_binary_ply_header


DEFAULT_PORT = 8089
RGB_FILE = "rgb.ply"
CLIP_FILE = "clip.npz"
DEFAULT_NEGATIVE_TEXT = "object"
DEFAULT_TEMPERATURE = 0.01
BEST_VIEWS_DIR = "best_views"
VLM_CONCURRENCY = 16
VLM_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "is_correct": {"type": "boolean"},
        "reason": {"type": "string", "maxLength": 120},
    },
    "required": ["is_correct", "reason"],
}
VLM_INSTRUCTIONS = (PROMPT_DIR / "open_vocab_vlm_instructions.txt").read_text().strip()
VLM_INSTANCE_PROMPT_TEMPLATE = (PROMPT_DIR / "open_vocab_vlm_instance_prompt.txt").read_text()
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
  <title>Open-Vocab CLIP PCD Viewer</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #111; color: #eee; font-family: ui-sans-serif, system-ui, sans-serif; }
    #bar { position: fixed; left: 14px; top: 14px; z-index: 10; display: flex; gap: 10px; align-items: center; padding: 10px 12px; background: rgba(20,20,20,.9); border: 1px solid #444; border-radius: 10px; }
    input, button { color: #eee; background: #222; border: 1px solid #555; border-radius: 6px; padding: 6px 8px; }
    button { cursor: pointer; }
    button:disabled { opacity: .45; cursor: wait; }
    #query { width: 260px; }
    #status { position: fixed; left: 14px; bottom: 14px; z-index: 10; max-width: min(900px, calc(100vw - 28px)); padding: 8px 10px; background: rgba(20,20,20,.84); border: 1px solid #444; border-radius: 8px; font-size: 13px; }
    #colorbar { display: none; position: fixed; right: 18px; bottom: 18px; z-index: 10; width: 320px; padding: 10px; background: rgba(20,20,20,.84); border: 1px solid #444; border-radius: 10px; }
    #gradient { height: 18px; border: 1px solid #222; border-radius: 4px; background: linear-gradient(to right, __HEATMAP_CSS__); }
    #ticks { display: flex; justify-content: space-between; margin-top: 6px; font-size: 12px; color: #ddd; }
    #legend { display: flex; gap: 14px; margin-top: 8px; font-size: 12px; color: #ddd; align-items: center; }
    .swatch { display: inline-block; width: 12px; height: 12px; margin-right: 5px; border: 1px solid #777; vertical-align: -2px; }
    .swatch.blue { background: rgb(49,54,149); }
    .swatch.white { background: #fff; }
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
    <input id="query" type="text" placeholder="chair, kitchen sink, ..." />
    <button id="submit">Submit</button>
    <button id="reset">Reset RGB</button>
    <label>Point size <input id="pointSize" type="number" min="0.001" step="0.001" value="__POINT_SIZE__" /></label>
    <label>Thresh <input id="threshold" type="number" min="0" max="1" step="0.01" value="0.95" /></label>
  </div>
  <div id="status">Loading RGB pointcloud...</div>
  <div id="colorbar">
    <div id="gradient"></div>
    <div id="ticks"><span id="tickMin"></span><span id="tickMid"></span><span id="tickMax"></span></div>
    <div id="legend">
      <span><span class="swatch blue"></span>threshold rejected</span>
      <span><span class="swatch white"></span>VLM rejected</span>
    </div>
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
    const tickMinEl = document.getElementById('tickMin');
    const tickMidEl = document.getElementById('tickMid');
    const tickMaxEl = document.getElementById('tickMax');

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
    let lastScores = null;
    let lastRejectedMask = null;
    let lastScoreMin = 0.0;
    let lastScoreMax = 1.0;

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

    function formatValue(value) {
      return Number(value).toFixed(3);
    }

    function setColorbar(minScore, maxScore) {
      const mid = (minScore + maxScore) * 0.5;
      tickMinEl.textContent = formatValue(minScore);
      tickMidEl.textContent = formatValue(mid);
      tickMaxEl.textContent = formatValue(maxScore);
      colorbarEl.style.display = 'block';
    }

    function heatmapColor(value, minScore, maxScore) {
      const stops = [
        [49, 54, 149], [69, 117, 180], [116, 173, 209], [171, 217, 233], [224, 243, 248],
        [254, 224, 144], [253, 174, 97], [244, 109, 67], [215, 48, 39], [165, 0, 38],
      ];
      const denom = Math.max(maxScore - minScore, 1e-8);
      const t = Math.max(0, Math.min(1, (value - minScore) / denom));
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

    function applyHeatmapScores(scores, rejectedMask, minScore, maxScore) {
      const colorAttr = pcd.geometry.attributes.color;
      const colors = colorAttr.array;
      if (scores.length * 3 !== colors.length) {
        throw new Error(`Score length mismatch: got ${scores.length}, expected ${colors.length / 3}`);
      }
      if (rejectedMask && rejectedMask.length !== scores.length) {
        throw new Error(`Rejected-mask length mismatch: got ${rejectedMask.length}, expected ${scores.length}`);
      }
      for (let i = 0; i < scores.length; i++) {
        const j = i * 3;
        if (rejectedMask && rejectedMask[i] > 0 && scores[i] > 0.0) {
          colors[j] = 1.0;
          colors[j + 1] = 1.0;
          colors[j + 2] = 1.0;
        } else {
          const value = Number.isFinite(scores[i]) ? scores[i] : 0.0;
          const color = heatmapColor(value, minScore, maxScore);
          colors[j] = color[0];
          colors[j + 1] = color[1];
          colors[j + 2] = color[2];
        }
      }
      colorAttr.needsUpdate = true;
    }

    function resetRgb() {
      if (!pcd || !originalColors) return;
      pcd.geometry.attributes.color.array.set(originalColors);
      pcd.geometry.attributes.color.needsUpdate = true;
      colorbarEl.style.display = 'none';
      lastScores = null;
      lastRejectedMask = null;
      setStatus(`RGB view: ${pcd.geometry.attributes.position.count.toLocaleString()} points`);
    }

    async function submitQuery() {
      const text = queryEl.value.trim();
      if (!text) {
        setStatus('Enter an object class first.');
        return;
      }
      setBusy(true);
      colorbarEl.style.display = 'none';
      const threshold = Number(thresholdEl.value || 0.0);
      setStatus(`Scoring "${text}" vs "object", thresholding instances, then VLM-checking passing instances...`);
      try {
        const response = await fetch('/query', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({text, threshold}),
        });
        if (!response.ok) throw new Error(await response.text());
        const minScore = Number(response.headers.get('X-Score-Min'));
        const maxScore = Number(response.headers.get('X-Score-Max'));
        const scoreBytes = Number(response.headers.get('X-Score-Bytes'));
        const rejectedBytes = Number(response.headers.get('X-Rejected-Bytes'));
        const topLabels = JSON.parse(decodeURIComponent(response.headers.get('X-Top-Labels')));
        const vlmResults = JSON.parse(decodeURIComponent(response.headers.get('X-VLM-Results')));
        const buffer = await response.arrayBuffer();
        lastScores = new Float32Array(buffer.slice(0, scoreBytes));
        lastRejectedMask = new Uint8Array(buffer.slice(scoreBytes, scoreBytes + rejectedBytes));
        lastScoreMin = minScore;
        lastScoreMax = maxScore;
        applyHeatmapScores(lastScores, lastRejectedMask, lastScoreMin, lastScoreMax);
        setColorbar(minScore, maxScore);
        const topText = topLabels.map((item) => `${item.label}=${Number(item.score).toFixed(3)}`).join(', ');
        const rejectedText = vlmResults.rejected_instance_ids.length > 0
          ? ` Rejected by VLM: ${vlmResults.rejected_instance_ids.join(', ')}.`
          : '';
        setStatus(
          `Heatmap for "${text}" vs "object" at threshold ${threshold.toFixed(3)}. ` +
          `VLM checked ${vlmResults.checked_count}, rejected ${vlmResults.rejected_count}.` +
          `${rejectedText} Top labels: ${topText}`
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
    thresholdEl.addEventListener('change', () => {
      if (lastScores) setStatus('Threshold changed. Press Submit to rerun instance thresholding and VLM rejection.');
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
    parser = argparse.ArgumentParser(description="Serve an RGB pointcloud viewer with open-vocab CLIP heatmaps.")
    parser.add_argument("pcd_dir", nargs="?", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    return parser.parse_args()


def load_instance_ids(pcd_dir: Path) -> np.ndarray:
    path = pcd_dir / "instance.ply"
    if not path.exists():
        raise FileNotFoundError(f"Missing required instance PLY: {path}")

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
    return vertices["instance_id"].astype(np.int32, copy=False)


def load_instance_label_texts(pcd_dir: Path, instance_ids: np.ndarray) -> dict[int, str]:
    data = np.load(pcd_dir / CLIP_FILE)
    point_label_ids = data["point_label_ids"].astype(np.int32, copy=False)
    point_instance_ids = load_instance_ids(pcd_dir)
    label_texts = [str(label) for label in data["label_texts"].tolist()]
    labels: dict[int, str] = {}
    for instance_id in instance_ids.tolist():
        mask = (point_instance_ids == int(instance_id)) & (point_label_ids >= 0)
        ids, counts = np.unique(point_label_ids[mask], return_counts=True)
        if len(ids):
            labels[int(instance_id)] = label_texts[int(ids[np.argmax(counts)])]
    return labels


class ClipScorer:
    def __init__(self, pcd_dir: Path, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)
        clip_path = pcd_dir / CLIP_FILE
        if not clip_path.exists():
            raise FileNotFoundError(f"Missing required CLIP file: {clip_path}")

        data = np.load(clip_path)
        self.point_label_ids = data["point_label_ids"].astype(np.int32, copy=False)
        self.point_instance_ids = load_instance_ids(pcd_dir)
        if len(self.point_instance_ids) != len(self.point_label_ids):
            raise ValueError(
                f"instance/clip point count mismatch: instance={len(self.point_instance_ids):,} "
                f"clip={len(self.point_label_ids):,}"
            )
        self.label_texts = [str(label) for label in data["label_texts"].tolist()]
        self.label_embeddings = data["label_embeddings"].astype(np.float32)
        self.label_embeddings /= np.maximum(np.linalg.norm(self.label_embeddings, axis=1, keepdims=True), 1e-8)
        self._model = None
        self._tokenizer = None

    def load_model(self):
        if self._model is None:
            import open_clip

            self._model = open_clip.create_model("ViT-L-14-336-quickgelu", pretrained="openai").eval()
            self._tokenizer = open_clip.get_tokenizer("ViT-L-14-336-quickgelu")
        return self._model, self._tokenizer

    def encode_text(self, text: str) -> np.ndarray:
        import torch

        model, tokenizer = self.load_model()
        tokens = tokenizer([text])
        with torch.inference_mode():
            embedding = model.encode_text(tokens).float()
            embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return embedding.cpu().numpy()[0].astype(np.float32)

    def score(
        self,
        text: str,
        negative_text: str = DEFAULT_NEGATIVE_TEXT,
    ) -> tuple[np.ndarray, dict[int, float], float, float, list[dict[str, object]]]:
        query_embedding = self.encode_text(text)
        negative_embedding = self.encode_text(negative_text)
        positive_logits = (self.label_embeddings @ query_embedding) / self.temperature
        negative_logits = (self.label_embeddings @ negative_embedding) / self.temperature
        logits = np.stack([positive_logits, negative_logits], axis=1)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
        label_scores = probs[:, 0]
        ids = self.point_label_ids
        valid = ids >= 0
        point_scores = np.full(len(ids), np.nan, dtype=np.float32)
        point_scores[valid] = label_scores[ids[valid]]

        instance_scores = np.full(len(point_scores), np.nan, dtype=np.float32)
        valid_instances = (self.point_instance_ids >= 0) & np.isfinite(point_scores)
        valid_indices = np.flatnonzero(valid_instances)
        instance_ids, inverse = np.unique(self.point_instance_ids[valid_indices], return_inverse=True)
        sums = np.bincount(inverse, weights=point_scores[valid_instances])
        counts = np.bincount(inverse)
        mean_scores = (sums / np.maximum(counts, 1)).astype(np.float32)
        instance_scores[valid_indices] = mean_scores[inverse]
        instance_score_map = {
            int(instance_id): float(mean_score)
            for instance_id, mean_score in zip(instance_ids.tolist(), mean_scores.tolist(), strict=True)
        }

        valid_scores = instance_scores[np.isfinite(instance_scores)]
        min_score = 0.0
        max_score = float(valid_scores.max())
        top_indices = np.argsort(-label_scores)[:5]
        top = [{"label": self.label_texts[int(idx)], "score": float(label_scores[int(idx)])} for idx in top_indices]
        return instance_scores.astype(np.float32, copy=False), instance_score_map, min_score, max_score, top


class VlmRejector:
    def __init__(self, pcd_dir: Path, instance_ids: np.ndarray) -> None:
        self.best_views_dir = pcd_dir / BEST_VIEWS_DIR
        self.instance_label_texts = load_instance_label_texts(pcd_dir, instance_ids)
        if not self.best_views_dir.exists():
            raise FileNotFoundError(f"Missing required best-view dir: {self.best_views_dir}")

        missing = [int(instance_id) for instance_id in instance_ids.tolist() if not (self.best_views_dir / f"{int(instance_id)}.png").exists()]
        if missing:
            raise FileNotFoundError(f"Missing best-view PNGs for instances: {missing[:20]}")

    async def _check_instance(self, client, text: str, instance_id: int, score: float, semaphore: asyncio.Semaphore) -> dict[str, object]:
        async with semaphore:
            prompt = (
                VLM_INSTANCE_PROMPT_TEMPLATE.replace("{{query_object_class}}", text)
                .replace("{{instance_id}}", str(int(instance_id)))
                .replace("{{clip_score}}", f"{score:.6f}")
            )
            result = await async_vlm_json_call(
                prompt,
                self.best_views_dir / f"{instance_id}.png",
                schema=VLM_RESPONSE_SCHEMA,
                schema_name="vlm_instance_check",
                model=DEFAULT_VLM_MODEL,
                instructions=VLM_INSTRUCTIONS,
                max_output_tokens=4096,
                client=client,
            )
            is_correct = result.get("is_correct")
            reason = result.get("reason")
            if not isinstance(is_correct, bool) or not isinstance(reason, str):
                raise ValueError(f"Invalid VLM structured output for instance {instance_id}: {result}")
            return {
                "instance_id": int(instance_id),
                "score": float(score),
                "is_correct": bool(is_correct),
                "reason": reason[:180],
            }

    async def reject(self, text: str, instance_scores: dict[int, float], threshold: float) -> dict[str, object]:
        query = text.lower()
        passed = [
            (int(instance_id), float(score))
            for instance_id, score in instance_scores.items()
            if (
                np.isfinite(score)
                and threshold <= float(score)
                and self.instance_label_texts.get(int(instance_id), "").lower() != query
            )
        ]
        passed.sort(key=lambda item: (-item[1], item[0]))
        if not passed:
            return {"checked_count": 0, "rejected_count": 0, "rejected_instance_ids": [], "results": []}

        client = make_async_client()
        semaphore = asyncio.Semaphore(VLM_CONCURRENCY)
        try:
            results = await asyncio.gather(
                *(self._check_instance(client, text, instance_id, score, semaphore) for instance_id, score in passed)
            )
        finally:
            await client.close()
        rejected = [int(item["instance_id"]) for item in results if not bool(item["is_correct"])]
        return {
            "checked_count": len(results),
            "rejected_count": len(rejected),
            "rejected_instance_ids": rejected,
            "results": results,
        }


def validate_inputs(pcd_dir: Path, temperature: float) -> tuple[int, ClipScorer, VlmRejector]:
    rgb_path = pcd_dir / RGB_FILE
    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing required RGB PLY: {rgb_path}")
    vertex_count, properties = read_ply_header(rgb_path)
    required = ["property uchar red", "property uchar green", "property uchar blue"]
    if vertex_count <= 0 or not all(prop in properties for prop in required):
        raise ValueError(f"{rgb_path} must be a colored binary PLY")

    scorer = ClipScorer(pcd_dir, temperature)
    if len(scorer.point_label_ids) != vertex_count:
        raise ValueError(
            f"clip point count mismatch: rgb={vertex_count:,} clip={len(scorer.point_label_ids):,}"
        )
    instance_ids = np.unique(scorer.point_instance_ids[scorer.point_instance_ids >= 0])
    rejector = VlmRejector(pcd_dir, instance_ids)
    return vertex_count, scorer, rejector


def make_handler(pcd_dir: Path, point_size: float, vertex_count: int, scorer: ClipScorer, rejector: VlmRejector):
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
                    .replace("__HEATMAP_CSS__", HEATMAP_CSS)
                    .encode("utf-8")
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if parsed.path == "/meta":
                self.send_json({"points": vertex_count, "point_size": point_size, "labels": scorer.label_texts})
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
                threshold = float(request["threshold"])
            except Exception as exc:
                self.send_plain_error(400, f"Invalid query request: {exc}")
                return
            if not text:
                self.send_plain_error(400, "Empty query")
                return
            if threshold < 0.0 or threshold > 1.0:
                self.send_plain_error(400, f"threshold must be in [0, 1], got {threshold}")
                return

            try:
                scores, instance_scores, min_score, max_score, top = scorer.score(text)
                vlm_results = asyncio.run(rejector.reject(text, instance_scores, threshold))
            except Exception as exc:
                self.send_plain_error(500, f"Query failed: {exc}")
                return

            display_scores = np.nan_to_num(scores, nan=0.0).astype(np.float32, copy=False)
            display_scores[display_scores < threshold] = 0.0
            rejected_ids = np.asarray(vlm_results["rejected_instance_ids"], dtype=np.int32)
            rejected_mask = np.zeros(len(display_scores), dtype=np.uint8)
            if len(rejected_ids) > 0:
                rejected_mask = np.isin(scorer.point_instance_ids, rejected_ids).astype(np.uint8, copy=False)
                rejected_mask[display_scores <= 0.0] = 0

            score_payload = display_scores.tobytes()
            rejected_payload = rejected_mask.tobytes()
            payload = score_payload + rejected_payload
            vlm_header = {
                "checked_count": int(vlm_results["checked_count"]),
                "rejected_count": int(vlm_results["rejected_count"]),
                "rejected_instance_ids": vlm_results["rejected_instance_ids"],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Score-Bytes", str(len(score_payload)))
            self.send_header("X-Rejected-Bytes", str(len(rejected_payload)))
            self.send_header("X-Score-Min", f"{min_score:.8f}")
            self.send_header("X-Score-Max", f"{max_score:.8f}")
            self.send_header("X-Top-Labels", urllib.parse.quote(json.dumps(top)))
            self.send_header("X-VLM-Results", urllib.parse.quote(json.dumps(vlm_header)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main() -> None:
    args = parse_args()
    pcd_dir = args.pcd_dir.expanduser().resolve()
    if not pcd_dir.exists():
        raise FileNotFoundError(f"Missing PCD dir: {pcd_dir}")

    vertex_count, scorer, rejector = validate_inputs(pcd_dir, float(args.temperature))
    server = ThreadingHTTPServer(
        (args.host, int(args.port)),
        make_handler(pcd_dir, float(args.point_size), vertex_count, scorer, rejector),
    )
    print(f"Serving {pcd_dir}")
    print(f"Open: http://127.0.0.1:{args.port}/")
    print("Stop with Ctrl-C")
    server.serve_forever()


if __name__ == "__main__":
    main()
