from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import sys
import urllib.parse

from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PROGRAM_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = PROGRAM_DIR.parent
PROMPT_DIR = PROGRAM_DIR / "prompts"
for import_dir in (PROGRAM_DIR, SCRIPT_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from openai_utils import DEFAULT_LLM_MODEL, async_llm_json_call, llm_json_call, make_async_client
from open_vocab_clip_view import DEFAULT_TEMPERATURE
from relation_judge import (
    DEFAULT_POINT_SELECTION_DIST_M,
    RELATION_JUDGE_INSTRUCTIONS,
    RELATION_JUDGE_SCHEMA,
    RELATION_LLM_CONCURRENCY,
    make_relation_judge_prompt,
)
from spatial_relation_view import RELATIONS, RGB_FILE, RelationEngine, validate_inputs
from view_pcd import DEFAULT_HOST, DEFAULT_PCD_DIR, DEFAULT_POINT_SIZE


DEFAULT_PORT = 8091
DEFAULT_THRESHOLD = 0.95
OPS = ("clip_grounding", *RELATIONS)


TASK_PROGRAM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "var": {"type": "string"},
                    "op": {"type": "string", "enum": list(OPS)},
                    "source": {"type": ["string", "null"]},
                    "class": {"type": "string"},
                    "wholeobj": {"type": "boolean"},
                    "threshold": {"type": "number"},
                },
                "required": ["var", "op", "source", "class", "wholeobj", "threshold"],
            },
        },
        "return": {"type": "string"},
    },
    "required": ["steps", "return"],
}


TASK_PROGRAM_INSTRUCTIONS = (
    (PROMPT_DIR / "task_program_instructions.txt")
    .read_text()
    .replace("{{DEFAULT_THRESHOLD}}", f"{DEFAULT_THRESHOLD:g}")
    .strip()
)
TASK_PROGRAM_USER_PROMPT_TEMPLATE = (PROMPT_DIR / "task_program_user_prompt.txt").read_text()


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Task Grounding Viewer</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #111; color: #eee; font-family: ui-sans-serif, system-ui, sans-serif; }
    #bar { position: fixed; left: 14px; top: 14px; right: 380px; z-index: 10; display: flex; gap: 10px; align-items: center; padding: 10px 12px; background: rgba(20,20,20,.9); border: 1px solid #444; border-radius: 10px; }
    textarea, input, button { color: #eee; background: #222; border: 1px solid #555; border-radius: 6px; padding: 6px 8px; }
    textarea { flex: 1; min-height: 38px; resize: vertical; }
    button { cursor: pointer; }
    button:disabled { opacity: .45; cursor: wait; }
    #side { position: fixed; right: 14px; top: 14px; bottom: 14px; width: 340px; z-index: 10; display: flex; flex-direction: column; gap: 10px; }
    .panel { min-height: 0; flex: 1; padding: 10px; background: rgba(20,20,20,.9); border: 1px solid #444; border-radius: 10px; overflow: auto; }
    .panel h3 { margin: 0 0 8px; font-size: 14px; }
    pre { margin: 0; white-space: pre-wrap; font-size: 12px; line-height: 1.35; }
    #status { position: fixed; left: 14px; bottom: 14px; right: 380px; z-index: 10; padding: 8px 10px; background: rgba(20,20,20,.84); border: 1px solid #444; border-radius: 8px; font-size: 13px; }
    #legend { position: fixed; left: 14px; bottom: 58px; z-index: 10; display: flex; gap: 14px; padding: 8px 10px; background: rgba(20,20,20,.84); border: 1px solid #444; border-radius: 8px; font-size: 13px; }
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
    <textarea id="task">Throw the trash in the trash can near the door</textarea>
    <button id="submit">Submit</button>
    <button id="reset">Reset RGB</button>
    <label>Point size <input id="pointSize" type="number" min="0.001" step="0.001" value="__POINT_SIZE__" /></label>
  </div>
  <div id="side">
    <div class="panel"><h3>Generated Program</h3><pre id="program">(none)</pre></div>
    <div class="panel"><h3>Logs</h3><pre id="logs">(none)</pre></div>
  </div>
  <div id="legend">
    <div><span class="swatch green"></span>final referential object</div>
    <div><span class="swatch blue"></span>returned floor points</div>
  </div>
  <div id="status">Loading RGB pointcloud...</div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

    const statusEl = document.getElementById('status');
    const taskEl = document.getElementById('task');
    const submitEl = document.getElementById('submit');
    const resetEl = document.getElementById('reset');
    const pointSizeEl = document.getElementById('pointSize');
    const programEl = document.getElementById('program');
    const logsEl = document.getElementById('logs');

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
      taskEl.disabled = isBusy;
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

    function decodeMask(maskB64) {
      const binary = atob(maskB64);
      const mask = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) mask[i] = binary.charCodeAt(i);
      return mask;
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

    function resetRgb() {
      if (!pcd || !originalColors) return;
      pcd.geometry.attributes.color.array.set(originalColors);
      pcd.geometry.attributes.color.needsUpdate = true;
      setStatus(`RGB view: ${pcd.geometry.attributes.position.count.toLocaleString()} points`);
    }

    async function submitTask() {
      const task = taskEl.value.trim();
      if (!task) {
        setStatus('Enter a task.');
        return;
      }
      setBusy(true);
      setStatus('Synthesizing program with LLM...');
      logsEl.textContent = 'running...';
      try {
        const programResponse = await fetch('/program', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({task}),
        });
        if (!programResponse.ok) throw new Error(await programResponse.text());
        const programPayload = await programResponse.json();
        programEl.textContent = programPayload.program_text;
        logsEl.textContent = 'program synthesized; executing grounding...';
        setStatus('Program ready. Executing grounding...');

        const executeResponse = await fetch('/execute', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({program: programPayload.program, task}),
        });
        if (!executeResponse.ok) throw new Error(await executeResponse.text());
        const payload = await executeResponse.json();
        logsEl.textContent = payload.meta.logs.join('\n');
        applyMask(decodeMask(payload.mask_b64));
        setStatus(
          `Task grounded: ${payload.meta.final_point_count.toLocaleString()} floor points, ` +
          `${payload.meta.green_instance_ids.length} referential object instance(s), ` +
          `${payload.meta.final_branch_count} final branch(es).`
        );
      } catch (err) {
        console.error(err);
        setStatus(`Task failed: ${err.message || err}`);
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

    submitEl.addEventListener('click', submitTask);
    resetEl.addEventListener('click', resetRgb);
    taskEl.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) submitTask();
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


@dataclass
class Branch:
    instance_id: int
    object_class: str
    point_indices: np.ndarray
    referential_instance_id: int
    trace: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an LLM task-to-floor-points grounding viewer.")
    parser.add_argument("pcd_dir", nargs="?", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    return parser.parse_args()


def synthesize_program(task: str, model: str) -> dict[str, object]:
    return llm_json_call(
        TASK_PROGRAM_USER_PROMPT_TEMPLATE.replace("{{task}}", task),
        schema=TASK_PROGRAM_SCHEMA,
        schema_name="mobile_robot_task_program",
        model=model,
        instructions=TASK_PROGRAM_INSTRUCTIONS,
        max_output_tokens=4096,
    )


def format_program_text(program: dict[str, object]) -> str:
    lines = []
    for step in program["steps"]:
        var = str(step["var"])
        op = str(step["op"])
        obj_class = str(step["class"])
        threshold = float(step["threshold"])
        threshold_text = "" if abs(threshold - DEFAULT_THRESHOLD) < 1e-8 else f", threshold={threshold:g}"
        if op == "clip_grounding":
            lines.append(f'{var} = clip_grounding("{obj_class}"{threshold_text})')
        else:
            source = str(step["source"])
            wholeobj = bool(step["wholeobj"])
            lines.append(
                f'{var} = {op}({source}, "{obj_class}", wholeobj={wholeobj}{threshold_text})'
            )
    lines.append(f'return {program["return"]}')
    return "\n".join(lines)


class TaskExecutor:
    def __init__(self, engine: RelationEngine, model: str) -> None:
        self.engine = engine
        self.model = model
        self.ground_cache: dict[tuple[str, float], dict[str, object]] = {}

    def ground_instances(self, obj_class: str, threshold: float) -> dict[str, object]:
        key = (obj_class, float(threshold))
        if key in self.ground_cache:
            return self.ground_cache[key]

        _, instance_scores, _, _, top_labels = self.engine.scorer.score(obj_class)
        import asyncio

        vlm_results = asyncio.run(self.engine.rejector.reject(obj_class, instance_scores, threshold))
        rejected = set(int(instance_id) for instance_id in vlm_results["rejected_instance_ids"])
        candidate_ids = [
            int(instance_id)
            for instance_id, score in instance_scores.items()
            if np.isfinite(score) and float(score) >= threshold and int(instance_id) not in rejected
        ]
        candidate_ids.sort(key=lambda instance_id: (-float(instance_scores[instance_id]), instance_id))
        result = {
            "obj_class": obj_class,
            "threshold": float(threshold),
            "instance_scores": instance_scores,
            "candidate_ids": candidate_ids,
            "top_labels": top_labels,
            "vlm": {
                "checked_count": int(vlm_results["checked_count"]),
                "rejected_count": int(vlm_results["rejected_count"]),
                "rejected_instance_ids": vlm_results["rejected_instance_ids"],
            },
        }
        self.ground_cache[key] = result
        return result

    def relation_measurements(
        self,
        source: Branch,
        target_id: int,
        target_points: np.ndarray,
        focus_points: np.ndarray,
        focus_centroid: np.ndarray,
        front: np.ndarray,
        side: np.ndarray,
        top: np.ndarray,
        nearest_distances: np.ndarray,
    ) -> dict[str, float]:
        from scipy.spatial import cKDTree

        target_centroid = target_points.mean(axis=0)
        delta = target_centroid - focus_centroid
        horizontal_surface, _ = cKDTree(focus_points[:, :2]).query(target_points[:, :2], k=1)
        return {
            "source_instance_id": int(source.instance_id),
            "target_instance_id": int(target_id),
            "horizontal_surface_distance_m": float(np.min(horizontal_surface)),
            "front_m": float(np.dot(delta, front)),
            "lateral_m": float(np.dot(delta, side)),
            "vertical_m": float(np.dot(delta, top)),
        }

    async def judge_whole_object_relations(
        self,
        cases: list[dict[str, object]],
    ) -> list[dict[str, object]]:
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
                    model=self.model,
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

    def relation_branches(
        self,
        source_branches: list[Branch],
        relation: str,
        obj_class: str,
        wholeobj: bool,
        threshold: float,
        task: str,
    ) -> tuple[list[Branch], list[str]]:
        if relation not in RELATIONS:
            raise ValueError(f"Unknown relation op: {relation}")

        from scipy.spatial import cKDTree

        grounding = self.ground_instances(obj_class, threshold)
        target_ids = grounding["candidate_ids"]
        logs = [
            f'{relation}(..., "{obj_class}", wholeobj={wholeobj}) targets={target_ids} '
            f'vlm={grounding["vlm"]}'
        ]
        outputs: list[Branch] = []
        llm_cases: list[dict[str, object]] = []
        workers = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))

        for source in source_branches:
            focus_indices = source.point_indices
            focus_points = self.engine.points[focus_indices]
            focus_centroid = focus_points.mean(axis=0)
            front, side, top = self.engine.orientation_for_indices(focus_indices)
            focus_tree = cKDTree(focus_points)

            for target_id in target_ids:
                if int(target_id) == int(source.instance_id):
                    continue
                target_indices = self.engine.indices_by_instance[int(target_id)]
                target_points = self.engine.points[target_indices]
                score = float(grounding["instance_scores"][int(target_id)])
                if wholeobj or not self.engine.is_floor_query(obj_class):
                    nearest_distances, _ = focus_tree.query(target_points, k=1, workers=workers)
                if wholeobj:
                    measurements = self.relation_measurements(
                        source,
                        int(target_id),
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
                            "source": source,
                            "target_id": int(target_id),
                            "target_indices": target_indices,
                            "score": score,
                            "measurements": measurements,
                            "prompt": make_relation_judge_prompt(
                                task=task,
                                relation=relation,
                                source_class=source.object_class,
                                target_class=obj_class,
                                target_score=score,
                                measurements=measurements,
                            ),
                        }
                    )
                    continue

                if self.engine.is_floor_query(obj_class):
                    selected, stats = self.engine.floor_point_band_selection(
                        relation,
                        target_indices,
                        focus_points,
                        focus_centroid,
                        front,
                        side,
                        top,
                        workers,
                    )
                    pass_count = int(stats["directional_unoccupied_point_count"])
                    floor_filtered_count = int(len(target_indices) - stats["unoccupied_floor_point_count"])
                    mode = f"floor-band width={stats['floor_band_max'] - stats['floor_band_min']:.3f}"
                else:
                    point_mask, stats = self.engine.point_relation_mask(
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
                    selected = target_indices[point_mask]
                    floor_filtered_count = 0
                    mode = f"point-threshold dist={DEFAULT_POINT_SELECTION_DIST_M:g}"
                if not len(selected):
                    logs.append(
                        f"  source {source.instance_id} -> target {target_id}: "
                        f"mode={mode}, "
                        f"pass_pts={pass_count}, selected_pts=0, floor_unoccupied_filtered={floor_filtered_count}, "
                        f"score={score:.3f}, nearest={stats['nearest_distance']:.3f}, "
                        f"horizontal={stats['horizontal_distance']:.3f}, "
                        f"front={stats['front']:.3f}, side={stats['side']:.3f}, top={stats['top']:.3f}"
                    )
                    continue
                outputs.append(
                    Branch(
                        instance_id=int(target_id),
                        object_class=obj_class,
                        point_indices=selected,
                        referential_instance_id=int(source.instance_id),
                        trace=(
                            *source.trace,
                            f"{relation}->{target_id} score={score:.3f} pass_pts={pass_count} selected_pts={len(selected)} "
                            f"floor_unoccupied_filtered={floor_filtered_count} stats={stats}",
                        ),
                    )
                )
                logs.append(
                    f"  source {source.instance_id} -> target {target_id}: "
                    f"mode={mode}, "
                    f"pass_pts={pass_count}, selected_pts={len(selected)}, "
                    f"floor_unoccupied_filtered={floor_filtered_count}, "
                    f"score={score:.3f}, nearest={stats['nearest_distance']:.3f}, "
                    f"horizontal={stats['horizontal_distance']:.3f}, "
                    f"front={stats['front']:.3f}, side={stats['side']:.3f}, top={stats['top']:.3f}"
                )

        if llm_cases:
            judged_cases = asyncio.run(self.judge_whole_object_relations(llm_cases))
            for case in judged_cases:
                source = case["source"]
                target_id = int(case["target_id"])
                measurements = case["measurements"]
                score = float(case["score"])
                keep = bool(case["keep"])
                reason = str(case["reason"])
                logs.append(
                    f"  source {source.instance_id} -> target {target_id}: "
                    f"mode=llm-object keep={keep}, score={score:.3f}, "
                    f"horizontal_surface={measurements['horizontal_surface_distance_m']:.3f}, "
                    f"front={measurements['front_m']:.3f}, lateral={measurements['lateral_m']:.3f}, "
                    f"vertical={measurements['vertical_m']:.3f}, reason={reason}"
                )
                if not keep:
                    continue
                target_indices = case["target_indices"]
                outputs.append(
                    Branch(
                        instance_id=target_id,
                        object_class=obj_class,
                        point_indices=target_indices,
                        referential_instance_id=int(source.instance_id),
                        trace=(
                            *source.trace,
                            f"{relation}->{target_id} score={score:.3f} selected_pts={len(target_indices)} "
                            f"measurements={measurements} llm_reason={reason}",
                        ),
                    )
                )
        logs.append(f"{relation} produced {len(outputs)} branch(es)")
        return outputs, logs

    def execute_program(self, program: dict[str, object], task: str = "") -> tuple[np.ndarray, dict[str, object]]:
        steps = program.get("steps")
        return_var = program.get("return")
        if not isinstance(steps, list) or not isinstance(return_var, str):
            raise ValueError("Program must contain steps[] and return")

        env: dict[str, list[Branch]] = {}
        step_by_var: dict[str, dict[str, object]] = {}
        logs: list[str] = []
        self.ground_cache.clear()

        for step_index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                raise ValueError(f"step {step_index} is not an object")
            var = str(step["var"])
            op = str(step["op"])
            obj_class = str(step["class"]).strip()
            threshold = float(step["threshold"])
            step_by_var[var] = step
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError(f"threshold for step {var} must be in [0, 1], got {threshold}")

            if op == "clip_grounding":
                grounding = self.ground_instances(obj_class, threshold)
                branches = [
                    Branch(
                        instance_id=int(instance_id),
                        object_class=obj_class,
                        point_indices=self.engine.indices_by_instance[int(instance_id)],
                        referential_instance_id=int(instance_id),
                        trace=(f'clip_grounding("{obj_class}") -> {instance_id}',),
                    )
                    for instance_id in grounding["candidate_ids"]
                ]
                env[var] = branches
                logs.append(
                    f'{var} = clip_grounding("{obj_class}") -> {grounding["candidate_ids"]} '
                    f'top={grounding["top_labels"]} vlm={grounding["vlm"]}'
                )
                continue

            source_name = step["source"]
            if not isinstance(source_name, str) or source_name not in env:
                raise ValueError(f"step {var} references unknown source: {source_name}")
            wholeobj = bool(step["wholeobj"])
            branches, step_logs = self.relation_branches(env[source_name], op, obj_class, wholeobj, threshold, task)
            env[var] = branches
            logs.extend([f"{var}: {line}" for line in step_logs])

        if return_var not in env:
            raise ValueError(f"return references unknown var: {return_var}")
        final_step = step_by_var[return_var]
        if (
            str(final_step["op"]) == "clip_grounding"
            or str(final_step["class"]).strip().lower() != "floor"
            or bool(final_step["wholeobj"])
        ):
            raise ValueError('Final returned step must target class="floor" with wholeobj=false')
        final_branches = env[return_var]

        mask = np.zeros(len(self.engine.points), dtype=np.uint8)
        green_ids = sorted({int(branch.referential_instance_id) for branch in final_branches})
        for instance_id in green_ids:
            if instance_id in self.engine.indices_by_instance:
                mask[self.engine.indices_by_instance[instance_id]] = 1
        for branch in final_branches:
            mask[branch.point_indices] = 2

        final_point_count = int(np.count_nonzero(mask == 2))
        logs.append(f"return {return_var}: {len(final_branches)} branch(es), {final_point_count} final floor point(s)")
        for idx, branch in enumerate(final_branches[:20], start=1):
            logs.append(f"  final branch {idx}: " + " | ".join(branch.trace))
        meta = {
            "program": program,
            "program_text": format_program_text(program),
            "logs": logs,
            "final_branch_count": len(final_branches),
            "final_point_count": final_point_count,
            "green_instance_ids": green_ids,
            "final_traces": [list(branch.trace) for branch in final_branches[:20]],
        }
        return mask, meta

    def execute_task(self, task: str) -> tuple[np.ndarray, dict[str, object]]:
        program = synthesize_program(task, self.model)
        return self.execute_program(program, task)

    def synthesize_task(self, task: str) -> dict[str, object]:
        program = synthesize_program(task, self.model)
        return {"program": program, "program_text": format_program_text(program)}


def make_handler(pcd_dir: Path, point_size: float, vertex_count: int, executor: TaskExecutor):
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
                self.send_json({"points": vertex_count, "point_size": point_size})
                return

            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path not in {"/program", "/execute"}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if parsed.path == "/program":
                    task = str(request["task"]).strip()
                    if not task:
                        raise ValueError("empty task")
                    self.send_json(executor.synthesize_task(task))
                    return
                program = request["program"]
                if not isinstance(program, dict):
                    raise ValueError("program must be an object")
                mask, meta = executor.execute_program(program, str(request.get("task", "")).strip())
            except Exception as exc:
                self.send_plain_error(500, f"Request failed: {exc}")
                return

            self.send_json(
                {
                    "mask_b64": base64.b64encode(mask.tobytes()).decode("ascii"),
                    "meta": meta,
                }
            )

    return Handler


def main() -> None:
    args = parse_args()
    pcd_dir = args.pcd_dir.expanduser().resolve()
    if not pcd_dir.exists():
        raise FileNotFoundError(f"Missing PCD dir: {pcd_dir}")
    vertex_count, engine = validate_inputs(pcd_dir, float(args.temperature))
    executor = TaskExecutor(engine, str(args.model))
    server = ThreadingHTTPServer(
        (args.host, int(args.port)),
        make_handler(pcd_dir, float(args.point_size), vertex_count, executor),
    )
    print(f"Serving {pcd_dir}")
    print(f"Open: http://127.0.0.1:{args.port}/")
    print("Stop with Ctrl-C")
    server.serve_forever()


if __name__ == "__main__":
    main()
