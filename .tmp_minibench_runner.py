#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from scripts import scannet_vipe_bench_evaluator as bench
from vipe.utils.config import load_yaml_config
from vipe.utils.determinism import seed_everything


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _set_nested(cfg: Any, dotted: str, value: Any) -> None:
    cur = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        cur = cur[key]
    cur[keys[-1]] = value


def _apply_overrides(cfg: Any, overrides: dict[str, Any]) -> Any:
    for key, value in overrides.items():
        _set_nested(cfg, key, value)
    return cfg


def _scene_result_path(run_dir: Path, scene: str) -> Path:
    return run_dir / "scenes" / f"{scene}.json"


def _scene_state(run_dir: Path, scene: str, status: str, **extra) -> None:
    payload = {"scene": scene, "status": status, "updated_at": time.time(), **extra}
    _atomic_write_json(_scene_result_path(run_dir, scene), payload)


def _load_pose_metric(work_dir: Path, scene: str) -> dict | None:
    path = work_dir / "metric_results" / "incremental_pose" / "scannet" / f"{scene}.json"
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return None
    return payload.get("metrics")


def _load_timing(work_dir: Path, scene: str, build_timing: dict, metric_timing: dict) -> dict:
    return {
        "build": build_timing.get(scene),
        "metric_eval": metric_timing.get(scene),
    }


def _run_vipe_pose_only(scene_dir: Path, pipeline_cfg, output_dir: Path) -> str:
    from vipe.pipeline import VipePipeline
    from vipe.streams.base import FrameDir
    from vipe.utils.logging import configure_logging

    seed_everything(pipeline_cfg.seed, temporary_determinism=pipeline_cfg.temporary_determinism)
    logger = configure_logging()
    pipeline = VipePipeline(
        slam=pipeline_cfg.pipeline.slam,
        output=pipeline_cfg.pipeline.output,
        output_dir=output_dir,
    )
    stream = FrameDir(path=scene_dir)
    logger.info(f"Running pose-only ViPE on {stream.name()}")
    frame_stream, intrinsics = pipeline._initialize(stream)
    old_trace_path = os.environ.get("VIPE_SLAM_DEBUG_TRACE_PATH")
    os.environ["VIPE_SLAM_DEBUG_TRACE_PATH"] = str(output_dir / "debug_trace.jsonl")
    try:
        slam_output = pipeline._run_slam(frame_stream, intrinsics)
    finally:
        if old_trace_path is None:
            os.environ.pop("VIPE_SLAM_DEBUG_TRACE_PATH", None)
        else:
            os.environ["VIPE_SLAM_DEBUG_TRACE_PATH"] = old_trace_path

    pose_path = output_dir / "pose" / f"{stream.name()}.npz"
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    poses = slam_output.trajectory.matrix().detach().cpu().numpy().astype(np.float32)
    inds = np.arange(len(poses), dtype=np.int64)
    np.savez(pose_path, data=poses, inds=inds)
    return stream.name()


def _write_pose_only_manifest(
    args: argparse.Namespace,
    evaluator,
    scene: str,
    full_scene_data,
    frame_indices: list[int],
    kept_scene_indices: list[int],
    vipe_output_dir: Path,
    artifact_name: str,
) -> None:
    pose_path = vipe_output_dir / "pose" / f"{artifact_name}.npz"
    if not pose_path.exists():
        raise FileNotFoundError(f"Missing ViPE pose artifact: {pose_path}")

    manifest = {
        "format": "vipe_pose_only_v1",
        "scene": scene,
        "artifact_name": artifact_name,
        "vipe_output_dir": str(vipe_output_dir.resolve()),
        "pose_path": str(pose_path.resolve()),
        "frame_indices": [int(idx) for idx in frame_indices],
    }

    exported_scene_data = bench._subset_scene_data(full_scene_data, kept_scene_indices)
    export_dir = Path(evaluator._export_dir("scannet", scene))
    exports_dir = export_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    (exports_dir / "vipe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    evaluator._save_gt_meta(str(export_dir), exported_scene_data)


def _prepare_pose_only_exports(
    args: argparse.Namespace,
    evaluator,
    pipeline_cfg,
) -> tuple[dict, dict]:
    build_timing = {}
    metric_timing = {}
    for scene in bench._scenes_for_worker(args, evaluator):
        start_scene = time.perf_counter()
        scene_dir = args.input_root / scene
        vipe_output_dir = args.work_dir / "vipe_outputs" / scene
        full_scene_data, frame_indices, kept_scene_indices = bench._benchmark_frame_request(evaluator, scene)
        start_build = time.perf_counter()
        artifact_name = _run_vipe_pose_only(scene_dir, pipeline_cfg, vipe_output_dir)
        build_seconds = time.perf_counter() - start_build
        _write_pose_only_manifest(
            args,
            evaluator,
            scene,
            full_scene_data,
            frame_indices,
            kept_scene_indices,
            vipe_output_dir,
            artifact_name,
        )
        frames = len(frame_indices)
        metric_seconds = max(0.0, time.perf_counter() - start_scene - build_seconds)
        build_timing[scene] = bench._timing_entry(frames, build_seconds)
        metric_timing[scene] = bench._timing_entry(frames, metric_seconds)
        bench._write_incremental_pose_metric(args, evaluator, scene)
    return build_timing, metric_timing


def _make_args(work_dir: Path, input_root: Path, raw_root: Path, scenes: list[str], gpu_id: int = 0) -> argparse.Namespace:
    return argparse.Namespace(
        scenes=scenes,
        work_dir=work_dir,
        input_root=input_root,
        raw_root=raw_root,
        print_only=False,
        gpu_id=gpu_id,
        total_gpus=1,
    )


def _run_worker(args: argparse.Namespace) -> None:
    run_dir = args.experiment_dir / "runs" / args.run_id
    work_dir = run_dir / "work"
    overrides = json.loads(args.overrides_json)

    pipeline_cfg = _apply_overrides(load_yaml_config(bench.PIPELINE_CONFIG_PATH), overrides)
    eval_config = load_yaml_config(bench.EVAL_CONFIG_PATH)
    seed_everything(int(eval_config.seed), temporary_determinism=pipeline_cfg.temporary_determinism)

    for scene in args.scenes:
        scene_start = time.perf_counter()
        _scene_state(run_dir, scene, "running", worker_id=args.worker_id)
        scene_args = _make_args(work_dir, args.input_root, args.raw_root, [scene], gpu_id=args.worker_id)
        evaluator = bench._load_evaluator(scene_args, eval_config)
        evaluator.scenes_filter = [scene]
        try:
            build_timing, metric_timing = _prepare_pose_only_exports(scene_args, evaluator, pipeline_cfg)
            metrics = _load_pose_metric(work_dir, scene)
            if metrics is None:
                raise RuntimeError(f"missing incremental pose metric for {scene}")
            _scene_state(
                run_dir,
                scene,
                "done",
                worker_id=args.worker_id,
                seconds=time.perf_counter() - scene_start,
                metrics=metrics,
                timing=_load_timing(work_dir, scene, build_timing, metric_timing),
            )
        except Exception as exc:
            _scene_state(
                run_dir,
                scene,
                "failed",
                worker_id=args.worker_id,
                seconds=time.perf_counter() - scene_start,
                error=str(exc),
                traceback=traceback.format_exc(),
            )


def _scene_weight(input_root: Path, scene: str) -> int:
    color_dir = input_root / scene / "color"
    if not color_dir.exists():
        return 1
    return sum(1 for path in color_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def _split_scenes(input_root: Path, scenes: list[str], n: int) -> list[list[str]]:
    bins: list[list[str]] = [[] for _ in range(n)]
    loads = [0 for _ in range(n)]
    weighted = sorted(((scene, _scene_weight(input_root, scene)) for scene in scenes), key=lambda item: item[1], reverse=True)
    for scene, weight in weighted:
        worker = min(range(n), key=lambda idx: loads[idx])
        bins[worker].append(scene)
        loads[worker] += weight
    return bins


def _aggregate(run_dir: Path, scenes: list[str]) -> dict:
    scene_payloads = []
    for scene in scenes:
        payload = _read_json(_scene_result_path(run_dir, scene))
        if isinstance(payload, dict):
            scene_payloads.append(payload)

    done = [item for item in scene_payloads if item.get("status") == "done"]
    failed = [item for item in scene_payloads if item.get("status") == "failed"]
    summary = {
        "status": "done" if len(done) == len(scenes) else ("failed" if failed else "running"),
        "total": len(scenes),
        "complete": len(done),
        "failed": len(failed),
        "updated_at": time.time(),
    }
    if done:
        for metric_key in ("auc03", "auc05", "auc15", "auc30"):
            vals = [float(item["metrics"][metric_key]) for item in done if metric_key in item.get("metrics", {})]
            if vals:
                summary[f"mean_{metric_key}"] = sum(vals) / len(vals)
        build_entries = [
            item.get("timing", {}).get("build")
            for item in done
            if isinstance(item.get("timing", {}).get("build"), dict)
        ]
        frames = sum(int(item.get("frames", 0)) for item in build_entries)
        seconds = sum(float(item.get("seconds", 0.0)) for item in build_entries)
        if seconds > 0.0:
            summary["build_fps"] = frames / seconds
            summary["build_frames"] = frames
            summary["build_seconds"] = seconds
    _atomic_write_json(run_dir / "summary.json", summary)
    return summary


def _run_parent(args: argparse.Namespace) -> None:
    run_dir = args.experiment_dir / "runs" / args.run_id
    if run_dir.exists() and not args.force:
        raise SystemExit(f"run already exists: {run_dir} (use --force to overwrite)")

    run_dir.mkdir(parents=True, exist_ok=True)
    scenes = list(args.scenes)
    if args.force:
        for path in (run_dir / "scenes").glob("*.json"):
            path.unlink()
        for path in (run_dir / "summary.json",):
            if path.exists():
                path.unlink()
    overrides = json.loads(args.overrides_json)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    meta = {
        "run_id": args.run_id,
        "description": args.description,
        "scenes": scenes,
        "overrides": overrides,
        "gpus": gpus,
        "status": "running",
        "started_at": time.time(),
        "updated_at": time.time(),
    }
    _atomic_write_json(run_dir / "meta.json", meta)
    for scene in scenes:
        _scene_state(run_dir, scene, "pending")

    workers = []
    for worker_id, worker_scenes in enumerate(_split_scenes(args.input_root, scenes, len(gpus))):
        if not worker_scenes:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus[worker_id]
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--experiment-dir",
            str(args.experiment_dir),
            "--run-id",
            args.run_id,
            "--input-root",
            str(args.input_root),
            "--raw-root",
            str(args.raw_root),
            "--worker-id",
            str(worker_id),
            "--overrides-json",
            args.overrides_json,
            "--scenes",
            *worker_scenes,
        ]
        workers.append(subprocess.Popen(cmd, env=env))

    while True:
        summary = _aggregate(run_dir, scenes)
        if all(process.poll() is not None for process in workers):
            break
        time.sleep(15)

    returncodes = [process.wait() for process in workers]
    summary = _aggregate(run_dir, scenes)
    meta["status"] = "done" if all(code == 0 for code in returncodes) and summary.get("failed", 0) == 0 else "failed"
    meta["finished_at"] = time.time()
    meta["updated_at"] = time.time()
    _atomic_write_json(run_dir / "meta.json", meta)
    summary["status"] = meta["status"]
    _atomic_write_json(run_dir / "summary.json", summary)
    if meta["status"] != "done":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--overrides-json", default="{}")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    args = parser.parse_args()

    if args.worker:
        _run_worker(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
