#!/usr/bin/env python3

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from vipe import get_config_path
from vipe.bench.gt_pose_checks import filter_scenes_with_bad_gt_pose_data
from vipe.bench.scannet import ScanNetEvaluator
from vipe.utils.config import load_yaml_config
from vipe.utils.data_format import scene_frame_count
from vipe.utils.determinism import seed_everything


WORKER_ENV = "_VIPE_SCANNET_BENCH_WORKER"
SCENE_QUEUE_ENV = "_VIPE_SCANNET_DYNAMIC_SCENE_QUEUE"
PIPELINE_CONFIG_PATH = get_config_path() / "default.yaml"
EVAL_CONFIG_PATH = get_config_path() / "eval_scannet_config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ViPE ScanNet outputs with the local ScanNet benchmark stack."
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        dest="scenes",
        help="ScanNet scene names, e.g. scene0000_00 scene0011_00. Defaults to all extracted scenes under --input-root.",
    )
    parser.add_argument("--work-dir", required=True, type=Path, help="Benchmark workspace/output directory")
    parser.add_argument("--input-root", required=True, type=Path, help="Canonical ViPE ScanNet scene root")
    parser.add_argument("--raw-root", required=True, type=Path, help="Raw ScanNet scans root with GT meshes")
    parser.add_argument("--print-only", action="store_true", help="Only print saved metrics")
    parser.add_argument("--do-final-eval", action="store_true", help="Run final pose/reconstruction eval after exports")
    parser.add_argument("--gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--total-gpus", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--scene-worker-scene", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scene-worker-output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--final-eval-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _resolve_vipe_output_dir(args: argparse.Namespace, scene: str) -> Path:
    return args.work_dir / "vipe_outputs" / scene


def _scene_dir(args: argparse.Namespace, scene: str) -> Path:
    return args.input_root / scene


def run_vipe(scene_dir: Path, pipeline_cfg, output_dir: Path) -> None:
    from vipe.pipeline import VipePipeline
    from vipe.stream import FrameDir
    from vipe.utils.logging import configure_logging

    seed_everything(pipeline_cfg.seed, temporary_determinism=pipeline_cfg.temporary_determinism)
    logger = configure_logging()
    pipeline = VipePipeline(
        slam=pipeline_cfg.pipeline.slam,
        output=pipeline_cfg.pipeline.output,
        output_dir=output_dir,
    )
    stream = FrameDir(scene_dir)
    logger.info(f"Running ViPE on {stream.name}")
    pipeline.run(stream)


def _artifact_name(scene_dir: Path) -> str:
    return scene_dir.name


def _write_vipe_manifest(
    args: argparse.Namespace,
    evaluator,
    scene: str,
    scene_data,
    vipe_output_dir: Path,
    artifact_name: str,
    pipeline_cfg,
) -> None:
    frame_count = len(scene_data.image_files)
    print(f"[INFO] Writing ViPE benchmark manifest | {scene} | frames={frame_count}", flush=True)

    pose_path = vipe_output_dir / "pose" / f"{artifact_name}.npz"
    tsdf_pcd_path = vipe_output_dir / "pcd" / f"{artifact_name}_tsdf.ply"
    required_artifacts = [pose_path, tsdf_pcd_path]
    missing_artifacts = [path for path in required_artifacts if not path.exists()]
    if missing_artifacts:
        raise FileNotFoundError("Missing ViPE artifacts:\n" + "\n".join(str(path) for path in missing_artifacts))

    manifest = {
        "format": "vipe_artifacts_v1",
        "scene": scene,
        "artifact_name": artifact_name,
        "vipe_output_dir": str(vipe_output_dir.resolve()),
        "pose_path": str(pose_path.resolve()),
        "tsdf_pcd_path": str(tsdf_pcd_path.resolve()),
        "output": {
            "pcd_max_points": int(pipeline_cfg.pipeline.output.pcd_max_points),
            "pcd_tsdf_voxel_edge_m": float(pipeline_cfg.pipeline.output.pcd_tsdf_voxel_edge_m),
            "pcd_tsdf_sdf_trunc_m": float(pipeline_cfg.pipeline.output.pcd_tsdf_sdf_trunc_m),
            "pcd_tsdf_depth_trunc_m": float(pipeline_cfg.pipeline.output.pcd_tsdf_depth_trunc_m),
            "pcd_tsdf_num_voxels_per_block_edge": int(pipeline_cfg.pipeline.output.pcd_tsdf_num_voxels_per_block_edge),
            "pcd_tsdf_depth_sampling_stride": int(pipeline_cfg.pipeline.output.pcd_tsdf_depth_sampling_stride),
        },
        "frame_count": frame_count,
    }

    export_dir = Path(evaluator._export_dir("scannet", scene))
    exports_dir = export_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = exports_dir / "vipe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote ViPE manifest | {scene} | {manifest_path}", flush=True)
    print(f"[INFO] Writing GT metadata | {scene}", flush=True)
    evaluator._save_gt_meta(str(export_dir), scene_data)

    print(f"[INFO] Exported ViPE benchmark manifest for {scene} under {args.work_dir}")


def _metric_dir(args: argparse.Namespace) -> Path:
    return args.work_dir / "metric_results"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4) + "\n", encoding="utf-8")


def _write_scene_queue(path: Path, scenes: list[str]) -> None:
    _write_json(path, {"next_index": 0, "scenes": scenes})


def _worker_timing_path(args: argparse.Namespace) -> Path:
    return _metric_dir(args) / "timing_workers" / f"scannet_timing_worker_{args.gpu_id}.json"


def _final_eval_worker_path(args: argparse.Namespace) -> Path:
    return _metric_dir(args) / "final_eval_workers" / f"scannet_final_eval_worker_{args.gpu_id}.json"


def _timing_entry(frames: int, seconds: float, peak_vram_mb: float | None = None) -> dict[str, float]:
    frames = int(frames)
    seconds = float(seconds)
    fps = frames / seconds if seconds > 0.0 else 0.0
    entry = {"frames": frames, "seconds": seconds, "fps": fps}
    if peak_vram_mb is not None:
        entry["peak_vram_mb"] = float(peak_vram_mb)
    return entry


def _write_worker_timing(args: argparse.Namespace, build_timing: dict, metric_timing: dict) -> None:
    _write_json(_worker_timing_path(args), {"build": build_timing, "metric_eval": metric_timing})


def _worker_failures_path(args: argparse.Namespace) -> Path:
    return _metric_dir(args) / "failed_scenes" / f"scannet_failed_worker_{args.gpu_id}.json"


def _write_worker_failures(args: argparse.Namespace, failures: dict) -> None:
    _write_json(_worker_failures_path(args), failures)


def _clear_failure_records(args: argparse.Namespace) -> None:
    shutil.rmtree(_metric_dir(args) / "failed_scenes", ignore_errors=True)
    (_metric_dir(args) / "failed_scenes.json").unlink(missing_ok=True)


def _clear_final_eval_outputs(args: argparse.Namespace) -> None:
    for filename in ("scannet_pose.json", "scannet_recon.json"):
        (_metric_dir(args) / filename).unlink(missing_ok=True)


def _cleanup_cuda_runtime() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _incremental_pose_dir(args: argparse.Namespace) -> Path:
    return _metric_dir(args) / "incremental_pose" / "scannet"


def _write_incremental_pose_metric(args: argparse.Namespace, evaluator, scene: str) -> None:
    dataset = evaluator.datasets["scannet"]
    export_dir = evaluator._export_dir("scannet", scene)
    result_path = dataset.result_path(export_dir)
    if not dataset.result_exists(result_path):
        raise FileNotFoundError(f"Result file not found for incremental pose eval: {result_path}")

    gt_meta = evaluator._load_gt_meta(export_dir)
    scene_data = gt_meta if gt_meta is not None else dataset.get_data(scene)
    start = time.perf_counter()
    result = evaluator._compute_pose_with_gt(dataset, result_path, scene_data)
    payload = {
        "scene": scene,
        "frames": int(len(scene_data["extrinsics"])),
        "seconds": float(time.perf_counter() - start),
        "metrics": evaluator._to_float_dict(result),
    }
    out_path = _incremental_pose_dir(args) / f"{scene}.json"
    _write_json(out_path, payload)
    print(f"[INFO] Incremental pose done | scannet | {scene} | {payload['metrics']} | {out_path}", flush=True)


def _load_parallel_timing(args: argparse.Namespace, total_gpus: int) -> tuple[dict, dict]:
    build_timing = {}
    metric_timing = {}
    for gpu_id in range(total_gpus):
        path = _metric_dir(args) / "timing_workers" / f"scannet_timing_worker_{gpu_id}.json"
        if not path.exists():
            print(f"[WARN] Missing timing file from ScanNet worker {gpu_id}: {path}", flush=True)
            continue
        with path.open(encoding="utf-8") as f:
            worker_timing = json.load(f)
        build_timing.update(worker_timing.get("build", {}))
        metric_timing.update(worker_timing.get("metric_eval", {}))
    return build_timing, metric_timing


def _load_parallel_failures(args: argparse.Namespace, total_gpus: int) -> dict:
    failures = {}
    for gpu_id in range(total_gpus):
        path = _metric_dir(args) / "failed_scenes" / f"scannet_failed_worker_{gpu_id}.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            failures.update(json.load(f))
    if failures:
        _write_json(_metric_dir(args) / "failed_scenes.json", failures)
    return failures


def _clear_final_eval_worker_outputs(args: argparse.Namespace) -> None:
    shutil.rmtree(_metric_dir(args) / "final_eval_workers", ignore_errors=True)


def _mean_metric_dict(scene_metrics: dict[str, dict]) -> dict[str, float]:
    metrics = [item for item in scene_metrics.values() if isinstance(item, dict)]
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(np.mean([float(item[key]) for item in metrics]).item()) for key in keys}


def _merge_metric_section(worker_payloads: list[dict], section: str) -> dict:
    merged = {}
    for payload in worker_payloads:
        section_metrics = payload.get("metrics", {}).get(section, {})
        if not isinstance(section_metrics, dict):
            continue
        for scene, metrics in section_metrics.items():
            if scene == "mean":
                continue
            merged[scene] = metrics
    if merged:
        merged["mean"] = _mean_metric_dict(merged)
    return merged


def _load_final_eval_worker_outputs(args: argparse.Namespace, total_gpus: int) -> list[dict]:
    payloads = []
    for gpu_id in range(total_gpus):
        path = _metric_dir(args) / "final_eval_workers" / f"scannet_final_eval_worker_{gpu_id}.json"
        if not path.exists():
            print(f"[WARN] Missing final eval file from ScanNet worker {gpu_id}: {path}", flush=True)
            continue
        with path.open(encoding="utf-8") as f:
            payloads.append(json.load(f))
    return payloads


def _merge_scene_timing(base: dict, extra: dict) -> dict:
    merged = dict(base)
    for scene, entry in extra.items():
        frames = int(entry["frames"])
        seconds = float(entry["seconds"])
        if scene in merged:
            frames = int(merged[scene].get("frames", frames))
            seconds += float(merged[scene].get("seconds", 0.0))
        merged[scene] = _timing_entry(frames, seconds)
    return merged


def _mean_fps(scene_timing: dict) -> float:
    fps_values = [float(entry["fps"]) for entry in scene_timing.values()]
    if not fps_values:
        return 0.0
    return float(np.mean(fps_values).item())


def _mean_peak_vram_mb(scene_timing: dict) -> float:
    values = [float(entry["peak_vram_mb"]) for entry in scene_timing.values() if "peak_vram_mb" in entry]
    return float(np.mean(values).item()) if values else 0.0


def _write_timing(args: argparse.Namespace, build_timing: dict, metric_timing: dict) -> dict:
    timing = {
        "build": {
            "mean_fps": _mean_fps(build_timing),
            "mean_peak_vram_mb": _mean_peak_vram_mb(build_timing),
            "scenes": build_timing,
        },
        "metric_eval": {
            "mean_fps": _mean_fps(metric_timing),
            "scenes": metric_timing,
        },
    }
    _write_json(_metric_dir(args) / "scannet_timing.json", timing)
    return timing


def _load_evaluator(args: argparse.Namespace, eval_config: dict[str, Any]):
    return ScanNetEvaluator(
        work_dir=str(args.work_dir),
        scenes=args.scenes,
        input_root=args.input_root,
        raw_root=args.raw_root,
        eval_config=eval_config,
    )


def _skip_bad_gt_pose_scenes(args: argparse.Namespace, evaluator) -> list[tuple[str, Any]]:
    dataset = evaluator.datasets["scannet"]
    kept, skipped = filter_scenes_with_bad_gt_pose_data(dataset, list(args.scenes))
    args.scenes = kept
    evaluator.scenes_filter = kept
    if os.environ.get(WORKER_ENV) != "1":
        _write_json(
            _metric_dir(args) / "skipped_gt_pose_scenes.json",
            {scene: report.summary() for scene, report in skipped},
        )
    if skipped:
        scene_names = "; ".join(f"{scene} ({report.summary()})" for scene, report in skipped)
        print(f"[INFO] Skipped {len(skipped)} ScanNet scene(s) with bad GT pose data: {scene_names}", flush=True)
    return skipped


def _append_common_cli_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    if args.scenes is not None:
        cmd += ["--scenes", *args.scenes]
    cmd += ["--work-dir", str(args.work_dir)]
    cmd += ["--input-root", str(args.input_root)]
    cmd += ["--raw-root", str(args.raw_root)]
    return cmd


def _cuda_visible_gpu_list() -> list[str]:
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_devices is not None and cuda_devices.strip():
        return [gpu.strip() for gpu in cuda_devices.split(",") if gpu.strip()]

    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


def maybe_spawn_workers(args: argparse.Namespace) -> tuple[dict, dict] | None:
    if args.print_only or os.environ.get(WORKER_ENV) == "1":
        return None

    gpu_list = _cuda_visible_gpu_list()

    if len(gpu_list) <= 1:
        return None

    base_cmd = _append_common_cli_args([sys.executable, os.path.abspath(__file__)], args)
    scene_queue_path = _metric_dir(args) / "dynamic_scene_queue.json"
    _write_scene_queue(scene_queue_path, list(args.scenes))

    print(f"[INFO] Detected {len(gpu_list)} GPUs for ViPE ScanNet benchmark: {gpu_list}")
    processes = []
    for idx, visible_gpu in enumerate(gpu_list):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        env[WORKER_ENV] = "1"
        env[SCENE_QUEUE_ENV] = str(scene_queue_path)
        cmd = base_cmd + ["--gpu-id", str(idx), "--total-gpus", str(len(gpu_list))]
        print(f"[INFO] Starting ViPE worker {idx} on GPU {visible_gpu}")
        processes.append(subprocess.Popen(cmd, env=env))

    failed_workers = []
    for idx, process in enumerate(processes):
        process.wait()
        if process.returncode != 0:
            failed_workers.append((idx, process.returncode))

    if failed_workers:
        print(f"[WARN] ScanNet worker failures: {failed_workers}. Continuing with completed scene artifacts.", flush=True)

    print("[INFO] All ViPE workers completed")
    _load_parallel_failures(args, len(gpu_list))
    return _load_parallel_timing(args, len(gpu_list))


def _claim_scene_from_queue(queue_path: Path) -> tuple[int, int, str | None]:
    import fcntl

    with queue_path.open("r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        state = json.load(f)
        scenes = state["scenes"]
        next_index = int(state["next_index"])
        if next_index >= len(scenes):
            scene = None
        else:
            scene = scenes[next_index]
            state["next_index"] = next_index + 1
            f.seek(0)
            json.dump(state, f, indent=4)
            f.write("\n")
            f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)
    return next_index, len(scenes), scene


def _dynamic_scenes_for_worker(args: argparse.Namespace) -> Iterable[str]:
    queue_path = Path(os.environ[SCENE_QUEUE_ENV])
    while True:
        scene_idx, total_scenes, scene = _claim_scene_from_queue(queue_path)
        if scene is None:
            print(f"[INFO] ViPE worker {args.gpu_id}/{args.total_gpus}: dynamic queue empty", flush=True)
            return
        print(
            f"[INFO] ViPE worker {args.gpu_id}/{args.total_gpus}: claimed {scene_idx + 1}/{total_scenes} {scene}",
            flush=True,
        )
        yield scene


def _scenes_for_worker(args: argparse.Namespace, evaluator) -> Iterable[str]:
    dataset = evaluator.datasets["scannet"]
    known_scenes = set(evaluator._get_scenes(dataset))
    missing = sorted(set(args.scenes) - known_scenes)
    if missing:
        raise ValueError(f"Unknown ScanNet scenes: {missing}")

    all_scenes = list(args.scenes)
    if args.total_gpus <= 1:
        print(f"[INFO] Total ViPE benchmark scenes: {len(all_scenes)}")
        return all_scenes

    if os.environ.get(SCENE_QUEUE_ENV):
        print(f"[INFO] ViPE worker {args.gpu_id}/{args.total_gpus}: using dynamic scene queue")
        return _dynamic_scenes_for_worker(args)

    scenes = [scene for idx, scene in enumerate(all_scenes) if idx % args.total_gpus == args.gpu_id]
    print(f"[INFO] ViPE worker {args.gpu_id}/{args.total_gpus}: {len(scenes)}/{len(all_scenes)} scenes")
    return scenes


def _gpu_memory_used_mb() -> float:
    visible_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    if "," in visible_gpu:
        raise ValueError(f"Scene worker requires one visible GPU, got {visible_gpu}")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            visible_gpu,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _run_scene_subprocess(args: argparse.Namespace, scene: str, vipe_output_dir: Path) -> float:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--work-dir",
        str(args.work_dir),
        "--input-root",
        str(args.input_root),
        "--raw-root",
        str(args.raw_root),
        "--scene-worker-scene",
        scene,
        "--scene-worker-output-dir",
        str(vipe_output_dir),
    ]
    env = os.environ.copy()
    env[WORKER_ENV] = "1"
    process = subprocess.Popen(cmd, env=env)
    peak_vram_mb = _gpu_memory_used_mb()
    while process.poll() is None:
        peak_vram_mb = max(peak_vram_mb, _gpu_memory_used_mb())
        time.sleep(0.2)
    if process.returncode != 0:
        raise RuntimeError(f"scene subprocess exited with code {process.returncode}")
    return peak_vram_mb


def _scene_export_manifest_path(args: argparse.Namespace, scene: str) -> Path:
    return args.work_dir / "model_results" / "scannet" / scene / "recon" / "exports" / "vipe_manifest.json"


def _scene_gt_meta_path(args: argparse.Namespace, scene: str) -> Path:
    return _scene_export_manifest_path(args, scene).with_name("gt_meta.npz")


def _scene_artifact_paths(scene_dir: Path, vipe_output_dir: Path) -> tuple[Path, Path]:
    artifact_name = _artifact_name(scene_dir)
    return (
        vipe_output_dir / "pose" / f"{artifact_name}.npz",
        vipe_output_dir / "pcd" / f"{artifact_name}_tsdf.ply",
    )


def _load_matching_manifest(manifest_path: Path, args: argparse.Namespace, scene: str) -> dict | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    scene_dir = _scene_dir(args, scene)
    if int(manifest.get("frame_count", -1)) != scene_frame_count(scene_dir):
        return None
    contract_paths = [scene_dir / "metadata.json", scene_dir / "intrinsic" / "intrinsic_color.json"]
    if manifest_path.stat().st_mtime_ns < max(path.stat().st_mtime_ns for path in contract_paths):
        return None
    return manifest


def _scene_build_matches_request(args: argparse.Namespace, scene: str) -> bool:
    manifest = _load_matching_manifest(_scene_export_manifest_path(args, scene), args, scene)
    return manifest is not None and (
        Path(manifest["pose_path"]).exists()
        and Path(manifest["tsdf_pcd_path"]).exists()
    )


def _scene_export_complete(args: argparse.Namespace, scene: str) -> bool:
    return _scene_build_matches_request(args, scene) and _scene_gt_meta_path(args, scene).exists()


def _completed_scenes(args: argparse.Namespace) -> list[str]:
    return [scene for scene in args.scenes if _scene_export_complete(args, scene)]


def _restrict_eval_to_completed_scenes(args: argparse.Namespace, evaluator) -> None:
    completed = _completed_scenes(args)
    dropped = [scene for scene in args.scenes if scene not in completed]
    if dropped:
        print(
            f"[WARN] Skipping {len(dropped)} ScanNet scene(s) without complete ViPE artifacts: {', '.join(dropped)}",
            flush=True,
        )
    args.scenes = completed
    evaluator.scenes_filter = completed


def _run_final_eval_worker(args: argparse.Namespace, evaluator) -> None:
    args.scenes = _scenes_for_worker(args, evaluator)
    evaluator.scenes_filter = args.scenes
    _restrict_eval_to_completed_scenes(args, evaluator)
    metrics = evaluator.eval(dump=False)
    _write_json(
        _final_eval_worker_path(args),
        {
            "metrics": metrics,
            "metric_eval": evaluator.metric_eval_timing(),
        },
    )


def _run_final_eval(args: argparse.Namespace, evaluator) -> tuple[dict, dict]:
    gpu_list = _cuda_visible_gpu_list()
    if len(gpu_list) <= 1 or os.environ.get(WORKER_ENV) == "1":
        _restrict_eval_to_completed_scenes(args, evaluator)
        metrics = evaluator.eval()
        return metrics, evaluator.metric_eval_timing()

    _clear_final_eval_worker_outputs(args)
    base_cmd = _append_common_cli_args(
        [sys.executable, os.path.abspath(__file__), "--do-final-eval", "--final-eval-worker"],
        args,
    )

    print(f"[INFO] Detected {len(gpu_list)} GPUs for ScanNet final eval: {gpu_list}")
    processes = []
    for idx, visible_gpu in enumerate(gpu_list):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        env[WORKER_ENV] = "1"
        cmd = base_cmd + ["--gpu-id", str(idx), "--total-gpus", str(len(gpu_list))]
        print(f"[INFO] Starting ScanNet final eval worker {idx} on GPU {visible_gpu}")
        processes.append(subprocess.Popen(cmd, env=env))

    failed_workers = []
    for idx, process in enumerate(processes):
        process.wait()
        if process.returncode != 0:
            failed_workers.append((idx, process.returncode))
    if failed_workers:
        print(f"[WARN] ScanNet final eval worker failures: {failed_workers}. Merging completed worker outputs.", flush=True)

    payloads = _load_final_eval_worker_outputs(args, len(gpu_list))
    metrics = {
        "scannet_pose": _merge_metric_section(payloads, "scannet_pose"),
        "scannet_recon": _merge_metric_section(payloads, "scannet_recon"),
    }
    _write_json(_metric_dir(args) / "scannet_pose.json", metrics["scannet_pose"])
    _write_json(_metric_dir(args) / "scannet_recon.json", metrics["scannet_recon"])

    metric_timing = {}
    for payload in payloads:
        metric_timing = _merge_scene_timing(metric_timing, payload.get("metric_eval", {}))
    return metrics, metric_timing


def prepare_vipe_benchmark_exports(
    args: argparse.Namespace,
    evaluator,
    pipeline_cfg,
) -> tuple[dict, dict]:
    build_timing = {}
    metric_timing = {}
    failures = {}
    for scene in _scenes_for_worker(args, evaluator):
        start_scene = time.perf_counter()
        scene_dir = _scene_dir(args, scene)
        vipe_output_dir = _resolve_vipe_output_dir(args, scene)
        scene_data = evaluator.datasets["scannet"].get_data(scene)
        pose_path, tsdf_pcd_path = _scene_artifact_paths(scene_dir, vipe_output_dir)
        if _scene_export_complete(args, scene):
            print(f"[INFO] Reusing existing ViPE artifacts | {scene}", flush=True)
            inc_path = _incremental_pose_dir(args) / f"{scene}.json"
            if not inc_path.exists():
                _write_incremental_pose_metric(args, evaluator, scene)
            continue
        if pose_path.exists() and tsdf_pcd_path.exists() and _scene_build_matches_request(args, scene):
            print(f"[INFO] Reusing existing ViPE build outputs and refreshing manifest | {scene}", flush=True)
            _write_vipe_manifest(
                args,
                evaluator,
                scene,
                scene_data,
                vipe_output_dir,
                _artifact_name(scene_dir),
                pipeline_cfg,
            )
            inc_path = _incremental_pose_dir(args) / f"{scene}.json"
            if not inc_path.exists():
                _write_incremental_pose_metric(args, evaluator, scene)
            continue
        if pose_path.exists() or tsdf_pcd_path.exists():
            print(f"[INFO] Removing stale ViPE artifacts for changed canonical input | {scene}", flush=True)
            shutil.rmtree(vipe_output_dir, ignore_errors=True)
            shutil.rmtree(args.work_dir / "model_results" / "scannet" / scene, ignore_errors=True)
        start_build = time.perf_counter()
        try:
            peak_vram_mb = _run_scene_subprocess(args, scene, vipe_output_dir)
        except Exception as exc:
            failures[scene] = str(exc)
            print(f"[WARN] Skipping ScanNet scene after ViPE failure | {scene} | {exc}", flush=True)
            shutil.rmtree(vipe_output_dir, ignore_errors=True)
            shutil.rmtree(args.work_dir / "model_results" / "scannet" / scene, ignore_errors=True)
            _write_worker_failures(args, failures)
            _cleanup_cuda_runtime()
            continue
        build_seconds = time.perf_counter() - start_build
        _write_vipe_manifest(
            args,
            evaluator,
            scene,
            scene_data,
            vipe_output_dir,
            _artifact_name(scene_dir),
            pipeline_cfg,
        )
        frames = len(scene_data.image_files)
        metric_seconds = max(0.0, time.perf_counter() - start_scene - build_seconds)
        build_timing[scene] = _timing_entry(frames, build_seconds, peak_vram_mb)
        metric_timing[scene] = _timing_entry(frames, metric_seconds)
        _write_incremental_pose_metric(args, evaluator, scene)
        _cleanup_cuda_runtime()
    if failures:
        _write_worker_failures(args, failures)
    return build_timing, metric_timing


def _fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def _fmt_fps(value):
    return "N/A" if value is None else f"{float(value):.2f}"


def _fmt_scale_summary(recon_metrics: dict) -> str:
    mean_scale = _fmt(_get_nested(recon_metrics, "mean", "scale_diagnostic"))
    scene_scales = [
        _fmt(result.get("scale_diagnostic"))
        for scene, result in recon_metrics.items()
        if scene != "mean" and isinstance(result, dict) and "scale_diagnostic" in result
    ]
    if not scene_scales:
        return mean_scale
    return f"{mean_scale} ({', '.join(scene_scales)})"


def _get_nested(d, *keys):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def print_scannet_summary(metrics) -> None:
    pose_mean = _get_nested(metrics, "scannet_pose", "mean") or {}
    recon_metrics = _get_nested(metrics, "scannet_recon") or {}
    recon = recon_metrics.get("mean") or {}
    timing = _get_nested(metrics, "scannet_timing") or {}

    auc3 = pose_mean.get("auc03")
    auc30 = pose_mean.get("auc30")

    col1 = 15
    col2 = 14
    print("\n" + "=" * 44)
    print("SCANNET VIPE BENCHMARK SUMMARY")
    print("=" * 44)
    print("\nPOSE ESTIMATION")
    print("-" * (col1 + col2))
    print(f"{'Metric':<{col1}}{'ScanNet':<{col2}}")
    print("-" * (col1 + col2))
    print(f"{'Auc3':<{col1}}{_fmt(auc3):<{col2}}")
    print(f"{'Auc30':<{col1}}{_fmt(auc30):<{col2}}")
    print("\nRECONSTRUCTION")
    print("-" * (col1 + col2))
    print(f"{'Metric':<{col1}}{'ScanNet':<{col2}}")
    print("-" * (col1 + col2))
    print(f"{'Overall':<{col1}}{_fmt(recon.get('overall')):<{col2}}")
    print(f"{'Scale':<{col1}}{_fmt_scale_summary(recon_metrics):<{col2}}")
    print(f"{'PSNR':<{col1}}{_fmt(recon.get('psnr')):<{col2}}")
    print(f"{'SSIM':<{col1}}{_fmt(recon.get('ssim')):<{col2}}")

    print("\nFPS")
    print("-" * (col1 + col2))
    print(f"{'Stage':<{col1}}{'Frames/sec':<{col2}}")
    print("-" * (col1 + col2))
    print(f"{'Build Run':<{col1}}{_fmt_fps(_get_nested(timing, 'build', 'mean_fps')):<{col2}}")
    print(f"{'Metric Eval':<{col1}}{_fmt_fps(_get_nested(timing, 'metric_eval', 'mean_fps')):<{col2}}")


def _finish_without_final_eval(args: argparse.Namespace, build_timing: dict, metric_timing: dict) -> None:
    _clear_final_eval_outputs(args)
    timing = _write_timing(args, build_timing, metric_timing)
    completed = len(_completed_scenes(args))
    print(
        f"[INFO] Skipping final ScanNet eval (--do-final-eval not set). "
        f"Completed artifacts: {completed}/{len(args.scenes)} | "
        f"build_fps={timing['build']['mean_fps']:.2f}",
        flush=True,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline_cfg = load_yaml_config(PIPELINE_CONFIG_PATH)
    eval_config = load_yaml_config(EVAL_CONFIG_PATH)
    seed_everything(int(eval_config.seed), temporary_determinism=eval_config.temporary_determinism)
    is_worker = os.environ.get(WORKER_ENV) == "1"

    if args.scene_worker_scene is not None:
        if args.scene_worker_output_dir is None:
            raise ValueError("--scene-worker-output-dir is required with --scene-worker-scene")
        run_vipe(
            _scene_dir(args, args.scene_worker_scene),
            pipeline_cfg,
            args.scene_worker_output_dir,
        )
        return

    evaluator = _load_evaluator(args, eval_config)
    if args.scenes is None:
        args.scenes = evaluator._get_scenes(evaluator.datasets["scannet"])
        if not args.scenes:
            raise ValueError(f"No extracted ScanNet scenes found under {args.input_root}")
        print(f"[INFO] Using all extracted ScanNet scenes from {args.input_root}: {len(args.scenes)} scenes")
    evaluator.scenes_filter = args.scenes

    if args.print_only:
        metrics = evaluator._load_metrics()
        print_scannet_summary(metrics)
        return

    _skip_bad_gt_pose_scenes(args, evaluator)
    if args.final_eval_worker:
        _run_final_eval_worker(args, evaluator)
        return

    if not is_worker:
        _clear_failure_records(args)
        if not args.do_final_eval:
            _clear_final_eval_outputs(args)

    build_timing = maybe_spawn_workers(args)
    if build_timing is not None:
        build_scene_timing, metric_scene_timing = build_timing
        if not args.do_final_eval:
            _finish_without_final_eval(args, build_scene_timing, metric_scene_timing)
            return
        metrics, final_metric_timing = _run_final_eval(args, evaluator)
        metric_scene_timing = _merge_scene_timing(metric_scene_timing, final_metric_timing)
        metrics["scannet_timing"] = _write_timing(args, build_scene_timing, metric_scene_timing)
        print_scannet_summary(metrics)
        return

    build_scene_timing, metric_scene_timing = prepare_vipe_benchmark_exports(args, evaluator, pipeline_cfg)
    if is_worker:
        _write_worker_timing(args, build_scene_timing, metric_scene_timing)
        return
    if not args.do_final_eval:
        _finish_without_final_eval(args, build_scene_timing, metric_scene_timing)
        return
    metrics, final_metric_timing = _run_final_eval(args, evaluator)
    metric_scene_timing = _merge_scene_timing(metric_scene_timing, final_metric_timing)
    metrics["scannet_timing"] = _write_timing(args, build_scene_timing, metric_scene_timing)
    print_scannet_summary(metrics)


if __name__ == "__main__":
    main()
