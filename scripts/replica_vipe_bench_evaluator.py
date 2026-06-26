#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time

from pathlib import Path
from typing import Any

import numpy as np
from vipe import get_config_path
from vipe.bench.gt_pose_checks import filter_scenes_with_bad_gt_pose_data
from vipe.bench.replica import ReplicaEvaluator
from vipe.bench.scannet import AttrDict
from vipe.utils.config import load_yaml_config
from vipe.utils.determinism import seed_everything


WORKER_ENV = "_VIPE_REPLICA_BENCH_WORKER"
PIPELINE_CONFIG_PATH = get_config_path() / "default.yaml"
EVAL_CONFIG_PATH = get_config_path() / "eval_replica_config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ViPE Replica outputs with the local Replica benchmark stack."
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        dest="scenes",
        help="Replica scene names, e.g. office0 office1 room0. Defaults to all extracted scenes under --input-root.",
    )
    parser.add_argument("--work-dir", required=True, type=Path, help="Benchmark workspace/output directory")
    parser.add_argument("--input-root", required=True, type=Path, help="Canonical ViPE Replica scene root")
    parser.add_argument("--raw-root", required=True, type=Path, help="Full Replica asset root with GT meshes")
    parser.add_argument("--print-only", action="store_true", help="Only print saved metrics")
    parser.add_argument("--gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--total-gpus", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--final-eval-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _resolve_vipe_output_dir(args: argparse.Namespace, scene: str) -> Path:
    return args.work_dir / "vipe_outputs" / scene


def _scene_dir(args: argparse.Namespace, scene: str) -> Path:
    return args.input_root / scene


def run_vipe(scene_dir: Path, pipeline_cfg, output_dir: Path) -> None:
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
    logger.info(f"Running ViPE on {stream.name()}")
    pipeline.run(stream)


def _subset_scene_data(scene_data, keep_indices: list[int]):
    subset = AttrDict()
    subset.image_files = [scene_data.image_files[i] for i in keep_indices]
    subset.extrinsics = scene_data.extrinsics[keep_indices]
    subset.intrinsics = scene_data.intrinsics[keep_indices]
    subset.aux = AttrDict()
    for key, val in scene_data.aux.items():
        if isinstance(val, list) and len(val) == len(scene_data.image_files):
            subset.aux[key] = [val[i] for i in keep_indices]
        elif isinstance(val, np.ndarray) and len(val) == len(scene_data.image_files):
            subset.aux[key] = val[keep_indices]
        else:
            subset.aux[key] = val
    return subset


def _benchmark_frame_request(
    evaluator,
    scene: str,
) -> tuple[Any, list[int], list[int]]:
    dataset = evaluator.datasets["replica"]
    full_scene_data = dataset.get_data(scene)
    frame_indices = list(range(len(full_scene_data.image_files)))
    kept_scene_indices = list(frame_indices)
    return full_scene_data, frame_indices, kept_scene_indices


def _artifact_name(scene_dir: Path) -> str:
    return scene_dir.name


def _write_vipe_manifest(
    args: argparse.Namespace,
    evaluator,
    scene: str,
    full_scene_data,
    frame_indices: list[int],
    kept_scene_indices: list[int],
    vipe_output_dir: Path,
    artifact_name: str,
    pipeline_cfg,
) -> None:
    print(f"[INFO] Writing ViPE benchmark manifest | {scene} | frames={len(frame_indices)}", flush=True)

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
        "frame_indices": [int(idx) for idx in frame_indices],
    }

    exported_scene_data = _subset_scene_data(full_scene_data, kept_scene_indices)
    export_dir = Path(evaluator._export_dir("replica", scene))
    exports_dir = export_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = exports_dir / "vipe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote ViPE manifest | {scene} | {manifest_path}", flush=True)
    print(f"[INFO] Writing GT metadata | {scene}", flush=True)
    evaluator._save_gt_meta(str(export_dir), exported_scene_data)

    print(f"[INFO] Exported ViPE benchmark manifest for {scene} under {args.work_dir}")


def _metric_dir(args: argparse.Namespace) -> Path:
    return args.work_dir / "metric_results"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4) + "\n", encoding="utf-8")


def _worker_timing_path(args: argparse.Namespace) -> Path:
    return _metric_dir(args) / "timing_workers" / f"replica_timing_worker_{args.gpu_id}.json"


def _final_eval_worker_path(args: argparse.Namespace) -> Path:
    return _metric_dir(args) / "final_eval_workers" / f"replica_final_eval_worker_{args.gpu_id}.json"


def _timing_entry(frames: int, seconds: float) -> dict[str, float]:
    frames = int(frames)
    seconds = float(seconds)
    fps = frames / seconds if seconds > 0.0 else 0.0
    return {"frames": frames, "seconds": seconds, "fps": fps}


def _write_worker_timing(args: argparse.Namespace, build_timing: dict, metric_timing: dict) -> None:
    _write_json(_worker_timing_path(args), {"build": build_timing, "metric_eval": metric_timing})


def _load_parallel_timing(args: argparse.Namespace, total_gpus: int) -> tuple[dict, dict]:
    build_timing = {}
    metric_timing = {}
    for gpu_id in range(total_gpus):
        path = _metric_dir(args) / "timing_workers" / f"replica_timing_worker_{gpu_id}.json"
        with path.open(encoding="utf-8") as f:
            worker_timing = json.load(f)
        build_timing.update(worker_timing.get("build", {}))
        metric_timing.update(worker_timing.get("metric_eval", {}))
    return build_timing, metric_timing


def _clear_final_eval_worker_outputs(args: argparse.Namespace) -> None:
    import shutil

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
        path = _metric_dir(args) / "final_eval_workers" / f"replica_final_eval_worker_{gpu_id}.json"
        if not path.exists():
            print(f"[WARN] Missing final eval file from Replica worker {gpu_id}: {path}", flush=True)
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


def _write_timing(args: argparse.Namespace, build_timing: dict, metric_timing: dict) -> dict:
    timing = {
        "build": {
            "mean_fps": _mean_fps(build_timing),
            "scenes": build_timing,
        },
        "metric_eval": {
            "mean_fps": _mean_fps(metric_timing),
            "scenes": metric_timing,
        },
    }
    _write_json(_metric_dir(args) / "replica_timing.json", timing)
    return timing


def _load_evaluator(args: argparse.Namespace, eval_config: dict[str, Any]):
    return ReplicaEvaluator(
        work_dir=str(args.work_dir),
        scenes=args.scenes,
        input_root=args.input_root,
        raw_root=args.raw_root,
        eval_config=eval_config,
    )


def _skip_bad_gt_pose_scenes(args: argparse.Namespace, evaluator) -> list[tuple[str, Any]]:
    dataset = evaluator.datasets["replica"]
    kept, skipped = filter_scenes_with_bad_gt_pose_data(dataset, list(args.scenes))
    args.scenes = kept
    evaluator.scenes_filter = kept
    if skipped:
        scene_names = "; ".join(f"{scene} ({report.summary()})" for scene, report in skipped)
        print(f"[INFO] Skipped {len(skipped)} Replica scene(s) with bad GT pose data: {scene_names}", flush=True)
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

    print(f"[INFO] Detected {len(gpu_list)} GPUs for ViPE Replica benchmark: {gpu_list}")
    processes = []
    for idx, visible_gpu in enumerate(gpu_list):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        env[WORKER_ENV] = "1"
        cmd = base_cmd + ["--gpu-id", str(idx), "--total-gpus", str(len(gpu_list))]
        print(f"[INFO] Starting ViPE worker {idx} on GPU {visible_gpu}")
        processes.append(subprocess.Popen(cmd, env=env))

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise SystemExit(process.returncode)

    print("[INFO] All ViPE workers completed")
    return _load_parallel_timing(args, len(gpu_list))


def _scenes_for_worker(args: argparse.Namespace, evaluator) -> list[str]:
    dataset = evaluator.datasets["replica"]
    known_scenes = set(evaluator._get_scenes(dataset))
    missing = sorted(set(args.scenes) - known_scenes)
    if missing:
        raise ValueError(f"Unknown Replica scenes: {missing}")

    all_scenes = list(args.scenes)
    if args.total_gpus <= 1:
        print(f"[INFO] Total ViPE benchmark scenes: {len(all_scenes)}")
        return all_scenes

    scenes = [scene for idx, scene in enumerate(all_scenes) if idx % args.total_gpus == args.gpu_id]
    print(f"[INFO] ViPE worker {args.gpu_id}/{args.total_gpus}: {len(scenes)}/{len(all_scenes)} scenes")
    return scenes


def _run_final_eval_worker(args: argparse.Namespace, evaluator) -> None:
    args.scenes = _scenes_for_worker(args, evaluator)
    evaluator.scenes_filter = args.scenes
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
        metrics = evaluator.eval()
        return metrics, evaluator.metric_eval_timing()

    _clear_final_eval_worker_outputs(args)
    base_cmd = _append_common_cli_args(
        [sys.executable, os.path.abspath(__file__), "--final-eval-worker"],
        args,
    )

    print(f"[INFO] Detected {len(gpu_list)} GPUs for Replica final eval: {gpu_list}")
    processes = []
    for idx, visible_gpu in enumerate(gpu_list):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        env[WORKER_ENV] = "1"
        cmd = base_cmd + ["--gpu-id", str(idx), "--total-gpus", str(len(gpu_list))]
        print(f"[INFO] Starting Replica final eval worker {idx} on GPU {visible_gpu}")
        processes.append(subprocess.Popen(cmd, env=env))

    failed_workers = []
    for idx, process in enumerate(processes):
        process.wait()
        if process.returncode != 0:
            failed_workers.append((idx, process.returncode))
    if failed_workers:
        print(f"[WARN] Replica final eval worker failures: {failed_workers}. Merging completed worker outputs.", flush=True)

    payloads = _load_final_eval_worker_outputs(args, len(gpu_list))
    metrics = {
        "replica_pose": _merge_metric_section(payloads, "replica_pose"),
        "replica_recon": _merge_metric_section(payloads, "replica_recon"),
    }
    _write_json(_metric_dir(args) / "replica_pose.json", metrics["replica_pose"])
    _write_json(_metric_dir(args) / "replica_recon.json", metrics["replica_recon"])

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
    for scene in _scenes_for_worker(args, evaluator):
        start_scene = time.perf_counter()
        scene_dir = _scene_dir(args, scene)
        vipe_output_dir = _resolve_vipe_output_dir(args, scene)
        full_scene_data, frame_indices, kept_scene_indices = _benchmark_frame_request(evaluator, scene)
        start_build = time.perf_counter()
        run_vipe(scene_dir, pipeline_cfg, vipe_output_dir)
        build_seconds = time.perf_counter() - start_build
        _write_vipe_manifest(
            args,
            evaluator,
            scene,
            full_scene_data,
            frame_indices,
            kept_scene_indices,
            vipe_output_dir,
            _artifact_name(scene_dir),
            pipeline_cfg,
        )
        frames = len(frame_indices)
        metric_seconds = max(0.0, time.perf_counter() - start_scene - build_seconds)
        build_timing[scene] = _timing_entry(frames, build_seconds)
        metric_timing[scene] = _timing_entry(frames, metric_seconds)
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


def print_replica_summary(metrics) -> None:
    pose_mean = _get_nested(metrics, "replica_pose", "mean") or {}
    recon_metrics = _get_nested(metrics, "replica_recon") or {}
    recon = recon_metrics.get("mean") or {}
    timing = _get_nested(metrics, "replica_timing") or {}

    auc3 = pose_mean.get("auc03")
    auc30 = pose_mean.get("auc30")

    col1 = 15
    col2 = 14
    print("\n" + "=" * 44)
    print("REPLICA VIPE BENCHMARK SUMMARY")
    print("=" * 44)
    print("\nPOSE ESTIMATION")
    print("-" * (col1 + col2))
    print(f"{'Metric':<{col1}}{'Replica':<{col2}}")
    print("-" * (col1 + col2))
    print(f"{'Auc3':<{col1}}{_fmt(auc3):<{col2}}")
    print(f"{'Auc30':<{col1}}{_fmt(auc30):<{col2}}")
    print("\nRECONSTRUCTION")
    print("-" * (col1 + col2))
    print(f"{'Metric':<{col1}}{'Replica':<{col2}}")
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline_cfg = load_yaml_config(PIPELINE_CONFIG_PATH)
    eval_config = load_yaml_config(EVAL_CONFIG_PATH)
    seed_everything(int(eval_config.seed), temporary_determinism=eval_config.temporary_determinism)
    is_worker = os.environ.get(WORKER_ENV) == "1"

    evaluator = _load_evaluator(args, eval_config)
    if args.scenes is None:
        args.scenes = evaluator._get_scenes(evaluator.datasets["replica"])
        if not args.scenes:
            raise ValueError(f"No extracted Replica scenes found under {args.input_root}")
        print(f"[INFO] Using all extracted Replica scenes from {args.input_root}: {len(args.scenes)} scenes")
    evaluator.scenes_filter = args.scenes

    if args.print_only:
        metrics = evaluator._load_metrics()
        print_replica_summary(metrics)
        return

    _skip_bad_gt_pose_scenes(args, evaluator)
    if args.final_eval_worker:
        _run_final_eval_worker(args, evaluator)
        return

    build_timing = maybe_spawn_workers(args)
    if build_timing is not None:
        build_scene_timing, metric_scene_timing = build_timing
        metrics, final_metric_timing = _run_final_eval(args, evaluator)
        metric_scene_timing = _merge_scene_timing(metric_scene_timing, final_metric_timing)
        metrics["replica_timing"] = _write_timing(args, build_scene_timing, metric_scene_timing)
        print_replica_summary(metrics)
        return

    build_scene_timing, metric_scene_timing = prepare_vipe_benchmark_exports(args, evaluator, pipeline_cfg)
    if is_worker:
        _write_worker_timing(args, build_scene_timing, metric_scene_timing)
        return
    metrics, final_metric_timing = _run_final_eval(args, evaluator)
    metric_scene_timing = _merge_scene_timing(metric_scene_timing, final_metric_timing)
    metrics["replica_timing"] = _write_timing(args, build_scene_timing, metric_scene_timing)
    print_replica_summary(metrics)


if __name__ == "__main__":
    main()
