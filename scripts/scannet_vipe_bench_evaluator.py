#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys

from pathlib import Path
from typing import Any

import numpy as np

from vipe.utils.determinism import seed_everything
from vipe.utils.misc import sort_image_sequence


DEFAULT_INPUT_ROOT = Path("/robodata/smodak/repos/ovo/data/input/ScanNet")
DEFAULT_RAW_ROOT = Path("/robodata/smodak/datasets/scannet_v2/scans")
DEFAULT_DA3_ROOT = Path("/robodata/smodak/repos/Depth-Anything-3")
WORKER_ENV = "_VIPE_SCANNET_BENCH_WORKER"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ViPE ScanNet outputs with the Depth-Anything-3 benchmark stack."
    )
    parser.add_argument(
        "--scenes",
        "--scene",
        required=True,
        nargs="+",
        dest="scenes",
        help="ScanNet scene names, e.g. scene0000_00 scene0011_00",
    )
    parser.add_argument("--work-dir", required=True, type=Path, help="DA3 benchmark workspace/output directory")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, type=Path, help="Processed ScanNet input root")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, type=Path, help="Raw ScanNet scans root with GT meshes")
    parser.add_argument("--da3-root", default=DEFAULT_DA3_ROOT, type=Path, help="Depth-Anything-3 repo root")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["pose", "recon_unposed", "recon_posed"],
        choices=["pose", "recon_unposed", "recon_posed"],
        help="Benchmark modes to run",
    )
    parser.add_argument("--max-frames", type=int, default=-1, help="Maximum frames to evaluate (-1 for all)")
    parser.add_argument("--num-fusion-workers", type=int, default=4, help="TSDF fusion workers")
    parser.add_argument("--seed", type=int, default=42, help="Seed for Python/NumPy/Torch/Open3D RNGs")
    parser.add_argument("--fps", type=float, default=30.0, help="Default streams.fps if not provided as an override")
    parser.add_argument("--vipe-output-dir", type=Path, default=None, help="ViPE output dir override")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip ViPE inference, write benchmark manifests from existing ViPE artifacts, then evaluate",
    )
    parser.add_argument("--print-only", action="store_true", help="Only print saved metrics")
    parser.add_argument("--gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--total-gpus", type=int, default=1, help=argparse.SUPPRESS)
    return parser


def _override_value(overrides: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for item in reversed(overrides):
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def _set_override(overrides: list[str], key: str, value: Any) -> list[str]:
    prefix = f"{key}="
    return [item for item in overrides if not item.startswith(prefix)] + [f"{key}={value}"]


def _image_files(frame_dir: Path) -> list[Path]:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    files: list[Path] = []
    for ext in exts:
        files.extend(frame_dir.glob(f"*{ext}"))
        files.extend(frame_dir.glob(f"*{ext.upper()}"))
    return sort_image_sequence(set(files))


def _format_scene_path(value: str | Path, scene: str) -> Path:
    return Path(str(value).format(scene=scene))


def _single_scene(args: argparse.Namespace) -> bool:
    return len(args.scenes) == 1


def _resolve_scene_value(args: argparse.Namespace, value: str | Path | None, scene: str, name: str) -> Path | None:
    if value is None:
        return None
    value_str = str(value)
    if "{scene}" in value_str:
        return _format_scene_path(value_str, scene)
    if _single_scene(args):
        return Path(value_str)
    raise ValueError(f"{name} must include '{{scene}}' when evaluating multiple scenes")


def _resolve_frame_dir(args: argparse.Namespace, vipe_overrides: list[str], scene: str) -> Path:
    base_path = _override_value(vipe_overrides, "streams.base_path")
    resolved = _resolve_scene_value(args, base_path, scene, "streams.base_path")
    if resolved is not None:
        return resolved
    return args.input_root / scene / "color"


def _resolve_vipe_output_dir(args: argparse.Namespace, vipe_overrides: list[str], scene: str) -> Path:
    resolved_arg = _resolve_scene_value(args, args.vipe_output_dir, scene, "--vipe-output-dir")
    if resolved_arg is not None:
        return resolved_arg
    output_path = _override_value(vipe_overrides, "pipeline.output.path")
    resolved_override = _resolve_scene_value(args, output_path, scene, "pipeline.output.path")
    if resolved_override is not None:
        return resolved_override
    return args.work_dir / "vipe_outputs" / scene


def _prepare_vipe_overrides(args: argparse.Namespace, vipe_overrides: list[str], scene: str) -> tuple[list[str], Path, Path]:
    frame_dir = _resolve_frame_dir(args, vipe_overrides, scene)
    vipe_output_dir = _resolve_vipe_output_dir(args, vipe_overrides, scene)

    overrides = list(vipe_overrides)
    overrides = _set_override(overrides, "streams.base_path", frame_dir)
    if _override_value(overrides, "streams.fps") is None:
        overrides = _set_override(overrides, "streams.fps", args.fps)
    overrides = _set_override(overrides, "pipeline.output.path", vipe_output_dir)
    overrides = _set_override(overrides, "pipeline.output.save_artifacts", "true")
    return overrides, frame_dir, vipe_output_dir


def run_vipe(overrides: list[str]) -> None:
    import hydra

    from vipe import get_config_path
    from vipe.pipeline import VipePipeline
    from vipe.streams.base import FrameDir
    from vipe.utils.logging import configure_logging

    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        cfg = hydra.compose("default", overrides=overrides)

    logger = configure_logging()
    pipeline = VipePipeline(
        slam=cfg.pipeline.slam,
        depth=cfg.pipeline.depth,
        output=cfg.pipeline.output,
    )
    stream = FrameDir(
        path=cfg.streams.base_path,
        fps=cfg.streams.fps,
        frame_start=cfg.streams.frame_start,
        frame_end=cfg.streams.frame_end,
        frame_skip=cfg.streams.frame_skip,
        load_sensor_depth=cfg.pipeline.depth.use_gt_sens_depths is not None,
    )
    logger.info(f"Running ViPE on {stream.name()}")
    pipeline.run(stream)


def _subset_scene_data(scene_data, keep_indices: list[int]):
    from addict import Dict

    subset = Dict()
    subset.image_files = [scene_data.image_files[i] for i in keep_indices]
    subset.extrinsics = scene_data.extrinsics[keep_indices]
    subset.intrinsics = scene_data.intrinsics[keep_indices]
    subset.aux = Dict()
    for key, val in scene_data.aux.items():
        if isinstance(val, list) and len(val) == len(scene_data.image_files):
            subset.aux[key] = [val[i] for i in keep_indices]
        elif isinstance(val, np.ndarray) and len(val) == len(scene_data.image_files):
            subset.aux[key] = val[keep_indices]
        else:
            subset.aux[key] = val
    return subset


def _benchmark_frame_request(
    args: argparse.Namespace,
    evaluator,
    scene: str,
    frame_dir: Path,
    scene_overrides: list[str],
) -> tuple[Any, list[int], list[int]]:
    dataset = evaluator.datasets["scannet"]
    full_scene_data = dataset.get_data(scene)
    scene_data = evaluator._sample_frames(full_scene_data, scene)

    frame_files = _image_files(frame_dir)
    frame_index_by_path = {str(path.resolve()): idx for idx, path in enumerate(frame_files)}
    full_index_by_image = {str(Path(path).resolve()): idx for idx, path in enumerate(full_scene_data.image_files)}
    frame_start = int(_override_value(scene_overrides, "streams.frame_start") or 0)
    frame_end_override = int(_override_value(scene_overrides, "streams.frame_end") or -1)
    frame_end = len(frame_files) if frame_end_override == -1 else min(frame_end_override, len(frame_files))
    frame_skip = int(_override_value(scene_overrides, "streams.frame_skip") or 1)
    frame_indices = []
    kept_scene_indices = []
    missing = []

    for image_file in scene_data.image_files:
        image_key = str(Path(image_file).resolve())
        if image_key not in frame_index_by_path:
            missing.append(f"{image_file}: not found in ViPE frame dir")
            continue
        raw_frame_idx = frame_index_by_path[image_key]
        if raw_frame_idx < frame_start or raw_frame_idx >= frame_end or (raw_frame_idx - frame_start) % frame_skip != 0:
            missing.append(f"{image_file}: not included by ViPE frame_start/frame_end/frame_skip")
            continue
        frame_idx = (raw_frame_idx - frame_start) // frame_skip
        if image_key not in full_index_by_image:
            missing.append(f"{image_file}: not found in full ScanNet scene data")
            continue

        frame_indices.append(frame_idx)
        kept_scene_indices.append(full_index_by_image[image_key])

    if missing:
        raise ValueError("ViPE input frames do not cover the benchmark frame set:\n" + "\n".join(missing[:20]))
    return full_scene_data, frame_indices, kept_scene_indices


def _artifact_name(frame_dir: Path) -> str:
    return frame_dir.name


def _write_vipe_manifest(
    args: argparse.Namespace,
    evaluator,
    scene: str,
    full_scene_data,
    frame_indices: list[int],
    kept_scene_indices: list[int],
    vipe_output_dir: Path,
    artifact_name: str,
) -> None:
    print(f"[INFO] Writing ViPE benchmark manifest | {scene} | frames={len(frame_indices)}", flush=True)

    pose_path = vipe_output_dir / "pose" / f"{artifact_name}.npz"
    depth_path = vipe_output_dir / "depth" / f"{artifact_name}.zip"
    intrinsics_path = vipe_output_dir / "intrinsics" / f"{artifact_name}.json"
    missing_artifacts = [path for path in [pose_path, depth_path, intrinsics_path] if not path.exists()]
    if missing_artifacts:
        raise FileNotFoundError("Missing ViPE artifacts:\n" + "\n".join(str(path) for path in missing_artifacts))

    manifest = {
        "format": "vipe_artifacts_v1",
        "scene": scene,
        "artifact_name": artifact_name,
        "vipe_output_dir": str(vipe_output_dir.resolve()),
        "pose_path": str(pose_path.resolve()),
        "depth_path": str(depth_path.resolve()),
        "intrinsics_path": str(intrinsics_path.resolve()),
        "frame_indices": [int(idx) for idx in frame_indices],
    }

    exported_scene_data = _subset_scene_data(full_scene_data, kept_scene_indices)
    need_unposed = {"pose", "recon_unposed"} & set(args.modes)
    need_posed = {"recon_posed"} & set(args.modes)
    for posed in [False, True]:
        if posed and not need_posed:
            continue
        if not posed and not need_unposed:
            continue
        export_dir = Path(evaluator._export_dir("scannet", scene, posed=posed))
        exports_dir = export_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = exports_dir / "vipe_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote ViPE manifest | {scene} | posed={posed} | {manifest_path}", flush=True)
        print(f"[INFO] Writing GT metadata | {scene} | posed={posed}", flush=True)
        evaluator._save_gt_meta(str(export_dir), exported_scene_data)

    print(f"[INFO] Exported ViPE benchmark manifest for {scene} under {args.work_dir}")


def _load_evaluator(args: argparse.Namespace):
    da3_src = args.da3_root / "src"
    if da3_src.is_dir() and str(da3_src) not in sys.path:
        sys.path.insert(0, str(da3_src))

    os.environ["DA3_SCANNET_INPUT_ROOT"] = str(args.input_root.resolve())
    os.environ["DA3_SCANNET_RAW_ROOT"] = str(args.raw_root.resolve())

    from depth_anything_3.bench.evaluator import Evaluator

    return Evaluator(
        work_dir=str(args.work_dir),
        datas=["scannet"],
        modes=args.modes,
        scenes=args.scenes,
        num_fusion_workers=args.num_fusion_workers,
        max_frames=args.max_frames,
        ref_view_strategy="unused_by_vipe",
        gpu_id=args.gpu_id,
        total_gpus=args.total_gpus,
    )


def _append_common_cli_args(cmd: list[str], args: argparse.Namespace, vipe_overrides: list[str]) -> list[str]:
    cmd += ["--scenes", *args.scenes]
    cmd += ["--work-dir", str(args.work_dir)]
    cmd += ["--input-root", str(args.input_root)]
    cmd += ["--raw-root", str(args.raw_root)]
    cmd += ["--da3-root", str(args.da3_root)]
    cmd += ["--modes", *args.modes]
    cmd += ["--max-frames", str(args.max_frames)]
    cmd += ["--num-fusion-workers", str(args.num_fusion_workers)]
    cmd += ["--seed", str(args.seed)]
    cmd += ["--fps", str(args.fps)]
    if args.vipe_output_dir is not None:
        cmd += ["--vipe-output-dir", str(args.vipe_output_dir)]
    cmd += vipe_overrides
    return cmd


def maybe_spawn_workers(args: argparse.Namespace, vipe_overrides: list[str]) -> bool:
    if args.eval_only or args.print_only or os.environ.get(WORKER_ENV) == "1":
        return False

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_devices is not None and cuda_devices.strip():
        gpu_list = [gpu.strip() for gpu in cuda_devices.split(",") if gpu.strip()]
    else:
        import torch

        gpu_list = [str(i) for i in range(torch.cuda.device_count())]

    if len(gpu_list) <= 1:
        return False

    base_cmd = _append_common_cli_args([sys.executable, os.path.abspath(__file__)], args, vipe_overrides)

    print(f"[INFO] Detected {len(gpu_list)} GPUs for ViPE ScanNet benchmark: {gpu_list}")
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
    return True


def _scenes_for_worker(args: argparse.Namespace, evaluator) -> list[str]:
    dataset = evaluator.datasets["scannet"]
    known_scenes = set(evaluator._get_scenes(dataset))
    missing = sorted(set(args.scenes) - known_scenes)
    if missing:
        raise ValueError(f"Unknown ScanNet scenes: {missing}")

    all_scenes = list(args.scenes)
    if args.total_gpus <= 1:
        print(f"[INFO] Total ViPE benchmark scenes: {len(all_scenes)}")
        return all_scenes

    scenes = [scene for idx, scene in enumerate(all_scenes) if idx % args.total_gpus == args.gpu_id]
    print(f"[INFO] ViPE worker {args.gpu_id}/{args.total_gpus}: {len(scenes)}/{len(all_scenes)} scenes")
    return scenes


def prepare_vipe_benchmark_exports(
    args: argparse.Namespace,
    evaluator,
    vipe_overrides: list[str],
    run_vipe_first: bool,
) -> None:
    for scene in _scenes_for_worker(args, evaluator):
        scene_overrides, frame_dir, vipe_output_dir = _prepare_vipe_overrides(args, vipe_overrides, scene)
        full_scene_data, frame_indices, kept_scene_indices = _benchmark_frame_request(
            args, evaluator, scene, frame_dir, scene_overrides
        )
        if run_vipe_first:
            run_vipe(scene_overrides)
        _write_vipe_manifest(
            args,
            evaluator,
            scene,
            full_scene_data,
            frame_indices,
            kept_scene_indices,
            vipe_output_dir,
            _artifact_name(frame_dir),
        )


def _fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def _get_nested(d, *keys):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def print_scannet_summary(metrics) -> None:
    pose_mean = _get_nested(metrics, "scannet_pose", "mean") or {}
    recon_u_tsdf = _get_nested(metrics, "scannet_recon_unposed_tsdf", "mean") or {}
    recon_u_backproject = _get_nested(metrics, "scannet_recon_unposed_backproject", "mean") or {}
    recon_p_tsdf = _get_nested(metrics, "scannet_recon_posed_tsdf", "mean") or {}
    recon_p_backproject = _get_nested(metrics, "scannet_recon_posed_backproject", "mean") or {}

    auc3 = next((pose_mean[key] for key in ["Auc_3", "auc03", "auc_3", "auc3", "Auc3"] if key in pose_mean), None)
    auc30 = next((pose_mean[key] for key in ["Auc_30", "auc30", "auc_30", "Auc30"] if key in pose_mean), None)

    col1 = 15
    col2 = 14
    col3 = 14
    print("\n" + "=" * 44)
    print("SCANNET VIPE BENCHMARK SUMMARY")
    print("=" * 44)
    print("\nPOSE ESTIMATION")
    print("-" * (col1 + col2))
    print(f"{'Metric':<{col1}}{'ScanNet':<{col2}}")
    print("-" * (col1 + col2))
    print(f"{'Auc3':<{col1}}{_fmt(auc3):<{col2}}")
    print(f"{'Auc30':<{col1}}{_fmt(auc30):<{col2}}")
    print("\nRECON_UNPOSED (ViPE Pose)")
    print("-" * (col1 + col2 + col3))
    print(f"{'Metric':<{col1}}{'TSDF':<{col2}}{'Backproject':<{col3}}")
    print("-" * (col1 + col2 + col3))
    print(f"{'F-score':<{col1}}{_fmt(recon_u_tsdf.get('fscore')):<{col2}}{_fmt(recon_u_backproject.get('fscore')):<{col3}}")
    print(f"{'Overall':<{col1}}{_fmt(recon_u_tsdf.get('overall')):<{col2}}{_fmt(recon_u_backproject.get('overall')):<{col3}}")
    print("\nRECON_POSED (GT Pose)")
    print("-" * (col1 + col2 + col3))
    print(f"{'Metric':<{col1}}{'TSDF':<{col2}}{'Backproject':<{col3}}")
    print("-" * (col1 + col2 + col3))
    print(f"{'F-score':<{col1}}{_fmt(recon_p_tsdf.get('fscore')):<{col2}}{_fmt(recon_p_backproject.get('fscore')):<{col3}}")
    print(f"{'Overall':<{col1}}{_fmt(recon_p_tsdf.get('overall')):<{col2}}{_fmt(recon_p_backproject.get('overall')):<{col3}}")


def main() -> None:
    parser = build_parser()
    args, vipe_overrides = parser.parse_known_args()
    seed_everything(args.seed)
    is_worker = os.environ.get(WORKER_ENV) == "1"

    evaluator = _load_evaluator(args)

    if args.print_only:
        metrics = evaluator._load_metrics()
        print_scannet_summary(metrics)
        return

    if args.eval_only:
        prepare_vipe_benchmark_exports(args, evaluator, vipe_overrides, run_vipe_first=False)
        metrics = evaluator.eval()
        print_scannet_summary(metrics)
        return

    if maybe_spawn_workers(args, vipe_overrides):
        metrics = evaluator.eval()
        print_scannet_summary(metrics)
        return

    prepare_vipe_benchmark_exports(args, evaluator, vipe_overrides, run_vipe_first=True)
    if is_worker:
        return
    metrics = evaluator.eval()
    print_scannet_summary(metrics)


if __name__ == "__main__":
    main()
