from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vipe.bench.gt_pose_checks import filter_scenes_with_bad_gt_pose_data
from vipe.utils.config import load_yaml_config
from vipe.utils.data_format import scene_frame_count
from vipe.utils.determinism import seed_everything


WORKER_ENV = "_VIPE_INSTANCE_WORKER"
SCENE_QUEUE_ENV = "_VIPE_INSTANCE_SCENE_QUEUE"


@dataclass(frozen=True)
class InstanceBenchmarkSpec:
    dataset_key: str
    dataset_label: str
    script_path: Path
    pipeline_config_path: Path
    instance_config_path: Path
    eval_config_path: Path
    dataset_type: type
    evaluate_scene: Callable


def _parser(spec: InstanceBenchmarkSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Build and evaluate ViPE {spec.dataset_label} instance hypotheses.")
    parser.add_argument(
        "--scenes", nargs="*", default=None, help=f"{spec.dataset_label} scenes; defaults to extracted scenes"
    )
    parser.add_argument("--work-dir", required=True, type=Path, help="Benchmark workspace/output directory")
    parser.add_argument(
        "--input-root", required=True, type=Path, help=f"Canonical ViPE {spec.dataset_label} scene root"
    )
    parser.add_argument("--raw-root", required=True, type=Path, help=f"Raw {spec.dataset_label} GT root")
    parser.add_argument("--print-only", action="store_true", help="Only print saved metrics")
    parser.add_argument("--do-final-eval", action="store_true", help="Evaluate completed instance artifacts")
    parser.add_argument("--gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--total-gpus", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--scene-worker-scene", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scene-worker-output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _metric_dir(args: argparse.Namespace) -> Path:
    return args.work_dir / "metric_results"


def _output_dir(args: argparse.Namespace, scene: str) -> Path:
    return args.work_dir / "vipe_outputs" / scene


def _scene_dir(args: argparse.Namespace, scene: str) -> Path:
    return args.input_root / scene


def _instance_paths(output_dir: Path, scene: str) -> tuple[Path, ...]:
    return (
        output_dir / "pose" / f"{scene}.npz",
        output_dir / "pcd" / f"{scene}_tsdf.ply",
        output_dir / "instances" / f"{scene}.npz",
        output_dir / "instances" / f"{scene}_summary.json",
        output_dir / "pcd" / f"{scene}_instances.ply",
        output_dir / "pcd" / f"{scene}_semantic_pca_A_dense.ply",
        output_dir / "pcd" / f"{scene}_semantic_pca_C_hypavg.ply",
    )


def _manifest_path(spec: InstanceBenchmarkSpec, args: argparse.Namespace, scene: str) -> Path:
    return args.work_dir / "model_results" / spec.dataset_key / scene / "instance" / "manifest.json"


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _request_fingerprint(spec: InstanceBenchmarkSpec, args: argparse.Namespace, scene: str) -> dict[str, str | int]:
    scene_dir = _scene_dir(args, scene)
    return {
        "frame_count": scene_frame_count(scene_dir),
        "scene_contract_sha256": _sha256_files(
            [scene_dir / "metadata.json", scene_dir / "intrinsic" / "intrinsic_color.json"]
        ),
        "pipeline_config_sha256": _sha256_files([spec.pipeline_config_path, spec.instance_config_path]),
    }


def _scene_complete(spec: InstanceBenchmarkSpec, args: argparse.Namespace, scene: str) -> bool:
    manifest_path = _manifest_path(spec, args, scene)
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("request") == _request_fingerprint(spec, args, scene) and all(
        path.is_file() for path in _instance_paths(_output_dir(args, scene), scene)
    )


def _write_manifest(spec: InstanceBenchmarkSpec, args: argparse.Namespace, scene: str) -> None:
    paths = _instance_paths(_output_dir(args, scene), scene)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ViPE instance artifacts:\n" + "\n".join(missing))
    names = (
        "pose",
        "tsdf",
        "instances",
        "instance_summary",
        "instance_ply",
        "semantic_A_dense_ply",
        "semantic_C_hypavg_ply",
    )
    _write_json(
        _manifest_path(spec, args, scene),
        {
            "format": "vipe_instance_manifest_v1",
            "dataset": spec.dataset_key,
            "scene": scene,
            "request": _request_fingerprint(spec, args, scene),
            "artifacts": {name: str(path.resolve()) for name, path in zip(names, paths)},
        },
    )


def _run_vipe(scene_dir: Path, output_dir: Path, pipeline_cfg, instance_cfg) -> None:
    from vipe.pipeline import VipePipeline
    from vipe.stream import FrameDir
    from vipe.utils.logging import configure_logging

    seed_everything(pipeline_cfg.seed, temporary_determinism=pipeline_cfg.temporary_determinism)
    logger = configure_logging()
    stream = FrameDir(scene_dir)
    logger.info("Running ViPE plus instance distillation on %s", stream.name)
    VipePipeline(
        slam=pipeline_cfg.pipeline.slam,
        output=pipeline_cfg.pipeline.output,
        instance=instance_cfg.pipeline.instance,
        output_dir=output_dir,
    ).run(stream)


def _visible_gpus() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip():
        return [value.strip() for value in visible.split(",") if value.strip()]
    import torch

    return [str(index) for index in range(torch.cuda.device_count())]


def _gpu_memory_used_mb() -> float:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if "," in visible:
        raise ValueError(f"Scene worker requires one visible GPU, got {visible}")
    result = subprocess.run(
        ["nvidia-smi", "--id", visible, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _run_scene_subprocess(spec: InstanceBenchmarkSpec, args: argparse.Namespace, scene: str) -> tuple[float, float]:
    output_dir = _output_dir(args, scene)
    cmd = [
        sys.executable,
        str(spec.script_path.resolve()),
        "--work-dir",
        str(args.work_dir),
        "--input-root",
        str(args.input_root),
        "--raw-root",
        str(args.raw_root),
        "--scene-worker-scene",
        scene,
        "--scene-worker-output-dir",
        str(output_dir),
    ]
    start = time.perf_counter()
    process = subprocess.Popen(cmd, env=os.environ.copy())
    peak_vram = _gpu_memory_used_mb()
    while process.poll() is None:
        peak_vram = max(peak_vram, _gpu_memory_used_mb())
        time.sleep(0.2)
    if process.returncode:
        raise RuntimeError(f"scene subprocess exited with code {process.returncode}")
    return time.perf_counter() - start, peak_vram


def _cleanup_runtime() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _queue_path(args: argparse.Namespace) -> Path:
    return _metric_dir(args) / "instance_scene_queue.json"


def _claim_scene(path: Path) -> tuple[int, int, str | None]:
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.load(handle)
        index = int(state["next_index"])
        scenes = state["scenes"]
        scene = None if index >= len(scenes) else scenes[index]
        if scene is not None:
            state["next_index"] = index + 1
            handle.seek(0)
            json.dump(state, handle, indent=2)
            handle.write("\n")
            handle.truncate()
        fcntl.flock(handle, fcntl.LOCK_UN)
    return index, len(scenes), scene


def _worker_scenes(args: argparse.Namespace) -> Iterable[str]:
    queue = os.environ.get(SCENE_QUEUE_ENV)
    if not queue:
        return list(args.scenes)

    def claim_iterator():
        while True:
            index, total, scene = _claim_scene(Path(queue))
            if scene is None:
                return
            print(f"[INFO] Instance worker {args.gpu_id}: claimed {index + 1}/{total} {scene}", flush=True)
            yield scene

    return claim_iterator()


def _worker_output_path(args: argparse.Namespace) -> Path:
    return _metric_dir(args) / "instance_workers" / f"worker_{args.gpu_id}.json"


def _timing_entry(frames: int, seconds: float, peak_vram_mb: float | None = None) -> dict:
    entry = {"frames": int(frames), "seconds": float(seconds), "fps": float(frames / seconds) if seconds else 0.0}
    if peak_vram_mb is not None:
        entry["peak_vram_mb"] = float(peak_vram_mb)
    return entry


def _evaluate(spec: InstanceBenchmarkSpec, args: argparse.Namespace, scene: str, eval_cfg, text_encoder) -> dict:
    cache_dir = args.work_dir / "model_results" / spec.dataset_key / scene / "instance" / "eval_cache"
    return spec.evaluate_scene(
        scene=scene,
        scene_dir=_scene_dir(args, scene),
        raw_root=args.raw_root,
        vipe_output_dir=_output_dir(args, scene),
        cache_dir=cache_dir,
        text_encoder=text_encoder,
        config=eval_cfg,
    )


def _run_worker(spec: InstanceBenchmarkSpec, args: argparse.Namespace, eval_cfg, instance_cfg) -> dict:
    payload = {"build": {}, "metric_eval": {}, "metrics": {}, "failures": {}}
    completed = []
    for scene in _worker_scenes(args):
        try:
            frames = scene_frame_count(_scene_dir(args, scene))
            if not _scene_complete(spec, args, scene):
                shutil.rmtree(_output_dir(args, scene), ignore_errors=True)
                shutil.rmtree(_manifest_path(spec, args, scene).parent, ignore_errors=True)
                seconds, peak_vram = _run_scene_subprocess(spec, args, scene)
                _write_manifest(spec, args, scene)
                payload["build"][scene] = _timing_entry(frames, seconds, peak_vram)
            else:
                print(f"[INFO] Reusing complete instance artifacts | {scene}", flush=True)
            completed.append(scene)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            payload["failures"][scene] = repr(exc)
            print(f"[WARN] Skipping failed {spec.dataset_label} instance scene | {scene} | {exc}", flush=True)
        finally:
            _write_json(_worker_output_path(args), payload)
            _cleanup_runtime()

    if args.do_final_eval and completed:
        from vipe.instance.semantic import FGCLIPBackbone

        features = instance_cfg.pipeline.instance.features
        text_encoder = FGCLIPBackbone(
            model_path=features.model_path,
            revision=features.revision,
            device="cuda",
        )
        for scene in completed:
            try:
                frames = scene_frame_count(_scene_dir(args, scene))
                start = time.perf_counter()
                payload["metrics"][scene] = _evaluate(
                    spec, args, scene, eval_cfg, text_encoder
                )
                payload["metric_eval"][scene] = _timing_entry(frames, time.perf_counter() - start)
                print(f"[INFO] Instance eval done | {scene} | {payload['metrics'][scene]}", flush=True)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                payload["failures"][scene] = repr(exc)
                print(f"[WARN] Skipping failed {spec.dataset_label} instance eval | {scene} | {exc}", flush=True)
            finally:
                _write_json(_worker_output_path(args), payload)
    return payload


def _spawn_workers(spec: InstanceBenchmarkSpec, args: argparse.Namespace) -> list[dict] | None:
    gpus = _visible_gpus()
    if os.environ.get(WORKER_ENV) == "1" or len(gpus) <= 1:
        return None
    shutil.rmtree(_metric_dir(args) / "instance_workers", ignore_errors=True)
    _write_json(_queue_path(args), {"next_index": 0, "scenes": list(args.scenes)})
    base = [
        sys.executable,
        str(spec.script_path.resolve()),
        "--scenes",
        *args.scenes,
        "--work-dir",
        str(args.work_dir),
        "--input-root",
        str(args.input_root),
        "--raw-root",
        str(args.raw_root),
    ]
    if args.do_final_eval:
        base.append("--do-final-eval")
    processes = []
    for worker_id, gpu in enumerate(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env[WORKER_ENV] = "1"
        env[SCENE_QUEUE_ENV] = str(_queue_path(args))
        print(f"[INFO] Starting {spec.dataset_label} instance worker {worker_id} on GPU {gpu}", flush=True)
        processes.append((worker_id, subprocess.Popen(base + ["--gpu-id", str(worker_id)], env=env)))
    failed = []
    for worker_id, process in processes:
        process.wait()
        if process.returncode:
            failed.append((worker_id, process.returncode))
    if failed:
        print(f"[WARN] {spec.dataset_label} instance worker failures: {failed}", flush=True)
    payloads = []
    for worker_id in range(len(gpus)):
        path = _metric_dir(args) / "instance_workers" / f"worker_{worker_id}.json"
        if path.is_file():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def _merge_payloads(payloads: list[dict]) -> dict:
    merged = {"build": {}, "metric_eval": {}, "metrics": {}, "failures": {}}
    for payload in payloads:
        for key in merged:
            merged[key].update(payload.get(key, {}))
    return merged


def _mean_metrics(scene_metrics: dict[str, dict]) -> dict[str, float]:
    keys = ("ar", "r50", "r75", "r90", "n_gt", "n_hyps", "mean_memb", "max_memb")
    mean = {
        key: float(np.mean([float(metrics[key]) for metrics in scene_metrics.values()]))
        for key in keys
        if scene_metrics and all(key in metrics for metrics in scene_metrics.values())
    }
    semantic = [
        metrics
        for metrics in scene_metrics.values()
        if metrics.get("semantic_top1") is not None and int(metrics.get("semantic_evaluated_points", 0)) > 0
    ]
    if semantic:
        total = sum(int(metrics["semantic_evaluated_points"]) for metrics in semantic)
        mean["semantic_top1"] = float(
            sum(float(metrics["semantic_top1"]) * int(metrics["semantic_evaluated_points"]) for metrics in semantic)
            / total
        )
        mean["semantic_evaluated_points"] = float(total)
    if scene_metrics and all("semantic_field_coverage" in metrics for metrics in scene_metrics.values()):
        mean["semantic_field_coverage"] = float(
            np.mean([float(metrics["semantic_field_coverage"]) for metrics in scene_metrics.values()])
        )
    return mean


def _write_results(spec: InstanceBenchmarkSpec, args: argparse.Namespace, payload: dict, eval_cfg) -> dict:
    metrics = dict(payload["metrics"])
    if metrics:
        metrics["mean"] = _mean_metrics(metrics)
    if args.do_final_eval:
        _write_json(_metric_dir(args) / str(eval_cfg.outputs.metrics_filename), metrics)
    timing_path = _metric_dir(args) / str(eval_cfg.outputs.timing_filename)
    previous = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {}
    build = dict(previous.get("build", {}).get("scenes", {}))
    metric = dict(previous.get("metric_eval", {}).get("scenes", {}))
    build.update(payload["build"])
    metric.update(payload["metric_eval"])
    timing = {
        "build": {
            "mean_fps": float(np.mean([entry["fps"] for entry in build.values()])) if build else 0.0,
            "mean_peak_vram_mb": float(np.mean([entry["peak_vram_mb"] for entry in build.values()])) if build else 0.0,
            "scenes": build,
        },
        "metric_eval": {
            "mean_fps": float(np.mean([entry["fps"] for entry in metric.values()])) if metric else 0.0,
            "scenes": metric,
        },
    }
    _write_json(timing_path, timing)
    if payload["failures"]:
        _write_json(_metric_dir(args) / f"{spec.dataset_key}_instance_failed_scenes.json", payload["failures"])
    return {"metrics": metrics, "timing": timing}


def _print_summary(spec: InstanceBenchmarkSpec, result: dict) -> None:
    mean = result.get("metrics", {}).get("mean", {})
    timing = result.get("timing", {})
    print("\n" + "=" * 54)
    print(f"{spec.dataset_label.upper()} INSTANCE DISTILLATION SUMMARY")
    print("=" * 54)
    print(f"{'AR':<22}{mean.get('ar', float('nan')):.4f}")
    print(f"{'R@0.50':<22}{mean.get('r50', float('nan')):.4f}")
    print(f"{'R@0.75':<22}{mean.get('r75', float('nan')):.4f}")
    print(f"{'R@0.90':<22}{mean.get('r90', float('nan')):.4f}")
    print(f"{'Semantic top-1':<22}{mean.get('semantic_top1', float('nan')):.4f}")
    print(f"{'Semantic coverage':<22}{mean.get('semantic_field_coverage', float('nan')):.4f}")
    print(f"{'Mean hypotheses':<22}{mean.get('n_hyps', float('nan')):.1f}")
    print(f"{'Build FPS':<22}{timing.get('build', {}).get('mean_fps', 0.0):.2f}")
    print(f"{'Peak VRAM (MB)':<22}{timing.get('build', {}).get('mean_peak_vram_mb', 0.0):.1f}")


def run_instance_benchmark(spec: InstanceBenchmarkSpec) -> None:
    args = _parser(spec).parse_args()
    pipeline_cfg = load_yaml_config(spec.pipeline_config_path)
    instance_cfg = load_yaml_config(spec.instance_config_path)
    eval_cfg = load_yaml_config(spec.eval_config_path)
    seed_everything(int(eval_cfg.seed), temporary_determinism=bool(eval_cfg.temporary_determinism))

    if args.scene_worker_scene is not None:
        if args.scene_worker_output_dir is None:
            raise ValueError("--scene-worker-output-dir is required with --scene-worker-scene")
        _run_vipe(_scene_dir(args, args.scene_worker_scene), args.scene_worker_output_dir, pipeline_cfg, instance_cfg)
        return

    dataset = spec.dataset_type(args.input_root, args.raw_root, eval_cfg)
    if args.scenes is None:
        args.scenes = list(dataset.SCENES)
    unknown = sorted(set(args.scenes) - set(dataset.SCENES))
    if unknown:
        raise ValueError(f"Unknown {spec.dataset_label} scenes: {unknown}")

    if args.print_only:
        metrics_path = _metric_dir(args) / str(eval_cfg.outputs.metrics_filename)
        timing_path = _metric_dir(args) / str(eval_cfg.outputs.timing_filename)
        _print_summary(
            spec,
            {
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {},
                "timing": json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {},
            },
        )
        return

    if os.environ.get(WORKER_ENV) != "1":
        (_metric_dir(args) / f"{spec.dataset_key}_instance_failed_scenes.json").unlink(missing_ok=True)
        if not args.do_final_eval:
            (_metric_dir(args) / str(eval_cfg.outputs.metrics_filename)).unlink(missing_ok=True)
        kept, skipped = filter_scenes_with_bad_gt_pose_data(dataset, list(args.scenes))
        args.scenes = kept
        if skipped:
            _write_json(
                _metric_dir(args) / f"{spec.dataset_key}_instance_skipped_gt_pose_scenes.json",
                {scene: report.summary() for scene, report in skipped},
            )

    spawned = _spawn_workers(spec, args)
    if spawned is None:
        payload = _run_worker(spec, args, eval_cfg, instance_cfg)
        if os.environ.get(WORKER_ENV) == "1":
            return
    else:
        payload = _merge_payloads(spawned)
    result = _write_results(spec, args, payload, eval_cfg)
    _print_summary(spec, result)
