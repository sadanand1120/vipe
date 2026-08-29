#!/usr/bin/env python3

import argparse
import json
import math
import sys

from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _scene_count(metrics: dict) -> int:
    return sum(key != "mean" for key in metrics)


def _expected_scene_count(workspace: Path, mode: str, pose: dict) -> int:
    queue_path = workspace / "metric_results" / "dynamic_scene_queue.json"
    if queue_path.is_file():
        return len(_load_json(queue_path)["scenes"])
    return 1 if mode == "smoke" else _scene_count(pose)


def _check_mean_metrics(errors: list[str], section: str, metrics: dict, names: tuple[str, ...]) -> None:
    mean = metrics.get("mean", {})
    for name in names:
        value = mean.get(name)
        if value is None or not math.isfinite(float(value)):
            errors.append(f"missing or non-finite {section}.mean.{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a complete ViPE ScanNet benchmark workspace.")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--write-success", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    metric_dir = workspace / "metric_results"
    errors = []

    try:
        pose = _load_json(metric_dir / "scannet_pose.json")
        recon = _load_json(metric_dir / "scannet_recon.json")
        timing = _load_json(metric_dir / "scannet_timing.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        pose, recon, timing = {}, {}, {}
        errors.append(f"missing or invalid metric file: {exc}")

    expected_scenes = _expected_scene_count(workspace, args.mode, pose)
    if expected_scenes < 1:
        errors.append("no benchmark scenes found")
    counts = {
        "pose_metrics": _scene_count(pose),
        "recon_metrics": _scene_count(recon),
        "timing_entries": len(timing.get("build", {}).get("scenes", {})),
        "pose_artifacts": len(list((workspace / "vipe_outputs").glob("*/pose/*.npz"))),
        "tsdf_artifacts": len(list((workspace / "vipe_outputs").glob("*/pcd/*_tsdf.ply"))),
    }
    for label, count in counts.items():
        if count != expected_scenes:
            errors.append(f"{label}: expected {expected_scenes}, found {count}")

    failed_path = metric_dir / "failed_scenes.json"
    if failed_path.exists():
        failed = _load_json(failed_path)
        if failed:
            errors.append(f"failed scenes: {sorted(failed)}")

    _check_mean_metrics(errors, "pose", pose, ("auc30", "auc03"))
    _check_mean_metrics(errors, "recon", recon, ("overall", "psnr", "ssim"))

    result = {
        "status": "passed" if not errors else "failed",
        "mode": args.mode,
        "workspace": str(workspace),
        "expected_scenes": expected_scenes,
        "counts": counts,
        "errors": errors,
    }
    validation_path = workspace / "validation.json"
    validation_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

    if errors:
        sys.exit(1)
    if args.write_success:
        (workspace / "_SUCCESS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
