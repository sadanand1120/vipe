#!/usr/bin/env python3

import argparse
import json
import sys

from pathlib import Path


SMOKE_TOLERANCES = {"auc30": 0.05, "auc03": 0.05, "overall": 0.01, "psnr": 1.0, "ssim": 0.05}
FULL_TOLERANCES = {"auc30": 0.02, "auc03": 0.02, "overall": 0.003, "psnr": 0.5, "ssim": 0.02}


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _scene_count(metrics: dict) -> int:
    return sum(key != "mean" for key in metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a complete ViPE ScanNet benchmark workspace.")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--write-success", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    reference = _load_json(args.reference)
    metric_dir = workspace / "metric_results"
    errors = []

    try:
        pose = _load_json(metric_dir / "scannet_pose.json")
        recon = _load_json(metric_dir / "scannet_recon.json")
        _load_json(metric_dir / "scannet_timing.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        pose, recon = {}, {}
        errors.append(f"missing or invalid metric file: {exc}")

    expected_scenes = 1 if args.mode == "smoke" else int(reference["scene_count"])
    counts = {
        "pose_metrics": _scene_count(pose),
        "recon_metrics": _scene_count(recon),
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

    skipped_path = metric_dir / "skipped_gt_pose_scenes.json"
    if skipped_path.exists():
        skipped = _load_json(skipped_path)
        if skipped:
            errors.append(f"unexpected skipped scenes: {sorted(skipped)}")

    metric_key = "scene0013_01" if args.mode == "smoke" else "mean"
    reference_key = "scene0013_01" if args.mode == "smoke" else "mean"
    actual = {"pose": pose.get(metric_key, {}), "recon": recon.get(metric_key, {})}
    expected = reference[reference_key]
    tolerances = SMOKE_TOLERANCES if args.mode == "smoke" else FULL_TOLERANCES
    deltas = {"pose": {}, "recon": {}}
    for section, names in (("pose", ("auc30", "auc03")), ("recon", ("overall", "psnr", "ssim"))):
        for name in names:
            if name not in actual[section]:
                errors.append(f"missing {section}.{name} for {metric_key}")
                continue
            delta = float(actual[section][name]) - float(expected[section][name])
            deltas[section][name] = delta
            if abs(delta) > tolerances[name]:
                errors.append(
                    f"{section}.{name}: actual={actual[section][name]:.8f}, "
                    f"reference={expected[section][name]:.8f}, delta={delta:+.8f}, tolerance={tolerances[name]}"
                )

    result = {
        "status": "passed" if not errors else "failed",
        "mode": args.mode,
        "workspace": str(workspace),
        "git_sha": reference["git_sha"],
        "expected_scenes": expected_scenes,
        "counts": counts,
        "actual": actual,
        "reference": expected,
        "deltas": deltas,
        "tolerances": tolerances,
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
