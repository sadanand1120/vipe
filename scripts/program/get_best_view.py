from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from orient_annotate import INSTANCE_VERTEX_DTYPE, read_binary_ply_header
from view_pcd import DEFAULT_PCD_DIR


DEFAULT_SCENE_DIR = Path("/robodata/smodak/repos/ovo/data/input/ScanNet/scene0000_00")
DEFAULT_RAW_ROOT = Path("/robodata/smodak/datasets/scannet_v2/scans")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find the best RGB image view for each pointcloud instance.")
    parser.add_argument("pcd_dir", nargs="?", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-output-dir", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--candidate-frames", type=int, default=300)
    parser.add_argument("--sample-points-per-instance", type=int, default=4096)
    parser.add_argument("--min-visible-points", type=int, default=25)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sorted_frame_files(path: Path) -> list[Path]:
    files = [p for p in path.iterdir() if p.is_file()]
    if all(p.stem.isdigit() for p in files):
        return sorted(files, key=lambda p: int(p.stem))
    return sorted(files, key=lambda p: p.name)


def parse_scannet_info(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("r") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().split()
            if len(value) == 1:
                try:
                    values[key] = float(value[0])
                except ValueError:
                    pass
    return values


def load_color_intrinsics(scene_dir: Path, raw_root: Path) -> tuple[np.ndarray, int, int]:
    scene = scene_dir.name
    info = parse_scannet_info(raw_root / scene / f"{scene}.txt")
    intrinsics = np.array(
        [
            [info["fx_color"], 0.0, info["mx_color"]],
            [0.0, info["fy_color"], info["my_color"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsics, int(info["colorWidth"]), int(info["colorHeight"])


def load_instance_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
        raise ValueError(f"Instance PLY ended early: expected {vertex_count}, got {len(vertices)}")

    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    return points, vertices["instance_id"].astype(np.int32, copy=False)


def load_semantic_labels(pcd_dir: Path, instance_ids: np.ndarray, point_instance_ids: np.ndarray) -> dict[int, dict[str, object]]:
    clip_path = pcd_dir / "clip.npz"
    if not clip_path.exists():
        return {}

    data = np.load(clip_path)
    semantic_ids = data["point_label_ids"].astype(np.int32, copy=False)
    label_texts = [str(x) for x in data["label_texts"].tolist()]
    if len(semantic_ids) != len(point_instance_ids):
        return {}

    labels = {}
    for instance_id in instance_ids:
        mask = point_instance_ids == int(instance_id)
        ids, counts = np.unique(semantic_ids[mask & (semantic_ids >= 0)], return_counts=True)
        if len(ids) == 0:
            continue
        best = int(ids[np.argmax(counts)])
        labels[int(instance_id)] = {"semantic_id": best, "semantic_label": label_texts[best]}
    return labels


def load_frames(scene_dir: Path, max_frames: int) -> list[dict[str, object]]:
    image_files = sorted_frame_files(scene_dir / "color")
    if max_frames > 0:
        image_files = image_files[:max_frames]

    frames = []
    for image_path in image_files:
        pose_path = scene_dir / "pose" / f"{image_path.stem}.txt"
        if not pose_path.exists():
            raise FileNotFoundError(f"Missing pose for {image_path}: {pose_path}")
        c2w = np.loadtxt(pose_path, dtype=np.float32)
        frames.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "pose_path": str(pose_path),
                "c2w": c2w,
                "w2c": np.linalg.inv(c2w).astype(np.float32),
            }
        )
    return frames


def build_instance_samples(
    points: np.ndarray,
    labels: np.ndarray,
    sample_points_per_instance: int,
    seed: int,
) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    instance_ids, point_counts = np.unique(labels[labels >= 0], return_counts=True)
    instance_ids = instance_ids.astype(np.int32)
    rng = np.random.default_rng(seed)
    samples: dict[int, dict[str, object]] = {}

    for instance_id, point_count in tqdm(
        zip(instance_ids, point_counts, strict=True),
        total=len(instance_ids),
        desc="Sampling instances",
        unit="inst",
    ):
        indices = np.flatnonzero(labels == int(instance_id))
        if len(indices) > sample_points_per_instance:
            indices = rng.choice(indices, size=sample_points_per_instance, replace=False)
        instance_points = points[indices].astype(np.float32, copy=False)
        centroid = instance_points.mean(axis=0)
        radius = float(np.linalg.norm(instance_points - centroid.reshape(1, 3), axis=1).max())
        samples[int(instance_id)] = {
            "points": instance_points,
            "point_count": int(point_count),
            "sample_count": int(len(instance_points)),
            "centroid": centroid.astype(np.float32),
            "radius": max(radius, 1e-4),
        }

    return instance_ids, samples


def project_points(points: np.ndarray, w2c: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    u = intrinsics[0, 0] * (cam[:, 0] / z) + intrinsics[0, 2]
    v = intrinsics[1, 1] * (cam[:, 1] / z) + intrinsics[1, 2]
    return u, v, z


def center_score(cx: float, cy: float, width: int, height: int) -> float:
    dx = (cx - width * 0.5) / max(width * 0.5, 1.0)
    dy = (cy - height * 0.5) / max(height * 0.5, 1.0)
    return float(max(0.0, 1.0 - np.sqrt(dx * dx + dy * dy) / np.sqrt(2.0)))


def approximate_candidates(
    frames: list[dict[str, object]],
    instance_ids: np.ndarray,
    samples: dict[int, dict[str, object]],
    intrinsics: np.ndarray,
    width: int,
    height: int,
    candidate_frames: int,
    min_depth: float,
) -> dict[int, list[int]]:
    centroids = np.stack([samples[int(instance_id)]["centroid"] for instance_id in instance_ids], axis=0)
    radii = np.asarray([samples[int(instance_id)]["radius"] for instance_id in instance_ids], dtype=np.float32)
    score_lists: dict[int, list[tuple[float, int]]] = {int(instance_id): [] for instance_id in instance_ids}

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    focal = max(fx, fy)
    image_area = float(width * height)
    for frame_idx, frame in enumerate(tqdm(frames, desc="Scanning camera poses", unit="frame")):
        w2c = frame["w2c"]
        cam = centroids @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        valid_depth = z > min_depth
        safe_z = np.maximum(z, min_depth)
        u = fx * (cam[:, 0] / safe_z) + float(intrinsics[0, 2])
        v = fy * (cam[:, 1] / safe_z) + float(intrinsics[1, 2])
        radius_px = focal * radii / safe_z
        in_frame = (u >= -radius_px) & (u < width + radius_px) & (v >= -radius_px) & (v < height + radius_px)
        valid = valid_depth & in_frame
        if not np.any(valid):
            continue

        area_frac = np.minimum(1.0, np.pi * radius_px * radius_px / image_area)
        for instance_pos in np.flatnonzero(valid):
            score = float(np.sqrt(area_frac[instance_pos]) * center_score(u[instance_pos], v[instance_pos], width, height))
            if score > 0.0:
                score_lists[int(instance_ids[instance_pos])].append((score, frame_idx))

    candidates = {}
    for instance_id, scored in score_lists.items():
        scored.sort(key=lambda item: item[0], reverse=True)
        candidates[instance_id] = [frame_idx for _, frame_idx in scored[:candidate_frames]]
    return candidates


def evaluate_view(
    points: np.ndarray,
    frame: dict[str, object],
    intrinsics: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
    min_visible_points: int,
) -> dict[str, object] | None:
    u, v, z = project_points(points, frame["w2c"], intrinsics)
    valid = (z > min_depth) & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    visible_count = int(valid.sum())
    if visible_count < min_visible_points:
        return None

    visible_u = u[valid]
    visible_v = v[valid]
    visible_z = z[valid]
    xmin = float(visible_u.min())
    ymin = float(visible_v.min())
    xmax = float(visible_u.max())
    ymax = float(visible_v.max())
    bbox_area = max(0.0, xmax - xmin + 1.0) * max(0.0, ymax - ymin + 1.0)
    bbox_area_fraction = float(bbox_area / float(width * height))
    cx = float((xmin + xmax) * 0.5)
    cy = float((ymin + ymax) * 0.5)
    visible_fraction = float(visible_count / len(points))
    centered = center_score(cx, cy, width, height)

    score = visible_fraction * (0.2 + np.sqrt(bbox_area_fraction)) * (0.75 + 0.25 * centered)
    return {
        "score": float(score),
        "visible_sample_count": visible_count,
        "visible_fraction": visible_fraction,
        "bbox_xyxy": [xmin, ymin, xmax, ymax],
        "bbox_area_pixels": float(bbox_area),
        "bbox_area_fraction": bbox_area_fraction,
        "bbox_center_xy": [cx, cy],
        "center_score": centered,
        "mean_depth_m": float(visible_z.mean()),
    }


def find_best_views(
    frames: list[dict[str, object]],
    instance_ids: np.ndarray,
    samples: dict[int, dict[str, object]],
    candidates: dict[int, list[int]],
    intrinsics: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
    min_visible_points: int,
) -> dict[int, dict[str, object]]:
    results = {}
    for instance_id in tqdm(instance_ids.tolist(), desc="Evaluating candidate views", unit="inst"):
        instance_id = int(instance_id)
        best = None
        best_frame = None
        points = samples[instance_id]["points"]
        frame_indices = candidates.get(instance_id, [])
        for pass_idx, indices in enumerate((frame_indices, range(len(frames)))):
            if pass_idx == 1 and best is not None:
                break
            if pass_idx == 1 and len(frame_indices) == len(frames):
                break
            for frame_idx in indices:
                metrics = evaluate_view(
                    points,
                    frames[frame_idx],
                    intrinsics,
                    width,
                    height,
                    min_depth,
                    min_visible_points,
                )
                if metrics is not None and (best is None or metrics["score"] > best["score"]):
                    best = metrics
                    best_frame = frames[frame_idx]

        if best is None:
            results[instance_id] = {
                "found": False,
                "reason": "no frame reached min_visible_points",
            }
            continue

        results[instance_id] = {
            "found": True,
            "image_id": best_frame["image_id"],
            "image_path": best_frame["image_path"],
            "pose_path": best_frame["pose_path"],
            **best,
        }
    return results


def apply_forced_best_views(
    best_views: dict[int, dict[str, object]],
    frames: list[dict[str, object]],
    samples: dict[int, dict[str, object]],
    intrinsics: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
    min_visible_points: int,
) -> None:
    forced_image_ids = {61: "616", 62: "647"}
    frames_by_id = {str(frame["image_id"]): frame for frame in frames}
    for instance_id, image_id in forced_image_ids.items():
        frame = frames_by_id[image_id]
        metrics = evaluate_view(
            samples[instance_id]["points"],
            frame,
            intrinsics,
            width,
            height,
            min_depth,
            min_visible_points,
        )
        if metrics is None:
            raise ValueError(f"Forced image {image_id} has too few visible points for instance {instance_id}")
        best_views[instance_id] = {
            "found": True,
            "image_id": frame["image_id"],
            "image_path": frame["image_path"],
            "pose_path": frame["pose_path"],
            **metrics,
        }


def dump_best_view_images(
    best_views: dict[int, dict[str, object]],
    samples: dict[int, dict[str, object]],
    frames: list[dict[str, object]],
    intrinsics: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
    output_dir: Path,
) -> None:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
        stale.unlink()
    for instance_id, best in tqdm(best_views.items(), desc="Copying best view images", unit="img"):
        if not best.get("found"):
            continue
        src = Path(str(best["image_path"]))
        dst = output_dir / f"{instance_id}.png"
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read best view image: {src}")

        overlay = bgr.copy()
        frame = next(frame for frame in frames if str(frame["image_path"]) == str(src))
        points = samples[int(instance_id)]["points"]
        u, v, z = project_points(points, frame["w2c"], intrinsics)
        valid = (z > min_depth) & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        if valid.any():
            uv = np.column_stack([u[valid], v[valid]]).round().astype(np.int32)
            for x, y in uv:
                cv2.circle(overlay, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            xmin, ymin, xmax, ymax = [int(round(x)) for x in best["bbox_xyxy"]]
            cv2.rectangle(
                overlay,
                (max(0, xmin), max(0, ymin)),
                (min(width - 1, xmax), min(height - 1, ymax)),
                color=(0, 0, 255),
                thickness=4,
                lineType=cv2.LINE_AA,
            )

        title_height = 58
        gutter = 18
        canvas = np.full((height + title_height, width * 2 + gutter, 3), 255, dtype=np.uint8)
        canvas[title_height:, :width] = bgr
        canvas[title_height:, width + gutter:] = overlay
        canvas[:, width : width + gutter] = np.array([255, 255, 255], dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        left_title = "RGB"
        right_title = "Projected points + bbox"
        left_size = cv2.getTextSize(left_title, font, 1.1, 2)[0]
        right_size = cv2.getTextSize(right_title, font, 1.1, 2)[0]
        cv2.putText(canvas, left_title, ((width - left_size[0]) // 2, 38), font, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            right_title,
            (width + gutter + (width - right_size[0]) // 2, 38),
            font,
            1.1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if not cv2.imwrite(str(dst), canvas):
            raise OSError(f"Could not write PNG: {dst}")
        best["best_view_image_path"] = str(dst)


def main() -> None:
    args = parse_args()
    pcd_dir = args.pcd_dir.expanduser().resolve()
    scene_dir = args.scene_dir.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else pcd_dir / "best_views.json"
    image_output_dir = (
        args.image_output_dir.expanduser().resolve() if args.image_output_dir else pcd_dir / "best_views"
    )

    intrinsics, width, height = load_color_intrinsics(scene_dir, raw_root)
    points, point_instance_ids = load_instance_ply(pcd_dir / "instance.ply")
    frames = load_frames(scene_dir, args.max_frames)
    instance_ids, samples = build_instance_samples(
        points,
        point_instance_ids,
        int(args.sample_points_per_instance),
        int(args.seed),
    )
    semantic_labels = load_semantic_labels(pcd_dir, instance_ids, point_instance_ids)
    candidates = approximate_candidates(
        frames,
        instance_ids,
        samples,
        intrinsics,
        width,
        height,
        int(args.candidate_frames),
        float(args.min_depth),
    )
    best_views = find_best_views(
        frames,
        instance_ids,
        samples,
        candidates,
        intrinsics,
        width,
        height,
        float(args.min_depth),
        int(args.min_visible_points),
    )
    apply_forced_best_views(
        best_views,
        frames,
        samples,
        intrinsics,
        width,
        height,
        float(args.min_depth),
        int(args.min_visible_points),
    )
    dump_best_view_images(
        best_views,
        samples,
        frames,
        intrinsics,
        width,
        height,
        float(args.min_depth),
        image_output_dir,
    )

    output = {
        "pcd_dir": str(pcd_dir),
        "scene_dir": str(scene_dir),
        "raw_root": str(raw_root),
        "image_size": {"width": width, "height": height},
        "intrinsics": intrinsics.tolist(),
        "best_view_image_dir": str(image_output_dir),
        "score_definition": (
            "visible_fraction * (0.2 + sqrt(bbox_area_fraction)) * "
            "(0.75 + 0.25 * center_score); candidates prefiltered by centroid/sphere projection"
        ),
        "params": {
            "max_frames": int(args.max_frames),
            "candidate_frames": int(args.candidate_frames),
            "sample_points_per_instance": int(args.sample_points_per_instance),
            "min_visible_points": int(args.min_visible_points),
            "min_depth": float(args.min_depth),
            "seed": int(args.seed),
        },
        "instances": {},
    }

    for instance_id in instance_ids.tolist():
        instance_id = int(instance_id)
        instance_info = {
            "instance_id": instance_id,
            "point_count": int(samples[instance_id]["point_count"]),
            "sample_count": int(samples[instance_id]["sample_count"]),
            "centroid_xyz": samples[instance_id]["centroid"].astype(float).tolist(),
            "radius_m": float(samples[instance_id]["radius"]),
            "candidate_count": len(candidates.get(instance_id, [])),
            "best_view": best_views[instance_id],
        }
        instance_info.update(semantic_labels.get(instance_id, {}))
        output["instances"][str(instance_id)] = instance_info

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"wrote: {output_path}")
    found = sum(1 for item in best_views.values() if item.get("found"))
    print(f"instances with best view: {found}/{len(best_views)}")


if __name__ == "__main__":
    main()
