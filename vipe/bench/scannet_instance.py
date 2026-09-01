from __future__ import annotations

import hashlib
import json

from pathlib import Path

import numpy as np

from vipe.bench.instance import build_gt_occupancy_cloud, evaluate_prediction, majority_nearest_labels
from vipe.utils.data_format import frame_stem, scene_frame_count


def annotation_paths(raw_root: str | Path, scene: str) -> tuple[Path, Path, Path]:
    scene_dir = Path(raw_root) / scene
    mesh = scene_dir / f"{scene}_vh_clean_2.ply"
    segments = scene_dir / f"{scene}_vh_clean_2.0.010000.segs.json"
    aggregation = scene_dir / f"{scene}.aggregation.json"
    missing = [str(path) for path in (mesh, segments, aggregation) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ScanNet instance annotations:\n" + "\n".join(missing))
    return mesh, segments, aggregation


def load_annotated_mesh(raw_root: str | Path, scene: str) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData

    mesh_path, segments_path, aggregation_path = annotation_paths(raw_root, scene)
    vertices = PlyData.read(str(mesh_path))["vertex"]
    points = np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=1).astype(np.float32)
    segment_ids = np.asarray(json.loads(segments_path.read_text(encoding="utf-8"))["segIndices"], dtype=np.int64)
    groups = json.loads(aggregation_path.read_text(encoding="utf-8"))["segGroups"]
    if len(segment_ids) != len(points):
        raise ValueError(f"{scene}: {len(segment_ids)} segment IDs for {len(points)} mesh vertices")

    segment_to_instance: dict[int, int] = {}
    for group in groups:
        instance_id = int(group["objectId"])
        for segment_id in group["segments"]:
            segment_id = int(segment_id)
            previous = segment_to_instance.setdefault(segment_id, instance_id)
            if previous != instance_id:
                raise ValueError(f"{scene}: segment {segment_id} belongs to multiple instances")
    labels = np.fromiter(
        (segment_to_instance.get(int(segment_id), -1) for segment_id in segment_ids),
        dtype=np.int32,
        count=len(segment_ids),
    )
    return points, labels


def load_semantic_classes(raw_root: str | Path, scene: str) -> tuple[dict[int, int], dict[int, str]]:
    aggregation_path = annotation_paths(raw_root, scene)[2]
    groups = json.loads(aggregation_path.read_text(encoding="utf-8"))["segGroups"]
    object_names: dict[int, str] = {}
    for group in groups:
        name = " ".join(str(group.get("label", "")).strip().lower().split())
        if not name:
            continue
        object_id = int(group["objectId"])
        previous = object_names.setdefault(object_id, name)
        if previous != name:
            raise ValueError(f"{scene}: ScanNet object {object_id} has conflicting labels {previous!r} and {name!r}")
    names = sorted(set(object_names.values()))
    name_to_class = {name: class_id for class_id, name in enumerate(names)}
    return {object_id: name_to_class[name] for object_id, name in object_names.items()}, dict(enumerate(names))


def _cache_fingerprint(scene_dir: Path, annotation_files: tuple[Path, ...], config: dict) -> str:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    files = [scene_dir / "metadata.json", scene_dir / "intrinsic" / "intrinsic_color.json", *annotation_files]
    files.extend(sorted((scene_dir / "pose").glob("*.txt")))
    files.extend(sorted((scene_dir / "depth").glob("*.png")))
    for path in files:
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def load_or_build_gt_cache(
    scene: str,
    scene_dir: str | Path,
    raw_root: str | Path,
    cache_path: str | Path,
    gt_c2w: np.ndarray,
    config,
) -> tuple[np.ndarray, np.ndarray]:
    scene_dir = Path(scene_dir)
    cache_path = Path(cache_path)
    annotations = annotation_paths(raw_root, scene)
    cache_config = {
        "voxel_m": float(config.cloud.voxel_m),
        "depth_min_m": float(config.cloud.depth_min_m),
        "depth_max_m": float(config.cloud.depth_max_m),
        "mesh_neighbors": int(config.labels.mesh_neighbors),
        "mesh_max_distance_m": float(config.labels.mesh_max_distance_m),
    }
    fingerprint = _cache_fingerprint(scene_dir, annotations, cache_config)
    if cache_path.is_file():
        with np.load(cache_path) as cached:
            if str(np.asarray(cached["fingerprint"]).item()) == fingerprint:
                return cached["points"].astype(np.float32), cached["labels"].astype(np.int32)

    points = build_gt_occupancy_cloud(
        scene_dir,
        gt_c2w,
        voxel_m=cache_config["voxel_m"],
        depth_min_m=cache_config["depth_min_m"],
        depth_max_m=cache_config["depth_max_m"],
    )
    mesh_points, mesh_labels = load_annotated_mesh(raw_root, scene)
    labels = majority_nearest_labels(
        mesh_points,
        mesh_labels,
        points,
        neighbors=cache_config["mesh_neighbors"],
        max_distance_m=cache_config["mesh_max_distance_m"],
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, points=points, labels=labels, fingerprint=np.asarray(fingerprint))
    return points, labels


def evaluate_scene(
    *,
    scene: str,
    scene_dir: str | Path,
    raw_root: str | Path,
    vipe_output_dir: str | Path,
    cache_dir: str | Path,
    feature_config,
    text_encoder,
    config,
) -> dict[str, object]:
    scene_dir = Path(scene_dir)
    gt_c2w = np.stack(
        [
            np.loadtxt(scene_dir / "pose" / f"{frame_stem(frame_idx)}.txt", dtype=np.float64)
            for frame_idx in range(scene_frame_count(scene_dir))
        ]
    )
    gt_points, gt_labels = load_or_build_gt_cache(
        scene,
        scene_dir,
        raw_root,
        Path(cache_dir) / str(config.outputs.gt_cache_filename),
        gt_c2w,
        config,
    )
    object_to_class, class_names = load_semantic_classes(raw_root, scene)
    return evaluate_prediction(
        scene=scene,
        scene_dir=scene_dir,
        vipe_output_dir=vipe_output_dir,
        gt_points=gt_points,
        gt_labels=gt_labels,
        excluded_ids=(),
        object_to_class=object_to_class,
        class_names=class_names,
        feature_config=feature_config,
        text_encoder=text_encoder,
        config=config,
    )
