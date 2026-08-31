from __future__ import annotations

import hashlib
import json

from pathlib import Path

import numpy as np

from vipe.bench.instance import (
    apply_exclusions,
    apply_se3,
    build_gt_occupancy_cloud,
    evaluate_prediction,
    gt_matching_hypotheses,
    kabsch_se3,
    load_instance_prediction,
    majority_nearest_labels,
    recall_ar,
)
from vipe.bench.replica import full_replica_scene_candidates
from vipe.utils.data_format import frame_stem, scene_frame_count


def semantic_mesh_path(raw_root: str | Path, scene: str) -> Path:
    checked = []
    for candidate in full_replica_scene_candidates(scene):
        path = Path(raw_root) / candidate / "habitat" / "mesh_semantic.ply"
        checked.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing Replica semantic mesh for {scene}. Checked: {checked}")


def label_points_from_semantic_mesh(
    mesh_path: str | Path,
    points: np.ndarray,
    *,
    neighbors: int,
    max_distance_m: float,
) -> np.ndarray:
    from plyfile import PlyData

    ply = PlyData.read(str(mesh_path))
    vertices = ply["vertex"]
    xyz = np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=1).astype(np.float32)
    faces = ply["face"]
    vertex_indices = faces["vertex_indices"]
    object_ids = np.asarray(faces["object_id"], dtype=np.int64)
    lengths = np.fromiter((len(face) for face in vertex_indices), dtype=np.int32, count=len(vertex_indices))
    if np.all(lengths == 3):
        triangles = np.stack(vertex_indices).astype(np.int64)
        triangle_objects = object_ids
    elif np.all(lengths == 4):
        quads = np.stack(vertex_indices).astype(np.int64)
        triangles = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]), axis=0)
        triangle_objects = np.concatenate((object_ids, object_ids))
    else:
        triangles_list: list[tuple[int, int, int]] = []
        objects_list: list[int] = []
        for face, object_id in zip(vertex_indices, object_ids):
            face = np.asarray(face, dtype=np.int64)
            for idx in range(1, len(face) - 1):
                triangles_list.append((int(face[0]), int(face[idx]), int(face[idx + 1])))
                objects_list.append(int(object_id))
        triangles = np.asarray(triangles_list, dtype=np.int64)
        triangle_objects = np.asarray(objects_list, dtype=np.int64)

    labels = majority_nearest_labels(
        xyz[triangles].mean(axis=1),
        triangle_objects,
        points,
        neighbors=neighbors,
        max_distance_m=max_distance_m,
    )
    labels[labels == 0] = -1
    return labels


def _cache_fingerprint(scene_dir: Path, mesh_path: Path, config: dict) -> str:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    files = [scene_dir / "metadata.json", scene_dir / "intrinsic" / "intrinsic_color.json", mesh_path]
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
    mesh_path = semantic_mesh_path(raw_root, scene)
    cache_config = {
        "voxel_m": float(config.cloud.voxel_m),
        "depth_min_m": float(config.cloud.depth_min_m),
        "depth_max_m": float(config.cloud.depth_max_m),
        "mesh_neighbors": int(config.labels.mesh_neighbors),
        "mesh_max_distance_m": float(config.labels.mesh_max_distance_m),
    }
    fingerprint = _cache_fingerprint(scene_dir, mesh_path, cache_config)
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
    labels = label_points_from_semantic_mesh(
        mesh_path,
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
    config,
) -> dict[str, float | int | list[float]]:
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
    return evaluate_prediction(
        scene=scene,
        scene_dir=scene_dir,
        vipe_output_dir=vipe_output_dir,
        gt_points=gt_points,
        gt_labels=gt_labels,
        excluded_ids=[int(value) for value in config.exclusions.get(scene, [])],
        config=config,
    )
