from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from vipe.instance.pipeline import write_instance_ply, write_labeled_instance_ply
from vipe.utils.data_format import frame_stem, intrinsic_matrix, read_pinhole_intrinsics, scene_frame_count


@dataclass(frozen=True)
class InstancePrediction:
    points: np.ndarray
    hypotheses: tuple[np.ndarray, ...]
    membership_budget: int


def load_instance_prediction(path: str | Path) -> InstancePrediction:
    with np.load(path) as data:
        required = {"points", "hypothesis_indices", "hypothesis_offsets", "K"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Missing instance artifact arrays in {path}: {sorted(missing)}")
        points = np.ascontiguousarray(data["points"], dtype=np.float32)
        indices = np.asarray(data["hypothesis_indices"], dtype=np.int64)
        offsets = np.asarray(data["hypothesis_offsets"], dtype=np.int64)
        budget = int(np.asarray(data["K"]).item())

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Instance points must have shape (N,3), got {points.shape}")
    if indices.ndim != 1 or offsets.ndim != 1 or len(offsets) == 0:
        raise ValueError("Packed hypotheses require one-dimensional indices and non-empty offsets")
    if offsets[0] != 0 or offsets[-1] != len(indices) or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("Invalid packed hypothesis offsets")
    if budget <= 0:
        raise ValueError(f"Invalid membership budget: {budget}")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(points)):
        raise ValueError("Hypothesis index is outside the instance surface")

    hypotheses = tuple(indices[start:end] for start, end in zip(offsets[:-1], offsets[1:]))
    membership = np.zeros(len(points), dtype=np.uint16)
    for hypothesis in hypotheses:
        np.add.at(membership, hypothesis, 1)
    if len(membership) and int(membership.max()) > budget:
        raise ValueError(f"Instance artifact exceeds K={budget}: max membership={int(membership.max())}")
    return InstancePrediction(points, hypotheses, budget)


def kabsch_se3(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError(f"Kabsch inputs must be matching (N,3), N>=3; got {source.shape} and {target.shape}")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def apply_se3(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


def build_gt_occupancy_cloud(
    scene_dir: str | Path,
    gt_c2w: np.ndarray,
    *,
    voxel_m: float,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    scene_dir = Path(scene_dir)
    count = scene_frame_count(scene_dir)
    if len(gt_c2w) != count:
        raise ValueError(f"GT trajectory has {len(gt_c2w)} poses for {count} canonical frames")
    intrinsics = intrinsic_matrix(read_pinhole_intrinsics(scene_dir / "intrinsic" / "intrinsic_color.json"))
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    frame_voxels: list[np.ndarray] = []
    for frame_idx, c2w in enumerate(gt_c2w):
        if not np.isfinite(c2w).all():
            continue
        depth_path = scene_dir / "depth" / f"{frame_stem(frame_idx)}.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Missing canonical depth frame: {depth_path}")
        depth = depth.astype(np.float32) / 1000.0
        ys, xs = np.where(np.isfinite(depth) & (depth > depth_min_m) & (depth < depth_max_m))
        if not len(xs):
            continue
        values = depth[ys, xs]
        camera = np.stack(((xs - cx) / fx * values, (ys - cy) / fy * values, values), axis=1).astype(np.float64)
        world = camera @ c2w[:3, :3].T + c2w[:3, 3]
        frame_voxels.append(np.unique(np.floor(world / voxel_m).astype(np.int64), axis=0))
    if not frame_voxels:
        raise ValueError(f"No valid GT depth points in {scene_dir}")
    voxels = np.unique(np.concatenate(frame_voxels, axis=0), axis=0)
    return ((voxels + 0.5) * voxel_m).astype(np.float32)


def majority_nearest_labels(
    reference_points: np.ndarray,
    reference_labels: np.ndarray,
    query_points: np.ndarray,
    *,
    neighbors: int,
    max_distance_m: float,
) -> np.ndarray:
    distances, indices = cKDTree(np.asarray(reference_points, dtype=np.float64)).query(
        np.asarray(query_points, dtype=np.float64), k=neighbors, workers=-1
    )
    if neighbors == 1:
        distances, indices = distances[:, None], indices[:, None]
    labels = np.asarray(reference_labels, dtype=np.int64)[indices]
    labels[distances > max_distance_m] = -1
    valid = labels >= 0
    equal = (labels[:, :, None] == labels[:, None, :]) & valid[:, None, :]
    votes = np.where(valid, equal.sum(axis=2), -1)
    winners = votes.argmax(axis=1)
    output = np.where(valid.any(axis=1), labels[np.arange(len(query_points)), winners], -1)
    return output.astype(np.int32)


def apply_exclusions(labels: np.ndarray, excluded_ids: list[int] | tuple[int, ...]) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if not excluded_ids:
        return labels.copy()
    return np.where(np.isin(labels, np.asarray(excluded_ids, dtype=np.int64)), -1, labels)


def recall_ar(
    labels: np.ndarray,
    hypotheses: tuple[np.ndarray, ...] | list[np.ndarray],
    thresholds: np.ndarray,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    gt_ids = np.unique(labels[labels >= 0])
    if not len(gt_ids):
        raise ValueError("No valid GT instances remain after exclusions")
    gt_sizes = np.bincount(np.searchsorted(gt_ids, labels[labels >= 0]), minlength=len(gt_ids))
    best_iou = np.zeros(len(gt_ids), dtype=np.float64)
    membership = np.zeros(len(labels), dtype=np.int32)
    for hypothesis in hypotheses:
        hypothesis = np.asarray(hypothesis, dtype=np.int64)
        if not len(hypothesis):
            continue
        membership[hypothesis] += 1
        hypothesis_labels = labels[hypothesis]
        valid = hypothesis_labels >= 0
        if not valid.any():
            continue
        intersection = np.bincount(np.searchsorted(gt_ids, hypothesis_labels[valid]), minlength=len(gt_ids))
        iou = intersection / np.maximum(gt_sizes + len(hypothesis) - intersection, 1)
        np.maximum(best_iou, iou, out=best_iou)
    covered = membership[membership > 0]
    return {
        "ar": float(np.mean([(best_iou >= threshold).mean() for threshold in thresholds])),
        "r50": float((best_iou >= 0.50).mean()),
        "r75": float((best_iou >= 0.75).mean()),
        "r90": float((best_iou >= 0.90).mean()),
        "n_gt": int(len(gt_ids)),
        "n_hyps": int(len(hypotheses)),
        "mean_memb": float(covered.mean()) if len(covered) else 0.0,
        "max_memb": int(membership.max()) if len(membership) else 0,
    }


def gt_matching_hypotheses(
    labels: np.ndarray,
    hypotheses: tuple[np.ndarray, ...] | list[np.ndarray],
) -> list[np.ndarray]:
    """Return each GT instance's unique best hypothesis when IoU is at least 0.30."""
    labels = np.asarray(labels, dtype=np.int64)
    gt_ids = np.unique(labels[labels >= 0])
    gt_sizes = np.bincount(np.searchsorted(gt_ids, labels[labels >= 0]), minlength=len(gt_ids))
    best_iou = np.zeros(len(gt_ids), dtype=np.float64)
    best_hypothesis = np.full(len(gt_ids), -1, dtype=np.int64)
    for hypothesis_id, hypothesis in enumerate(hypotheses):
        hypothesis = np.asarray(hypothesis, dtype=np.int64)
        hypothesis_labels = labels[hypothesis]
        valid = hypothesis_labels >= 0
        if not valid.any():
            continue
        intersection = np.bincount(np.searchsorted(gt_ids, hypothesis_labels[valid]), minlength=len(gt_ids))
        iou = intersection / np.maximum(gt_sizes + len(hypothesis) - intersection, 1)
        update = iou > best_iou
        best_iou[update] = iou[update]
        best_hypothesis[update] = hypothesis_id
    representatives = sorted(set(best_hypothesis[best_iou >= 0.30].tolist()) - {-1})
    return [np.asarray(hypotheses[idx], dtype=np.int64) for idx in representatives]


def evaluate_prediction(
    *,
    scene: str,
    scene_dir: str | Path,
    vipe_output_dir: str | Path,
    gt_points: np.ndarray,
    gt_labels: np.ndarray,
    excluded_ids: list[int] | tuple[int, ...],
    config,
) -> dict[str, float | int | list[float]]:
    scene_dir = Path(scene_dir)
    vipe_output_dir = Path(vipe_output_dir)
    prediction = load_instance_prediction(vipe_output_dir / "instances" / f"{scene}.npz")
    with np.load(vipe_output_dir / "pose" / f"{scene}.npz") as pose_data:
        pred_c2w = pose_data["data"].astype(np.float64)
        indices = pose_data["inds"].astype(np.int64)
    expected_indices = np.arange(scene_frame_count(scene_dir))
    if not np.array_equal(indices, expected_indices) or len(pred_c2w) != len(expected_indices):
        raise ValueError(f"{scene}: predicted poses do not cover the contiguous canonical sequence")

    gt_c2w = np.stack(
        [np.loadtxt(scene_dir / "pose" / f"{frame_stem(idx)}.txt", dtype=np.float64) for idx in expected_indices]
    )
    valid = np.isfinite(pred_c2w).all(axis=(1, 2)) & np.isfinite(gt_c2w).all(axis=(1, 2))
    if int(valid.sum()) < 3:
        raise ValueError(f"{scene}: fewer than three finite pose pairs for SE3 alignment")
    rotation, translation = kabsch_se3(pred_c2w[valid, :3, 3], gt_c2w[valid, :3, 3])
    aligned_centers = apply_se3(pred_c2w[valid, :3, 3], rotation, translation)
    ate = float(np.sqrt(np.mean(np.sum((aligned_centers - gt_c2w[valid, :3, 3]) ** 2, axis=1))))

    aligned_prediction = apply_se3(prediction.points, rotation, translation)
    distances, nearest = cKDTree(np.asarray(gt_points, dtype=np.float64)).query(aligned_prediction, k=1, workers=-1)
    transferred = apply_exclusions(np.asarray(gt_labels)[nearest], excluded_ids)
    thresholds = np.round(
        np.arange(float(config.metric.iou_min), float(config.metric.iou_max) + 1e-9, float(config.metric.iou_step)),
        2,
    )
    metrics = recall_ar(transferred, prediction.hypotheses, thresholds)
    pcd_dir = vipe_output_dir / "pcd"
    write_labeled_instance_ply(pcd_dir / f"{scene}_instances_gt.ply", prediction.points, transferred)
    write_instance_ply(
        pcd_dir / f"{scene}_instances_gtmatch.ply",
        prediction.points,
        gt_matching_hypotheses(transferred, prediction.hypotheses),
    )
    metrics.update(
        {
            "membership_budget": prediction.membership_budget,
            "ate_se3_m": ate,
            "label_transfer_distance_p50_m": float(np.percentile(distances, 50)),
            "label_transfer_distance_p90_m": float(np.percentile(distances, 90)),
            "label_transfer_distance_p99_m": float(np.percentile(distances, 99)),
            "excluded_gt_ids": list(excluded_ids),
        }
    )
    return metrics
