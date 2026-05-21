# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gc
import json
import os
import random
import time

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from tqdm import tqdm

from vipe.utils.misc import sort_image_sequence


class AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _scene_sort_key(name: str) -> tuple[int, int]:
    prefix = name.replace("scene", "")
    major, minor = prefix.split("_")
    return int(major), int(minor)


def _load_scene_list(input_root: Path) -> list[str]:
    if not input_root.exists():
        return []
    scenes = [p.name for p in input_root.iterdir() if p.is_dir() and p.name.startswith("scene")]
    return sorted(scenes, key=_scene_sort_key)


def as_homogeneous(ext: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if ext.shape[-2:] == (4, 4):
        return ext
    if ext.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsics with shape (...,3,4) or (...,4,4), got {ext.shape}")
    if isinstance(ext, torch.Tensor):
        bottom = torch.zeros_like(ext[..., :1, :4])
        bottom[..., 0, 3] = 1.0
        return torch.cat([ext, bottom], dim=-2)
    bottom = np.zeros_like(ext[..., :1, :4])
    bottom[..., 0, 3] = 1.0
    return np.concatenate([ext, bottom], axis=-2)


def _camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = as_homogeneous(extrinsics).astype(np.float64, copy=False)
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    return -(np.swapaxes(rotation, 1, 2) @ translation[..., None])[..., 0]


def _umeyama_sim3(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if src_xyz.shape != dst_xyz.shape or len(src_xyz) < 2:
        raise ValueError("Need at least two matched camera centers for Sim3 alignment")

    src_mean = src_xyz.mean(axis=0)
    dst_mean = dst_xyz.mean(axis=0)
    src_centered = src_xyz - src_mean
    dst_centered = dst_xyz - dst_mean
    covariance = (dst_centered.T @ src_centered) / len(src_xyz)
    u, singular_vals, vh = np.linalg.svd(covariance)

    sign = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1, -1] = -1.0

    rotation = u @ sign @ vh
    src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
    if src_var <= 0.0:
        raise ValueError("Degenerate camera centers for Sim3 alignment")
    scale = np.sum(singular_vals * np.diag(sign)) / src_var
    translation = dst_mean - scale * (rotation @ src_mean)
    return float(scale), rotation, translation


def _apply_sim3(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def _ransac_umeyama_sim3(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    max_iters: int = 10,
    random_state: int = 42,
) -> tuple[float, np.ndarray, np.ndarray]:
    if len(src_xyz) < 3:
        return _umeyama_sim3(src_xyz, dst_xyz)

    rng = np.random.default_rng(random_state)
    all_indices = np.arange(len(src_xyz))
    sample_count = max(3, (len(src_xyz) + 1) // 2)
    scale0, rotation0, translation0 = _umeyama_sim3(src_xyz, dst_xyz)
    aligned0 = _apply_sim3(src_xyz, scale0, rotation0, translation0)
    nearest = [np.linalg.norm(dst_xyz - point[None, :], axis=1).min() for point in aligned0]
    inlier_thresh = float(np.median(nearest))

    best_model = (scale0, rotation0, translation0)
    best_inliers = None
    best_score = (-1, float("inf"))
    for _ in range(max_iters):
        sample = rng.choice(all_indices, size=sample_count, replace=False)
        try:
            scale, rotation, translation = _umeyama_sim3(src_xyz[sample], dst_xyz[sample])
        except ValueError:
            continue
        errors = np.linalg.norm(_apply_sim3(src_xyz, scale, rotation, translation) - dst_xyz, axis=1)
        inliers = errors <= inlier_thresh
        inlier_count = int(inliers.sum())
        mean_error = float(errors[inliers].mean()) if inlier_count > 0 else float("inf")
        if (inlier_count > best_score[0]) or (inlier_count == best_score[0] and mean_error < best_score[1]):
            best_model = (scale, rotation, translation)
            best_inliers = inliers
            best_score = (inlier_count, mean_error)

    if best_inliers is not None and int(best_inliers.sum()) >= 3:
        return _umeyama_sim3(src_xyz[best_inliers], dst_xyz[best_inliers])
    return best_model


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    positive_mask = x > 0
    ret = torch.zeros_like(x)
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    out = out[..., [1, 2, 3, 0]]
    return torch.where(out[..., 3:4] < 0, -out, out)


def closed_form_inverse_se3(se3: torch.Tensor) -> torch.Tensor:
    rotation = se3[:, :3, :3]
    translation = se3[:, :3, 3:]
    out = torch.eye(4, dtype=se3.dtype, device=se3.device)[None].repeat(len(se3), 1, 1)
    out[:, :3, :3] = rotation.transpose(1, 2)
    out[:, :3, 3:] = -rotation.transpose(1, 2).bmm(translation)
    return out


def align_to_first_camera(camera_poses: torch.Tensor) -> torch.Tensor:
    first_inverse = closed_form_inverse_se3(camera_poses[0][None])
    return torch.matmul(camera_poses, first_inverse)


def rotation_angle(rot_gt: torch.Tensor, rot_pred: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    return torch.arccos(1 - 2 * loss_q) * 180 / np.pi


def translation_angle(tvec_gt: torch.Tensor, tvec_pred: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    t_pred_norm = torch.norm(tvec_pred, dim=1, keepdim=True)
    t_gt_norm = torch.norm(tvec_gt, dim=1, keepdim=True)
    t_pred = tvec_pred / (t_pred_norm + eps)
    t_gt = tvec_gt / (t_gt_norm + eps)
    loss_t = torch.clamp_min(1.0 - torch.sum(t_pred * t_gt, dim=1) ** 2, eps)
    err = torch.acos(torch.sqrt(1 - loss_t)) * 180.0 / np.pi
    err = torch.min(err, (180 - err).abs())
    err[torch.isnan(err) | torch.isinf(err)] = 1e6
    return err


@torch.no_grad()
def compute_pose(pred_se3: torch.Tensor, gt_se3: torch.Tensor) -> AttrDict:
    device = torch.device("cuda:0")
    pred_se3 = align_to_first_camera(pred_se3.to(device=device, dtype=torch.float32))
    gt_se3 = align_to_first_camera(gt_se3.to(device=device, dtype=torch.float32))
    num_frames = len(pred_se3)
    num_pairs = num_frames * (num_frames - 1) // 2
    thresholds = (30, 15, 5, 3)
    histograms = {threshold: torch.zeros(threshold, dtype=torch.float64, device=device) for threshold in thresholds}

    pair_indices = torch.combinations(torch.arange(num_frames, device=device), 2, with_replacement=False)
    chunk_size = 1_000_000
    for start in range(0, num_pairs, chunk_size):
        end = min(start + chunk_size, num_pairs)
        i1 = pair_indices[start:end, 0]
        i2 = pair_indices[start:end, 1]
        relative_pose_gt = closed_form_inverse_se3(gt_se3[i1]).bmm(gt_se3[i2])
        relative_pose_pred = closed_form_inverse_se3(pred_se3[i1]).bmm(pred_se3[i2])
        max_errors = torch.maximum(
            rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]),
            translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]),
        )
        for threshold, hist in histograms.items():
            valid = (max_errors >= 0) & (max_errors <= threshold)
            if bool(valid.any().item()):
                bins = torch.floor(max_errors[valid]).to(torch.long).clamp(max=threshold - 1)
                hist += torch.bincount(bins, minlength=threshold)[:threshold].to(torch.float64)

    output = AttrDict()
    for threshold, key in [(30, "auc30"), (15, "auc15"), (5, "auc05"), (3, "auc03")]:
        output[key] = float(torch.cumsum(histograms[threshold] / float(num_pairs), dim=0).mean().item())
    return output


def mean_squared_nn_distance(
    reference: np.ndarray,
    query: np.ndarray,
    chunk_size: int = 1_000_000,
    progress_desc: str | None = None,
) -> float:
    if len(reference) == 0 or len(query) == 0:
        return float("inf")

    device = o3d.core.Device("CUDA:0")
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    if progress_desc is not None:
        tqdm.write(f"[ScanNet] Open3D CUDA NN index start | {progress_desc} | ref={len(reference):,} query={len(query):,}")
    start_tree = time.perf_counter()
    ref_tensor = o3d.core.Tensor(reference, dtype=o3d.core.Dtype.Float32, device=device)
    nns = o3d.core.nns.NearestNeighborSearch(ref_tensor)
    nns.knn_index()
    if progress_desc is not None:
        tqdm.write(f"[ScanNet] Open3D CUDA NN index done  | {progress_desc} | {time.perf_counter() - start_tree:.2f}s")

    total = 0.0
    count = 0
    iterator = range(0, len(query), chunk_size)
    if progress_desc is not None:
        iterator = tqdm(
            iterator,
            total=(len(query) + chunk_size - 1) // chunk_size,
            desc=f"{progress_desc} CUDA",
            leave=False,
        )
    for start in iterator:
        end = min(start + chunk_size, len(query))
        query_tensor = o3d.core.Tensor(query[start:end], dtype=o3d.core.Dtype.Float32, device=device)
        _, squared_distances = nns.knn_search(query_tensor, 1)
        total += float(squared_distances.sum().cpu().numpy())
        count += end - start
    return total / count


def nn_correspondance(verts1: np.ndarray, verts2: np.ndarray) -> np.ndarray:
    if len(verts1) == 0 or len(verts2) == 0:
        return np.array([])

    device = o3d.core.Device("CUDA:0")
    reference = o3d.core.Tensor(np.ascontiguousarray(verts1, dtype=np.float32), device=device)
    query = np.ascontiguousarray(verts2, dtype=np.float32)
    nns = o3d.core.nns.NearestNeighborSearch(reference)
    nns.knn_index()

    distances = []
    chunk_size = 1_000_000
    for start in range(0, len(query), chunk_size):
        end = min(start + chunk_size, len(query))
        query_tensor = o3d.core.Tensor(query[start:end], device=device)
        _, squared_distances = nns.knn_search(query_tensor, 1)
        distances.append(np.sqrt(squared_distances.cpu().numpy().reshape(-1)))
    return np.concatenate(distances)


def evaluate_3d_reconstruction_l2(
    pcd_pred: o3d.geometry.PointCloud | np.ndarray,
    pcd_trgt: o3d.geometry.PointCloud | np.ndarray,
    progress_desc: str | None = None,
    chunk_size: int = 1_000_000,
) -> dict[str, float]:
    if isinstance(pcd_pred, np.ndarray):
        pcd_pred = _point_cloud_from_arrays(pcd_pred)
    if isinstance(pcd_trgt, np.ndarray):
        pcd_trgt = _point_cloud_from_arrays(pcd_trgt)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)
    if len(verts_pred) == 0 or len(verts_trgt) == 0:
        return {"acc": float("inf"), "comp": float("inf"), "overall": float("inf")}

    accuracy = mean_squared_nn_distance(
        verts_trgt,
        verts_pred,
        chunk_size=chunk_size,
        progress_desc=None if progress_desc is None else f"{progress_desc} pred->gt",
    )
    completeness = mean_squared_nn_distance(
        verts_pred,
        verts_trgt,
        chunk_size=chunk_size,
        progress_desc=None if progress_desc is None else f"{progress_desc} gt->pred",
    )
    return {"acc": accuracy, "comp": completeness, "overall": (accuracy + completeness) / 2}


def evaluate_3d_reconstruction(
    pcd_pred: o3d.geometry.PointCloud | np.ndarray,
    pcd_trgt: o3d.geometry.PointCloud | np.ndarray,
    threshold: float = 0.05,
    down_sample: float | None = None,
) -> dict[str, float]:
    if isinstance(pcd_pred, np.ndarray):
        pcd_pred = _point_cloud_from_arrays(pcd_pred)
    if isinstance(pcd_trgt, np.ndarray):
        pcd_trgt = _point_cloud_from_arrays(pcd_trgt)
    if down_sample is not None and down_sample > 0:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)
    if len(verts_pred) == 0 or len(verts_trgt) == 0:
        return {
            "acc": float("inf"),
            "comp": float("inf"),
            "overall": float("inf"),
            "precision": 0.0,
            "recall": 0.0,
            "fscore": 0.0,
        }

    dist_pred_to_gt = nn_correspondance(verts_trgt, verts_pred)
    dist_gt_to_pred = nn_correspondance(verts_pred, verts_trgt)
    accuracy = float(np.mean(dist_pred_to_gt))
    completeness = float(np.mean(dist_gt_to_pred))
    precision = float(np.mean((dist_pred_to_gt < threshold).astype(float)))
    recall = float(np.mean((dist_gt_to_pred < threshold).astype(float)))
    fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "acc": accuracy,
        "comp": completeness,
        "overall": (accuracy + completeness) / 2,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def sample_points_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int = 10_000_000,
) -> o3d.geometry.PointCloud:
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    if pcd.has_colors():
        pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(pcd.colors), 0.0, 1.0))
    return pcd


def _image_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def _image_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    scores = []
    for channel in range(3):
        x = pred[:, :, channel].astype(np.float32, copy=False)
        y = target[:, :, channel].astype(np.float32, copy=False)
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        scores.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return float(np.mean(scores))


def _point_cloud_from_arrays(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def _voxelized_render_arrays(pcd: o3d.geometry.PointCloud, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    render_pcd = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(render_pcd.points).astype(np.float32, copy=False)
    colors = np.asarray(render_pcd.colors).astype(np.float32, copy=False)
    if len(colors) != len(points):
        colors = np.zeros((len(points), 3), dtype=np.float32)
    return points, np.clip(colors, 0.0, 1.0)


def _load_cached_point_cloud(cache_path: Path) -> o3d.geometry.PointCloud | None:
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as data:
        points = data["points"]
        colors = data["colors"] if "colors" in data else None
    return _point_cloud_from_arrays(points, colors)


def _write_cached_point_cloud(cache_path: Path, pcd: o3d.geometry.PointCloud) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(pcd.points).astype(np.float32, copy=False)
    colors = np.asarray(pcd.colors).astype(np.float32, copy=False)
    if len(colors) != len(points):
        colors = np.zeros((len(points), 3), dtype=np.float32)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez(f, points=points, colors=np.clip(colors, 0.0, 1.0))
    os.replace(tmp_path, cache_path)


def _load_cached_render_arrays(
    cache_path: Path,
    source_path: str,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.exists():
        return None
    source_stat = os.stat(source_path)
    with np.load(cache_path, allow_pickle=False) as data:
        if int(data["source_mtime_ns"]) != source_stat.st_mtime_ns:
            return None
        if int(data["source_size"]) != source_stat.st_size:
            return None
        if float(data["voxel_size"]) != float(voxel_size):
            return None
        return data["points"], data["colors"]


def _write_cached_render_arrays(
    cache_path: Path,
    source_path: str,
    voxel_size: float,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    source_stat = os.stat(source_path)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez(
            f,
            points=points.astype(np.float32, copy=False),
            colors=colors.astype(np.float32, copy=False),
            source_mtime_ns=np.array(source_stat.st_mtime_ns, dtype=np.int64),
            source_size=np.array(source_stat.st_size, dtype=np.int64),
            voxel_size=np.array(voxel_size, dtype=np.float32),
        )
    os.replace(tmp_path, cache_path)


def _make_offset_shells(radius_cap: int, device: torch.device) -> list[tuple[int, torch.Tensor]]:
    shells = []
    for radius in range(radius_cap + 1):
        offsets = [
            (dx, dy)
            for dy in range(-radius_cap, radius_cap + 1)
            for dx in range(-radius_cap, radius_cap + 1)
            if max(abs(dx), abs(dy)) == radius
        ]
        shells.append((radius, torch.tensor(offsets, dtype=torch.int64, device=device)))
    return shells


@torch.no_grad()
def _render_voxel_cloud_cuda(
    points: torch.Tensor,
    colors: torch.Tensor,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    voxel_size: float,
    chunk_size: int,
    radius_cap: int,
    offset_shells: list[tuple[int, torch.Tensor]],
) -> np.ndarray:
    device = points.device
    zbuf = torch.full((height * width,), float("inf"), dtype=torch.float32, device=device)
    image = torch.zeros((height * width, 3), dtype=torch.float32, device=device)
    rotation = torch.as_tensor(extrinsic[:3, :3], dtype=torch.float32, device=device)
    translation = torch.as_tensor(extrinsic[:3, 3], dtype=torch.float32, device=device)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    max_focal = max(abs(fx), abs(fy))

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        cam = points[start:end] @ rotation.T + translation
        z = cam[:, 2]
        valid = z > 1e-4
        if not bool(valid.any().item()):
            continue
        cam = cam[valid]
        z = z[valid]
        cols = colors[start:end][valid]
        u = torch.round(fx * cam[:, 0] / z + cx).to(torch.int64)
        v = torch.round(fy * cam[:, 1] / z + cy).to(torch.int64)
        radii = torch.ceil(0.5 * voxel_size * max_focal / z).to(torch.int64).clamp_(0, radius_cap)
        near_image = (u >= -radius_cap) & (u < width + radius_cap) & (v >= -radius_cap) & (v < height + radius_cap)
        if not bool(near_image.any().item()):
            continue
        u = u[near_image]
        v = v[near_image]
        z = z[near_image]
        cols = cols[near_image]
        radii = radii[near_image]

        for radius, offsets in offset_shells:
            if radius > 0:
                active = radii >= radius
                if not bool(active.any().item()):
                    continue
                u0, v0, z0, cols0 = u[active], v[active], z[active], cols[active]
            else:
                u0, v0, z0, cols0 = u, v, z, cols
            uu = u0[:, None] + offsets[:, 0][None, :]
            vv = v0[:, None] + offsets[:, 1][None, :]
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            if not bool(inside.any().item()):
                continue
            rows, cols_idx = inside.nonzero(as_tuple=True)
            pix = vv[rows, cols_idx] * width + uu[rows, cols_idx]
            pix_z = z0[rows]
            zbuf.scatter_reduce_(0, pix, pix_z, reduce="amin", include_self=True)
            winners = torch.abs(pix_z - zbuf[pix]) <= 1e-6
            if bool(winners.any().item()):
                image[pix[winners]] = cols0[rows[winners]]
    return image.reshape(height, width, 3).cpu().numpy()


class ScanNetDataset:
    render_voxel_size = 0.001
    render_num_images = 800
    render_chunk_size = 1_000_000
    render_radius_cap = 8
    eval_threshold = 0.05
    down_sample = None

    def __init__(self, input_root: Path, raw_root: Path):
        self.data_root = Path(input_root)
        self.raw_root = Path(raw_root)
        self.SCENES = _load_scene_list(self.data_root)
        self._scene_cache: dict[str, AttrDict] = {}

    def get_data(self, scene: str) -> AttrDict:
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        scene_dir = self.data_root / scene
        color_dir = scene_dir / "color"
        pose_dir = scene_dir / "pose"
        depth_dir = scene_dir / "depth"
        intrinsic_path = scene_dir / "intrinsic" / "intrinsic_color.txt"
        if not color_dir.is_dir():
            raise FileNotFoundError(f"Missing ScanNet color dir: {color_dir}")
        if not pose_dir.is_dir():
            raise FileNotFoundError(f"Missing ScanNet pose dir: {pose_dir}")
        if not intrinsic_path.is_file():
            raise FileNotFoundError(f"Missing ScanNet intrinsic file: {intrinsic_path}")

        raw_scene_dir = self.raw_root / scene
        gt_mesh_path = raw_scene_dir / f"{scene}_vh_clean_2.ply"
        if not gt_mesh_path.is_file():
            fallback = raw_scene_dir / f"{scene}_vh_clean.ply"
            if fallback.is_file():
                gt_mesh_path = fallback
            else:
                raise FileNotFoundError(f"Missing ScanNet GT mesh: {gt_mesh_path} (and fallback {fallback})")

        ixt_shared = np.loadtxt(intrinsic_path, dtype=np.float32)[:3, :3]
        image_files = sort_image_sequence(color_dir.glob("*.jpg"))
        out = AttrDict(image_files=[], extrinsics=[], intrinsics=[], aux=AttrDict(gt_mesh_path=str(gt_mesh_path), gt_depth_files=[]))
        for img_path in tqdm(image_files, desc=f"[ScanNet] {scene} load poses", leave=False):
            frame_id = img_path.stem
            pose_path = pose_dir / f"{frame_id}.txt"
            depth_path = depth_dir / f"{frame_id}.png"
            if not pose_path.is_file():
                continue
            c2w = np.loadtxt(pose_path, dtype=np.float32)
            if c2w.shape != (4, 4):
                continue
            out.image_files.append(str(img_path))
            out.extrinsics.append(np.linalg.inv(c2w).astype(np.float32))
            out.intrinsics.append(ixt_shared.copy())
            out.aux.gt_depth_files.append(str(depth_path))

        out.extrinsics = np.asarray(out.extrinsics, dtype=np.float32)
        out.intrinsics = np.asarray(out.intrinsics, dtype=np.float32)
        self._scene_cache[scene] = out
        tqdm.write(f"[ScanNet] {scene}: {len(out.image_files)} images")
        return out

    def result_path(self, export_dir: str) -> str:
        return os.path.join(export_dir, "exports", "vipe_manifest.json")

    def result_exists(self, result_path: str) -> bool:
        return os.path.exists(result_path)

    def _load_gt_meta(self, result_path: str) -> AttrDict | None:
        gt_meta_path = os.path.join(os.path.dirname(result_path), "gt_meta.npz")
        if not os.path.exists(gt_meta_path):
            return None
        data = np.load(gt_meta_path, allow_pickle=True)
        return AttrDict(
            extrinsics=data["extrinsics"],
            intrinsics=data["intrinsics"],
            image_files=list(data["image_files"]),
        )

    def _load_vipe_manifest(self, result_path: str) -> dict:
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_vipe_pose_map(self, manifest: dict) -> dict[int, np.ndarray]:
        pose_npz = np.load(manifest["pose_path"])
        return {int(idx): pose.astype(np.float32) for idx, pose in zip(pose_npz["inds"], pose_npz["data"])}

    def load_pred_extrinsics(self, result_path: str) -> np.ndarray:
        manifest = self._load_vipe_manifest(result_path)
        pose_map = self._load_vipe_pose_map(manifest)
        return np.stack(
            [np.linalg.inv(pose_map[int(frame_idx)]).astype(np.float32) for frame_idx in manifest["frame_indices"]]
        )

    def _tsdf_pcd_path(self, result_path: str) -> str:
        manifest = self._load_vipe_manifest(result_path)
        path = manifest["tsdf_pcd_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing ViPE TSDF PLY: {path}")
        return path

    def _aligned_tsdf_pcd_path(self, result_path: str) -> Path:
        return Path(result_path).parents[2] / "eval_cache" / "pcd_tsdf_aligned.ply"

    def _tsdf_alignment_is_current(self, aligned_path: Path, source_path: str, result_path: str, manifest: dict) -> bool:
        if not aligned_path.exists():
            return False
        input_paths = [source_path, result_path, manifest["pose_path"]]
        gt_meta_path = Path(result_path).with_name("gt_meta.npz")
        if gt_meta_path.exists():
            input_paths.append(str(gt_meta_path))
        newest_input = max(Path(path).stat().st_mtime_ns for path in input_paths)
        return aligned_path.stat().st_mtime_ns >= newest_input

    def _align_tsdf_pcd(self, scene: str, result_path: str) -> str:
        manifest = self._load_vipe_manifest(result_path)
        source_path = self._tsdf_pcd_path(result_path)
        aligned_path = self._aligned_tsdf_pcd_path(result_path)
        if self._tsdf_alignment_is_current(aligned_path, source_path, result_path, manifest):
            tqdm.write(f"[ScanNet] aligned TSDF cache hit | {scene} | {aligned_path}")
            return str(aligned_path)

        gt_meta = self._load_gt_meta(result_path)
        if gt_meta is None:
            gt_meta = self.get_data(scene)
        frame_indices = [int(idx) for idx in manifest["frame_indices"]]
        if len(gt_meta.extrinsics) != len(frame_indices):
            raise ValueError(
                f"GT metadata length ({len(gt_meta.extrinsics)}) does not match manifest frame count ({len(frame_indices)})"
            )

        pose_map = self._load_vipe_pose_map(manifest)
        pred_extrinsics = np.stack([np.linalg.inv(pose_map[idx]).astype(np.float64) for idx in frame_indices])
        gt_centers = _camera_centers_from_extrinsics(gt_meta.extrinsics)
        pred_centers = _camera_centers_from_extrinsics(pred_extrinsics)
        scale, rotation, translation = _ransac_umeyama_sim3(pred_centers, gt_centers)

        start = time.perf_counter()
        tqdm.write(f"[ScanNet] align TSDF start | {scene} | scale={scale:.6f} | {source_path}")
        pcd = o3d.io.read_point_cloud(source_path)
        points = np.asarray(pcd.points)
        if len(points) > 0:
            pcd.points = o3d.utility.Vector3dVector(_apply_sim3(points, scale, rotation, translation))
        if pcd.has_normals():
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) @ rotation.T)

        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = aligned_path.with_name(f"{aligned_path.stem}.tmp{aligned_path.suffix}")
        o3d.io.write_point_cloud(str(tmp_path), pcd)
        os.replace(tmp_path, aligned_path)
        tqdm.write(
            f"[ScanNet] align TSDF done  | {scene} | "
            f"points={len(pcd.points):,} | {time.perf_counter() - start:.2f}s"
        )
        return str(aligned_path)

    def _artifact_sample_count(self, result_path: str) -> int:
        manifest = self._load_vipe_manifest(result_path)
        return int(manifest["output"]["pcd_max_points"])

    def eval3d(self, scene: str, result_path: str) -> dict[str, float]:
        start_eval = time.perf_counter()
        full_gt_data = self.get_data(scene)
        pcd_path = self._align_tsdf_pcd(scene, result_path)
        method_name = "pcd_tsdf"
        eval_cache_dir = Path(result_path).parents[2] / "eval_cache"
        tqdm.write(f"[ScanNet] eval3d start | {scene} | {pcd_path}")
        sample_count = self._artifact_sample_count(result_path)
        gt_cache_path = eval_cache_dir / f"gt_sample_{sample_count}.npz"
        start_gt = time.perf_counter()
        gt_pcd = _load_cached_point_cloud(gt_cache_path)
        if gt_pcd is None:
            tqdm.write(f"[ScanNet] sample GT start | {scene} | points={sample_count:,}")
            gt_mesh = o3d.io.read_triangle_mesh(full_gt_data.aux.gt_mesh_path)
            gt_pcd = sample_points_from_mesh(gt_mesh, sample_count)
            _write_cached_point_cloud(gt_cache_path, gt_pcd)
            tqdm.write(f"[ScanNet] sample GT done  | {scene} | {time.perf_counter() - start_gt:.2f}s")
        else:
            tqdm.write(
                f"[ScanNet] sample GT cache hit | {scene} | "
                f"points={len(gt_pcd.points):,} | {time.perf_counter() - start_gt:.2f}s"
            )

        start_read = time.perf_counter()
        pred_pcd = o3d.io.read_point_cloud(pcd_path)
        tqdm.write(f"[ScanNet] read pred done | {scene} | points={len(pred_pcd.points):,} | {time.perf_counter() - start_read:.2f}s")
        pred_eval_pcd = pred_pcd

        start_crop = time.perf_counter()
        aabb = gt_pcd.get_axis_aligned_bounding_box()
        points = np.asarray(pred_pcd.points)
        if points.size > 0:
            inside_mask = (
                (points[:, 0] >= aabb.min_bound[0] - 0.1)
                & (points[:, 0] <= aabb.max_bound[0] + 0.1)
                & (points[:, 1] >= aabb.min_bound[1] - 0.1)
                & (points[:, 1] <= aabb.max_bound[1] + 0.1)
                & (points[:, 2] >= aabb.min_bound[2] - 0.1)
                & (points[:, 2] <= aabb.max_bound[2] + 0.1)
            )
            pred_eval_pcd = pred_pcd.select_by_index(np.nonzero(inside_mask)[0])
        tqdm.write(
            f"[ScanNet] AABB crop done | {scene} | "
            f"eval_points={len(pred_eval_pcd.points):,}/{len(pred_pcd.points):,} | {time.perf_counter() - start_crop:.2f}s"
        )

        start_geom = time.perf_counter()
        result = evaluate_3d_reconstruction_l2(
            pred_eval_pcd,
            gt_pcd,
            progress_desc=f"{scene} {method_name} L2 NN",
        )
        tqdm.write(f"[ScanNet] geometry metric done | {scene} | {result} | {time.perf_counter() - start_geom:.2f}s")
        del gt_pcd, pred_eval_pcd, points
        gc.collect()

        render_data = self._load_gt_meta(result_path) or full_gt_data
        result.update(self._compute_render_metrics(scene, method_name, pred_pcd, render_data, pcd_path, eval_cache_dir))
        tqdm.write(f"[ScanNet] eval3d done | {scene} | {method_name} | {time.perf_counter() - start_eval:.2f}s")
        return result

    def _compute_render_metrics(
        self,
        scene: str,
        method_name: str,
        pcd: o3d.geometry.PointCloud,
        scene_data: AttrDict,
        source_path: str,
        cache_dir: Path,
    ) -> dict[str, float]:
        start_total = time.perf_counter()
        label = f"{scene} {method_name}"
        tqdm.write(f"[ScanNet] render metric voxelize start | {label} | voxel={self.render_voxel_size}m")
        cache_name = f"{method_name}_render_vox_{str(self.render_voxel_size).replace('.', 'p')}.npz"
        render_cache_path = cache_dir / cache_name
        cached = _load_cached_render_arrays(render_cache_path, source_path, self.render_voxel_size)
        if cached is None:
            points, colors = _voxelized_render_arrays(pcd, self.render_voxel_size)
            _write_cached_render_arrays(render_cache_path, source_path, self.render_voxel_size, points, colors)
            tqdm.write(
                f"[ScanNet] render metric voxelize done  | {label} | "
                f"voxels={len(points):,} | {time.perf_counter() - start_total:.2f}s"
            )
        else:
            points, colors = cached
            tqdm.write(
                f"[ScanNet] render metric voxelize cache hit | {label} | "
                f"voxels={len(points):,} | {time.perf_counter() - start_total:.2f}s"
            )

        frame_indices = np.arange(len(scene_data.image_files))
        if self.render_num_images != -1:
            sample_count = min(self.render_num_images, len(frame_indices))
            frame_indices = np.sort(np.random.default_rng(seed=42).choice(frame_indices, size=sample_count, replace=False))

        tasks = [
            (int(idx), str(scene_data.image_files[idx]), scene_data.extrinsics[idx], scene_data.intrinsics[idx])
            for idx in frame_indices
        ]
        tqdm.write(
            f"[ScanNet] render metric project start | {label} | "
            f"frames={len(tasks):,}/{len(scene_data.image_files):,} device=cuda:0"
        )
        device = torch.device("cuda:0")
        points_t = torch.as_tensor(np.ascontiguousarray(points), dtype=torch.float32, device=device)
        colors_t = torch.as_tensor(np.ascontiguousarray(colors), dtype=torch.float32, device=device)
        offset_shells = _make_offset_shells(self.render_radius_cap, device)
        psnr_values = []
        ssim_values = []
        progress = tqdm(total=len(tasks), desc=f"{label} render PSNR/SSIM", unit="frame", leave=False)
        for _, image_file, extrinsic, intrinsic in tasks:
            bgr = cv2.imread(image_file, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(image_file)
            target = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            height, width = target.shape[:2]
            render = _render_voxel_cloud_cuda(
                points_t,
                colors_t,
                extrinsic,
                intrinsic,
                height,
                width,
                self.render_voxel_size,
                self.render_chunk_size,
                self.render_radius_cap,
                offset_shells,
            )
            psnr_values.append(_image_psnr(render, target))
            ssim_values.append(_image_ssim(render, target))
            progress.update()
            progress.set_postfix(psnr=f"{np.mean(psnr_values):.2f}", ssim=f"{np.mean(ssim_values):.4f}")
        del points_t, colors_t, offset_shells
        torch.cuda.empty_cache()
        progress.close()
        tqdm.write(f"[ScanNet] render metric project done  | {label} | {time.perf_counter() - start_total:.2f}s")
        return {"psnr": float(np.mean(psnr_values)), "ssim": float(np.mean(ssim_values))}


class ScanNetEvaluator:
    VALID_MODES = {"pose", "recon"}

    def __init__(
        self,
        work_dir: str | Path,
        input_root: str | Path,
        raw_root: str | Path,
        modes: list[str],
        scenes: list[str] | None = None,
        max_frames: int = 100,
        gpu_id: int = 0,
        total_gpus: int = 1,
    ):
        self.work_dir = str(work_dir)
        self.modes = set(modes)
        unknown = self.modes - self.VALID_MODES
        if unknown:
            raise ValueError(f"Unknown modes: {sorted(unknown)}")
        self.scenes_filter = scenes
        self.max_frames = max_frames
        self.gpu_id = gpu_id
        self.total_gpus = total_gpus
        self.datasets = AttrDict(scannet=ScanNetDataset(Path(input_root), Path(raw_root)))
        os.makedirs(self.work_dir, exist_ok=True)

    def _get_scenes(self, dataset: ScanNetDataset) -> list[str]:
        if self.scenes_filter:
            return [scene for scene in dataset.SCENES if scene in self.scenes_filter]
        return list(dataset.SCENES)

    def eval(self) -> dict[str, dict]:
        summary: dict[str, dict] = {}
        if "pose" in self.modes:
            print(f"\n{'=' * 60}")
            print("Evaluating POSE for ScanNet")
            print(f"{'=' * 60}")
            for data, result in self._eval_pose():
                summary[f"{data}_pose"] = result
        if "recon" in self.modes:
            print(f"\n{'=' * 60}")
            print("Evaluating RECON for ScanNet")
            print(f"{'=' * 60}")
            for key, result in self._eval_reconstruction():
                summary[key] = result
        return summary

    def _eval_pose(self):
        os.makedirs(self._metric_dir, exist_ok=True)
        dataset = self.datasets.scannet
        dataset_results = AttrDict()
        for scene in tqdm(self._get_scenes(dataset), desc="scannet scenes", leave=False):
            export_dir = self._export_dir("scannet", scene)
            result_path = dataset.result_path(export_dir)
            if not dataset.result_exists(result_path):
                tqdm.write(f"[ERROR] Result file not found: {result_path}")
                continue
            gt_meta = self._load_gt_meta(export_dir)
            if gt_meta is not None:
                num_frames = len(gt_meta["extrinsics"])
                num_pairs = num_frames * (num_frames - 1) // 2
                tqdm.write(f"[INFO] Pose start | scannet | {scene} | frames={num_frames} pairs={num_pairs}")
                result = self._compute_pose_with_gt(dataset, result_path, gt_meta)
            else:
                scene_data = dataset.get_data(scene)
                num_frames = len(scene_data.image_files)
                num_pairs = num_frames * (num_frames - 1) // 2
                tqdm.write(f"[INFO] Pose start | scannet | {scene} | frames={num_frames} pairs={num_pairs}")
                result = self._compute_pose_with_gt(dataset, result_path, scene_data)
            dataset_results[scene] = self._to_float_dict(result)
            tqdm.write(f"[INFO] Pose done  | scannet | {scene} | {result}")

        if dataset_results:
            dataset_results["mean"] = self._mean_of_dicts(dataset_results.values())
        self._dump_json(os.path.join(self._metric_dir, "scannet_pose.json"), dataset_results)
        yield "scannet", dataset_results

    def _eval_reconstruction(self):
        os.makedirs(self._metric_dir, exist_ok=True)
        dataset = self.datasets.scannet
        scenes = self._get_scenes(dataset)
        dataset_results = AttrDict()
        tqdm.write(f"[INFO] Starting recon eval for dataset=scannet with {len(scenes)} scene(s)")
        for scene in scenes:
            export_dir = self._export_dir("scannet", scene)
            result_path = dataset.result_path(export_dir)
            result = dataset.eval3d(scene, result_path)
            dataset_results[scene] = self._to_float_dict(result)
            tqdm.write(f"  recon | scannet | {scene}: {result}")

        dataset_results["mean"] = self._mean_of_dicts(dataset_results.values())
        key = "scannet_recon"
        self._dump_json(os.path.join(self._metric_dir, f"{key}.json"), dataset_results)
        yield key, dataset_results

    def _save_gt_meta(self, export_dir: str, scene_data: AttrDict) -> None:
        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        np.savez_compressed(
            meta_path,
            extrinsics=scene_data.extrinsics,
            intrinsics=scene_data.intrinsics,
            image_files=np.array(scene_data.image_files, dtype=object),
        )

    def _load_gt_meta(self, export_dir: str) -> AttrDict | None:
        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        if not os.path.exists(meta_path):
            return None
        data = np.load(meta_path, allow_pickle=True)
        return AttrDict(extrinsics=data["extrinsics"], intrinsics=data["intrinsics"], image_files=list(data["image_files"]))

    def _compute_pose_with_gt(self, dataset: ScanNetDataset, result_path: str, gt_meta: AttrDict) -> dict[str, float]:
        pred_extrinsics = dataset.load_pred_extrinsics(result_path)
        return compute_pose(
            torch.from_numpy(as_homogeneous(pred_extrinsics)),
            torch.from_numpy(as_homogeneous(gt_meta["extrinsics"])),
        )

    def _sample_frames(self, scene_data: AttrDict, scene: str) -> AttrDict:
        if self.max_frames <= 0:
            return scene_data
        num_frames = len(scene_data.image_files)
        if num_frames <= self.max_frames:
            return scene_data
        random.seed(42)
        indices = list(range(num_frames))
        random.shuffle(indices)
        sampled_indices = sorted(indices[: self.max_frames])
        tqdm.write(f"  [Sampling] {scene}: {num_frames} -> {self.max_frames} frames")
        sampled = AttrDict()
        sampled.image_files = [scene_data.image_files[i] for i in sampled_indices]
        sampled.extrinsics = scene_data.extrinsics[sampled_indices]
        sampled.intrinsics = scene_data.intrinsics[sampled_indices]
        sampled.aux = AttrDict()
        for key, value in scene_data.aux.items():
            if isinstance(value, list) and len(value) == num_frames:
                sampled.aux[key] = [value[i] for i in sampled_indices]
            elif isinstance(value, np.ndarray) and len(value) == num_frames:
                sampled.aux[key] = value[sampled_indices]
            else:
                sampled.aux[key] = value
        return sampled

    @property
    def _metric_dir(self) -> str:
        return os.path.join(self.work_dir, "metric_results")

    def _export_dir(self, data: str, scene: str) -> str:
        export_dir = os.path.join(self.work_dir, "model_results", data, scene, "recon")
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    @staticmethod
    def _to_float_dict(data: dict[str, float]) -> dict[str, float]:
        return {key: float(value) for key, value in data.items()}

    @staticmethod
    def _mean_of_dicts(dicts) -> dict[str, float]:
        dicts = list(dicts)
        if not dicts:
            return {}
        keys = dicts[0].keys()
        return {key: float(np.mean([item[key] for item in dicts]).item()) for key in keys}

    @staticmethod
    def _dump_json(path: str, obj: dict, indent: int = 4) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)

    def _load_metrics(self) -> dict[str, dict]:
        metrics = {}
        if not os.path.exists(self._metric_dir):
            return metrics
        for filename in os.listdir(self._metric_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self._metric_dir, filename), encoding="utf-8") as f:
                    metrics[filename[:-5]] = json.load(f)
        return metrics
