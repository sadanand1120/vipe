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
import time

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from tqdm import tqdm

from vipe.utils.data_format import intrinsic_matrix, read_pinhole_intrinsics, read_scene_frames


class AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({key: _to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


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


def _camera_centers_from_c2w(c2w: np.ndarray) -> np.ndarray:
    c2w = as_homogeneous(c2w).astype(np.float64, copy=False)
    return c2w[:, :3, 3]


def _apply_se3(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _first_camera_se3(pred_c2w: np.ndarray, gt_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_c2w = as_homogeneous(pred_c2w).astype(np.float64, copy=False)
    gt_c2w = np.linalg.inv(as_homogeneous(gt_w2c).astype(np.float64, copy=False))
    return gt_c2w[0] @ np.linalg.inv(pred_c2w[0]), gt_c2w


def _valid_pose_mask(*pose_arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(pose_arrays[0]), dtype=bool)
    for poses in pose_arrays:
        mask &= np.isfinite(poses).all(axis=(-2, -1))
    return mask


def _filter_valid_poses(pred_poses: np.ndarray, gt_poses: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    if len(pred_poses) != len(gt_poses):
        raise ValueError(f"{label}: predicted pose count {len(pred_poses)} does not match GT pose count {len(gt_poses)}")
    valid = _valid_pose_mask(pred_poses, gt_poses)
    if not bool(valid.any()):
        raise ValueError(f"{label}: no finite predicted/GT pose pairs")
    dropped = int((~valid).sum())
    if dropped:
        tqdm.write(f"[WARN] {label}: dropping {dropped:,}/{len(valid):,} frames with non-finite predicted/GT poses")
    return pred_poses[valid], gt_poses[valid]


def _first_camera_scale_diagnostic(pred_c2w: np.ndarray, gt_c2w: np.ndarray, se3: np.ndarray) -> float:
    pred_centers = _apply_se3(_camera_centers_from_c2w(pred_c2w), se3)
    gt_centers = _camera_centers_from_c2w(gt_c2w)
    pred_delta = pred_centers - gt_centers[0]
    gt_delta = gt_centers - gt_centers[0]
    denom = float(np.sum(pred_delta * pred_delta))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(pred_delta * gt_delta) / denom)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    positive_mask = x > 0
    ret = torch.zeros_like(x)
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def mat_to_quat(matrix: torch.Tensor, floor_value: float) -> torch.Tensor:
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
    floor = torch.tensor(floor_value, dtype=q_abs.dtype, device=q_abs.device)
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


def rotation_angle(rot_gt: torch.Tensor, rot_pred: torch.Tensor, eps: float, quat_floor: float) -> torch.Tensor:
    q_pred = mat_to_quat(rot_pred, quat_floor)
    q_gt = mat_to_quat(rot_gt, quat_floor)
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    return torch.arccos(1 - 2 * loss_q) * 180 / np.pi


def translation_angle(
    tvec_gt: torch.Tensor,
    tvec_pred: torch.Tensor,
    eps: float,
    invalid_error_value: float,
) -> torch.Tensor:
    t_pred_norm = torch.norm(tvec_pred, dim=1, keepdim=True)
    t_gt_norm = torch.norm(tvec_gt, dim=1, keepdim=True)
    t_pred = tvec_pred / (t_pred_norm + eps)
    t_gt = tvec_gt / (t_gt_norm + eps)
    loss_t = torch.clamp_min(1.0 - torch.sum(t_pred * t_gt, dim=1) ** 2, eps)
    err = torch.acos(torch.sqrt(1 - loss_t)) * 180.0 / np.pi
    err = torch.min(err, (180 - err).abs())
    err[torch.isnan(err) | torch.isinf(err)] = invalid_error_value
    return err


@torch.no_grad()
def compute_pose(pred_se3: torch.Tensor, gt_se3: torch.Tensor, pose_config: AttrDict) -> AttrDict:
    device = torch.device(str(pose_config.device))
    pred_se3 = align_to_first_camera(pred_se3.to(device=device, dtype=torch.float32))
    gt_se3 = align_to_first_camera(gt_se3.to(device=device, dtype=torch.float32))
    num_frames = len(pred_se3)
    num_pairs = num_frames * (num_frames - 1) // 2
    thresholds = tuple(int(threshold) for threshold in pose_config.auc_threshold_degrees)
    histograms = {threshold: torch.zeros(threshold, dtype=torch.float64, device=device) for threshold in thresholds}

    pair_indices = torch.combinations(torch.arange(num_frames, device=device), 2, with_replacement=False)
    chunk_size = int(pose_config.pair_chunk_size)
    eps = float(pose_config.angle_epsilon)
    quat_floor = float(pose_config.quaternion_floor)
    invalid_error_value = float(pose_config.invalid_error_value)
    for start in range(0, num_pairs, chunk_size):
        end = min(start + chunk_size, num_pairs)
        i1 = pair_indices[start:end, 0]
        i2 = pair_indices[start:end, 1]
        relative_pose_gt = closed_form_inverse_se3(gt_se3[i1]).bmm(gt_se3[i2])
        relative_pose_pred = closed_form_inverse_se3(pred_se3[i1]).bmm(pred_se3[i2])
        max_errors = torch.maximum(
            rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3], eps, quat_floor),
            translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3], eps, invalid_error_value),
        )
        for threshold, hist in histograms.items():
            valid = (max_errors >= 0) & (max_errors <= threshold)
            if bool(valid.any().item()):
                bins = torch.floor(max_errors[valid]).to(torch.long).clamp(max=threshold - 1)
                hist += torch.bincount(bins, minlength=threshold)[:threshold].to(torch.float64)

    output = AttrDict()
    for threshold in thresholds:
        key = f"auc{threshold:02d}"
        output[key] = float(torch.cumsum(histograms[threshold] / float(num_pairs), dim=0).mean().item())
    return output


def mean_squared_nn_distance(
    reference: np.ndarray,
    query: np.ndarray,
    chunk_size: int,
    device_name: str,
    progress_desc: str | None = None,
    log_label: str = "ScanNet",
) -> float:
    if len(reference) == 0 or len(query) == 0:
        return float("inf")

    device = o3d.core.Device(device_name)
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    if progress_desc is not None:
        tqdm.write(
            f"[{log_label}] Open3D CUDA NN index start | {progress_desc} | "
            f"ref={len(reference):,} query={len(query):,}"
        )
    start_tree = time.perf_counter()
    ref_tensor = o3d.core.Tensor(reference, dtype=o3d.core.Dtype.Float32, device=device)
    nns = o3d.core.nns.NearestNeighborSearch(ref_tensor)
    nns.knn_index()
    if progress_desc is not None:
        tqdm.write(f"[{log_label}] Open3D CUDA NN index done  | {progress_desc} | {time.perf_counter() - start_tree:.2f}s")

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


def evaluate_3d_reconstruction_l2(
    pcd_pred: o3d.geometry.PointCloud | np.ndarray,
    pcd_trgt: o3d.geometry.PointCloud | np.ndarray,
    geometry_config: AttrDict,
    progress_desc: str | None = None,
    log_label: str = "ScanNet",
) -> dict[str, float]:
    if isinstance(pcd_pred, np.ndarray):
        pcd_pred = _point_cloud_from_arrays(pcd_pred)
    if isinstance(pcd_trgt, np.ndarray):
        pcd_trgt = _point_cloud_from_arrays(pcd_trgt)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)
    if len(verts_pred) == 0 or len(verts_trgt) == 0:
        return {"acc": float("inf"), "comp": float("inf"), "overall": float("inf")}

    chunk_size = int(geometry_config.nn_chunk_size)
    device_name = str(geometry_config.nn_device)
    accuracy = mean_squared_nn_distance(
        verts_trgt,
        verts_pred,
        chunk_size=chunk_size,
        device_name=device_name,
        progress_desc=None if progress_desc is None else f"{progress_desc} pred->gt",
        log_label=log_label,
    )
    completeness = mean_squared_nn_distance(
        verts_pred,
        verts_trgt,
        chunk_size=chunk_size,
        device_name=device_name,
        progress_desc=None if progress_desc is None else f"{progress_desc} gt->pred",
        log_label=log_label,
    )
    return {"acc": accuracy, "comp": completeness, "overall": (accuracy + completeness) / 2}


def sample_points_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
) -> o3d.geometry.PointCloud:
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    if pcd.has_colors():
        pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(pcd.colors), 0.0, 1.0))
    return pcd


def _image_psnr(pred: np.ndarray, target: np.ndarray, max_value: float) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(max_value / np.sqrt(mse)))


def _image_ssim(pred: np.ndarray, target: np.ndarray, image_config: AttrDict) -> float:
    c1 = float(image_config.ssim_k1) ** 2
    c2 = float(image_config.ssim_k2) ** 2
    window = int(image_config.ssim_window_size)
    kernel = (window, window)
    sigma = float(image_config.ssim_sigma)
    denominator_epsilon = float(image_config.ssim_denominator_epsilon)
    scores = []
    for channel in range(3):
        x = pred[:, :, channel].astype(np.float32, copy=False)
        y = target[:, :, channel].astype(np.float32, copy=False)
        mu_x = cv2.GaussianBlur(x, kernel, sigma)
        mu_y = cv2.GaussianBlur(y, kernel, sigma)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = cv2.GaussianBlur(x * x, kernel, sigma) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, kernel, sigma) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, kernel, sigma) - mu_xy
        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        scores.append(float(np.mean(numerator / np.maximum(denominator, denominator_epsilon))))
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
    near_plane_m: float,
    splat_radius_scale: float,
    z_tie_epsilon: float,
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
        valid = z > near_plane_m
        if not bool(valid.any().item()):
            continue
        cam = cam[valid]
        z = z[valid]
        cols = colors[start:end][valid]
        u = torch.round(fx * cam[:, 0] / z + cx).to(torch.int64)
        v = torch.round(fy * cam[:, 1] / z + cy).to(torch.int64)
        radii = torch.ceil(splat_radius_scale * voxel_size * max_focal / z).to(torch.int64).clamp_(0, radius_cap)
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
            winners = torch.abs(pix_z - zbuf[pix]) <= z_tie_epsilon
            if bool(winners.any().item()):
                image[pix[winners]] = cols0[rows[winners]]
    return image.reshape(height, width, 3).cpu().numpy()


class ScanNetDataset:
    DATASET_LABEL = "ScanNet"

    def __init__(self, input_root: Path, raw_root: Path, eval_config: AttrDict):
        self.data_root = Path(input_root)
        self.raw_root = Path(raw_root)
        self.config = eval_config
        self.SCENES = self._load_scene_list(self.data_root)
        self._scene_cache: dict[str, AttrDict] = {}

    def _load_scene_list(self, input_root: Path) -> list[str]:
        return _load_scene_list(input_root)

    def _gt_mesh_path(self, scene: str) -> Path:
        dataset_config = self.config.dataset
        raw_scene_dir = self.raw_root / scene
        gt_mesh_path = raw_scene_dir / f"{scene}{dataset_config.gt_mesh_suffix}"
        if gt_mesh_path.is_file():
            return gt_mesh_path
        fallback = raw_scene_dir / f"{scene}{dataset_config.gt_mesh_fallback_suffix}"
        if fallback.is_file():
            return fallback
        raise FileNotFoundError(f"Missing {self.DATASET_LABEL} GT mesh: {gt_mesh_path} (and fallback {fallback})")

    def _load_gt_mesh(self, mesh_path: str | Path) -> o3d.geometry.TriangleMesh:
        return o3d.io.read_triangle_mesh(str(mesh_path))

    def get_data(self, scene: str) -> AttrDict:
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        scene_dir = self.data_root / scene
        intrinsic_path = scene_dir / "intrinsic" / "intrinsic_color.json"
        if not intrinsic_path.is_file():
            raise FileNotFoundError(f"Missing {self.DATASET_LABEL} intrinsic file: {intrinsic_path}")

        gt_mesh_path = self._gt_mesh_path(scene)

        ixt_shared = intrinsic_matrix(read_pinhole_intrinsics(intrinsic_path))
        frames = read_scene_frames(scene_dir, require_pose=True)
        out = AttrDict(image_files=[], extrinsics=[], intrinsics=[], aux=AttrDict(gt_mesh_path=str(gt_mesh_path)))
        for frame in tqdm(frames, desc=f"[{self.DATASET_LABEL}] {scene} load poses", leave=False):
            img_path = scene_dir / frame["color_file"]
            pose_path = scene_dir / frame["pose_file"]
            if not img_path.is_file():
                raise FileNotFoundError(f"Missing {self.DATASET_LABEL} color frame: {img_path}")
            if not pose_path.is_file():
                raise FileNotFoundError(f"Missing {self.DATASET_LABEL} pose file: {pose_path}")
            c2w = np.loadtxt(pose_path, dtype=np.float32)
            if c2w.shape != (4, 4):
                raise ValueError(f"Expected 4x4 {self.DATASET_LABEL} pose matrix: {pose_path}")
            out.image_files.append(str(img_path))
            out.extrinsics.append(np.linalg.inv(c2w).astype(np.float32))
            out.intrinsics.append(ixt_shared.copy())

        out.extrinsics = np.asarray(out.extrinsics, dtype=np.float32)
        out.intrinsics = np.asarray(out.intrinsics, dtype=np.float32)
        self._scene_cache[scene] = out
        tqdm.write(f"[{self.DATASET_LABEL}] {scene}: {len(out.image_files)} images")
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
        return Path(result_path).parents[2] / "eval_cache" / self.config.outputs.aligned_ply_filename

    def _tsdf_alignment_is_current(self, aligned_path: Path, source_path: str, result_path: str, manifest: dict) -> bool:
        if not aligned_path.exists():
            return False
        input_paths = [source_path, result_path, manifest["pose_path"]]
        gt_meta_path = Path(result_path).with_name("gt_meta.npz")
        if gt_meta_path.exists():
            input_paths.append(str(gt_meta_path))
        newest_input = max(Path(path).stat().st_mtime_ns for path in input_paths)
        return aligned_path.stat().st_mtime_ns >= newest_input

    def _align_tsdf_pcd(self, scene: str, result_path: str) -> tuple[str, float]:
        manifest = self._load_vipe_manifest(result_path)
        source_path = self._tsdf_pcd_path(result_path)
        aligned_path = self._aligned_tsdf_pcd_path(result_path)

        gt_meta = self._load_gt_meta(result_path)
        if gt_meta is None:
            gt_meta = self.get_data(scene)
        frame_indices = [int(idx) for idx in manifest["frame_indices"]]
        if len(gt_meta.extrinsics) != len(frame_indices):
            raise ValueError(
                f"GT metadata length ({len(gt_meta.extrinsics)}) does not match manifest frame count ({len(frame_indices)})"
            )

        pose_map = self._load_vipe_pose_map(manifest)
        pred_c2w = np.stack([as_homogeneous(pose_map[idx]).astype(np.float64) for idx in frame_indices])
        pred_c2w_valid, gt_w2c_valid = _filter_valid_poses(pred_c2w, gt_meta.extrinsics, f"{self.DATASET_LABEL} {scene} align")
        se3, gt_c2w_valid = _first_camera_se3(pred_c2w_valid, gt_w2c_valid)
        scale_diagnostic = _first_camera_scale_diagnostic(pred_c2w_valid, gt_c2w_valid, se3)
        if self._tsdf_alignment_is_current(aligned_path, source_path, result_path, manifest):
            tqdm.write(
                f"[{self.DATASET_LABEL}] aligned TSDF cache hit | {scene} | "
                f"scale_diagnostic={scale_diagnostic:.6f} | {aligned_path}"
            )
            return str(aligned_path), scale_diagnostic

        start = time.perf_counter()
        tqdm.write(
            f"[{self.DATASET_LABEL}] align TSDF start | {scene} | first-camera SE3 | "
            f"scale_diagnostic={scale_diagnostic:.6f} | {source_path}"
        )
        pcd = o3d.io.read_point_cloud(source_path)
        points = np.asarray(pcd.points)
        if len(points) > 0:
            pcd.points = o3d.utility.Vector3dVector(_apply_se3(points, se3))
        if pcd.has_normals():
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) @ se3[:3, :3].T)

        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = aligned_path.with_name(f"{aligned_path.stem}.tmp{aligned_path.suffix}")
        o3d.io.write_point_cloud(str(tmp_path), pcd)
        os.replace(tmp_path, aligned_path)
        tqdm.write(
            f"[{self.DATASET_LABEL}] align TSDF done  | {scene} | "
            f"points={len(pcd.points):,} | {time.perf_counter() - start:.2f}s"
        )
        return str(aligned_path), scale_diagnostic

    def _artifact_sample_count(self, result_path: str) -> int:
        manifest = self._load_vipe_manifest(result_path)
        return int(manifest["output"]["pcd_max_points"])

    def eval3d(self, scene: str, result_path: str) -> dict[str, float]:
        start_eval = time.perf_counter()
        full_gt_data = self.get_data(scene)
        pcd_path, scale_diagnostic = self._align_tsdf_pcd(scene, result_path)
        method_name = str(self.config.outputs.reconstruction_method_name)
        eval_cache_dir = Path(result_path).parents[2] / "eval_cache"
        tqdm.write(f"[{self.DATASET_LABEL}] eval3d start | {scene} | {pcd_path}")
        sample_count = self._artifact_sample_count(result_path)
        gt_cache_path = eval_cache_dir / f"gt_sample_{sample_count}.npz"
        start_gt = time.perf_counter()
        gt_pcd = _load_cached_point_cloud(gt_cache_path)
        if gt_pcd is None:
            tqdm.write(f"[{self.DATASET_LABEL}] sample GT start | {scene} | points={sample_count:,}")
            gt_mesh = self._load_gt_mesh(full_gt_data.aux.gt_mesh_path)
            gt_pcd = sample_points_from_mesh(gt_mesh, sample_count)
            _write_cached_point_cloud(gt_cache_path, gt_pcd)
            tqdm.write(f"[{self.DATASET_LABEL}] sample GT done  | {scene} | {time.perf_counter() - start_gt:.2f}s")
        else:
            tqdm.write(
                f"[{self.DATASET_LABEL}] sample GT cache hit | {scene} | "
                f"points={len(gt_pcd.points):,} | {time.perf_counter() - start_gt:.2f}s"
            )

        start_read = time.perf_counter()
        pred_pcd = o3d.io.read_point_cloud(pcd_path)
        tqdm.write(
            f"[{self.DATASET_LABEL}] read pred done | {scene} | "
            f"points={len(pred_pcd.points):,} | {time.perf_counter() - start_read:.2f}s"
        )
        pred_eval_pcd = pred_pcd

        start_crop = time.perf_counter()
        aabb = gt_pcd.get_axis_aligned_bounding_box()
        points = np.asarray(pred_pcd.points)
        padding = float(self.config.geometry.gt_aabb_padding_m)
        if points.size > 0:
            inside_mask = (
                (points[:, 0] >= aabb.min_bound[0] - padding)
                & (points[:, 0] <= aabb.max_bound[0] + padding)
                & (points[:, 1] >= aabb.min_bound[1] - padding)
                & (points[:, 1] <= aabb.max_bound[1] + padding)
                & (points[:, 2] >= aabb.min_bound[2] - padding)
                & (points[:, 2] <= aabb.max_bound[2] + padding)
            )
            pred_eval_pcd = pred_pcd.select_by_index(np.nonzero(inside_mask)[0])
        tqdm.write(
            f"[{self.DATASET_LABEL}] AABB crop done | {scene} | "
            f"eval_points={len(pred_eval_pcd.points):,}/{len(pred_pcd.points):,} | {time.perf_counter() - start_crop:.2f}s"
        )

        start_geom = time.perf_counter()
        result = evaluate_3d_reconstruction_l2(
            pred_eval_pcd,
            gt_pcd,
            self.config.geometry,
            progress_desc=f"{scene} {method_name} L2 NN",
            log_label=self.DATASET_LABEL,
        )
        result["scale_diagnostic"] = scale_diagnostic
        tqdm.write(f"[{self.DATASET_LABEL}] geometry metric done | {scene} | {result} | {time.perf_counter() - start_geom:.2f}s")
        del gt_pcd, pred_eval_pcd, points
        gc.collect()

        render_data = self._load_gt_meta(result_path) or full_gt_data
        result.update(self._compute_render_metrics(scene, method_name, pred_pcd, render_data, pcd_path, eval_cache_dir))
        tqdm.write(f"[{self.DATASET_LABEL}] eval3d done | {scene} | {method_name} | {time.perf_counter() - start_eval:.2f}s")
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
        render_config = self.config.render
        image_config = self.config.image_metrics
        voxel_size = float(render_config.voxel_size_m)
        tqdm.write(f"[{self.DATASET_LABEL}] render metric voxelize start | {label} | voxel={voxel_size}m")
        cache_name = f"{method_name}_render_vox_{str(voxel_size).replace('.', 'p')}.npz"
        render_cache_path = cache_dir / cache_name
        cached = _load_cached_render_arrays(render_cache_path, source_path, voxel_size)
        if cached is None:
            points, colors = _voxelized_render_arrays(pcd, voxel_size)
            _write_cached_render_arrays(render_cache_path, source_path, voxel_size, points, colors)
            tqdm.write(
                f"[{self.DATASET_LABEL}] render metric voxelize done  | {label} | "
                f"voxels={len(points):,} | {time.perf_counter() - start_total:.2f}s"
            )
        else:
            points, colors = cached
            tqdm.write(
                f"[{self.DATASET_LABEL}] render metric voxelize cache hit | {label} | "
                f"voxels={len(points):,} | {time.perf_counter() - start_total:.2f}s"
            )

        valid_pose_mask = _valid_pose_mask(scene_data.extrinsics)
        dropped = int((~valid_pose_mask).sum())
        if dropped:
            tqdm.write(
                f"[WARN] {self.DATASET_LABEL} {scene} render: "
                f"dropping {dropped:,}/{len(valid_pose_mask):,} frames with non-finite GT poses"
            )
        frame_indices = np.nonzero(valid_pose_mask)[0]
        if len(frame_indices) == 0:
            raise ValueError(f"{self.DATASET_LABEL} {scene} render: no finite GT poses")
        num_images = int(render_config.num_images)
        if num_images != -1:
            sample_count = min(num_images, len(frame_indices))
            rng = np.random.default_rng(seed=int(render_config.frame_sample_seed))
            frame_indices = np.sort(rng.choice(frame_indices, size=sample_count, replace=False))

        tasks = [
            (int(idx), str(scene_data.image_files[idx]), scene_data.extrinsics[idx], scene_data.intrinsics[idx])
            for idx in frame_indices
        ]
        render_device = str(render_config.device)
        tqdm.write(
            f"[{self.DATASET_LABEL}] render metric project start | {label} | "
            f"frames={len(tasks):,}/{len(scene_data.image_files):,} device={render_device}"
        )
        device = torch.device(render_device)
        points_t = torch.as_tensor(np.ascontiguousarray(points), dtype=torch.float32, device=device)
        colors_t = torch.as_tensor(np.ascontiguousarray(colors), dtype=torch.float32, device=device)
        radius_cap = int(render_config.radius_cap_px)
        offset_shells = _make_offset_shells(radius_cap, device)
        psnr_values = []
        ssim_values = []
        progress = tqdm(total=len(tasks), desc=f"{label} render PSNR/SSIM", unit="frame", leave=False)
        for _, image_file, extrinsic, intrinsic in tasks:
            bgr = cv2.imread(image_file, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(image_file)
            target = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / float(image_config.rgb_normalizer)
            height, width = target.shape[:2]
            render = _render_voxel_cloud_cuda(
                points_t,
                colors_t,
                extrinsic,
                intrinsic,
                height,
                width,
                voxel_size,
                int(render_config.chunk_size),
                radius_cap,
                float(render_config.near_plane_m),
                float(render_config.splat_radius_scale),
                float(render_config.z_tie_epsilon),
                offset_shells,
            )
            psnr_values.append(_image_psnr(render, target, float(image_config.psnr_max_value)))
            ssim_values.append(_image_ssim(render, target, image_config))
            progress.update()
            progress.set_postfix(psnr=f"{np.mean(psnr_values):.2f}", ssim=f"{np.mean(ssim_values):.4f}")
        del points_t, colors_t, offset_shells
        torch.cuda.empty_cache()
        progress.close()
        tqdm.write(f"[{self.DATASET_LABEL}] render metric project done  | {label} | {time.perf_counter() - start_total:.2f}s")
        return {"psnr": float(np.mean(psnr_values)), "ssim": float(np.mean(ssim_values))}


class ScanNetEvaluator:
    DATASET_KEY = "scannet"
    DATASET_LABEL = "ScanNet"

    def __init__(
        self,
        work_dir: str | Path,
        input_root: str | Path,
        raw_root: str | Path,
        eval_config: dict[str, Any] | AttrDict,
        scenes: list[str] | None = None,
    ):
        self.work_dir = str(work_dir)
        self.config = _to_attr_dict(eval_config)
        self.scenes_filter = scenes
        self.datasets = self._build_datasets(Path(input_root), Path(raw_root))
        self._metric_eval_seconds_by_scene: dict[str, float] = {}
        self._metric_eval_frames_by_scene: dict[str, int] = {}
        os.makedirs(self.work_dir, exist_ok=True)

    def _build_datasets(self, input_root: Path, raw_root: Path) -> AttrDict:
        return AttrDict(scannet=ScanNetDataset(input_root, raw_root, self.config))

    def _get_scenes(self, dataset: ScanNetDataset) -> list[str]:
        if self.scenes_filter is not None:
            return [scene for scene in dataset.SCENES if scene in self.scenes_filter]
        return list(dataset.SCENES)

    def eval(self, dump: bool = True) -> dict[str, dict]:
        self._metric_eval_seconds_by_scene = {}
        self._metric_eval_frames_by_scene = {}
        summary: dict[str, dict] = {}
        print(f"\n{'=' * 60}")
        print(f"Evaluating POSE for {self.DATASET_LABEL}")
        print(f"{'=' * 60}")
        for data, result in self._eval_pose(dump=dump):
            summary[f"{data}_pose"] = result
        print(f"\n{'=' * 60}")
        print(f"Evaluating RECON for {self.DATASET_LABEL}")
        print(f"{'=' * 60}")
        for key, result in self._eval_reconstruction(dump=dump):
            summary[key] = result
        return summary

    def _eval_pose(self, dump: bool = True):
        os.makedirs(self._metric_dir, exist_ok=True)
        dataset = self.datasets[self.DATASET_KEY]
        dataset_results = AttrDict()
        for scene in tqdm(self._get_scenes(dataset), desc=f"{self.DATASET_KEY} scenes", leave=False):
            start_scene = time.perf_counter()
            export_dir = self._export_dir(self.DATASET_KEY, scene)
            result_path = dataset.result_path(export_dir)
            if not dataset.result_exists(result_path):
                tqdm.write(f"[ERROR] Result file not found: {result_path}")
                continue
            gt_meta = self._load_gt_meta(export_dir)
            if gt_meta is not None:
                num_frames = len(gt_meta["extrinsics"])
                num_pairs = num_frames * (num_frames - 1) // 2
                tqdm.write(f"[INFO] Pose start | {self.DATASET_KEY} | {scene} | frames={num_frames} pairs={num_pairs}")
                result = self._compute_pose_with_gt(dataset, result_path, gt_meta)
            else:
                scene_data = dataset.get_data(scene)
                num_frames = len(scene_data.image_files)
                num_pairs = num_frames * (num_frames - 1) // 2
                tqdm.write(f"[INFO] Pose start | {self.DATASET_KEY} | {scene} | frames={num_frames} pairs={num_pairs}")
                result = self._compute_pose_with_gt(dataset, result_path, scene_data)
            self._metric_eval_frames_by_scene[scene] = int(num_frames)
            self._metric_eval_seconds_by_scene[scene] = (
                self._metric_eval_seconds_by_scene.get(scene, 0.0) + time.perf_counter() - start_scene
            )
            dataset_results[scene] = self._to_float_dict(result)
            tqdm.write(f"[INFO] Pose done  | {self.DATASET_KEY} | {scene} | {result}")

        if dataset_results:
            dataset_results["mean"] = self._mean_of_dicts(dataset_results.values())
        if dump:
            self._dump_json(os.path.join(self._metric_dir, f"{self.DATASET_KEY}_pose.json"), dataset_results)
        yield self.DATASET_KEY, dataset_results

    def _eval_reconstruction(self, dump: bool = True):
        os.makedirs(self._metric_dir, exist_ok=True)
        dataset = self.datasets[self.DATASET_KEY]
        scenes = self._get_scenes(dataset)
        dataset_results = AttrDict()
        tqdm.write(f"[INFO] Starting recon eval for dataset={self.DATASET_KEY} with {len(scenes)} scene(s)")
        for scene in scenes:
            start_scene = time.perf_counter()
            export_dir = self._export_dir(self.DATASET_KEY, scene)
            result_path = dataset.result_path(export_dir)
            result = dataset.eval3d(scene, result_path)
            self._metric_eval_seconds_by_scene[scene] = (
                self._metric_eval_seconds_by_scene.get(scene, 0.0) + time.perf_counter() - start_scene
            )
            dataset_results[scene] = self._to_float_dict(result)
            tqdm.write(f"  recon | {self.DATASET_KEY} | {scene}: {result}")

        dataset_results["mean"] = self._mean_of_dicts(dataset_results.values())
        key = f"{self.DATASET_KEY}_recon"
        if dump:
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
        pred_extrinsics, gt_extrinsics = _filter_valid_poses(
            as_homogeneous(pred_extrinsics),
            as_homogeneous(gt_meta["extrinsics"]),
            f"{self.DATASET_LABEL} pose",
        )
        return compute_pose(
            torch.from_numpy(pred_extrinsics),
            torch.from_numpy(gt_extrinsics),
            self.config.pose,
        )

    def metric_eval_timing(self) -> dict[str, dict[str, float]]:
        timing = {}
        for scene, seconds in self._metric_eval_seconds_by_scene.items():
            frames = self._metric_eval_frames_by_scene.get(scene)
            if frames is None:
                continue
            timing[scene] = {"frames": int(frames), "seconds": float(seconds)}
        return timing

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
