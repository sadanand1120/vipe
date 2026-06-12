# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vipe.streams.base import FrameData
from vipe.utils.logging import pbar
from vipe.utils.tsdf import TSDFVolume, write_binary_ply


logger = logging.getLogger(__name__)


@dataclass
class ArtifactPath:
    base_path: Path
    artifact_name: str

    @property
    def pose_path(self) -> Path:
        return self.base_path / "pose" / f"{self.artifact_name}.npz"

    @property
    def tsdf_pcd_path(self) -> Path:
        return self.base_path / "pcd" / f"{self.artifact_name}_tsdf.ply"

    @property
    def slam_debug_path(self) -> Path:
        return self.base_path / "debug" / f"{self.artifact_name}_slam_debug.npz"


def _image_valid_numpy(frame_data: FrameData, shape: tuple[int, int]) -> np.ndarray | None:
    if frame_data.image_valid_mask is None:
        return None
    image_valid = frame_data.image_valid_mask.detach().cpu().numpy().astype(bool)
    assert image_valid.shape == shape
    return image_valid


def _make_tsdf_volume(
    voxel_edge_m: float,
    sdf_trunc_m: float,
    num_voxels_per_block_edge: int,
    depth_sampling_stride: int,
):
    return TSDFVolume(
        voxel_edge_m=voxel_edge_m,
        sdf_trunc_m=sdf_trunc_m,
        num_voxels_per_block_edge=num_voxels_per_block_edge,
        depth_sampling_stride=depth_sampling_stride,
    )


def _integrate_tsdf_frame(volume, frame_data: FrameData, depth_trunc: float) -> None:
    if frame_data.metric_depth is None or frame_data.pose is None or frame_data.intrinsics is None:
        return

    depth = frame_data.metric_depth.detach().cpu().numpy().astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0.0] = 0.0
    image_valid = _image_valid_numpy(frame_data, depth.shape)
    if image_valid is not None:
        depth[~image_valid] = 0.0

    color = (frame_data.rgb.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    fx, fy, cx, cy = frame_data.intrinsics[:4].detach().cpu().numpy().astype(np.float32)
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
    w2c = frame_data.pose.inv().matrix().detach().cpu().numpy().astype(np.float32)
    volume.integrate(depth, color, intrinsics, w2c, depth_trunc)


def _write_tsdf_pcd(out_path: ArtifactPath, volume, max_points: int) -> None:
    points, colors, normals = volume.extract_point_cloud(max_points)
    if len(points) == 0:
        return

    write_binary_ply(out_path.tsdf_pcd_path, points, colors, normals)


def save_slam_debug(out_path: ArtifactPath, debug: dict[str, np.ndarray]) -> None:
    out_path.slam_debug_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(out_path.slam_debug_path, **debug)


def save_artifacts(
    out_path: ArtifactPath,
    final_frames,
    n_frames: int,
    max_pcd_points: int = 10_000_000,
    pcd_tsdf_voxel_edge_m: float = 0.02,
    pcd_tsdf_sdf_trunc_m: float = 0.15,
    pcd_tsdf_depth_trunc_m: float = 5.0,
    pcd_tsdf_num_voxels_per_block_edge: int = 16,
    pcd_tsdf_depth_sampling_stride: int = 4,
) -> None:
    """
    Save artifacts in a single streaming pass to avoid retaining the full sequence in RAM.
    """

    pose_list = []
    tsdf_volume = _make_tsdf_volume(
        pcd_tsdf_voxel_edge_m,
        pcd_tsdf_sdf_trunc_m,
        pcd_tsdf_num_voxels_per_block_edge,
        pcd_tsdf_depth_sampling_stride,
    )

    for frame_idx, frame_data in pbar(
        enumerate(final_frames),
        total=n_frames,
        desc="Saving artifacts",
    ):
        assert isinstance(frame_data, FrameData)

        if frame_data.pose is not None:
            pose_list.append((frame_idx, frame_data.pose.matrix().cpu().numpy()))

        _integrate_tsdf_frame(tsdf_volume, frame_data, pcd_tsdf_depth_trunc_m)

    if len(pose_list) > 0:
        pose_data = np.stack([pose for _, pose in pose_list], axis=0)
        pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
        out_path.pose_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(out_path.pose_path, data=pose_data, inds=pose_inds)

    _write_tsdf_pcd(out_path, tsdf_volume, max_pcd_points)
