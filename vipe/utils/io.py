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

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

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


@dataclass(slots=True)
class ArtifactFrame:
    frame_idx: int
    color: np.ndarray
    depth: np.ndarray
    pose_matrix: np.ndarray
    w2c_matrix: np.ndarray
    intrinsics: np.ndarray


def _prefetch_frames(
    load_frame: Callable[[int], ArtifactFrame],
    n_frames: int,
    max_prefetch: int = 16,
    num_workers: int = 4,
) -> Iterator[ArtifactFrame]:
    def load(frame_idx: int) -> ArtifactFrame:
        return load_frame(frame_idx)

    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="vipe-artifact-load") as pool:
        next_submit = 0
        pending: dict[int, Future[ArtifactFrame]] = {}

        def submit_ready() -> None:
            nonlocal next_submit
            while next_submit < n_frames and len(pending) < max_prefetch:
                pending[next_submit] = pool.submit(load, next_submit)
                next_submit += 1

        submit_ready()
        for frame_idx in range(n_frames):
            frame_data = pending.pop(frame_idx).result()
            submit_ready()
            yield frame_data


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


def _integrate_tsdf_frame(
    volume,
    frame_data: ArtifactFrame,
    depth_trunc: float,
) -> None:
    depth = frame_data.depth
    color = frame_data.color
    intrinsics = frame_data.intrinsics

    volume.integrate(depth, color, intrinsics, frame_data.w2c_matrix, depth_trunc)


def _write_tsdf_pcd(out_path: ArtifactPath, volume, max_points: int) -> None:
    points, colors, normals = volume.extract_point_cloud_tensors(max_points)
    if len(points) == 0:
        return

    write_binary_ply(out_path.tsdf_pcd_path, points, colors, normals)


def save_artifacts(
    out_path: ArtifactPath,
    load_frame: Callable[[int], ArtifactFrame],
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

    frame_iter = _prefetch_frames(load_frame, n_frames)
    for _ in pbar(range(n_frames), desc="Saving artifacts"):
        try:
            frame_data = next(frame_iter)
        except StopIteration as exc:
            raise ValueError("Final frame iterator ended before n_frames") from exc
        assert isinstance(frame_data, ArtifactFrame)

        pose_list.append((frame_data.frame_idx, frame_data.pose_matrix))

        _integrate_tsdf_frame(tsdf_volume, frame_data, pcd_tsdf_depth_trunc_m)

    if len(pose_list) > 0:
        pose_data = np.stack([pose for _, pose in pose_list], axis=0)
        pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
        out_path.pose_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(out_path.pose_path, data=pose_data, inds=pose_inds)

    _write_tsdf_pcd(out_path, tsdf_volume, max_pcd_points)
