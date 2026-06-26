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
import time

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

    @property
    def timing_path(self) -> Path:
        return self.base_path / "timing" / f"{self.artifact_name}.json"


@dataclass(slots=True)
class ArtifactFrame:
    frame_idx: int
    color: np.ndarray
    depth: np.ndarray
    pose_matrix: np.ndarray
    w2c_matrix: np.ndarray
    intrinsics: np.ndarray


def _add_timing(timing: dict[str, float | int], key: str, seconds: float) -> None:
    timing[key] = float(timing.get(key, 0.0)) + float(seconds)


def _prefetch_frames(
    load_frame: Callable[[int], ArtifactFrame],
    n_frames: int,
    max_prefetch: int = 16,
    num_workers: int = 4,
) -> Iterator[tuple[ArtifactFrame, float]]:
    def load(frame_idx: int) -> tuple[ArtifactFrame, float]:
        start = time.perf_counter()
        return load_frame(frame_idx), time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="vipe-artifact-load") as pool:
        next_submit = 0
        pending: dict[int, Future[tuple[ArtifactFrame, float]]] = {}

        def submit_ready() -> None:
            nonlocal next_submit
            while next_submit < n_frames and len(pending) < max_prefetch:
                pending[next_submit] = pool.submit(load, next_submit)
                next_submit += 1

        submit_ready()
        for frame_idx in range(n_frames):
            frame_data, load_seconds = pending.pop(frame_idx).result()
            submit_ready()
            yield frame_data, load_seconds


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
    timing: dict[str, float | int],
) -> None:
    prep_start = time.perf_counter()
    depth = frame_data.depth
    color = frame_data.color
    intrinsics = frame_data.intrinsics
    _add_timing(timing, "tsdf_prepare_s", time.perf_counter() - prep_start)

    integrate_start = time.perf_counter()
    volume.integrate(depth, color, intrinsics, frame_data.w2c_matrix, depth_trunc)
    _add_timing(timing, "tsdf_integrate_s", time.perf_counter() - integrate_start)


def _write_tsdf_pcd(out_path: ArtifactPath, volume, max_points: int, timing: dict[str, float | int]) -> None:
    extract_start = time.perf_counter()
    points, colors, normals = volume.extract_point_cloud_tensors(max_points)
    timing["tsdf_extract_s"] = time.perf_counter() - extract_start
    timing["tsdf_points"] = int(len(points))
    if len(points) == 0:
        return

    write_start = time.perf_counter()
    write_binary_ply(out_path.tsdf_pcd_path, points, colors, normals)
    timing["tsdf_ply_write_s"] = time.perf_counter() - write_start


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
) -> dict[str, float | int]:
    """
    Save artifacts in a single streaming pass to avoid retaining the full sequence in RAM.
    """

    total_start = time.perf_counter()
    timing: dict[str, float | int] = {"frames": int(n_frames)}
    pose_list = []
    tsdf_volume = _make_tsdf_volume(
        pcd_tsdf_voxel_edge_m,
        pcd_tsdf_sdf_trunc_m,
        pcd_tsdf_num_voxels_per_block_edge,
        pcd_tsdf_depth_sampling_stride,
    )

    frame_iter = _prefetch_frames(load_frame, n_frames)
    for _ in pbar(range(n_frames), desc="Saving artifacts"):
        wait_start = time.perf_counter()
        try:
            frame_data, load_seconds = next(frame_iter)
        except StopIteration as exc:
            raise ValueError("Final frame iterator ended before n_frames") from exc
        assert isinstance(frame_data, ArtifactFrame)
        _add_timing(timing, "frame_prefetch_wait_s", time.perf_counter() - wait_start)
        _add_timing(timing, "frame_load_attach_s", load_seconds)

        pose_list.append((frame_data.frame_idx, frame_data.pose_matrix))

        _integrate_tsdf_frame(tsdf_volume, frame_data, pcd_tsdf_depth_trunc_m, timing)

    if len(pose_list) > 0:
        pose_write_start = time.perf_counter()
        pose_data = np.stack([pose for _, pose in pose_list], axis=0)
        pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
        out_path.pose_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(out_path.pose_path, data=pose_data, inds=pose_inds)
        timing["pose_npz_write_s"] = time.perf_counter() - pose_write_start

    _write_tsdf_pcd(out_path, tsdf_volume, max_pcd_points, timing)
    timing["total_s"] = time.perf_counter() - total_start
    return timing
