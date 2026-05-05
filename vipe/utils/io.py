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

import json
import logging
import math
import shutil
import tempfile
import zipfile

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

from vipe.streams.base import FrameData
from vipe.utils.logging import pbar


logger = logging.getLogger(__name__)


@dataclass
class ArtifactPath:
    base_path: Path
    artifact_name: str

    @property
    def pose_path(self) -> Path:
        return self.base_path / "pose" / f"{self.artifact_name}.npz"

    @property
    def depth_path(self) -> Path:
        return self.base_path / "depth" / f"{self.artifact_name}.zip"

    @property
    def backproject_pcd_path(self) -> Path:
        return self.base_path / "pcd" / f"{self.artifact_name}_backproject.ply"

    @property
    def tsdf_pcd_path(self) -> Path:
        return self.base_path / "pcd" / f"{self.artifact_name}_tsdf.ply"

    @property
    def intrinsics_path(self) -> Path:
        return self.base_path / "intrinsics" / f"{self.artifact_name}.json"


def _backproject_vertices(
    frame_data: FrameData,
    max_points_per_frame: int,
    conf_threshold_coef: float,
    sample_ratio: float,
) -> np.ndarray | None:
    if (
        frame_data.metric_depth is None
        or frame_data.pose is None
        or frame_data.intrinsics is None
        or max_points_per_frame <= 0
    ):
        return None

    depth = frame_data.metric_depth.detach().cpu().numpy()
    valid = np.isfinite(depth) & (depth > 0.0)
    confidence = None
    if frame_data.depth_confidence is not None:
        confidence = frame_data.depth_confidence.detach().cpu().numpy()
        assert confidence.shape == depth.shape
        valid &= (confidence >= float(np.mean(confidence)) * conf_threshold_coef) & (confidence > 1e-5)

    valid_flat = np.flatnonzero(valid.ravel())
    if len(valid_flat) == 0:
        return None

    if confidence is not None and sample_ratio < 1.0:
        sample_count = int(len(valid_flat) * sample_ratio)
    else:
        sample_count = len(valid_flat)
    sample_count = min(sample_count, max_points_per_frame)
    if sample_count <= 0:
        return None

    if sample_count < len(valid_flat):
        if confidence is not None:
            rng = np.random.default_rng(frame_data.raw_frame_idx)
            valid_flat = rng.choice(valid_flat, sample_count, replace=False)
        else:
            stride = max(1, math.ceil(len(valid_flat) / sample_count))
            valid_flat = valid_flat[::stride][:sample_count]

    height, width = depth.shape
    ys, xs = np.divmod(valid_flat, width)
    zs = depth.ravel()[valid_flat].astype(np.float32)
    fx, fy, cx, cy = frame_data.intrinsics[:4].detach().cpu().numpy().astype(np.float32)

    points_cam = np.empty((len(valid_flat), 4), dtype=np.float32)
    points_cam[:, 0] = (xs.astype(np.float32) - cx) * zs / fx
    points_cam[:, 1] = (ys.astype(np.float32) - cy) * zs / fy
    points_cam[:, 2] = zs
    points_cam[:, 3] = 1.0

    pose_c2w = frame_data.pose.matrix().detach().cpu().numpy().astype(np.float32)
    points_world = (pose_c2w @ points_cam.T).T[:, :3]
    colors = (frame_data.rgb.detach().cpu().numpy().reshape(-1, 3)[valid_flat] * 255.0).clip(0, 255).astype(np.uint8)

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(len(points_world), dtype=vertex_dtype)
    vertices["x"] = points_world[:, 0]
    vertices["y"] = points_world[:, 1]
    vertices["z"] = points_world[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    return vertices


def _write_backproject_pcd(out_path: ArtifactPath, body_file, vertex_count: int) -> None:
    if vertex_count == 0:
        return

    out_path.backproject_pcd_path.parent.mkdir(exist_ok=True, parents=True)
    body_file.seek(0)
    with out_path.backproject_pcd_path.open("wb") as ply_file:
        ply_file.write(
            (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {vertex_count}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            ).encode("ascii")
        )
        shutil.copyfileobj(body_file, ply_file)


def _make_tsdf_volume(voxel_length: float, sdf_trunc: float):
    import open3d as o3d

    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def _integrate_tsdf_frame(volume, frame_data: FrameData, depth_trunc: float) -> None:
    if frame_data.metric_depth is None or frame_data.pose is None or frame_data.intrinsics is None:
        return

    import open3d as o3d

    depth = frame_data.metric_depth.detach().cpu().numpy().astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0.0] = 0.0

    color = (frame_data.rgb.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    height, width = depth.shape
    fx, fy, cx, cy = frame_data.intrinsics[:4].detach().cpu().numpy().astype(np.float32)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(fx),
        float(fy),
        float(cx),
        float(cy),
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    w2c = frame_data.pose.inv().matrix().detach().cpu().numpy().astype(np.float64)
    volume.integrate(rgbd, intrinsics, w2c)


def _write_tsdf_pcd(out_path: ArtifactPath, volume, max_points: int) -> None:
    import open3d as o3d

    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return

    out_path.tsdf_pcd_path.parent.mkdir(exist_ok=True, parents=True)
    pcd = mesh.sample_points_uniformly(number_of_points=max_points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    o3d.io.write_point_cloud(str(out_path.tsdf_pcd_path), pcd, write_ascii=False)


def _write_intrinsics_json(out_path: ArtifactPath, intrinsics: np.ndarray, frame_size: tuple[int, int]) -> None:
    height, width = frame_size
    fx, fy, cx, cy = intrinsics[:4]
    out_path.intrinsics_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.intrinsics_path.write_text(
        json.dumps(
            {
                "camera_model": "pinhole",
                "width": int(width),
                "height": int(height),
                "params": [float(fx), float(fy), float(cx), float(cy)],
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_depth_frame(depth_zip: zipfile.ZipFile, frame_idx: int, frame_data: FrameData) -> None:
    if frame_data.metric_depth is None:
        raise ValueError(f"Frame {frame_idx} is missing metric depth")

    depth = frame_data.metric_depth.detach().cpu().numpy().astype(np.float16)
    buffer = BytesIO()
    np.save(buffer, depth, allow_pickle=False)
    depth_zip.writestr(f"{frame_idx:06d}.npy", buffer.getvalue())


def save_artifacts(
    out_path: ArtifactPath,
    final_frames,
    n_frames: int,
    pcd_fusion_mode: str = "both",
    max_pcd_points: int = 8_000_000,
    pcd_conf_threshold_coef: float = 0.75,
    pcd_sample_ratio: float = 0.015,
    pcd_tsdf_voxel_length: float = 0.02,
    pcd_tsdf_sdf_trunc: float = 0.15,
    pcd_tsdf_depth_trunc: float = 5.0,
) -> None:
    """
    Save artifacts in a single streaming pass to avoid retaining the full sequence in RAM.
    """

    pose_list = []
    intrinsics = None
    intrinsics_frame_size = None
    if pcd_fusion_mode not in {"backproject", "tsdf", "both"}:
        raise ValueError(f"Invalid pcd_fusion_mode: {pcd_fusion_mode}")

    write_backproject = pcd_fusion_mode in {"backproject", "both"}
    write_tsdf = pcd_fusion_mode in {"tsdf", "both"}
    pcd_body_file = tempfile.TemporaryFile() if write_backproject else None
    pcd_vertex_count = 0
    max_points_per_frame = math.ceil(max_pcd_points / max(n_frames, 1))
    tsdf_volume = _make_tsdf_volume(pcd_tsdf_voxel_length, pcd_tsdf_sdf_trunc) if write_tsdf else None

    try:
        out_path.depth_path.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(out_path.depth_path, "w", compression=zipfile.ZIP_STORED) as depth_zip:
            for frame_idx, frame_data in pbar(
                enumerate(final_frames),
                total=n_frames,
                desc="Saving artifacts",
            ):
                assert isinstance(frame_data, FrameData)
                _write_depth_frame(depth_zip, frame_idx, frame_data)

                if frame_data.pose is not None:
                    pose_list.append((frame_idx, frame_data.pose.matrix().cpu().numpy()))

                if intrinsics is None and frame_data.intrinsics is not None:
                    intrinsics = frame_data.intrinsics.cpu().numpy()
                    intrinsics_frame_size = frame_data.size()

                remaining_points = max_pcd_points - pcd_vertex_count
                if write_backproject and remaining_points > 0:
                    assert pcd_body_file is not None
                    vertices = _backproject_vertices(
                        frame_data,
                        min(max_points_per_frame, remaining_points),
                        pcd_conf_threshold_coef,
                        pcd_sample_ratio,
                    )
                    if vertices is not None:
                        vertices.tofile(pcd_body_file)
                        pcd_vertex_count += len(vertices)
                if write_tsdf:
                    _integrate_tsdf_frame(tsdf_volume, frame_data, pcd_tsdf_depth_trunc)
    except Exception:
        if pcd_body_file is not None:
            pcd_body_file.close()
        raise

    if len(pose_list) > 0:
        pose_data = np.stack([pose for _, pose in pose_list], axis=0)
        pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
        out_path.pose_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(out_path.pose_path, data=pose_data, inds=pose_inds)

    if intrinsics is not None:
        assert intrinsics_frame_size is not None
        _write_intrinsics_json(out_path, intrinsics, intrinsics_frame_size)

    try:
        if write_backproject:
            assert pcd_body_file is not None
            _write_backproject_pcd(out_path, pcd_body_file, pcd_vertex_count)
        if write_tsdf:
            _write_tsdf_pcd(out_path, tsdf_volume, max_pcd_points)
    finally:
        if pcd_body_file is not None:
            pcd_body_file.close()
