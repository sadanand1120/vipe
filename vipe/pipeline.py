# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from pathlib import Path

import torch

from vipe.slam.interface import SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.streams.base import FrameStream
from vipe.utils import io
from vipe.utils.cameras import CameraType


logger = logging.getLogger(__name__)


class VipePipeline:
    def __init__(self, slam, output, output_dir: str | Path) -> None:
        self.slam_cfg = slam
        self.out_cfg = output
        self.out_path = Path(output_dir)
        self.out_path.mkdir(exist_ok=True, parents=True)

    def _initialize(self, frame_stream: FrameStream) -> tuple[FrameStream, torch.Tensor]:
        camera = frame_stream.sensor_camera()
        if camera is None:
            raise ValueError("Input stream must provide external RGB/color intrinsics")

        intrinsics = camera.pinhole_intrinsics()
        logger.info(
            "Using loaded pinhole intrinsics from %s: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            camera.source_path,
            intrinsics[0].item(),
            intrinsics[1].item(),
            intrinsics[2].item(),
            intrinsics[3].item(),
        )
        return frame_stream, intrinsics

    def _run_slam(self, frame_stream: FrameStream, intrinsics: torch.Tensor) -> SLAMOutput:
        slam_pipeline = SLAMSystem(
            device=torch.device("cuda"),
            config=self.slam_cfg,
        )
        return slam_pipeline.run(frame_stream, intrinsics, camera_type=CameraType.PINHOLE)

    def _make_artifact_frame_loader(self, frame_stream: FrameStream, slam_output: SLAMOutput):
        intrinsics = slam_output.intrinsics[:4].detach().cpu().numpy().astype("float32")
        pose_mats = slam_output.trajectory.matrix().detach().cpu().numpy().astype("float32")
        w2c_mats = slam_output.trajectory.inv().matrix().detach().cpu().numpy().astype("float32")

        def load(frame_idx: int) -> io.ArtifactFrame:
            color, depth = frame_stream.artifact_arrays(frame_idx)
            return io.ArtifactFrame(
                frame_idx=frame_idx,
                color=color,
                depth=depth,
                pose_matrix=pose_mats[frame_idx],
                w2c_matrix=w2c_mats[frame_idx],
                intrinsics=intrinsics,
            )

        return load

    def _save_outputs(
        self,
        artifact_path: io.ArtifactPath,
        frame_stream: FrameStream,
        slam_output: SLAMOutput,
    ) -> None:
        logger.info(f"Saving artifacts to {artifact_path}")
        io.save_artifacts(
            artifact_path,
            self._make_artifact_frame_loader(frame_stream, slam_output),
            n_frames=len(frame_stream),
            max_pcd_points=self.out_cfg.pcd_max_points,
            pcd_tsdf_voxel_edge_m=self.out_cfg.pcd_tsdf_voxel_edge_m,
            pcd_tsdf_sdf_trunc_m=self.out_cfg.pcd_tsdf_sdf_trunc_m,
            pcd_tsdf_depth_trunc_m=self.out_cfg.pcd_tsdf_depth_trunc_m,
            pcd_tsdf_num_voxels_per_block_edge=self.out_cfg.pcd_tsdf_num_voxels_per_block_edge,
            pcd_tsdf_depth_sampling_stride=self.out_cfg.pcd_tsdf_depth_sampling_stride,
            pcd_tsdf_depth_filter=self.out_cfg.pcd_tsdf_depth_filter,
            pcd_tsdf_depth_filter_thresh=self.out_cfg.pcd_tsdf_depth_filter_thresh,
            pcd_tsdf_bilinear_color=self.out_cfg.pcd_tsdf_bilinear_color,
        )

    def run(self, frame_stream: FrameStream) -> SLAMOutput:
        frame_stream, intrinsics = self._initialize(frame_stream)

        artifact_path = io.ArtifactPath(self.out_path, frame_stream.name())
        slam_output = self._run_slam(frame_stream, intrinsics)
        self._save_outputs(artifact_path, frame_stream, slam_output)
        return slam_output
