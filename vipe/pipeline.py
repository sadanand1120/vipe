# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from pathlib import Path

import torch

from vipe.slam.interface import SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.stream import FrameDir
from vipe.utils import io


logger = logging.getLogger(__name__)


class VipePipeline:
    def __init__(self, slam, output, output_dir: str | Path) -> None:
        self.slam_cfg = slam
        self.out_cfg = output
        self.out_path = Path(output_dir)
        self.out_path.mkdir(exist_ok=True, parents=True)

    def _load_intrinsics(self, frame_stream: FrameDir) -> torch.Tensor:
        intrinsics = frame_stream.intrinsics()
        logger.info(
            "Using loaded pinhole intrinsics from %s: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            frame_stream.intrinsics_path,
            intrinsics[0].item(),
            intrinsics[1].item(),
            intrinsics[2].item(),
            intrinsics[3].item(),
        )
        return intrinsics

    def _run_slam(self, frame_stream: FrameDir, intrinsics: torch.Tensor) -> SLAMOutput:
        slam_pipeline = SLAMSystem(
            device=torch.device("cuda"),
            config=self.slam_cfg,
        )
        return slam_pipeline.run(frame_stream, intrinsics)

    def _save_outputs(
        self,
        artifact_path: io.ArtifactPath,
        frame_stream: FrameDir,
        slam_output: SLAMOutput,
    ) -> None:
        logger.info(f"Saving artifacts to {artifact_path}")
        intrinsics = slam_output.intrinsics[:4].detach().cpu().numpy().astype("float32")
        pose_matrices = slam_output.trajectory.matrix().detach().cpu().numpy().astype("float32")
        w2c_matrices = slam_output.trajectory.inv().matrix().detach().cpu().numpy().astype("float32")

        def load_frame(frame_idx: int) -> io.TSDFFrame:
            color, depth = frame_stream.artifact_arrays(frame_idx)
            return io.TSDFFrame(color, depth, w2c_matrices[frame_idx], intrinsics)

        io.save_artifacts(
            artifact_path,
            pose_matrices,
            load_frame,
            max_pcd_points=self.out_cfg.pcd_max_points,
            pcd_tsdf_voxel_edge_m=self.out_cfg.pcd_tsdf_voxel_edge_m,
            pcd_tsdf_sdf_trunc_m=self.out_cfg.pcd_tsdf_sdf_trunc_m,
            pcd_tsdf_depth_trunc_m=self.out_cfg.pcd_tsdf_depth_trunc_m,
            pcd_tsdf_num_voxels_per_block_edge=self.out_cfg.pcd_tsdf_num_voxels_per_block_edge,
            pcd_tsdf_depth_sampling_stride=self.out_cfg.pcd_tsdf_depth_sampling_stride,
        )

    def run(self, frame_stream: FrameDir) -> SLAMOutput:
        intrinsics = self._load_intrinsics(frame_stream)

        artifact_path = io.ArtifactPath(self.out_path, frame_stream.name)
        slam_output = self._run_slam(frame_stream, intrinsics)
        self._save_outputs(artifact_path, frame_stream, slam_output)
        return slam_output
