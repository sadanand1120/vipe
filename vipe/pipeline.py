# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import time

from pathlib import Path

import torch

from vipe.slam.interface import SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.stream import FrameDir
from vipe.utils import io


logger = logging.getLogger(__name__)


class VipePipeline:
    def __init__(self, slam, output, output_dir: str | Path, instance=None) -> None:
        self.slam_cfg = slam
        self.out_cfg = output
        self.instance_cfg = instance
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
        pose_matrices,
        w2c_matrices,
        intrinsics,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        logger.info(f"Saving artifacts to {artifact_path}")

        def load_frame(frame_idx: int) -> io.TSDFFrame:
            color, depth = frame_stream.artifact_arrays(frame_idx)
            return io.TSDFFrame(color, depth, w2c_matrices[frame_idx], intrinsics)

        return io.save_artifacts(
            artifact_path,
            pose_matrices,
            load_frame,
            max_pcd_points=self.out_cfg.pcd_max_points,
            pcd_tsdf_voxel_edge_m=self.out_cfg.pcd_tsdf_voxel_edge_m,
            pcd_tsdf_sdf_trunc_m=self.out_cfg.pcd_tsdf_sdf_trunc_m,
            pcd_tsdf_depth_trunc_m=self.out_cfg.pcd_tsdf_depth_trunc_m,
            pcd_tsdf_num_voxels_per_block_edge=self.out_cfg.pcd_tsdf_num_voxels_per_block_edge,
            pcd_tsdf_depth_sampling_stride=self.out_cfg.pcd_tsdf_depth_sampling_stride,
            retain_tsdf_surface=self.instance_cfg is not None,
        )

    def run(self, frame_stream: FrameDir) -> None:
        intrinsics = self._load_intrinsics(frame_stream)

        artifact_path = io.ArtifactPath(self.out_path, frame_stream.name)
        slam_output = self._run_slam(frame_stream, intrinsics)
        intrinsics_cpu = slam_output.intrinsics[:4].detach().cpu().numpy().astype("float32")
        pose_matrices = slam_output.trajectory.matrix().detach().cpu().numpy().astype("float32")
        w2c_matrices = slam_output.trajectory.inv().matrix().detach().cpu().numpy().astype("float32")
        tsdf_surface = self._save_outputs(
            artifact_path, frame_stream, pose_matrices, w2c_matrices, intrinsics_cpu
        )

        if self.instance_cfg is None:
            return
        if tsdf_surface is None:
            raise ValueError("TSDF extraction produced no surface for instance distillation")

        del slam_output
        gc.collect()
        torch.cuda.empty_cache()

        from vipe.instance.pipeline import InstancePipeline, reduce_tsdf_surface

        tick = time.perf_counter()
        surface_points, surface_normals = reduce_tsdf_surface(
            *tsdf_surface,
            float(self.out_cfg.pcd_tsdf_voxel_edge_m),
        )
        surface_reduce_seconds = time.perf_counter() - tick
        source_surface_points = len(tsdf_surface[0])
        del tsdf_surface

        logger.info("Starting post-TSDF instance distillation for %s", frame_stream.name)
        InstancePipeline(self.instance_cfg, device=torch.device("cuda")).run(
            frame_stream,
            pose_matrices,
            intrinsics_cpu,
            artifact_path,
            surface_points,
            surface_normals,
            float(self.out_cfg.pcd_tsdf_voxel_edge_m),
            source_surface_points,
            surface_reduce_seconds,
        )
