# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

from omegaconf import DictConfig
from torch.nn import functional as F

from vipe.slam.interface import SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.streams.base import FrameData, FrameStream, SensorCamera
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar


logger = logging.getLogger(__name__)


class OpenCVPinholeNormalizedFrameStream(FrameStream):
    def __init__(self, frame_stream: FrameStream, camera: SensorCamera) -> None:
        super().__init__()
        self.frame_stream = frame_stream
        self.camera = camera
        self.grid = self._build_undistort_grid(camera)
        self.grid_valid_mask = self._grid_valid_mask(self.grid)

    @staticmethod
    def _build_undistort_grid(camera: SensorCamera) -> torch.Tensor:
        assert camera.distortion_coefficients is not None
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera.input_k,
            camera.distortion_coefficients,
            np.eye(3, dtype=np.float32),
            camera.output_k,
            (camera.width, camera.height),
            cv2.CV_32FC1,
        )
        grid = torch.as_tensor(np.stack((mapx, mapy), axis=-1), dtype=torch.float32).cuda()[None]
        scale = torch.tensor([camera.width - 1, camera.height - 1], dtype=torch.float32).cuda()
        return 2.0 * grid / scale - 1.0

    @staticmethod
    def _grid_valid_mask(grid: torch.Tensor) -> torch.Tensor:
        xy = grid[0]
        return (
            torch.isfinite(xy).all(dim=-1)
            & (xy[..., 0] >= -1.0)
            & (xy[..., 0] <= 1.0)
            & (xy[..., 1] >= -1.0)
            & (xy[..., 1] <= 1.0)
        )

    def _remap_image(self, image: torch.Tensor) -> torch.Tensor:
        remapped = F.grid_sample(
            image.permute(2, 0, 1)[None],
            self.grid,
            mode="bilinear",
            align_corners=True,
        )[0]
        return remapped.permute(1, 2, 0).clamp(0.0, 1.0)

    def _remap_map(self, image: torch.Tensor, mode: str) -> torch.Tensor:
        return F.grid_sample(image[None, None], self.grid, mode=mode, align_corners=True)[0, 0]

    def _remap_valid_mask(self, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return self.grid_valid_mask
        return (self._remap_map(mask.float(), mode="nearest") > 0.5) & self.grid_valid_mask

    def frame_size(self) -> tuple[int, int]:
        return self.frame_stream.frame_size()

    def name(self) -> str:
        return self.frame_stream.name()

    def fps(self) -> float:
        return self.frame_stream.fps()

    def sensor_camera(self) -> SensorCamera | None:
        return self.frame_stream.sensor_camera()

    def __len__(self) -> int:
        return len(self.frame_stream)

    def __getitem__(self, index: int) -> FrameData:
        frame = self.frame_stream[index]
        image_valid_mask = self._remap_valid_mask(frame.image_valid_mask)
        rgb = self._remap_image(frame.rgb)
        rgb = torch.where(image_valid_mask[..., None], rgb, torch.zeros_like(rgb))

        sensor_depth = self._remap_map(frame.sensor_depth, mode="nearest")
        valid_depth = torch.isfinite(sensor_depth) & (sensor_depth > 0.0) & image_valid_mask
        sensor_depth = torch.where(valid_depth, sensor_depth, torch.zeros_like(sensor_depth))

        return FrameData(
            raw_frame_idx=frame.raw_frame_idx,
            rgb=rgb,
            sensor_depth=sensor_depth,
            pose=frame.pose,
            camera_type=frame.camera_type,
            intrinsics=frame.intrinsics,
            image_valid_mask=image_valid_mask,
            information=frame.information,
        )

    def __iter__(self) -> Iterator[FrameData]:
        for frame_idx in range(len(self)):
            yield self[frame_idx]


class VipePipeline:
    def __init__(self, slam: DictConfig, output: DictConfig) -> None:
        self.slam_cfg = slam
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)

    def _initialize(self, frame_stream: FrameStream) -> tuple[FrameStream, torch.Tensor]:
        camera = frame_stream.sensor_camera()
        if camera is None:
            raise ValueError("Input stream must provide external RGB/color intrinsics")

        intrinsics = camera.pinhole_intrinsics()
        if camera.has_distortion:
            logger.info(
                "Normalizing loaded %s camera to pinhole from %s: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
                camera.distortion_model,
                camera.source_path,
                intrinsics[0].item(),
                intrinsics[1].item(),
                intrinsics[2].item(),
                intrinsics[3].item(),
            )
            return OpenCVPinholeNormalizedFrameStream(frame_stream, camera), intrinsics

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

    @staticmethod
    def _attach_final_depth(frame: FrameData, frame_idx: int, slam_output: SLAMOutput) -> FrameData:
        frame.pose = slam_output.trajectory[frame_idx]
        frame.intrinsics = slam_output.intrinsics
        frame.camera_type = CameraType.PINHOLE

        sensor_depth = frame.sensor_depth.float()
        valid = torch.isfinite(sensor_depth) & (sensor_depth > 0.0)
        if frame.image_valid_mask is not None:
            valid &= frame.image_valid_mask.to(valid.device)
        frame.metric_depth = torch.where(valid, sensor_depth, torch.zeros_like(sensor_depth))
        return frame

    def _final_frames(self, frame_stream: FrameStream, slam_output: SLAMOutput) -> Iterator[FrameData]:
        for frame_idx in pbar(range(len(frame_stream)), desc="Loading sensor depth"):
            yield self._attach_final_depth(frame_stream[frame_idx], frame_idx, slam_output)

    def _save_outputs(
        self,
        artifact_path: io.ArtifactPath,
        frame_stream: FrameStream,
        slam_output: SLAMOutput,
    ) -> None:
        if not self.out_cfg.save_artifacts:
            return

        logger.info(f"Saving artifacts to {artifact_path}")
        io.save_artifacts(
            artifact_path,
            self._final_frames(frame_stream, slam_output),
            n_frames=len(frame_stream),
            pcd_fusion_mode=self.out_cfg.pcd_fusion_mode,
            max_pcd_points=self.out_cfg.pcd_max_points,
            pcd_sample_ratio=self.out_cfg.pcd_sample_ratio,
            pcd_tsdf_voxel_length=self.out_cfg.pcd_tsdf_voxel_length,
            pcd_tsdf_sdf_trunc=self.out_cfg.pcd_tsdf_sdf_trunc,
            pcd_tsdf_depth_trunc=self.out_cfg.pcd_tsdf_depth_trunc,
        )

    def run(self, frame_stream: FrameStream) -> SLAMOutput:
        frame_stream, intrinsics = self._initialize(frame_stream)
        artifact_path = io.ArtifactPath(self.out_path, frame_stream.name())
        slam_output = self._run_slam(frame_stream, intrinsics)
        self._save_outputs(artifact_path, frame_stream, slam_output)
        return slam_output
