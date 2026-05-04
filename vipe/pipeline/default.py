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

from pathlib import Path
from typing import Iterator

import torch

from omegaconf import DictConfig

from vipe.slam.system import SLAMOutput, SLAMSystem
from vipe.streams.base import FrameData, FrameStream
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.visualization import save_projection_video

from .processors import (
    DAV3DepthStream,
    estimate_geocalib_intrinsics,
)


logger = logging.getLogger(__name__)


class SharedIntrinsicsFrameStream(FrameStream):
    def __init__(self, stream: FrameStream, intrinsics: torch.Tensor):
        self.stream = stream
        self.intrinsics = intrinsics

    def frame_size(self) -> tuple[int, int]:
        return self.stream.frame_size()

    def fps(self) -> float:
        return self.stream.fps()

    def name(self) -> str:
        return self.stream.name()

    def __len__(self) -> int:
        return len(self.stream)

    def __iter__(self) -> Iterator[FrameData]:
        for frame in self.stream:
            frame.intrinsics = self.intrinsics
            frame.camera_type = CameraType.PINHOLE
            yield frame


class SLAMOutputFrameStream(FrameStream):
    def __init__(self, stream: FrameStream, slam_output: SLAMOutput):
        self.stream = stream
        self.slam_output = slam_output

    def frame_size(self) -> tuple[int, int]:
        return self.stream.frame_size()

    def fps(self) -> float:
        return self.stream.fps()

    def name(self) -> str:
        return self.stream.name()

    def __len__(self) -> int:
        return len(self.stream)

    def __iter__(self) -> Iterator[FrameData]:
        for frame_idx, frame in enumerate(self.stream):
            frame.pose = self.slam_output.trajectory[frame_idx]
            frame.intrinsics = self.slam_output.intrinsics
            frame.camera_type = CameraType.PINHOLE
            yield frame


class DefaultAnnotationPipeline:
    def __init__(self, slam: DictConfig, output: DictConfig) -> None:
        self.slam_cfg = slam
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)

    def run(self, frame_stream: FrameStream) -> SLAMOutput:
        artifact_path = io.ArtifactPath(self.out_path, frame_stream.name())
        intrinsics = estimate_geocalib_intrinsics(frame_stream)
        init_stream = SharedIntrinsicsFrameStream(frame_stream, intrinsics)

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run(init_stream, camera_type=CameraType.PINHOLE)

        output_stream = DAV3DepthStream(SLAMOutputFrameStream(frame_stream, slam_output), slam_output)

        if self.out_cfg.save_artifacts:
            logger.info(f"Saving artifacts to {artifact_path}")
            io.save_artifacts(
                artifact_path,
                output_stream,
                pcd_fusion_mode=self.out_cfg.pcd_fusion_mode,
                max_pcd_points=self.out_cfg.pcd_max_points,
                pcd_conf_threshold_coef=self.out_cfg.pcd_conf_threshold_coef,
                pcd_sample_ratio=self.out_cfg.pcd_sample_ratio,
                pcd_tsdf_voxel_length=self.out_cfg.pcd_tsdf_voxel_length,
                pcd_tsdf_sdf_trunc=self.out_cfg.pcd_tsdf_sdf_trunc,
                pcd_tsdf_depth_trunc=self.out_cfg.pcd_tsdf_depth_trunc,
            )

        if self.out_cfg.save_viz:
            viz_stream = io.ArtifactFrameStream(artifact_path) if self.out_cfg.save_artifacts else output_stream
            save_projection_video(
                artifact_path.meta_vis_path,
                viz_stream,
                slam_output,
                self.out_cfg.viz_downsample,
                self.out_cfg.viz_attributes,
            )

        return slam_output
