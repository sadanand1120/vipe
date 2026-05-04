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
import pickle

from pathlib import Path

import numpy as np
import torch

from omegaconf import DictConfig

from vipe.slam.system import SLAMOutput, SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    ProcessedFrameStream,
    FrameProcessor,
    FrameStream,
)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.geometry import se3_matrix_to_se3
from vipe.utils.logging import pbar
from vipe.utils.visualization import save_projection_video

from .processors import (
    DAV3DepthProcessor,
    GeoCalibIntrinsicsProcessor,
)


logger = logging.getLogger(__name__)


def _cpu_value(value):
    return value.cpu() if hasattr(value, "cpu") else value


def _cuda_value(value):
    return value.cuda() if hasattr(value, "cuda") else value


class InitAttributeRecorder(FrameProcessor):
    def __init__(self) -> None:
        self.recorded_attributes = [
            FrameAttribute.INTRINSICS,
            FrameAttribute.CAMERA_TYPE,
        ]
        self.stream_attributes: dict[FrameAttribute, list] = {attribute: [] for attribute in self.recorded_attributes}

    def __call__(self, frame_idx: int, frame):
        for attribute in self.recorded_attributes:
            self.stream_attributes[attribute].append(_cpu_value(frame.get_attribute(attribute)))
        return frame

    def replay_processors(self) -> list[FrameProcessor]:
        stream_attributes = {
            attribute: values
            for attribute, values in self.stream_attributes.items()
            if any(value is not None for value in values)
        }
        return [AssignRecordedInitProcessor(stream_attributes)]


class AssignRecordedInitProcessor(FrameProcessor):
    def __init__(
        self,
        stream_attributes: dict[FrameAttribute, list],
    ) -> None:
        self.stream_attributes = stream_attributes

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes.union(self.stream_attributes.keys())

    def __call__(self, frame_idx: int, frame):
        for attribute, attribute_values in self.stream_attributes.items():
            frame.set_attribute(attribute, _cuda_value(attribute_values[frame_idx]))
        return frame


class InitializedFrameStream(FrameStream):
    def __init__(self, stream: FrameStream, init_processors: list[FrameProcessor], recorder: InitAttributeRecorder):
        self.stream = stream
        self.init_processors = init_processors
        self.recorder = recorder
        self.initialized = False

    def frame_size(self) -> tuple[int, int]:
        return self.stream.frame_size()

    def fps(self) -> float:
        return self.stream.fps()

    def name(self) -> str:
        return self.stream.name()

    def __len__(self) -> int:
        return len(self.stream)

    def attributes(self) -> set[FrameAttribute]:
        attributes = self.stream.attributes()
        for processor in self.init_processors:
            attributes = processor.update_attributes(attributes)
        return attributes

    def replay(self) -> ProcessedFrameStream:
        if not self.initialized:
            raise RuntimeError("Initialization attributes are not available until the first full stream pass completes")
        return ProcessedFrameStream(self.stream, self.recorder.replay_processors())

    def __iter__(self):
        if self.initialized:
            return iter(self.replay())

        def _recording_iterator():
            processed = ProcessedFrameStream(self.stream, self.init_processors + [self.recorder])
            for frame in processed:
                yield frame
            self.initialized = True
            torch.cuda.empty_cache()

        return _recording_iterator()


class DefaultAnnotationPipeline:
    def __init__(
        self,
        init: DictConfig,
        slam: DictConfig,
        post: DictConfig,
        output: DictConfig,
        use_gt_pose: bool = False,
        use_gt_depth: bool = False,
    ) -> None:
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.use_gt_pose = use_gt_pose
        self.use_gt_depth = use_gt_depth
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)

    def _add_init_processors(self, frame_stream: FrameStream) -> InitializedFrameStream:
        init_processors: list[FrameProcessor] = []

        # The assertions make sure that the attributes are not estimated previously.
        # Otherwise it will be overwritten by the processors.
        assert FrameAttribute.INTRINSICS not in frame_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in frame_stream.attributes()

        init_processors.append(GeoCalibIntrinsicsProcessor(frame_stream, camera_type=self.camera_type))
        recorder = InitAttributeRecorder()
        return InitializedFrameStream(frame_stream, init_processors, recorder)

    def _gt_slam_output(self, frame_stream: FrameStream) -> SLAMOutput:
        pose_list = []
        intrinsics = None
        for frame in pbar(frame_stream, desc="Loading GT pose stream", total=len(frame_stream)):
            if frame.pose is None:
                raise ValueError("pipeline.use_gt_pose=true requires ScanNet pose files for every frame")
            pose_list.append(frame.pose.matrix().detach().cpu().numpy())
            if intrinsics is None:
                intrinsics = frame.intrinsics.detach().cpu()
        if intrinsics is None:
            raise ValueError("Intrinsics initialization failed")
        trajectory = se3_matrix_to_se3(np.stack(pose_list, axis=0)).cuda()
        return SLAMOutput(trajectory=trajectory, intrinsics=intrinsics)

    def _add_post_processors(self, frame_stream: FrameStream, slam_output: SLAMOutput) -> ProcessedFrameStream:
        post_processors: list[FrameProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.trajectory,
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics] * len(frame_stream),
                }
            )
        ]
        if not self.use_gt_depth and (depth_align_model := self.post_cfg.depth_align_model) is not None:
            post_processors.append(DAV3DepthProcessor(slam_output, model=depth_align_model))
        return ProcessedFrameStream(frame_stream, post_processors)

    def run(self, frame_stream: FrameStream, source_frame_dir: Path | None = None) -> SLAMOutput:
        artifact_path = io.ArtifactPath(self.out_path, frame_stream.name())
        init_stream = self._add_init_processors(frame_stream)
        source_frame_dir = source_frame_dir if source_frame_dir is not None else frame_stream.path

        skip_slam = self.use_gt_pose and (self.use_gt_depth or self.post_cfg.depth_align_model is None)
        if skip_slam:
            slam_output = self._gt_slam_output(init_stream)
        else:
            slam_cfg = self.slam_cfg.copy()
            if self.use_gt_depth:
                slam_cfg.keyframe_depth = None
            slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=slam_cfg)
            slam_output = slam_pipeline.run(init_stream, camera_type=self.camera_type)
            if self.use_gt_pose:
                gt_output = self._gt_slam_output(init_stream.replay())
                slam_output.trajectory = gt_output.trajectory

        output_stream = self._add_post_processors(init_stream.replay(), slam_output)

        artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
        if self.out_cfg.save_artifacts:
            logger.info(f"Saving artifacts to {artifact_path}")
            io.save_artifacts(
                artifact_path,
                output_stream,
                source_frame_dir=source_frame_dir,
                pcd_fusion_mode=self.out_cfg.pcd_fusion_mode,
                max_pcd_points=self.out_cfg.backproject_pcd_max_points,
                pcd_conf_threshold_coef=self.out_cfg.backproject_pcd_conf_threshold_coef,
                pcd_sample_ratio=self.out_cfg.backproject_pcd_sample_ratio,
                pcd_tsdf_voxel_length=self.out_cfg.pcd_tsdf_voxel_length,
                pcd_tsdf_sdf_trunc=self.out_cfg.pcd_tsdf_sdf_trunc,
                pcd_tsdf_depth_trunc=self.out_cfg.pcd_tsdf_depth_trunc,
            )
            with artifact_path.meta_info_path.open("wb") as f:
                pickle.dump({"ba_residual": slam_output.ba_residual}, f)

        if self.out_cfg.save_viz:
            viz_stream = io.ArtifactFrameStream(artifact_path) if self.out_cfg.save_artifacts else output_stream
            save_projection_video(
                artifact_path.meta_vis_path,
                viz_stream,
                slam_output,
                self.out_cfg.viz_downsample,
                self.out_cfg.viz_attributes,
            )

        if self.out_cfg.save_slam_map and slam_output.slam_map is not None:
            logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
            slam_output.slam_map.save(artifact_path.slam_map_path)

        return slam_output
