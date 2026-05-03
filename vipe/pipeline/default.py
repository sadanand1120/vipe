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
import gc

from pathlib import Path

import torch

from omegaconf import DictConfig

from vipe.slam.system import SLAMOutput, SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    CachedVideoStream,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.visualization import save_projection_video

from . import AnnotationPipelineOutput, Pipeline
from .processors import (
    GeoCalibIntrinsicsProcessor,
    MultiviewDepthProcessor,
    TrackAnythingProcessor,
)


logger = logging.getLogger(__name__)


class AssignCachedInitProcessor(StreamProcessor):
    def __init__(
        self,
        stream_attributes: dict[FrameAttribute, list],
        instance_phrases: list[dict[int, str] | None],
    ) -> None:
        self.stream_attributes = stream_attributes
        self.instance_phrases = instance_phrases

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes.union(self.stream_attributes.keys())

    def __call__(self, frame_idx: int, frame):
        for attribute, attribute_values in self.stream_attributes.items():
            frame.set_attribute(attribute, attribute_values[frame_idx])
        frame.instance_phrases = self.instance_phrases[frame_idx]
        return frame


class DefaultAnnotationPipeline(Pipeline):
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)

    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        init_processors: list[StreamProcessor] = []

        # The assertions make sure that the attributes are not estimated previously.
        # Otherwise it will be overwritten by the processors.
        assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        assert FrameAttribute.METRIC_DEPTH not in video_stream.attributes()
        assert FrameAttribute.INSTANCE not in video_stream.attributes()

        init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type))
        if self.init_cfg.instance is not None:
            init_processors.append(
                TrackAnythingProcessor(
                    self.init_cfg.instance.phrases,
                    add_sky=self.init_cfg.instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                )
            )
        return ProcessedVideoStream(video_stream, init_processors)

    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),  # type: ignore
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics[view_idx]] * len(video_stream),
                }
            )
        ]
        if (depth_align_model := self.post_cfg.depth_align_model) is not None:
            post_processors.append(MultiviewDepthProcessor(slam_output, model=depth_align_model))
        return ProcessedVideoStream(video_stream, post_processors)

    def _rebuild_init_streams(
        self,
        video_streams: list[VideoStream],
        slam_streams: list[VideoStream],
    ) -> list[VideoStream]:
        rebuilt_streams = []
        cached_attributes = [
            FrameAttribute.INTRINSICS,
            FrameAttribute.CAMERA_TYPE,
            FrameAttribute.INSTANCE,
            FrameAttribute.MASK,
        ]

        for video_stream, slam_stream in zip(video_streams, slam_streams):
            assert isinstance(slam_stream, CachedVideoStream)
            cached_frames = slam_stream.data
            stream_attributes = {}
            for attribute in cached_attributes:
                attribute_values = [frame.get_attribute(attribute) for frame in cached_frames]
                if any(value is not None for value in attribute_values):
                    stream_attributes[attribute] = attribute_values
            instance_phrases = [frame.instance_phrases for frame in cached_frames]
            rebuilt_streams.append(
                ProcessedVideoStream(
                    video_stream,
                    [AssignCachedInitProcessor(stream_attributes, instance_phrases)],
                )
            )
            slam_stream.data = []
            slam_stream.iterator = None

        gc.collect()
        torch.cuda.empty_cache()
        return rebuilt_streams

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()

        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None

        annotate_output = AnnotationPipelineOutput()

        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream).cache("process", online=True) for video_stream in video_streams
        ]

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        output_input_streams = self._rebuild_init_streams(video_streams, slam_streams)
        del slam_streams

        output_streams = [
            self._add_post_processors(view_idx, init_stream, slam_output)
            for view_idx, init_stream in enumerate(output_input_streams)
        ]

        # Dumping artifacts for all views in the streams
        for output_stream, artifact_path in zip(output_streams, artifact_paths):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            if self.out_cfg.save_artifacts:
                logger.info(f"Saving artifacts to {artifact_path}")
                io.save_artifacts(
                    artifact_path,
                    output_stream,
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
                viz_stream = io.ArtifactVideoStream(artifact_path) if self.out_cfg.save_artifacts else output_stream
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

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        return annotate_output
