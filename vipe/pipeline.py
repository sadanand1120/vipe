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

import numpy as np
import torch

from omegaconf import DictConfig

from vipe.priors.geocalib.extractor import GeoCalib
from vipe.slam.interface import SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.streams.base import FrameData, FrameStream
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar


logger = logging.getLogger(__name__)


def estimate_geocalib_intrinsics(frame_stream: FrameStream, gap_sec: float = 1.0) -> torch.Tensor:
    gap_frame = int(gap_sec * frame_stream.fps())
    gap_frame = min(gap_frame, (len(frame_stream) - 1) // 2)
    sample_frame_inds = [0, gap_frame, gap_frame * 2]
    sample_frame_set = set(sample_frame_inds)

    model = GeoCalib(weights="pinhole").cuda()
    sample_by_idx = {}
    for frame_idx, frame in enumerate(frame_stream):
        if frame_idx in sample_frame_set:
            sample_by_idx[frame_idx] = frame.rgb.moveaxis(-1, 0)
        if frame_idx >= sample_frame_inds[-1]:
            break

    sample_frames = torch.stack([sample_by_idx[i] for i in sample_frame_inds])
    res = model.calibrate(sample_frames, shared_intrinsics=True)
    fov_y = res["camera"].vfov[0].item()

    frame_height, frame_width = frame_stream.frame_size()
    fx = fy = frame_height / (2 * np.tan(fov_y / 2))
    return torch.as_tensor([fx, fy, frame_width / 2, frame_height / 2]).float().cuda()


class DAV3DepthEstimator:
    """
    Use DAV3 to estimate depth for each frame.
    Depth is conditioned on camera poses and intrinsics from SLAM.

    Depth is estimated in a sliding-window manner, and overlapped frames are linearly averaged to sharp transitions.
    """

    def __init__(
        self,
        model_name: str,
        window_size: int = 10,
        overlap_size: int = 3,
    ):
        super().__init__()
        self.window_size = window_size
        self.overlap_size = overlap_size

        try:
            from depth_anything_3.api import DepthAnything3
            from depth_anything_3.api import logger as dav3_logger
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "depth-anything-3 not found. Please reinstall vipe with `pip install --no-build-isolation -e .[dav3]`"
            )

        dav3_logger.level = 0
        self.dav3_api = DepthAnything3.from_pretrained(model_name)
        self.dav3_api = self.dav3_api.cuda().eval()

    @staticmethod
    def _probe_keyframe_indices(keyframe_indices: list[int], frame_idx: int) -> list[int]:
        assert keyframe_indices
        left = max(0, np.searchsorted(keyframe_indices, frame_idx, side="right").item() - 1)
        indices = [keyframe_indices[left]]
        if frame_idx < keyframe_indices[-1] and left + 1 < len(keyframe_indices):
            indices.append(keyframe_indices[left + 1])
        return indices

    def _attach_slam_output(self, frame_stream: FrameStream, frame_idx: int, slam_output: SLAMOutput) -> FrameData:
        frame = frame_stream[frame_idx]
        frame.pose = slam_output.trajectory[frame_idx]
        frame.intrinsics = slam_output.intrinsics
        frame.camera_type = CameraType.PINHOLE
        return frame

    def estimate(self, frame_stream: FrameStream, slam_output: SLAMOutput) -> Iterator[FrameData]:
        n_frames = len(frame_stream)

        current_sliding_window: list[FrameData] = []
        current_sliding_window_idx: list[int] = []
        trailing_depth: torch.Tensor | None = None
        trailing_confidence: torch.Tensor | None = None
        for frame_idx in pbar(range(n_frames), desc="Estimating DAV3 depth"):
            frame = self._attach_slam_output(frame_stream, frame_idx, slam_output)
            current_sliding_window.append(frame)
            current_sliding_window_idx.append(frame_idx)
            is_last_frame = frame_idx == n_frames - 1

            if len(current_sliding_window) == self.window_size or is_last_frame:
                context_indices = sorted(
                    {
                        keyframe_idx
                        for i in current_sliding_window_idx
                        for keyframe_idx in self._probe_keyframe_indices(slam_output.keyframe_indices, i)
                    }
                )
                context_indices = [i for i in context_indices if i not in current_sliding_window_idx]

                sw_images, sw_exts, sw_ints = zip(*[frame.dav3_conditions() for frame in current_sliding_window])
                if context_indices:
                    ctx_images, ctx_exts, ctx_ints = zip(
                        *[
                            self._attach_slam_output(frame_stream, frame_idx, slam_output).dav3_conditions()
                            for frame_idx in context_indices
                        ]
                    )
                else:
                    ctx_images, ctx_exts, ctx_ints = tuple(), tuple(), tuple()

                dav3_inference_result = self.dav3_api.inference(
                    list(sw_images + ctx_images),
                    extrinsics=np.stack(sw_exts + ctx_exts, axis=0),
                    intrinsics=np.stack(sw_ints + ctx_ints, axis=0),
                    process_res_method="lower_bound_resize",
                )
                sw_depth = torch.from_numpy(dav3_inference_result.depth[: len(sw_images)]).float().cuda()
                sw_depth = torch.nn.functional.interpolate(sw_depth[:, None], frame.size(), mode="bilinear")[:, 0]
                sw_confidence = None
                if dav3_inference_result.conf is not None:
                    sw_confidence = torch.from_numpy(dav3_inference_result.conf[: len(sw_images)]).float().cuda()
                    sw_confidence = torch.nn.functional.interpolate(
                        sw_confidence[:, None], frame.size(), mode="bilinear"
                    )[:, 0]

                n_frames_to_yield = (
                    self.window_size - self.overlap_size if not is_last_frame else len(current_sliding_window)
                )

                if trailing_depth is not None:
                    n_interp_frames = len(trailing_depth)
                    alpha = torch.linspace(0, 1, n_interp_frames + 2)[1:-1].float().cuda()[:, None, None]
                    sw_depth[:n_interp_frames] = trailing_depth * (1 - alpha) + sw_depth[:n_interp_frames] * alpha
                    if trailing_confidence is not None and sw_confidence is not None:
                        sw_confidence[:n_interp_frames] = (
                            trailing_confidence * (1 - alpha) + sw_confidence[:n_interp_frames] * alpha
                        )

                for sw_idx, frame in enumerate(current_sliding_window[:n_frames_to_yield]):
                    frame.metric_depth = sw_depth[sw_idx]
                    frame.depth_confidence = None if sw_confidence is None else sw_confidence[sw_idx]
                    yield frame

                trailing_depth = sw_depth[n_frames_to_yield:]
                trailing_confidence = None if sw_confidence is None else sw_confidence[n_frames_to_yield:]
                current_sliding_window = current_sliding_window[n_frames_to_yield:]
                current_sliding_window_idx = current_sliding_window_idx[n_frames_to_yield:]

        assert len(current_sliding_window) == 0, "Current sliding window should be empty"


class VipePipeline:
    def __init__(self, slam: DictConfig, depth: DictConfig, output: DictConfig) -> None:
        self.slam_cfg = slam
        self.depth_cfg = depth
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)

    def _initialize(self, frame_stream: FrameStream) -> torch.Tensor:
        return estimate_geocalib_intrinsics(frame_stream)

    def _run_slam(self, frame_stream: FrameStream, intrinsics: torch.Tensor) -> SLAMOutput:
        slam_pipeline = SLAMSystem(
            device=torch.device("cuda"),
            config=self.slam_cfg,
            keyframe_depth_model=self.depth_cfg.keyframe_model,
        )
        return slam_pipeline.run(frame_stream, intrinsics, camera_type=CameraType.PINHOLE)

    def _run_final_depth(self, frame_stream: FrameStream, slam_output: SLAMOutput) -> Iterator[FrameData]:
        depth_estimator = DAV3DepthEstimator(
            model_name=self.depth_cfg.final_model,
            window_size=self.depth_cfg.window_size,
            overlap_size=self.depth_cfg.overlap_size,
        )
        return depth_estimator.estimate(frame_stream, slam_output)

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
            self._run_final_depth(frame_stream, slam_output),
            n_frames=len(frame_stream),
            pcd_fusion_mode=self.out_cfg.pcd_fusion_mode,
            max_pcd_points=self.out_cfg.pcd_max_points,
            pcd_conf_threshold_coef=self.out_cfg.pcd_conf_threshold_coef,
            pcd_sample_ratio=self.out_cfg.pcd_sample_ratio,
            pcd_tsdf_voxel_length=self.out_cfg.pcd_tsdf_voxel_length,
            pcd_tsdf_sdf_trunc=self.out_cfg.pcd_tsdf_sdf_trunc,
            pcd_tsdf_depth_trunc=self.out_cfg.pcd_tsdf_depth_trunc,
        )

    def run(self, frame_stream: FrameStream) -> SLAMOutput:
        artifact_path = io.ArtifactPath(self.out_path, frame_stream.name())
        intrinsics = self._initialize(frame_stream)
        slam_output = self._run_slam(frame_stream, intrinsics)
        self._save_outputs(artifact_path, frame_stream, slam_output)

        return slam_output
