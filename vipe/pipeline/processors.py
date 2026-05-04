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

import cv2
import numpy as np
import torch

from vipe.priors.geocalib import GeoCalib
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import FrameAttribute, FrameProcessor, FrameData, FrameStream
from vipe.utils.cameras import CameraType
from vipe.utils.geometry import se3_matrix_to_se3
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional

logger = logging.getLogger(__name__)


class ScanNetGTProcessor(FrameProcessor):
    def __init__(
        self,
        frame_files: list[Path],
        scene_dir: Path,
        use_gt_pose: bool,
        use_gt_depth: bool,
    ) -> None:
        self.frame_files = frame_files
        self.pose_dir = scene_dir / "pose"
        self.depth_dir = scene_dir / "depth"
        self.use_gt_pose = use_gt_pose
        self.use_gt_depth = use_gt_depth

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        attributes = set(previous_attributes)
        if self.use_gt_pose:
            attributes.add(FrameAttribute.POSE)
        if self.use_gt_depth:
            attributes.add(FrameAttribute.METRIC_DEPTH)
        return attributes

    def __call__(self, frame_idx: int, frame: FrameData) -> FrameData:
        frame_id = self.frame_files[frame.raw_frame_idx].stem

        if self.use_gt_pose:
            pose_path = self.pose_dir / f"{frame_id}.txt"
            pose = np.loadtxt(pose_path, dtype=np.float32)
            frame.pose = se3_matrix_to_se3(pose).cuda()

        if self.use_gt_depth:
            depth_path = self.depth_dir / f"{frame_id}.png"
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Could not read ScanNet depth: {depth_path}")
            depth = depth.astype(np.float32) / 1000.0
            if depth.shape != frame.size():
                depth = cv2.resize(depth, (frame.size()[1], frame.size()[0]), interpolation=cv2.INTER_NEAREST)
            frame.metric_depth = torch.from_numpy(depth).float().cuda()

        return frame


class IntrinsicEstimationProcessor(FrameProcessor):
    """Override existing intrinsics with estimated intrinsics."""

    def __init__(self, frame_stream: FrameStream, gap_sec: float = 1.0) -> None:
        super().__init__()
        gap_frame = int(gap_sec * frame_stream.fps())
        gap_frame = min(gap_frame, (len(frame_stream) - 1) // 2)
        self.sample_frame_inds = [0, gap_frame, gap_frame * 2]
        self.fov_y = -1.0
        self.camera_type = CameraType.PINHOLE

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INTRINSICS}

    def __call__(self, frame_idx: int, frame: FrameData) -> FrameData:
        assert self.fov_y > 0, "FOV not set"
        frame_height, frame_width = frame.size()
        fx = fy = frame_height / (2 * np.tan(self.fov_y / 2))
        frame.intrinsics = torch.as_tensor(
            [fx, fy, frame_width / 2, frame_height / 2],
        ).float()
        frame.camera_type = self.camera_type
        return frame


class GeoCalibIntrinsicsProcessor(IntrinsicEstimationProcessor):
    def __init__(
        self,
        frame_stream: FrameStream,
        gap_sec: float = 1.0,
        camera_type: CameraType = CameraType.PINHOLE,
    ) -> None:
        super().__init__(frame_stream, gap_sec)
        assert camera_type == CameraType.PINHOLE, "Only pinhole camera intrinsics are supported"

        model = GeoCalib(weights="pinhole").cuda()
        sample_frame_set = set(self.sample_frame_inds)
        sample_by_idx = {}
        for frame_idx, frame in enumerate(frame_stream):
            if frame_idx in sample_frame_set:
                sample_by_idx[frame_idx] = frame.rgb.moveaxis(-1, 0)
            if frame_idx >= self.sample_frame_inds[-1]:
                break
        sample_frames = torch.stack([sample_by_idx[i] for i in self.sample_frame_inds])
        res = model.calibrate(
            sample_frames,
            shared_intrinsics=True,
        )

        self.fov_y = res["camera"].vfov[0].item()
        self.camera_type = camera_type


class DAV3DepthProcessor(FrameProcessor):
    """
    Use DAV3 to estimate depth for each frame.
    Depth is conditioned on camera poses and intrinsics from SLAM.

    Depth is estimated in a sliding-window manner, and overlapped frames are linearly averaged to sharp transitions.
    To create enough parallex to improve estimation confidence, for each window we optionally also include
    neighboring keyframes, and their secondary neighboring keyframes.
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        model: str = "mvd_dav3",
        window_size: int = 10,                  # Practically this should be as large as possible if memory permits.
        overlap_size: int = 3,
        secondary_keyframe: bool = False,       # This is found to cause jittering for some scenes due to abrupt context changes.
    ):
        super().__init__()
        self.slam_output = slam_output
        self.model = model
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.secondary_keyframe = secondary_keyframe

        self.keyframes_inds = unpack_optional(self.slam_output.slam_map).dense_disp_frame_inds
        self.keyframes_data: list[FrameData] = []
        self.n_frames = 0

        # Need two passes for this iterator to work.
        self.n_passes_required = 2

        if self.model != "mvd_dav3":
            raise ValueError(f"Only mvd_dav3 is supported, got {self.model}")

        try:
            from depth_anything_3.api import DepthAnything3
            from depth_anything_3.api import logger as dav3_logger
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "depth-anything-3 not found. Please reinstall vipe with `pip install --no-build-isolation -e .[dav3]`"
            )

        dav3_logger.level = 0  # Disable logging timing information
        self.dav3_api = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
        self.dav3_api = self.dav3_api.cuda().eval()

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH, FrameAttribute.DEPTH_CONFIDENCE}

    def __call__(self, frame_idx: int, frame: FrameData) -> FrameData:
        raise NotImplementedError("DAV3DepthProcessor should not be called directly.")

    def _probe_keyframe_indices(self, frame_idx: int) -> list[int]:
        inds: list[int] = []
        left_idx = np.searchsorted(self.keyframes_inds, frame_idx, side="right").item() - 1
        inds.append(left_idx)
        if frame_idx < self.keyframes_inds[-1]:
            inds.append(left_idx + 1)
        # Pick the farthest secondary keyframe from the left keyframe.
        if self.secondary_keyframe:
            slam_graph = unpack_optional(self.slam_output.slam_map).backend_graph
            if slam_graph is not None:
                matching_secondary_j = slam_graph[slam_graph[:, 0] == left_idx, 1].tolist()
                picked_sj_idx = np.argmax([abs(self.keyframes_inds[j] - frame_idx) for j in matching_secondary_j])
                inds.append(matching_secondary_j[picked_sj_idx])
        return inds

    def record_keyframes(self, previous_iterator: Iterator[FrameData]) -> Iterator[FrameData]:
        for frame_idx, frame in enumerate(previous_iterator):
            self.n_frames += 1
            if frame_idx in self.keyframes_inds:
                self.keyframes_data.append(frame)
            yield frame

    def estimate_depth_sliding_window(self, previous_iterator: Iterator[FrameData]) -> Iterator[FrameData]:
        current_sliding_window: list[FrameData] = []
        current_sliding_window_idx: list[int] = []
        trailing_depth: torch.Tensor | None = None
        trailing_confidence: torch.Tensor | None = None
        for frame_idx, frame in pbar(enumerate(previous_iterator), desc="Estimating DAV3 depth"):
            current_sliding_window.append(frame)
            current_sliding_window_idx.append(frame_idx)
            is_last_frame = frame_idx == self.n_frames - 1

            if len(current_sliding_window) == self.window_size or is_last_frame:
                # Grab all neighboring keyframes to anchor the current sliding window.
                # Note that we remove redundant keyframes that already exist in the current sliding window.
                sw_keyframe_inds = list(
                    set(sum([self._probe_keyframe_indices(i) for i in current_sliding_window_idx], []))
                )
                sw_keyframe_inds = [
                    t for t in sw_keyframe_inds if self.keyframes_inds[t] not in current_sliding_window_idx
                ]

                sw_images, sw_exts, sw_ints = zip(*[frame.dav3_conditions() for frame in current_sliding_window])

                if len(sw_keyframe_inds) > 0:
                    kf_images, kf_exts, kf_ints = zip(*[self.keyframes_data[t].dav3_conditions() for t in sw_keyframe_inds])
                else:
                    kf_images, kf_exts, kf_ints = tuple(), tuple(), tuple()

                # Perform inference
                dav3_inference_result = self.dav3_api.inference(
                    list(sw_images + kf_images),
                    extrinsics=np.stack(sw_exts + kf_exts, axis=0),
                    intrinsics=np.stack(sw_ints + kf_ints, axis=0),
                    process_res_method="lower_bound_resize",  # Keep aspect ratio
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

                # Linearly interpolate the trailing depth with new depth
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

    def update_iterator(self, previous_iterator: Iterator[FrameData], pass_idx: int) -> Iterator[FrameData]:
        if pass_idx == 0:
            yield from self.record_keyframes(previous_iterator)
        elif pass_idx == 1:
            yield from self.estimate_depth_sliding_window(previous_iterator)
        else:
            raise ValueError(f"Invalid pass index: {pass_idx}")
