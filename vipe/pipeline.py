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
from vipe.utils.misc import unpack_optional


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


class DAV3DepthStream(FrameStream):
    """
    Use DAV3 to estimate depth for each frame.
    Depth is conditioned on camera poses and intrinsics from SLAM.

    Depth is estimated in a sliding-window manner, and overlapped frames are linearly averaged to sharp transitions.
    To create enough parallax to improve estimation confidence, each window also includes neighboring keyframes.
    """

    def __init__(
        self,
        frame_stream: FrameStream,
        slam_output: SLAMOutput,
        window_size: int = 10,
        overlap_size: int = 3,
    ):
        super().__init__()
        self.frame_stream = frame_stream
        self.slam_output = slam_output
        self.window_size = window_size
        self.overlap_size = overlap_size

        self.keyframes_inds = unpack_optional(self.slam_output.slam_map).dense_disp_frame_inds

        try:
            from depth_anything_3.api import DepthAnything3
            from depth_anything_3.api import logger as dav3_logger
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "depth-anything-3 not found. Please reinstall vipe with `pip install --no-build-isolation -e .[dav3]`"
            )

        dav3_logger.level = 0
        self.dav3_api = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
        self.dav3_api = self.dav3_api.cuda().eval()

    def frame_size(self) -> tuple[int, int]:
        return self.frame_stream.frame_size()

    def fps(self) -> float:
        return self.frame_stream.fps()

    def name(self) -> str:
        return self.frame_stream.name()

    def __len__(self) -> int:
        return len(self.frame_stream)

    def _probe_keyframe_indices(self, frame_idx: int) -> list[int]:
        inds: list[int] = []
        left_idx = np.searchsorted(self.keyframes_inds, frame_idx, side="right").item() - 1
        inds.append(left_idx)
        if frame_idx < self.keyframes_inds[-1]:
            inds.append(left_idx + 1)
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
                # Neighboring keyframes anchor each current sliding window.
                sw_keyframe_inds = list(
                    set(sum([self._probe_keyframe_indices(i) for i in current_sliding_window_idx], []))
                )
                sw_keyframe_inds = [
                    t for t in sw_keyframe_inds if self.keyframes_inds[t] not in current_sliding_window_idx
                ]

                sw_images, sw_exts, sw_ints = zip(*[frame.dav3_conditions() for frame in current_sliding_window])

                if len(sw_keyframe_inds) > 0:
                    kf_images, kf_exts, kf_ints = zip(
                        *[self.keyframes_data[t].dav3_conditions() for t in sw_keyframe_inds]
                    )
                else:
                    kf_images, kf_exts, kf_ints = tuple(), tuple(), tuple()

                dav3_inference_result = self.dav3_api.inference(
                    list(sw_images + kf_images),
                    extrinsics=np.stack(sw_exts + kf_exts, axis=0),
                    intrinsics=np.stack(sw_ints + kf_ints, axis=0),
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

    def __iter__(self) -> Iterator[FrameData]:
        self.keyframes_data: list[FrameData] = []
        self.n_frames = 0
        for _ in pbar(self.record_keyframes(iter(self.frame_stream)), total=len(self), desc="Collecting DAV3 keyframes"):
            pass
        yield from self.estimate_depth_sliding_window(iter(self.frame_stream))


class VipePipeline:
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

        return slam_output
