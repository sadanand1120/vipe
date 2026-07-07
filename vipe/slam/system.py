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

import numpy as np
import torch

from vipe.streams.base import FrameData, FrameStream
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional

from .components.backend import SLAMBackend
from .components.buffer import GraphBuffer
from .components.frontend import SLAMFrontend
from .components.inner_filler import InnerFiller
from .components.motion_filter import MotionFilter, MotionFilterResult
from .interface import SLAMOutput
from .networks.droid_net import DroidNet


logger = logging.getLogger(__name__)


class StandardResizeFrameProcessor:
    def __init__(self, target_pixels: int = 384 * 512) -> None:
        super().__init__()
        self.fac_x, self.fac_y = 1.0, 1.0
        self.target_pixels = int(target_pixels)

    def _compute_frame_size_crop(self, previous_frame_size: tuple[int, int]):
        h0, w0 = previous_frame_size
        scale_factor = np.sqrt(self.target_pixels / (h0 * w0))
        h1 = int(h0 * scale_factor)
        w1 = int(w0 * scale_factor)

        crop_h, crop_w = h1 % 8, w1 % 8
        crop_top, crop_bottom = crop_h // 2, crop_h - crop_h // 2
        crop_left, crop_right = crop_w // 2, crop_w - crop_w // 2

        self.fac_x, self.fac_y = w0 / w1, h0 / h1
        self.scx, self.scy = crop_left, crop_top
        return (h1, w1), (crop_top, crop_bottom, crop_left, crop_right)

    def update_frame_size(self, previous_frame_size: tuple[int, int]):
        (h1, w1), (crop_top, crop_bottom, crop_left, crop_right) = self._compute_frame_size_crop(previous_frame_size)
        return h1 - (crop_top + crop_bottom), w1 - (crop_left + crop_right)

    def __call__(self, frame_data: FrameData) -> FrameData:
        (h1, w1), (crop_top, crop_bottom, crop_left, crop_right) = self._compute_frame_size_crop(frame_data.size())
        frame_data = frame_data.resize((h1, w1))
        frame_data = frame_data.crop(top=crop_top, bottom=crop_bottom, left=crop_left, right=crop_right)
        return frame_data

    def sensor_depth(self, sensor_depth: torch.Tensor) -> torch.Tensor:
        (h1, w1), (crop_top, crop_bottom, crop_left, crop_right) = self._compute_frame_size_crop(sensor_depth.shape)
        depth = torch.nn.functional.interpolate(sensor_depth[None, None], (h1, w1), mode="nearest")[0, 0]
        bottom = depth.shape[0] - crop_bottom
        right = depth.shape[1] - crop_right
        return depth[crop_top:bottom, crop_left:right]

    def recover_intrinsics(self, after_intrinsics: torch.Tensor) -> torch.Tensor:
        new_intrinsics = after_intrinsics.clone()
        new_intrinsics[2] += self.scx
        new_intrinsics[3] += self.scy
        new_intrinsics[0:4:2] *= self.fac_x
        new_intrinsics[1:4:2] *= self.fac_y
        return new_intrinsics


class SLAMSystem:
    """Solver-defined SLAM"""

    def __init__(
        self,
        device: torch.device,
        config,
    ) -> None:
        self.device = device
        self.config = config.copy()

    def _build_components(self):
        self.droid_net = DroidNet().to(self.device)
        self.buffer = GraphBuffer(
            height=self.config.height,
            width=self.config.width,
            buffer_size=self.config.buffer,
            init_disp=self.config.init_disp,
            ba_config=self.config.ba,
            camera_type=self.config.camera_type,
            device=self.device,
        )
        self.motion_filter = MotionFilter(
            self.droid_net,
            thresh=self.config.filter_thresh,
            device=self.device,
        )
        self.frontend = SLAMFrontend(self.droid_net, self.buffer, self.config, device=self.device)
        self.backend = SLAMBackend(self.droid_net, self.buffer, self.config, device=self.device)
        self.inner_filler = InnerFiller(self.droid_net, self.buffer, self.config, device=self.device)

    def _store_buffer_frame(
        self,
        frame_idx: int,
        images: torch.Tensor,
        frame_data: FrameData,
        fmap: torch.Tensor | None = None,
        net: torch.Tensor | None = None,
        inp: torch.Tensor | None = None,
    ) -> int:
        kf_idx = self.buffer.n_frames
        self.buffer.tstamp[kf_idx] = frame_idx
        self.buffer.fmaps[kf_idx] = (self.droid_net.encode_features(images) if fmap is None else fmap)[0]
        if net is None or inp is None:
            net, inp = self.droid_net.encode_context(images)
        self.buffer.nets[kf_idx], self.buffer.inps[kf_idx] = net[0], inp[0]

        if kf_idx == 0:
            self.buffer.intrinsics = unpack_optional(frame_data.intrinsics).to(self.device)

        self.buffer.n_frames += 1
        return kf_idx

    def _add_frontend_keyframe(
        self,
        frame_idx: int,
        images: torch.Tensor,
        frame_data: FrameData,
        motion_result: MotionFilterResult,
    ):
        kf_idx = self._store_buffer_frame(
            frame_idx,
            images,
            frame_data,
            fmap=motion_result.fmap,
            net=motion_result.net,
            inp=motion_result.inp,
        )
        self.buffer.update_disps_sens(frame_idx=kf_idx, frame_data=frame_data)

    def _add_infill_frame(self, frame_idx: int, fmap: torch.Tensor):
        frame_idx_in_buffer = self.buffer.n_frames
        self.buffer.tstamp[frame_idx_in_buffer] = frame_idx
        self.buffer.fmaps[frame_idx_in_buffer] = fmap[0]
        self.buffer.n_frames += 1

    def _rgb_bchw(self, frame_data: FrameData):
        images = frame_data.rgb.permute(2, 0, 1)[None]
        return images

    @staticmethod
    def _attach_intrinsics(frame_data: FrameData, intrinsics: torch.Tensor, camera_type: CameraType) -> FrameData:
        frame_data.intrinsics = intrinsics
        frame_data.camera_type = camera_type
        return frame_data

    @torch.no_grad()
    def run(
        self,
        frame_stream: FrameStream,
        intrinsics: torch.Tensor,
        camera_type: CameraType = CameraType.PINHOLE,
    ) -> SLAMOutput:
        resizer = StandardResizeFrameProcessor(self.config.resize_target_pixels)
        frame_size = resizer.update_frame_size(frame_stream.frame_size())
        total_n_frames = len(frame_stream)

        self.config.update(
            {
                "height": frame_size[0],
                "width": frame_size[1],
                "camera_type": camera_type,
            }
        )

        self._build_components()

        pass1_fmaps: list[torch.Tensor | None] = [None] * total_n_frames
        pass1_pbar = pbar(range(total_n_frames), desc="SLAM Pass (1/2)", total=total_n_frames)
        for frame_idx in pass1_pbar:
            frame_data = frame_stream.rgb_frame(frame_idx)
            frame_data = self._attach_intrinsics(frame_data, intrinsics, camera_type)
            frame_data = resizer(frame_data)
            images = self._rgb_bchw(frame_data)
            motion_result = self.motion_filter.check(images)

            max_keyframe_gap = int(self.config.max_keyframe_gap)
            force_keyframe = False
            if max_keyframe_gap > 0 and self.buffer.n_frames > 0:
                last_keyframe_idx = int(self.buffer.tstamp[self.buffer.n_frames - 1].item())
                force_keyframe = frame_idx - last_keyframe_idx >= max_keyframe_gap
            if force_keyframe and not motion_result.is_keyframe:
                motion_result = self.motion_filter.promote_keyframe(images, motion_result.fmap)
            pass1_fmaps[frame_idx] = motion_result.fmap.detach()

            if motion_result.is_keyframe or force_keyframe or frame_idx == total_n_frames - 1:
                keyframe_data = FrameData(
                    raw_frame_idx=frame_data.raw_frame_idx,
                    rgb=frame_data.rgb,
                    sensor_depth=resizer.sensor_depth(frame_stream.sensor_depth(frame_idx)),
                    intrinsics=frame_data.intrinsics,
                    camera_type=camera_type,
                )
                self._add_frontend_keyframe(frame_idx, images, keyframe_data, motion_result)

            self.frontend.run()

            if hasattr(pass1_pbar, "set_postfix"):
                pass1_pbar.set_postfix(
                    kf=self.buffer.n_frames,
                    act_fac=self.frontend.graph.num_factors,
                )

        logger.info(
            "SLAM pass 1 complete: keyframes=%d act_fac=%d",
            self.buffer.n_frames,
            self.frontend.graph.num_factors,
        )

        # Run a global BA over the keyframes.
        backend_active_factors = self.backend.run(self.config.backend_iters)
        logger.info(
            "SLAM backend complete: keyframes=%d act_fac=%d",
            self.buffer.n_frames,
            backend_active_factors,
        )

        keyframe_indices = [int(t) for t in self.buffer.tstamp[: self.buffer.n_frames].detach().cpu().tolist()]

        # Infill poses and attributes for non-keyframe frames.
        self.inner_filler.start_after_keyframes(self.buffer.n_frames)
        for frame_idx in pbar(range(total_n_frames), desc="SLAM Pass (2/2)", total=total_n_frames):
            fmap = pass1_fmaps[frame_idx]
            if fmap is None:
                raise RuntimeError(f"missing pass-1 feature map for frame {frame_idx}")
            self._add_infill_frame(frame_idx, fmap)
            if self.inner_filler.chunk_ready() or frame_idx == total_n_frames - 1:
                self.inner_filler.fill_pending_chunk()

        infill_result = self.inner_filler.get_result()

        # This means the iterator is exhausted early than expected in the above loop.
        if infill_result.poses.shape[0] != total_n_frames:
            raise ValueError("Your video might be malformed or unreadable.")

        # Scale back the intrinsics to the original size.
        original_intrinsics = resizer.recover_intrinsics(self.buffer.intrinsics)
        trajectory = infill_result.poses.inv()

        output = SLAMOutput(
            trajectory=trajectory,
            intrinsics=original_intrinsics,
            keyframe_indices=keyframe_indices,
        )
        return output
