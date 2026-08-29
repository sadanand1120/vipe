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

from vipe.stream import FrameDir
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar

from .components.backend import SLAMBackend
from .components.buffer import GraphBuffer
from .components.frontend import SLAMFrontend
from .components.inner_filler import InnerFiller
from .components.motion_filter import MotionFilter, MotionFilterResult
from .interface import SLAMOutput
from .networks.droid_net import DroidNet


logger = logging.getLogger(__name__)


class SLAMInputResizer:
    """Fixed source-to-SLAM resize/crop transform for one canonical scene."""

    def __init__(self, source_size: tuple[int, int], target_pixels: int) -> None:
        h0, w0 = source_size
        scale_factor = np.sqrt(int(target_pixels) / (h0 * w0))
        h1 = int(h0 * scale_factor)
        w1 = int(w0 * scale_factor)

        crop_h, crop_w = h1 % 8, w1 % 8
        self.resize_size = (h1, w1)
        self.crop_top = crop_h // 2
        self.crop_bottom = crop_h - self.crop_top
        self.crop_left = crop_w // 2
        self.crop_right = crop_w - self.crop_left
        self.output_size = (h1 - crop_h, w1 - crop_w)
        self.scale_x = w1 / w0
        self.scale_y = h1 / h0

    def _crop(self, value: torch.Tensor) -> torch.Tensor:
        bottom = value.shape[0] - self.crop_bottom
        right = value.shape[1] - self.crop_right
        return value[self.crop_top:bottom, self.crop_left:right]

    def rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        resized = torch.nn.functional.interpolate(
            rgb.permute(2, 0, 1)[None], self.resize_size, mode="bilinear"
        )[0].permute(1, 2, 0)
        return self._crop(resized)

    def depth(self, depth: torch.Tensor) -> torch.Tensor:
        resized = torch.nn.functional.interpolate(depth[None, None], self.resize_size, mode="nearest")[0, 0]
        return self._crop(resized)

    def intrinsics(self, intrinsics: torch.Tensor) -> torch.Tensor:
        resized = intrinsics.clone()
        resized[0:4:2] *= self.scale_x
        resized[1:4:2] *= self.scale_y
        resized[2] -= self.crop_left
        resized[3] -= self.crop_top
        return resized


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
            dense_disp_alpha=self.config.dense_disp_alpha,
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
        intrinsics: torch.Tensor,
        fmap: torch.Tensor | None = None,
        net: torch.Tensor | None = None,
        inp: torch.Tensor | None = None,
    ) -> int:
        kf_idx = self.buffer.n_frames
        self.buffer.frame_indices[kf_idx] = frame_idx
        self.buffer.fmaps[kf_idx] = (self.droid_net.encode_features(images) if fmap is None else fmap)[0]
        if net is None or inp is None:
            net, inp = self.droid_net.encode_context(images)
        self.buffer.nets[kf_idx], self.buffer.inps[kf_idx] = net[0], inp[0]

        if kf_idx == 0:
            self.buffer.intrinsics = intrinsics.to(self.device)

        self.buffer.n_frames += 1
        return kf_idx

    def _add_frontend_keyframe(
        self,
        frame_idx: int,
        images: torch.Tensor,
        sensor_depth: torch.Tensor,
        intrinsics: torch.Tensor,
        motion_result: MotionFilterResult,
    ):
        kf_idx = self._store_buffer_frame(
            frame_idx,
            images,
            intrinsics,
            fmap=motion_result.fmap,
            net=motion_result.net,
            inp=motion_result.inp,
        )
        self.buffer.update_disps_sens(frame_idx=kf_idx, sensor_depth=sensor_depth)

    def _add_infill_frame(self, frame_idx: int, fmap: torch.Tensor):
        frame_idx_in_buffer = self.buffer.n_frames
        self.buffer.frame_indices[frame_idx_in_buffer] = frame_idx
        self.buffer.fmaps[frame_idx_in_buffer] = fmap[0]
        self.buffer.n_frames += 1

    @torch.no_grad()
    def run(
        self,
        frame_stream: FrameDir,
        intrinsics: torch.Tensor,
    ) -> SLAMOutput:
        resizer = SLAMInputResizer(frame_stream.frame_size, self.config.resize_target_pixels)
        working_intrinsics = resizer.intrinsics(intrinsics)
        total_n_frames = len(frame_stream)

        self.config.update(
            {
                "height": resizer.output_size[0],
                "width": resizer.output_size[1],
                "camera_type": CameraType.PINHOLE,
            }
        )

        self._build_components()

        pass1_fmaps: list[torch.Tensor | None] = [None] * total_n_frames
        pass1_pbar = pbar(range(total_n_frames), desc="SLAM Pass (1/2)", total=total_n_frames)
        for frame_idx in pass1_pbar:
            rgb = resizer.rgb(frame_stream.rgb(frame_idx))
            images = rgb.permute(2, 0, 1)[None]
            motion_result = self.motion_filter.check(images)

            max_keyframe_gap = int(self.config.max_keyframe_gap)
            force_keyframe = False
            if max_keyframe_gap > 0 and self.buffer.n_frames > 0:
                last_keyframe_idx = int(self.buffer.frame_indices[self.buffer.n_frames - 1].item())
                force_keyframe = frame_idx - last_keyframe_idx >= max_keyframe_gap
            if force_keyframe and not motion_result.is_keyframe:
                motion_result = self.motion_filter.promote_keyframe(images, motion_result.fmap)
            pass1_fmaps[frame_idx] = motion_result.fmap.detach()

            if motion_result.is_keyframe or force_keyframe or frame_idx == total_n_frames - 1:
                self._add_frontend_keyframe(
                    frame_idx,
                    images,
                    resizer.depth(frame_stream.sensor_depth(frame_idx)),
                    working_intrinsics,
                    motion_result,
                )

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
        del rgb, images, motion_result
        del self.frontend, self.motion_filter

        # Run a global BA over the keyframes.
        backend_active_factors = self.backend.run(self.config.backend_iters)
        logger.info(
            "SLAM backend complete: keyframes=%d act_fac=%d",
            self.buffer.n_frames,
            backend_active_factors,
        )

        keyframe_indices = [
            int(frame_idx)
            for frame_idx in self.buffer.frame_indices[: self.buffer.n_frames].detach().cpu().tolist()
        ]

        # Produce one final pose slot per input frame from the fixed keyframe references.
        self.inner_filler.start_after_keyframes(self.buffer.n_frames)
        for frame_idx in pbar(range(total_n_frames), desc="SLAM Pass (2/2)", total=total_n_frames):
            fmap = pass1_fmaps[frame_idx]
            if fmap is None:
                raise RuntimeError(f"missing pass-1 feature map for frame {frame_idx}")
            self._add_infill_frame(frame_idx, fmap)
            pass1_fmaps[frame_idx] = None
            if self.inner_filler.chunk_ready() or frame_idx == total_n_frames - 1:
                self.inner_filler.fill_pending_chunk()

        infill_result = self.inner_filler.get_result()

        # This means the iterator is exhausted early than expected in the above loop.
        if infill_result.poses.shape[0] != total_n_frames:
            raise ValueError("Your video might be malformed or unreadable.")

        trajectory = infill_result.poses.inv()

        output = SLAMOutput(
            trajectory=trajectory,
            intrinsics=intrinsics,
            keyframe_indices=keyframe_indices,
        )
        return output
