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
# -------------------------------------------------------------------------------------------------
# This file includes code originally from the DROID-SLAM repository:
# https://github.com/cvg/DROID-SLAM
# Licensed under the MIT License. See THIRD_PARTY_LICENSES.md for details.
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass

import torch

from vipe.ext.lietorch.groups import SE3, cat

from ..networks.droid_net import DroidNet
from .buffer import GraphBuffer
from .factor_graph import FactorGraph


@dataclass
class PoseInfillResult:
    poses: SE3  # Inverse of c2w

    def scale(self, factor: float):
        self.poses.data[..., :3] *= factor


class InnerFiller:
    """Optimize the final full-frame trajectory against fixed keyframe references."""

    def __init__(self, net: DroidNet, video: GraphBuffer, args, device: torch.device):
        self.video = video
        self.net = net
        self.device = device
        self.start_idx = -1
        self.args = args

        self.filled_poses = []

    def start_after_keyframes(self, start_idx: int):
        self.start_idx = start_idx

    def chunk_ready(self) -> bool:
        assert self.start_idx >= 0
        return self.video.n_frames - self.start_idx >= self.args.infill_chunk_size

    def fill_pending_chunk(self):
        total_frames = self.video.n_frames

        # Setup initial value (for pose and disp)
        pending_frame_indices = self.video.frame_indices[self.start_idx : total_frames]
        keyframe_indices = self.video.frame_indices[: self.start_idx]

        # Find left (inclusive) nearest keyframe
        t0 = torch.searchsorted(keyframe_indices, pending_frame_indices, right=True) - 1
        t1 = torch.where(t0 < self.start_idx - 1, t0 + 1, t0)

        frame_gap = keyframe_indices[t1] - keyframe_indices[t0] + 1e-3
        n_pose = SE3(self.video.poses[: self.start_idx])
        d_pose = n_pose[t1] * n_pose[t0].inv()
        pose_step = d_pose.log() / frame_gap.unsqueeze(-1)
        w = pose_step * (pending_frame_indices - keyframe_indices[t0]).unsqueeze(-1)
        m_pose = SE3.exp(w) * n_pose[t0]

        self.video.poses[self.start_idx : total_frames] = m_pose.data

        # Build factor graph and optimize for the interpolated information.
        graph = FactorGraph(
            self.net,
            self.video,
            self.device,
            max_factors=-1,
            incremental=True,
        )
        infill_inds = torch.arange(self.start_idx, total_frames).to(self.device)
        graph.add_factors(t0, infill_inds)
        graph.add_factors(t1, infill_inds)

        infill_ba_iters = int(self.args.infill_ba_iters)
        for _ in range(int(self.args.infill_update_steps)):
            graph.update(
                self.start_idx,
                total_frames,
                itrs=infill_ba_iters,
                motion_only=True,
            )

        current_poses = SE3(self.video.poses[self.start_idx : total_frames].clone())
        self.filled_poses.append(current_poses)

        self.video.n_frames = self.start_idx

    def get_result(self) -> PoseInfillResult:
        return PoseInfillResult(
            poses=cat(self.filled_poses, dim=0),
        )
