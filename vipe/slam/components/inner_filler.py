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
    """This class is used to fill in non-keyframe poses"""

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
        m_tstamp = self.video.tstamp[self.start_idx : total_frames]
        n_tstamp = self.video.tstamp[: self.start_idx]

        # Find left (inclusive) nearest keyframe
        t0 = torch.searchsorted(n_tstamp, m_tstamp, right=True) - 1
        t1 = torch.where(t0 < self.start_idx - 1, t0 + 1, t0)

        d_time = n_tstamp[t1] - n_tstamp[t0] + 1e-3  # Avoid if time is out of bound of kfs
        n_pose = SE3(self.video.poses[: self.start_idx])
        d_pose = n_pose[t1] * n_pose[t0].inv()
        vel = d_pose.log() / d_time.unsqueeze(-1)
        w = vel * (m_tstamp - n_tstamp[t0]).unsqueeze(-1)
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

        chunk_idx = len(self.filled_poses) + 1
        frame_start = int(m_tstamp[0].item()) if len(m_tstamp) > 0 else None
        frame_end = int(m_tstamp[-1].item()) if len(m_tstamp) > 0 else None
        infill_ba_iters = int(self.args.infill_ba_iters)
        for outer_iter in range(1, int(self.args.infill_update_steps) + 1):
            graph.update(
                self.start_idx,
                total_frames,
                itrs=infill_ba_iters,
                motion_only=True,
                ba_trace_context={
                    "stage": "infill",
                    "event": "chunk",
                    "phase": "motion_only",
                    "chunk_idx": chunk_idx,
                    "chunk_start_buffer_idx": int(self.start_idx),
                    "chunk_end_buffer_idx_exclusive": int(total_frames),
                    "chunk_frames": int(total_frames - self.start_idx),
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "outer_iter": outer_iter,
                    "outer_total": int(self.args.infill_update_steps),
                    "ba_iters": infill_ba_iters,
                    "cycle_base": (outer_iter - 1) * infill_ba_iters,
                    "num_factors": graph.num_factors,
                },
            )

        current_poses = SE3(self.video.poses[self.start_idx : total_frames].clone())
        self.filled_poses.append(current_poses)

        self.video.n_frames = self.start_idx

    def get_result(self) -> PoseInfillResult:
        return PoseInfillResult(
            poses=cat(self.filled_poses, dim=0),
        )
