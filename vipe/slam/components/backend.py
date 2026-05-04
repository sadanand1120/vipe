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

import torch

from omegaconf import DictConfig

from ..networks.droid_net import DroidNet
from .buffer import GraphBuffer
from .factor_graph import FactorGraph


class SLAMBackend:
    """
    Mainly used to run a pretty dense bundle adjustment for all the frames in the graph.
    """

    def __init__(self, net: DroidNet, video: GraphBuffer, args: DictConfig, device: torch.device):
        self.net = net
        self.video = video
        self.args = args
        self.device = device
        self.last_graph: torch.Tensor | None = None

    def _iterate(self, graph: FactorGraph, steps: int):
        graph.update_batch(
            itrs=8,
            steps=steps,
            solver_verbose=True,
        )

    @torch.no_grad()
    def run(self, steps: int = 12):
        """main update (reset GRU state)"""

        t = self.video.n_frames

        graph = FactorGraph(
            self.net,
            self.video,
            self.device,
            max_factors=16 * t,
            incremental=False,
        )

        graph.add_proximity_factors(
            rad=self.args.backend_radius,
            nms=self.args.backend_nms,
            thresh=self.args.backend_thresh,
            beta=self.args.beta,
        )

        if len(graph.ii) > 0:
            self._iterate(graph, steps)
        else:
            # Empty graph with only one keyframe, assign sensor depth
            self.video.disps[0] = torch.where(
                self.video.disps_sens[0] > 0,
                self.video.disps_sens[0],
                self.video.disps[0],
            )

        self.last_graph = torch.stack([graph.ii, graph.jj], dim=-1)
