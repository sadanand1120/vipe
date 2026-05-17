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

import logging

import torch

from einops import rearrange
from omegaconf.dictconfig import DictConfig

from vipe.ext.lietorch.groups import SE3
from vipe.priors.depth.base import DepthEstimationInput, DepthEstimationModel
from vipe.streams.base import FrameData
from vipe.utils.cameras import CameraType

from ..ba.solver import Solver, SparseBlockVector
from ..ba.terms import DenseDepthFlowTerm, DispSensRegularizationTerm
from ..maths import geom
from ..maths.retractor import DenseDispRetractor, PoseRetractor


logger = logging.getLogger(__name__)


class GraphBuffer:
    def __init__(
        self,
        height: int,
        width: int,
        buffer_size: int,
        init_disp: float,
        ba_config: DictConfig,
        camera_type: CameraType,
        device: torch.device = torch.device("cuda"),
    ):
        self.n_frames: int = 0

        self.height = height
        self.width = width
        self.device = device
        self.ba_config = ba_config
        self.camera_type = camera_type

        assert self.height % 8 == 0 and self.width % 8 == 0
        dd_height, dd_width = self.height // 8, self.width // 8

        # Frame index in the original stream.
        self.tstamp = torch.zeros(buffer_size, device=device, dtype=torch.int)

        # RGB image at resized SLAM resolution, shape: (frame, channel, height, width).
        self.images = torch.zeros(buffer_size, 3, self.height, self.width, device=device, dtype=torch.float16)

        # World-to-camera pose for each buffered frame.
        self.poses = torch.zeros(buffer_size, 7, device=device, dtype=torch.float)
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)

        # Shared pinhole intrinsics for this single stream.
        self.intrinsics = torch.zeros(self.camera_type.intrinsics_dim(), device=device, dtype=torch.float)

        # Dense disparity tensors at 1/8 SLAM resolution.
        self.disps = torch.ones(buffer_size, dd_height, dd_width, device=device, dtype=torch.float) * init_disp
        self.disps_sens = torch.zeros(buffer_size, dd_height, dd_width, device=device, dtype=torch.float)
        self.disps_sens_weight = torch.ones(buffer_size, dd_height, dd_width, device=device, dtype=torch.float)

        # DROID feature/context state, all at 1/8 SLAM resolution.
        self.fmaps = torch.zeros(buffer_size, 128, dd_height, dd_width, device=device, dtype=torch.half)
        self.nets = torch.zeros(buffer_size, 128, dd_height, dd_width, device=device, dtype=torch.half)
        self.inps = torch.zeros(buffer_size, 128, dd_height, dd_width, device=device, dtype=torch.half)

    def remove_second_newest(self, ix: int):
        assert ix == self.n_frames - 2
        self.tstamp[ix] = self.tstamp[ix + 1]
        self.images[ix] = self.images[ix + 1]
        self.poses[ix] = self.poses[ix + 1]
        self.disps[ix] = self.disps[ix + 1]
        self.disps_sens[ix] = self.disps_sens[ix + 1]
        self.disps_sens_weight[ix] = self.disps_sens_weight[ix + 1]
        self.nets[ix] = self.nets[ix + 1]
        self.inps[ix] = self.inps[ix + 1]
        self.fmaps[ix] = self.fmaps[ix + 1]
        self.n_frames -= 1

    def update_disps_sens(self, depth_model: DepthEstimationModel, frame_idx: int, frame_data: FrameData):
        depth_input = DepthEstimationInput(
            rgb=self.images[frame_idx].moveaxis(0, -1).float(),
            intrinsics=self.intrinsics,
            sensor_depth=frame_data.sensor_depth,
            camera_type=self.camera_type,
        )
        result = depth_model.estimate(depth_input)
        metric_depth = result.metric_depth
        assert metric_depth is not None
        disp_sens = metric_depth[3::8, 3::8]
        disp_sens = torch.where(disp_sens > 0, disp_sens.reciprocal(), disp_sens)
        self.disps_sens[frame_idx] = disp_sens
        if result.valid_mask is None:
            self.disps_sens_weight[frame_idx] = 1.0
        else:
            self.disps_sens_weight[frame_idx] = result.valid_mask[3::8, 3::8].float()

    def bundle_adjustment(
        self,
        target: torch.Tensor,
        weight: torch.Tensor,
        disp_damping: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        t0: int,
        t1: int,
        n_iters: int,
        pose_damping: float,
        pose_ep: float,
        motion_only: bool,
        verbose: bool,
    ):
        assert t0 <= t1
        weight_dense_disp = 0.001

        di = ii
        di_unique = torch.unique(di)
        pose_i_unique = torch.unique(ii)

        solver = Solver(compute_energy=verbose)
        solver.add_term(
            DenseDepthFlowTerm(
                pose_i_inds=ii,
                pose_j_inds=jj,
                dense_disp_i_inds=di,
                target=target,
                weight=weight_dense_disp * weight,
                intrinsics=self.intrinsics,
                intrinsics_factor=8.0,
                image_size=(self.height // 8, self.width // 8),
                camera_type=self.camera_type,
            )
        )

        solver.set_fixed(
            "pose",
            (torch.cat([pose_i_unique[pose_i_unique < t0], pose_i_unique[pose_i_unique >= t1]]) if t0 < t1 else None),
        )
        solver.set_retractor("pose", PoseRetractor())
        solver.set_damping("pose", damping=pose_damping, ep=pose_ep)

        if not motion_only:
            disps_sens = rearrange(self.disps_sens, "n h w -> n (h w)")
            disps_sens_weight = rearrange(self.disps_sens_weight, "n h w -> n (h w)")
            sens_i_inds = di_unique[disps_sens[di_unique].sum(1) > 0.0]
            if len(sens_i_inds) > 0:
                solver.add_term(
                    DispSensRegularizationTerm(
                        i_inds=sens_i_inds,
                        alpha=self.ba_config.dense_disp_alpha,
                        disps_sens=disps_sens,
                        disps_sens_weight=disps_sens_weight,
                    )
                )
            solver.set_retractor("dense_disp", DenseDispRetractor())
            disp_damping = rearrange(disp_damping, "n h w -> n (h w)")
            solver.set_damping(
                "dense_disp",
                damping=SparseBlockVector(
                    inds=di_unique,
                    data=0.2 * disp_damping[di_unique] + 1e-7,
                ),
                ep=1e-7,
            )
        else:
            solver.set_fixed("dense_disp")
        solver.set_marginilized("dense_disp")

        disps_flattened = rearrange(self.disps, "n h w -> n (h w)")

        ba_energy = []
        for _ in range(n_iters):
            cur_energy = solver.run_inplace(
                {
                    "pose": SE3(self.poses),
                    "dense_disp": disps_flattened,
                }
            )
            ba_energy.append(cur_energy)

        if verbose:
            logger.info(f"BA iters = {n_iters}, energy: {ba_energy[0]} -> {ba_energy[-1]}")

        self.disps.clamp_(min=0.001)

    def reproject_dense_disp(self, ii: torch.Tensor, jj: torch.Tensor):
        """Project each source dense-disparity map from frame ii into frame jj."""
        ii = ii.reshape(-1).to(device=self.device, dtype=torch.long)
        jj = jj.reshape(-1).to(device=self.device, dtype=torch.long)
        intrinsics = self.camera_type.build_camera_model(self.intrinsics).scaled(1 / 8.0).intrinsics
        coords, valid_mask, _, _ = geom.iproj_i_proj_j_disp(
            SE3(self.poses),
            self.disps,
            None,
            intrinsics,
            self.camera_type,
            ii,
            jj,
            ii,
            jacobian_p_d=False,
            jacobian_f=False,
        )
        return coords, valid_mask

    def frame_distance_dense_disp(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        beta: float = 0.3,
        bidirectional=True,
    ):
        ii = ii.reshape(-1).to(device=self.device, dtype=torch.long)
        jj = jj.reshape(-1).to(device=self.device, dtype=torch.long)
        poses = self.poses[: self.n_frames]
        intrinsics = self.camera_type.build_camera_model(self.intrinsics).scaled(1 / 8.0).intrinsics

        d = geom.frame_distance_dense_disp(
            SE3(poses),
            self.disps[: self.n_frames],
            intrinsics,
            self.camera_type,
            ii,
            jj,
            ii,
            beta,
        )

        if bidirectional:
            d2 = geom.frame_distance_dense_disp(
                SE3(poses),
                self.disps[: self.n_frames],
                intrinsics,
                self.camera_type,
                jj,
                ii,
                jj,
                beta,
            )
            d = 0.5 * (d + d2)

        return d
