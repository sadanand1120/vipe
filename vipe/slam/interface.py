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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vipe.ext import utils_ext
from vipe.ext.lietorch import SE3
from vipe.utils.cameras import CameraType


@dataclass(kw_only=True)
class SLAMMap:
    # (M, 3) tensor of XYZ coordinates
    dense_disp_xyz: torch.Tensor
    # (M, 3) tensor of RGB colors (0-1)
    dense_disp_rgb: torch.Tensor
    # (N, 2) range of corresponding keyframe indices
    dense_disp_packinfo: torch.Tensor
    # Actual frame indices of the dense_disp_xyz (assert sorted)
    dense_disp_frame_inds: list[int]

    def scale(self, factor: float):
        self.dense_disp_xyz *= factor

    def save(self, path: Path):
        """
        Save the SLAM map to a directory.
        """
        map_device = self.dense_disp_xyz.device
        torch.save(
            {
                "dense_disp_xyz": self.dense_disp_xyz.cpu(),
                "dense_disp_rgb": self.dense_disp_rgb.cpu(),
                "dense_disp_packinfo": self.dense_disp_packinfo.cpu(),
                "dense_disp_frame_inds": self.dense_disp_frame_inds,
                "device": map_device,
            },
            path,
        )

    @staticmethod
    def load(path: Path, device: torch.device | None = None):
        """
        Load the SLAM map from a directory.
        """
        data = torch.load(path)
        if device is None:
            device = data["device"]
        return SLAMMap(
            dense_disp_xyz=data["dense_disp_xyz"].to(device),
            dense_disp_rgb=data["dense_disp_rgb"].to(device),
            dense_disp_packinfo=data["dense_disp_packinfo"].to(device),
            dense_disp_frame_inds=data["dense_disp_frame_inds"],
        )

    @staticmethod
    def from_masked_dense_disp(
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        mask: torch.Tensor,
        tstamps: torch.Tensor,
    ):
        """
        xyz: (N, H, W, 3)
        rgb: (N, H, W, 3)
        mask: (N, H, W)
        tstamps: (N,)
        """
        assert torch.all(tstamps[1:] > tstamps[:-1]), "Timestamps should be sorted."
        N, H, W, C = xyz.shape
        xyz = xyz.reshape(-1, C)[mask.reshape(-1)]
        rgb = rgb.reshape(-1, C)[mask.reshape(-1)]
        valid_count = mask.sum([1, 2]).reshape(-1)
        packinfo = torch.stack([torch.cumsum(valid_count, 0) - valid_count, valid_count], dim=-1).reshape(N, 2)
        assert tstamps.shape[0] == N
        return SLAMMap(
            dense_disp_xyz=xyz,
            dense_disp_rgb=rgb,
            dense_disp_packinfo=packinfo,
            dense_disp_frame_inds=tstamps.tolist(),
        )

    def get_dense_disp_pcd(self, keyframe_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start, count = self.dense_disp_packinfo[keyframe_idx]
        return (
            self.dense_disp_xyz[start : start + count],
            self.dense_disp_rgb[start : start + count],
        )

    def get_dense_disp_full_pcd(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the full point cloud of the dense disparity map.
        """
        xyz_list, color_list = [], []
        for keyframe_idx in range(len(self.dense_disp_frame_inds)):
            xyz, color = self.get_dense_disp_pcd(keyframe_idx)
            xyz_list.append(xyz)
            color_list.append(color)
        return torch.cat(xyz_list, dim=0), torch.cat(color_list, dim=0)

    def project_map(
        self,
        frame_tstamp: int,
        target_size: tuple[int, int],
        target_intrinsics: torch.Tensor,
        target_pose: SE3,
        target_camera_type: CameraType,
        infill: bool = False,
        tstamp_nn: int = 3,
    ) -> torch.Tensor:
        right_keyframe_idx = np.searchsorted(self.dense_disp_frame_inds, frame_tstamp).item()
        right_keyframe_idx = min(right_keyframe_idx + tstamp_nn, len(self.dense_disp_frame_inds) - 1)
        left_keyframe_idx = max(right_keyframe_idx - 2 * tstamp_nn, 0)

        xyz_list = []
        for keyframe_idx in range(left_keyframe_idx, right_keyframe_idx + 1):
            xyz, _ = self.get_dense_disp_pcd(keyframe_idx)
            xyz_list.append(xyz)
        all_xyz = torch.cat(xyz_list, dim=0)

        target_pose_mat = target_pose.inv().matrix()
        all_xyz = all_xyz @ target_pose_mat[:3, :3].T + target_pose_mat[:3, 3]

        xyz_h = torch.cat(
            [all_xyz, torch.ones(all_xyz.shape[0], device="cuda").unsqueeze(-1)],
            dim=-1,
        )
        disp = 1.0 / all_xyz[:, 2]

        camera_model = target_camera_type.build_camera_model(target_intrinsics)
        uv, _, _ = camera_model.proj_points(xyz_h, limit_min_depth=False)
        uu, vv = uv[..., 0], uv[..., 1]

        in_mask = (uu > 0) & (uu < target_size[1]) & (vv > 0) & (vv < target_size[0]) & (disp > 0)
        uu, vv, depth = uu[in_mask], vv[in_mask], disp[in_mask].reciprocal()

        if not infill:
            target_depth = torch.zeros(target_size, device="cuda")
            target_depth[vv.floor().long(), uu.floor().long()] = depth
        else:
            tree = torch.stack((uu, vv), dim=-1)
            query = torch.stack(
                torch.meshgrid(
                    torch.arange(target_size[1], device="cuda").float() + 0.5,
                    torch.arange(target_size[0], device="cuda").float() + 0.5,
                    indexing="xy",
                ),
                dim=-1,
            ).reshape(-1, 2)
            _, inds = utils_ext.nearest_neighbours(query, tree, 1)
            target_depth = depth[inds.view(-1)].reshape(target_size)
        return target_depth


@dataclass(kw_only=True)
class SLAMOutput:
    trajectory: SE3  # (N,)
    intrinsics: torch.Tensor  # (4,)

    slam_map: SLAMMap | None = None

    @property
    def keyframe_ids(self) -> np.ndarray:
        assert self.slam_map is not None, "SLAM map not available."
        return np.array(self.slam_map.dense_disp_frame_inds)
