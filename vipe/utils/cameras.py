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

from enum import Enum

import torch


class CameraType(Enum):
    PINHOLE = "pinhole"

    def build_camera_model(self, intrinsics: torch.Tensor):
        cls = self.camera_model_cls()
        return cls(intrinsics)

    def intrinsics_dim(self) -> int:
        return self.camera_model_cls().intrinsics_dim()

    def camera_model_cls(self) -> type["BaseCameraModel"]:
        return PinholeCameraModel


class BaseCameraModel:
    """Represent a batch of camera models of the same type."""

    MIN_DEPTH: float = 0.1

    @classmethod
    def intrinsics_dim(cls) -> int:
        raise NotImplementedError

    def __init__(self, intrinsics: torch.Tensor):
        self.intrinsics = intrinsics
        assert self.intrinsics.shape[-1] == self.intrinsics_dim(), (
            f"Intrinsics should have shape (..., {self.intrinsics_dim()})"
        )

    def iproj_disp(
        self,
        disps: torch.Tensor,
        disps_u: torch.Tensor,
        disps_v: torch.Tensor,
        compute_jz: bool = False,
        compute_jf: bool = False,
    ):
        """
        Args:
            disps: (N, ...) tensor of disparities
            disps_u: (N, ...) tensor of u coordinates
            disps_v: (N, ...) tensor of v coordinates
            compute_jz: bool to compute jacobian of the disps
            compute_jf: bool to compute jacobian of the intrinsics

        Returns:
            pts: (N, ..., 4) tensor of homogeneous points
            Jz: (N, ..., 4) tensor of jacobian of the disps
            Jf: (N, ..., 4, 1+D) tensor of jacobian of the intrinsics (focal + distortion)
        """
        raise NotImplementedError

    def proj_points(
        self,
        ps: torch.Tensor,
        compute_jp: bool = False,
        compute_jf: bool = False,
        limit_min_depth: bool = True,
    ):
        """
        Args:
            ps: (N, ..., 4) tensor of homogeneous points
            compute_jp: bool to compute jacobian of the homogeneous points
            compute_jf: bool to compute jacobian of the intrinsics
            limit_min_depth: bool to limit the minimum depth to self.MIN_DEPTH

        Returns:
            coords: (N, ..., 2) tensor of coordinates
            Jp: (N, ..., 2, 4) tensor of jacobian of the homogeneous points
            Jf: (N, ..., 2, 1+D) tensor of jacobian of the focal + distortion
        """
        raise NotImplementedError

    def pinhole(self) -> "PinholeCameraModel":
        """
        Returns:
            PinholeCameraModel.
        """
        raise NotImplementedError

    def scaled(self, scale: float) -> "BaseCameraModel":
        """
        Args:
            scale: scale factor to apply to the camera model.
        """
        raise NotImplementedError

    @classmethod
    def J_scale(cls, scale: float, J: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PinholeCameraModel(BaseCameraModel):
    def __init__(self, intrinsics: torch.Tensor):
        super().__init__(intrinsics)

    @classmethod
    def intrinsics_dim(self) -> int:
        return 4

    def iproj_disp(
        self,
        disps: torch.Tensor,
        disps_u: torch.Tensor,
        disps_v: torch.Tensor,
        compute_jz: bool = False,
        compute_jf: bool = False,
    ):
        # Expand intrinsics.
        intrinsics = self.intrinsics.view((-1,) + (1,) * (disps.dim() - 1) + (4,))
        fx, fy, cx, cy = intrinsics.unbind(dim=-1)

        i = torch.ones_like(disps)
        X = (disps_u - cx) / fx
        Y = (disps_v - cy) / fy
        pts = torch.stack([X, Y, i, disps], dim=-1)

        Jz = None
        if compute_jz:
            Jz = torch.zeros_like(pts)
            Jz[..., -1] = 1.0

        Jf = None
        if compute_jf:
            Jf = torch.zeros_like(pts)[..., None]
            Jf[..., 0, 0] = -X / fx
            Jf[..., 1, 0] = -Y / fy

        return pts, Jz, Jf

    def proj_points(
        self,
        ps: torch.Tensor,
        compute_jp: bool = False,
        compute_jf: bool = False,
        limit_min_depth: bool = True,
    ):
        extra_dim_shapes = ps.shape[1:-1]
        n_extra_dim = len(extra_dim_shapes)  # Dim of "..."

        fx, fy, cx, cy = self.intrinsics.view((-1,) + (1,) * n_extra_dim + (4,)).unbind(dim=-1)
        # Ignore the last component since it will be cancelled out
        X, Y, Z, _ = ps.unbind(dim=-1)

        if limit_min_depth:
            Z = torch.where(Z < self.MIN_DEPTH, torch.ones_like(Z), Z)
        d = Z.reciprocal()

        x = fx * (X * d) + cx
        y = fy * (Y * d) + cy
        coords = torch.stack([x, y], dim=-1)

        Jp = None
        if compute_jp:
            N = d.shape[0]
            o = torch.zeros_like(d)
            Jp = torch.stack(
                [
                    fx * d,
                    o,
                    -fx * X * d * d,
                    o,
                    o,
                    fy * d,
                    -fy * Y * d * d,
                    o,
                ],
                dim=-1,
            ).view(N, *extra_dim_shapes, 2, 4)

        Jf = None
        if compute_jf:
            Jf = torch.zeros_like(coords)[..., None]
            Jf[..., 0, 0] = X * d
            Jf[..., 1, 0] = Y * d

        return coords, Jp, Jf

    def pinhole(self):
        return self

    def scaled(self, scale: float) -> "PinholeCameraModel":
        return PinholeCameraModel(self.intrinsics * scale)

    @classmethod
    def J_scale(cls, scale: float, J: torch.Tensor) -> torch.Tensor:
        return J * scale
