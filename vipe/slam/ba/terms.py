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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from einops import rearrange

from vipe.ext.lietorch.groups import SE3
from vipe.utils.cameras import BaseCameraModel, CameraType

from ..maths import geom
from ..maths.matrix import SparseBlockMatrix, SparseDenseBlockMatrix, SparseMDiagonalBlockMatrix
from ..maths.vector import SparseBlockVector
from .kernel import RobustKernel


class TermEvalReturn(ABC):
    @abstractmethod
    def jtwj(self, group_name_row: str, group_name_col: str) -> SparseBlockMatrix: ...

    @abstractmethod
    def nwjtr(self, group_name: str) -> SparseBlockVector: ...

    @abstractmethod
    def remove_jcol_inds(self, group_name: str, col_inds: torch.Tensor): ...

    @abstractmethod
    def residual(self) -> torch.Tensor: ...

    def apply_robust_kernel(self, kernel: RobustKernel):
        raise NotImplementedError


@dataclass(kw_only=True)
class ConcreteTermEvalReturn(TermEvalReturn):
    J: dict[str, SparseBlockMatrix]  # group_name -> (n_occ, res_dim, manifold_dim)
    w: torch.Tensor  # (n_terms, res_dim, )
    r: torch.Tensor  # (n_terms, res_dim, )

    # n_occ = number of occurrences of this group_name in all the terms.
    # i.e. the number of blocks in the sparse Jacobian matrix with size n_terms x n_vars

    def jtwj(self, group_name_row: str, group_name_col: str) -> SparseBlockMatrix:
        wJ = self.J[group_name_col].scale_w_left(self.w)
        try:
            return self.J[group_name_row].tmult_mat(wJ).coalesce()
        except NotImplementedError:
            return wJ.tmult_mat(self.J[group_name_row]).transpose().coalesce()

    def nwjtr(self, group_name: str) -> SparseBlockVector:
        return self.J[group_name].tmult_vec(-self.w * self.r).coalesce()

    def remove_jcol_inds(self, group_name: str, col_inds: torch.Tensor):
        j_group = self.J[group_name]
        keep_mask = torch.isin(j_group.j_inds, col_inds, invert=True)
        self.J[group_name] = j_group.subset(keep_mask)

    def apply_robust_kernel(self, kernel: RobustKernel):
        robust_weight = kernel.apply(self.r)
        self.w = self.w * robust_weight

    def residual(self) -> torch.Tensor:
        return torch.sum(self.r * self.r * self.w, dim=1)


class SolverTerm(ABC):
    @abstractmethod
    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn: ...

    @abstractmethod
    def group_names(self) -> set[str]: ...

    def update(self, solver):
        # Default implementation do nothing.
        pass


class DenseDepthFlowTerm(SolverTerm):
    """
    E(pose_pi, pose_pj, dense_disp_di, intrinsics) = \
        proj(pose_j * pose_i.inv(), dense_disp_di) - target_[ij di]

        Pose is the world2cam transform.
        target_[ij di] is the target projected location.
    res_dim = H*W*2
    """

    def __init__(
        self,
        pose_i_inds: torch.Tensor,
        pose_j_inds: torch.Tensor,
        dense_disp_i_inds: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        intrinsics: torch.Tensor,
        intrinsics_factor: float,
        image_size: tuple[int, int],
        camera_type: CameraType,
    ) -> None:
        super().__init__()

        self.n_terms = pose_i_inds.shape[0]
        assert pose_i_inds.shape == (self.n_terms,)
        assert pose_j_inds.shape == (self.n_terms,)
        assert dense_disp_i_inds.shape == (self.n_terms,)

        self.pose_i_inds = pose_i_inds
        self.pose_j_inds = pose_j_inds
        self.dense_disp_i_inds = dense_disp_i_inds
        self.image_size = image_size
        self.camera_type = camera_type

        n_pixels = image_size[0] * image_size[1]

        self.target = target.reshape(self.n_terms, n_pixels, 2)  # (n_terms, H*W, 2)
        self.weight = weight.reshape(self.n_terms, n_pixels, 2)  # (n_terms, H*W, 2)
        self.intrinsics = intrinsics.reshape(-1, 4)  # (Q, 4)
        self.intrinsics_factor = intrinsics_factor

    def group_names(self) -> set[str]:
        return {"pose", "dense_disp"}

    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn:
        """
        variables contain:
            - pose: (n_var, ) SE3 of poses
            - dense_disp: (n_var, H*W) tensor of disparities
            - intrinsics are fixed and stored in this term

        # TODO: To accelerate, you can return a PrecomputedTermEvalReturn with kernels from Droid-SLAM.
        """
        pose, dense_disp = variables["pose"], variables["dense_disp"]
        intrinsics = self.intrinsics

        assert isinstance(pose, SE3) and isinstance(dense_disp, torch.Tensor)
        assert dense_disp.shape[1] == self.image_size[0] * self.image_size[1]

        camera_model_cls = self.camera_type.camera_model_cls()

        coords, valid, (Ji, Jj, Jz), _ = geom.iproj_i_proj_j_disp(
            pose,
            dense_disp.view(-1, self.image_size[0], self.image_size[1]),
            None,
            (camera_model_cls(intrinsics).scaled(1.0 / self.intrinsics_factor).intrinsics),
            self.camera_type,
            self.pose_i_inds,
            self.pose_j_inds,
            self.dense_disp_i_inds,
            jacobian_p_d=jacobian,
            jacobian_f=False,
        )
        coords = rearrange(coords, "n h w c -> n (h w) c", c=2)
        weight = rearrange(valid, "n h w 1 -> n (h w) 1") * self.weight  # (n_terms, H*W, 2)
        weight = rearrange(weight, "n hw c -> n (hw c)", c=2)

        J_dict = {}
        if jacobian:
            assert Ji is not None and Jj is not None and Jz is not None
            Ji = rearrange(Ji, "n h w c d -> n (h w c) d", c=2, d=6)
            Jj = rearrange(Jj, "n h w c d -> n (h w c) d", c=2, d=6)
            Jz = rearrange(Jz, "n h w c d -> n (h w) (c d)", c=2, d=1)
            term_inds = torch.arange(self.n_terms).to(pose.device)
            J_dict = {
                "pose": SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),
                    data=torch.cat([Ji, Jj], dim=0),
                ),
                "dense_disp": SparseMDiagonalBlockMatrix(
                    i_inds=term_inds,
                    j_inds=self.dense_disp_i_inds,
                    data=Jz,
                ),
            }
        return ConcreteTermEvalReturn(
            J=J_dict,
            w=weight,
            r=rearrange(coords - self.target, "n hw c -> n (hw c)", c=2),
        )


class DispSensRegularizationTerm(SolverTerm):
    """
    E(dense_disp_i) = dense_disp_i - dense_disps_sens_i
    res_dim = H*W
    """

    @dataclass(kw_only=True)
    class ThisTermEvalReturn(TermEvalReturn):
        alpha: float
        i_inds: torch.Tensor
        disps_sens_res: torch.Tensor
        disps_sens_weight: torch.Tensor

        def jtwj(self, group_name_row: str, group_name_col: str) -> SparseBlockMatrix:
            assert group_name_row == group_name_col == "dense_disp"
            return SparseMDiagonalBlockMatrix(
                i_inds=self.i_inds,
                j_inds=self.i_inds,
                data=(self.alpha * self.disps_sens_weight).unsqueeze(-1),
            )

        def nwjtr(self, group_name: str) -> SparseBlockVector:
            assert group_name == "dense_disp"
            return SparseBlockVector(
                inds=self.i_inds,
                data=-self.alpha * self.disps_sens_weight * self.disps_sens_res,
            )

        def remove_jcol_inds(self, group_name: str, col_inds: torch.Tensor):
            assert group_name == "dense_disp"
            keep_mask = torch.isin(self.i_inds, col_inds, invert=True)
            self.i_inds = self.i_inds[keep_mask]
            self.disps_sens_res = self.disps_sens_res[keep_mask]
            self.disps_sens_weight = self.disps_sens_weight[keep_mask]

        def residual(self) -> torch.Tensor:
            return self.alpha * (self.disps_sens_weight * self.disps_sens_res**2).sum(dim=1)

    def __init__(
        self,
        i_inds: torch.Tensor,
        alpha: float,
        disps_sens: torch.Tensor,
        disps_sens_weight: torch.Tensor,
    ) -> None:
        super().__init__()

        self.i_inds = i_inds
        self.alpha = alpha
        self.disps_sens = disps_sens
        self.disps_sens_weight = disps_sens_weight

    def group_names(self) -> set[str]:
        return {"dense_disp"}

    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn:
        """
        variables contain:
            - dense_disp: (n_var, H*W) tensor of disparities
        """
        dense_disp = variables["dense_disp"]

        assert isinstance(dense_disp, torch.Tensor)
        assert dense_disp.shape == self.disps_sens.shape
        assert dense_disp.shape == self.disps_sens_weight.shape

        return self.ThisTermEvalReturn(
            alpha=self.alpha,
            i_inds=self.i_inds,
            disps_sens_res=dense_disp[self.i_inds] - self.disps_sens[self.i_inds],
            disps_sens_weight=self.disps_sens_weight[self.i_inds],
        )


class PoseSmoothnessTerm(SolverTerm):
    """
    Weak adjacent-pose velocity prior.

    The Jacobian is a first-order approximation in SE(3) tangent space. It is
    intentionally weak; dense flow factors should dominate whenever they are
    well-conditioned.
    """

    def __init__(
        self,
        pose_i_inds: torch.Tensor,
        pose_j_inds: torch.Tensor,
        alpha: float,
        scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_terms = pose_i_inds.shape[0]
        assert pose_i_inds.shape == (self.n_terms,)
        assert pose_j_inds.shape == (self.n_terms,)
        assert scale.shape == (self.n_terms,)

        self.pose_i_inds = pose_i_inds
        self.pose_j_inds = pose_j_inds
        self.alpha = alpha
        self.scale = scale

    def group_names(self) -> set[str]:
        return {"pose"}

    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn:
        pose = variables["pose"]
        assert isinstance(pose, SE3)

        rel = pose[self.pose_j_inds] * pose[self.pose_i_inds].inv()
        residual = rel.log() * self.scale[:, None]

        j_dict = {}
        if jacobian:
            term_inds = torch.arange(self.n_terms, device=self.pose_i_inds.device)
            eye = torch.eye(6, device=self.pose_i_inds.device, dtype=residual.dtype)
            scaled_eye = self.scale[:, None, None] * eye[None]
            j_dict["pose"] = SparseDenseBlockMatrix(
                i_inds=torch.cat([term_inds, term_inds]),
                j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),
                data=torch.cat([-scaled_eye, scaled_eye], dim=0),
            )

        return ConcreteTermEvalReturn(
            J=j_dict,
            w=self.alpha * torch.ones_like(residual),
            r=residual,
        )


class SensorDepthGeometryTerm(SolverTerm):
    """Backend-only point-to-plane RGB-D geometry term on the fixed sensor-depth grid."""

    def __init__(
        self,
        pose_i_inds: torch.Tensor,
        pose_j_inds: torch.Tensor,
        sensor_disps: torch.Tensor,
        sensor_weights: torch.Tensor,
        intrinsics: torch.Tensor,
        intrinsics_factor: float,
        image_size: tuple[int, int],
        camera_type: CameraType,
        alpha: float,
        points_per_factor: int,
        max_residual_m: float,
    ) -> None:
        super().__init__()
        self.n_terms = pose_i_inds.shape[0]
        assert pose_i_inds.shape == (self.n_terms,)
        assert pose_j_inds.shape == (self.n_terms,)

        self.pose_i_inds = pose_i_inds
        self.pose_j_inds = pose_j_inds
        self.sensor_disps = sensor_disps
        self.sensor_weights = sensor_weights
        self.intrinsics = intrinsics.reshape(-1, camera_type.intrinsics_dim())
        self.intrinsics_factor = intrinsics_factor
        self.image_size = image_size
        self.camera_type = camera_type
        self.alpha = float(alpha)
        self.max_residual_m = float(max_residual_m)

        self.sample_xy = self._sample_grid(image_size, int(points_per_factor), pose_i_inds.device)

    @staticmethod
    def _sample_grid(image_size: tuple[int, int], points_per_factor: int, device: torch.device) -> torch.Tensor:
        height, width = image_size
        points_per_factor = max(1, points_per_factor)
        stride = max(1, int(round((height * width / points_per_factor) ** 0.5)))
        y, x = torch.meshgrid(
            torch.arange(0, height, stride, device=device),
            torch.arange(0, width, stride, device=device),
            indexing="ij",
        )
        xy = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
        if xy.shape[0] > points_per_factor:
            keep = torch.linspace(0, xy.shape[0] - 1, points_per_factor, device=device).round().long()
            xy = xy[keep]
        return xy

    def group_names(self) -> set[str]:
        return {"pose"}

    def _intrinsics_for(self, inds: torch.Tensor) -> torch.Tensor:
        if self.intrinsics.shape[0] == 1:
            intr = self.intrinsics.expand(inds.shape[0], -1)
        else:
            intr = self.intrinsics[inds]
        return self.camera_type.build_camera_model(intr).scaled(1.0 / self.intrinsics_factor).intrinsics

    def _gather(self, maps: torch.Tensor, frame_inds: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        height, width = self.image_size
        x = x.clamp(0, width - 1)
        y = y.clamp(0, height - 1)
        flat_inds = y * width + x
        return maps[frame_inds].reshape(frame_inds.shape[0], -1).gather(1, flat_inds)

    def _iproj(self, disps: torch.Tensor, x: torch.Tensor, y: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        disps = torch.where(disps > 0.0, disps, torch.ones_like(disps))
        uv = torch.stack([x.float(), y.float()], dim=-1)
        pts, _, _ = geom.iproj_disp(disps, uv, intrinsics, self.camera_type)
        return pts

    def _target_points_normals(
        self,
        frame_inds: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = self.image_size
        interior = (x > 0) & (x < width - 1) & (y > 0) & (y < height - 1)

        disp_c = self._gather(self.sensor_disps, frame_inds, x, y)
        disp_l = self._gather(self.sensor_disps, frame_inds, x - 1, y)
        disp_r = self._gather(self.sensor_disps, frame_inds, x + 1, y)
        disp_u = self._gather(self.sensor_disps, frame_inds, x, y - 1)
        disp_d = self._gather(self.sensor_disps, frame_inds, x, y + 1)

        weight_c = self._gather(self.sensor_weights, frame_inds, x, y)
        weight_l = self._gather(self.sensor_weights, frame_inds, x - 1, y)
        weight_r = self._gather(self.sensor_weights, frame_inds, x + 1, y)
        weight_u = self._gather(self.sensor_weights, frame_inds, x, y - 1)
        weight_d = self._gather(self.sensor_weights, frame_inds, x, y + 1)

        valid = (
            interior
            & (disp_c > 0.0)
            & (disp_l > 0.0)
            & (disp_r > 0.0)
            & (disp_u > 0.0)
            & (disp_d > 0.0)
            & (weight_c > 0.0)
            & (weight_l > 0.0)
            & (weight_r > 0.0)
            & (weight_u > 0.0)
            & (weight_d > 0.0)
        )

        pts_c = self._iproj(disp_c, x, y, intrinsics)[..., :3]
        pts_l = self._iproj(disp_l, x - 1, y, intrinsics)[..., :3]
        pts_r = self._iproj(disp_r, x + 1, y, intrinsics)[..., :3]
        pts_u = self._iproj(disp_u, x, y - 1, intrinsics)[..., :3]
        pts_d = self._iproj(disp_d, x, y + 1, intrinsics)[..., :3]

        normal = torch.linalg.cross(pts_r - pts_l, pts_d - pts_u, dim=-1)
        normal_norm = torch.linalg.norm(normal, dim=-1, keepdim=True)
        valid &= normal_norm[..., 0] > 1e-4
        normal = normal / normal_norm.clamp_min(1e-6)
        normal = torch.where((normal * pts_c).sum(dim=-1, keepdim=True) > 0.0, -normal, normal)
        return pts_c, normal, valid

    def forward(self, variables: dict[str, Any], jacobian: bool = True) -> TermEvalReturn:
        pose = variables["pose"]
        assert isinstance(pose, SE3)

        m = self.n_terms
        xy = self.sample_xy
        x0 = xy[:, 0].reshape(1, -1).expand(m, -1)
        y0 = xy[:, 1].reshape(1, -1).expand(m, -1)

        source_disp = self._gather(self.sensor_disps, self.pose_i_inds, x0, y0)
        source_weight = self._gather(self.sensor_weights, self.pose_i_inds, x0, y0)
        source_valid = (source_disp > 0.0) & (source_weight > 0.0)

        intr_i = self._intrinsics_for(self.pose_i_inds)
        intr_j = self._intrinsics_for(self.pose_j_inds)
        x_source = self._iproj(source_disp, x0, y0, intr_i)

        rel_pose = pose[self.pose_j_inds] * pose[self.pose_i_inds].inv()
        x_pred, ja = geom.actp(rel_pose, x_source, compute_jp=jacobian)
        coords, _, _ = geom.proj_points(x_pred, intr_j, self.camera_type)
        x1 = coords[..., 0].round().long()
        y1 = coords[..., 1].round().long()

        x_target, normal, target_valid = self._target_points_normals(self.pose_j_inds, x1, y1, intr_j)
        residual = (normal * (x_pred[..., :3] - x_target)).sum(dim=-1)

        valid = source_valid & target_valid & (x_source[..., 2] > BaseCameraModel.MIN_DEPTH) & (
            x_pred[..., 2] > BaseCameraModel.MIN_DEPTH
        )
        if self.max_residual_m > 0.0:
            valid &= residual.abs() <= self.max_residual_m

        weight = self.alpha * valid.float()

        j_dict = {}
        if jacobian:
            assert ja is not None
            j_j = torch.einsum("mpc,mpck->mpk", normal, ja[..., :3, :])
            j_j = j_j.unsqueeze(-2)
            j_i = -rel_pose[:, None, None].adjT(j_j)
            term_inds = torch.arange(m, device=self.pose_i_inds.device)
            j_dict["pose"] = SparseDenseBlockMatrix(
                i_inds=torch.cat([term_inds, term_inds]),
                j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),
                data=torch.cat([j_i[:, :, 0], j_j[:, :, 0]], dim=0),
            )

        return ConcreteTermEvalReturn(J=j_dict, w=weight, r=residual)

