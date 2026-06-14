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

import warnings

import numpy as np
import torch

from einops import rearrange

from ..networks.droid_net import AltCorrBlock, CorrBlock, DroidNet
from .buffer import GraphBuffer


warnings.simplefilter(action="ignore", category=FutureWarning)


class FactorGraph:
    @staticmethod
    def coords_grid(ht, wd, **kwargs):
        y, x = torch.meshgrid(
            torch.arange(ht).to(**kwargs).float(),
            torch.arange(wd).to(**kwargs).float(),
            indexing="ij",
        )
        return torch.stack([x, y], dim=-1)

    def __init__(
        self,
        net: DroidNet,
        buffer: GraphBuffer,
        device: torch.device,
        max_factors: int,
        incremental: bool,
        use_sparse_tracks: bool = False,
    ):
        self.net = net
        self.buffer = buffer
        self.device = device
        self.max_factors = max_factors
        self.incremental = incremental
        self.use_sparse_tracks = use_sparse_tracks

        ht = buffer.height // 8
        wd = buffer.width // 8
        self.coords0 = self.coords_grid(ht, wd, device=device)

        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.damping = 1e-6 * torch.ones_like(self.buffer.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.sparse_target = None
        self.sparse_weight = None

        self.corr, self.f_net, self.inp = None, None, None

    @property
    def num_factors(self) -> int:
        return int(self.ii.shape[0])

    def __filter_repeated_edges(self, ii, jj):
        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set((i.item(), j.item()) for i, j in zip(self.ii, self.jj))

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    @torch.amp.autocast("cuda", enabled=True)
    def add_factors(self, ii, jj, remove=False):
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)
        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        ii, jj = self.__filter_repeated_edges(ii, jj)
        if ii.shape[0] == 0:
            return

        if (
            self.max_factors > 0
            and self.ii.shape[0] + ii.shape[0] > self.max_factors
            and self.corr is not None
            and remove
        ):
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0])

        if self.incremental:
            fmap1 = self.buffer.fmaps[ii][None]
            fmap2 = self.buffer.fmaps[jj][None]
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.buffer.inps[ii][None]
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.buffer.reproject_dense_disp(ii, jj)
            target = target[None]
            weight = torch.zeros_like(target)
            sparse_target, sparse_weight = self._compute_sparse_targets(ii, jj)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        net = self.buffer.nets[ii][None]
        self.f_net = net if self.f_net is None else torch.cat([self.f_net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)
        self._append_sparse_targets(sparse_target, sparse_weight)

    def _compute_sparse_targets(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.use_sparse_tracks or self.buffer.sparse_tracks is None:
            return None, None

        return self.buffer.sparse_tracks.compute_dense_flow_target_weight(
            source_frame_inds=self.buffer.tstamp[ii],
            target_frame_inds=self.buffer.tstamp[jj],
            image_size=(self.buffer.height, self.buffer.width),
            dense_disp_size=(self.buffer.height // 8, self.buffer.width // 8),
            device=self.device,
        )

    def _append_sparse_targets(
        self,
        sparse_target: torch.Tensor | None,
        sparse_weight: torch.Tensor | None,
    ) -> None:
        if sparse_target is None or sparse_weight is None:
            return
        if self.sparse_target is None or self.sparse_weight is None:
            self.sparse_target = sparse_target
            self.sparse_weight = sparse_weight
            return
        self.sparse_target = torch.cat([self.sparse_target, sparse_target], 0)
        self.sparse_weight = torch.cat([self.sparse_weight, sparse_weight], 0)

    @torch.amp.autocast("cuda", enabled=True)
    def rm_factors(self, mask: torch.Tensor):
        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]

        if self.corr is not None:
            self.corr = self.corr[~mask]
        if self.f_net is not None:
            self.f_net = self.f_net[:, ~mask]
        if self.inp is not None:
            self.inp = self.inp[:, ~mask]

        self.target = self.target[:, ~mask]
        self.weight = self.weight[:, ~mask]
        if self.sparse_target is not None:
            self.sparse_target = self.sparse_target[~mask]
        if self.sparse_weight is not None:
            self.sparse_weight = self.sparse_weight[~mask]

    @torch.amp.autocast("cuda", enabled=True)
    def rm_second_newest_keyframe(self, ix: int):
        self.buffer.remove_second_newest(ix)

        m = (self.ii == ix) | (self.jj == ix)
        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m)

    @torch.amp.autocast("cuda", enabled=True)
    def update(
        self,
        t0: int | None = None,
        t1: int | None = None,
        itrs: int = 3,
        motion_only: bool = False,
    ):
        assert self.incremental
        assert self.corr is not None and self.inp is not None and self.f_net is not None

        if t0 is None:
            t0 = int(max(1, self.ii.min().item() + 1))
        if t1 is None:
            t1 = int(max(self.ii.max().item(), self.jj.max().item()) + 1)

        with torch.cuda.amp.autocast(enabled=False):
            coords1, _ = self.buffer.reproject_dense_disp(self.ii, self.jj)
            coords1 = coords1[None]
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        corr = self.corr(coords1)

        di, dix = torch.unique(self.ii, return_inverse=True)
        self.f_net, delta, weight, damping, _ = self.net.update.forward(  # type: ignore
            self.f_net, self.inp, corr, motn, ix=dix
        )

        with torch.cuda.amp.autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)
            self.damping[di] = damping

            ht, wd = self.coords0.shape[0:2]
            target = rearrange(self.target, "1 k h w c -> k (h w) c", c=2, h=ht, w=wd)
            weight = rearrange(self.weight, "1 k h w c -> k (h w) c", c=2, h=ht, w=wd)

            self.buffer.bundle_adjustment(
                target=target,
                weight=weight,
                disp_damping=self.damping,
                ii=self.ii,
                jj=self.jj,
                t0=t0,
                t1=t1,
                n_iters=itrs,
                pose_damping=1e-3,
                pose_ep=0.1,
                motion_only=motion_only,
                sparse_target=self.sparse_target,
                sparse_weight=self.sparse_weight,
                verbose=False,
            )

        self.age += 1

    @torch.amp.autocast("cuda", enabled=False)
    def update_batch(
        self,
        itrs: int,
        steps: int,
        batch_size: int,
        solver_verbose: bool = False,
    ):
        if self.incremental:
            warnings.warn("Calling update_batch with incremental=True could be slow.")
        assert self.f_net is not None

        t = self.buffer.n_frames
        corr_op = AltCorrBlock(self.buffer.fmaps[None])

        for _ in range(steps):
            with torch.cuda.amp.autocast(enabled=False):
                coords1, _ = self.buffer.reproject_dense_disp(self.ii, self.jj)
                coords1 = coords1[None]
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

            assert self.jj.max() >= self.ii.max()
            for i in range(0, self.jj.max() + 1, batch_size):
                v = (self.ii >= i) & (self.ii < i + batch_size)
                if not torch.any(v):
                    continue
                iis, jjs = self.ii[v], self.jj[v]
                corr1 = corr_op(coords1[:, v], iis, jjs)
                dis, dixs = torch.unique(iis, return_inverse=True)

                with torch.cuda.amp.autocast(enabled=True):
                    net, delta, weight, damping, _ = self.net.update.forward(  # type: ignore
                        self.f_net[:, v],
                        self.buffer.inps[iis][None],
                        corr1,
                        motn[:, v],
                        ix=dixs,
                    )

                self.f_net[:, v] = net
                self.target[:, v] = coords1[:, v] + delta.float()
                self.weight[:, v] = weight.float()
                self.damping[dis] = damping

            ht, wd = self.coords0.shape[0:2]
            target = rearrange(self.target, "1 k h w c -> k (h w) c", c=2, h=ht, w=wd)
            weight = rearrange(self.weight, "1 k h w c -> k (h w) c", c=2, h=ht, w=wd)

            self.buffer.bundle_adjustment(
                target=target,
                weight=weight,
                disp_damping=self.damping,
                ii=self.ii,
                jj=self.jj,
                t0=1,
                t1=t,
                n_iters=itrs,
                pose_damping=1e-5,
                pose_ep=1e-2,
                motion_only=False,
                sparse_target=self.sparse_target,
                sparse_weight=self.sparse_weight,
                verbose=solver_verbose,
            )

    def add_neighborhood_factors(self, t0, t1, r: int = 3):
        ii, jj = torch.meshgrid(torch.arange(t0, t1), torch.arange(t0, t1), indexing="ij")
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    def add_proximity_factors(
        self,
        t0: int = 0,
        t1: int = 0,
        rad: int = 2,
        nms: int = 2,
        beta: float = 0.25,
        thresh: float = 16.0,
        remove: bool = False,
    ):
        assert t0 >= t1, "t0 should be a subset of t1"

        t = self.buffer.n_frames
        ix = torch.arange(t0, t).to(self.device)
        jx = torch.arange(t1, t).to(self.device)

        ii, jj = torch.meshgrid(ix, jx, indexing="ij")
        ii, jj = ii.reshape(-1), jj.reshape(-1)

        d = self.buffer.frame_distance_dense_disp(ii, jj, beta=beta)

        def _suppress(i: int, j: int):
            if (t0 <= i < t) and (t1 <= j < t):
                d[(i - t0) * (t - t1) + (j - t1)] = np.inf

        def _suppress_nms(i: int, j: int):
            for di in range(-nms, nms + 1):
                for dj in range(-nms, nms + 1):
                    if abs(di) + abs(dj) <= max(min(abs(i - j) - 2, nms), 0):
                        _suppress(i + di, j + dj)

        for i, j in zip(self.ii.cpu().numpy(), self.jj.cpu().numpy()):
            _suppress_nms(i, j)
        d[(ii - rad < jj) | (d > thresh)] = np.inf

        es = []
        for i in range(t0, t):
            for j in range(max(i - rad - 1, 0), i):
                es.append((i, j))
                es.append((j, i))
                _suppress(i, j)

        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                continue
            if len(es) > self.max_factors:
                break

            i, j = int(ii[k].item()), int(jj[k].item())
            es.append((i, j))
            es.append((j, i))
            _suppress_nms(i, j)

        if len(es) == 0:
            return
        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)
