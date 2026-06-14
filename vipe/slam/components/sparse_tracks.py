# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import numpy as np
import torch

from vipe.streams.base import FrameData
from vipe.utils.misc import unpack_optional


logger = logging.getLogger(__name__)


class CuvslamSparseTracks:
    """cuVSLAM feature tracks exported as sparse optical-flow targets for BA."""

    def __init__(self, fps: float) -> None:
        if fps <= 0.0:
            raise ValueError(f"sparse_tracks_fps must be positive, got {fps}")
        self.fps = float(fps)
        self.tracker = None
        self.observations: list[dict[int, np.ndarray]] = []

    def _build_tracker(self, frame_data: FrameData):
        import cuvslam

        height, width = frame_data.size()
        fx, fy, cx, cy = unpack_optional(frame_data.intrinsics).detach().cpu().numpy().tolist()

        distortion = cuvslam.Distortion()
        distortion.model = cuvslam.Distortion.Pinhole

        camera = cuvslam.Camera()
        camera.distortion = distortion
        camera.focal = [float(fx), float(fy)]
        camera.principal = [float(cx), float(cy)]
        camera.size = [int(width), int(height)]
        camera.rig_from_camera = cuvslam.Pose()

        rig = cuvslam.Rig()
        rig.cameras = [camera]
        rig.imus = []

        cfg = cuvslam.Tracker.OdometryConfig()
        cfg.odometry_mode = cuvslam.Tracker.OdometryMode.Mono
        cfg.enable_observations_export = True
        cfg.enable_landmarks_export = False
        cfg.enable_final_landmarks_export = False

        self.tracker = cuvslam.Tracker(rig, odom_config=cfg)
        logger.info(
            "Initialized cuVSLAM sparse tracks: %dx%d fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            width,
            height,
            fx,
            fy,
            cx,
            cy,
        )

    @staticmethod
    def _rgb_uint8(frame_data: FrameData) -> np.ndarray:
        rgb = frame_data.rgb.detach().mul(255.0).clamp_(0.0, 255.0).to(torch.uint8).cpu().numpy()
        return np.ascontiguousarray(rgb)

    @staticmethod
    def _invalid_mask_uint8(frame_data: FrameData) -> np.ndarray:
        if frame_data.image_valid_mask is None:
            mask = np.zeros(frame_data.size(), dtype=np.uint8)
        else:
            mask_t = (~frame_data.image_valid_mask).to(torch.uint8)
            mask = mask_t.detach().cpu().numpy()
        return np.ascontiguousarray(mask)

    def track_frame(self, frame_idx: int, frame_data: FrameData) -> None:
        if self.tracker is None:
            self._build_tracker(frame_data)
        assert self.tracker is not None

        if frame_idx != len(self.observations):
            raise ValueError(f"cuVSLAM sparse tracks require sequential frames, got {frame_idx} after {len(self.observations)}")

        timestamp_ns = int(round(frame_idx * 1_000_000_000.0 / self.fps))
        self.tracker.track(
            timestamp_ns,
            [self._rgb_uint8(frame_data)],
            [self._invalid_mask_uint8(frame_data)],
        )

        frame_obs: dict[int, np.ndarray] = {}
        for obs in self.tracker.get_last_observations(0):
            frame_obs[int(obs.id)] = np.asarray([obs.u, obs.v], dtype=np.float32)
        self.observations.append(frame_obs)

    def _correspondences(self, source_frame_idx: int, target_frame_idx: int) -> list[int]:
        if source_frame_idx >= len(self.observations) or target_frame_idx >= len(self.observations):
            return []
        source_kps = self.observations[source_frame_idx]
        target_kps = self.observations[target_frame_idx]
        return list(source_kps.keys() & target_kps.keys())

    @staticmethod
    def _splat_flow(
        source_uv: torch.Tensor,
        flow: torch.Tensor,
        dense_disp_size: tuple[int, int],
        uv_factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        disp_h, disp_w = dense_disp_size
        source_uv = source_uv * uv_factor
        flow = flow * uv_factor

        x, y = source_uv.unbind(-1)
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        fx = x - x0.float()
        fy = y - y0.float()

        flat_flow = torch.zeros(disp_h * disp_w, 2, dtype=torch.float32, device=source_uv.device)
        flat_weight = torch.zeros(disp_h * disp_w, dtype=torch.float32, device=source_uv.device)

        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            for dy, wy in ((0, 1.0 - fy), (1, fy)):
                xi = x0 + dx
                yi = y0 + dy
                valid = (xi >= 0) & (xi < disp_w) & (yi >= 0) & (yi < disp_h)
                if not torch.any(valid):
                    continue

                w = (wx * wy)[valid]
                flat_idx = yi[valid] * disp_w + xi[valid]
                flat_flow.index_add_(0, flat_idx, flow[valid] * w[:, None])
                flat_weight.index_add_(0, flat_idx, w)

        target = torch.zeros(disp_h * disp_w, 2, dtype=torch.float32, device=source_uv.device)
        valid_weight = flat_weight >= 0.1
        target[valid_weight] = flat_flow[valid_weight] / flat_weight[valid_weight, None]

        yy, xx = torch.meshgrid(
            torch.arange(disp_h, dtype=torch.float32, device=source_uv.device),
            torch.arange(disp_w, dtype=torch.float32, device=source_uv.device),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        target += grid

        weight = torch.zeros(disp_h * disp_w, 2, dtype=torch.float32, device=source_uv.device)
        weight[valid_weight] = flat_weight[valid_weight, None]
        return target, weight

    def compute_dense_flow_target_weight(
        self,
        source_frame_inds: torch.Tensor,
        target_frame_inds: torch.Tensor,
        image_size: tuple[int, int],
        dense_disp_size: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_frames = source_frame_inds.detach().cpu().numpy().astype(np.int64).tolist()
        target_frames = target_frame_inds.detach().cpu().numpy().astype(np.int64).tolist()
        n_terms = len(source_frames)
        disp_h, disp_w = dense_disp_size
        target = torch.zeros(n_terms, disp_h * disp_w, 2, dtype=torch.float32, device=device)
        weight = torch.zeros_like(target)
        uv_factor = torch.tensor(
            [disp_w / image_size[1], disp_h / image_size[0]],
            dtype=torch.float32,
            device=device,
        )

        for term_idx, (source_idx, target_idx) in enumerate(zip(source_frames, target_frames)):
            keypoints = self._correspondences(source_idx, target_idx)
            if not keypoints:
                continue

            source_obs = self.observations[source_idx]
            target_obs = self.observations[target_idx]
            source_uv = torch.as_tensor(
                np.stack([source_obs[keypoint] for keypoint in keypoints], axis=0),
                dtype=torch.float32,
                device=device,
            )
            target_uv = torch.as_tensor(
                np.stack([target_obs[keypoint] for keypoint in keypoints], axis=0),
                dtype=torch.float32,
                device=device,
            )
            target[term_idx], weight[term_idx] = self._splat_flow(
                source_uv,
                target_uv - source_uv,
                dense_disp_size,
                uv_factor,
            )

        return target, weight

    def stats(self) -> dict[str, int]:
        return {
            "frames": len(self.observations),
            "observations": sum(len(frame_obs) for frame_obs in self.observations),
        }
