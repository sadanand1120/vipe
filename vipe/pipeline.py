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

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

from omegaconf import DictConfig
from torch.nn import functional as F

from vipe.slam.interface import SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.streams.base import FrameData, FrameStream, SensorCamera
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.depth import scale_depth_to_sensor
from vipe.utils.logging import pbar


logger = logging.getLogger(__name__)
SUPPORTED_INPUT_CAMERA_MODELS = {"pinhole", "radial", "simple_radial", "simple_divisional", "simple_mei"}


def _validate_gt_sens_depth_mode(mode: str | None) -> None:
    if mode not in {None, "scale", "direct"}:
        raise ValueError(f"Invalid pipeline.depth.use_gt_sens_depths: {mode}")


def _validate_input_camera_model(input_camera_model: str) -> None:
    if input_camera_model not in SUPPORTED_INPUT_CAMERA_MODELS:
        supported = ", ".join(sorted(SUPPORTED_INPUT_CAMERA_MODELS))
        raise ValueError(f"Invalid streams.input_camera_model: {input_camera_model}. Supported: {supported}")


def _geocalib_sample_indices(frame_stream: FrameStream, gap_sec: float = 1.0) -> list[int]:
    gap_frame = int(gap_sec * frame_stream.fps())
    gap_frame = min(gap_frame, (len(frame_stream) - 1) // 2)
    return [0, gap_frame, gap_frame * 2]


def _sample_geocalib_frames(frame_stream: FrameStream, sample_frame_inds: list[int]) -> torch.Tensor:
    sample_frame_set = set(sample_frame_inds)

    sample_by_idx = {}
    for frame_idx, frame in enumerate(frame_stream):
        if frame_idx in sample_frame_set:
            sample_by_idx[frame_idx] = frame.rgb.moveaxis(-1, 0)
        if frame_idx >= sample_frame_inds[-1]:
            break

    return torch.stack([sample_by_idx[i] for i in sample_frame_inds])


def estimate_geocalib_intrinsics(frame_stream: FrameStream) -> torch.Tensor:
    from vipe.priors.geocalib.extractor import GeoCalib

    sample_frame_inds = _geocalib_sample_indices(frame_stream)
    sample_frames = _sample_geocalib_frames(frame_stream, sample_frame_inds)

    model = GeoCalib(weights="pinhole").cuda()
    res = model.calibrate(sample_frames, camera_model="pinhole", shared_intrinsics=True)
    fov_y = res["camera"].vfov[0].item()

    frame_height, frame_width = frame_stream.frame_size()
    fx = fy = frame_height / (2 * np.tan(fov_y / 2))
    return torch.as_tensor([fx, fy, frame_width / 2, frame_height / 2]).float().cuda()


def estimate_geocalib_distorted_camera(frame_stream: FrameStream, input_camera_model: str):
    from vipe.priors.geocalib.extractor import GeoCalib

    sample_frame_inds = _geocalib_sample_indices(frame_stream)
    sample_frames = _sample_geocalib_frames(frame_stream, sample_frame_inds)

    model = GeoCalib(weights="distorted").cuda()
    cameras = [
        model.calibrate(frame, camera_model=input_camera_model, shared_intrinsics=False)["camera"]
        for frame in sample_frames
    ]
    camera_data = torch.stack([camera._data[0] for camera in cameras]).mean(dim=0, keepdim=True)
    return cameras[0].__class__(camera_data)


def camera_to_pinhole_intrinsics(camera) -> torch.Tensor:
    pinhole_camera = camera.pinhole()
    return torch.cat([pinhole_camera.f[0], pinhole_camera.c[0]]).float().cuda()


class GridNormalizedFrameStream(FrameStream):
    def __init__(self, frame_stream: FrameStream, grid: torch.Tensor) -> None:
        super().__init__()
        self.frame_stream = frame_stream
        self.grid = grid
        self.grid_valid_mask = self._grid_valid_mask(grid)

    @staticmethod
    def _grid_valid_mask(grid: torch.Tensor) -> torch.Tensor:
        xy = grid[0]
        return (
            torch.isfinite(xy).all(dim=-1)
            & (xy[..., 0] >= -1.0)
            & (xy[..., 0] <= 1.0)
            & (xy[..., 1] >= -1.0)
            & (xy[..., 1] <= 1.0)
        )

    def _remap_image(self, image: torch.Tensor) -> torch.Tensor:
        remapped = F.grid_sample(
            image.permute(2, 0, 1)[None],
            self.grid,
            mode="bilinear",
            align_corners=True,
        )[0]
        return remapped.permute(1, 2, 0).clamp(0.0, 1.0)

    def _remap_map(self, image: torch.Tensor, mode: str) -> torch.Tensor:
        return F.grid_sample(image[None, None], self.grid, mode=mode, align_corners=True)[0, 0]

    def _remap_valid_mask(self, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return self.grid_valid_mask
        return (self._remap_map(mask.float(), mode="nearest") > 0.5) & self.grid_valid_mask

    def frame_size(self) -> tuple[int, int]:
        return self.frame_stream.frame_size()

    def name(self) -> str:
        return self.frame_stream.name()

    def fps(self) -> float:
        return self.frame_stream.fps()

    def __len__(self) -> int:
        return len(self.frame_stream)

    def __getitem__(self, index: int) -> FrameData:
        frame = self.frame_stream[index]
        image_valid_mask = self._remap_valid_mask(frame.image_valid_mask)
        rgb = self._remap_image(frame.rgb)
        rgb = torch.where(image_valid_mask[..., None], rgb, torch.zeros_like(rgb))

        sensor_depth = None
        if frame.sensor_depth is not None:
            sensor_depth = self._remap_map(frame.sensor_depth, mode="nearest")
            valid = torch.isfinite(sensor_depth) & (sensor_depth > 0.0) & image_valid_mask
            sensor_depth = torch.where(valid, sensor_depth, torch.zeros_like(sensor_depth))

        metric_depth = None
        if frame.metric_depth is not None:
            metric_depth = self._remap_map(frame.metric_depth, mode="bilinear")
            valid = torch.isfinite(metric_depth) & (metric_depth > 0.0) & image_valid_mask
            metric_depth = torch.where(valid, metric_depth, torch.zeros_like(metric_depth))

        depth_confidence = None
        if frame.depth_confidence is not None:
            depth_confidence = self._remap_map(frame.depth_confidence, mode="bilinear")
            depth_confidence = torch.where(image_valid_mask, depth_confidence, torch.zeros_like(depth_confidence))

        return FrameData(
            raw_frame_idx=frame.raw_frame_idx,
            rgb=rgb,
            pose=frame.pose,
            camera_type=frame.camera_type,
            intrinsics=frame.intrinsics,
            metric_depth=metric_depth,
            sensor_depth=sensor_depth,
            image_valid_mask=image_valid_mask,
            depth_confidence=depth_confidence,
            information=frame.information,
        )

    def __iter__(self) -> Iterator[FrameData]:
        for frame_idx in range(len(self)):
            yield self[frame_idx]


class PinholeNormalizedFrameStream(GridNormalizedFrameStream):
    def __init__(self, frame_stream: FrameStream, camera) -> None:
        self.camera = camera
        super().__init__(frame_stream, self._build_undistort_grid(camera))

    @staticmethod
    def _build_undistort_grid(camera) -> torch.Tensor:
        width, height = camera.size[0].round().int().tolist()
        device = camera.device
        dtype = camera.dtype
        x, y = torch.meshgrid(
            torch.arange(0, width, device=device, dtype=dtype),
            torch.arange(0, height, device=device, dtype=dtype),
            indexing="xy",
        )
        coords = torch.stack((x, y), dim=-1).reshape(-1, 2)
        p3d, _ = camera.pinhole().image2world(coords)
        p2d, _ = camera.world2image(p3d)
        mapx = p2d[..., 0].reshape(1, height, width)
        mapy = p2d[..., 1].reshape(1, height, width)
        grid = torch.stack((mapx, mapy), dim=-1)
        scale = torch.tensor([width - 1, height - 1], device=device, dtype=dtype)
        return 2.0 * grid / scale - 1.0


class OpenCVPinholeNormalizedFrameStream(GridNormalizedFrameStream):
    def __init__(self, frame_stream: FrameStream, camera: SensorCamera) -> None:
        self.camera = camera
        super().__init__(frame_stream, self._build_undistort_grid(camera))

    @staticmethod
    def _build_undistort_grid(camera: SensorCamera) -> torch.Tensor:
        assert camera.distortion_coefficients is not None
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera.input_k,
            camera.distortion_coefficients,
            np.eye(3, dtype=np.float32),
            camera.output_k,
            (camera.width, camera.height),
            cv2.CV_32FC1,
        )
        grid = torch.as_tensor(np.stack((mapx, mapy), axis=-1), dtype=torch.float32).cuda()[None]
        scale = torch.tensor([camera.width - 1, camera.height - 1], dtype=torch.float32).cuda()
        return 2.0 * grid / scale - 1.0


class DAV3DepthEstimator:
    """
    Use DAV3 to estimate depth for each frame.
    Depth is conditioned on camera poses and intrinsics from SLAM.

    Depth is estimated in a sliding-window manner, and overlapped frames are linearly averaged to sharp transitions.
    """

    def __init__(
        self,
        model_name: str,
        window_size: int = 10,
        overlap_size: int = 3,
        use_gt_sens_depths: str | None = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.use_gt_sens_depths = use_gt_sens_depths

        if self.use_gt_sens_depths == "direct":
            self.dav3_api = None
            return

        try:
            from depth_anything_3.api import DepthAnything3
            from depth_anything_3.api import logger as dav3_logger
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "depth-anything-3 not found. Please reinstall vipe with `pip install --no-build-isolation -e .[dav3]`"
            )

        dav3_logger.level = 0
        self.dav3_api = DepthAnything3.from_pretrained(model_name)
        self.dav3_api = self.dav3_api.cuda().eval()

    @staticmethod
    def _probe_keyframe_indices(keyframe_indices: list[int], frame_idx: int) -> list[int]:
        assert keyframe_indices
        left = max(0, np.searchsorted(keyframe_indices, frame_idx, side="right").item() - 1)
        indices = [keyframe_indices[left]]
        if frame_idx < keyframe_indices[-1] and left + 1 < len(keyframe_indices):
            indices.append(keyframe_indices[left + 1])
        return indices

    def _attach_slam_output(self, frame_stream: FrameStream, frame_idx: int, slam_output: SLAMOutput) -> FrameData:
        frame = frame_stream[frame_idx]
        frame.pose = slam_output.trajectory[frame_idx]
        frame.intrinsics = slam_output.intrinsics
        frame.camera_type = CameraType.PINHOLE
        return frame

    @staticmethod
    def _image_valid_stack(frames: list[FrameData], depth: torch.Tensor) -> torch.Tensor:
        masks = []
        for frame in frames:
            if frame.image_valid_mask is None:
                masks.append(torch.ones(frame.size(), device=depth.device, dtype=torch.bool))
            else:
                masks.append(frame.image_valid_mask.to(device=depth.device, dtype=torch.bool))
        return torch.stack(masks)

    def _estimate_direct(self, frame_stream: FrameStream, slam_output: SLAMOutput) -> Iterator[FrameData]:
        for frame_idx in pbar(range(len(frame_stream)), desc="Loading sensor depth"):
            frame = self._attach_slam_output(frame_stream, frame_idx, slam_output)
            if frame.sensor_depth is None:
                raise ValueError("Sensor depth is required when pipeline.depth.use_gt_sens_depths=direct")
            sensor_depth = frame.sensor_depth.float()
            valid = torch.isfinite(sensor_depth) & (sensor_depth > 0.0)
            if frame.image_valid_mask is not None:
                valid &= frame.image_valid_mask.to(valid.device)
            frame.metric_depth = torch.where(valid, sensor_depth, torch.zeros_like(sensor_depth))
            frame.depth_confidence = valid.float()
            yield frame

    def estimate(self, frame_stream: FrameStream, slam_output: SLAMOutput) -> Iterator[FrameData]:
        if self.use_gt_sens_depths == "direct":
            yield from self._estimate_direct(frame_stream, slam_output)
            return

        n_frames = len(frame_stream)

        current_sliding_window: list[FrameData] = []
        current_sliding_window_idx: list[int] = []
        trailing_depth: torch.Tensor | None = None
        trailing_confidence: torch.Tensor | None = None
        for frame_idx in pbar(range(n_frames), desc="Estimating DAV3 depth"):
            frame = self._attach_slam_output(frame_stream, frame_idx, slam_output)
            current_sliding_window.append(frame)
            current_sliding_window_idx.append(frame_idx)
            is_last_frame = frame_idx == n_frames - 1

            if len(current_sliding_window) == self.window_size or is_last_frame:
                context_indices = sorted(
                    {
                        keyframe_idx
                        for i in current_sliding_window_idx
                        for keyframe_idx in self._probe_keyframe_indices(slam_output.keyframe_indices, i)
                    }
                )
                context_indices = [i for i in context_indices if i not in current_sliding_window_idx]

                sw_images, sw_exts, sw_ints = zip(*[frame.dav3_conditions() for frame in current_sliding_window])
                if context_indices:
                    ctx_images, ctx_exts, ctx_ints = zip(
                        *[
                            self._attach_slam_output(frame_stream, frame_idx, slam_output).dav3_conditions()
                            for frame_idx in context_indices
                        ]
                    )
                else:
                    ctx_images, ctx_exts, ctx_ints = tuple(), tuple(), tuple()

                dav3_inference_result = self.dav3_api.inference(
                    list(sw_images + ctx_images),
                    extrinsics=np.stack(sw_exts + ctx_exts, axis=0),
                    intrinsics=np.stack(sw_ints + ctx_ints, axis=0),
                    process_res_method="lower_bound_resize",
                )
                sw_depth = torch.from_numpy(dav3_inference_result.depth[: len(sw_images)]).float().cuda()
                sw_depth = torch.nn.functional.interpolate(sw_depth[:, None], frame.size(), mode="bilinear")[:, 0]
                sw_valid = torch.isfinite(sw_depth) & (sw_depth > 0.0)
                sw_valid &= self._image_valid_stack(current_sliding_window, sw_depth)
                sw_confidence = None
                if dav3_inference_result.conf is not None:
                    sw_confidence = torch.from_numpy(dav3_inference_result.conf[: len(sw_images)]).float().cuda()
                    sw_confidence = torch.nn.functional.interpolate(
                        sw_confidence[:, None], frame.size(), mode="bilinear"
                    )[:, 0]

                if self.use_gt_sens_depths == "scale":
                    if any(frame.sensor_depth is None for frame in current_sliding_window):
                        raise ValueError("Sensor depth is required when pipeline.depth.use_gt_sens_depths=scale")
                    sw_sensor = torch.stack([frame.sensor_depth for frame in current_sliding_window]).to(sw_depth)
                    sw_depth, _ = scale_depth_to_sensor(sw_depth, sw_sensor, sw_valid)

                sw_depth = torch.where(sw_valid, sw_depth, torch.zeros_like(sw_depth))
                if sw_confidence is not None:
                    sw_confidence = torch.where(sw_valid, sw_confidence, torch.zeros_like(sw_confidence))

                n_frames_to_yield = (
                    self.window_size - self.overlap_size if not is_last_frame else len(current_sliding_window)
                )

                if trailing_depth is not None:
                    n_interp_frames = len(trailing_depth)
                    alpha = torch.linspace(0, 1, n_interp_frames + 2)[1:-1].float().cuda()[:, None, None]
                    sw_depth[:n_interp_frames] = trailing_depth * (1 - alpha) + sw_depth[:n_interp_frames] * alpha
                    if trailing_confidence is not None and sw_confidence is not None:
                        sw_confidence[:n_interp_frames] = (
                            trailing_confidence * (1 - alpha) + sw_confidence[:n_interp_frames] * alpha
                        )

                for sw_idx, frame in enumerate(current_sliding_window[:n_frames_to_yield]):
                    frame.metric_depth = sw_depth[sw_idx]
                    frame.depth_confidence = None if sw_confidence is None else sw_confidence[sw_idx]
                    yield frame

                trailing_depth = sw_depth[n_frames_to_yield:]
                trailing_confidence = None if sw_confidence is None else sw_confidence[n_frames_to_yield:]
                current_sliding_window = current_sliding_window[n_frames_to_yield:]
                current_sliding_window_idx = current_sliding_window_idx[n_frames_to_yield:]

        assert len(current_sliding_window) == 0, "Current sliding window should be empty"


class VipePipeline:
    def __init__(self, slam: DictConfig, depth: DictConfig, output: DictConfig) -> None:
        self.slam_cfg = slam
        self.depth_cfg = depth
        self.out_cfg = output
        _validate_gt_sens_depth_mode(self.depth_cfg.use_gt_sens_depths)
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)

    def _initialize_sensor_intrinsics(self, frame_stream: FrameStream) -> tuple[FrameStream, torch.Tensor]:
        camera = frame_stream.sensor_camera()
        if camera is None:
            raise ValueError("GT sensor intrinsics requested but stream did not provide sensor camera metadata")

        intrinsics = camera.pinhole_intrinsics()
        if camera.has_distortion:
            logger.info(
                "Normalizing loaded %s camera to pinhole from %s: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
                camera.distortion_model,
                camera.source_path,
                intrinsics[0].item(),
                intrinsics[1].item(),
                intrinsics[2].item(),
                intrinsics[3].item(),
            )
            return OpenCVPinholeNormalizedFrameStream(frame_stream, camera), intrinsics

        logger.info(
            "Using loaded pinhole intrinsics from %s: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            camera.source_path,
            intrinsics[0].item(),
            intrinsics[1].item(),
            intrinsics[2].item(),
            intrinsics[3].item(),
        )
        return frame_stream, intrinsics

    def _initialize(
        self,
        frame_stream: FrameStream,
        input_camera_model: str,
        use_gt_intrinsics: bool,
    ) -> tuple[FrameStream, torch.Tensor]:
        if use_gt_intrinsics:
            return self._initialize_sensor_intrinsics(frame_stream)

        _validate_input_camera_model(input_camera_model)
        if input_camera_model == "pinhole":
            return frame_stream, estimate_geocalib_intrinsics(frame_stream)

        camera = estimate_geocalib_distorted_camera(frame_stream, input_camera_model)
        intrinsics = camera_to_pinhole_intrinsics(camera)
        logger.info(
            "Normalizing %s input camera to pinhole: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            input_camera_model,
            intrinsics[0].item(),
            intrinsics[1].item(),
            intrinsics[2].item(),
            intrinsics[3].item(),
        )
        return PinholeNormalizedFrameStream(frame_stream, camera), intrinsics

    def _run_slam(self, frame_stream: FrameStream, intrinsics: torch.Tensor) -> SLAMOutput:
        slam_pipeline = SLAMSystem(
            device=torch.device("cuda"),
            config=self.slam_cfg,
            keyframe_depth_model=self.depth_cfg.keyframe_model,
            use_gt_sens_depths=self.depth_cfg.use_gt_sens_depths,
        )
        return slam_pipeline.run(frame_stream, intrinsics, camera_type=CameraType.PINHOLE)

    def _run_final_depth(self, frame_stream: FrameStream, slam_output: SLAMOutput) -> Iterator[FrameData]:
        depth_estimator = DAV3DepthEstimator(
            model_name=self.depth_cfg.final_model,
            window_size=self.depth_cfg.window_size,
            overlap_size=self.depth_cfg.overlap_size,
            use_gt_sens_depths=self.depth_cfg.use_gt_sens_depths,
        )
        return depth_estimator.estimate(frame_stream, slam_output)

    def _save_outputs(
        self,
        artifact_path: io.ArtifactPath,
        frame_stream: FrameStream,
        slam_output: SLAMOutput,
    ) -> None:
        if not self.out_cfg.save_artifacts:
            return

        logger.info(f"Saving artifacts to {artifact_path}")
        io.save_artifacts(
            artifact_path,
            self._run_final_depth(frame_stream, slam_output),
            n_frames=len(frame_stream),
            pcd_fusion_mode=self.out_cfg.pcd_fusion_mode,
            max_pcd_points=self.out_cfg.pcd_max_points,
            pcd_conf_threshold_coef=self.out_cfg.pcd_conf_threshold_coef,
            pcd_sample_ratio=self.out_cfg.pcd_sample_ratio,
            pcd_tsdf_voxel_length=self.out_cfg.pcd_tsdf_voxel_length,
            pcd_tsdf_sdf_trunc=self.out_cfg.pcd_tsdf_sdf_trunc,
            pcd_tsdf_depth_trunc=self.out_cfg.pcd_tsdf_depth_trunc,
        )

    def run(
        self,
        frame_stream: FrameStream,
        input_camera_model: str = "pinhole",
        use_gt_intrinsics: bool = False,
    ) -> SLAMOutput:
        frame_stream, intrinsics = self._initialize(frame_stream, input_camera_model, use_gt_intrinsics)
        artifact_path = io.ArtifactPath(self.out_path, frame_stream.name())
        slam_output = self._run_slam(frame_stream, intrinsics)
        self._save_outputs(artifact_path, frame_stream, slam_output)

        return slam_output
