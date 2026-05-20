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

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

from vipe.ext.lietorch.groups import SE3
from vipe.utils.cameras import CameraType
from vipe.utils.misc import sort_image_sequence


@dataclass(frozen=True, slots=True)
class SensorCamera:
    """
    Loaded RGB/color camera calibration for the input frame directory.
    """

    source_path: Path
    input_k: np.ndarray
    output_k: np.ndarray
    width: int
    height: int
    distortion_model: str | None = None
    distortion_coefficients: np.ndarray | None = None

    @property
    def has_distortion(self) -> bool:
        return self.distortion_coefficients is not None and not np.allclose(self.distortion_coefficients, 0.0)

    def pinhole_intrinsics(self) -> torch.Tensor:
        k = self.output_k
        return torch.as_tensor([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype=torch.float32).cuda()


@dataclass(kw_only=True, slots=True)
class FrameData:
    """
    Frame data from one RGB image.
    - raw_frame_idx: The index of the frame in the sorted frame directory.
    - rgb: The RGB image of the frame. The shape is (H, W, 3), RGB, with range 0-1.
    - sensor_depth: Loaded external sensor/GT depth. This is input data, not the pipeline output depth.
    - pose: The pose of the camera at the time the frame was captured (c2w aka. Twc, opencv convention).
    - camera_type: The downstream camera model. The current pipeline always uses pinhole.
    - intrinsics: Pinhole intrinsics torch Tensor of shape (4,), [fx, fy, cx, cy].
    - metric_depth: The final depth map of the frame. The shape is (H, W). Value is in metric scale.
    - image_valid_mask: Optional input-image validity mask. False means this output pixel has no valid source RGB.
    - information: Additional information about the frame
    """

    raw_frame_idx: int
    rgb: torch.Tensor
    sensor_depth: torch.Tensor
    pose: SE3 | None = None
    camera_type: CameraType | None = None
    intrinsics: torch.Tensor | None = None
    metric_depth: torch.Tensor | None = None
    image_valid_mask: torch.Tensor | None = None
    information: str = ""

    def size(self) -> tuple[int, int]:
        return (self.rgb.shape[0], self.rgb.shape[1])

    def resize(self, size: tuple[int, int]) -> "FrameData":
        """
        Resize the frame to a given size.
        """
        h0, w0 = self.size()
        h1, w1 = size

        new_rgb = (
            torch.nn.functional.interpolate(self.rgb.permute(2, 0, 1)[None], size, mode="bilinear")
            .squeeze(0)
            .permute(1, 2, 0)
        )

        new_sensor_depth = torch.nn.functional.interpolate(self.sensor_depth[None, None], size, mode="nearest")[0, 0]

        new_image_valid_mask = None
        if self.image_valid_mask is not None:
            new_image_valid_mask = torch.nn.functional.interpolate(
                self.image_valid_mask.float()[None, None], size, mode="nearest"
            )[0, 0].bool()

        new_intrinsics = None
        if self.intrinsics is not None:
            new_intrinsics = self.intrinsics.clone()
            new_intrinsics[0:4:2] *= w1 / w0
            new_intrinsics[1:4:2] *= h1 / h0
        new_camera_type = self.camera_type

        return FrameData(
            raw_frame_idx=self.raw_frame_idx,
            rgb=new_rgb,
            sensor_depth=new_sensor_depth,
            image_valid_mask=new_image_valid_mask,
            pose=self.pose,
            intrinsics=new_intrinsics,
            camera_type=new_camera_type,
            information=self.information,
        )

    def crop(self, top: int, bottom: int, left: int, right: int) -> "FrameData":
        """
        Crop the frame with given top, bottom, left, right.
        """
        bottom = self.size()[0] - bottom
        right = self.size()[1] - right

        new_rgb = self.rgb[top:bottom, left:right]
        new_sensor_depth = self.sensor_depth[top:bottom, left:right]

        new_image_valid_mask = None
        if self.image_valid_mask is not None:
            new_image_valid_mask = self.image_valid_mask[top:bottom, left:right]

        new_intrinsics = None
        if self.intrinsics is not None:
            new_intrinsics = self.intrinsics.clone()
            new_intrinsics[2] -= left
            new_intrinsics[3] -= top

        new_camera_type = self.camera_type

        return FrameData(
            raw_frame_idx=self.raw_frame_idx,
            rgb=new_rgb,
            sensor_depth=new_sensor_depth,
            image_valid_mask=new_image_valid_mask,
            pose=self.pose,
            intrinsics=new_intrinsics,
            camera_type=new_camera_type,
            information=self.information,
        )


class FrameStream:
    """
    Minimal frame sequence interface used by the pipeline.
    """

    def frame_size(self) -> tuple[int, int]:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError

    def fps(self) -> float:
        raise NotImplementedError

    def sensor_camera(self) -> SensorCamera | None:
        return None

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> FrameData:
        raise NotImplementedError


class FrameDir(FrameStream):
    """
    Re-iterable RGB frame directory reader.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        name: str | None = None,
    ) -> None:
        path = Path(path)
        self.path = path
        self._name = name if name is not None else path.name

        if not path.is_dir():
            raise ValueError(f"Frame directory not found: {path}")

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        frame_files = []
        for ext in image_extensions:
            frame_files.extend(path.glob(f"*{ext}"))
            frame_files.extend(path.glob(f"*{ext.upper()}"))

        self.frame_files = sort_image_sequence(set(frame_files))
        if not self.frame_files:
            raise ValueError(f"No image files found in directory: {path}")

        first_frame = cv2.imread(str(self.frame_files[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {self.frame_files[0]}")

        self._height, self._width = first_frame.shape[:2]
        self._fps = fps
        self._sensor_camera = self._load_sensor_camera()

        depth_dir = path.parent / "depth"
        missing = [
            depth_dir / f"{frame_path.stem}.png"
            for frame_path in self.frame_files
            if not (depth_dir / f"{frame_path.stem}.png").exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing sensor depth file: {missing[0]}")

    @staticmethod
    def _json_matrix(meta: dict, key: str, shape: tuple[int, int]) -> np.ndarray | None:
        value = meta.get(key)
        if value is None:
            return None
        if isinstance(value, dict):
            value = value["data"]
        matrix = np.asarray(value, dtype=np.float32)
        return matrix.reshape(shape)

    def _load_sensor_camera_json(self, path: Path) -> SensorCamera:
        meta = json.loads(path.read_text(encoding="utf-8"))
        camera_matrix = self._json_matrix(meta, "camera_matrix", (3, 3))
        scannet_matrix = self._json_matrix(meta, "scannet_intrinsic_matrix", (4, 4))
        projection_matrix = self._json_matrix(meta, "projection_matrix", (3, 4))

        if camera_matrix is not None:
            input_k = camera_matrix
        elif scannet_matrix is not None:
            input_k = scannet_matrix[:3, :3]
        else:
            intr = meta["intrinsics"]
            input_k = np.array(
                [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

        distortion_coefficients = meta.get("distortion_coefficients")
        if distortion_coefficients is not None:
            distortion_coefficients = np.asarray(distortion_coefficients, dtype=np.float32).reshape(-1)

        output_k = input_k
        if distortion_coefficients is not None and not np.allclose(distortion_coefficients, 0.0):
            if projection_matrix is not None:
                output_k = projection_matrix[:3, :3]
            distortion_model = meta.get("distortion_model")
            if distortion_model not in {"plumb_bob", "rational_polynomial"}:
                raise ValueError(f"Unsupported distortion model in {path}: {distortion_model}")
        else:
            distortion_model = meta.get("distortion_model")

        width = int(meta.get("width", self._width))
        height = int(meta.get("height", self._height))
        if (height, width) != (self._height, self._width):
            raise ValueError(
                f"Sensor intrinsics size {width}x{height} does not match RGB frames {self._width}x{self._height}"
            )

        return SensorCamera(
            source_path=path,
            input_k=input_k.astype(np.float32),
            output_k=output_k.astype(np.float32),
            width=width,
            height=height,
            distortion_model=distortion_model,
            distortion_coefficients=distortion_coefficients,
        )

    def _load_sensor_camera_txt(self, path: Path) -> SensorCamera:
        matrix = np.loadtxt(path, dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 ScanNet intrinsic matrix in {path}, got {matrix.shape}")
        k = matrix[:3, :3].astype(np.float32)
        return SensorCamera(
            source_path=path,
            input_k=k,
            output_k=k,
            width=self._width,
            height=self._height,
        )

    def _load_sensor_camera(self) -> SensorCamera:
        intrinsic_dir = self.path.parent / "intrinsic"
        json_path = intrinsic_dir / "intrinsic_color.json"
        txt_path = intrinsic_dir / "intrinsic_color.txt"

        if json_path.exists():
            return self._load_sensor_camera_json(json_path)
        if txt_path.exists():
            return self._load_sensor_camera_txt(txt_path)
        raise FileNotFoundError(f"Missing sensor intrinsics file: {json_path} or {txt_path}")

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def sensor_camera(self) -> SensorCamera:
        return self._sensor_camera

    def __len__(self) -> int:
        return len(self.frame_files)

    def __getitem__(self, index: int) -> FrameData:
        n_frames = len(self)
        if index < 0:
            index += n_frames
        if index < 0 or index >= n_frames:
            raise IndexError(index)

        frame_idx = index
        frame_path = self.frame_files[index]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = torch.as_tensor(frame).float().cuda() / 255.0

        depth_path = self.path.parent / "depth" / f"{frame_path.stem}.png"
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw_depth is None:
            raise ValueError(f"Could not read sensor depth: {depth_path}")
        if raw_depth.shape[:2] != frame.shape[:2]:
            raw_depth = cv2.resize(raw_depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        sensor = raw_depth.astype(np.float32) / 1000.0
        sensor[~np.isfinite(sensor)] = 0.0
        sensor[sensor <= 0.0] = 0.0
        sensor_depth = torch.as_tensor(sensor).float().cuda()

        return FrameData(raw_frame_idx=frame_idx, rgb=frame_rgb, sensor_depth=sensor_depth)

    def __iter__(self) -> Iterator[FrameData]:
        for frame_idx in range(len(self)):
            yield self[frame_idx]
