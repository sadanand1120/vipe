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
from typing import Iterator

import cv2
import numpy as np
import torch

from vipe.ext.lietorch.groups import SE3
from vipe.utils.cameras import CameraType
from vipe.utils.data_format import intrinsic_matrix, read_pinhole_intrinsics, read_scene_frames, read_scene_metadata


@dataclass(frozen=True, slots=True)
class SensorCamera:
    """
    Loaded pinhole RGB camera calibration for a canonical ViPE scene.
    """

    source_path: Path
    k: np.ndarray
    width: int
    height: int

    def pinhole_intrinsics(self) -> torch.Tensor:
        k = self.k
        return torch.as_tensor([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype=torch.float32).cuda()


@dataclass(kw_only=True, slots=True)
class FrameData:
    """
    Frame data from one RGB image.
    - raw_frame_idx: The sequential frame index from metadata.json.
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

    def sensor_camera(self) -> SensorCamera | None:
        return None

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> FrameData:
        raise NotImplementedError


class FrameDir(FrameStream):
    """
    Re-iterable canonical ViPE RGB-D scene reader.
    """

    def __init__(
        self,
        path: str | Path,
        name: str | None = None,
    ) -> None:
        path = Path(path)
        self.path = path
        self._name = name if name is not None else path.name

        if not path.is_dir():
            raise ValueError(f"Canonical ViPE scene directory not found: {path}")

        self.metadata = read_scene_metadata(path)
        self.frames = read_scene_frames(path)
        self.frame_files = [path / frame["color_file"] for frame in self.frames]
        self.depth_files = [path / frame["depth_file"] for frame in self.frames]
        for frame_path, depth_path in zip(self.frame_files, self.depth_files):
            if not frame_path.is_file():
                raise FileNotFoundError(f"Missing canonical RGB frame: {frame_path}")
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing sensor depth file: {depth_path}")

        first_frame = cv2.imread(str(self.frame_files[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {self.frame_files[0]}")

        self._height, self._width = first_frame.shape[:2]
        if (self._width, self._height) != (int(self.metadata["width"]), int(self.metadata["height"])):
            raise ValueError(
                f"metadata.json declares {self.metadata['width']}x{self.metadata['height']}, "
                f"but RGB frames are {self._width}x{self._height}"
            )
        self._sensor_camera = self._load_sensor_camera()

    def _load_sensor_camera(self) -> SensorCamera:
        intrinsic_path = self.path / "intrinsic" / "intrinsic_color.json"
        intrinsics = read_pinhole_intrinsics(intrinsic_path)
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        if (height, width) != (self._height, self._width):
            raise ValueError(
                f"Sensor intrinsics size {width}x{height} does not match RGB frames {self._width}x{self._height}"
            )

        return SensorCamera(
            source_path=intrinsic_path,
            k=intrinsic_matrix(intrinsics),
            width=width,
            height=height,
        )

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

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

        frame_idx = int(self.frames[index].get("seq", index))
        frame_path = self.frame_files[index]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = torch.as_tensor(frame).float().cuda() / 255.0

        depth_path = self.depth_files[index]
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw_depth is None:
            raise ValueError(f"Could not read sensor depth: {depth_path}")
        if raw_depth.shape[:2] != frame.shape[:2]:
            raise ValueError(
                f"Depth size {raw_depth.shape[1]}x{raw_depth.shape[0]} does not match RGB "
                f"{frame.shape[1]}x{frame.shape[0]} for {depth_path}"
            )
        sensor = raw_depth.astype(np.float32) / 1000.0
        sensor[~np.isfinite(sensor)] = 0.0
        sensor[sensor <= 0.0] = 0.0
        sensor_depth = torch.as_tensor(sensor).float().cuda()

        return FrameData(raw_frame_idx=frame_idx, rgb=frame_rgb, sensor_depth=sensor_depth)

    def __iter__(self) -> Iterator[FrameData]:
        for frame_idx in range(len(self)):
            yield self[frame_idx]
