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

from vipe.ext.lietorch import SE3
from vipe.utils.cameras import CameraType
from vipe.utils.misc import sort_image_sequence


@dataclass(kw_only=True, slots=True)
class FrameData:
    """
    Frame data from one RGB image.
    - raw_frame_idx: The index of the frame in the sorted frame directory.
    - rgb: The RGB image of the frame. The shape is (H, W, 3), RGB, with range 0-1.
    - pose: The pose of the camera at the time the frame was captured (c2w aka. Twc, opencv convention).
    - camera_type: The type of camera used to capture the raw frame.
    - intrinsics: Pinhole intrinsics torch Tensor of shape (4+D,), [fx, fy, cx, cy, ...].
      - For the D part, this will be the distortion coefficients of the camera.
      - For panorama images, this will all be zeros.
    - metric_depth: The depth map of the frame. The shape is (H, W). Value is in metric scale.
    - depth_confidence: The depth confidence map of the frame. The shape is (H, W).
    - information: Additional information about the frame
    """

    raw_frame_idx: int
    rgb: torch.Tensor
    pose: SE3 | None = None
    camera_type: CameraType | None = None
    intrinsics: torch.Tensor | None = None
    metric_depth: torch.Tensor | None = None
    depth_confidence: torch.Tensor | None = None
    information: str = ""

    def size(self) -> tuple[int, int]:
        return (self.rgb.shape[0], self.rgb.shape[1])

    @property
    def device(self) -> torch.device:
        return self.rgb.device

    def cpu(self) -> "FrameData":
        map_cpu = lambda x: x.cpu() if x is not None else None

        return FrameData(
            raw_frame_idx=self.raw_frame_idx,
            rgb=self.rgb.cpu(),
            metric_depth=map_cpu(self.metric_depth),
            depth_confidence=map_cpu(self.depth_confidence),
            pose=map_cpu(self.pose),
            intrinsics=map_cpu(self.intrinsics),
            camera_type=self.camera_type,
            information=self.information,
        )

    def cuda(self) -> "FrameData":
        map_cuda = lambda x: x.cuda() if x is not None else None

        return FrameData(
            raw_frame_idx=self.raw_frame_idx,
            rgb=self.rgb.cuda(),
            metric_depth=map_cuda(self.metric_depth),
            depth_confidence=map_cuda(self.depth_confidence),
            pose=map_cuda(self.pose),
            intrinsics=map_cuda(self.intrinsics),
            camera_type=self.camera_type,
            information=self.information,
        )

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

        new_metric_depth = None
        if self.metric_depth is not None:
            new_metric_depth = torch.nn.functional.interpolate(self.metric_depth[None, None], size, mode="bilinear")[
                0, 0
            ]

        new_depth_confidence = None
        if self.depth_confidence is not None:
            new_depth_confidence = torch.nn.functional.interpolate(
                self.depth_confidence[None, None], size, mode="bilinear"
            )[0, 0]

        new_intrinsics = None
        if self.intrinsics is not None:
            new_intrinsics = self.intrinsics.clone()
            new_intrinsics[0:4:2] *= w1 / w0
            new_intrinsics[1:4:2] *= h1 / h0
        # Distortion coefficients are usually w.r.t normalized coordinates so no need to change here.
        new_camera_type = self.camera_type

        return FrameData(
            raw_frame_idx=self.raw_frame_idx,
            rgb=new_rgb,
            metric_depth=new_metric_depth,
            depth_confidence=new_depth_confidence,
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

        new_metric_depth = None
        if self.metric_depth is not None:
            new_metric_depth = self.metric_depth[top:bottom, left:right]

        new_depth_confidence = None
        if self.depth_confidence is not None:
            new_depth_confidence = self.depth_confidence[top:bottom, left:right]

        new_intrinsics = None
        if self.intrinsics is not None:
            new_intrinsics = self.intrinsics.clone()
            new_intrinsics[2] -= left
            new_intrinsics[3] -= top

        new_camera_type = self.camera_type

        return FrameData(
            raw_frame_idx=self.raw_frame_idx,
            rgb=new_rgb,
            metric_depth=new_metric_depth,
            depth_confidence=new_depth_confidence,
            pose=self.pose,
            intrinsics=new_intrinsics,
            camera_type=new_camera_type,
            information=self.information,
        )

    def dav3_conditions(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        dav3_rgb = (self.rgb.cpu().numpy() * 255).astype(np.uint8)
        dav3_ext = None
        if self.pose is not None:
            dav3_ext = self.pose.inv().matrix().cpu().numpy()
        dav3_int = None
        if self.intrinsics is not None:
            assert self.camera_type == CameraType.PINHOLE
            fx, fy, cx, cy = self.intrinsics.cpu().numpy()
            dav3_int = np.array(
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1],
                ]
            )
        return dav3_rgb, dav3_ext, dav3_int


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

    def __len__(self) -> int:
        raise NotImplementedError


class FrameDir(FrameStream):
    """
    Re-iterable RGB frame directory reader.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_start: int = 0,
        frame_end: int = -1,
        frame_skip: int = 1,
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
        self.start = frame_start
        self.end = len(self.frame_files) if frame_end == -1 else min(frame_end, len(self.frame_files))
        self.step = frame_skip
        self._fps = fps / self.step

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(range(self.start, self.end, self.step))

    def __iter__(self) -> Iterator[FrameData]:
        for frame_idx in range(self.start, self.end, self.step):
            frame_path = self.frame_files[frame_idx]
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Could not read frame: {frame_path}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = torch.as_tensor(frame).float().cuda() / 255.0
            yield FrameData(raw_frame_idx=frame_idx, rgb=frame_rgb)
