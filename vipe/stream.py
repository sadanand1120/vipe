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

from pathlib import Path

import cv2
import numpy as np
import torch

from vipe.utils.data_format import frame_stem, read_pinhole_intrinsics, read_scene_metadata


class FrameDir:
    """Reader for one canonical ViPE RGB-D scene."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        self.path = path
        self.name = path.name

        if not path.is_dir():
            raise ValueError(f"Canonical ViPE scene directory not found: {path}")

        metadata = read_scene_metadata(path)
        frames = metadata.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"{path / 'metadata.json'} has no frame records")
        self._n_frames = len(frames)
        self.frame_size = (int(metadata["height"]), int(metadata["width"]))

        self.intrinsics_path = path / "intrinsic" / "intrinsic_color.json"
        intrinsics = read_pinhole_intrinsics(self.intrinsics_path)
        if (int(intrinsics["height"]), int(intrinsics["width"])) != self.frame_size:
            raise ValueError(
                f"Sensor intrinsics size {intrinsics['width']}x{intrinsics['height']} does not match "
                f"scene size {self.frame_size[1]}x{self.frame_size[0]}"
            )
        self._intrinsics = torch.tensor(
            [intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]], dtype=torch.float32
        )

    def intrinsics(self) -> torch.Tensor:
        return self._intrinsics.cuda()

    def __len__(self) -> int:
        return self._n_frames

    def _frame_paths(self, index: int) -> tuple[Path, Path]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        stem = frame_stem(index)
        return self.path / "color" / f"{stem}.png", self.path / "depth" / f"{stem}.png"

    def rgb(self, index: int) -> torch.Tensor:
        frame_path, _ = self._frame_paths(index)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")
        if frame.shape[:2] != self.frame_size:
            raise ValueError(
                f"RGB size {frame.shape[1]}x{frame.shape[0]} does not match scene size "
                f"{self.frame_size[1]}x{self.frame_size[0]} for {frame_path}"
            )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.as_tensor(frame).float().cuda() / 255.0

    def _read_depth(self, index: int, rgb_shape: tuple[int, int] | None = None) -> np.ndarray:
        _, depth_path = self._frame_paths(index)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not read sensor depth: {depth_path}")
        if rgb_shape is not None and depth.shape[:2] != rgb_shape:
            raise ValueError(
                f"Depth size {depth.shape[1]}x{depth.shape[0]} does not match RGB "
                f"{rgb_shape[1]}x{rgb_shape[0]} for {depth_path}"
            )
        depth = depth.astype(np.float32) / 1000.0
        depth[depth <= 0.0] = 0.0
        return depth

    def sensor_depth(self, index: int) -> torch.Tensor:
        return torch.as_tensor(self._read_depth(index, self.frame_size)).float().cuda()

    def artifact_arrays(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        frame_path, _ = self._frame_paths(index)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")
        if frame.shape[:2] != self.frame_size:
            raise ValueError(
                f"RGB size {frame.shape[1]}x{frame.shape[0]} does not match scene size "
                f"{self.frame_size[1]}x{self.frame_size[0]} for {frame_path}"
            )
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return color, self._read_depth(index, frame.shape[:2])
