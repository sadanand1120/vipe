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
from enum import Enum

import torch

from vipe.utils.cameras import CameraType


class DepthType(Enum):
    """
    Type of depth estimated.
    """

    # DAV3 metric depth is proportional to focal length.
    METRIC_DEPTH = "metric_depth"


@dataclass(slots=True, kw_only=True)
class DepthEstimationResult:
    """
    Dataclass for depth estimation results.

    - metric_depth: The estimated depth map ([B,], H, W) in metric scale.
    - confidence: The confidence map ([B,], H, W).
    """

    metric_depth: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None


@dataclass(slots=True, kw_only=True)
class DepthEstimationInput:
    """
    Dataclass for depth estimation inputs.

    - rgb: The source image ([B,], H, W, 3), should be within 0-1 float.
    - intrinsics: The intrinsics of the camera.
    - camera_type: The type of camera.
    """

    rgb: torch.Tensor | None = None
    intrinsics: torch.Tensor | None = None
    sensor_depth: torch.Tensor | None = None
    camera_type: CameraType = CameraType.PINHOLE


class DepthEstimationModel(ABC):
    """
    Unified interface for depth prediction models.
    """

    @property
    def depth_type(self) -> DepthType:
        """
        Type of depth estimated.
        """
        raise NotImplementedError

    @abstractmethod
    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        """
        Estimate a single optical flow result from two images.
        """
