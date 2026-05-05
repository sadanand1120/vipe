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

import torch

from vipe.ext.lietorch.groups import SE3


class BaseRetractor:
    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor):
        x[inds] += dx


class PoseRetractor(BaseRetractor):
    def oplus(self, x: SE3, inds: torch.Tensor, dx: torch.Tensor):
        x.data[inds] = SE3(x.data[inds]).retr(dx).data


class DenseDispRetractor(BaseRetractor):
    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor):
        dx = torch.where(dx > 10, torch.zeros_like(dx), dx)
        return super().oplus(x, inds, dx)
