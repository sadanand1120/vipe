# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def scale_depth_to_sensor(pred_depth: torch.Tensor, sensor_depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (
        torch.isfinite(pred_depth)
        & torch.isfinite(sensor_depth)
        & (pred_depth > 0.0)
        & (sensor_depth > 0.0)
    )
    if valid.sum().item() == 0:
        raise ValueError("Cannot scale depth: no valid overlapping predicted/sensor depth pixels")

    pred = pred_depth[valid].float()
    sensor = sensor_depth[valid].float()
    scale = torch.dot(pred, sensor) / torch.dot(pred, pred).clamp_min(1e-12)
    return pred_depth * scale, scale
