from pathlib import Path

import numpy as np
import torch

from vipe.ext import tsdf_ext


class TSDFVolume:
    def __init__(
        self,
        voxel_edge_m: float,
        sdf_trunc_m: float,
        num_voxels_per_block_edge: int,
        depth_sampling_stride: int,
    ) -> None:
        self.volume = tsdf_ext.TSDFVolume(
            float(voxel_edge_m),
            float(sdf_trunc_m),
            int(num_voxels_per_block_edge),
            int(depth_sampling_stride),
        )

    def integrate(
        self,
        depth: np.ndarray | torch.Tensor,
        color: np.ndarray | torch.Tensor,
        intrinsics: np.ndarray | torch.Tensor,
        extrinsic_w2c: np.ndarray | torch.Tensor,
        depth_trunc: float,
    ) -> None:
        depth_t = _cpu_tensor(depth, torch.float32).contiguous()
        color_t = _cpu_tensor(color, torch.uint8).contiguous()
        intrinsics_t = _cpu_tensor(intrinsics, torch.float32).contiguous()
        extrinsic_t = _cpu_tensor(extrinsic_w2c, torch.float32).contiguous()
        self.volume.integrate(
            depth_t,
            color_t,
            intrinsics_t,
            extrinsic_t,
            float(depth_trunc),
        )

    def write_point_cloud(
        self,
        path: Path,
        max_points: int,
        select_representatives: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        path.parent.mkdir(exist_ok=True, parents=True)
        points, normals, source_count = self.volume.write_point_cloud(
            str(path),
            int(max_points),
            bool(select_representatives),
        )
        if source_count == 0 or not select_representatives:
            return None
        return points, normals, int(source_count)


def _cpu_tensor(value: np.ndarray | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.device.type != "cpu" or value.dtype != dtype:
            value = value.to(device="cpu", dtype=dtype)
        return value

    value = np.asarray(value)
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    return torch.as_tensor(value, dtype=dtype, device="cpu")
