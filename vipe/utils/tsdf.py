from pathlib import Path

import numpy as np
import torch

from vipe.ext import tsdf_ext


class TSDFVolume:
    def __init__(
        self,
        voxel_length: float,
        sdf_trunc: float,
        volume_unit_resolution: int = 16,
        depth_sampling_stride: int = 4,
    ) -> None:
        self.volume = tsdf_ext.TSDFVolume(
            float(voxel_length),
            float(sdf_trunc),
            int(volume_unit_resolution),
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
        self.volume.integrate(depth_t, color_t, intrinsics_t, extrinsic_t, float(depth_trunc))

    def extract_point_cloud(self, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        points, colors = self.volume.extract_point_cloud(int(max_points))
        return points.numpy(), colors.numpy()


def _cpu_tensor(value: np.ndarray | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=dtype)
    return torch.as_tensor(np.ascontiguousarray(value), dtype=dtype, device="cpu")


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if len(points) == 0:
        return

    path.parent.mkdir(exist_ok=True, parents=True)
    colors = np.clip(colors, 0, 255).astype(np.uint8, copy=False)
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(len(points), dtype=vertex_dtype)
    vertices["x"] = points[:, 0].astype(np.float32, copy=False)
    vertices["y"] = points[:, 1].astype(np.float32, copy=False)
    vertices["z"] = points[:, 2].astype(np.float32, copy=False)
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    with path.open("wb") as ply_file:
        ply_file.write(
            (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {len(vertices)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            ).encode("ascii")
        )
        vertices.tofile(ply_file)
