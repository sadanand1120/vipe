from pathlib import Path

import numpy as np
import torch

from vipe.ext import tsdf_ext


class TSDFVolume:
    def __init__(
        self,
        voxel_edge_m: float,
        sdf_trunc_m: float,
        num_voxels_per_block_edge: int = 16,
        depth_sampling_stride: int = 4,
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
        self.volume.integrate(depth_t, color_t, intrinsics_t, extrinsic_t, float(depth_trunc))

    def extract_point_cloud(self, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points, colors, normals = self.volume.extract_point_cloud(int(max_points))
        return points.numpy(), colors.numpy(), normals.numpy()


def _cpu_tensor(value: np.ndarray | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=dtype)
    return torch.as_tensor(np.ascontiguousarray(value), dtype=dtype, device="cpu")


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray, normals: np.ndarray) -> None:
    if len(points) == 0:
        return

    path.parent.mkdir(exist_ok=True, parents=True)
    colors = np.clip(colors, 0, 255).astype(np.uint8, copy=False)
    normals = normals.astype(np.float32, copy=False)
    normal_colors = np.clip(np.rint((normals * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("normals_red", "u1"),
            ("normals_green", "u1"),
            ("normals_blue", "u1"),
        ]
    )
    vertices = np.empty(len(points), dtype=vertex_dtype)
    vertices["x"] = points[:, 0].astype(np.float32, copy=False)
    vertices["y"] = points[:, 1].astype(np.float32, copy=False)
    vertices["z"] = points[:, 2].astype(np.float32, copy=False)
    vertices["nx"] = normals[:, 0]
    vertices["ny"] = normals[:, 1]
    vertices["nz"] = normals[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    vertices["normals_red"] = normal_colors[:, 0]
    vertices["normals_green"] = normal_colors[:, 1]
    vertices["normals_blue"] = normal_colors[:, 2]

    with path.open("wb") as ply_file:
        ply_file.write(
            (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {len(vertices)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property float nx\n"
                "property float ny\n"
                "property float nz\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "property uchar normals_red\n"
                "property uchar normals_green\n"
                "property uchar normals_blue\n"
                "end_header\n"
            ).encode("ascii")
        )
        vertices.tofile(ply_file)
