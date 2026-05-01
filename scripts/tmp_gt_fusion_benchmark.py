#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


DEFAULT_INPUT_ROOT = Path("/robodata/smodak/repos/ovo/data/input/ScanNet")
DEFAULT_RAW_ROOT = Path("/robodata/smodak/datasets/scannet_v2/scans")
DEFAULT_DA3_ROOT = Path("/robodata/smodak/repos/Depth-Anything-3")
VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)
COUNT_PLACEHOLDER = "0" * 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GT-only ScanNet backprojection-vs-TSDF speed/accuracy benchmark."
    )
    parser.add_argument("--scene", required=True, help="ScanNet scene name, e.g. scene0000_00")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for sampled output PLYs")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, type=Path, help="Processed ScanNet input root")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, type=Path, help="Raw ScanNet scans root")
    parser.add_argument("--da3-root", default=DEFAULT_DA3_ROOT, type=Path, help="Depth-Anything-3 repo root")
    parser.add_argument("--sample-points", default=1_000_000, type=int, help="Points to sample per method")
    parser.add_argument("--max-frames", default=-1, type=int, help="Use first N frames only (-1 for all)")
    parser.add_argument("--seed", default=42, type=int, help="Open3D random seed")
    return parser


def setup_da3(args: argparse.Namespace):
    os.environ["DA3_SCANNET_INPUT_ROOT"] = str(args.input_root)
    os.environ["DA3_SCANNET_RAW_ROOT"] = str(args.raw_root)
    sys.path.insert(0, str(args.da3_root / "src"))

    from depth_anything_3.bench.datasets.scannet import ScanNetDataset
    from depth_anything_3.bench.utils import (
        create_tsdf_volume,
        evaluate_3d_reconstruction,
        sample_points_from_mesh,
    )

    return ScanNetDataset, create_tsdf_volume, evaluate_3d_reconstruction, sample_points_from_mesh


def load_depth_meters(path: str | Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    depth = raw.astype(np.float32) / 1000.0 if np.issubdtype(raw.dtype, np.integer) else raw.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0.0] = 0.0
    return depth


def load_color_for_depth(path: str | Path, depth_hw: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = depth_hw
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb.astype(np.uint8))


def load_depth_intrinsics(input_root: Path, scene: str) -> np.ndarray:
    path = input_root / scene / "intrinsic" / "intrinsic_depth.txt"
    return np.loadtxt(path, dtype=np.float32)[:3, :3]


def ply_header(vertex_count: int | str) -> bytes:
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")


def write_vertices_ply(path: Path, vertices: np.ndarray) -> None:
    with path.open("wb") as f:
        f.write(ply_header(len(vertices)))
        vertices.tofile(f)


def init_streamed_ply(path: Path):
    header = ply_header(COUNT_PLACEHOLDER)
    count_offset = header.index(COUNT_PLACEHOLDER.encode("ascii"))
    f = path.open("w+b")
    f.write(header)
    return f, count_offset, len(header)


def finish_streamed_ply(f, count_offset: int, vertex_count: int) -> None:
    f.seek(count_offset)
    f.write(f"{vertex_count:020d}".encode("ascii"))
    f.close()


def backproject_frame(
    depth: np.ndarray,
    color: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    max_depth: float,
) -> np.ndarray:
    valid = (depth > 0.0) & (depth <= max_depth)
    ys, xs = np.nonzero(valid)
    z = depth[ys, xs]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    points_cam = np.empty((len(z), 3), dtype=np.float32)
    points_cam[:, 0] = (xs.astype(np.float32) - cx) * z / fx
    points_cam[:, 1] = (ys.astype(np.float32) - cy) * z / fy
    points_cam[:, 2] = z

    points_world = points_cam @ c2w[:3, :3].T + c2w[:3, 3]
    colors = color[ys, xs]

    vertices = np.empty(len(z), dtype=VERTEX_DTYPE)
    vertices["x"] = points_world[:, 0]
    vertices["y"] = points_world[:, 1]
    vertices["z"] = points_world[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    return vertices


def build_backproject_ply(
    scene_data,
    intrinsics: np.ndarray,
    frame_count: int,
    max_depth: float,
    tmp_ply: Path,
) -> tuple[int, int]:
    f, count_offset, body_offset = init_streamed_ply(tmp_ply)
    vertex_count = 0
    try:
        for i in tqdm(range(frame_count), desc="Backproject GT frames", unit="frame"):
            depth = load_depth_meters(scene_data.aux.gt_depth_files[i])
            color = load_color_for_depth(scene_data.image_files[i], depth.shape)
            c2w = np.linalg.inv(scene_data.extrinsics[i]).astype(np.float32)
            vertices = backproject_frame(depth, color, intrinsics, c2w, max_depth)
            vertices.tofile(f)
            vertex_count += len(vertices)
    except Exception:
        f.close()
        raise

    finish_streamed_ply(f, count_offset, vertex_count)
    return vertex_count, body_offset


def sample_indices(total: int, count: int) -> np.ndarray:
    return np.floor(np.arange(count, dtype=np.float64) * total / count).astype(np.int64)


def pcd_from_vertices(vertices: np.ndarray) -> o3d.geometry.PointCloud:
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)
    colors = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]).astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def sample_backproject_ply(
    tmp_ply: Path,
    body_offset: int,
    vertex_count: int,
    sample_count: int,
    output_path: Path,
) -> o3d.geometry.PointCloud:
    indices = sample_indices(vertex_count, sample_count)
    sampled = np.empty(sample_count, dtype=VERTEX_DTYPE)
    chunk_vertices = 2_000_000

    with tmp_ply.open("rb") as f, tqdm(total=sample_count, desc="Sample backproject PLY", unit="pts") as pbar:
        f.seek(body_offset)
        read_start = 0
        out_start = 0
        while read_start < vertex_count:
            chunk_count = min(chunk_vertices, vertex_count - read_start)
            chunk = np.fromfile(f, dtype=VERTEX_DTYPE, count=chunk_count)
            read_end = read_start + len(chunk)
            out_end = np.searchsorted(indices, read_end, side="left")
            if out_end > out_start:
                rel = indices[out_start:out_end] - read_start
                sampled[out_start:out_end] = chunk[rel]
                pbar.update(out_end - out_start)
                out_start = out_end
            read_start = read_end

    write_vertices_ply(output_path, sampled)
    return pcd_from_vertices(sampled)


def integrate_tsdf_online(
    scene_data,
    create_tsdf_volume,
    intrinsics: np.ndarray,
    frame_count: int,
    max_depth: float,
    voxel_length: float,
    sdf_trunc: float,
):
    volume = create_tsdf_volume(voxel_length=voxel_length, sdf_trunc=sdf_trunc)
    for i in tqdm(range(frame_count), desc="Fuse GT TSDF", unit="frame"):
        depth = np.ascontiguousarray(load_depth_meters(scene_data.aux.gt_depth_files[i]).astype(np.float32))
        color = load_color_for_depth(scene_data.image_files[i], depth.shape)
        h, w = depth.shape

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_trunc=max_depth,
            depth_scale=1.0,
            convert_rgb_to_intensity=False,
        )
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            w,
            h,
            float(intrinsics[0, 0]),
            float(intrinsics[1, 1]),
            float(intrinsics[0, 2]),
            float(intrinsics[1, 2]),
        )
        volume.integrate(rgbd, intrinsic_o3d, scene_data.extrinsics[i].astype(np.float64))

    return volume


def extract_tsdf_mesh(volume, tmp_mesh: Path) -> None:
    tqdm.write("Extract TSDF mesh")
    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("TSDF fusion produced an empty mesh")
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(tmp_mesh), mesh, write_ascii=False)


def sample_tsdf_mesh(
    tmp_mesh: Path,
    sample_count: int,
    output_path: Path,
    sample_points_from_mesh,
) -> o3d.geometry.PointCloud:
    tqdm.write("Sample TSDF mesh from dumped mesh")
    mesh = o3d.io.read_triangle_mesh(str(tmp_mesh))
    pcd = sample_points_from_mesh(mesh, sample_count)
    o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
    return pcd


def format_table(rows: list[dict[str, float | int | str]]) -> str:
    headers = [
        "method",
        "online_s",
        "extract_dump_s",
        "sample_dump_s",
        "total_s",
        "sampled_pts",
        "acc",
        "comp",
        "overall",
        "precision",
        "recall",
        "fscore",
    ]
    formatted = []
    for row in rows:
        formatted.append(
            {
                "method": str(row["method"]),
                "online_s": f"{row['online_s']:.2f}",
                "extract_dump_s": f"{row['extract_dump_s']:.2f}",
                "sample_dump_s": f"{row['sample_dump_s']:.2f}",
                "total_s": f"{row['total_s']:.2f}",
                "sampled_pts": f"{row['sampled_pts']:,}",
                "acc": f"{row['acc']:.5f}",
                "comp": f"{row['comp']:.5f}",
                "overall": f"{row['overall']:.5f}",
                "precision": f"{row['precision']:.4f}",
                "recall": f"{row['recall']:.4f}",
                "fscore": f"{row['fscore']:.4f}",
            }
        )
    widths = {header: max(len(header), *(len(row[header]) for row in formatted)) for header in headers}
    lines = [
        " | ".join(header.ljust(widths[header]) for header in headers),
        "-+-".join("-" * widths[header] for header in headers),
    ]
    for row in formatted:
        lines.append(" | ".join(row[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    o3d.utility.random.seed(args.seed)

    ScanNetDataset, create_tsdf_volume, evaluate_3d_reconstruction, sample_points_from_mesh = setup_da3(args)
    dataset = ScanNetDataset()
    scene_data = dataset.get_data(args.scene)
    frame_count = len(scene_data.image_files)
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    if frame_count <= 0:
        raise ValueError(f"No frames selected for {args.scene}")

    depth_intrinsics = load_depth_intrinsics(args.input_root, args.scene)
    tmp_dir = args.output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_backproject = tmp_dir / f"{args.scene}_backproject_full.ply"
    tmp_tsdf_mesh = tmp_dir / f"{args.scene}_tsdf_mesh.ply"
    backproject_sampled_path = args.output_dir / f"{args.scene}_backproject_sampled.ply"
    tsdf_sampled_path = args.output_dir / f"{args.scene}_tsdf_sampled.ply"

    try:
        start = time.perf_counter()
        backproject_vertices, body_offset = build_backproject_ply(
            scene_data,
            depth_intrinsics,
            frame_count,
            dataset.max_depth,
            tmp_backproject,
        )
        backproject_time = time.perf_counter() - start

        if backproject_vertices <= 0:
            raise RuntimeError("Backprojection produced no valid points")
        sample_count = min(args.sample_points, backproject_vertices)

        start = time.perf_counter()
        tsdf_volume = integrate_tsdf_online(
            scene_data,
            create_tsdf_volume,
            depth_intrinsics,
            frame_count,
            dataset.max_depth,
            dataset.voxel_length,
            dataset.sdf_trunc,
        )
        tsdf_online_time = time.perf_counter() - start

        start = time.perf_counter()
        extract_tsdf_mesh(tsdf_volume, tmp_tsdf_mesh)
        tsdf_extract_time = time.perf_counter() - start

        start = time.perf_counter()
        backproject_pcd = sample_backproject_ply(
            tmp_backproject,
            body_offset,
            backproject_vertices,
            sample_count,
            backproject_sampled_path,
        )
        backproject_sample_time = time.perf_counter() - start

        start = time.perf_counter()
        tsdf_pcd = sample_tsdf_mesh(
            tmp_tsdf_mesh,
            sample_count,
            tsdf_sampled_path,
            sample_points_from_mesh,
        )
        tsdf_sample_time = time.perf_counter() - start

        gt_mesh = o3d.io.read_triangle_mesh(scene_data.aux.gt_mesh_path)
        gt_pcd = sample_points_from_mesh(gt_mesh, sample_count)

        rows = []
        for method, online_time, extract_time, sample_time, pcd in [
            ("backproject", backproject_time, 0.0, backproject_sample_time, backproject_pcd),
            ("tsdf", tsdf_online_time, tsdf_extract_time, tsdf_sample_time, tsdf_pcd),
        ]:
            metrics = evaluate_3d_reconstruction(
                pcd,
                gt_pcd,
                threshold=dataset.eval_threshold,
                down_sample=dataset.down_sample,
            )
            rows.append(
                {
                    "method": method,
                    "online_s": online_time,
                    "extract_dump_s": extract_time,
                    "sample_dump_s": sample_time,
                    "total_s": online_time + extract_time + sample_time,
                    "sampled_pts": sample_count,
                    **metrics,
                }
            )

        print()
        print("timing: online_s=per-frame update; extract_dump_s=TSDF mesh extraction+dump; sample_dump_s=final sampled PLY from dumped intermediate")
        print(format_table(rows))
        print()
        print(f"saved: {backproject_sampled_path}")
        print(f"saved: {tsdf_sampled_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
