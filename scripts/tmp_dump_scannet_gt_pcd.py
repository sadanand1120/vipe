from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vipe.utils.io import RGB_VERTEX_DTYPE, _instance_colors, _load_aligned_scannet_gt, _write_ply_header


INPUT_ROOT = Path("/robodata/smodak/repos/ovo/data/input/ScanNet")
RAW_ROOT = Path("/robodata/smodak/datasets/scannet_v2/scans")
REFERENCE_OUTPUT = Path("/robodata/smodak/repos/vipe/outputs_gtequiv/scene00_dav3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump the aligned ScanNet GT cloud used for ViPE property transfer.")
    parser.add_argument("--scene", default="scene0000_00")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--reference-output", type=Path, default=REFERENCE_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("tmpgt"))
    parser.add_argument("--overlay-gt-points", type=int, default=8_000_000)
    return parser.parse_args()


def write_rgb_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(len(points), dtype=RGB_VERTEX_DTYPE)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    with path.open("wb") as f:
        _write_ply_header(
            f,
            len(vertices),
            [
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ],
        )
        vertices.tofile(f)


def read_rgb_ply_points(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        vertex_count = None
        for raw in f:
            line = raw.decode("ascii").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
        assert vertex_count is not None
        vertices = np.frombuffer(f.read(vertex_count * RGB_VERTEX_DTYPE.itemsize), dtype=RGB_VERTEX_DTYPE)
    return np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()

    import open3d as o3d

    pose_data = np.load(args.reference_output / "pose" / "color.npz")
    first_raw_frame_idx = int(pose_data["inds"][0])
    first_pose_c2w = pose_data["data"][0].astype(np.float32)

    scene_dir = args.input_root / args.scene
    raw_scene_dir = args.raw_root / args.scene
    source_frame_dir = scene_dir / "color"

    gt_points, _, gt_instances, _, _ = _load_aligned_scannet_gt(
        source_frame_dir,
        first_raw_frame_idx,
        first_pose_c2w,
    )

    mesh = o3d.io.read_triangle_mesh(str(raw_scene_dir / f"{args.scene}_vh_clean_2.ply"))
    vertex_colors = (np.asarray(mesh.vertex_colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    assert len(vertex_colors) == len(gt_points)

    sampled_gt = mesh.sample_points_uniformly(number_of_points=int(args.overlay_gt_points))
    sampled_gt_points = np.asarray(sampled_gt.points, dtype=np.float32)
    sampled_gt_colors = (np.asarray(sampled_gt.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)

    from scipy.spatial import cKDTree

    vertex_tree = cKDTree(np.asarray(mesh.vertices, dtype=np.float32))
    _, sample_vertex_ids = vertex_tree.query(sampled_gt_points, k=1, workers=1)
    sampled_gt_instances = gt_instances[sample_vertex_ids]

    gt_points_h = np.concatenate([sampled_gt_points, np.ones((len(sampled_gt_points), 1), dtype=np.float32)], axis=1)
    frame_files = sorted(source_frame_dir.iterdir(), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
    frame_id = frame_files[first_raw_frame_idx].stem
    scannet_first_pose = np.loadtxt(scene_dir / "pose" / f"{frame_id}.txt", dtype=np.float32)
    vipe_from_scannet = first_pose_c2w @ np.linalg.inv(scannet_first_pose)
    sampled_gt_points = (vipe_from_scannet @ gt_points_h.T).T[:, :3].astype(np.float32)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rgb_ply(out_dir / "rgb.ply", sampled_gt_points, sampled_gt_colors)
    write_rgb_ply(out_dir / "instance_viz.ply", sampled_gt_points, _instance_colors(sampled_gt_instances))

    pred_points = read_rgb_ply_points(args.reference_output / "pcd" / "rgb.ply")
    overlay_dir = out_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_points = np.concatenate([sampled_gt_points, pred_points], axis=0)
    overlay_colors = np.empty((len(overlay_points), 3), dtype=np.uint8)
    overlay_colors[: len(sampled_gt_points)] = np.array([0, 255, 0], dtype=np.uint8)
    overlay_colors[len(sampled_gt_points) :] = np.array([255, 0, 0], dtype=np.uint8)
    write_rgb_ply(overlay_dir / "overlay.ply", overlay_points, overlay_colors)

    print(f"GT source mesh: {raw_scene_dir / f'{args.scene}_vh_clean_2.ply'}")
    print(f"GT segmentation: {raw_scene_dir / f'{args.scene}_vh_clean_2.0.010000.segs.json'}")
    print(f"GT aggregation: {raw_scene_dir / f'{args.scene}.aggregation.json'}")
    print(f"Aligned to: {args.reference_output}")
    print(f"GT mesh vertices: {len(gt_points):,}")
    print(f"sampled GT points: {len(sampled_gt_points):,}")
    print(f"wrote: {out_dir / 'rgb.ply'}")
    print(f"wrote: {out_dir / 'instance_viz.ply'}")
    print(f"wrote: {overlay_dir / 'overlay.ply'}")


if __name__ == "__main__":
    main()
