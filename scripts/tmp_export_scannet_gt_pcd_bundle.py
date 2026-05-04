from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vipe.utils.io import (
    INSTANCE_VERTEX_DTYPE,
    NORMAL_VERTEX_DTYPE,
    RGB_VERTEX_DTYPE,
    _clip_pca_colors,
    _encode_clip_labels,
    _instance_colors,
    _load_scannet_vertex_annotations,
    _normal_colors,
    _write_ply_header,
)


INPUT_ROOT = Path("/robodata/smodak/repos/ovo/data/input/ScanNet")
RAW_ROOT = Path("/robodata/smodak/datasets/scannet_v2/scans")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a GT ScanNet PCD bundle matching ViPE pcd/ files.")
    parser.add_argument("--scene", default="scene0000_00")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("tmpgt/pcd"))
    parser.add_argument("--points", type=int, default=2_000_000)
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


def write_normals_ply(path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    vertices = np.empty(len(points), dtype=NORMAL_VERTEX_DTYPE)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["nx"] = normals[:, 0]
    vertices["ny"] = normals[:, 1]
    vertices["nz"] = normals[:, 2]
    with path.open("wb") as f:
        _write_ply_header(
            f,
            len(vertices),
            [
                "property float x",
                "property float y",
                "property float z",
                "property float nx",
                "property float ny",
                "property float nz",
            ],
        )
        vertices.tofile(f)


def write_instance_ply(path: Path, points: np.ndarray, instance_ids: np.ndarray) -> None:
    vertices = np.empty(len(points), dtype=INSTANCE_VERTEX_DTYPE)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["instance_id"] = instance_ids
    with path.open("wb") as f:
        _write_ply_header(
            f,
            len(vertices),
            [
                "property float x",
                "property float y",
                "property float z",
                "property int instance_id",
            ],
        )
        vertices.tofile(f)


def main() -> None:
    args = parse_args()

    import open3d as o3d
    from scipy.spatial import cKDTree

    scene_dir = args.input_root / args.scene
    raw_scene_dir = args.raw_root / args.scene
    mesh = o3d.io.read_triangle_mesh(str(raw_scene_dir / f"{args.scene}_vh_clean_2.ply"))
    mesh.compute_vertex_normals()

    mesh_points = np.asarray(mesh.vertices, dtype=np.float32)
    vertex_normals = np.load(scene_dir / f"{args.scene}_vh_clean_2.vertex_normals.npy").astype(np.float32)
    vertex_instances, vertex_semantics, semantic_labels = _load_scannet_vertex_annotations(
        raw_scene_dir,
        args.scene,
        len(mesh_points),
    )

    sampled = mesh.sample_points_uniformly(number_of_points=int(args.points))
    points = np.asarray(sampled.points, dtype=np.float32)
    colors = (np.asarray(sampled.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)

    _, vertex_ids = cKDTree(mesh_points).query(points, k=1, workers=1)
    normals = vertex_normals[vertex_ids].astype(np.float32)
    instance_ids = vertex_instances[vertex_ids]
    semantic_ids = vertex_semantics[vertex_ids]

    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)

    clip_embeddings = _encode_clip_labels(semantic_labels)
    clip_colors = _clip_pca_colors(clip_embeddings)
    semantic_colors = np.full((len(points), 3), 255, dtype=np.uint8)
    valid_semantics = semantic_ids >= 0
    semantic_colors[valid_semantics] = clip_colors[semantic_ids[valid_semantics]]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rgb_ply(out_dir / "rgb.ply", points, colors)
    write_normals_ply(out_dir / "normals.ply", points, normals)
    write_rgb_ply(out_dir / "normals_viz.ply", points, _normal_colors(normals))
    write_instance_ply(out_dir / "instance.ply", points, instance_ids)
    write_rgb_ply(out_dir / "instance_viz.ply", points, _instance_colors(instance_ids))
    write_rgb_ply(out_dir / "clip_viz.ply", points, semantic_colors)
    np.savez_compressed(
        out_dir / "clip.npz",
        point_label_ids=semantic_ids.astype(np.int16, copy=False),
        label_texts=np.asarray(semantic_labels),
        label_embeddings=clip_embeddings,
        embedding_model=np.asarray("ViT-L-14-336-quickgelu"),
        pretrained=np.asarray("openai"),
        normalized=np.asarray(True),
    )

    print(f"source: {raw_scene_dir / f'{args.scene}_vh_clean_2.ply'}")
    print("coords: raw ScanNet world frame")
    print(f"points: {len(points):,}")
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
