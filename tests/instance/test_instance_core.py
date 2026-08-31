from pathlib import Path

import numpy as np
import torch

from vipe.instance.association import frame_coreset_poses
from vipe.instance.atoms import atom_adjacency, atom_voxels
from vipe.instance.lift import visible_points
from vipe.instance.masks import StreamingMasks, _UnionFind
from vipe.instance.pipeline import (
    _contract_geometry,
    _pack_hypotheses,
    _spatial_instance_colors,
    write_instance_ply,
    reduce_tsdf_surface,
)


def test_tsdf_surface_reduction_keeps_nearest_sample_and_native_normal() -> None:
    points = torch.tensor(
        [
            [0.10, 0.10, 0.10],
            [0.49, 0.49, 0.49],
            [-0.10, 0.10, 0.10],
            [-0.49, 0.49, 0.49],
            [1.25, 0.50, 0.50],
            [1.75, 0.50, 0.50],
        ],
        dtype=torch.float32,
    )
    normals = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    reduced_points, reduced_normals = reduce_tsdf_surface(points, normals, 1.0)
    np.testing.assert_array_equal(
        reduced_points,
        np.array([[-0.49, 0.49, 0.49], [0.49, 0.49, 0.49], [1.25, 0.50, 0.50]], np.float32),
    )
    np.testing.assert_array_equal(reduced_normals, normals[[3, 1, 4]].numpy())


def test_touching_instances_receive_different_colors() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
    owner = np.array([0, 1, 2], np.int32)
    colors = _spatial_instance_colors(points, owner, 3)
    assert not np.array_equal(colors[0], colors[1])


def test_resolution_contract_uses_frontier_nearest_index_map() -> None:
    width, height, x, y = _contract_geometry(6, 4, 3)
    assert (width, height) == (3, 2)
    np.testing.assert_array_equal(x, [0, 2, 4])
    np.testing.assert_array_equal(y, [0, 2])


def test_pose_coreset_references_last_kept_frame() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0)
    poses[:, 0, 3] = [0.0, 0.04, 0.09, 0.13, 0.18]
    kept = frame_coreset_poses(range(5), poses.__getitem__, 8.0, 8.0, log=lambda _: None)
    assert kept == [0, 2, 4]


def test_projection_is_occlusion_correct_and_ascending() -> None:
    points = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    indices, u, v = visible_points(
        points,
        np.eye(4, dtype=np.float32),
        np.array([1.0, 1.0, 0.0, 0.0]),
        width=2,
        height=1,
        depth=np.array([[1.0, 1.0]], np.float32),
        tolerance=0.05,
    )
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_array_equal(u, [0, 1])
    np.testing.assert_array_equal(v, [0, 0])


def test_global_track_linking_uses_minimum_id() -> None:
    masks = StreamingMasks.__new__(StreamingMasks)
    masks.chunk_starts = [0, 1]
    masks.stitch_config = {"min_iou": 0.8, "margin": 1.2}
    masks.union_find = _UnionFind()
    masks.stats = {"linked_tracks": 0, "low_iou": 0, "ambiguous": 0}
    framed = [
        {"masks": [np.array([1, 2, 3], np.int32)], "mtid": np.array([0]), "mgid": np.array([0])},
        {"masks": [np.array([1, 2, 3], np.int32)], "mtid": np.array([1]), "mgid": np.array([1])},
    ]
    masks.finalize(framed)
    assert framed[0]["mgid"].tolist() == [0]
    assert framed[1]["mgid"].tolist() == [0]
    assert masks.stats["linked_tracks"] == 1


def test_sam1_and_sam2_consume_the_same_chunk_jpeg() -> None:
    from PIL import Image

    seen = {}

    class SAM1:
        def generate(self, rgb):
            seen["sam1"] = rgb.copy()
            return [(np.ones(rgb.shape[:2], bool), 0.9)]

    class SAM2:
        def track(self, directory, frame_count, seeds):
            seen["directory"] = directory
            seen["sam2"] = np.asarray(Image.open(directory / "0.jpg").convert("RGB"))
            return [{seeds[0][0]: seeds[0][1]}] + [{} for _ in range(frame_count - 1)]

    config = {
        "chunk_keyframes": 1,
        "seed_topk": 100,
        "sam1": {},
        "sam2": {},
        "stitch": {"min_iou": 0.8, "margin": 1.2},
    }
    rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    masks = StreamingMasks([0], lambda _: rgb, 8, 6, config, 83, torch.device("cpu"), lambda _: None)
    masks._models = lambda: (SAM1(), SAM2())
    masks._build_chunk(0)

    np.testing.assert_array_equal(seen["sam1"], seen["sam2"])
    assert not seen["directory"].exists()


def test_atom_grouping_and_adjacency_are_stably_ordered() -> None:
    atom_of = np.array([1, 0, 1, 2, 0], np.int64)
    grouped = atom_voxels(atom_of)
    assert [group.tolist() for group in grouped] == [[1, 4], [0, 2], [3]]
    edges = (
        np.array([0, 1, 2], np.int64),
        np.array([1, 2, 3], np.int64),
        np.ones(3),
    )
    left, right, counts = atom_adjacency(atom_of, edges)
    np.testing.assert_array_equal(left, [0, 1])
    np.testing.assert_array_equal(right, [1, 2])
    np.testing.assert_array_equal(counts, [2, 1])


def test_hypothesis_packing_and_ply_instance_field(tmp_path: Path) -> None:
    hypotheses = [np.array([0, 2], np.int32), np.array([1, 2, 3], np.int32)]
    indices, offsets = _pack_hypotheses(hypotheses)
    np.testing.assert_array_equal(indices, [0, 2, 1, 2, 3])
    np.testing.assert_array_equal(offsets, [0, 2, 5])

    path = tmp_path / "instances.ply"
    write_instance_ply(path, np.zeros((4, 3), np.float32), hypotheses)
    header = path.read_bytes().split(b"end_header\n", 1)[0]
    assert b"property int instance" in header
