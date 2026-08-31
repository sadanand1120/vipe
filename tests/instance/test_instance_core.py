from pathlib import Path

import numpy as np
import torch

from vipe.instance.association import frame_coreset_poses
from vipe.instance.atoms import atom_adjacency, atom_voxels
from vipe.instance.lift import lift_masks, visible_points
from vipe.instance.masks import StreamingMasks, _UnionFind
from vipe.instance.pipeline import (
    _contract_geometry,
    _pack_hypotheses,
    _spatial_instance_colors,
    write_instance_ply,
)


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


def test_lift_streams_atom_csr_and_exact_track_unions() -> None:
    points = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]])
    masks = [
        (np.array([[1, 0, 1, 0]], bool), 0.9, 4, 4),
        (np.array([[0, 1, 1, 1]], bool), 0.8, 5, 5),
    ]
    evidence, track_unions = lift_masks(
        points,
        np.array([0, 0, 1, 1], np.int32),
        (np.array([0], np.int64), np.array([1], np.int64)),
        [0],
        lambda _: np.eye(4, dtype=np.float32),
        lambda _: masks,
        lambda _: np.ones((1, 4), np.float32),
        np.array([1.0, 1.0, 0.0, 0.0], np.float32),
        4,
        1,
        0.05,
        1,
    )
    np.testing.assert_array_equal(evidence["leafI_ptr"], [0, 2, 4])
    np.testing.assert_array_equal(evidence["leafI_gm"], [0, 1, 0, 1])
    np.testing.assert_array_equal(evidence["leafI_c"], [1, 1, 1, 2])
    np.testing.assert_array_equal(evidence["leafN_ptr"], [0, 1, 2])
    np.testing.assert_array_equal(evidence["leafN_c"], [2, 2])
    np.testing.assert_array_equal(evidence["affinity_edge"], [0])
    np.testing.assert_allclose(evidence["affinity_num"], [3.0 / np.sqrt(20.0)])
    np.testing.assert_array_equal(evidence["affinity_weight"], [1.0])
    np.testing.assert_array_equal(track_unions[4], [0, 2])
    np.testing.assert_array_equal(track_unions[5], [1, 2, 3])


def test_affinity_counts_one_sided_mask_frames_as_disagreement() -> None:
    points = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    masks = {
        0: [(np.array([[1, 1]], bool), 0.9, 0, 0)],
        1: [(np.array([[1, 0]], bool), 0.9, 1, 1)],
    }
    evidence, _ = lift_masks(
        points,
        np.array([0, 1], np.int32),
        (np.array([0], np.int64), np.array([1], np.int64)),
        [0, 1],
        lambda _: np.eye(4, dtype=np.float32),
        masks.__getitem__,
        lambda _: np.ones((1, 2), np.float32),
        np.array([1.0, 1.0, 0.0, 0.0], np.float32),
        2,
        1,
        0.05,
        1,
    )
    np.testing.assert_allclose(evidence["affinity_num"], [1.0])
    np.testing.assert_array_equal(evidence["affinity_weight"], [2.0])


def test_streamed_affinity_matches_sequence_wide_sparse_formula() -> None:
    import scipy.sparse as sp

    rng = np.random.default_rng(4)
    atom_count, frame_count = 6, 4
    points = torch.tensor([[float(index), 0.0, 1.0] for index in range(atom_count)])
    adjacent_a, adjacent_b = np.triu_indices(atom_count, 1)
    masks = {}
    track_id = 0
    for frame in range(frame_count):
        frame_masks = []
        for _ in range(5):
            mask = rng.random(atom_count) > 0.45
            mask[rng.integers(atom_count)] = True
            frame_masks.append((mask[None], 0.9, track_id, track_id))
            track_id += 1
        masks[frame] = frame_masks
    evidence, _ = lift_masks(
        points,
        np.arange(atom_count, dtype=np.int32),
        (adjacent_a, adjacent_b),
        range(frame_count),
        lambda _: np.eye(4, dtype=np.float32),
        masks.__getitem__,
        lambda _: np.ones((1, atom_count), np.float32),
        np.array([1.0, 1.0, 0.0, 0.0], np.float32),
        atom_count,
        1,
        0.05,
        1,
    )

    gm_frame = evidence["gm_frame"]
    row = np.repeat(np.arange(atom_count), np.diff(evidence["leafI_ptr"]))
    frame = gm_frame[evidence["leafI_gm"]]
    nkey = np.repeat(np.arange(atom_count), np.diff(evidence["leafN_ptr"])) * frame_count + evidence["leafN_f"]
    key = row * frame_count + frame
    visible = evidence["leafN_c"][np.searchsorted(nkey, key)]
    values = evidence["leafI_c"] / visible
    starts = np.r_[True, (row[1:] != row[:-1]) | (frame[1:] != frame[:-1])]
    groups = np.cumsum(starts) - 1
    norms = np.sqrt(np.bincount(groups, weights=values * values))
    maximum = np.maximum.reduceat(values, np.flatnonzero(starts))
    values *= np.sqrt(np.minimum(maximum, 1.0))[groups] / norms[groups]
    signature = sp.csr_matrix((values, (row, evidence["leafI_gm"])), shape=(atom_count, len(gm_frame)))
    visible_matrix = sp.csr_matrix(
        (np.ones(len(evidence["leafN_f"])),
         (np.repeat(np.arange(atom_count), np.diff(evidence["leafN_ptr"])), evidence["leafN_f"])),
        shape=(atom_count, frame_count),
    )
    masked_matrix = sp.csr_matrix(
        (np.ones(len(row)), (row, frame)), shape=(atom_count, frame_count)
    )
    masked_matrix.data[:] = 1.0
    numerator = np.asarray(
        signature[adjacent_a].multiply(signature[adjacent_b]).sum(axis=1)
    ).ravel()
    weight = np.asarray(
        (visible_matrix[adjacent_a].multiply(visible_matrix[adjacent_b])
         .multiply(masked_matrix[adjacent_a] + masked_matrix[adjacent_b]) > 0)
        .sum(axis=1)
    ).ravel()
    positive = np.flatnonzero((numerator > 0.0) & (weight > 0.0))
    np.testing.assert_array_equal(evidence["affinity_edge"], positive)
    np.testing.assert_allclose(evidence["affinity_num"], numerator[positive], rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(evidence["affinity_weight"], weight[positive])


def test_global_track_linking_uses_minimum_id() -> None:
    masks = StreamingMasks.__new__(StreamingMasks)
    masks.chunk_starts = [0, 1]
    masks.stitch_config = {"min_iou": 0.8, "margin": 1.2}
    masks.union_find = _UnionFind()
    masks.stats = {"linked_tracks": 0, "low_iou": 0, "ambiguous": 0}
    evidence = {"gm_track_id": np.array([0, 1], np.int64)}
    masks.finalize(
        evidence,
        {0: np.array([1, 2, 3], np.int32), 1: np.array([1, 2, 3], np.int32)},
    )
    assert evidence["global_track_ids"].tolist() == [0, 0]
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
    masks = StreamingMasks([0], lambda _: rgb, 8, 6, config, 83, torch.device("cpu"))
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
