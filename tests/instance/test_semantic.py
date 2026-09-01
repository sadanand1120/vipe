from pathlib import Path

import numpy as np
import torch

import vipe.instance.semantic as semantic
from vipe.instance.semantic import (
    CANONICAL_NEGATIVES,
    OverlapField,
    ProjectiveFeatureAccumulator,
    pool_instance_descriptors,
    text_scores,
    write_semantic_pca_ply,
)


def test_distillation_interface_uses_selected_config_and_ascending_frames(monkeypatch) -> None:
    seen = []

    class Backbone:
        grid = 24
        dimension = 2

        @staticmethod
        def dense_features(_):
            return torch.tensor([[[1.0, 0.0]]])

    features = {
        "weight_a": 1.0,
        "weight_b": 1.0,
        "occlusion_tolerance_m": 0.05,
        "model_path": "model",
        "revision": "revision",
    }
    monkeypatch.setattr(
        semantic,
        "FGCLIPBackbone",
        lambda **_: Backbone(),
    )
    monkeypatch.setattr(semantic, "pbar", lambda frames, **_: frames)

    descriptors, metrics = semantic.distill_semantic_features(
        features=features,
        points=np.array([[0.0, 0.0, 1.0]], np.float32),
        normals=np.array([[0.0, 0.0, 1.0]], np.float32),
        hypotheses=[np.array([0])],
        frame_indices=[1, 0],
        rgb_of=lambda index: seen.append(index) or np.zeros((1, 1, 3), np.uint8),
        depth_of=lambda _: np.ones((1, 1), np.float32),
        poses=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        intrinsics=np.array([1.0, 1.0, 0.0, 0.0], np.float32),
        width=1,
        height=1,
        device="cpu",
    )

    assert seen == [0, 1]
    np.testing.assert_array_equal(descriptors, [[1.0, 0.0]])
    assert metrics == {
        "grid": 24,
        "descriptor_dimension": 2,
        "selected_frames": 2,
        "valid_descriptor_count": 1,
        "direct_point_hit_fraction": 1.0,
        "instance_field_coverage": 1.0,
    }


def test_projective_fusion_uses_dense_depth_and_frontier_weight() -> None:
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 0.0, 2.0]], np.float32)
    normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
    feature_map = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [-1.0, 0.0]]]
    )
    pool = ProjectiveFeatureAccumulator(points, normals, 2, device="cpu")

    hits = pool.integrate(
        feature_map,
        np.array([2.0, 1.0, 0.0, 0.0], np.float32),
        np.eye(4, dtype=np.float32),
        width=2,
        height=1,
        depth=np.array([[1.0, 2.0]], np.float32),
    )

    assert hits == 2
    torch.testing.assert_close(pool.sum_w, torch.tensor([1.0, 0.25, 0.0]))
    torch.testing.assert_close(
        pool.sum_wf,
        torch.tensor([[1.0, 0.0], [0.0, 0.25], [0.0, 0.0]]),
    )


def test_frontier_text_scores_use_learned_temperature_and_canonical_negatives() -> None:
    class Backbone:
        temperature = 0.5

        @staticmethod
        def encode_text(names, template):
            assert names == ["chair", "table", *CANONICAL_NEGATIVES]
            assert template == "a photo of a {}"
            return torch.tensor(
                [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.6, 0.8]]
            )

    features = np.array([[1.0, 0.0]], np.float32)
    scores = text_scores(Backbone(), features, ["chair", "table"])
    similarities = np.array([[1.0, 0.8, 0.0, -1.0, 0.0, 0.6]], np.float32)
    exponentials = np.exp((similarities - similarities.max()) / 0.5)
    negative_sum = exponentials[:, 2:].sum(1, keepdims=True)
    expected = exponentials[:, :2] / (exponentials[:, :2] + negative_sum)

    np.testing.assert_allclose(scores, expected, rtol=1e-6)


def test_fgclip_temperature_comes_from_learned_logit_scale() -> None:
    model = torch.nn.Module()
    model.logit_scale = torch.nn.Parameter(torch.tensor(np.log(4.0), dtype=torch.float32))

    assert semantic._learned_temperature(model) == 0.25


def test_descriptor_pooling_and_overlap_field_match_frontier() -> None:
    sum_wf = torch.tensor([[2.0, 0.0], [0.0, 3.0], [4.0, 4.0], [0.0, 0.0]])
    sum_w = torch.tensor([2.0, 1.0, 0.0, 0.0])
    hypotheses = [np.array([0, 1, 2]), np.array([2, 3]), np.array([1])]

    descriptors = pool_instance_descriptors(sum_wf, sum_w, hypotheses, chunk_size=1)

    assert descriptors.dtype == np.float16
    np.testing.assert_allclose(descriptors[0], np.sqrt(0.5), atol=5e-4)
    np.testing.assert_array_equal(descriptors[1], 0.0)
    np.testing.assert_array_equal(descriptors[2], [0.0, 1.0])
    field = OverlapField(4, hypotheses, descriptors)
    np.testing.assert_array_equal(field.covered, [True, True, True, False])
    np.testing.assert_allclose(
        field.rows(np.arange(4)),
        [[np.sqrt(0.5), np.sqrt(0.5)], [0.382683, 0.92388], [np.sqrt(0.5), np.sqrt(0.5)], [0, 0]],
        atol=5e-4,
    )


def test_semantic_pca_ply_is_bounded_and_marks_uncovered_gray(tmp_path: Path) -> None:
    points = np.arange(15, dtype=np.float32).reshape(5, 3)
    normals = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (5, 1))
    hypotheses = [np.array([0, 1, 2]), np.array([1, 2, 3])]
    descriptors = np.array([[1.0, 0.0], [0.0, 1.0]], np.float16)
    path = tmp_path / "semantic.ply"

    coverage = write_semantic_pca_ply(
        path,
        points,
        normals,
        hypotheses,
        descriptors,
        chunk_size=2,
        max_pca_samples=2,
    )

    assert coverage == 0.8
    header, records = path.read_bytes().split(b"end_header\n", 1)
    assert b"element vertex 5" in header
    dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ]
    )
    colors = np.frombuffer(records, dtype=dtype)[["red", "green", "blue"]]
    assert tuple(colors[-1]) == (128, 128, 128)
