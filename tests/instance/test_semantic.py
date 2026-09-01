from pathlib import Path

import numpy as np
import pytest
import torch

import vipe.instance.semantic as semantic
from vipe.instance.semantic import (
    OverlapField,
    ProjectiveFeatureAccumulator,
    pool_instance_descriptors,
    write_semantic_pca_ply,
)


def test_distillation_interface_uses_selected_config_and_ascending_frames(monkeypatch) -> None:
    seen = []

    class Backbone:
        dimension = 2

        @staticmethod
        def dense_features(_):
            return torch.tensor([[[1.0, 0.0]]])

    features = {
        "backbone": "fgclip",
        "grid": 1,
        "weight_a": 1.0,
        "weight_b": 1.0,
        "occlusion_tolerance_m": 0.05,
        "fgclip": {"model_path": "model", "revision": "revision"},
    }
    monkeypatch.setattr(
        semantic,
        "load_backbone",
        lambda name, grid, config, device: Backbone(),
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
        "backbone": "fgclip",
        "grid": 1,
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


def test_dinotxt_checkpoint_paths_resolve_from_vipe_root(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "dinov3"
    repo.mkdir()
    monkeypatch.setattr(semantic, "_verify_local_revision", lambda _: None)

    with pytest.raises(FileNotFoundError, match=str(semantic._REPO_ROOT / "models" / "backbone.pth")):
        semantic.DINOTextBackbone(
            grid=64,
            repo_path=repo,
            model_name="dinov3_vitl16_dinotxt_tet1280d20h24l",
            backbone_model_path="models/backbone.pth",
            text_model_path="models/text.pth",
            device="cpu",
        )


def test_dinotxt_loads_checkpoints_directly(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "dinov3"
    repo.mkdir()
    backbone_path = tmp_path / "backbone.pth"
    text_path = tmp_path / "text.pth"
    backbone_path.touch()
    text_path.touch()
    monkeypatch.setattr(semantic, "_verify_local_revision", lambda _: None)

    calls = []

    class StateTarget:
        def load_state_dict(self, state, *, strict):
            calls.append((state, strict))

    class Model(StateTarget):
        visual_model = type("Visual", (), {"backbone": StateTarget()})()

        def to(self, _):
            return self

        def eval(self):
            return self

    hub_kwargs = {}
    monkeypatch.setattr(
        torch.hub,
        "load",
        lambda *args, **kwargs: (hub_kwargs.update(kwargs) or Model(), object()),
    )
    monkeypatch.setattr(torch, "load", lambda path, **_: {"path": Path(path).name})

    semantic.DINOTextBackbone(
        grid=64,
        repo_path=repo,
        model_name="dinov3_vitl16_dinotxt_tet1280d20h24l",
        backbone_model_path=backbone_path,
        text_model_path=text_path,
        device="cpu",
    )

    assert hub_kwargs["pretrained"] is False
    assert "weights" not in hub_kwargs and "backbone_weights" not in hub_kwargs
    assert calls == [({"path": "backbone.pth"}, True), ({"path": "text.pth"}, False)]


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
