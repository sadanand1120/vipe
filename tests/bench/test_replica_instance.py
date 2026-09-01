from pathlib import Path

import cv2
import numpy as np
import yaml

import vipe.bench.replica_instance as replica_instance
from vipe.bench.replica_instance import (
    apply_exclusions,
    apply_se3,
    build_gt_occupancy_cloud,
    kabsch_se3,
    load_instance_prediction,
    recall_ar,
)
from vipe.utils.config import AttrDict
from vipe.utils.data_format import write_pinhole_intrinsics, write_scene_metadata


def test_kabsch_se3_recovers_rigid_transform() -> None:
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float64)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    translation = np.array([2, -3, 4], dtype=np.float64)
    target = apply_se3(source, rotation, translation)

    estimated_rotation, estimated_translation = kabsch_se3(source, target)

    np.testing.assert_allclose(estimated_rotation, rotation, atol=1e-12)
    np.testing.assert_allclose(estimated_translation, translation, atol=1e-12)


def test_instance_artifact_loading_and_membership_budget(tmp_path: Path) -> None:
    path = tmp_path / "scene.npz"
    np.savez(
        path,
        points=np.zeros((4, 3), dtype=np.float32),
        hypothesis_indices=np.array([0, 1, 1, 2], dtype=np.int32),
        hypothesis_offsets=np.array([0, 2, 4], dtype=np.int64),
        K=np.array(2, dtype=np.int32),
        instance_features=np.array([[1, 0], [0, 1]], dtype=np.float16),
        feature_grid=np.int32(64),
    )

    prediction = load_instance_prediction(path)

    assert prediction.membership_budget == 2
    assert [hypothesis.tolist() for hypothesis in prediction.hypotheses] == [[0, 1], [1, 2]]


def test_recall_ar_keeps_excluded_voxels_in_prediction_union() -> None:
    labels = apply_exclusions(np.array([1, 1, 2, 2, 9]), [9])
    hypotheses = [np.array([0, 1, 4]), np.array([2, 3])]

    result = recall_ar(labels, hypotheses, np.array([0.50, 0.75]))

    # GT 1 has IoU 2/3 because the excluded voxel remains in its hypothesis union; GT 2 has IoU 1.
    assert result["ar"] == 0.75
    assert result["r50"] == 1.0
    assert result["r75"] == 0.5
    assert result["n_gt"] == 2
    assert result["max_memb"] == 1


def test_gt_cloud_uses_canonical_depth_and_gt_pose(tmp_path: Path) -> None:
    scene_dir = tmp_path / "office0"
    (scene_dir / "depth").mkdir(parents=True)
    write_scene_metadata(
        scene_dir,
        name="office0",
        width=2,
        height=1,
        fps=5.0,
        frames=[{"source_frame_id": 0}],
    )
    write_pinhole_intrinsics(
        scene_dir / "intrinsic" / "intrinsic_color.json",
        width=2,
        height=1,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
    )
    assert cv2.imwrite(str(scene_dir / "depth" / "000000.png"), np.array([[1000, 0]], dtype=np.uint16))
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 1.0

    points = build_gt_occupancy_cloud(
        scene_dir,
        pose[None],
        voxel_m=0.5,
        depth_min_m=0.1,
        depth_max_m=12.0,
    )

    np.testing.assert_array_equal(points, np.array([[1.25, 0.25, 1.25]], dtype=np.float32))


def test_replica_frontier_exclusions_are_exact() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "eval_replica_instance_config.yaml"
    exclusions = yaml.safe_load(config_path.read_text(encoding="utf-8"))["exclusions"]

    assert exclusions == {
        "office0": [1, 13, 18, 26, 56, 65, 67],
        "office2": [5, 6, 7, 78, 83],
        "room0": [17, 26, 28, 29, 37, 38, 42, 52, 53, 62, 66, 82, 91],
    }


def test_evaluate_scene_aligns_cloud_then_transfers_gt_labels(tmp_path: Path, monkeypatch) -> None:
    scene = "office0"
    scene_dir = tmp_path / "input" / scene
    output_dir = tmp_path / "outputs" / scene
    (scene_dir / "pose").mkdir(parents=True)
    (output_dir / "pose").mkdir(parents=True)
    (output_dir / "instances").mkdir(parents=True)
    write_scene_metadata(
        scene_dir,
        name=scene,
        width=1,
        height=1,
        fps=5.0,
        frames=[{"source_frame_id": idx} for idx in range(3)],
    )
    write_pinhole_intrinsics(
        scene_dir / "intrinsic" / "intrinsic_color.json",
        width=1,
        height=1,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
    )
    gt_centers = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    pred_centers = gt_centers + np.array([10, 0, 0])
    gt_poses = np.repeat(np.eye(4)[None], 3, axis=0)
    pred_poses = gt_poses.copy()
    gt_poses[:, :3, 3] = gt_centers
    pred_poses[:, :3, 3] = pred_centers
    for idx, pose in enumerate(gt_poses):
        np.savetxt(scene_dir / "pose" / f"{idx:06d}.txt", pose)
    np.savez(output_dir / "pose" / f"{scene}.npz", data=pred_poses, inds=np.arange(3))

    gt_points = np.array([[0, 0, 1], [0.02, 0, 1], [1, 0, 1], [1.02, 0, 1]], dtype=np.float32)
    np.savez(
        output_dir / "instances" / f"{scene}.npz",
        points=gt_points + np.array([10, 0, 0], dtype=np.float32),
        hypothesis_indices=np.arange(4, dtype=np.int32),
        hypothesis_offsets=np.array([0, 2, 4]),
        K=np.array(1),
        instance_features=np.array([[1, 0], [0, 1]], dtype=np.float16),
        feature_grid=np.int32(64),
    )
    monkeypatch.setattr(
        replica_instance,
        "load_or_build_gt_cache",
        lambda *args, **kwargs: (gt_points, np.array([1, 1, 2, 2], dtype=np.int32)),
    )
    monkeypatch.setattr(
        replica_instance,
        "load_semantic_classes",
        lambda *args, **kwargs: ({1: 10, 2: 20}, {10: "chair", 20: "table"}),
    )

    class TextEncoder:
        @staticmethod
        def encode_text(names, template=None):
            assert names == ["chair", "table"]
            assert template == "a photo of a {}"
            return np.eye(2, dtype=np.float32)

    config = AttrDict(
        outputs=AttrDict(gt_cache_filename="unused.npz"),
        exclusions=AttrDict(),
        metric=AttrDict(iou_min=0.5, iou_max=0.95, iou_step=0.05),
    )

    result = replica_instance.evaluate_scene(
        scene=scene,
        scene_dir=scene_dir,
        raw_root=tmp_path / "raw",
        vipe_output_dir=output_dir,
        cache_dir=tmp_path / "cache",
        feature_config=AttrDict(grid=64),
        text_encoder=TextEncoder(),
        config=config,
    )

    assert result["ar"] == 1.0
    assert result["r90"] == 1.0
    assert result["n_hyps"] == 2
    assert result["ate_se3_m"] < 1e-12
    assert result["semantic_top1"] == 1.0
    assert (output_dir / "pcd" / f"{scene}_instances_gt.ply").is_file()
    assert (output_dir / "pcd" / f"{scene}_instances_gtmatch.ply").is_file()
