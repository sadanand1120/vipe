import json

from pathlib import Path

import numpy as np

from vipe.bench.scannet_instance import load_annotated_mesh, load_semantic_classes


def test_load_annotated_scannet_mesh_maps_segments_to_instances(tmp_path: Path) -> None:
    scene = "scene0000_00"
    scene_dir = tmp_path / scene
    scene_dir.mkdir()
    (scene_dir / f"{scene}_vh_clean_2.ply").write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 0\nproperty list uchar int vertex_indices\n"
        "end_header\n"
        "0 0 0\n1 0 0\n0 1 0\n1 1 0\n",
        encoding="ascii",
    )
    (scene_dir / f"{scene}_vh_clean_2.0.010000.segs.json").write_text(
        json.dumps({"segIndices": [10, 10, 20, 30]}), encoding="utf-8"
    )
    (scene_dir / f"{scene}.aggregation.json").write_text(
        json.dumps(
            {
                "segGroups": [
                    {"objectId": 0, "segments": [10], "label": "floor"},
                    {"objectId": 7, "segments": [20], "label": "chair"},
                ]
            }
        ),
        encoding="utf-8",
    )

    points, labels = load_annotated_mesh(tmp_path, scene)
    object_to_class, class_names = load_semantic_classes(tmp_path, scene)

    assert points.shape == (4, 3)
    np.testing.assert_array_equal(labels, np.array([0, 0, 7, -1], dtype=np.int32))
    assert {class_names[class_id] for class_id in object_to_class.values()} == {"floor", "chair"}
    assert class_names[object_to_class[0]] == "floor"
    assert class_names[object_to_class[7]] == "chair"
