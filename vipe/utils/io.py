# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import shutil
import tempfile
import zipfile

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import imageio
import Imath
import numpy as np
import OpenEXR
import torch

from vipe.ext.lietorch import SE3
from vipe.streams.base import FrameAttribute, FrameData, FrameStream
from vipe.utils.cameras import CameraType
from vipe.utils.geometry import se3_matrix_to_se3
from vipe.utils.logging import pbar
from vipe.utils.misc import sort_image_sequence
from vipe.utils.visualization import VideoWriter


logger = logging.getLogger(__name__)
SCANNET_RAW_ROOT = Path(os.environ.get("DA3_SCANNET_RAW_ROOT", "/robodata/smodak/datasets/scannet_v2/scans"))
VISIBLE_INVALID_COLOR = np.array([255, 255, 255], dtype=np.uint8)
MIN_VISIBLE_COLOR_VALUE = 30
MIN_VISIBLE_COLOR_SUM = 120
RGB_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)
NORMAL_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
    ]
)
INSTANCE_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("instance_id", "<i4"),
    ]
)


@dataclass
class ArtifactPath:
    base_path: Path
    artifact_name: str

    @property
    def rgb_path(self) -> Path:
        return self.base_path / "rgb" / f"{self.artifact_name}.mp4"

    @property
    def pose_path(self) -> Path:
        return self.base_path / "pose" / f"{self.artifact_name}.npz"

    @property
    def depth_path(self) -> Path:
        return self.base_path / "depth" / f"{self.artifact_name}.zip"

    @property
    def rgb_pcd_path(self) -> Path:
        return self.base_path / "pcd" / "rgb.ply"

    @property
    def normals_pcd_path(self) -> Path:
        return self.base_path / "pcd" / "normals.ply"

    @property
    def normals_viz_pcd_path(self) -> Path:
        return self.base_path / "pcd" / "normals_viz.ply"

    @property
    def instance_pcd_path(self) -> Path:
        return self.base_path / "pcd" / "instance.ply"

    @property
    def instance_viz_pcd_path(self) -> Path:
        return self.base_path / "pcd" / "instance_viz.ply"

    @property
    def clip_path(self) -> Path:
        return self.base_path / "pcd" / "clip.npz"

    @property
    def clip_viz_pcd_path(self) -> Path:
        return self.base_path / "pcd" / "clip_viz.ply"

    @property
    def tsdf_pcd_path(self) -> Path:
        return self.base_path / "pcd" / f"{self.artifact_name}_tsdf.ply"

    @property
    def intrinsics_path(self) -> Path:
        return self.base_path / "intrinsics" / f"{self.artifact_name}.npz"

    @property
    def camera_type_path(self) -> Path:
        return self.base_path / "intrinsics" / f"{self.artifact_name}_camera.txt"

    @property
    def meta_info_path(self) -> Path:
        return self.base_path / "vipe" / f"{self.artifact_name}_info.pkl"

    @classmethod
    def glob_artifacts(cls, base_path: Path, use_video: bool = False) -> Iterator["ArtifactPath"]:
        if use_video:
            for artifact_path in (base_path / "rgb").glob("*.mp4"):
                artifact_name = artifact_path.stem
                yield cls(base_path, artifact_name)
        else:
            for artifact_path in (base_path / "vipe").glob("*_info.pkl"):
                artifact_name = artifact_path.stem.replace("_info", "")
                yield cls(base_path, artifact_name)

    @property
    def meta_vis_path(self) -> Path:
        return self.base_path / "vipe" / f"{self.artifact_name}_vis.mp4"

    @property
    def slam_map_path(self) -> Path:
        return self.base_path / "vipe" / f"{self.artifact_name}_slam_map.pt"


def read_pose_artifacts(npz_file_path: Path) -> tuple[np.ndarray, SE3]:
    data = np.load(npz_file_path)
    return data["inds"], se3_matrix_to_se3(data["data"])


def read_pose_artifacts_benchmark(npz_file_path: Path) -> dict:
    data = np.load(npz_file_path)
    return dict(
        ids=data["inds"],
        trajectory=se3_matrix_to_se3(data["data"]),
        runtime=data.get("runtime", None),
        keyframe_ids=data.get("keyframe_ids", None),
        frame_num=len(data["inds"]),
    )


def read_intrinsics_artifacts(
    intr_file_path: Path, camera_file_path: Path | None = None
) -> tuple[np.ndarray, torch.Tensor, list[CameraType]]:
    data = np.load(intr_file_path)
    inds, intrinsics = data["inds"], torch.from_numpy(data["data"])
    if camera_file_path is None or not camera_file_path.exists():
        assert intrinsics.shape[1] == 4
        camera_types = [CameraType.PINHOLE] * intrinsics.shape[0]

    else:
        with camera_file_path.open("r") as f:
            camera_types = [CameraType[line.split(":")[1].strip()] for line in f.readlines()]

    return inds, intrinsics, camera_types


def read_rgb_artifacts(rgb_file_path: Path) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read RGB from H264-encoded video.
    """
    reader = imageio.get_reader(rgb_file_path, "ffmpeg")
    for frame_idx, rgb in enumerate(reader):
        rgb = torch.from_numpy(rgb) / 255.0
        yield frame_idx, rgb


def read_depth_artifacts(zip_file_path: Path) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read metric depth from zipped exr files.
    """
    valid_width, valid_height = 0, 0
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                try:
                    exr = OpenEXR.InputFile(f)
                except OSError:
                    # Sometimes EXR loader might fail, we return all nan maps.
                    logger.warning(f"Failed to load EXR file {zip_file_path}-{file_name}. Returning all nan maps.")
                    assert valid_width > 0 and valid_height > 0
                    yield (
                        frame_idx,
                        torch.full(
                            (valid_height, valid_width),
                            float("nan"),
                            dtype=torch.float32,
                        ),
                    )
                    continue
                header = exr.header()
                dw = header["dataWindow"]
                valid_width = width = dw.max.x - dw.min.x + 1
                valid_height = height = dw.max.y - dw.min.y + 1
                channels = exr.channels(["Z"])
                depth_data = np.frombuffer(channels[0], dtype=np.float16).reshape((height, width))
                yield frame_idx, torch.from_numpy(depth_data.copy()).float()


class ArtifactFrameStream(FrameStream):
    def __init__(self, artifact_path: ArtifactPath) -> None:
        self.artifact_path = artifact_path
        self._name = artifact_path.artifact_name

        rgb_reader = imageio.get_reader(artifact_path.rgb_path, "ffmpeg")
        try:
            first_rgb = rgb_reader.get_data(0)
            self._frame_size = (first_rgb.shape[0], first_rgb.shape[1])
            self._fps = float(rgb_reader.get_meta_data()["fps"])
            self._len = rgb_reader.count_frames()
        finally:
            rgb_reader.close()

        self._attributes: set[FrameAttribute] = set()
        self.pose_by_idx: dict[int, SE3] = {}
        self.intrinsics_by_idx: dict[int, torch.Tensor] = {}
        self.camera_type_by_idx: dict[int, CameraType] = {}

        if artifact_path.pose_path.exists():
            pose_inds, pose_data = read_pose_artifacts(artifact_path.pose_path)
            self.pose_by_idx = {int(frame_idx): pose_data[list_idx] for list_idx, frame_idx in enumerate(pose_inds.tolist())}
            self._attributes.add(FrameAttribute.POSE)

        if artifact_path.intrinsics_path.exists():
            intr_inds, intrinsics_data, camera_types = read_intrinsics_artifacts(
                artifact_path.intrinsics_path,
                artifact_path.camera_type_path,
            )
            self.intrinsics_by_idx = {
                int(frame_idx): intrinsics_data[list_idx]
                for list_idx, frame_idx in enumerate(intr_inds.tolist())
            }
            self.camera_type_by_idx = {
                int(frame_idx): camera_types[list_idx]
                for list_idx, frame_idx in enumerate(intr_inds.tolist())
            }
            self._attributes.update({FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE})

        if artifact_path.depth_path.exists():
            self._attributes.add(FrameAttribute.METRIC_DEPTH)

    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    def name(self) -> str:
        return self._name

    def fps(self) -> float:
        return self._fps

    def __len__(self) -> int:
        return self._len

    def attributes(self) -> set[FrameAttribute]:
        return self._attributes

    def __iter__(self):
        depth_iterator = read_depth_artifacts(self.artifact_path.depth_path) if self.artifact_path.depth_path.exists() else None
        next_depth = next(depth_iterator, None) if depth_iterator is not None else None

        for frame_idx, rgb in read_rgb_artifacts(self.artifact_path.rgb_path):
            metric_depth = None
            while next_depth is not None and next_depth[0] < frame_idx:
                next_depth = next(depth_iterator, None)
            if next_depth is not None and next_depth[0] == frame_idx:
                metric_depth = next_depth[1]
                next_depth = next(depth_iterator, None)

            yield FrameData(
                raw_frame_idx=frame_idx,
                rgb=rgb,
                pose=self.pose_by_idx.get(frame_idx),
                camera_type=self.camera_type_by_idx.get(frame_idx),
                intrinsics=self.intrinsics_by_idx.get(frame_idx),
                metric_depth=metric_depth,
            )


def _backproject_vertices(
    frame_data: FrameData,
    max_points_per_frame: int,
    conf_threshold_coef: float,
    sample_ratio: float,
) -> np.ndarray | None:
    if (
        frame_data.metric_depth is None
        or frame_data.pose is None
        or frame_data.intrinsics is None
        or max_points_per_frame <= 0
    ):
        return None

    depth = frame_data.metric_depth.detach().cpu().numpy()
    valid = np.isfinite(depth) & (depth > 0.0)
    confidence = None
    if frame_data.depth_confidence is not None:
        confidence = frame_data.depth_confidence.detach().cpu().numpy()
        assert confidence.shape == depth.shape
        valid &= (confidence >= float(np.mean(confidence)) * conf_threshold_coef) & (confidence > 1e-5)

    valid_flat = np.flatnonzero(valid.ravel())
    if len(valid_flat) == 0:
        return None

    if confidence is not None and sample_ratio < 1.0:
        sample_count = int(len(valid_flat) * sample_ratio)
    else:
        sample_count = len(valid_flat)
    sample_count = min(sample_count, max_points_per_frame)
    if sample_count <= 0:
        return None

    if sample_count < len(valid_flat):
        if confidence is not None:
            rng = np.random.default_rng(frame_data.raw_frame_idx)
            valid_flat = rng.choice(valid_flat, sample_count, replace=False)
        else:
            stride = max(1, math.ceil(len(valid_flat) / sample_count))
            valid_flat = valid_flat[::stride][:sample_count]

    height, width = depth.shape
    ys, xs = np.divmod(valid_flat, width)
    zs = depth.ravel()[valid_flat].astype(np.float32)
    fx, fy, cx, cy = frame_data.intrinsics[:4].detach().cpu().numpy().astype(np.float32)

    points_cam = np.empty((len(valid_flat), 4), dtype=np.float32)
    points_cam[:, 0] = (xs.astype(np.float32) - cx) * zs / fx
    points_cam[:, 1] = (ys.astype(np.float32) - cy) * zs / fy
    points_cam[:, 2] = zs
    points_cam[:, 3] = 1.0

    pose_c2w = frame_data.pose.matrix().detach().cpu().numpy().astype(np.float32)
    points_world = (pose_c2w @ points_cam.T).T[:, :3]
    colors = (frame_data.rgb.detach().cpu().numpy().reshape(-1, 3)[valid_flat] * 255.0).clip(0, 255).astype(np.uint8)

    vertices = np.empty(len(points_world), dtype=RGB_VERTEX_DTYPE)
    vertices["x"] = points_world[:, 0]
    vertices["y"] = points_world[:, 1]
    vertices["z"] = points_world[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    return vertices


def _write_ply_header(ply_file, vertex_count: int, properties: list[str]) -> None:
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {vertex_count}", *properties, "end_header"]
    ply_file.write(("\n".join(header) + "\n").encode("ascii"))


def _write_backproject_pcd(out_path: ArtifactPath, body_file, vertex_count: int) -> None:
    if vertex_count == 0:
        return

    out_path.rgb_pcd_path.parent.mkdir(exist_ok=True, parents=True)
    body_file.seek(0)
    with out_path.rgb_pcd_path.open("wb") as ply_file:
        _write_ply_header(
            ply_file,
            vertex_count,
            [
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ],
        )
        shutil.copyfileobj(body_file, ply_file)


def _source_frame_files(source_frame_dir: Path) -> list[Path]:
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    frame_files = []
    for ext in image_extensions:
        frame_files.extend(source_frame_dir.glob(f"*{ext}"))
        frame_files.extend(source_frame_dir.glob(f"*{ext.upper()}"))
    return sort_image_sequence(set(frame_files))


def _load_scannet_vertex_annotations(raw_scene_dir: Path, scene: str, vertex_count: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    seg_path = raw_scene_dir / f"{scene}_vh_clean_2.0.010000.segs.json"
    agg_path = raw_scene_dir / f"{scene}.aggregation.json"

    with seg_path.open("r") as f:
        seg_indices = np.asarray(json.load(f)["segIndices"], dtype=np.int64)
    assert len(seg_indices) == vertex_count

    with agg_path.open("r") as f:
        seg_groups = json.load(f)["segGroups"]

    semantic_labels = sorted({str(group["label"]) for group in seg_groups})
    assert len(semantic_labels) <= np.iinfo(np.int16).max
    label_to_id = {label: label_id for label_id, label in enumerate(semantic_labels)}

    seg_to_instance = {}
    seg_to_semantic = {}
    for group in seg_groups:
        instance_id = int(group["objectId"])
        semantic_id = label_to_id[str(group["label"])]
        for segment_id in group["segments"]:
            seg_to_instance[int(segment_id)] = instance_id
            seg_to_semantic[int(segment_id)] = semantic_id

    instance_ids = np.full(vertex_count, -1, dtype=np.int32)
    semantic_ids = np.full(vertex_count, -1, dtype=np.int16)
    for vertex_idx, segment_id in enumerate(seg_indices.tolist()):
        instance_ids[vertex_idx] = seg_to_instance.get(segment_id, -1)
        semantic_ids[vertex_idx] = seg_to_semantic.get(segment_id, -1)
    return instance_ids, semantic_ids, semantic_labels


def _load_aligned_scannet_gt(
    source_frame_dir: Path,
    first_raw_frame_idx: int,
    first_pose_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    import open3d as o3d

    scene_dir = source_frame_dir.parent
    scene = scene_dir.name
    raw_scene_dir = SCANNET_RAW_ROOT / scene

    mesh = o3d.io.read_triangle_mesh(str(raw_scene_dir / f"{scene}_vh_clean_2.ply"))
    gt_points = np.asarray(mesh.vertices, dtype=np.float32)
    gt_normals = np.load(scene_dir / f"{scene}_vh_clean_2.vertex_normals.npy").astype(np.float32)
    gt_instances, gt_semantics, semantic_labels = _load_scannet_vertex_annotations(raw_scene_dir, scene, len(gt_points))
    assert gt_normals.shape == gt_points.shape

    frame_files = _source_frame_files(source_frame_dir)
    frame_id = frame_files[first_raw_frame_idx].stem
    scannet_first_pose = np.loadtxt(scene_dir / "pose" / f"{frame_id}.txt", dtype=np.float32)
    vipe_from_scannet = first_pose_c2w @ np.linalg.inv(scannet_first_pose)

    gt_points_h = np.concatenate([gt_points, np.ones((len(gt_points), 1), dtype=np.float32)], axis=1)
    aligned_points = (vipe_from_scannet @ gt_points_h.T).T[:, :3].astype(np.float32)
    aligned_normals = (gt_normals @ vipe_from_scannet[:3, :3].T).astype(np.float32)
    aligned_normals /= np.maximum(np.linalg.norm(aligned_normals, axis=1, keepdims=True), 1e-8)
    return aligned_points, aligned_normals, gt_instances, gt_semantics, semantic_labels


def _vertex_points(vertices: np.ndarray) -> np.ndarray:
    points = np.empty((len(vertices), 3), dtype=np.float32)
    points[:, 0] = vertices["x"]
    points[:, 1] = vertices["y"]
    points[:, 2] = vertices["z"]
    return points


def _normal_colors(normals: np.ndarray) -> np.ndarray:
    return _avoid_black_colors(((normals * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8))


def _instance_colors(instance_ids: np.ndarray) -> np.ndarray:
    ids = np.maximum(instance_ids, 0).astype(np.uint32)
    colors = np.empty((len(instance_ids), 3), dtype=np.uint8)
    colors[:, 0] = ((ids * 37 + 29) % 255).astype(np.uint8)
    colors[:, 1] = ((ids * 67 + 71) % 255).astype(np.uint8)
    colors[:, 2] = ((ids * 97 + 113) % 255).astype(np.uint8)
    colors[instance_ids < 0] = VISIBLE_INVALID_COLOR
    return _avoid_black_colors(colors)


def _avoid_black_colors(colors: np.ndarray) -> np.ndarray:
    near_black = (colors.max(axis=1) < MIN_VISIBLE_COLOR_VALUE) | (colors.astype(np.uint16).sum(axis=1) < MIN_VISIBLE_COLOR_SUM)
    colors[near_black] = VISIBLE_INVALID_COLOR
    return colors


def _average_knn_normals(neighbor_normals: np.ndarray) -> np.ndarray:
    normals = neighbor_normals.mean(axis=1).astype(np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return normals


def _knn_vote_ids(neighbor_ids: np.ndarray) -> np.ndarray:
    valid_ids = np.unique(neighbor_ids[neighbor_ids >= 0])
    voted_ids = np.full(len(neighbor_ids), -1, dtype=neighbor_ids.dtype)
    if len(valid_ids) == 0:
        return voted_ids

    mapped_ids = np.full(neighbor_ids.shape, -1, dtype=np.int32)
    counts = np.zeros((len(neighbor_ids), len(valid_ids)), dtype=np.uint8)
    row_ids = np.arange(len(neighbor_ids))
    for neighbor_idx in range(neighbor_ids.shape[1]):
        ids = neighbor_ids[:, neighbor_idx]
        valid = ids >= 0
        compact_ids = np.searchsorted(valid_ids, ids[valid])
        mapped_ids[valid, neighbor_idx] = compact_ids
        counts[row_ids[valid], compact_ids] += 1

    max_counts = counts.max(axis=1)
    neighbor_counts = np.take_along_axis(counts, np.maximum(mapped_ids, 0), axis=1)
    neighbor_counts[mapped_ids < 0] = 0
    nearest_winner = ((neighbor_counts == max_counts[:, None]) & (mapped_ids >= 0)).argmax(axis=1)
    voted_ids = neighbor_ids[row_ids, nearest_winner]
    voted_ids[max_counts == 0] = -1
    return voted_ids


def _encode_clip_labels(label_texts: list[str]) -> np.ndarray:
    import open_clip

    model = open_clip.create_model("ViT-L-14-336-quickgelu", pretrained="openai").eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14-336-quickgelu")
    tokens = tokenizer(label_texts)
    with torch.inference_mode():
        embeddings = model.encode_text(tokens).float()
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return embeddings.cpu().numpy().astype(np.float16)


def _clip_pca_colors(label_embeddings: np.ndarray) -> np.ndarray:
    features = label_embeddings.astype(np.float32)
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    n_components = min(3, len(vh))
    projected = np.zeros((len(features), 3), dtype=np.float32)
    if n_components > 0:
        projected[:, :n_components] = centered @ vh[:n_components].T

    colors = np.empty((len(features), 3), dtype=np.uint8)
    for dim in range(3):
        values = projected[:, dim]
        min_value = float(values.min())
        max_value = float(values.max())
        if max_value > min_value:
            colors[:, dim] = ((values - min_value) / (max_value - min_value) * 255.0).astype(np.uint8)
        else:
            colors[:, dim] = 128
    return _avoid_black_colors(colors)


def _read_vertex_chunk(body_file, max_vertices: int) -> np.ndarray:
    data = body_file.read(max_vertices * RGB_VERTEX_DTYPE.itemsize)
    return np.frombuffer(data, dtype=RGB_VERTEX_DTYPE)


def _write_scannet_annotation_pcds(
    out_path: ArtifactPath,
    body_file,
    vertex_count: int,
    source_frame_dir: Path,
    first_raw_frame_idx: int,
    first_pose_c2w: np.ndarray,
) -> None:
    if vertex_count == 0:
        return
    scene_dir = source_frame_dir.parent
    scene = scene_dir.name
    raw_scene_dir = SCANNET_RAW_ROOT / scene
    required_paths = [
        raw_scene_dir / f"{scene}_vh_clean_2.ply",
        raw_scene_dir / f"{scene}_vh_clean_2.0.010000.segs.json",
        raw_scene_dir / f"{scene}.aggregation.json",
        scene_dir / f"{scene}_vh_clean_2.vertex_normals.npy",
    ]
    if any(not path.exists() for path in required_paths):
        logger.info(f"Skipping ScanNet annotation PLYs; GT files not found for scene {scene}")
        return

    from scipy.spatial import cKDTree

    gt_points, gt_normals, gt_instances, gt_semantics, semantic_labels = _load_aligned_scannet_gt(
        source_frame_dir,
        first_raw_frame_idx,
        first_pose_c2w,
    )
    tree = cKDTree(gt_points)
    clip_embeddings = _encode_clip_labels(semantic_labels)
    clip_colors = _clip_pca_colors(clip_embeddings)
    point_clip_label_ids = np.empty(vertex_count, dtype=np.int16)

    out_path.normals_pcd_path.parent.mkdir(exist_ok=True, parents=True)
    body_file.seek(0)
    written = 0
    chunk_size = 500_000
    query_workers = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
    with (
        out_path.normals_pcd_path.open("wb") as normals_file,
        out_path.normals_viz_pcd_path.open("wb") as normals_viz_file,
        out_path.instance_pcd_path.open("wb") as instance_file,
        out_path.instance_viz_pcd_path.open("wb") as instance_viz_file,
        out_path.clip_viz_pcd_path.open("wb") as clip_viz_file,
    ):
        _write_ply_header(
            normals_file,
            vertex_count,
            [
                "property float x",
                "property float y",
                "property float z",
                "property float nx",
                "property float ny",
                "property float nz",
            ],
        )
        rgb_properties = [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        _write_ply_header(normals_viz_file, vertex_count, rgb_properties)
        _write_ply_header(
            instance_file,
            vertex_count,
            [
                "property float x",
                "property float y",
                "property float z",
                "property int instance_id",
            ],
        )
        _write_ply_header(instance_viz_file, vertex_count, rgb_properties)
        _write_ply_header(clip_viz_file, vertex_count, rgb_properties)

        for _ in pbar(range(math.ceil(vertex_count / chunk_size)), desc="Writing ScanNet annotation PLYs"):
            vertices = _read_vertex_chunk(body_file, min(chunk_size, vertex_count - written))
            if len(vertices) == 0:
                break

            points = _vertex_points(vertices)
            _, nn_indices = tree.query(points, k=5, workers=query_workers)
            normals = _average_knn_normals(gt_normals[nn_indices])
            instance_ids = _knn_vote_ids(gt_instances[nn_indices])
            semantic_ids = _knn_vote_ids(gt_semantics[nn_indices])
            point_clip_label_ids[written : written + len(vertices)] = semantic_ids

            normal_vertices = np.empty(len(vertices), dtype=NORMAL_VERTEX_DTYPE)
            normal_vertices["x"] = points[:, 0]
            normal_vertices["y"] = points[:, 1]
            normal_vertices["z"] = points[:, 2]
            normal_vertices["nx"] = normals[:, 0]
            normal_vertices["ny"] = normals[:, 1]
            normal_vertices["nz"] = normals[:, 2]
            normal_vertices.tofile(normals_file)

            normal_viz_vertices = np.empty(len(vertices), dtype=RGB_VERTEX_DTYPE)
            normal_viz_vertices["x"] = points[:, 0]
            normal_viz_vertices["y"] = points[:, 1]
            normal_viz_vertices["z"] = points[:, 2]
            normal_colors = _normal_colors(normals)
            normal_viz_vertices["red"] = normal_colors[:, 0]
            normal_viz_vertices["green"] = normal_colors[:, 1]
            normal_viz_vertices["blue"] = normal_colors[:, 2]
            normal_viz_vertices.tofile(normals_viz_file)

            instance_vertices = np.empty(len(vertices), dtype=INSTANCE_VERTEX_DTYPE)
            instance_vertices["x"] = points[:, 0]
            instance_vertices["y"] = points[:, 1]
            instance_vertices["z"] = points[:, 2]
            instance_vertices["instance_id"] = instance_ids
            instance_vertices.tofile(instance_file)

            instance_viz_vertices = np.empty(len(vertices), dtype=RGB_VERTEX_DTYPE)
            instance_viz_vertices["x"] = points[:, 0]
            instance_viz_vertices["y"] = points[:, 1]
            instance_viz_vertices["z"] = points[:, 2]
            instance_colors = _instance_colors(instance_ids)
            instance_viz_vertices["red"] = instance_colors[:, 0]
            instance_viz_vertices["green"] = instance_colors[:, 1]
            instance_viz_vertices["blue"] = instance_colors[:, 2]
            instance_viz_vertices.tofile(instance_viz_file)

            clip_viz_vertices = np.empty(len(vertices), dtype=RGB_VERTEX_DTYPE)
            clip_viz_vertices["x"] = points[:, 0]
            clip_viz_vertices["y"] = points[:, 1]
            clip_viz_vertices["z"] = points[:, 2]
            semantic_colors = np.tile(VISIBLE_INVALID_COLOR, (len(vertices), 1))
            valid_semantics = semantic_ids >= 0
            semantic_colors[valid_semantics] = clip_colors[semantic_ids[valid_semantics]]
            semantic_colors = _avoid_black_colors(semantic_colors)
            clip_viz_vertices["red"] = semantic_colors[:, 0]
            clip_viz_vertices["green"] = semantic_colors[:, 1]
            clip_viz_vertices["blue"] = semantic_colors[:, 2]
            clip_viz_vertices.tofile(clip_viz_file)
            written += len(vertices)
    assert written == vertex_count
    np.savez_compressed(
        out_path.clip_path,
        point_label_ids=point_clip_label_ids,
        label_texts=np.asarray(semantic_labels),
        label_embeddings=clip_embeddings,
        embedding_model=np.asarray("ViT-L-14-336-quickgelu"),
        pretrained=np.asarray("openai"),
        normalized=np.asarray(True),
    )


def _make_tsdf_volume(voxel_length: float, sdf_trunc: float):
    import open3d as o3d

    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def _integrate_tsdf_frame(volume, frame_data: FrameData, depth_trunc: float) -> None:
    if frame_data.metric_depth is None or frame_data.pose is None or frame_data.intrinsics is None:
        return

    import open3d as o3d

    depth = frame_data.metric_depth.detach().cpu().numpy().astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0.0] = 0.0

    color = (frame_data.rgb.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    height, width = depth.shape
    fx, fy, cx, cy = frame_data.intrinsics[:4].detach().cpu().numpy().astype(np.float32)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(fx),
        float(fy),
        float(cx),
        float(cy),
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    w2c = frame_data.pose.inv().matrix().detach().cpu().numpy().astype(np.float64)
    volume.integrate(rgbd, intrinsics, w2c)


def _write_tsdf_pcd(out_path: ArtifactPath, volume, max_points: int) -> None:
    import open3d as o3d

    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return

    out_path.tsdf_pcd_path.parent.mkdir(exist_ok=True, parents=True)
    pcd = mesh.sample_points_uniformly(number_of_points=max_points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    o3d.io.write_point_cloud(str(out_path.tsdf_pcd_path), pcd, write_ascii=False)


def save_artifacts(
    out_path: ArtifactPath,
    final_stream: FrameStream,
    source_frame_dir: Path,
    pcd_fusion_mode: str = "backproject",
    max_pcd_points: int = 8_000_000,
    pcd_conf_threshold_coef: float = 0.75,
    pcd_sample_ratio: float = 0.015,
    pcd_tsdf_voxel_length: float = 0.02,
    pcd_tsdf_sdf_trunc: float = 0.15,
    pcd_tsdf_depth_trunc: float = 5.0,
) -> None:
    """
    Save artifacts in a single streaming pass to avoid retaining the full sequence in RAM.
    """

    pose_list = []
    intrinsics_list = []
    camera_type_list = []
    first_raw_frame_idx = None
    first_pose_c2w = None
    depth_zip: zipfile.ZipFile | None = None
    if pcd_fusion_mode not in {"backproject", "tsdf"}:
        raise ValueError(f"Invalid pcd_fusion_mode: {pcd_fusion_mode}")

    pcd_body_file = tempfile.TemporaryFile() if pcd_fusion_mode == "backproject" else None
    pcd_vertex_count = 0
    max_points_per_frame = math.ceil(max_pcd_points / max(len(final_stream), 1))
    tsdf_volume = (
        _make_tsdf_volume(pcd_tsdf_voxel_length, pcd_tsdf_sdf_trunc) if pcd_fusion_mode == "tsdf" else None
    )

    try:
        with VideoWriter(out_path.rgb_path, final_stream.fps()) as rgb_writer:
            for frame_idx, frame_data in pbar(
                enumerate(final_stream),
                total=len(final_stream),
                desc="Saving artifacts",
            ):
                assert isinstance(frame_data, FrameData)

                if frame_data.pose is not None:
                    pose_list.append((frame_idx, frame_data.pose.matrix().cpu().numpy()))
                    if first_pose_c2w is None:
                        first_raw_frame_idx = frame_data.raw_frame_idx
                        first_pose_c2w = pose_list[-1][1].astype(np.float32)

                if frame_data.intrinsics is not None:
                    intrinsics_list.append((frame_idx, frame_data.intrinsics.cpu().numpy()))

                if frame_data.camera_type is not None:
                    camera_type_list.append((frame_idx, frame_data.camera_type))

                rgb_writer.write((frame_data.rgb.cpu().numpy() * 255).astype(np.uint8))

                if frame_data.metric_depth is not None:
                    if depth_zip is None:
                        out_path.depth_path.parent.mkdir(exist_ok=True, parents=True)
                        depth_zip = zipfile.ZipFile(out_path.depth_path, "w", zipfile.ZIP_DEFLATED)

                    metric_depth = frame_data.metric_depth.cpu().numpy()
                    height, width = metric_depth.shape
                    header = OpenEXR.Header(width, height)
                    header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
                    with tempfile.NamedTemporaryFile(suffix=".exr") as f:
                        exr = OpenEXR.OutputFile(f.name, header)
                        exr.writePixels({"Z": metric_depth.astype(np.float16).tobytes()})
                        exr.close()
                        depth_zip.write(f.name, f"{frame_idx:05d}.exr")

                remaining_points = max_pcd_points - pcd_vertex_count
                if pcd_fusion_mode == "backproject" and remaining_points > 0:
                    assert pcd_body_file is not None
                    vertices = _backproject_vertices(
                        frame_data,
                        min(max_points_per_frame, remaining_points),
                        pcd_conf_threshold_coef,
                        pcd_sample_ratio,
                    )
                    if vertices is not None:
                        vertices.tofile(pcd_body_file)
                        pcd_vertex_count += len(vertices)
                elif pcd_fusion_mode == "tsdf":
                    _integrate_tsdf_frame(tsdf_volume, frame_data, pcd_tsdf_depth_trunc)
    except Exception:
        if pcd_body_file is not None:
            pcd_body_file.close()
        raise
    finally:
        if depth_zip is not None:
            depth_zip.close()

    if len(pose_list) > 0:
        pose_data = np.stack([pose for _, pose in pose_list], axis=0)
        pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
        out_path.pose_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(out_path.pose_path, data=pose_data, inds=pose_inds)

    if len(intrinsics_list) > 0:
        intrinsics_data = np.stack([intrinsics for _, intrinsics in intrinsics_list], axis=0)
        intrinsics_inds = np.array([frame_idx for frame_idx, _ in intrinsics_list])
        out_path.intrinsics_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(out_path.intrinsics_path, data=intrinsics_data, inds=intrinsics_inds)

    if len(camera_type_list) > 0:
        out_path.camera_type_path.parent.mkdir(exist_ok=True, parents=True)
        with out_path.camera_type_path.open("w") as f:
            for frame_idx, camera_type_data in camera_type_list:
                f.write(f"{frame_idx}: {camera_type_data.name}\n")

    try:
        if pcd_fusion_mode == "backproject":
            assert pcd_body_file is not None
            _write_backproject_pcd(out_path, pcd_body_file, pcd_vertex_count)
            assert first_raw_frame_idx is not None and first_pose_c2w is not None
            _write_scannet_annotation_pcds(
                out_path,
                pcd_body_file,
                pcd_vertex_count,
                source_frame_dir,
                first_raw_frame_idx,
                first_pose_c2w,
            )
        else:
            _write_tsdf_pcd(out_path, tsdf_volume, max_pcd_points)
    finally:
        if pcd_body_file is not None:
            pcd_body_file.close()
