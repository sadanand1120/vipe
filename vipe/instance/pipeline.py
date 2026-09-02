"""Post-ViPE class-agnostic 3D instance distillation pipeline."""

import colorsys
import gc
import json
import logging
import time

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from vipe.instance.association import associate, build_atom_graph, frame_coreset_poses
from vipe.instance.masks import generate_and_lift
from vipe.instance.semantic import (
    distill_semantic_features,
    write_hypothesis_average_semantic_pca_ply,
    write_point_semantic_pca_ply,
)
from vipe.utils.io import ArtifactPath


logger = logging.getLogger(__name__)


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _contract_geometry(width: int, height: int, long_side: int):
    scale = long_side / max(width, height)
    out_width, out_height = int(round(width * scale)), int(round(height * scale))
    x = np.minimum((np.arange(out_width) / (out_width / width)).astype(np.int64), width - 1)
    y = np.minimum((np.arange(out_height) / (out_height / height)).astype(np.int64), height - 1)
    return out_width, out_height, x, y


def _pack_hypotheses(hypotheses: list[np.ndarray]):
    lengths = np.fromiter((len(hypothesis) for hypothesis in hypotheses), np.int64, len(hypotheses))
    offsets = np.empty(len(hypotheses) + 1, np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    indices = (
        np.concatenate(hypotheses).astype(np.int32, copy=False)
        if hypotheses
        else np.empty(0, np.int32)
    )
    return indices, offsets


def _spatial_instance_colors(points: np.ndarray, owner: np.ndarray, count: int) -> np.ndarray:
    """Assign maximally different colors to instances that touch in the output cloud."""
    from scipy.spatial import cKDTree

    palette = np.asarray(
        [
            tuple(
                int(channel * 255)
                for channel in colorsys.hsv_to_rgb(
                    hue / 12.0,
                    0.92 if band == 0 else 0.68,
                    1.0 if band == 0 else 0.78,
                )
            )
            for band in range(2)
            for hue in range(12)
        ],
        dtype=np.uint8,
    )
    visible = np.unique(owner[owner >= 0])
    colors = np.zeros((count, 3), dtype=np.uint8)
    if not len(visible):
        return colors

    neighbor_count = min(17, len(points))
    _, neighbors = cKDTree(points).query(points, k=neighbor_count, workers=1)
    if neighbor_count == 1:
        neighbors = neighbors[:, None]
    left = np.repeat(owner, neighbor_count - 1)
    right = owner[neighbors[:, 1:].reshape(-1)]
    keep = (left >= 0) & (right >= 0) & (left != right)
    lo, hi = np.minimum(left[keep], right[keep]), np.maximum(left[keep], right[keep])
    edges = np.unique(lo.astype(np.int64) * count + hi) if len(lo) else np.empty(0, np.int64)
    adjacency = [set() for _ in range(count)]
    for edge in edges:
        left_id, right_id = divmod(int(edge), count)
        adjacency[left_id].add(right_id)
        adjacency[right_id].add(left_id)

    assigned = np.full(count, -1, dtype=np.int32)
    usage = np.zeros(len(palette), dtype=np.int32)
    palette_float = palette.astype(np.float64)
    for instance_id in sorted(visible.tolist(), key=lambda value: (-len(adjacency[value]), value)):
        neighbor_colors = [assigned[value] for value in adjacency[instance_id] if assigned[value] >= 0]
        if neighbor_colors:
            separation = (
                (palette_float[:, None] - palette_float[neighbor_colors][None]) ** 2
            ).sum(axis=2).min(axis=1)
            best = np.flatnonzero(separation == separation.max())
            color_id = int(best[np.argmin(usage[best])])
        else:
            color_id = int(np.argmin(usage))
        assigned[instance_id] = color_id
        usage[color_id] += 1
        colors[instance_id] = palette[color_id]
    return colors


def _write_labeled_ply(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    labels = np.asarray(labels, dtype=np.int32)
    if labels.shape != (len(points),):
        raise ValueError(f"Point labels must have shape ({len(points)},), got {labels.shape}")
    visible = np.unique(labels[labels >= 0])
    compact = np.full(len(labels), -1, dtype=np.int32)
    compact[labels >= 0] = np.searchsorted(visible, labels[labels >= 0])
    colors = np.tile(np.array([60, 60, 60], np.uint8), (len(points), 1))
    assigned = compact >= 0
    colors[assigned] = _spatial_instance_colors(points, compact, len(visible))[compact[assigned]]

    records = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("instance", "<i4"),
        ],
    )
    records["x"], records["y"], records["z"] = points.T
    records["red"], records["green"], records["blue"] = colors.T
    records["instance"] = labels
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int instance\nend_header\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(records.tobytes())


def write_labeled_instance_ply(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    """Write one preserved instance ID per point with spatially contrasting colors."""
    _write_labeled_ply(path, points, labels)


def write_instance_ply(path: Path, points: np.ndarray, hypotheses: list[np.ndarray]) -> None:
    """Write a smallest-hypothesis-wins visualization of the TSDF surface prediction."""
    owner = np.full(len(points), -1, np.int32)
    owner_size = np.full(len(points), np.iinfo(np.int64).max, np.int64)
    for hypothesis_id, voxels in enumerate(hypotheses):
        take = voxels[len(voxels) < owner_size[voxels]]
        owner[take] = hypothesis_id
        owner_size[take] = len(voxels)
    _write_labeled_ply(path, points, owner)


class InstancePipeline:
    """Run the fixed instance pipeline after ViPE pose and TSDF stages."""

    def __init__(self, config, device: torch.device = torch.device("cuda")) -> None:
        self.config = config
        self.device = device

    @staticmethod
    def _artifact_paths(artifact_path: ArtifactPath):
        name = artifact_path.artifact_name
        instance_dir = artifact_path.base_path / "instances"
        return (
            instance_dir / f"{name}.npz",
            instance_dir / f"{name}_summary.json",
            artifact_path.base_path / "pcd" / f"{name}_instances.ply",
            artifact_path.base_path / "pcd" / f"{name}_semantic_pca_A_dense.ply",
            artifact_path.base_path / "pcd" / f"{name}_semantic_pca_C_hypavg.ply",
        )

    def run(
        self,
        frame_stream,
        pose_matrices: np.ndarray,
        intrinsics: np.ndarray,
        artifact_path: ArtifactPath,
        points: np.ndarray,
        normals: np.ndarray,
        tsdf_voxel_edge_m: float,
    ) -> dict:
        """Distill overlapping 3D instances and return the persisted scene summary."""
        started = time.perf_counter()
        poses = np.ascontiguousarray(pose_matrices, dtype=np.float32)
        intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(4)
        if poses.shape != (len(frame_stream), 4, 4):
            raise ValueError(
                f"Expected {(len(frame_stream), 4, 4)} c2w poses, received {poses.shape}"
            )

        points = np.ascontiguousarray(points, dtype=np.float32)
        normals = np.ascontiguousarray(normals, dtype=np.float32)
        if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Reduced TSDF points and normals must have matching (N, 3) shapes")
        logger.info("Stage 6: using %d native TSDF surface points", len(points))
        frames_config = self.config["frames"]
        frame_indices = frame_coreset_poses(
            list(range(len(poses))),
            poses.__getitem__,
            float(frames_config["move_cm"]),
            float(frames_config["move_deg"]),
        )
        source_height, source_width = frame_stream.frame_size
        image_config = self.config["images"]
        width, height, x_index, y_index = _contract_geometry(
            source_width, source_height, int(image_config["long_side"])
        )
        scaled_intrinsics = intrinsics.copy()
        scaled_intrinsics[[0, 2]] *= width / source_width
        scaled_intrinsics[[1, 3]] *= height / source_height

        def rgb_of(frame_index: int) -> np.ndarray:
            from PIL import Image

            rgb, _ = frame_stream.artifact_arrays(frame_index)
            return np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS))

        def depth_of(frame_index: int) -> np.ndarray:
            depth = frame_stream._read_depth(frame_index, frame_stream.frame_size)
            return depth[np.ix_(y_index, x_index)]

        logger.info("Stage 7: building normal-aware surface atoms")
        tick = time.perf_counter()
        atom_graph = build_atom_graph(
            points,
            normals,
            self.config["association"],
            tsdf_voxel_edge_m,
        )
        atom_seconds = time.perf_counter() - tick

        logger.info("Stage 8: generating, propagating, and lifting instance masks")
        tick = time.perf_counter()
        points_device = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        lifted_evidence, mask_stats = generate_and_lift(
            points_device,
            atom_graph["atom_of"],
            (atom_graph["aa"], atom_graph["ab"]),
            frame_indices,
            rgb_of,
            depth_of,
            poses.__getitem__,
            scaled_intrinsics,
            width,
            height,
            self.config["masks"],
            self.config["lift"],
            int(image_config["jpeg_quality"]),
            self.device,
        )
        mask_seconds = time.perf_counter() - tick
        del points_device
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Stages 9-11: selecting overlapping 3D hypotheses")
        association_result = associate(
            atom_graph,
            lifted_evidence,
            len(points),
            self.config["association"],
            atom_seconds=atom_seconds,
        )
        hypotheses = association_result["hypotheses"]
        lifted_frames = int(lifted_evidence["n_frames"])
        lifted_masks = int(len(lifted_evidence["gm_frame"]))
        del atom_graph, lifted_evidence
        gc.collect()

        logger.info("Stage 12: distilling FG-CLIP semantic descriptors")
        tick = time.perf_counter()
        point_features, feature_metrics = distill_semantic_features(
            features=self.config["features"],
            points=points,
            normals=normals,
            frame_indices=frame_indices,
            rgb_of=rgb_of,
            depth_of=depth_of,
            poses=poses,
            intrinsics=scaled_intrinsics,
            width=width,
            height=height,
            device=self.device,
        )
        feature_seconds = time.perf_counter() - tick
        gc.collect()
        torch.cuda.empty_cache()

        indices, offsets = _pack_hypotheses(hypotheses)
        prediction_path, summary_path, ply_path, semantic_a_path, semantic_c_path = (
            self._artifact_paths(artifact_path)
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            prediction_path,
            points=points,
            hypothesis_indices=indices,
            hypothesis_offsets=offsets,
            K=np.int32(self.config["association"]["membership_budget"]),
            domain=np.array("tsdf_surface"),
            voxel_edge_m=np.float32(tsdf_voxel_edge_m),
            point_features=point_features,
            feature_grid=np.int32(feature_metrics["grid"]),
        )
        write_instance_ply(ply_path, points, hypotheses)
        tick = time.perf_counter()
        direct_coverage = write_point_semantic_pca_ply(
            semantic_a_path,
            points,
            normals,
            point_features,
        )
        semantic_a_seconds = time.perf_counter() - tick
        if not np.isclose(direct_coverage, feature_metrics["direct_point_hit_fraction"]):
            raise RuntimeError("Direct semantic coverage changed during artifact serialization")
        tick = time.perf_counter()
        semantic_coverage, valid_hypotheses = write_hypothesis_average_semantic_pca_ply(
            semantic_c_path,
            points,
            normals,
            point_features,
            hypotheses,
            device=self.device,
        )
        semantic_c_seconds = time.perf_counter() - tick
        feature_metrics["valid_hypothesis_descriptor_count"] = valid_hypotheses
        feature_metrics["instance_field_coverage"] = semantic_coverage

        summary = {
            "schema_version": 4,
            "scene": artifact_path.artifact_name,
            "config": _plain(self.config),
            "frames": {"total": len(poses), "selected": len(frame_indices)},
            "counts": {
                "points": len(points),
                "lifted_frames": lifted_frames,
                "lifted_masks": lifted_masks,
                **mask_stats,
                **association_result["counts"],
                "semantic_point_descriptors": feature_metrics["valid_point_descriptor_count"],
                "semantic_hypothesis_descriptors": valid_hypotheses,
            },
            "features": feature_metrics,
            "timings": {
                "masks_and_lift_s": round(mask_seconds, 3),
                **association_result["timings"],
                "semantic_features_s": round(feature_seconds, 3),
                "semantic_A_dense_visualization_s": round(semantic_a_seconds, 3),
                "semantic_C_hypavg_visualization_s": round(semantic_c_seconds, 3),
                "wall_s": round(time.perf_counter() - started, 3),
            },
            "artifacts": {
                "prediction": str(prediction_path),
                "visualization": str(ply_path),
                "semantic_A_dense_visualization": str(semantic_a_path),
                "semantic_C_hypavg_visualization": str(semantic_c_path),
            },
        }
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        logger.info(
            "Instance distillation complete: %d hypotheses over %d TSDF surface points",
            len(hypotheses),
            len(points),
        )
        return summary
