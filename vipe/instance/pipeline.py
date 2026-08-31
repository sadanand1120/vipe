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

from vipe.instance.association import associate, frame_coreset_poses
from vipe.instance.masks import generate_and_lift
from vipe.utils.io import ArtifactPath
from vipe.utils.logging import pbar


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


def _build_occupancy_cloud(
    frame_stream,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    voxel_size: float,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Backproject all valid sensor depths into the frontier's occupied-voxel cloud."""
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    batches = []
    for frame_index in pbar(range(len(poses)), desc="Stage 6 occupancy cloud"):
        c2w = poses[frame_index]
        depth = frame_stream._read_depth(frame_index, frame_stream.frame_size)
        y, x = np.where(
            np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
        )
        if not y.size:
            continue
        z = depth[y, x]
        camera = np.stack(((x - cx) / fx * z, (y - cy) / fy * z, z), axis=1).astype(
            np.float64
        )
        world = camera @ c2w[:3, :3].T + c2w[:3, 3]
        batches.append(np.unique(np.floor(world / voxel_size).astype(np.int64), axis=0))
    if not batches:
        raise ValueError("No valid sensor-depth points were available for instance distillation")
    voxels = np.unique(np.concatenate(batches), axis=0)
    return ((voxels + 0.5) * voxel_size).astype(np.float32)


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


def _palette(count: int) -> np.ndarray:
    colors = np.zeros((count, 3), np.uint8)
    hue = 0.0
    for index in range(count):
        hue = (hue + 0.61803398875) % 1.0
        rgb = colorsys.hsv_to_rgb(
            hue,
            0.55 + 0.35 * ((index % 3) / 2),
            0.75 + 0.25 * (index % 2),
        )
        colors[index] = tuple(int(channel * 255) for channel in rgb)
    return colors


def _write_instance_ply(path: Path, points: np.ndarray, hypotheses: list[np.ndarray]) -> None:
    """Write a smallest-hypothesis-wins visualization of the overlapping prediction."""
    owner = np.full(len(points), -1, np.int32)
    owner_size = np.full(len(points), np.iinfo(np.int64).max, np.int64)
    for hypothesis_id, voxels in enumerate(hypotheses):
        take = voxels[len(voxels) < owner_size[voxels]]
        owner[take] = hypothesis_id
        owner_size[take] = len(voxels)
    colors = np.tile(np.array([60, 60, 60], np.uint8), (len(points), 1))
    assigned = owner >= 0
    colors[assigned] = _palette(len(hypotheses))[owner[assigned]]

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
    records["instance"] = owner
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
        )

    def run(
        self,
        frame_stream,
        pose_matrices: np.ndarray,
        intrinsics: np.ndarray,
        artifact_path: ArtifactPath,
    ) -> dict:
        """Distill overlapping 3D instances and return the persisted scene summary."""
        started = time.perf_counter()
        poses = np.ascontiguousarray(pose_matrices, dtype=np.float32)
        intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(4)
        if poses.shape != (len(frame_stream), 4, 4):
            raise ValueError(
                f"Expected {(len(frame_stream), 4, 4)} c2w poses, received {poses.shape}"
            )

        cloud_config = self.config["cloud"]
        logger.info("Stage 6: building the 2 cm instance occupancy cloud and frame coreset")
        tick = time.perf_counter()
        points = _build_occupancy_cloud(
            frame_stream,
            poses,
            intrinsics,
            float(cloud_config["voxel_m"]),
            float(cloud_config["depth_min_m"]),
            float(cloud_config["depth_max_m"]),
        )
        cloud_seconds = time.perf_counter() - tick

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

        logger.info("Stages 7-8: generating, propagating, and lifting instance masks")
        tick = time.perf_counter()
        points_device = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        framed, mask_stats = generate_and_lift(
            points_device,
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

        logger.info("Stages 9-12: atomizing and selecting overlapping 3D hypotheses")
        association_result = associate(
            framed,
            points,
            self.config["association"],
            float(cloud_config["voxel_m"]),
        )
        hypotheses = association_result["hypotheses"]
        indices, offsets = _pack_hypotheses(hypotheses)
        prediction_path, summary_path, ply_path = self._artifact_paths(artifact_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            prediction_path,
            points=points,
            hypothesis_indices=indices,
            hypothesis_offsets=offsets,
            K=np.int32(self.config["association"]["membership_budget"]),
        )
        _write_instance_ply(ply_path, points, hypotheses)

        summary = {
            "schema_version": 1,
            "scene": artifact_path.artifact_name,
            "config": _plain(self.config),
            "frames": {"total": len(poses), "selected": len(frame_indices)},
            "counts": {
                "points": len(points),
                "lifted_frames": len(framed),
                "lifted_masks": sum(len(frame["masks"]) for frame in framed),
                **mask_stats,
                **association_result["counts"],
            },
            "timings": {
                "cloud_s": round(cloud_seconds, 3),
                "masks_and_lift_s": round(mask_seconds, 3),
                **association_result["timings"],
                "wall_s": round(time.perf_counter() - started, 3),
            },
            "artifacts": {
                "prediction": str(prediction_path),
                "visualization": str(ply_path),
            },
        }
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        logger.info(
            "Instance distillation complete: %d hypotheses over %d occupancy points",
            len(hypotheses),
            len(points),
        )
        return summary
