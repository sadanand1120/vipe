import argparse
import io
import json
import shutil
import zipfile

from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from tqdm import tqdm

from vipe.utils.data_format import (
    DEFAULT_VIPE_FPS,
    frame_stem,
    long_side_size,
    rescale_pinhole_matrix,
    resize_image,
    subsampled_timestamp_indices,
    write_pinhole_intrinsics,
    write_scene_metadata,
)


def quaternion_wxyz_to_matrix(quaternion: list[float]) -> np.ndarray:
    normalized = np.asarray(quaternion, dtype=np.float64)
    normalized /= np.linalg.norm(normalized)
    w, x, y, z = normalized
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def usd_pose_to_opencv_c2w(position: list[float], quaternion: list[float]) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = quaternion_wxyz_to_matrix(quaternion)
    c2w[:3, 1:3] *= -1.0  # USD camera +Y up/-Z forward -> OpenCV +Y down/+Z forward.
    c2w[:3, 3] = np.asarray(position, dtype=np.float64)
    return c2w


def frame_members(archive: zipfile.ZipFile, root: str, kind: str, suffix: str) -> dict[int, str]:
    prefix = f"{root}/cam/{kind}/"
    return {
        int(PurePosixPath(member.filename).stem): member.filename
        for member in archive.infolist()
        if not member.is_dir() and member.filename.startswith(prefix) and member.filename.endswith(suffix)
    }


def prepare_output(output_dir: Path, archive_path: Path) -> None:
    output_dir = output_dir.resolve()
    archive_path = archive_path.resolve()
    if output_dir == archive_path.parent or archive_path.is_relative_to(output_dir):
        raise ValueError(f"Refusing unsafe output path: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for subdir in ("color", "depth", "pose", "intrinsic"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def depth_to_mm(depth: np.ndarray) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}")
    depth_mm = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= np.iinfo(np.uint16).max / 1000.0)
    depth_mm[valid] = np.rint(depth[valid] * 1000.0).astype(np.uint16)
    return depth_mm


def extract_scene(archive_path: Path, output_dir: Path, vipe_res: int, vipe_fps: float) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        roots = {member.filename.split("/", 1)[0] for member in archive.infolist() if "/cam/" in member.filename}
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one IsaacSim scene root, found: {sorted(roots)}")
        root = roots.pop()

        intrinsics = json.loads(archive.read(f"{root}/cam/intrinsics.json"))
        if intrinsics.get("projection") != "pinhole" or intrinsics.get("distortion"):
            raise ValueError("IsaacSim extractor requires an undistorted pinhole RGB-D camera")

        pose_records = [
            json.loads(line) for line in archive.read(f"{root}/cam/poses.jsonl").decode("utf-8").splitlines()
        ]
        pose_by_id = {int(record["frame_id"]): record for record in pose_records}
        rgb_by_id = frame_members(archive, root, "rgb", ".png")
        depth_by_id = frame_members(archive, root, "depth", ".npy")
        frame_ids = sorted(set(pose_by_id) & set(rgb_by_id) & set(depth_by_id))
        if not frame_ids or set(frame_ids) != set(pose_by_id) or set(frame_ids) != set(rgb_by_id) or set(frame_ids) != set(depth_by_id):
            raise ValueError("IsaacSim RGB, depth, and pose frame ids do not match")

        timestamps_ns = [int(pose_by_id[frame_id]["phys_t_ns"]) for frame_id in frame_ids]
        selected_indices = subsampled_timestamp_indices(timestamps_ns, vipe_fps)
        selected_ids = [frame_ids[index] for index in selected_indices]

        source_size = (int(intrinsics["width"]), int(intrinsics["height"]))
        output_size = long_side_size(*source_size, vipe_res)
        source_k = np.array(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        output_k = rescale_pinhole_matrix(source_k, source_size, output_size)

        prepare_output(output_dir, archive_path)
        write_pinhole_intrinsics(
            output_dir / "intrinsic" / "intrinsic_color.json",
            width=output_size[0],
            height=output_size[1],
            fx=output_k[0, 0],
            fy=output_k[1, 1],
            cx=output_k[0, 2],
            cy=output_k[1, 2],
            source={"dataset": "IsaacSim", "scene": root, "file": str(archive_path)},
        )

        frames = []
        for output_index, frame_id in enumerate(tqdm(selected_ids, desc=f"Extract {root}", unit="frame")):
            color = cv2.imdecode(np.frombuffer(archive.read(rgb_by_id[frame_id]), dtype=np.uint8), cv2.IMREAD_COLOR)
            depth = np.load(io.BytesIO(archive.read(depth_by_id[frame_id])), allow_pickle=False)
            if color is None or color.shape[1::-1] != source_size or depth.shape[::-1] != source_size:
                raise ValueError(f"Unexpected RGB-D dimensions for IsaacSim frame {frame_id}")

            color = resize_image(color, output_size)
            depth_mm = resize_image(depth_to_mm(depth), output_size, nearest=True)
            pose = pose_by_id[frame_id]
            c2w = usd_pose_to_opencv_c2w(pose["position"], pose["quaternion_wxyz"])
            stem = frame_stem(output_index)
            if not cv2.imwrite(str(output_dir / "color" / f"{stem}.png"), color):
                raise RuntimeError(f"Failed to write RGB frame {frame_id}")
            if not cv2.imwrite(str(output_dir / "depth" / f"{stem}.png"), depth_mm):
                raise RuntimeError(f"Failed to write depth frame {frame_id}")
            np.savetxt(output_dir / "pose" / f"{stem}.txt", c2w, fmt="%.9f")
            frames.append({"source_frame_id": frame_id, "source_timestamp_ns": int(pose["phys_t_ns"])})

        source_fps = 1e9 / float(np.median(np.diff(np.asarray(timestamps_ns, dtype=np.float64))))
        write_scene_metadata(
            output_dir,
            name=root,
            width=output_size[0],
            height=output_size[1],
            fps=vipe_fps,
            frames=frames,
            source={
                "dataset": "IsaacSim",
                "scene": root,
                "archive": str(archive_path),
                "source_fps": source_fps,
                "vipe_res": vipe_res,
                "depth_source": "camera_z_depth",
                "depth_unit_original": "float32_meter",
                "depth_unit_exported": "millimeter",
                "pose_conversion": "USD/OpenGL camera axes to OpenCV camera axes",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract an IsaacSim RGB-D ZIP into one canonical ViPE scene.")
    parser.add_argument("archive", type=Path, help="IsaacSim scene ZIP")
    parser.add_argument("--output-dir", required=True, type=Path, help="Canonical ViPE scene directory")
    parser.add_argument("--vipe-res", type=int, default=1280, help="Canonical output image long side (default: 1280)")
    parser.add_argument("--vipe-fps", type=float, default=DEFAULT_VIPE_FPS, help="Canonical output FPS (default: 5)")
    args = parser.parse_args()
    if args.vipe_res < 1 or args.vipe_fps <= 0.0:
        raise ValueError("--vipe-res and --vipe-fps must be positive")
    extract_scene(args.archive, args.output_dir, args.vipe_res, args.vipe_fps)


if __name__ == "__main__":
    main()
