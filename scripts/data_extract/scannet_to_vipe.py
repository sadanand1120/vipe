import argparse
import shutil
import struct
import zlib

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from vipe.utils.data_format import frame_stem, write_pinhole_intrinsics, write_scene_metadata


COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


class RGBDFrame:
    def load(self, handle) -> None:
        self.camera_to_world = np.asarray(struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", handle.read(8))[0]
        self.color_data = handle.read(self.color_size_bytes)
        self.depth_data = handle.read(self.depth_size_bytes)

    def decompress_color(self, compression_type: str) -> np.ndarray:
        if compression_type in {"jpeg", "png"}:
            color = cv2.imdecode(np.frombuffer(self.color_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if color is None:
                raise ValueError("Failed to decode compressed color frame")
            return color
        raise ValueError(f"Unsupported color compression: {compression_type}")

    def decompress_depth(self, compression_type: str) -> bytes:
        if compression_type == "zlib_ushort":
            return zlib.decompress(self.depth_data)
        if compression_type == "raw_ushort":
            return self.depth_data
        raise ValueError(f"Unsupported depth compression: {compression_type}")


class SensorData:
    def __init__(self, filename: Path) -> None:
        self.load(filename)

    def load(self, filename: Path) -> None:
        with open(filename, "rb") as handle:
            version = struct.unpack("I", handle.read(4))[0]
            if version != 4:
                raise ValueError(f"Unsupported .sens version {version}, expected 4")

            strlen = struct.unpack("Q", handle.read(8))[0]
            self.sensor_name = handle.read(strlen).decode("utf-8", errors="ignore")
            self.intrinsic_color = np.asarray(struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", handle.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack("i", handle.read(4))[0]]
            self.color_width = struct.unpack("I", handle.read(4))[0]
            self.color_height = struct.unpack("I", handle.read(4))[0]
            self.depth_width = struct.unpack("I", handle.read(4))[0]
            self.depth_height = struct.unpack("I", handle.read(4))[0]
            self.depth_shift = struct.unpack("f", handle.read(4))[0]
            num_frames = struct.unpack("Q", handle.read(8))[0]

            self.frames = []
            for _ in tqdm(range(num_frames), desc=f"Load {filename.name}", unit="frame"):
                frame = RGBDFrame()
                frame.load(handle)
                self.frames.append(frame)


def ensure_free_space(target: Path, min_free_gb: float) -> None:
    free_gb = shutil.disk_usage(target).free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(f"Free space on {target} is {free_gb:.1f} GB, below {min_free_gb:.1f} GB")


def prepare_scene_dir(scene_dir: Path, output_scene_dir: Path) -> None:
    scene_dir = scene_dir.resolve()
    output_scene_dir = output_scene_dir.resolve()
    if scene_dir.is_relative_to(output_scene_dir) or output_scene_dir.is_relative_to(scene_dir):
        raise ValueError(f"Refusing unsafe output path {output_scene_dir} for source scene {scene_dir}")
    if output_scene_dir.exists():
        shutil.rmtree(output_scene_dir)
    for subdir in ("color", "depth", "pose", "intrinsic"):
        (output_scene_dir / subdir).mkdir(parents=True, exist_ok=True)


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    np.savetxt(path, matrix, fmt="%.9f")


def decode_scene(scene_dir: Path, output_scene_dir: Path, frame_skip: int, source_fps: float, min_free_gb: float) -> None:
    sens_files = list(scene_dir.glob("*.sens"))
    if len(sens_files) != 1:
        raise ValueError(f"Expected exactly one .sens file in {scene_dir}, found {len(sens_files)}")

    prepare_scene_dir(scene_dir, output_scene_dir)
    ensure_free_space(output_scene_dir, min_free_gb)

    sensor_data = SensorData(sens_files[0])
    if not np.isclose(sensor_data.depth_shift, 1000.0):
        raise ValueError(f"Unexpected ScanNet depth_shift={sensor_data.depth_shift}; expected 1000.0")

    k = sensor_data.intrinsic_color[:3, :3]
    write_pinhole_intrinsics(
        output_scene_dir / "intrinsic" / "intrinsic_color.json",
        width=sensor_data.color_width,
        height=sensor_data.color_height,
        fx=k[0, 0],
        fy=k[1, 1],
        cx=k[0, 2],
        cy=k[1, 2],
        source={"dataset": "ScanNet", "file": str(sens_files[0]), "matrix": "intrinsic_color"},
    )

    frame_indices = list(range(0, len(sensor_data.frames), frame_skip))
    frames = []
    print(f"Decoding {scene_dir.name}: {len(frame_indices)} frames @ {source_fps / frame_skip:.2f} FPS")

    for export_idx, source_idx in enumerate(tqdm(frame_indices, desc=scene_dir.name, unit="frame")):
        if export_idx % 25 == 0:
            ensure_free_space(output_scene_dir, min_free_gb)

        frame = sensor_data.frames[source_idx]
        stem = frame_stem(export_idx)
        color = frame.decompress_color(sensor_data.color_compression_type)
        depth = np.frombuffer(frame.decompress_depth(sensor_data.depth_compression_type), dtype=np.uint16).reshape(
            sensor_data.depth_height, sensor_data.depth_width
        )
        if depth.shape[:2] != color.shape[:2]:
            depth = cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

        color_file = output_scene_dir / "color" / f"{stem}.png"
        depth_file = output_scene_dir / "depth" / f"{stem}.png"
        pose_file = output_scene_dir / "pose" / f"{stem}.txt"
        if not cv2.imwrite(str(color_file), color):
            raise RuntimeError(f"Failed to write color frame: {color_file}")
        if not cv2.imwrite(str(depth_file), depth):
            raise RuntimeError(f"Failed to write depth frame: {depth_file}")
        save_matrix(frame.camera_to_world, pose_file)
        frames.append(
            {
                "seq": export_idx,
                "stem": stem,
                "color_file": f"color/{stem}.png",
                "depth_file": f"depth/{stem}.png",
                "pose_file": f"pose/{stem}.txt",
                "source_frame_id": source_idx,
                "source_color_timestamp": int(frame.timestamp_color),
                "source_depth_timestamp": int(frame.timestamp_depth),
            }
        )

    write_scene_metadata(
        output_scene_dir,
        name=scene_dir.name,
        fps=source_fps / frame_skip,
        width=sensor_data.color_width,
        height=sensor_data.color_height,
        frames=frames,
        source={
            "dataset": "ScanNet",
            "scene": scene_dir.name,
            "sens_file": str(sens_files[0]),
            "source_fps": round(float(source_fps), 2),
            "frame_skip": frame_skip,
            "depth_unit": "millimeter",
        },
    )
    ensure_free_space(output_scene_dir, min_free_gb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ScanNet .sens files into canonical ViPE RGB-D scenes.")
    parser.add_argument("--scans-root", required=True, type=Path, help="Directory containing ScanNet scene folders.")
    parser.add_argument("--output-root", required=True, type=Path, help="Directory to write canonical scene folders into.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene names to extract. Defaults to all scenes.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Write every Nth source frame.")
    parser.add_argument("--fps", type=float, default=30.0, help="Source ScanNet frame rate before frame skipping.")
    parser.add_argument("--min-free-gb", type=float, default=10.0, help="Abort if free disk drops below this amount.")
    args = parser.parse_args()

    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be >= 1")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.scenes:
        scenes = [args.scans_root / scene for scene in args.scenes]
    else:
        scenes = sorted(path for path in args.scans_root.iterdir() if path.is_dir())

    for scene_dir in scenes:
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        decode_scene(scene_dir, args.output_root / scene_dir.name, args.frame_skip, args.fps, args.min_free_gb)


if __name__ == "__main__":
    main()
