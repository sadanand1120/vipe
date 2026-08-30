import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCENE_FORMAT = "vipe_rgbd_v1"
INTRINSICS_FORMAT = "vipe_pinhole_intrinsics_v1"
FRAME_STEM_WIDTH = 6
DEFAULT_VIPE_FPS = 5.0


def frame_stem(index: int) -> str:
    return f"{index:0{FRAME_STEM_WIDTH}d}"


def long_side_size(width: int, height: int, target_long_side: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or target_long_side <= 0:
        raise ValueError("Image dimensions and target long side must be positive")
    scale = float(target_long_side) / float(max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_image(image: np.ndarray, size: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    if image.shape[1::-1] == size:
        return image
    if nearest:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_AREA if size[0] * size[1] < image.shape[1] * image.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def rescale_pinhole_matrix(
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    scaled = intrinsics.copy()
    scaled[0, [0, 2]] *= float(target_size[0]) / float(source_size[0])
    scaled[1, [1, 2]] *= float(target_size[1]) / float(source_size[1])
    return scaled


def subsampled_frame_indices(frame_count: int, source_fps: float, target_fps: float) -> list[int]:
    if frame_count < 1:
        raise ValueError("Frame count must be positive")
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("Source and target FPS must be positive")
    if target_fps > source_fps:
        raise ValueError(f"Target FPS ({target_fps:g}) cannot exceed source FPS ({source_fps:g})")

    output_count = math.floor((frame_count - 1) * target_fps / source_fps + 1e-9) + 1
    return [math.floor(index * source_fps / target_fps + 1e-9) for index in range(output_count)]


def subsampled_timestamp_indices(timestamps_ns: list[int], target_fps: float) -> list[int]:
    if not timestamps_ns:
        raise ValueError("Timestamp sequence must not be empty")
    if target_fps <= 0.0:
        raise ValueError("Target FPS must be positive")
    if any(a >= b for a, b in zip(timestamps_ns, timestamps_ns[1:])):
        raise ValueError("Timestamps must be strictly increasing")
    if len(timestamps_ns) == 1:
        return [0]

    source_fps = 1e9 / float(np.median(np.diff(np.asarray(timestamps_ns, dtype=np.float64))))
    if target_fps > source_fps * (1.0 + 1e-3):
        raise ValueError(f"Target FPS ({target_fps:g}) cannot exceed measured source FPS ({source_fps:.3f})")

    period_ns = 1e9 / target_fps
    selected = [0]
    target_ns = timestamps_ns[0] + period_ns
    source_idx = 1
    while target_ns <= timestamps_ns[-1] + 1e-6:
        while source_idx < len(timestamps_ns) and timestamps_ns[source_idx] < target_ns:
            source_idx += 1
        if source_idx >= len(timestamps_ns):
            break
        before_idx = source_idx - 1
        best_idx = min((before_idx, source_idx), key=lambda idx: abs(timestamps_ns[idx] - target_ns))
        if best_idx <= selected[-1]:
            best_idx = source_idx
        selected.append(best_idx)
        source_idx = best_idx + 1
        target_ns += period_ns
    return selected


def read_scene_metadata(scene_dir: str | Path) -> dict[str, Any]:
    path = Path(scene_dir) / "metadata.json"
    with open(path, "r") as f:
        metadata = json.load(f)
    if metadata.get("format") != SCENE_FORMAT:
        raise ValueError(f"{path} is not a {SCENE_FORMAT} scene")
    if not isinstance(metadata.get("fps"), (int, float)) or metadata["fps"] <= 0.0:
        raise ValueError(f"{path} has no valid canonical FPS")
    return metadata


def scene_frame_count(scene_dir: str | Path) -> int:
    metadata = read_scene_metadata(scene_dir)
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{Path(scene_dir) / 'metadata.json'} has no frame records")
    return len(frames)


def write_scene_metadata(
    scene_dir: str | Path,
    *,
    name: str,
    width: int,
    height: int,
    fps: float,
    frames: list[dict[str, Any]],
    source: dict[str, Any] | None = None,
) -> None:
    metadata = {
        "format": SCENE_FORMAT,
        "name": name,
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "frames": frames,
    }
    if source is not None:
        metadata["source"] = source

    path = Path(scene_dir) / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def read_pinhole_intrinsics(path: str | Path) -> dict[str, float | int]:
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)
    if data.get("format") != INTRINSICS_FORMAT:
        raise ValueError(f"{path} is not a {INTRINSICS_FORMAT} file")
    return {
        "width": int(data["width"]),
        "height": int(data["height"]),
        "fx": float(data["fx"]),
        "fy": float(data["fy"]),
        "cx": float(data["cx"]),
        "cy": float(data["cy"]),
    }


def write_pinhole_intrinsics(
    path: str | Path,
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    source: dict[str, Any] | None = None,
) -> None:
    data = {
        "format": INTRINSICS_FORMAT,
        "camera_model": "pinhole",
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }
    if source is not None:
        data["source"] = source

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def intrinsic_matrix(intrinsics: dict[str, float | int]) -> np.ndarray:
    return np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
