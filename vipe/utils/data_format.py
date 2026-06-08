import json
from pathlib import Path
from typing import Any

import numpy as np


SCENE_FORMAT = "vipe_rgbd_v1"
INTRINSICS_FORMAT = "vipe_pinhole_intrinsics_v1"
FRAME_STEM_WIDTH = 6


def frame_stem(index: int) -> str:
    return f"{index:0{FRAME_STEM_WIDTH}d}"


def read_scene_metadata(scene_dir: str | Path) -> dict[str, Any]:
    path = Path(scene_dir) / "metadata.json"
    with open(path, "r") as f:
        metadata = json.load(f)
    if metadata.get("format") != SCENE_FORMAT:
        raise ValueError(f"{path} is not a {SCENE_FORMAT} scene")
    return metadata


def read_scene_frames(scene_dir: str | Path, *, require_pose: bool = False) -> list[dict[str, Any]]:
    metadata = read_scene_metadata(scene_dir)
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{Path(scene_dir) / 'metadata.json'} has no frame records")

    required = ("color_file", "depth_file")
    if require_pose:
        required += ("pose_file",)
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Frame record {idx} is not an object")
        for key in required:
            if not isinstance(frame.get(key), str) or not frame[key]:
                raise ValueError(f"Frame record {idx} is missing required '{key}'")
    return frames


def write_scene_metadata(
    scene_dir: str | Path,
    *,
    name: str,
    width: int,
    height: int,
    frames: list[dict[str, Any]],
    source: dict[str, Any] | None = None,
) -> None:
    metadata = {
        "format": SCENE_FORMAT,
        "name": name,
        "width": int(width),
        "height": int(height),
        "color": {"dir": "color", "encoding": "rgb8_png"},
        "depth": {"dir": "depth", "encoding": "uint16_png", "unit": "millimeter"},
        "intrinsics": "intrinsic/intrinsic_color.json",
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
