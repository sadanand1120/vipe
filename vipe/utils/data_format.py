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
    frames: list[dict[str, Any]],
    source: dict[str, Any] | None = None,
) -> None:
    metadata = {
        "format": SCENE_FORMAT,
        "name": name,
        "width": int(width),
        "height": int(height),
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
