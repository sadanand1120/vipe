import argparse
import bisect
import json
import shutil

from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tqdm import tqdm

from vipe.utils.data_format import (
    frame_stem,
    write_pinhole_intrinsics,
    write_scene_metadata,
)


COLOR_TOPIC = "/rgb/image_raw"
DEPTH_TOPIC = "/depth_to_rgb/image_raw"
COLOR_INFO_TOPIC = "/rgb/camera_info"


def fps_from_timestamps_ns(timestamps_ns: list[int]) -> float:
    if len(timestamps_ns) < 2:
        return 0.0
    deltas = np.diff(np.asarray(timestamps_ns, dtype=np.float64)) / 1e9
    valid = deltas[deltas > 0]
    if valid.size == 0:
        return 0.0
    return float(1.0 / np.median(valid))


def open_reader(bag_path: Path) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return reader


def reader_message_count(reader) -> int | None:
    try:
        return int(reader.get_metadata().message_count)
    except RuntimeError:
        return None


def stamp_ns(msg, bag_stamp_ns: int) -> int:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return int(bag_stamp_ns)
    ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return ns if ns > 0 else int(bag_stamp_ns)


def image_to_cv(msg, bridge, kind: str) -> np.ndarray:
    image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    encoding = msg.encoding.lower()
    if kind == "color":
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.nan_to_num(image * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
    return image.astype(np.uint16, copy=False)


def prepare_dirs(output_dir: Path, bag_path: Path) -> None:
    output_dir = output_dir.resolve()
    bag_path = bag_path.resolve()
    if bag_path.is_relative_to(output_dir):
        raise ValueError(f"Refusing to overwrite output dir {output_dir} because it contains input bag {bag_path}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for path in [
        output_dir / "raw" / "color",
        output_dir / "raw" / "depth",
        output_dir / "color",
        output_dir / "depth",
        output_dir / "intrinsic",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_camera_info(args: argparse.Namespace) -> dict:
    reader = open_reader(args.bag_path)
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    if args.color_info_topic not in topic_types:
        raise KeyError(f"Missing required camera info topic in bag: {args.color_info_topic}")

    with tqdm(total=reader_message_count(reader), desc="Read camera info", unit="msg") as progress:
        while reader.has_next():
            topic, data, bag_stamp_ns = reader.read_next()
            progress.update(1)
            if topic != args.color_info_topic:
                continue
            msg = deserialize_message(data, get_message(topic_types[topic]))
            k = np.asarray(msg.k, dtype=np.float32).reshape(3, 3)
            p = np.asarray(msg.p, dtype=np.float32).reshape(3, 4)
            output_k = p[:3, :3] if np.any(p) else k
            d = np.asarray(msg.d, dtype=np.float32).reshape(-1)
            return {
                "topic": topic,
                "stamp_ns": stamp_ns(msg, bag_stamp_ns),
                "bag_stamp_ns": int(bag_stamp_ns),
                "frame_id": msg.header.frame_id,
                "width": int(msg.width),
                "height": int(msg.height),
                "distortion_model": msg.distortion_model,
                "input_k": k,
                "output_k": output_k,
                "distortion_coefficients": d,
                "has_distortion": d.size > 0 and not np.allclose(d, 0.0),
            }
    raise RuntimeError(f"No CameraInfo messages found for: {args.color_info_topic}")


def build_rectifier(camera_info: dict) -> tuple[np.ndarray, np.ndarray] | None:
    if not camera_info["has_distortion"]:
        return None
    if camera_info["distortion_model"] not in {"plumb_bob", "rational_polynomial"}:
        raise ValueError(f"Unsupported distortion model: {camera_info['distortion_model']}")
    return cv2.initUndistortRectifyMap(
        camera_info["input_k"],
        camera_info["distortion_coefficients"],
        np.eye(3, dtype=np.float32),
        camera_info["output_k"],
        (camera_info["width"], camera_info["height"]),
        cv2.CV_32FC1,
    )


def write_intrinsics(output_dir: Path, camera_info: dict) -> None:
    k = camera_info["output_k"]
    write_pinhole_intrinsics(
        output_dir / "intrinsic" / "intrinsic_color.json",
        width=camera_info["width"],
        height=camera_info["height"],
        fx=k[0, 0],
        fy=k[1, 1],
        cx=k[0, 2],
        cy=k[1, 2],
        source={
            "type": "sensor_msgs/msg/CameraInfo",
            "topic": camera_info["topic"],
            "stamp_ns": camera_info["stamp_ns"],
            "frame_id": camera_info["frame_id"],
            "rectified_from_distortion": bool(camera_info["has_distortion"]),
        },
    )


def export_raw(args: argparse.Namespace) -> dict[str, float]:
    reader = open_reader(args.bag_path)
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    wanted_topics = {
        args.color_topic: ("color", args.output_dir / "raw" / "color", ".png"),
        args.depth_topic: ("depth", args.output_dir / "raw" / "depth", ".png"),
    }
    for topic in wanted_topics:
        if topic not in topic_types:
            raise KeyError(f"Missing required topic in bag: {topic}")

    bridge = CvBridge()
    counts = {topic: 0 for topic in wanted_topics}
    metas = {
        "color": {"topic": args.color_topic, "items": []},
        "depth": {"topic": args.depth_topic, "items": []},
    }

    with tqdm(total=reader_message_count(reader), desc="Export raw bag", unit="msg") as progress:
        while reader.has_next():
            topic, data, bag_stamp_ns = reader.read_next()
            progress.update(1)
            if topic not in wanted_topics:
                continue

            msg = deserialize_message(data, get_message(topic_types[topic]))
            kind, out_dir, suffix = wanted_topics[topic]
            seq = counts[topic]
            file_name = f"{frame_stem(seq)}{suffix}"
            out_path = out_dir / file_name
            image = image_to_cv(msg, bridge, kind)
            if not cv2.imwrite(str(out_path), image):
                raise RuntimeError(f"Failed to write image: {out_path}")

            metas[kind]["items"].append(
                {
                    "seq": seq,
                    "file": file_name,
                    "stamp_ns": stamp_ns(msg, bag_stamp_ns),
                    "bag_stamp_ns": int(bag_stamp_ns),
                    "encoding": msg.encoding,
                    "width": int(msg.width),
                    "height": int(msg.height),
                }
            )
            counts[topic] += 1
            if sum(counts.values()) % 100 == 0:
                progress.set_postfix(color=counts[args.color_topic], depth=counts[args.depth_topic])

    for kind, meta in metas.items():
        meta["count"] = len(meta["items"])
        write_json(args.output_dir / "raw" / kind / "meta.json", meta)

    print(f"Raw export: color={len(metas['color']['items'])}, depth={len(metas['depth']['items'])}")
    return {
        "raw_color_fps": round(fps_from_timestamps_ns([item["stamp_ns"] for item in metas["color"]["items"]]), 2),
        "raw_depth_fps": round(fps_from_timestamps_ns([item["stamp_ns"] for item in metas["depth"]["items"]]), 2),
    }


def load_meta(path: Path) -> list[dict]:
    return json.loads(path.read_text())["items"]


def nearest(items: list[dict], stamps: list[int], stamp_ns_value: int) -> dict:
    idx = bisect.bisect_left(stamps, stamp_ns_value)
    candidates = []
    if idx < len(items):
        candidates.append(items[idx])
    if idx > 0:
        candidates.append(items[idx - 1])
    return min(candidates, key=lambda item: abs(item["stamp_ns"] - stamp_ns_value))


def write_synced_image(src: Path, dst: Path, kind: str, camera_info: dict, rectifier) -> None:
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read raw image: {src}")

    target_size = (camera_info["width"], camera_info["height"])
    if kind == "color" and image.shape[:2] != (target_size[1], target_size[0]):
        raise ValueError(
            f"Color frame size {image.shape[1]}x{image.shape[0]} does not match "
            f"CameraInfo {target_size[0]}x{target_size[1]} for {src}"
        )
    if rectifier is not None and image.shape[:2] != (target_size[1], target_size[0]):
        raise ValueError(
            f"Cannot rectify {kind} frame {src}: size {image.shape[1]}x{image.shape[0]} "
            f"does not match CameraInfo {target_size[0]}x{target_size[1]}"
        )

    if rectifier is not None:
        interpolation = cv2.INTER_LINEAR if kind == "color" else cv2.INTER_NEAREST
        image = cv2.remap(image, rectifier[0], rectifier[1], interpolation=interpolation)
    elif image.shape[:2] != (target_size[1], target_size[0]):
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

    if not cv2.imwrite(str(dst), image):
        raise RuntimeError(f"Failed to write synced image: {dst}")


def sync_outputs(args: argparse.Namespace, camera_info: dict) -> tuple[float, list[dict]]:
    raw_dir = args.output_dir / "raw"
    color = sorted(load_meta(raw_dir / "color" / "meta.json"), key=lambda x: x["stamp_ns"])
    depth = sorted(load_meta(raw_dir / "depth" / "meta.json"), key=lambda x: x["stamp_ns"])
    if len(color) == 0:
        raise RuntimeError("No raw color frames were exported")
    if len(depth) == 0:
        raise RuntimeError("No raw depth frames were exported")

    depth_stamps = [item["stamp_ns"] for item in depth]
    max_depth_dt_ns = int(args.max_depth_dt * 1_000_000_000)
    rectifier = build_rectifier(camera_info)

    sync_items = []
    frames = []
    for color_item in tqdm(color, desc="Sync frames", unit="frame"):
        depth_item = nearest(depth, depth_stamps, color_item["stamp_ns"])
        depth_dt_ns = abs(depth_item["stamp_ns"] - color_item["stamp_ns"])
        if depth_dt_ns > max_depth_dt_ns:
            continue

        seq = len(sync_items)
        stem = frame_stem(seq)
        color_name = f"{stem}.png"
        depth_name = f"{stem}.png"
        write_synced_image(
            raw_dir / "color" / color_item["file"],
            args.output_dir / "color" / color_name,
            "color",
            camera_info,
            rectifier,
        )
        write_synced_image(
            raw_dir / "depth" / depth_item["file"],
            args.output_dir / "depth" / depth_name,
            "depth",
            camera_info,
            rectifier,
        )

        item = {
            "seq": seq,
            "stem": stem,
            "color_file": f"color/{color_name}",
            "depth_file": f"depth/{depth_name}",
            "color_stamp_ns": color_item["stamp_ns"],
            "depth_stamp_ns": depth_item["stamp_ns"],
            "depth_dt_ns": depth_dt_ns,
            "raw_color_file": color_item["file"],
            "raw_depth_file": depth_item["file"],
        }
        sync_items.append(item)
        frames.append(
            {
                "color_stamp_ns": color_item["stamp_ns"],
                "depth_stamp_ns": depth_item["stamp_ns"],
                "depth_dt_ns": depth_dt_ns,
                "raw_color_file": color_item["file"],
                "raw_depth_file": depth_item["file"],
            }
        )

    write_json(
        args.output_dir / "sync_meta.json",
        {
            "bag_path": str(args.bag_path),
            "color_topic": args.color_topic,
            "depth_topic": args.depth_topic,
            "max_depth_dt_sec": args.max_depth_dt,
            "count": len(sync_items),
            "items": sync_items,
        },
    )
    write_scene_metadata(
        args.output_dir,
        name=args.output_dir.name,
        width=camera_info["width"],
        height=camera_info["height"],
        frames=frames,
        source={
            "type": "rosbag2_mcap",
            "bag_path": str(args.bag_path),
            "color_topic": args.color_topic,
            "depth_topic": args.depth_topic,
            "color_info_topic": args.color_info_topic,
            "depth_unit": "millimeter",
        },
    )
    print(f"Synchronized export: {len(sync_items)} frames")
    synced_fps = round(fps_from_timestamps_ns([item["color_stamp_ns"] for item in sync_items]), 2)
    return synced_fps, sync_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract an MCAP rosbag into a canonical ViPE RGB-D scene.")
    parser.add_argument("bag_path", type=Path, help="Required path to the input .mcap bag file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Required canonical scene output directory.")
    parser.add_argument("--color-topic", default=COLOR_TOPIC)
    parser.add_argument("--depth-topic", default=DEPTH_TOPIC)
    parser.add_argument("--color-info-topic", default=COLOR_INFO_TOPIC)
    parser.add_argument("--max-depth-dt", type=float, default=0.05)
    args = parser.parse_args()

    args.bag_path = args.bag_path.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.bag_path.is_file():
        raise FileNotFoundError(f"Input bag file not found: {args.bag_path}")
    if args.bag_path.suffix.lower() != ".mcap":
        raise ValueError(f"Expected an .mcap bag file, got: {args.bag_path}")
    if args.max_depth_dt < 0.0:
        raise ValueError("--max-depth-dt must be non-negative")

    prepare_dirs(args.output_dir, args.bag_path)
    camera_info = read_camera_info(args)
    raw_fps = export_raw(args)
    synced_fps, _ = sync_outputs(args, camera_info)
    write_intrinsics(args.output_dir, camera_info)
    print(
        "FPS summary: "
        f"raw_color={raw_fps['raw_color_fps']:.2f}, "
        f"raw_depth={raw_fps['raw_depth_fps']:.2f}, "
        f"synced={synced_fps:.2f}"
    )


if __name__ == "__main__":
    main()
