import argparse
import bisect
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tqdm import tqdm


DEFAULT_BAG_DIR = Path("/robodata/smodak/repos/vipe/data/rosbags/distilled_bag")
COLOR_TOPIC = "/rgb/image_raw"
DEPTH_TOPIC = "/depth_to_rgb/image_raw"
COLOR_INFO_TOPIC = "/rgb/camera_info"
POSE_TOPIC = "/lidar_odometry/pose"
BASE_FROM_CAMERA = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def resolve_bag_uri(path):
    if path.is_file():
        return path
    mcap_files = sorted(path.glob("*.mcap"))
    if len(mcap_files) == 1:
        return mcap_files[0]
    return path


def open_reader(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(resolve_bag_uri(bag_path)), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    return reader


def reader_message_count(reader):
    try:
        return int(reader.get_metadata().message_count)
    except RuntimeError:
        return None


def stamp_ns(msg, bag_stamp_ns):
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return int(bag_stamp_ns)
    ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return ns if ns > 0 else int(bag_stamp_ns)


def image_to_cv(msg, bridge, kind):
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
        return np.nan_to_num(image * 1000.0, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.uint16
        )
    return image


def quat_to_matrix(qx, qy, qz, qw):
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def odom_to_camera_pose(msg):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    world_from_base = np.eye(4, dtype=np.float64)
    world_from_base[:3, :3] = quat_to_matrix(q.x, q.y, q.z, q.w)
    world_from_base[:3, 3] = [p.x, p.y, p.z]
    base_from_camera = np.eye(4, dtype=np.float64)
    base_from_camera[:3, :3] = BASE_FROM_CAMERA
    return world_from_base @ base_from_camera


def write_pose(path, msg):
    np.savetxt(path, odom_to_camera_pose(msg), fmt="%.9f")


def prepare_dirs(bag_dir, overwrite, sync_only):
    raw_targets = [
        bag_dir / "raw" / "color",
        bag_dir / "raw" / "depth",
        bag_dir / "raw" / "pose",
    ]
    sync_targets = [
        bag_dir / "color",
        bag_dir / "depth",
        bag_dir / "pose",
    ]
    targets = sync_targets if sync_only else raw_targets + sync_targets
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output directories already exist; rerun with --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    for path in existing:
        shutil.rmtree(path)
    for path in targets:
        path.mkdir(parents=True, exist_ok=True)


def prepare_pose_dirs(bag_dir, overwrite):
    targets = [bag_dir / "raw" / "pose", bag_dir / "pose"]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Pose output directories already exist; rerun with --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    for path in existing:
        shutil.rmtree(path)
    for path in targets:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def camera_info_to_json(topic, msg, bag_stamp_ns):
    distortion_model = msg.distortion_model
    camera_model_name = f"pinhole_{distortion_model}" if distortion_model else "pinhole"
    scannet_intrinsic_matrix = [
        [float(msg.k[0]), 0.0, float(msg.k[2]), 0.0],
        [0.0, float(msg.k[4]), float(msg.k[5]), 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "topic": topic,
        "message_type": "sensor_msgs/msg/CameraInfo",
        "stamp_ns": stamp_ns(msg, bag_stamp_ns),
        "bag_stamp_ns": int(bag_stamp_ns),
        "frame_id": msg.header.frame_id,
        "width": int(msg.width),
        "height": int(msg.height),
        "camera_model_name": camera_model_name,
        "projection_model": "pinhole",
        "distortion_model": distortion_model,
        "scannet_intrinsic_matrix": scannet_intrinsic_matrix,
        "intrinsics": {
            "fx": float(msg.k[0]),
            "fy": float(msg.k[4]),
            "cx": float(msg.k[2]),
            "cy": float(msg.k[5]),
        },
        "distortion_coefficients": list(map(float, msg.d)),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": list(map(float, msg.k)),
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": list(map(float, msg.r)),
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": list(map(float, msg.p)),
        },
        "binning": {
            "x": int(msg.binning_x),
            "y": int(msg.binning_y),
        },
        "roi": {
            "x_offset": int(msg.roi.x_offset),
            "y_offset": int(msg.roi.y_offset),
            "height": int(msg.roi.height),
            "width": int(msg.roi.width),
            "do_rectify": bool(msg.roi.do_rectify),
        },
    }


def write_intrinsics_files(bag_dir, intrinsics):
    intrinsic_dir = bag_dir / "intrinsic"
    intrinsic_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        intrinsic_dir / "intrinsic_color.txt",
        np.asarray(intrinsics["scannet_intrinsic_matrix"], dtype=np.float64),
        fmt="%.9f",
    )
    write_json(intrinsic_dir / "intrinsic_color.json", intrinsics)
    for base in (bag_dir / "raw" / "color", bag_dir / "color"):
        base.mkdir(parents=True, exist_ok=True)
        write_json(base / "intrinsics.json", intrinsics)


def export_intrinsics(args):
    reader = open_reader(args.bag_dir)
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    if args.color_info_topic not in topic_types:
        raise KeyError(f"Missing required camera info topic in bag: {args.color_info_topic}")

    intrinsics = None
    while reader.has_next():
        topic, data, bag_stamp_ns = reader.read_next()
        if topic != args.color_info_topic:
            continue
        msg = deserialize_message(data, get_message(topic_types[topic]))
        intrinsics = camera_info_to_json(topic, msg, bag_stamp_ns)
        break

    if intrinsics is None:
        raise RuntimeError(f"No CameraInfo messages found for: {args.color_info_topic}")
    write_intrinsics_files(args.bag_dir, intrinsics)
    print(f"RGB intrinsics export: {args.color_info_topic}")


def export_raw(args):
    reader = open_reader(args.bag_dir)
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    wanted_topics = {
        args.color_topic: ("color", args.bag_dir / "raw" / "color", ".png"),
        args.depth_topic: ("depth", args.bag_dir / "raw" / "depth", ".png"),
        args.pose_topic: ("pose", args.bag_dir / "raw" / "pose", ".txt"),
    }
    for topic in wanted_topics:
        if topic not in topic_types:
            raise KeyError(f"Missing required topic in bag: {topic}")

    bridge = CvBridge()
    counts = {topic: 0 for topic in wanted_topics}
    metas = {
        "color": {"topic": args.color_topic, "items": []},
        "depth": {"topic": args.depth_topic, "items": []},
        "pose": {"topic": args.pose_topic, "items": []},
    }

    with tqdm(
        total=reader_message_count(reader), desc="Export raw bag", unit="msg"
    ) as progress:
        while reader.has_next():
            topic, data, bag_stamp_ns = reader.read_next()
            progress.update(1)
            if topic not in wanted_topics:
                continue
            msg = deserialize_message(data, get_message(topic_types[topic]))
            kind, out_dir, suffix = wanted_topics[topic]
            seq = counts[topic]
            file_name = f"{seq:05d}{suffix}"
            out_path = out_dir / file_name
            item = {
                "seq": seq,
                "file": file_name,
                "stamp_ns": stamp_ns(msg, bag_stamp_ns),
                "bag_stamp_ns": int(bag_stamp_ns),
            }

            if kind in {"color", "depth"}:
                image = image_to_cv(msg, bridge, kind)
                if not cv2.imwrite(str(out_path), image):
                    raise RuntimeError(f"Failed to write image: {out_path}")
                item.update(
                    {
                        "encoding": msg.encoding,
                        "width": int(msg.width),
                        "height": int(msg.height),
                    }
                )
            else:
                write_pose(out_path, msg)
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                item.update(
                    {
                        "frame_id": msg.header.frame_id,
                        "child_frame_id": msg.child_frame_id,
                        "position": [p.x, p.y, p.z],
                        "orientation_xyzw": [q.x, q.y, q.z, q.w],
                        "written_pose": "world_from_camera",
                        "source_pose": "world_from_base",
                        "base_from_camera_rotation": BASE_FROM_CAMERA.tolist(),
                    }
                )

            metas[kind]["items"].append(item)
            counts[topic] += 1
            if sum(counts.values()) % 100 == 0:
                progress.set_postfix(
                    color=counts[args.color_topic],
                    depth=counts[args.depth_topic],
                    pose=counts[args.pose_topic],
                )

    for kind, meta in metas.items():
        meta["count"] = len(meta["items"])
        write_json(args.bag_dir / "raw" / kind / "meta.json", meta)
    print(
        "Raw export: "
        f"color={len(metas['color']['items'])}, "
        f"depth={len(metas['depth']['items'])}, "
        f"pose={len(metas['pose']['items'])}"
    )


def export_raw_poses(args):
    reader = open_reader(args.bag_dir)
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    if args.pose_topic not in topic_types:
        raise KeyError(f"Missing required topic in bag: {args.pose_topic}")

    items = []
    with tqdm(
        total=reader_message_count(reader), desc="Export raw poses", unit="msg"
    ) as progress:
        while reader.has_next():
            topic, data, bag_stamp_ns = reader.read_next()
            progress.update(1)
            if topic != args.pose_topic:
                continue
            msg = deserialize_message(data, get_message(topic_types[topic]))
            seq = len(items)
            file_name = f"{seq:05d}.txt"
            write_pose(args.bag_dir / "raw" / "pose" / file_name, msg)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            items.append(
                {
                    "seq": seq,
                    "file": file_name,
                    "stamp_ns": stamp_ns(msg, bag_stamp_ns),
                    "bag_stamp_ns": int(bag_stamp_ns),
                    "frame_id": msg.header.frame_id,
                    "child_frame_id": msg.child_frame_id,
                    "position": [p.x, p.y, p.z],
                    "orientation_xyzw": [q.x, q.y, q.z, q.w],
                    "written_pose": "world_from_camera",
                    "source_pose": "world_from_base",
                    "base_from_camera_rotation": BASE_FROM_CAMERA.tolist(),
                }
            )
            if len(items) % 100 == 0:
                progress.set_postfix(pose=len(items))
    write_json(
        args.bag_dir / "raw" / "pose" / "meta.json",
        {"topic": args.pose_topic, "count": len(items), "items": items},
    )
    print(f"Raw pose export: pose={len(items)}")


def rewrite_synced_poses_from_meta(args):
    meta = json.loads((args.bag_dir / "sync_meta.json").read_text())
    pose_dir = args.bag_dir / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)
    for item in tqdm(meta["items"], desc="Rewrite synced poses", unit="pose"):
        shutil.copy2(
            args.bag_dir / "raw" / "pose" / item["raw_pose_file"],
            pose_dir / item["pose_file"],
        )
    meta["pose_transform"] = {
        "written_pose": "world_from_camera",
        "source_pose": "world_from_base",
        "base_from_camera_rotation": BASE_FROM_CAMERA.tolist(),
    }
    write_json(args.bag_dir / "sync_meta.json", meta)
    print(f"Synchronized pose rewrite: {len(meta['items'])} poses")


def load_meta(path):
    return json.loads(path.read_text())["items"]


def nearest(items, stamps, stamp_ns):
    idx = bisect.bisect_left(stamps, stamp_ns)
    candidates = []
    if idx < len(items):
        candidates.append(items[idx])
    if idx > 0:
        candidates.append(items[idx - 1])
    return min(candidates, key=lambda item: abs(item["stamp_ns"] - stamp_ns))


def sync_outputs(args):
    raw_dir = args.bag_dir / "raw"
    color = sorted(load_meta(raw_dir / "color" / "meta.json"), key=lambda x: x["stamp_ns"])
    depth = sorted(load_meta(raw_dir / "depth" / "meta.json"), key=lambda x: x["stamp_ns"])
    pose = sorted(load_meta(raw_dir / "pose" / "meta.json"), key=lambda x: x["stamp_ns"])
    color = color[args.skip_rgb_frames :]
    depth_stamps = [item["stamp_ns"] for item in depth]
    pose_stamps = [item["stamp_ns"] for item in pose]
    max_depth_dt_ns = int(args.max_depth_dt * 1_000_000_000)
    max_pose_dt_ns = int(args.max_pose_dt * 1_000_000_000)

    sync_items = []
    for color_item in tqdm(color, desc="Sync frames", unit="frame"):
        depth_item = nearest(depth, depth_stamps, color_item["stamp_ns"])
        pose_item = nearest(pose, pose_stamps, color_item["stamp_ns"])
        depth_dt_ns = abs(depth_item["stamp_ns"] - color_item["stamp_ns"])
        pose_dt_ns = abs(pose_item["stamp_ns"] - color_item["stamp_ns"])
        if depth_dt_ns > max_depth_dt_ns or pose_dt_ns > max_pose_dt_ns:
            continue

        seq = len(sync_items)
        stem = f"{seq:05d}"
        color_name = f"{stem}.png"
        depth_name = f"{stem}.png"
        pose_name = f"{stem}.txt"
        shutil.copy2(raw_dir / "color" / color_item["file"], args.bag_dir / "color" / color_name)
        shutil.copy2(raw_dir / "depth" / depth_item["file"], args.bag_dir / "depth" / depth_name)
        shutil.copy2(raw_dir / "pose" / pose_item["file"], args.bag_dir / "pose" / pose_name)
        sync_items.append(
            {
                "seq": seq,
                "color_file": color_name,
                "depth_file": depth_name,
                "pose_file": pose_name,
                "color_stamp_ns": color_item["stamp_ns"],
                "depth_stamp_ns": depth_item["stamp_ns"],
                "pose_stamp_ns": pose_item["stamp_ns"],
                "depth_dt_ns": depth_dt_ns,
                "pose_dt_ns": pose_dt_ns,
                "raw_color_file": color_item["file"],
                "raw_depth_file": depth_item["file"],
                "raw_pose_file": pose_item["file"],
            }
        )

    write_json(
        args.bag_dir / "sync_meta.json",
        {
            "color_topic": args.color_topic,
            "depth_topic": args.depth_topic,
            "pose_topic": args.pose_topic,
            "max_depth_dt_sec": args.max_depth_dt,
            "max_pose_dt_sec": args.max_pose_dt,
            "skip_rgb_frames": args.skip_rgb_frames,
            "count": len(sync_items),
            "items": sync_items,
        },
    )
    print(f"Synchronized export: {len(sync_items)} frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_dir", nargs="?", type=Path, default=DEFAULT_BAG_DIR)
    parser.add_argument("--color-topic", default=COLOR_TOPIC)
    parser.add_argument("--depth-topic", default=DEPTH_TOPIC)
    parser.add_argument("--color-info-topic", default=COLOR_INFO_TOPIC)
    parser.add_argument("--pose-topic", default=POSE_TOPIC)
    parser.add_argument("--max-depth-dt", type=float, default=0.05)
    parser.add_argument("--max-pose-dt", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--poses-only", action="store_true")
    parser.add_argument("--intrinsics-only", action="store_true")
    parser.add_argument("--skip-rgb-frames", type=int, default=0)
    args = parser.parse_args()

    args.bag_dir = args.bag_dir.resolve()
    if args.intrinsics_only:
        export_intrinsics(args)
        return
    if args.poses_only:
        prepare_pose_dirs(args.bag_dir, args.overwrite)
        export_raw_poses(args)
        rewrite_synced_poses_from_meta(args)
        export_intrinsics(args)
        return
    prepare_dirs(args.bag_dir, args.overwrite, args.sync_only)
    if not args.sync_only:
        export_raw(args)
    sync_outputs(args)
    export_intrinsics(args)


if __name__ == "__main__":
    main()
