import argparse
import bisect
import shutil
import tempfile
import zipfile

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mcap.exceptions import EndOfFile
from mcap.records import Channel, Message, Schema
from mcap.stream_reader import StreamReader
from mcap_ros2.decoder import DecoderFactory
from tqdm import tqdm

from vipe.utils.data_format import fps_from_timestamps_ns, frame_stem, write_pinhole_intrinsics, write_scene_metadata


COLOR_TOPIC = "/camera/color/image/compressed"
DEPTH_TOPIC = "/camera/depth/image_rect"
COLOR_INFO_TOPIC = "/camera/color/camera_info"
DEPTH_INFO_TOPIC = "/camera/depth/camera_info"


@dataclass(frozen=True, slots=True)
class CameraInfo:
    topic: str
    stamp_ns: int
    frame_id: str
    width: int
    height: int
    k: np.ndarray
    distortion_model: str
    distortion_coefficients: list[float]


def prepare_output_dir(output_dir: Path, input_path: Path) -> None:
    output_dir = output_dir.resolve()
    input_path = input_path.resolve()
    if input_path.is_relative_to(output_dir):
        raise ValueError(f"Refusing to overwrite output dir {output_dir} because it contains input {input_path}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for subdir in ("color", "depth", "intrinsic"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def extract_zip(zip_path: Path, temp_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        members = []
        for member in archive.infolist():
            member_path = Path(member.filename)
            if not member.filename or member_path.parts[0] == "__MACOSX":
                continue
            members.append(member)

        total_bytes = sum(member.file_size for member in members if not member.is_dir())
        with tqdm(total=total_bytes, desc=f"Unzip {zip_path.name}", unit="B", unit_scale=True) as progress:
            for member in members:
                member_path = Path(member.filename)
                target = (temp_dir / member.filename).resolve()
                if not target.is_relative_to(temp_dir.resolve()):
                    raise ValueError(f"Unsafe zip member path: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                        progress.update(len(chunk))

    mcap_files = sorted(temp_dir.glob("*/recording.mcap"))
    if len(mcap_files) != 1:
        raise ValueError(f"Expected exactly one recording.mcap inside {zip_path}, found {len(mcap_files)}")
    return mcap_files[0]


def resolve_mcap_path(input_path: Path, temp_dir: Path) -> Path:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        return extract_zip(input_path, temp_dir)
    if input_path.is_file() and input_path.name == "recording.mcap":
        return input_path
    if input_path.is_dir():
        mcap_path = input_path / "recording.mcap"
        if mcap_path.is_file():
            return mcap_path
    raise ValueError(f"Expected a DepthCaptureLab .zip, recording.mcap, or session dir, got: {input_path}")


def iter_decoded_messages(mcap_path: Path, *, progress_desc: str | None = None):
    schemas: dict[int, Schema] = {}
    channels: dict[int, Channel] = {}
    decoders = {}
    decoder_factory = DecoderFactory()
    progress = (
        tqdm(total=mcap_path.stat().st_size, desc=progress_desc, unit="B", unit_scale=True)
        if progress_desc is not None
        else None
    )

    try:
        with open(mcap_path, "rb") as handle:
            reader = StreamReader(handle)
            last_pos = handle.tell()
            for record in reader.records:
                if progress is not None:
                    pos = handle.tell()
                    progress.update(max(0, pos - last_pos))
                    last_pos = pos
                if isinstance(record, Schema):
                    schemas[record.id] = record
                    continue
                if isinstance(record, Channel):
                    channels[record.id] = record
                    continue
                if not isinstance(record, Message):
                    continue

                channel = channels[record.channel_id]
                schema = schemas[channel.schema_id]
                decoder = decoders.get(record.channel_id)
                if decoder is None:
                    decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
                    decoders[record.channel_id] = decoder
                yield schema, channel, record, decoder(record.data)
    except EndOfFile:
        return
    finally:
        if progress is not None:
            progress.close()


def stamp_ns(msg, record: Message) -> int:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return int(record.log_time)
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else int(record.log_time)


def camera_info_from_msg(topic: str, msg, record: Message) -> CameraInfo:
    k = np.asarray(msg.p, dtype=np.float32).reshape(3, 4)[:3, :3]
    if not np.any(k):
        k = np.asarray(msg.k, dtype=np.float32).reshape(3, 3)
    return CameraInfo(
        topic=topic,
        stamp_ns=stamp_ns(msg, record),
        frame_id=msg.header.frame_id,
        width=int(msg.width),
        height=int(msg.height),
        k=k,
        distortion_model=str(msg.distortion_model),
        distortion_coefficients=[float(x) for x in msg.d],
    )


def inspect_mcap(mcap_path: Path, max_depth_dt: float) -> tuple[CameraInfo, CameraInfo, list[dict]]:
    color_info = None
    depth_info = None
    color_stamps = []
    depth_stamps = []

    for _, channel, record, msg in iter_decoded_messages(mcap_path, progress_desc=f"Scan {mcap_path.parent.name} MCAP"):
        if channel.topic == COLOR_INFO_TOPIC:
            color_info = camera_info_from_msg(channel.topic, msg, record)
        elif channel.topic == DEPTH_INFO_TOPIC:
            depth_info = camera_info_from_msg(channel.topic, msg, record)
        elif channel.topic == COLOR_TOPIC:
            color_stamps.append(int(record.log_time))
        elif channel.topic == DEPTH_TOPIC:
            depth_stamps.append(int(record.log_time))

    if color_info is None:
        raise KeyError(f"Missing required topic: {COLOR_INFO_TOPIC}")
    if depth_info is None:
        raise KeyError(f"Missing required topic: {DEPTH_INFO_TOPIC}")
    if color_info.distortion_coefficients or color_info.distortion_model:
        raise ValueError("DepthCaptureLab color CameraInfo is expected to be rectified pinhole")
    if depth_info.distortion_coefficients or depth_info.distortion_model:
        raise ValueError("DepthCaptureLab depth CameraInfo is expected to be rectified pinhole")

    max_dt_ns = int(max_depth_dt * 1_000_000_000)
    depth_sorted = sorted(depth_stamps)
    used_depth = set()
    pairs = []
    for color_stamp in color_stamps:
        idx = bisect.bisect_left(depth_sorted, color_stamp)
        candidates = []
        for candidate_idx in (idx - 1, idx):
            if 0 <= candidate_idx < len(depth_sorted):
                depth_stamp = depth_sorted[candidate_idx]
                if depth_stamp not in used_depth:
                    candidates.append(depth_stamp)
        if not candidates:
            continue
        depth_stamp = min(candidates, key=lambda value: abs(value - color_stamp))
        depth_dt_ns = abs(depth_stamp - color_stamp)
        if depth_dt_ns > max_dt_ns:
            continue
        used_depth.add(depth_stamp)
        pairs.append(
            {
                "seq": len(pairs),
                "color_stamp_ns": color_stamp,
                "depth_stamp_ns": depth_stamp,
                "depth_dt_ns": depth_dt_ns,
            }
        )

    if not pairs:
        raise RuntimeError("No synchronized RGB/depth pairs found")
    return color_info, depth_info, pairs


def depth_to_color_maps(color_info: CameraInfo, depth_info: CameraInfo) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(color_info.width, dtype=np.float32), np.arange(color_info.height, dtype=np.float32))
    map_x = (u - color_info.k[0, 2]) * depth_info.k[0, 0] / color_info.k[0, 0] + depth_info.k[0, 2]
    map_y = (v - color_info.k[1, 2]) * depth_info.k[1, 1] / color_info.k[1, 1] + depth_info.k[1, 2]
    return map_x.astype(np.float32, copy=False), map_y.astype(np.float32, copy=False)


def decode_color(msg) -> np.ndarray:
    if str(msg.format).lower() != "jpeg":
        raise ValueError(f"Unsupported DepthCaptureLab color format: {msg.format}")
    image = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG color frame")
    return image


def decode_depth_meters(msg) -> np.ndarray:
    if msg.encoding != "32FC1":
        raise ValueError(f"Unsupported DepthCaptureLab depth encoding: {msg.encoding}")
    depth = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(int(msg.height), int(msg.width))
    if int(msg.is_bigendian):
        depth = depth.byteswap()
    return depth


def write_depth_png(depth_m: np.ndarray, path: Path, maps: tuple[np.ndarray, np.ndarray]) -> None:
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    aligned_m = cv2.remap(depth_m, maps[0], maps[1], interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    depth_mm = np.clip(np.rint(aligned_m * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if not cv2.imwrite(str(path), depth_mm):
        raise RuntimeError(f"Failed to write depth frame: {path}")


def export_scene(input_path: Path, mcap_path: Path, output_dir: Path, max_depth_dt: float) -> None:
    color_info, depth_info, pairs = inspect_mcap(mcap_path, max_depth_dt)
    pair_by_color = {item["color_stamp_ns"]: item for item in pairs}
    pair_by_depth = {item["depth_stamp_ns"]: item for item in pairs}
    color_cache = {}
    depth_cache = {}
    maps = depth_to_color_maps(color_info, depth_info)
    frames = []

    print(
        f"Exporting {output_dir.name}: {len(pairs)} synchronized frames "
        f"@ {fps_from_timestamps_ns([item['color_stamp_ns'] for item in pairs]):.2f} FPS"
    )
    with tqdm(total=len(pairs), desc=output_dir.name, unit="frame") as progress:
        for _, channel, record, msg in iter_decoded_messages(mcap_path):
            if channel.topic == COLOR_TOPIC and record.log_time in pair_by_color:
                pair = pair_by_color[record.log_time]
                color_cache[pair["seq"]] = decode_color(msg)
            elif channel.topic == DEPTH_TOPIC and record.log_time in pair_by_depth:
                pair = pair_by_depth[record.log_time]
                depth_cache[pair["seq"]] = decode_depth_meters(msg)
            else:
                continue

            seq = pair["seq"]
            if seq not in color_cache or seq not in depth_cache:
                continue

            stem = frame_stem(seq)
            color_file = output_dir / "color" / f"{stem}.png"
            depth_file = output_dir / "depth" / f"{stem}.png"
            if not cv2.imwrite(str(color_file), color_cache.pop(seq)):
                raise RuntimeError(f"Failed to write color frame: {color_file}")
            write_depth_png(depth_cache.pop(seq), depth_file, maps)

            frame = {
                "seq": seq,
                "stem": stem,
                "color_file": f"color/{stem}.png",
                "depth_file": f"depth/{stem}.png",
                "color_stamp_ns": pair["color_stamp_ns"],
                "depth_stamp_ns": pair["depth_stamp_ns"],
                "depth_dt_ns": pair["depth_dt_ns"],
            }
            frames.append(frame)
            progress.update(1)

    frames.sort(key=lambda item: item["seq"])
    if len(frames) != len(pairs):
        raise RuntimeError(f"Wrote {len(frames)} frames but expected {len(pairs)}")

    write_pinhole_intrinsics(
        output_dir / "intrinsic" / "intrinsic_color.json",
        width=color_info.width,
        height=color_info.height,
        fx=color_info.k[0, 0],
        fy=color_info.k[1, 1],
        cx=color_info.k[0, 2],
        cy=color_info.k[1, 2],
        source={
            "type": "sensor_msgs/msg/CameraInfo",
            "topic": color_info.topic,
            "stamp_ns": color_info.stamp_ns,
            "frame_id": color_info.frame_id,
            "rectified_from_distortion": False,
        },
    )
    write_scene_metadata(
        output_dir,
        name=output_dir.name,
        fps=fps_from_timestamps_ns([item["color_stamp_ns"] for item in frames]),
        width=color_info.width,
        height=color_info.height,
        frames=frames,
        source={
            "type": "depthcapturelab_ros2_mcap",
            "input_path": str(input_path),
            "mcap_path": str(mcap_path),
            "color_topic": COLOR_TOPIC,
            "depth_topic": DEPTH_TOPIC,
            "color_info_topic": COLOR_INFO_TOPIC,
            "depth_info_topic": DEPTH_INFO_TOPIC,
            "input_depth_unit": "meter",
            "depth_unit": "millimeter",
            "depth_alignment": "depth_intrinsics_to_color_intrinsics_identity_extrinsics",
            "depth_frame_id": depth_info.frame_id,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DepthCaptureLab MCAP exports into canonical ViPE RGB-D scenes.")
    parser.add_argument("input_path", type=Path, help="DepthCaptureLab .zip, session directory, or recording.mcap.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Canonical ViPE scene output directory.")
    parser.add_argument("--max-depth-dt", type=float, default=0.05, help="Maximum RGB/depth timestamp delta in seconds.")
    args = parser.parse_args()

    args.input_path = args.input_path.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input path not found: {args.input_path}")
    if args.max_depth_dt < 0.0:
        raise ValueError("--max-depth-dt must be non-negative")

    with tempfile.TemporaryDirectory(prefix="depthcapture_to_vipe_") as temp:
        mcap_path = resolve_mcap_path(args.input_path, Path(temp))
        prepare_output_dir(args.output_dir, args.input_path)
        export_scene(args.input_path, mcap_path, args.output_dir, args.max_depth_dt)


if __name__ == "__main__":
    main()
