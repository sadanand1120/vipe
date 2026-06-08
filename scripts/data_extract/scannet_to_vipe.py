import argparse
import multiprocessing as mp
import signal
import shutil
import struct
import traceback
import zlib

from pathlib import Path
from queue import Empty

import cv2
import numpy as np
from tqdm import tqdm

from vipe.utils.data_format import frame_stem, write_pinhole_intrinsics, write_scene_metadata


COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}
PROGRESS_UPDATE_INTERVAL = 16


def emit_progress(progress_queue, worker_idx: int | None, kind: str, **payload) -> None:
    if progress_queue is None or worker_idx is None:
        return
    progress_queue.put({"kind": kind, "worker_idx": worker_idx, **payload})


def reset_worker_bar(bar: tqdm, desc: str, total: int) -> None:
    bar.reset(total=max(total, 1))
    bar.n = 0
    bar.set_description_str(desc)
    bar.refresh()


def update_worker_bar(bar: tqdm, current: int) -> None:
    bar.n = current
    bar.refresh()


def set_worker_idle(bar: tqdm, worker_idx: int) -> None:
    bar.reset(total=1)
    bar.n = 0
    bar.set_description_str(f"W{worker_idx} idle")
    bar.refresh()


def stop_workers(workers: list[mp.Process]) -> None:
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(timeout=2.0)
    for worker in workers:
        if worker.is_alive():
            worker.kill()
    for worker in workers:
        worker.join()


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
    def __init__(self, filename: Path, show_progress: bool = True) -> None:
        self.load(filename, show_progress)

    def load(
        self,
        filename: Path,
        show_progress: bool,
        progress_queue=None,
        worker_idx: int | None = None,
        scene_name: str | None = None,
    ) -> None:
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

            emit_progress(progress_queue, worker_idx, "load_start", scene=scene_name, total=num_frames)
            self.frames = []
            frame_iter = range(num_frames)
            if show_progress:
                frame_iter = tqdm(frame_iter, desc=f"Load {filename.name}", unit="frame")
            for frame_idx in frame_iter:
                frame = RGBDFrame()
                frame.load(handle)
                self.frames.append(frame)
                if progress_queue is not None and ((frame_idx + 1) % PROGRESS_UPDATE_INTERVAL == 0 or frame_idx + 1 == num_frames):
                    emit_progress(
                        progress_queue,
                        worker_idx,
                        "load_progress",
                        scene=scene_name,
                        current=frame_idx + 1,
                    )


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


def decode_scene(
    scene_dir: Path,
    output_scene_dir: Path,
    frame_skip: int,
    show_progress: bool = True,
    progress_queue=None,
    worker_idx: int | None = None,
) -> None:
    sens_files = list(scene_dir.glob("*.sens"))
    if len(sens_files) != 1:
        raise ValueError(f"Expected exactly one .sens file in {scene_dir}, found {len(sens_files)}")

    prepare_scene_dir(scene_dir, output_scene_dir)

    sensor_data = SensorData.__new__(SensorData)
    sensor_data.load(
        sens_files[0],
        show_progress=show_progress,
        progress_queue=progress_queue,
        worker_idx=worker_idx,
        scene_name=scene_dir.name,
    )
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
    if show_progress:
        print(f"Decoding {scene_dir.name}: {len(frame_indices)} frames")
    else:
        emit_progress(progress_queue, worker_idx, "export_start", scene=scene_dir.name, total=len(frame_indices))

    frame_iter = frame_indices
    if show_progress:
        frame_iter = tqdm(frame_iter, desc=scene_dir.name, unit="frame")
    for export_idx, source_idx in enumerate(frame_iter):
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
        if progress_queue is not None and ((export_idx + 1) % PROGRESS_UPDATE_INTERVAL == 0 or export_idx + 1 == len(frame_indices)):
            emit_progress(
                progress_queue,
                worker_idx,
                "export_progress",
                scene=scene_dir.name,
                current=export_idx + 1,
            )

    write_scene_metadata(
        output_scene_dir,
        name=scene_dir.name,
        width=sensor_data.color_width,
        height=sensor_data.color_height,
        frames=frames,
        source={
            "dataset": "ScanNet",
            "scene": scene_dir.name,
            "sens_file": str(sens_files[0]),
            "frame_skip": frame_skip,
            "depth_unit": "millimeter",
        },
    )


def decode_scene_worker(task_queue, progress_queue, output_root: Path, frame_skip: int, worker_idx: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        scene_dir = task_queue.get()
        if scene_dir is None:
            return
        try:
            decode_scene(
                scene_dir,
                output_root / scene_dir.name,
                frame_skip,
                show_progress=False,
                progress_queue=progress_queue,
                worker_idx=worker_idx,
            )
            emit_progress(progress_queue, worker_idx, "scene_done", scene=scene_dir.name)
        except Exception:
            emit_progress(progress_queue, worker_idx, "error", scene=scene_dir.name, error=traceback.format_exc())
            return


def run_parallel_scenes(scenes: list[Path], output_root: Path, frame_skip: int, num_workers: int) -> None:
    worker_count = min(num_workers, len(scenes))
    ctx = mp.get_context("fork")
    task_queue = ctx.Queue()
    progress_queue = ctx.Queue()
    for scene_dir in scenes:
        task_queue.put(scene_dir)
    for _ in range(worker_count):
        task_queue.put(None)

    workers = [
        ctx.Process(target=decode_scene_worker, args=(task_queue, progress_queue, output_root, frame_skip, worker_idx))
        for worker_idx in range(worker_count)
    ]
    for worker in workers:
        worker.daemon = True
        worker.start()

    scene_bar = tqdm(total=len(scenes), desc="Scenes", unit="scene", position=0)
    worker_bars = [tqdm(total=1, desc=f"W{idx} idle", unit="frame", position=idx + 1) for idx in range(worker_count)]
    completed = 0
    try:
        try:
            while completed < len(scenes):
                try:
                    message = progress_queue.get(timeout=0.5)
                except Empty:
                    if any(worker.exitcode not in (None, 0) for worker in workers):
                        raise RuntimeError("A ScanNet extraction worker exited unexpectedly.")
                    continue

                kind = message["kind"]
                worker_idx = message["worker_idx"]
                worker_bar = worker_bars[worker_idx]
                if kind == "load_start":
                    reset_worker_bar(worker_bar, f"W{worker_idx} load {message['scene']}", message["total"])
                elif kind == "load_progress":
                    update_worker_bar(worker_bar, message["current"])
                elif kind == "export_start":
                    reset_worker_bar(worker_bar, f"W{worker_idx} {message['scene']}", message["total"])
                elif kind == "export_progress":
                    update_worker_bar(worker_bar, message["current"])
                elif kind == "scene_done":
                    completed += 1
                    scene_bar.update(1)
                    set_worker_idle(worker_bar, worker_idx)
                elif kind == "error":
                    raise RuntimeError(message["error"])
        except KeyboardInterrupt:
            stop_workers(workers)
            raise SystemExit(130)
    finally:
        stop_workers(workers)
        task_queue.close()
        progress_queue.close()
        scene_bar.close()
        for worker_bar in worker_bars:
            worker_bar.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ScanNet .sens files into canonical ViPE RGB-D scenes.")
    parser.add_argument("--scans-root", required=True, type=Path, help="Directory containing ScanNet scene folders.")
    parser.add_argument("--output-root", required=True, type=Path, help="Directory to write canonical scene folders into.")
    scene_group = parser.add_mutually_exclusive_group()
    scene_group.add_argument("--scenes", nargs="*", default=None, help="Scene names to extract. Defaults to all scenes.")
    scene_group.add_argument("--totN", type=int, default=None, help="Extract the first N sorted scenes.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Write every Nth source frame.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of scenes to extract in parallel.")
    args = parser.parse_args()

    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.totN is not None and args.totN < 1:
        raise ValueError("--totN must be >= 1")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.scenes:
        scenes = [args.scans_root / scene for scene in args.scenes]
    else:
        scenes = sorted(path for path in args.scans_root.iterdir() if path.is_dir())
        if args.totN is not None:
            scenes = scenes[: args.totN]

    for scene_dir in scenes:
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    if args.num_workers == 1 or len(scenes) <= 1:
        for scene_dir in scenes:
            decode_scene(scene_dir, args.output_root / scene_dir.name, args.frame_skip)
        return

    run_parallel_scenes(scenes, args.output_root, args.frame_skip, args.num_workers)


if __name__ == "__main__":
    main()
