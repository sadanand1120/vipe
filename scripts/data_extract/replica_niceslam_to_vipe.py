import argparse
import multiprocessing as mp
import re
import signal
import shutil
import traceback

from pathlib import Path
from queue import Empty

import cv2
import numpy as np
from tqdm import tqdm

from vipe.utils.data_format import frame_stem, write_pinhole_intrinsics, write_scene_metadata


STANDARD_REPLICA_SCENES = (
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
)
FULL_REPLICA_REQUIRED_NAMES = (
    "mesh.ply",
    "semantic.json",
    "semantic.bin",
    "habitat/info_semantic.json",
    "habitat/mesh_semantic.ply",
)
REPLICA_BASE_WIDTH = 1200
REPLICA_BASE_HEIGHT = 680
REPLICA_BASE_FX = 600.0
REPLICA_BASE_FY = 600.0
REPLICA_BASE_CX = 599.5
REPLICA_BASE_CY = 339.5
REPLICA_DEPTH_SCALE = 6553.5
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


def frame_id_from_stem(stem: str) -> int:
    match = re.search(r"(\d+)$", stem)
    if match is None:
        raise ValueError(f"Could not parse Replica frame id from: {stem}")
    return int(match.group(1))


def full_replica_scene_candidates(scene_name: str) -> list[str]:
    candidates = [scene_name]
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", scene_name)
    if match is not None:
        candidates.append(f"{match.group(1)}_{match.group(2)}")
    return candidates


def resolve_full_scene(full_root: Path, scene_name: str) -> Path:
    checked = []
    for candidate in full_replica_scene_candidates(scene_name):
        scene_dir = full_root / candidate
        checked.append(str(scene_dir))
        if not scene_dir.is_dir():
            continue
        missing = [rel_name for rel_name in FULL_REPLICA_REQUIRED_NAMES if not (scene_dir / rel_name).exists()]
        if not missing:
            return scene_dir
    raise FileNotFoundError(f"No valid full Replica scene for {scene_name}. Checked: {checked}")


def sorted_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: frame_id_from_stem(path.stem))


def load_pose_matrices(path: Path) -> list[np.ndarray]:
    poses = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            values = line.strip().split()
            if not values:
                continue
            if len(values) != 16:
                raise ValueError(f"Expected 16 pose values on line {line_idx + 1} in {path}, got {len(values)}")
            poses.append(np.asarray(list(map(float, values)), dtype=np.float32).reshape(4, 4))
    if not poses:
        raise RuntimeError(f"No Replica poses found in: {path}")
    return poses


def load_replica_intrinsics(width: int, height: int) -> np.ndarray:
    scale_x = float(width) / float(REPLICA_BASE_WIDTH)
    scale_y = float(height) / float(REPLICA_BASE_HEIGHT)
    return np.array(
        [
            [REPLICA_BASE_FX * scale_x, 0.0, REPLICA_BASE_CX * scale_x],
            [0.0, REPLICA_BASE_FY * scale_y, REPLICA_BASE_CY * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    np.savetxt(path, matrix, fmt="%.9f")


def convert_depth_to_mm(raw_depth: np.ndarray) -> np.ndarray:
    if raw_depth.dtype != np.uint16:
        raise ValueError(f"Replica depth must be uint16, got {raw_depth.dtype}")
    depth_mm = np.rint(raw_depth.astype(np.float32) * (1000.0 / REPLICA_DEPTH_SCALE))
    depth_mm[raw_depth == 0] = 0.0
    return depth_mm.astype(np.uint16)


def prepare_scene_dir(output_scene_dir: Path, protected_roots: list[Path]) -> None:
    output_scene_dir = output_scene_dir.resolve()
    for protected_root in protected_roots:
        protected_root = protected_root.resolve()
        if protected_root.is_relative_to(output_scene_dir) or output_scene_dir.is_relative_to(protected_root):
            raise ValueError(f"Refusing unsafe output path {output_scene_dir} for source root {protected_root}")
    if output_scene_dir.exists():
        shutil.rmtree(output_scene_dir)
    for subdir in ("color", "depth", "pose", "intrinsic"):
        (output_scene_dir / subdir).mkdir(parents=True, exist_ok=True)


def validate_scene(niceslam_root: Path, full_root: Path, scene_name: str) -> dict:
    if scene_name not in STANDARD_REPLICA_SCENES:
        raise ValueError(
            f"Replica ViPE extractor only supports the 8 NICE-SLAM scenes: {', '.join(STANDARD_REPLICA_SCENES)}"
        )

    scene_dir = niceslam_root / scene_name
    results_dir = scene_dir / "results"
    traj_path = scene_dir / "traj.txt"
    mesh_path = niceslam_root / f"{scene_name}_mesh.ply"
    full_scene_dir = resolve_full_scene(full_root, scene_name)

    if not scene_dir.is_dir():
        raise FileNotFoundError(scene_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(results_dir)
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    color_paths = sorted_paths(list(results_dir.glob("frame*.jpg")))
    if not color_paths:
        color_paths = sorted_paths(list(results_dir.glob("frame*.png")))
    depth_paths = sorted_paths(list(results_dir.glob("depth*.png")))
    if not color_paths:
        raise RuntimeError(f"No Replica color frames found under {results_dir}")
    if not depth_paths:
        raise RuntimeError(f"No Replica depth frames found under {results_dir}")

    color_ids = [frame_id_from_stem(path.stem) for path in color_paths]
    depth_ids = [frame_id_from_stem(path.stem) for path in depth_paths]
    poses = load_pose_matrices(traj_path)

    if len(color_paths) != len(depth_paths):
        raise RuntimeError(
            f"Replica frame mismatch for {scene_name}: {len(color_paths)} color vs {len(depth_paths)} depth"
        )
    if color_ids != depth_ids:
        raise RuntimeError(f"Replica color/depth frame ids differ for {scene_name}")
    if len(color_paths) != len(poses):
        raise RuntimeError(f"Replica frame/pose mismatch for {scene_name}: {len(color_paths)} frames vs {len(poses)} poses")

    first_color = cv2.imread(str(color_paths[0]), cv2.IMREAD_COLOR)
    first_depth = cv2.imread(str(depth_paths[0]), cv2.IMREAD_UNCHANGED)
    if first_color is None:
        raise ValueError(f"Could not read Replica color frame: {color_paths[0]}")
    if first_depth is None:
        raise ValueError(f"Could not read Replica depth frame: {depth_paths[0]}")
    if first_depth.dtype != np.uint16:
        raise ValueError(f"Replica depth must be uint16, got {first_depth.dtype} in {depth_paths[0]}")
    if first_color.shape[:2] != first_depth.shape[:2]:
        raise ValueError(
            f"Replica color/depth size mismatch in {scene_name}: "
            f"{first_color.shape[1]}x{first_color.shape[0]} vs {first_depth.shape[1]}x{first_depth.shape[0]}"
        )

    return {
        "scene_name": scene_name,
        "scene_dir": scene_dir,
        "results_dir": results_dir,
        "traj_path": traj_path,
        "mesh_path": mesh_path,
        "full_scene_dir": full_scene_dir,
        "color_paths": color_paths,
        "depth_paths": depth_paths,
        "frame_ids": color_ids,
        "poses": poses,
        "width": int(first_color.shape[1]),
        "height": int(first_color.shape[0]),
    }


def export_scene(
    scene_info: dict,
    output_root: Path,
    frame_skip: int,
    show_progress: bool = True,
    progress_queue=None,
    worker_idx: int | None = None,
) -> None:
    scene_name = scene_info["scene_name"]
    output_scene_dir = output_root / scene_name
    prepare_scene_dir(output_scene_dir, [scene_info["scene_dir"], scene_info["full_scene_dir"]])

    width = int(scene_info["width"])
    height = int(scene_info["height"])
    intrinsics = load_replica_intrinsics(width, height)
    write_pinhole_intrinsics(
        output_scene_dir / "intrinsic" / "intrinsic_color.json",
        width=width,
        height=height,
        fx=intrinsics[0, 0],
        fy=intrinsics[1, 1],
        cx=intrinsics[0, 2],
        cy=intrinsics[1, 2],
        source={
            "dataset": "Replica",
            "sequence_format": "NICE-SLAM/iMAP rendered RGB-D subset",
            "scene": scene_name,
            "niceslam_scene_dir": str(scene_info["scene_dir"]),
        },
    )

    selected_indices = list(range(0, len(scene_info["frame_ids"]), frame_skip))
    if show_progress:
        print(f"Exporting {scene_name}: {len(selected_indices)} frames")
    else:
        emit_progress(progress_queue, worker_idx, "export_start", scene=scene_name, total=len(selected_indices))

    frames = []
    frame_iter = selected_indices
    if show_progress:
        frame_iter = tqdm(frame_iter, desc=scene_name, unit="frame")
    for export_idx, source_idx in enumerate(frame_iter):
        color_path = scene_info["color_paths"][source_idx]
        depth_path = scene_info["depth_paths"][source_idx]
        pose = scene_info["poses"][source_idx]
        source_frame_id = int(scene_info["frame_ids"][source_idx])

        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color is None:
            raise ValueError(f"Could not read Replica color frame: {color_path}")
        if depth_raw is None:
            raise ValueError(f"Could not read Replica depth frame: {depth_path}")
        if color.shape[:2] != (height, width):
            raise ValueError(
                f"Replica color size changed within {scene_name}: expected {width}x{height}, "
                f"got {color.shape[1]}x{color.shape[0]} for {color_path}"
            )
        if depth_raw.shape[:2] != (height, width):
            raise ValueError(
                f"Replica depth size changed within {scene_name}: expected {width}x{height}, "
                f"got {depth_raw.shape[1]}x{depth_raw.shape[0]} for {depth_path}"
            )

        depth_mm = convert_depth_to_mm(depth_raw)
        stem = frame_stem(export_idx)
        out_color_path = output_scene_dir / "color" / f"{stem}.png"
        out_depth_path = output_scene_dir / "depth" / f"{stem}.png"
        out_pose_path = output_scene_dir / "pose" / f"{stem}.txt"

        if not cv2.imwrite(str(out_color_path), color):
            raise RuntimeError(f"Failed to write color frame: {out_color_path}")
        if not cv2.imwrite(str(out_depth_path), depth_mm):
            raise RuntimeError(f"Failed to write depth frame: {out_depth_path}")
        save_matrix(pose, out_pose_path)

        frames.append(
            {
                "source_frame_id": source_frame_id,
                "source_color_file": color_path.name,
                "source_depth_file": depth_path.name,
            }
        )
        if progress_queue is not None and ((export_idx + 1) % PROGRESS_UPDATE_INTERVAL == 0 or export_idx + 1 == len(selected_indices)):
            emit_progress(
                progress_queue,
                worker_idx,
                "export_progress",
                scene=scene_name,
                current=export_idx + 1,
            )

    write_scene_metadata(
        output_scene_dir,
        name=scene_name,
        width=width,
        height=height,
        frames=frames,
        source={
            "dataset": "Replica",
            "sequence_format": "NICE-SLAM/iMAP rendered RGB-D subset",
            "scene": scene_name,
            "niceslam_scene_dir": str(scene_info["scene_dir"]),
            "niceslam_mesh_file": str(scene_info["mesh_path"]),
            "full_replica_scene_dir": str(scene_info["full_scene_dir"]),
            "full_replica_mesh_file": str(scene_info["full_scene_dir"] / "mesh.ply"),
            "full_replica_habitat_semantic": str(scene_info["full_scene_dir"] / "habitat" / "mesh_semantic.ply"),
            "frame_skip": int(frame_skip),
            "depth_unit_original": "uint16_scaled_depth",
            "depth_scale_original": REPLICA_DEPTH_SCALE,
            "depth_unit_exported": "millimeter",
        },
    )


def process_scene(
    scene_name: str,
    niceslam_root: Path,
    full_root: Path,
    output_root: Path,
    frame_skip: int,
    show_progress: bool = True,
    progress_queue=None,
    worker_idx: int | None = None,
) -> str:
    if not show_progress:
        emit_progress(progress_queue, worker_idx, "validate_start", scene=scene_name)
    scene_info = validate_scene(niceslam_root, full_root, scene_name)
    export_scene(
        scene_info,
        output_root,
        frame_skip,
        show_progress=show_progress,
        progress_queue=progress_queue,
        worker_idx=worker_idx,
    )
    return scene_name


def replica_worker(task_queue, progress_queue, niceslam_root: Path, full_root: Path, output_root: Path, frame_skip: int, worker_idx: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        scene_name = task_queue.get()
        if scene_name is None:
            return
        try:
            process_scene(
                scene_name,
                niceslam_root,
                full_root,
                output_root,
                frame_skip,
                show_progress=False,
                progress_queue=progress_queue,
                worker_idx=worker_idx,
            )
            emit_progress(progress_queue, worker_idx, "scene_done", scene=scene_name)
        except Exception:
            emit_progress(progress_queue, worker_idx, "error", scene=scene_name, error=traceback.format_exc())
            return


def run_parallel_scenes(
    scene_names: tuple[str, ...],
    niceslam_root: Path,
    full_root: Path,
    output_root: Path,
    frame_skip: int,
    num_workers: int,
) -> None:
    worker_count = min(num_workers, len(scene_names))
    ctx = mp.get_context("fork")
    task_queue = ctx.Queue()
    progress_queue = ctx.Queue()
    for scene_name in scene_names:
        task_queue.put(scene_name)
    for _ in range(worker_count):
        task_queue.put(None)

    workers = [
        ctx.Process(
            target=replica_worker,
            args=(task_queue, progress_queue, niceslam_root, full_root, output_root, frame_skip, worker_idx),
        )
        for worker_idx in range(worker_count)
    ]
    for worker in workers:
        worker.daemon = True
        worker.start()

    scene_bar = tqdm(total=len(scene_names), desc="Scenes", unit="scene", position=0)
    worker_bars = [tqdm(total=1, desc=f"W{idx} idle", unit="frame", position=idx + 1) for idx in range(worker_count)]
    completed = 0
    try:
        try:
            while completed < len(scene_names):
                try:
                    message = progress_queue.get(timeout=0.5)
                except Empty:
                    if any(worker.exitcode not in (None, 0) for worker in workers):
                        raise RuntimeError("A Replica extraction worker exited unexpectedly.")
                    continue

                kind = message["kind"]
                worker_idx = message["worker_idx"]
                worker_bar = worker_bars[worker_idx]
                if kind == "validate_start":
                    reset_worker_bar(worker_bar, f"W{worker_idx} prep {message['scene']}", 1)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the 8-scene NICE-SLAM Replica RGB-D subset into canonical ViPE RGB-D scene directories."
    )
    parser.add_argument("--niceslam-root", required=True, type=Path, help="Replica NICE-SLAM RGB-D root")
    parser.add_argument("--full-root", required=True, type=Path, help="Full Replica asset root")
    parser.add_argument("--output-root", required=True, type=Path, help="Directory to write canonical ViPE scenes into")
    parser.add_argument("--frame-skip", type=int, default=1, help="Write every Nth source frame")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of scenes to extract in parallel")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.num_workers == 1:
        for scene_name in STANDARD_REPLICA_SCENES:
            process_scene(scene_name, args.niceslam_root, args.full_root, args.output_root, args.frame_skip)
        return

    run_parallel_scenes(
        STANDARD_REPLICA_SCENES,
        args.niceslam_root,
        args.full_root,
        args.output_root,
        args.frame_skip,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
