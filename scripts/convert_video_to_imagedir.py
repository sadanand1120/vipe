#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a video into a ViPE-compatible image directory.")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace output_dir if it already exists")
    parser.add_argument(
        "--downsize-factor",
        type=float,
        default=1.0,
        help="Divide frame height and width by this factor before writing PNGs",
    )
    parser.add_argument(
        "--store-fps",
        type=float,
        default=None,
        help="FPS to store on disk by subsampling the source video. Defaults to the input FPS.",
    )
    parser.add_argument(
        "--orig-frame-idx",
        action="store_true",
        help="Name stored frames using their original source-video frame indices instead of sequential kept-frame indices.",
    )
    args = parser.parse_args()
    if args.downsize_factor <= 0.0:
        raise ValueError("--downsize-factor must be positive")
    if args.store_fps is not None and args.store_fps <= 0.0:
        raise ValueError("--store-fps must be positive")

    if not args.video_path.is_file():
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    if args.output_dir.exists():
        if not args.overwrite and any(args.output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
        if args.overwrite:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = total if total > 0 else None
    print(f"Detected FPS: {fps:.6g}")
    if total is not None:
        print(f"Detected frames: {total}")

    if args.store_fps is not None:
        if fps <= 0.0:
            raise ValueError("--store-fps requires a valid positive input FPS from video metadata")
        if args.store_fps > fps:
            raise ValueError(
                f"--store-fps ({args.store_fps:.6g}) cannot exceed input FPS ({fps:.6g})"
            )
    store_fps = fps if args.store_fps is None else args.store_fps
    if store_fps is not None:
        print(f"Storing FPS: {store_fps:.6g}")

    decoded_frame_idx = 0
    stored_frame_idx = 0
    first_stored_time = None
    last_stored_time = None
    next_store_time = 0.0
    time_epsilon = 1e-9
    try:
        with tqdm(total=total, desc="Extracting frames", unit="frame") as pbar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                should_store = True
                if args.store_fps is not None:
                    frame_time = decoded_frame_idx / fps
                    should_store = frame_time + time_epsilon >= next_store_time

                if should_store:
                    frame_time = decoded_frame_idx / fps
                    if args.downsize_factor != 1.0:
                        height, width = frame.shape[:2]
                        new_width = max(1, int(round(width / args.downsize_factor)))
                        new_height = max(1, int(round(height / args.downsize_factor)))
                        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    output_frame_idx = decoded_frame_idx if args.orig_frame_idx else stored_frame_idx
                    out_path = args.output_dir / f"{output_frame_idx:05d}.png"
                    if not cv2.imwrite(str(out_path), frame):
                        raise IOError(f"Failed to write frame: {out_path}")
                    if first_stored_time is None:
                        first_stored_time = frame_time
                    last_stored_time = frame_time
                    stored_frame_idx += 1
                    if args.store_fps is not None:
                        next_store_time += 1.0 / args.store_fps

                decoded_frame_idx += 1
                pbar.update(1)
    finally:
        cap.release()

    if stored_frame_idx == 0:
        raise ValueError(f"No frames decoded from video: {args.video_path}")

    print(f"Wrote {stored_frame_idx} frames to {args.output_dir}")
    if stored_frame_idx == 1:
        effective_store_fps = fps
    else:
        assert first_stored_time is not None and last_stored_time is not None
        effective_store_fps = (stored_frame_idx - 1) / (last_stored_time - first_stored_time)
    print(f"Effective stored FPS: {effective_store_fps:.12g}")


if __name__ == "__main__":
    main()
