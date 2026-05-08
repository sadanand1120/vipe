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
    args = parser.parse_args()
    if args.downsize_factor <= 0.0:
        raise ValueError("--downsize-factor must be positive")

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

    frame_idx = 0
    try:
        with tqdm(total=total, desc="Extracting frames", unit="frame") as pbar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if args.downsize_factor != 1.0:
                    height, width = frame.shape[:2]
                    new_width = max(1, int(round(width / args.downsize_factor)))
                    new_height = max(1, int(round(height / args.downsize_factor)))
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                out_path = args.output_dir / f"{frame_idx:05d}.png"
                if not cv2.imwrite(str(out_path), frame):
                    raise IOError(f"Failed to write frame: {out_path}")
                frame_idx += 1
                pbar.update(1)
    finally:
        cap.release()

    if frame_idx == 0:
        raise ValueError(f"No frames decoded from video: {args.video_path}")

    print(f"Wrote {frame_idx} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
