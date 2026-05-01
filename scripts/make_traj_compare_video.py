#!/usr/bin/env python3

import argparse
import math

from pathlib import Path

import cv2
import imageio
import numpy as np

from vipe.utils.logging import pbar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a passive trajectory-comparison MP4 from ViPE poses and GT poses."
    )
    parser.add_argument(
        "vipe_output_dir",
        type=Path,
        help="ViPE output directory containing pose/*.npz",
    )
    parser.add_argument(
        "--gt-pose-dir",
        type=Path,
        required=True,
        help="Directory containing per-frame GT pose txt files, e.g. ScanNet pose/*.txt",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Sequence name if pose/*.npz contains more than one file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mp4 path. Defaults to vipe_aux_vis/<sequence>_traj.mp4",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS",
    )
    parser.add_argument(
        "--plane",
        type=str,
        choices=["xy", "xz", "yz"],
        default="xy",
        help="2D plane used for plotting",
    )
    parser.add_argument(
        "--alignment",
        type=str,
        choices=["sim3", "se3"],
        default="sim3",
        help="Alignment used to overlay prediction onto GT",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=900,
        help="Square canvas size in pixels",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export aligned predicted poses as per-frame txt files in ScanNet GT pose format.",
    )
    return parser.parse_args()


def discover_pred_pose(vipe_output_dir: Path, sequence: str | None) -> tuple[str, Path]:
    pose_dir = vipe_output_dir / "pose"
    pose_files = sorted(pose_dir.glob("*.npz"))
    if len(pose_files) == 0:
        raise FileNotFoundError(f"No pose npz found under {pose_dir}")

    if sequence is not None:
        pose_path = pose_dir / f"{sequence}.npz"
        if not pose_path.exists():
            raise FileNotFoundError(f"Could not find pose file for sequence '{sequence}' at {pose_path}")
        return sequence, pose_path

    if len(pose_files) > 1:
        raise ValueError(f"Found multiple pose files under {pose_dir}; pass --sequence explicitly")

    pose_path = pose_files[0]
    return pose_path.stem, pose_path


def load_pred_pose_npz(pose_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(pose_path)
    frame_ids = data["inds"].astype(np.int64)
    pose_mats = data["data"].astype(np.float64)
    xyz = pose_mats[:, :3, 3]
    return frame_ids, pose_mats, xyz


def load_gt_pose_dir(gt_pose_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pose_files = sorted(gt_pose_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if len(pose_files) == 0:
        raise FileNotFoundError(f"No GT pose txt files found under {gt_pose_dir}")

    frame_ids = []
    xyz = []
    for pose_file in pbar(pose_files, desc="Loading GT poses"):
        pose = np.loadtxt(pose_file, dtype=np.float64)
        if pose.shape != (4, 4):
            continue
        if not np.isfinite(pose).all():
            continue
        frame_ids.append(int(pose_file.stem))
        xyz.append(pose[:3, 3])

    if len(frame_ids) == 0:
        raise ValueError(f"No valid GT poses found under {gt_pose_dir}")

    return np.asarray(frame_ids, dtype=np.int64), np.asarray(xyz, dtype=np.float64)


def umeyama_alignment(src_xyz: np.ndarray, dst_xyz: np.ndarray, estimate_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    if src_xyz.shape != dst_xyz.shape or src_xyz.shape[0] < 2:
        raise ValueError("Need at least two matched points for alignment")

    src_mean = src_xyz.mean(axis=0)
    dst_mean = dst_xyz.mean(axis=0)
    src_centered = src_xyz - src_mean
    dst_centered = dst_xyz - dst_mean

    cov = (dst_centered.T @ src_centered) / src_xyz.shape[0]
    u, singular_vals, vh = np.linalg.svd(cov)

    sign = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1, -1] = -1.0

    rotation = u @ sign @ vh

    if estimate_scale:
        src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
        scale = np.sum(singular_vals * np.diag(sign)) / src_var
    else:
        scale = 1.0

    translation = dst_mean - scale * (rotation @ src_mean)
    return float(scale), rotation, translation


def apply_alignment(xyz: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (xyz @ rotation.T) + translation


def apply_alignment_to_pose_mats(
    pose_mats: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    aligned_pose_mats = pose_mats.copy()
    aligned_pose_mats[:, :3, :3] = rotation[None] @ aligned_pose_mats[:, :3, :3]
    aligned_pose_mats[:, :3, 3] = apply_alignment(aligned_pose_mats[:, :3, 3], scale, rotation, translation)
    aligned_pose_mats[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return aligned_pose_mats


def export_aligned_pose_dir(export_dir: Path, frame_ids: np.ndarray, pose_mats: np.ndarray) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    for frame_id, pose_mat in pbar(zip(frame_ids.tolist(), pose_mats), total=len(frame_ids), desc="Exporting aligned poses"):
        np.savetxt(export_dir / f"{frame_id}.txt", pose_mat, fmt="%.6f")


def plane_indices(plane: str) -> tuple[int, int]:
    return {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
    }[plane]


def nice_grid_step(extent: float) -> float:
    if extent <= 0:
        return 1.0
    raw = extent / 8.0
    power = 10 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 5.0, 10.0):
        step = power * mult
        if step >= raw:
            return step
    return power * 10.0


def make_world_to_image(all_xy: np.ndarray, size: int, margin: int = 70):
    xy_min = all_xy.min(axis=0)
    xy_max = all_xy.max(axis=0)
    center = 0.5 * (xy_min + xy_max)
    half_extent = 0.5 * np.max(xy_max - xy_min)
    half_extent = max(half_extent, 1e-3)
    half_extent *= 1.1

    usable = size - 2 * margin
    scale = usable / (2.0 * half_extent)

    def project(xy: np.ndarray) -> np.ndarray:
        proj = (xy - center) * scale
        proj[:, 0] += size / 2.0
        proj[:, 1] = size / 2.0 - proj[:, 1]
        return np.round(proj).astype(np.int32)

    return center, half_extent, scale, project


def draw_grid(canvas: np.ndarray, center: np.ndarray, half_extent: float, scale: float, plane: str) -> None:
    grid_step = nice_grid_step(2.0 * half_extent)
    size = canvas.shape[0]
    origin_px = np.array([size / 2.0, size / 2.0])

    n_steps = int(math.ceil(half_extent / grid_step))
    grid_color = (225, 225, 225)
    axis_color = (170, 170, 170)

    for i in range(-n_steps, n_steps + 1):
        x_val = i * grid_step
        x_px = int(round(origin_px[0] + x_val * scale))
        cv2.line(canvas, (x_px, 0), (x_px, size - 1), grid_color, 1, cv2.LINE_AA)

        y_px = int(round(origin_px[1] - x_val * scale))
        cv2.line(canvas, (0, y_px), (size - 1, y_px), grid_color, 1, cv2.LINE_AA)

    cv2.line(canvas, (int(round(origin_px[0])), 0), (int(round(origin_px[0])), size - 1), axis_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (0, int(round(origin_px[1]))), (size - 1, int(round(origin_px[1]))), axis_color, 1, cv2.LINE_AA)

    axis_names = {
        "xy": ("x", "y"),
        "xz": ("x", "z"),
        "yz": ("y", "z"),
    }[plane]
    cv2.putText(canvas, axis_names[0], (size - 24, int(round(origin_px[1])) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, axis_names[1], (int(round(origin_px[0])) + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 1, cv2.LINE_AA)


def draw_polyline(canvas: np.ndarray, pts_px: np.ndarray, color: tuple[int, int, int]) -> None:
    if len(pts_px) >= 2:
        cv2.polylines(canvas, [pts_px.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
    if len(pts_px) >= 1:
        cv2.circle(canvas, tuple(pts_px[-1]), 5, color, -1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    sequence, pred_pose_path = discover_pred_pose(args.vipe_output_dir, args.sequence)
    pred_inds, pred_pose_mats, pred_xyz = load_pred_pose_npz(pred_pose_path)
    gt_inds, gt_xyz = load_gt_pose_dir(args.gt_pose_dir)

    pred_map = {int(idx): xyz for idx, xyz in zip(pred_inds.tolist(), pred_xyz)}
    gt_map = {int(idx): xyz for idx, xyz in zip(gt_inds.tolist(), gt_xyz)}
    overlap_ids = np.asarray(sorted(set(pred_map) & set(gt_map)), dtype=np.int64)
    if overlap_ids.size < 2:
        raise ValueError("Not enough overlapping GT/pred frames to align trajectories")

    pred_overlap = np.stack([pred_map[int(idx)] for idx in overlap_ids], axis=0)
    gt_overlap = np.stack([gt_map[int(idx)] for idx in overlap_ids], axis=0)
    scale, rotation, translation = umeyama_alignment(
        pred_overlap,
        gt_overlap,
        estimate_scale=args.alignment == "sim3",
    )

    pred_xyz_aligned = apply_alignment(pred_xyz, scale, rotation, translation)
    pred_pose_mats_aligned = apply_alignment_to_pose_mats(pred_pose_mats, scale, rotation, translation)

    ax0, ax1 = plane_indices(args.plane)
    gt_xy_all = gt_xyz[:, [ax0, ax1]]
    pred_xy_all = pred_xyz_aligned[:, [ax0, ax1]]
    all_xy = np.concatenate([gt_xy_all, pred_xy_all], axis=0)
    center, half_extent, scale_px, project = make_world_to_image(all_xy, size=args.size)

    gt_xy_px = project(gt_xy_all)
    pred_xy_px = project(pred_xy_all)
    gt_xyz_aligned_metric = gt_xyz
    pred_xyz_metric = pred_xyz_aligned

    output_path = args.output
    if output_path is None:
        output_path = args.vipe_output_dir / "vipe_aux_vis" / f"{sequence}_traj.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_dir = args.vipe_output_dir / "pose_aligned"
    if args.export:
        export_aligned_pose_dir(export_dir, pred_inds, pred_pose_mats_aligned)

    gt_inds_np = np.asarray(gt_inds)
    pred_inds_np = np.asarray(pred_inds)
    frame_ids = np.arange(0, int(max(gt_inds_np.max(), pred_inds_np.max())) + 1, dtype=np.int64)

    with imageio.get_writer(str(output_path), fps=args.fps, codec="libx264", macro_block_size=None) as writer:
        for frame_id in pbar(frame_ids, desc="Writing trajectory video"):
            canvas = np.full((args.size, args.size, 3), 255, dtype=np.uint8)
            draw_grid(canvas, center, half_extent, scale_px, args.plane)

            gt_mask = gt_inds_np <= frame_id
            pred_mask = pred_inds_np <= frame_id
            draw_polyline(canvas, gt_xy_px[gt_mask], (60, 180, 75))
            draw_polyline(canvas, pred_xy_px[pred_mask], (215, 110, 30))

            pos_err_text = "Pos err N/A"
            if np.any(gt_mask) and np.any(pred_mask):
                gt_current = gt_xyz_aligned_metric[np.flatnonzero(gt_mask)[-1]]
                pred_current = pred_xyz_metric[np.flatnonzero(pred_mask)[-1]]
                pos_err = np.linalg.norm(gt_current - pred_current)
                pos_err_text = f"Pos err {pos_err:.3f} m"

            cv2.putText(
                canvas,
                f"Frame {frame_id}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"GT: green  Pred: blue  Align: {args.alignment}",
                (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{pos_err_text}  |  Metric scale: ScanNet GT",
                (20, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    print(output_path)
    if args.export:
        print(export_dir)


if __name__ == "__main__":
    main()
