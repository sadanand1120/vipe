from __future__ import annotations

import math

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


VIDEO_H = 720
TRAJ_SIZE = 720
FPS = 30
PLANE = (0, 1)


def _world_to_canvas(points_xy: np.ndarray, size: int, margin: int = 45):
    mn = points_xy.min(axis=0)
    mx = points_xy.max(axis=0)
    center = 0.5 * (mn + mx)
    half = max(0.5 * float(np.max(mx - mn)) * 1.15, 1e-3)
    scale = (size - 2 * margin) / (2 * half)

    def project(xy: np.ndarray) -> np.ndarray:
        out = (xy - center) * scale
        out[:, 0] += size / 2
        out[:, 1] = size / 2 - out[:, 1]
        return np.round(out).astype(np.int32)

    return project, half, scale


def _draw_grid(canvas: np.ndarray, half: float, scale: float) -> None:
    size = canvas.shape[0]
    canvas[:] = 250
    raw = 2 * half / 8
    power = 10 ** math.floor(math.log10(max(raw, 1e-6)))
    step = next((power * m for m in (1, 2, 5, 10) if power * m >= raw), power * 10)
    grid_color = (225, 225, 225)
    axis_color = (175, 175, 175)
    n = int(math.ceil(half / step)) + 1
    for i in range(-n, n + 1):
        delta = i * step * scale
        x = int(round(size / 2 + delta))
        y = int(round(size / 2 - delta))
        if 0 <= x < size:
            cv2.line(canvas, (x, 0), (x, size - 1), grid_color, 1, cv2.LINE_AA)
        if 0 <= y < size:
            cv2.line(canvas, (0, y), (size - 1, y), grid_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (size // 2, 0), (size // 2, size - 1), axis_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (0, size // 2), (size - 1, size // 2), axis_color, 1, cv2.LINE_AA)


def _draw_poly(canvas: np.ndarray, pts: np.ndarray, upto: int, color, pale) -> None:
    if len(pts) >= 2:
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, pale, 1, cv2.LINE_AA)
    if upto >= 1:
        cv2.polylines(canvas, [pts[: upto + 1].reshape(-1, 1, 2)], False, color, 3, cv2.LINE_AA)
    if upto >= 0:
        cv2.circle(canvas, tuple(pts[upto]), 5, color, -1, cv2.LINE_AA)


def _heading_xy(pose: np.ndarray) -> np.ndarray:
    forward = pose[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    xy = forward[list(PLANE)]
    norm = np.linalg.norm(xy)
    if norm < 1e-9:
        return np.array([1.0, 0.0])
    return xy / norm


def _draw_heading(canvas: np.ndarray, pose: np.ndarray, project, color, length_m: float) -> None:
    p = pose[:3, 3][list(PLANE)]
    q = p + _heading_xy(pose) * length_m
    pq = project(np.stack([p, q], axis=0))
    cv2.arrowedLine(canvas, tuple(pq[0]), tuple(pq[1]), color, 3, cv2.LINE_AA, tipLength=0.35)


def _resize_rgb(path: str | Path, height: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    h, w = bgr.shape[:2]
    out_w = int(round(w * height / h))
    if out_w % 2 == 1:
        out_w += 1
    return cv2.resize(bgr, (out_w, height), interpolation=cv2.INTER_AREA)


def _load_pose_npz(pose_path: str | Path) -> dict[int, np.ndarray]:
    with np.load(pose_path) as data:
        return {int(idx): pose.astype(np.float64) for idx, pose in zip(data["inds"], data["data"])}


def save_trajectory_debug_video(
    *,
    scene: str,
    image_files: list[str],
    gt_w2c: np.ndarray,
    frame_indices: list[int],
    pred_pose_path: str | Path,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_map = _load_pose_npz(pred_pose_path)
    gt_w2c = gt_w2c.astype(np.float64)
    valid_gt = np.isfinite(gt_w2c).all(axis=(1, 2))
    valid_frame_indices = [int(idx) for idx, valid in zip(frame_indices, valid_gt) if bool(valid)]
    valid_image_files = [str(path) for path, valid in zip(image_files, valid_gt) if bool(valid)]
    gt_c2w = np.linalg.inv(gt_w2c[valid_gt])
    gt_map = {frame_idx: pose for frame_idx, pose in zip(valid_frame_indices, gt_c2w)}
    overlap = sorted(set(pred_map) & set(gt_map))
    if not overlap:
        raise RuntimeError(f"No pred/GT pose overlap for trajectory video: {scene}")

    first = overlap[0]
    first_align = gt_map[first] @ np.linalg.inv(pred_map[first])
    pred_ids = [frame_idx for frame_idx in valid_frame_indices if frame_idx in pred_map]
    pred_aligned = {idx: first_align @ pred_map[idx] for idx in pred_ids}

    gt_xy = gt_c2w[:, :3, 3][:, list(PLANE)]
    pred_xy = np.stack([pred_aligned[idx][:3, 3][list(PLANE)] for idx in pred_ids], axis=0)
    project, half, scale = _world_to_canvas(np.concatenate([gt_xy, pred_xy], axis=0), TRAJ_SIZE)
    gt_px = project(gt_xy)
    pred_px = project(pred_xy)
    gt_index = {idx: n for n, idx in enumerate(valid_frame_indices)}
    pred_index = {idx: n for n, idx in enumerate(pred_ids)}
    heading_len = max(0.25, 0.06 * half)

    with imageio.get_writer(str(out_path), fps=FPS, codec="libx264", macro_block_size=None) as writer:
        iterator = zip(valid_frame_indices, valid_image_files, strict=True)
        for frame_id, image_file in tqdm(iterator, total=len(valid_frame_indices), desc=f"traj video {scene}", unit="frame"):
            frame_id = int(frame_id)
            if frame_id not in pred_index:
                continue
            rgb_panel = _resize_rgb(image_file, VIDEO_H)
            traj = np.empty((TRAJ_SIZE, TRAJ_SIZE, 3), dtype=np.uint8)
            _draw_grid(traj, half, scale)

            gi = gt_index[frame_id]
            pi = pred_index[frame_id]
            pred_pose = pred_aligned[frame_id]
            _draw_poly(traj, gt_px, gi, (50, 170, 70), (195, 230, 200))
            _draw_poly(traj, pred_px, pi, (210, 100, 25), (235, 205, 180))
            _draw_heading(traj, gt_c2w[gi], project, (25, 130, 45), heading_len)
            _draw_heading(traj, pred_pose, project, (185, 70, 15), heading_len)

            pos_err = np.linalg.norm(gt_c2w[gi][:3, 3] - pred_pose[:3, 3])
            cv2.putText(rgb_panel, scene, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(rgb_panel, scene, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(traj, f"frame {frame_id}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(traj, "GT green | Pred orange | first-pose align", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(traj, f"position error {pos_err:.3f} m", (18, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)

            writer.append_data(cv2.cvtColor(np.concatenate([rgb_panel, traj], axis=1), cv2.COLOR_BGR2RGB))
    return out_path
