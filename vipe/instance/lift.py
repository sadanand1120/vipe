"""Occlusion-correct projection and 2D-mask lifting."""

from collections.abc import Callable, Sequence

import numpy as np
import torch


@torch.no_grad()
def visible_voxels(
    points: torch.Tensor,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    depth: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ascending visible point indices and their integer image coordinates."""
    device = points.device
    c2w_t = torch.as_tensor(np.asarray(c2w), dtype=torch.float32, device=device)
    fx, fy, cx, cy = (float(value) for value in intrinsics)

    # Preserve the fixed arithmetic order that removed cuBLAS projection jitter in the frontier.
    dx = points[:, 0] - c2w_t[0, 3]
    dy = points[:, 1] - c2w_t[1, 3]
    dz = points[:, 2] - c2w_t[2, 3]
    rotation = c2w_t[:3, :3]
    x = dx * rotation[0, 0] + dy * rotation[1, 0] + dz * rotation[2, 0]
    y = dx * rotation[0, 1] + dy * rotation[1, 1] + dz * rotation[2, 1]
    z = dx * rotation[0, 2] + dy * rotation[1, 2] + dz * rotation[2, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    in_bounds = (
        (z > 1e-3)
        & torch.isfinite(u)
        & torch.isfinite(v)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    ui = u.long().clamp(0, width - 1)
    vi = v.long().clamp(0, height - 1)
    depth_t = torch.as_tensor(np.asarray(depth), dtype=torch.float32, device=device)
    measured = depth_t[vi, ui]
    visible = in_bounds & (measured > 1e-3) & ((z - measured).abs() <= tolerance)
    indices = torch.nonzero(visible, as_tuple=False).squeeze(1)
    return (
        indices.cpu().numpy(),
        ui[indices].cpu().numpy(),
        vi[indices].cpu().numpy(),
    )


def lift_masks(
    points: torch.Tensor,
    frames: Sequence[int],
    c2w_of: Callable[[int], np.ndarray],
    masks_of: Callable[[int], Sequence[tuple[np.ndarray, float, int, int]]],
    depth_of: Callable[[int], np.ndarray],
    intrinsics: np.ndarray,
    width: int,
    height: int,
    occlusion_tolerance: float,
    min_voxels: int,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Lift all masks while retaining frame visibility and track provenance."""
    lifted = []
    for position, frame_index in enumerate(frames, 1):
        if position % 100 == 0 or position == len(frames):
            log(f"  [instance] lift {position}/{len(frames)} frames={len(lifted)}")
        indices, u, v = visible_voxels(
            points,
            c2w_of(frame_index),
            intrinsics,
            width,
            height,
            depth_of(frame_index),
            occlusion_tolerance,
        )
        if not indices.size:
            continue
        visible = indices.astype(np.int64)
        masks, scores, track_ids, global_ids = [], [], [], []
        for mask, score, track_id, global_id in masks_of(frame_index):
            selected = mask[v, u]
            if not selected.any():
                continue
            voxels = visible[selected]
            if voxels.size >= min_voxels:
                masks.append(voxels.astype(np.int32))
                scores.append(float(score))
                track_ids.append(int(track_id))
                global_ids.append(int(global_id))
        if masks:
            lifted.append(
                {
                    "frame": int(frame_index),
                    "V": visible.astype(np.int32),
                    "masks": masks,
                    "c2w": np.asarray(c2w_of(frame_index), np.float32),
                    "mscore": np.asarray(scores, np.float32),
                    "mtid": np.asarray(track_ids, np.int64),
                    "mgid": np.asarray(global_ids, np.int64),
                }
            )
    return lifted
