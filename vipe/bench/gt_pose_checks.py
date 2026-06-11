from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LargePoseJumpReport:
    has_large_jump: bool
    threshold_m: float
    max_jump_m: float
    jump_count: int
    worst_from_frame: int | None
    worst_to_frame: int | None
    worst_gap: int | None


def large_finite_pose_jump_report(
    extrinsics_w2c: np.ndarray,
    *,
    frame_indices: list[int] | np.ndarray | None = None,
    min_jump_m: float = 0.35,
    p99_multiplier: float = 8.0,
    median_multiplier: float = 25.0,
) -> LargePoseJumpReport:
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float64)
    valid = np.isfinite(extrinsics_w2c).all(axis=(1, 2))
    valid_rows = np.flatnonzero(valid)
    if len(valid_rows) < 2:
        return LargePoseJumpReport(False, float(min_jump_m), 0.0, 0, None, None, None)

    frames = valid_rows if frame_indices is None else np.asarray(frame_indices, dtype=np.int64)[valid]
    c2w = np.linalg.inv(extrinsics_w2c[valid])
    deltas = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    frame_gaps = np.diff(frames)
    adjacent = deltas[frame_gaps == 1]
    threshold = float(min_jump_m)
    if len(adjacent) > 0:
        threshold = max(
            threshold,
            float(p99_multiplier * np.percentile(adjacent, 99)),
            float(median_multiplier * np.median(adjacent)),
        )

    jump_mask = deltas > threshold
    if not bool(jump_mask.any()):
        max_idx = int(np.argmax(deltas))
        return LargePoseJumpReport(
            False,
            threshold,
            float(deltas[max_idx]),
            0,
            int(frames[max_idx]),
            int(frames[max_idx + 1]),
            int(frame_gaps[max_idx]),
        )

    jump_indices = np.flatnonzero(jump_mask)
    worst_idx = int(jump_indices[np.argmax(deltas[jump_indices])])
    return LargePoseJumpReport(
        True,
        threshold,
        float(deltas[worst_idx]),
        int(len(jump_indices)),
        int(frames[worst_idx]),
        int(frames[worst_idx + 1]),
        int(frame_gaps[worst_idx]),
    )


def filter_scenes_with_large_gt_jumps(dataset, scenes: list[str]) -> tuple[list[str], list[tuple[str, LargePoseJumpReport]]]:
    kept = []
    skipped = []
    for scene in scenes:
        scene_data = dataset.get_data(scene)
        report = large_finite_pose_jump_report(scene_data.extrinsics)
        if report.has_large_jump:
            skipped.append((scene, report))
        else:
            kept.append(scene)
    return kept, skipped
