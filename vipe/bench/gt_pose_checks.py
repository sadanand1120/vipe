from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GTPoseQualityReport:
    should_skip: bool
    skip_reasons: tuple[str, ...]
    finite_coverage: float
    valid_run_count: int
    invalid_run_count: int
    invalid_frame_count: int
    threshold_m: float
    max_jump_m: float
    jump_count: int
    worst_from_frame: int | None
    worst_to_frame: int | None
    worst_gap: int | None

    def summary(self) -> str:
        reasons = ",".join(self.skip_reasons) if self.skip_reasons else "ok"
        return (
            f"{reasons} "
            f"finite_cov={self.finite_coverage:.3f} "
            f"valid_runs={self.valid_run_count} "
            f"invalid_runs={self.invalid_run_count} "
            f"invalid_frames={self.invalid_frame_count} "
            f"max_jump={self.max_jump_m:.3f}m "
            f"jump_thresh={self.threshold_m:.3f}m"
        )


def _run_lengths(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    runs = []
    i = 0
    while i < len(mask):
        j = i + 1
        while j < len(mask) and bool(mask[j]) == bool(mask[i]):
            j += 1
        if bool(mask[i]) == value:
            runs.append((i, j))
        i = j
    return runs


def gt_pose_quality_report(
    extrinsics_w2c: np.ndarray,
    *,
    frame_indices: list[int] | np.ndarray | None = None,
    min_jump_m: float = 0.35,
    p99_multiplier: float = 8.0,
    median_multiplier: float = 25.0,
    min_finite_coverage: float = 0.80,
    max_valid_runs: int = 8,
    max_invalid_runs: int = 8,
) -> GTPoseQualityReport:
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float64)
    valid = np.isfinite(extrinsics_w2c).all(axis=(1, 2))
    valid_runs = _run_lengths(valid, True)
    invalid_runs = _run_lengths(valid, False)
    finite_coverage = float(valid.mean()) if len(valid) > 0 else 0.0
    invalid_frame_count = int((~valid).sum())

    skip_reasons = []
    if finite_coverage < min_finite_coverage:
        skip_reasons.append("low_finite_coverage")
    if len(valid_runs) > max_valid_runs:
        skip_reasons.append("fragmented_valid_runs")
    if len(invalid_runs) > max_invalid_runs:
        skip_reasons.append("fragmented_invalid_runs")

    valid_rows = np.flatnonzero(valid)
    if len(valid_rows) < 2:
        skip_reasons.append("too_few_finite_poses")
        return GTPoseQualityReport(
            True,
            tuple(skip_reasons),
            finite_coverage,
            len(valid_runs),
            len(invalid_runs),
            invalid_frame_count,
            float(min_jump_m),
            0.0,
            0,
            None,
            None,
            None,
        )

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
        return GTPoseQualityReport(
            bool(skip_reasons),
            tuple(skip_reasons),
            finite_coverage,
            len(valid_runs),
            len(invalid_runs),
            invalid_frame_count,
            threshold,
            float(deltas[max_idx]),
            0,
            int(frames[max_idx]),
            int(frames[max_idx + 1]),
            int(frame_gaps[max_idx]),
        )

    jump_indices = np.flatnonzero(jump_mask)
    worst_idx = int(jump_indices[np.argmax(deltas[jump_indices])])
    skip_reasons.append("large_finite_jump")
    return GTPoseQualityReport(
        True,
        tuple(skip_reasons),
        finite_coverage,
        len(valid_runs),
        len(invalid_runs),
        invalid_frame_count,
        threshold,
        float(deltas[worst_idx]),
        int(len(jump_indices)),
        int(frames[worst_idx]),
        int(frames[worst_idx + 1]),
        int(frame_gaps[worst_idx]),
    )


def filter_scenes_with_bad_gt_pose_data(dataset, scenes: list[str]) -> tuple[list[str], list[tuple[str, GTPoseQualityReport]]]:
    kept = []
    skipped = []
    for scene in scenes:
        scene_data = dataset.get_data(scene)
        report = gt_pose_quality_report(scene_data.extrinsics)
        if report.should_skip:
            skipped.append((scene, report))
        else:
            kept.append(scene)
    return kept, skipped
