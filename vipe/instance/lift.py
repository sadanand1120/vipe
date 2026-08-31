"""Occlusion-correct projection and 2D-mask lifting."""

from collections.abc import Callable, Sequence

import numpy as np
import torch


@torch.no_grad()
def visible_points(
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
    atom_of: np.ndarray,
    adjacency: tuple[np.ndarray, np.ndarray],
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
) -> tuple[dict, dict[int, np.ndarray]]:
    """Lift masks into sparse atom evidence and accumulate adjacent-atom affinity."""
    atom_of = np.asarray(atom_of, np.int64)
    atom_count = int(atom_of.max()) + 1
    adjacent_a, adjacent_b = (np.asarray(values, np.int64) for values in adjacency)
    edge_num = np.zeros(len(adjacent_a), np.float64)
    edge_weight = np.zeros(len(adjacent_a), np.int32)
    visible_flag = np.zeros(atom_count, bool)
    masked_flag = np.zeros(atom_count, bool)
    i_atom, i_mask, i_count = [], [], []
    n_atom, n_frame, n_count = [], [], []
    gm_frame, gm_size, gm_track_id = [], [], []
    lifted_frame_count = 0
    track_unions: dict[int, np.ndarray] = {}

    for position, frame_index in enumerate(frames, 1):
        if position % 100 == 0 or position == len(frames):
            log(f"  [instance] lift {position}/{len(frames)} frames={lifted_frame_count}")
        indices, u, v = visible_points(
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
        visible = indices.astype(np.int32)
        visible_atoms, visible_counts = np.unique(atom_of[visible], return_counts=True)
        retained_frame = lifted_frame_count
        retained_masks = 0
        frame_atom, frame_mask, frame_count = [], [], []
        for mask, _, track_id, _ in masks_of(frame_index):
            selected = mask[v, u]
            if not selected.any():
                continue
            voxels = visible[selected]
            if voxels.size >= min_voxels:
                global_mask = len(gm_frame)
                atoms, counts = np.unique(atom_of[voxels], return_counts=True)
                i_atom.append(atoms.astype(np.int32))
                i_mask.append(np.full(len(atoms), global_mask, np.int32))
                i_count.append(counts.astype(np.int32))
                frame_atom.append(atoms.astype(np.int32))
                frame_mask.append(np.full(len(atoms), retained_masks, np.int32))
                frame_count.append(counts.astype(np.int32))
                gm_frame.append(retained_frame)
                gm_size.append(int(voxels.size))
                gm_track_id.append(int(track_id))
                previous = track_unions.get(int(track_id))
                track_unions[int(track_id)] = (
                    voxels.copy() if previous is None else np.union1d(previous, voxels)
                )
                retained_masks += 1
        if retained_masks:
            n_atom.append(visible_atoms.astype(np.int32))
            n_frame.append(np.full(len(visible_atoms), retained_frame, np.int32))
            n_count.append(visible_counts.astype(np.int32))

            rows = np.concatenate(frame_atom)
            columns = np.concatenate(frame_mask)
            counts = np.concatenate(frame_count)
            order = np.argsort(rows, kind="stable")
            rows, columns, counts = rows[order], columns[order], counts[order]
            starts = np.r_[True, rows[1:] != rows[:-1]]
            groups = np.cumsum(starts) - 1
            visible_by_atom = np.zeros(atom_count, np.int32)
            visible_by_atom[visible_atoms] = visible_counts
            values = counts / visible_by_atom[rows]
            norms = np.sqrt(np.bincount(groups, weights=values * values))
            maximum = np.maximum.reduceat(values, np.flatnonzero(starts))
            values *= np.sqrt(np.minimum(maximum, 1.0))[groups] / norms[groups]

            from scipy.sparse import csr_matrix

            signature = csr_matrix(
                (values, (rows, columns)), shape=(atom_count, retained_masks)
            )
            masked_atoms = rows[starts]
            visible_flag[visible_atoms] = True
            masked_flag[masked_atoms] = True
            opportunities = (
                visible_flag[adjacent_a]
                & visible_flag[adjacent_b]
                & (masked_flag[adjacent_a] | masked_flag[adjacent_b])
            )
            edge_weight[opportunities] += 1
            agreements = np.flatnonzero(
                masked_flag[adjacent_a] & masked_flag[adjacent_b]
            )
            edge_num[agreements] += np.asarray(
                signature[adjacent_a[agreements]]
                .multiply(signature[adjacent_b[agreements]])
                .sum(axis=1)
            ).ravel()
            visible_flag[visible_atoms] = False
            masked_flag[masked_atoms] = False
            lifted_frame_count += 1

    def _csr(rows, keys, counts):
        if not rows:
            return (
                np.zeros(atom_count + 1, np.int64),
                np.empty(0, np.int32),
                np.empty(0, np.int32),
            )
        row = np.concatenate(rows)
        key = np.concatenate(keys)
        count = np.concatenate(counts)
        order = np.argsort(row, kind="stable")
        ptr = np.zeros(atom_count + 1, np.int64)
        np.cumsum(np.bincount(row, minlength=atom_count), out=ptr[1:])
        return ptr, key[order], count[order]

    leafI_ptr, leafI_gm, leafI_c = _csr(i_atom, i_mask, i_count)
    leafN_ptr, leafN_f, leafN_c = _csr(n_atom, n_frame, n_count)
    positive_edges = np.flatnonzero((edge_num > 0.0) & (edge_weight > 0))
    evidence = {
        "n_frames": lifted_frame_count,
        "gm_frame": np.asarray(gm_frame, np.int64),
        "gm_size": np.asarray(gm_size, np.int64),
        "gm_track_id": np.asarray(gm_track_id, np.int64),
        "leafI_ptr": leafI_ptr,
        "leafI_gm": leafI_gm,
        "leafI_c": leafI_c,
        "leafN_ptr": leafN_ptr,
        "leafN_f": leafN_f,
        "leafN_c": leafN_c,
        "affinity_edge": positive_edges.astype(np.int64),
        "affinity_num": edge_num[positive_edges],
        "affinity_weight": edge_weight[positive_edges].astype(np.float64),
    }
    return evidence, track_unions
