#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TRACE = Path("workspace/.tmp_ba_trace_smoke/vipe_outputs/scene0013_01/ba_trace/scene0013_01.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ViPE BA trace JSONL logs.")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    return parser.parse_args()


def load_rows(trace_path: Path) -> list[dict]:
    with trace_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def add_outer_loop_lines(ax, boundary_positions):
    for x_val in sorted(set(boundary_positions)):
        ax.axvline(x_val, color="black", linestyle=":", linewidth=0.8, alpha=0.28)


def boundary_positions_for_rows(rows: list[dict]) -> list[float]:
    boundaries = []
    for idx in range(1, len(rows)):
        if rows[idx]["outer_iter"] != rows[idx - 1]["outer_iter"]:
            boundaries.append(idx)
    return boundaries


def save_backend(rows: list[dict], out_dir: Path):
    data = [row for row in rows if row["stage"] == "backend"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot([row["stage_cycle"] for row in data], [row["loss"] for row in data], lw=1.8, color="#0b6e4f")

    boundary_positions = []
    for idx in range(1, len(data)):
        if data[idx]["outer_iter"] != data[idx - 1]["outer_iter"]:
            boundary_positions.append(data[idx - 1]["stage_cycle"])
    add_outer_loop_lines(ax, boundary_positions)

    ax.set_title("Backend Global BA Loss")
    ax.set_xlabel("Backend solver cycle")
    ax.set_ylabel("Post-cycle BA loss")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "ba_trace_backend.png", dpi=180)
    plt.close(fig)


def grouped_frontend_rows(rows: list[dict], phase: str) -> dict[tuple[int, int], list[dict]]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row["stage"] == "frontend" and row["phase"] == phase and row.get("kf_idx") is not None:
            groups[(int(row["kf_idx"]), int(row["frame_idx"]))].append(row)
    return groups


def save_frontend_warmup(rows: list[dict], out_dir: Path):
    data = [row for row in rows if row["stage"] == "frontend" and row["phase"] == "init"]
    data.sort(key=lambda row: (row["outer_iter"], row["ba_iter"]))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(1, len(data) + 1), [row["loss"] for row in data], lw=1.8, color="#3b6ea8")
    add_outer_loop_lines(ax, boundary_positions_for_rows(data))

    ax.set_title("Frontend Warmup Init BA Loss")
    ax.set_xlabel("Solver cycle within warmup init")
    ax.set_ylabel("Post-cycle BA loss")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.99, 0.98, f"{len(data)} cycles", transform=ax.transAxes, ha="right", va="top")
    fig.tight_layout()
    fig.savefig(out_dir / "ba_trace_frontend_warmup.png", dpi=180)
    plt.close(fig)


def save_frontend_phase(rows: list[dict], out_dir: Path, phase: str, title: str, filename: str):
    groups = grouped_frontend_rows(rows, phase)

    fig, ax = plt.subplots(figsize=(14, 7))
    boundary_positions = []
    for _, data in sorted(groups.items()):
        data.sort(key=lambda row: (row["outer_iter"], row["ba_iter"]))
        x_vals = range(1, len(data) + 1)
        y_vals = [row["loss"] for row in data]
        ax.plot(x_vals, y_vals, lw=0.95, alpha=0.25)
        boundary_positions.extend(boundary_positions_for_rows(data))
    add_outer_loop_lines(ax, boundary_positions)

    ax.set_title(title)
    ax.set_xlabel("Solver cycle within that frontend block")
    ax.set_ylabel("Post-cycle BA loss")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.99, 0.98, f"{len(groups)} keyframe events", transform=ax.transAxes, ha="right", va="top")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def save_frontend(rows: list[dict], out_dir: Path):
    save_frontend_warmup(rows, out_dir)
    save_frontend_phase(
        rows,
        out_dir,
        "update_iters1",
        "Frontend Update Iters 1 BA Loss: One Curve Per Keyframe",
        "ba_trace_frontend_update_iters1.png",
    )
    save_frontend_phase(
        rows,
        out_dir,
        "update_iters2",
        "Frontend Update Iters 2 BA Loss: One Curve Per Kept Keyframe",
        "ba_trace_frontend_update_iters2.png",
    )


def save_infill(rows: list[dict], out_dir: Path):
    groups = defaultdict(list)
    for row in rows:
        if row["stage"] == "infill":
            groups[row["chunk_idx"]].append(row)

    fig, ax = plt.subplots(figsize=(14, 7))
    boundary_positions = []
    for _, data in sorted(groups.items()):
        data.sort(key=lambda row: (row["outer_iter"], row["ba_iter"]))
        x_vals = range(1, len(data) + 1)
        y_vals = [row["loss"] for row in data]
        ax.plot(x_vals, y_vals, lw=1.0, alpha=0.35)
        boundary_positions.extend(boundary_positions_for_rows(data))
    add_outer_loop_lines(ax, boundary_positions)

    ax.set_title("Infill Motion-Only BA Loss: One Curve Per Chunk")
    ax.set_xlabel("Solver cycle within chunk")
    ax.set_ylabel("Post-cycle BA loss")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.99, 0.98, f"{len(groups)} chunks", transform=ax.transAxes, ha="right", va="top")
    fig.tight_layout()
    fig.savefig(out_dir / "ba_trace_infill.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.trace)
    save_backend(rows, args.out_dir)
    save_frontend(rows, args.out_dir)
    save_infill(rows, args.out_dir)


if __name__ == "__main__":
    main()
