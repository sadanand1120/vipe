#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


EPS = 1e-12


STAGES = {
    "backend": {"stage": "backend", "phase": "backend"},
    "infill": {"stage": "infill", "phase": "motion_only"},
    "frontend_warmup": {"stage": "frontend", "phase": "init"},
    "frontend_update_iters1": {"stage": "frontend", "phase": "update_iters1"},
    "frontend_update_iters2": {"stage": "frontend", "phase": "update_iters2"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ViPE BA trace loss-contribution curves.")
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def stage_rows(rows: list[dict[str, Any]], stage_name: str) -> list[dict[str, Any]]:
    spec = STAGES[stage_name]
    return [row for row in rows if row["stage"] == spec["stage"] and row["phase"] == spec["phase"]]


def instance_key(row: dict[str, Any], stage_name: str) -> str:
    if stage_name == "backend":
        return "backend"
    if stage_name == "frontend_warmup":
        return "warmup"
    if stage_name == "infill":
        return f"chunk:{row['chunk_idx']}"
    return f"frame:{row['frame_idx']}:kf:{row['kf_idx']}"


def group_rows_by_instance(rows: list[dict[str, Any]], stage_name: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[instance_key(row, stage_name)].append(row)
    return groups


def pct_list(values: list[float], total: float) -> list[float]:
    return [float(value / total * 100.0) if abs(total) > EPS else 0.0 for value in values]


def summarize_inner(instances: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    delta_by_ba_iter: dict[int, float] = defaultdict(float)
    group_pcts: dict[int, list[float]] = defaultdict(list)
    total_drop = 0.0
    groups_used = 0
    skipped_nonpositive_groups = 0
    per_group = []
    ba_iters_seen = set()

    for inst_id, inst_rows in sorted(instances.items()):
        outer_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in inst_rows:
            outer_groups[int(row["outer_iter"])].append(row)

        for outer_iter, group in sorted(outer_groups.items()):
            group.sort(key=lambda row: int(row["ba_iter"]))
            deltas = [float(row["loss_delta"]) for row in group]
            ba_iters = [int(row["ba_iter"]) for row in group]
            ba_iters_seen.update(ba_iters)
            group_drop = sum(deltas)
            if group_drop <= EPS:
                skipped_nonpositive_groups += 1
                continue
            groups_used += 1
            total_drop += group_drop
            contrib = pct_list(deltas, group_drop)
            for ba_iter, delta, pct in zip(ba_iters, deltas, contrib):
                delta_by_ba_iter[ba_iter] += delta
                group_pcts[ba_iter].append(pct)
            per_group.append(
                {
                    "instance": inst_id,
                    "outer_iter": outer_iter,
                    "total_drop": group_drop,
                    "pct_by_ba_iter": {
                        str(ba_iter): pct for ba_iter, pct in zip(ba_iters, contrib)
                    },
                }
            )

    cumulative = 0.0
    by_iter = []
    for ba_iter in sorted(ba_iters_seen):
        delta = delta_by_ba_iter[ba_iter]
        pct = float(delta / total_drop * 100.0) if total_drop > EPS else 0.0
        cumulative += pct
        pcts = group_pcts.get(ba_iter, [])
        by_iter.append(
            {
                "ba_iter": ba_iter,
                "delta_sum": delta,
                "pct": pct,
                "cumulative_pct": cumulative,
                "median_group_pct": float(median(pcts)) if pcts else 0.0,
            }
        )

    return {
        "total_drop": total_drop,
        "groups_used": groups_used,
        "skipped_nonpositive_groups": skipped_nonpositive_groups,
        "by_ba_iter": by_iter,
        "per_outer_group": per_group,
    }


def summarize_outer(instances: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    delta_by_outer_iter: dict[int, float] = defaultdict(float)
    instance_pcts: dict[int, list[float]] = defaultdict(list)
    total_drop = 0.0
    instances_used = 0
    skipped_nonpositive_instances = 0
    per_instance = []
    outer_iters_seen = set()

    for inst_id, inst_rows in sorted(instances.items()):
        outer_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in inst_rows:
            outer_groups[int(row["outer_iter"])].append(row)

        outer_drops = []
        outer_iters = []
        for outer_iter, group in sorted(outer_groups.items()):
            drop = sum(float(row["loss_delta"]) for row in group)
            outer_iters.append(outer_iter)
            outer_iters_seen.add(outer_iter)
            outer_drops.append(drop)

        inst_drop = sum(outer_drops)
        if inst_drop <= EPS:
            skipped_nonpositive_instances += 1
            continue

        instances_used += 1
        total_drop += inst_drop
        contrib = pct_list(outer_drops, inst_drop)
        for outer_iter, drop, pct in zip(outer_iters, outer_drops, contrib):
            delta_by_outer_iter[outer_iter] += drop
            instance_pcts[outer_iter].append(pct)
        per_instance.append(
            {
                "instance": inst_id,
                "total_drop": inst_drop,
                "pct_by_outer_iter": {
                    str(outer_iter): pct for outer_iter, pct in zip(outer_iters, contrib)
                },
            }
        )

    cumulative = 0.0
    by_iter = []
    for outer_iter in sorted(outer_iters_seen):
        delta = delta_by_outer_iter[outer_iter]
        pct = float(delta / total_drop * 100.0) if total_drop > EPS else 0.0
        cumulative += pct
        pcts = instance_pcts.get(outer_iter, [])
        by_iter.append(
            {
                "outer_iter": outer_iter,
                "delta_sum": delta,
                "pct": pct,
                "cumulative_pct": cumulative,
                "median_instance_pct": float(median(pcts)) if pcts else 0.0,
            }
        )

    return {
        "total_drop": total_drop,
        "instances_used": instances_used,
        "skipped_nonpositive_instances": skipped_nonpositive_instances,
        "by_outer_iter": by_iter,
        "per_instance": per_instance,
    }


def summarize_stage(rows: list[dict[str, Any]], stage_name: str) -> dict[str, Any]:
    selected = stage_rows(rows, stage_name)
    instances = group_rows_by_instance(selected, stage_name)
    return {
        "rows": len(selected),
        "instances": len(instances),
        "inner": summarize_inner(instances),
        "outer": summarize_outer(instances),
    }


def merge_aggregate_scene(stage_totals: dict[str, Any], scene: str, stage_name: str, stage_summary: dict[str, Any]) -> None:
    stage_total = stage_totals[stage_name]
    stage_total["scenes"].append(scene)
    stage_total["rows"] += int(stage_summary["rows"])
    stage_total["instances"] += int(stage_summary["instances"])

    inner = stage_summary["inner"]
    stage_total["inner_total_drop"] += float(inner["total_drop"])
    stage_total["inner_groups_used"] += int(inner["groups_used"])
    stage_total["inner_skipped_nonpositive_groups"] += int(inner["skipped_nonpositive_groups"])
    for item in inner["by_ba_iter"]:
        stage_total["inner_delta_by_ba_iter"][int(item["ba_iter"])] += float(item["delta_sum"])

    outer = stage_summary["outer"]
    stage_total["outer_total_drop"] += float(outer["total_drop"])
    stage_total["outer_instances_used"] += int(outer["instances_used"])
    stage_total["outer_skipped_nonpositive_instances"] += int(outer["skipped_nonpositive_instances"])
    for item in outer["by_outer_iter"]:
        stage_total["outer_delta_by_outer_iter"][int(item["outer_iter"])] += float(item["delta_sum"])


def finalize_aggregate(stage_totals: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for stage_name, total in stage_totals.items():
        inner_total = total["inner_total_drop"]
        inner_cumulative = 0.0
        inner_items = []
        for ba_iter, delta in sorted(total["inner_delta_by_ba_iter"].items()):
            pct = float(delta / inner_total * 100.0) if inner_total > EPS else 0.0
            inner_cumulative += pct
            inner_items.append(
                {
                    "ba_iter": ba_iter,
                    "delta_sum": delta,
                    "pct": pct,
                    "cumulative_pct": inner_cumulative,
                }
            )

        outer_total = total["outer_total_drop"]
        outer_cumulative = 0.0
        outer_items = []
        for outer_iter, delta in sorted(total["outer_delta_by_outer_iter"].items()):
            pct = float(delta / outer_total * 100.0) if outer_total > EPS else 0.0
            outer_cumulative += pct
            outer_items.append(
                {
                    "outer_iter": outer_iter,
                    "delta_sum": delta,
                    "pct": pct,
                    "cumulative_pct": outer_cumulative,
                }
            )

        out[stage_name] = {
            "scenes": sorted(total["scenes"]),
            "rows": total["rows"],
            "instances": total["instances"],
            "inner": {
                "total_drop": inner_total,
                "groups_used": total["inner_groups_used"],
                "skipped_nonpositive_groups": total["inner_skipped_nonpositive_groups"],
                "by_ba_iter": inner_items,
            },
            "outer": {
                "total_drop": outer_total,
                "instances_used": total["outer_instances_used"],
                "skipped_nonpositive_instances": total["outer_skipped_nonpositive_instances"],
                "by_outer_iter": outer_items,
            },
        }
    return out


def analyze_scene(trace_path: Path) -> dict[str, Any]:
    scene = trace_path.stem
    rows = read_jsonl(trace_path)
    return {
        "scene": scene,
        "trace_path": str(trace_path.resolve()),
        "stages": {
            stage_name: summarize_stage(rows, stage_name)
            for stage_name in STAGES
        },
    }


def main():
    args = parse_args()
    trace_paths = sorted((args.work_dir / "vipe_outputs").glob("*/ba_trace/*.jsonl"))
    stage_totals: dict[str, Any] = {
        stage_name: {
            "scenes": [],
            "rows": 0,
            "instances": 0,
            "inner_total_drop": 0.0,
            "inner_groups_used": 0,
            "inner_skipped_nonpositive_groups": 0,
            "inner_delta_by_ba_iter": defaultdict(float),
            "outer_total_drop": 0.0,
            "outer_instances_used": 0,
            "outer_skipped_nonpositive_instances": 0,
            "outer_delta_by_outer_iter": defaultdict(float),
        }
        for stage_name in STAGES
    }

    scene_summaries = {}
    for trace_path in trace_paths:
        scene_summary = analyze_scene(trace_path)
        scene = str(scene_summary["scene"])
        out_path = trace_path.with_name(f"{scene}_ba_contrib.json")
        out_path.write_text(json.dumps(scene_summary, indent=2) + "\n", encoding="utf-8")
        scene_summaries[scene] = {
            "trace_path": str(trace_path),
            "contrib_path": str(out_path),
        }
        for stage_name, stage_summary in scene_summary["stages"].items():
            merge_aggregate_scene(stage_totals, scene, stage_name, stage_summary)

    aggregate = {
        "work_dir": str(args.work_dir.resolve()),
        "num_scenes": len(trace_paths),
        "scene_outputs": scene_summaries,
        "stages": finalize_aggregate(stage_totals),
    }
    out_path = args.work_dir / "metric_results" / "ba_trace_contrib_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
