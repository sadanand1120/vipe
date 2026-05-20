#!/usr/bin/env python3
"""
Visualize the exact ScanNet reconstruction metric inputs.

For each fused prediction PLY, this inspects the DA3 ScanNet eval path:
GT mesh -> sampled GT point cloud -> AABB-crop predicted cloud -> nearest-neighbor squared distances.
It also projects the fused point cloud into every ScanNet RGB view with a 0.1cm voxel z-buffer
and reports mean image PSNR/SSIM.

Outputs:
- pred_eval.ply: prediction points actually used for metric computation, with original RGB.
- gt_eval.ply: GT sampled mesh points actually used for metric computation, with original RGB.
- pred_accuracy_colored.ply: prediction eval points colored by pred->GT distance.
- gt_completion_colored.ply: GT eval points colored by GT->pred distance.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

from pathlib import Path
from typing import Any

import cv2
import open3d as o3d
import numpy as np

from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm


DEFAULT_DA3_ROOT = Path("/robodata/smodak/repos/Depth-Anything-3")
DEFAULT_INPUT_ROOT = Path("/robodata/smodak/repos/ovo/data/input/ScanNet")
DEFAULT_RAW_ROOT = Path("/robodata/smodak/datasets/scannet_v2/scans")
_RENDER_POINTS = None
_RENDER_COLORS = None
_RENDER_VOXEL_SIZE = None
_RENDER_CHUNK_SIZE = None
_RENDER_RADIUS_CAP = None


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Color ScanNet fused PLYs by reconstruction metric errors."
    )
    parser.add_argument("workspaces", nargs="+", type=Path, help="Benchmark workspace root(s).")
    parser.add_argument("--scene", default="scene0000_00", help="ScanNet scene name.")
    parser.add_argument("--mode", default="unposed", choices=["unposed", "posed", "recon_unposed", "recon_posed"])
    parser.add_argument("--method", default="backproject", choices=["backproject", "tsdf"])
    parser.add_argument("--da3-root", default=DEFAULT_DA3_ROOT, type=Path)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, type=Path)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, type=Path)
    parser.add_argument("--aabb-margin", default=0.1, type=float, help="GT AABB margin used to crop predicted points.")
    parser.add_argument("--gt-sample-points", default=10_000_000, type=int)
    parser.add_argument("--render-voxel-size", default=0.001, type=float, help="Voxel size for all-view image projection in meters.")
    parser.add_argument("--render-chunk-size", default=1_000_000, type=int, help="Projection chunk size in voxels.")
    parser.add_argument("--render-radius-cap", default=8, type=int, help="Max pixel radius for one projected voxel.")
    parser.add_argument("--render-workers", default=16, type=int, help="Parallel frame render workers.")
    parser.add_argument("--render-num-images", default=-1, type=int, help="-1 renders all images; otherwise sample this many images.")
    parser.add_argument("--no-render-metrics", action="store_true", help="Skip all-view PSNR/SSIM projection.")
    parser.add_argument("--write-rejected", action="store_true", help="Also write predicted points ignored by AABB crop.")
    parser.add_argument("--no-write", action="store_true", help="Only print metrics; do not write colored PLYs.")
    parser.add_argument(
        "--out-subdir",
        default="metric_debug",
        help="Output subdir under each workspace.",
    )
    return parser.parse_args()


def setup_da3(args: argparse.Namespace):
    start = time.perf_counter()
    log(f"Importing DA3 benchmark code from {args.da3_root}")
    os.environ["DA3_SCANNET_INPUT_ROOT"] = str(args.input_root)
    os.environ["DA3_SCANNET_RAW_ROOT"] = str(args.raw_root)
    da3_src = args.da3_root / "src"
    if da3_src.is_dir() and str(da3_src) not in sys.path:
        sys.path.insert(0, str(da3_src))

    from depth_anything_3.bench.datasets.scannet import ScanNetDataset
    from depth_anything_3.bench.utils import sample_points_from_mesh

    log(f"Imported DA3 benchmark code in {time.perf_counter() - start:.2f}s")
    return ScanNetDataset, sample_points_from_mesh


def normalize_mode(mode: str) -> str:
    if mode == "recon_unposed":
        return "unposed"
    if mode == "recon_posed":
        return "posed"
    return mode


def fuse_path(workspace: Path, scene: str, mode: str, method: str) -> Path:
    return workspace / "model_results" / "scannet" / scene / mode / "exports" / "fuse" / f"pcd_{method}.ply"


def metric_json_path(workspace: Path, mode: str, method: str) -> Path:
    recon_mode = f"recon_{mode}"
    return workspace / "metric_results" / f"scannet_{recon_mode}_{method}.json"


def point_cloud_from_points(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64, copy=False))
    return pcd


def colors_from_distances(dist: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(dist), 3), dtype=np.float32)
    if len(dist) == 0:
        return colors

    scale = max(float(np.percentile(dist, 95)), 1e-12)
    t = np.clip(dist / scale, 0.0, 1.0).astype(np.float32)
    first_half = t <= 0.5

    colors[first_half, 0] = 2.0 * t[first_half]
    colors[first_half, 1] = 1.0
    colors[first_half, 2] = 0.0

    second_half = ~first_half
    colors[second_half, 0] = 1.0
    colors[second_half, 1] = 2.0 * (1.0 - t[second_half])
    colors[second_half, 2] = 0.0
    return colors


def distance_stats(dist: np.ndarray) -> dict[str, float]:
    if len(dist) == 0:
        return {key: float("inf") for key in ["mean", "p50", "p90", "p95", "p99", "max"]}
    return {
        "mean": float(np.mean(dist)),
        "p50": float(np.percentile(dist, 50)),
        "p90": float(np.percentile(dist, 90)),
        "p95": float(np.percentile(dist, 95)),
        "p99": float(np.percentile(dist, 99)),
        "max": float(np.max(dist)),
    }


def nearest_neighbor_distances(pred_points: np.ndarray, gt_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist_pred_to_gt, _ = KDTree(gt_points).query(pred_points, workers=-1)
    dist_gt_to_pred, _ = KDTree(pred_points).query(gt_points, workers=-1)
    return dist_pred_to_gt.reshape(-1), dist_gt_to_pred.reshape(-1)


def compute_eval(
    pred_pcd: o3d.geometry.PointCloud,
    gt_pcd: o3d.geometry.PointCloud,
    aabb_margin: float,
) -> dict[str, Any]:
    start_total = time.perf_counter()
    log("Geometry eval: AABB crop start")
    pred_points_raw = np.asarray(pred_pcd.points)
    gt_points_raw = np.asarray(gt_pcd.points)
    if len(pred_points_raw) == 0 or len(gt_points_raw) == 0:
        raise ValueError("Empty predicted or GT point cloud")

    aabb = gt_pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound - aabb_margin
    max_bound = aabb.max_bound + aabb_margin
    inside = np.all((pred_points_raw >= min_bound) & (pred_points_raw <= max_bound), axis=1)

    pred_inside = pred_pcd.select_by_index(np.nonzero(inside)[0])
    pred_rejected = pred_pcd.select_by_index(np.nonzero(~inside)[0])
    pred_eval = pred_inside
    gt_eval = gt_pcd

    pred_points = np.asarray(pred_eval.points)
    gt_points = np.asarray(gt_eval.points)
    if len(pred_points) == 0 or len(gt_points) == 0:
        raise ValueError("Empty eval point cloud after AABB crop")
    log(
        "Geometry eval: AABB crop done | "
        f"pred_raw={len(pred_points_raw):,} pred_eval={len(pred_points):,} gt_eval={len(gt_points):,} | "
        f"{time.perf_counter() - start_total:.2f}s"
    )

    start = time.perf_counter()
    log("Geometry eval: nearest-neighbor queries start")
    dist_pred_to_gt, dist_gt_to_pred = nearest_neighbor_distances(pred_points, gt_points)
    log(f"Geometry eval: nearest-neighbor queries done | {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    log("Geometry eval: metric reductions start")
    dist_pred_to_gt_l2 = dist_pred_to_gt * dist_pred_to_gt
    dist_gt_to_pred_l2 = dist_gt_to_pred * dist_gt_to_pred
    accuracy = float(np.mean(dist_pred_to_gt_l2))
    completeness = float(np.mean(dist_gt_to_pred_l2))
    log(
        "Geometry eval: metric reductions done | "
        f"total={time.perf_counter() - start_total:.2f}s reductions={time.perf_counter() - start:.2f}s"
    )

    return {
        "metrics": {
            "acc": accuracy,
            "comp": completeness,
            "overall": (accuracy + completeness) / 2.0,
        },
        "counts": {
            "pred_raw": int(len(pred_points_raw)),
            "pred_inside_aabb": int(inside.sum()),
            "pred_rejected_aabb": int((~inside).sum()),
            "pred_eval": int(len(pred_points)),
            "gt_sampled_raw": int(len(gt_points_raw)),
            "gt_eval": int(len(gt_points)),
        },
        "dist_pred_to_gt": dist_pred_to_gt,
        "dist_gt_to_pred": dist_gt_to_pred,
        "pred_eval": pred_eval,
        "gt_eval": gt_eval,
        "pred_rejected": pred_rejected,
        "pred_distance_stats": distance_stats(dist_pred_to_gt),
        "gt_distance_stats": distance_stats(dist_gt_to_pred),
        "pred_squared_distance_stats": distance_stats(dist_pred_to_gt_l2),
        "gt_squared_distance_stats": distance_stats(dist_gt_to_pred_l2),
    }


def write_outputs(out_dir: Path, result: dict[str, Any], write_rejected: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    log(f"Writing pred_eval.ply | {out_dir / 'pred_eval.ply'}")
    o3d.io.write_point_cloud(str(out_dir / "pred_eval.ply"), result["pred_eval"])
    log(f"Wrote pred_eval.ply in {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    log(f"Writing gt_eval.ply | {out_dir / 'gt_eval.ply'}")
    o3d.io.write_point_cloud(str(out_dir / "gt_eval.ply"), result["gt_eval"])
    log(f"Wrote gt_eval.ply in {time.perf_counter() - start:.2f}s")

    pred_points = np.asarray(result["pred_eval"].points)
    pred_colors = colors_from_distances(result["dist_pred_to_gt"])
    start = time.perf_counter()
    log(f"Writing pred_accuracy_colored.ply | {out_dir / 'pred_accuracy_colored.ply'}")
    o3d.io.write_point_cloud(str(out_dir / "pred_accuracy_colored.ply"), point_cloud_from_points(pred_points, pred_colors))
    log(f"Wrote pred_accuracy_colored.ply in {time.perf_counter() - start:.2f}s")

    gt_points = np.asarray(result["gt_eval"].points)
    gt_colors = colors_from_distances(result["dist_gt_to_pred"])
    start = time.perf_counter()
    log(f"Writing gt_completion_colored.ply | {out_dir / 'gt_completion_colored.ply'}")
    o3d.io.write_point_cloud(str(out_dir / "gt_completion_colored.ply"), point_cloud_from_points(gt_points, gt_colors))
    log(f"Wrote gt_completion_colored.ply in {time.perf_counter() - start:.2f}s")

    if write_rejected and len(result["pred_rejected"].points) > 0:
        rejected_points = np.asarray(result["pred_rejected"].points)
        rejected_colors = np.tile(np.array([[0.25, 0.25, 1.0]], dtype=np.float32), (len(rejected_points), 1))
        o3d.io.write_point_cloud(
            str(out_dir / "pred_aabb_rejected_not_evaluated.ply"),
            point_cloud_from_points(rejected_points, rejected_colors),
        )
        log(f"Wrote pred_aabb_rejected_not_evaluated.ply | {out_dir / 'pred_aabb_rejected_not_evaluated.ply'}")


def image_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def image_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    scores = []
    for channel in range(3):
        x = pred[:, :, channel].astype(np.float32, copy=False)
        y = target[:, :, channel].astype(np.float32, copy=False)

        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        scores.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return float(np.mean(scores))


def voxelized_render_arrays(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    render_pcd = pcd.voxel_down_sample(voxel_size) if voxel_size > 0.0 else pcd
    points = np.asarray(render_pcd.points).astype(np.float32, copy=False)
    colors = np.asarray(render_pcd.colors).astype(np.float32, copy=False)
    if len(colors) != len(points):
        colors = np.zeros((len(points), 3), dtype=np.float32)
    return points, np.clip(colors, 0.0, 1.0)


def render_voxel_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    voxel_size: float,
    chunk_size: int,
    radius_cap: int,
) -> np.ndarray:
    zbuf = np.full(height * width, np.inf, dtype=np.float32)
    image = np.zeros((height * width, 3), dtype=np.float32)

    r = extrinsic[:3, :3].astype(np.float32, copy=False)
    t = extrinsic[:3, 3].astype(np.float32, copy=False)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    radius_cap = max(int(radius_cap), 0)
    offsets = [
        (dx, dy, max(abs(dx), abs(dy)))
        for dy in range(-radius_cap, radius_cap + 1)
        for dx in range(-radius_cap, radius_cap + 1)
    ]
    max_focal = max(abs(fx), abs(fy))
    chunk_size = max(int(chunk_size), 1)

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        cam = points[start:end] @ r.T + t
        z = cam[:, 2]
        valid = z > 1e-4
        if not np.any(valid):
            continue

        cam = cam[valid]
        z = z[valid]
        cols = colors[start:end][valid]
        u = np.rint(fx * cam[:, 0] / z + cx).astype(np.int32)
        v = np.rint(fy * cam[:, 1] / z + cy).astype(np.int32)
        radii = np.ceil(0.5 * voxel_size * max_focal / z).astype(np.int32) if voxel_size > 0.0 else np.zeros_like(u)
        radii = np.clip(radii, 0, radius_cap)

        margin = radius_cap
        near_image = (u >= -margin) & (u < width + margin) & (v >= -margin) & (v < height + margin)
        if not np.any(near_image):
            continue

        u = u[near_image]
        v = v[near_image]
        z = z[near_image]
        cols = cols[near_image]
        radii = radii[near_image]

        for dx, dy, offset_radius in offsets:
            if offset_radius > 0:
                active = radii >= offset_radius
                if not np.any(active):
                    continue
                uu = u[active] + dx
                vv = v[active] + dy
                zz = z[active]
                cc = cols[active]
            else:
                uu = u + dx
                vv = v + dy
                zz = z
                cc = cols

            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            if not np.any(inside):
                continue

            pix = vv[inside] * width + uu[inside]
            pix_z = zz[inside]
            pix_colors = cc[inside]
            np.minimum.at(zbuf, pix, pix_z)
            winners = np.abs(pix_z - zbuf[pix]) <= 1e-6
            if np.any(winners):
                image[pix[winners]] = pix_colors[winners]

    return image.reshape(height, width, 3)


def init_render_worker(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    chunk_size: int,
    radius_cap: int,
) -> None:
    global _RENDER_POINTS, _RENDER_COLORS, _RENDER_VOXEL_SIZE, _RENDER_CHUNK_SIZE, _RENDER_RADIUS_CAP
    cv2.setNumThreads(0)
    _RENDER_POINTS = points
    _RENDER_COLORS = colors
    _RENDER_VOXEL_SIZE = voxel_size
    _RENDER_CHUNK_SIZE = chunk_size
    _RENDER_RADIUS_CAP = radius_cap


def render_metric_task(task) -> tuple[int, float, float]:
    idx, image_file, extrinsic, intrinsic = task
    bgr = cv2.imread(image_file, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_file)
    target = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    height, width = target.shape[:2]
    render = render_voxel_cloud(
        _RENDER_POINTS,
        _RENDER_COLORS,
        extrinsic,
        intrinsic,
        height,
        width,
        _RENDER_VOXEL_SIZE,
        _RENDER_CHUNK_SIZE,
        _RENDER_RADIUS_CAP,
    )
    return idx, image_psnr(render, target), image_ssim(render, target)


def compute_render_metrics(
    label: str,
    pcd: o3d.geometry.PointCloud,
    scene_data,
    voxel_size: float,
    chunk_size: int,
    radius_cap: int,
    workers: int,
    num_images: int,
) -> dict[str, Any]:
    start_total = time.perf_counter()
    log(f"Render metrics: voxelizing cloud start | {label} | voxel={voxel_size}m")
    points, colors = voxelized_render_arrays(pcd, voxel_size)
    log(
        "Render metrics: voxelizing cloud done | "
        f"{label} | voxels={len(points):,} | {time.perf_counter() - start_total:.2f}s"
    )
    psnr_values = []
    ssim_values = []

    frame_indices = np.arange(len(scene_data.image_files))
    if num_images != -1:
        if num_images <= 0:
            raise ValueError("--render-num-images must be -1 or a positive integer")
        sample_count = min(num_images, len(frame_indices))
        rng = np.random.default_rng(seed=42)
        frame_indices = np.sort(rng.choice(frame_indices, size=sample_count, replace=False))

    tasks = [
        (
            int(idx),
            str(scene_data.image_files[idx]),
            scene_data.extrinsics[idx],
            scene_data.intrinsics[idx],
        )
        for idx in frame_indices
    ]
    workers = min(max(int(workers), 1), len(tasks))
    log(
        "Render metrics: projection start | "
        f"{label} | frames={len(tasks):,}/{len(scene_data.image_files):,} workers={workers}"
    )
    start_render = time.perf_counter()
    iterator = tqdm(total=len(tasks), desc=f"{label} render PSNR/SSIM", unit="frame")
    if workers == 1:
        init_render_worker(points, colors, voxel_size, chunk_size, radius_cap)
        for task in tasks:
            _, psnr, ssim = render_metric_task(task)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            iterator.update()
            iterator.set_postfix(psnr=f"{np.mean(psnr_values):.2f}", ssim=f"{np.mean(ssim_values):.4f}")
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=workers,
            initializer=init_render_worker,
            initargs=(points, colors, voxel_size, chunk_size, radius_cap),
        ) as pool:
            for _, psnr, ssim in pool.imap_unordered(render_metric_task, tasks, chunksize=1):
                psnr_values.append(psnr)
                ssim_values.append(ssim)
                iterator.update()
                iterator.set_postfix(psnr=f"{np.mean(psnr_values):.2f}", ssim=f"{np.mean(ssim_values):.4f}")
    iterator.close()
    log(
        "Render metrics: projection done | "
        f"{label} | render={time.perf_counter() - start_render:.2f}s total={time.perf_counter() - start_total:.2f}s"
    )

    return {
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "num_images": int(len(psnr_values)),
        "total_scene_images": int(len(scene_data.image_files)),
        "voxel_size": float(voxel_size),
        "voxel_points": int(len(points)),
        "workers": int(workers),
    }


def load_saved_metric(workspace: Path, scene: str, mode: str, method: str) -> dict[str, float] | None:
    path = metric_json_path(workspace, mode, method)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    scene_metrics = data.get(scene)
    return scene_metrics if isinstance(scene_metrics, dict) else None


def print_row(label: str, result: dict[str, Any], saved: dict[str, float] | None) -> None:
    m = result["metrics"]
    c = result["counts"]
    rej_pct = 100.0 * c["pred_rejected_aabb"] / max(c["pred_raw"], 1)
    saved_text = ""
    if saved is not None:
        saved_text = f" | saved_json_l1_ov={saved.get('overall', float('nan')):.4f}"
    render_text = ""
    if "render_metrics" in result:
        r = result["render_metrics"]
        render_text = f" psnr={r['psnr']:.2f} ssim={r['ssim']:.4f}"
    print(
        f"{label}: ov_l2={m['overall']:.6f} "
        f"acc_l2={m['acc']:.6f} comp_l2={m['comp']:.6f}{render_text} | "
        f"pts raw={c['pred_raw']:,} eval={c['pred_eval']:,} gt_eval={c['gt_eval']:,} "
        f"aabb_reject={rej_pct:.2f}%{saved_text}",
        flush=True,
    )


def main() -> None:
    start_all = time.perf_counter()
    args = parse_args()
    mode = normalize_mode(args.mode)
    ScanNetDataset, sample_points_from_mesh = setup_da3(args)

    start = time.perf_counter()
    log(f"Loading ScanNet scene metadata | scene={args.scene}")
    dataset = ScanNetDataset()
    gt_data = dataset.get_data(args.scene)
    log(f"Loaded ScanNet scene metadata | frames={len(gt_data.image_files):,} | {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    log(f"Reading GT mesh | {gt_data.aux.gt_mesh_path}")
    gt_mesh = o3d.io.read_triangle_mesh(gt_data.aux.gt_mesh_path)
    log(f"Read GT mesh in {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    log(f"Sampling GT point cloud | points={args.gt_sample_points:,}")
    gt_pcd = sample_points_from_mesh(gt_mesh, args.gt_sample_points)
    log(f"Sampled GT point cloud | points={len(gt_pcd.points):,} | {time.perf_counter() - start:.2f}s")

    summaries = {}
    for workspace in args.workspaces:
        start_workspace = time.perf_counter()
        log(f"Workspace start | {workspace}")
        path = fuse_path(workspace, args.scene, mode, args.method)
        if not path.exists():
            raise FileNotFoundError(path)

        start = time.perf_counter()
        log(f"Reading prediction PLY | {path}")
        pred_pcd = o3d.io.read_point_cloud(str(path))
        log(f"Read prediction PLY | points={len(pred_pcd.points):,} | {time.perf_counter() - start:.2f}s")

        result = compute_eval(
            pred_pcd,
            gt_pcd,
            aabb_margin=args.aabb_margin,
        )

        label = workspace.name
        if not args.no_render_metrics:
            result["render_metrics"] = compute_render_metrics(
                label,
                pred_pcd,
                gt_data,
                args.render_voxel_size,
                args.render_chunk_size,
                args.render_radius_cap,
                args.render_workers,
                args.render_num_images,
            )

        saved = load_saved_metric(workspace, args.scene, mode, args.method)
        print_row(label, result, saved)

        out_dir = workspace / args.out_subdir / args.scene / mode / args.method
        summary = {
            "workspace": str(workspace.resolve()),
            "scene": args.scene,
            "mode": mode,
            "method": args.method,
            "fuse_path": str(path.resolve()),
            "aabb_margin": args.aabb_margin,
            "render_metrics_enabled": not args.no_render_metrics,
            "metrics": result["metrics"],
            "render_metrics": result.get("render_metrics"),
            "saved_json_metrics": saved,
            "counts": result["counts"],
            "pred_distance_stats": result["pred_distance_stats"],
            "gt_distance_stats": result["gt_distance_stats"],
            "pred_squared_distance_stats": result["pred_squared_distance_stats"],
            "gt_squared_distance_stats": result["gt_squared_distance_stats"],
            "outputs": {
                "pred_eval": str((out_dir / "pred_eval.ply").resolve()),
                "gt_eval": str((out_dir / "gt_eval.ply").resolve()),
                "pred_accuracy_colored": str((out_dir / "pred_accuracy_colored.ply").resolve()),
                "gt_completion_colored": str((out_dir / "gt_completion_colored.ply").resolve()),
            },
        }
        summaries[label] = summary

        if not args.no_write:
            start = time.perf_counter()
            log(f"Writing analyzer outputs | {out_dir}")
            write_outputs(out_dir, result, args.write_rejected)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            log(f"Wrote analyzer outputs and summary.json in {time.perf_counter() - start:.2f}s")

        log(f"Workspace done | {workspace} | {time.perf_counter() - start_workspace:.2f}s")

    if len(summaries) > 1:
        print("\nInterpretation: lower ov_l2 is better; higher PSNR/SSIM is better.")
        print("pred_accuracy_colored and gt_completion_colored are continuous nearest-neighbor distance visualizations.")
    log(f"Analyzer done | total={time.perf_counter() - start_all:.2f}s")


if __name__ == "__main__":
    main()
