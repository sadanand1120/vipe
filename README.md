# ViPE: DAV3 Frame-Directory Fork

This fork keeps the single-camera frame-directory path for running ViPE SLAM with Depth-Anything-3 dense depth.

## Installation

Manual setup:

```bash
conda create -n vipe-manual -c nvidia/label/cuda-12.8.0 -c conda-forge python=3.10 pip cuda-nvcc eigen zlib libcusparse-dev libcublas-dev libcusolver-dev -y
conda activate vipe-manual
pip3 install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip3 install --no-build-isolation -e .
```

Depth-Anything-3 / `da3_streaming` benchmark support:

```bash
pip3 install faiss-gpu pandas prettytable numba pypose
pip3 install -e /robodata/smodak/repos/Depth-Anything-3
```

## Standalone Run

```bash
python run.py \
  streams.base_path=/path/to/scene/color \
  streams.fps=30 \
  pipeline.output.path=/path/to/output \
  pipeline.output.save_artifacts=true
```

Useful output knobs:

- `pipeline.output.pcd_fusion_mode=both`: save both point-cloud exports. This is the default.
- `pipeline.output.pcd_fusion_mode=backproject`: save `pcd/color_backproject.ply`.
- `pipeline.output.pcd_fusion_mode=tsdf`: save `pcd/color_tsdf.ply`.
- `pipeline.output.pcd_max_points=8000000`: cap saved point cloud points.
- `pipeline.depth.keyframe_model=depth-anything/DA3METRIC-LARGE`: DAV3 metric model used to anchor SLAM keyframe depth.
- `pipeline.depth.final_model=depth-anything/DA3-GIANT-1.1`: DAV3 posed model used for final per-frame dense depth.
- `pipeline.depth.window_size=10`: final DAV3 sliding-window size.
- `pipeline.depth.overlap_size=3`: final DAV3 overlap blended between windows.

Saved artifacts:

- `pose/color.npz`: camera-to-world pose per selected frame.
- `depth/color.zip`: per-frame final dense depth as NumPy `.npy` entries.
- `intrinsics/color.json`: one shared original-resolution pinhole intrinsics record.
- `pcd/color_backproject.ply`: direct backprojected point cloud, if enabled.
- `pcd/color_tsdf.ply`: TSDF-fused sampled point cloud, if enabled.

## ScanNet Benchmark

```bash
python3 scripts/scannet_vipe_bench_evaluator.py \
  --scenes scene0000_00 scene0011_00 scene0378_00 \
  --work-dir ./workspace/evaluation_scannet_vipe_dav3 \
  --input-root /robodata/smodak/repos/ovo/data/input/ScanNet \
  --raw-root /robodata/smodak/datasets/scannet_v2/scans \
  --max-frames -1 \
  --num-fusion-workers 16 \
  streams.fps=30
```

## Notes

The repo is intentionally configured through `configs/default.yaml`; `run.py` and the ScanNet benchmark both compose that config and instantiate `VipePipeline` directly.

Logging is centralized through `vipe.utils.logging.configure_logging()`. `run.py` and the ScanNet benchmark use the same ViPE logger setup, so SLAM/backend progress logs from `vipe.*` child modules are emitted consistently.

The standalone artifact path is intentionally lean but complete for benchmarking: RGB videos are not written, while pose, depth, intrinsics, and configured point clouds are written. The ScanNet benchmark reads those artifacts through a lightweight DA3-side ViPE manifest and reports reconstruction metrics for both TSDF and direct backprojection.
