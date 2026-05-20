# ViPE: External-Depth Frame-Directory Fork

This fork keeps one supported runtime path: a single RGB frame directory with external RGB/color intrinsics and external sensor depth. ViPE estimates poses with the DROID/ViPE SLAM stack, uses the provided depth as the dense depth source, and writes pose/depth/intrinsics plus configured point-cloud exports.

## Installation

Manual setup:

```bash
conda create -n vipe-manual -c nvidia/label/cuda-12.8.0 -c conda-forge python=3.10 pip cuda-nvcc eigen zlib libcusparse-dev libcublas-dev libcusolver-dev -y
conda activate vipe-manual
pip3 install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip3 install --no-build-isolation -e .
```

## Input Layout

`streams.base_path` must point to the RGB directory. The sibling `depth/` and `intrinsic/` directories are required:

```text
<scene>/color/<frame_id>.jpg|png
<scene>/depth/<frame_id>.png
<scene>/intrinsic/intrinsic_color.json  # preferred when present
<scene>/intrinsic/intrinsic_color.txt   # ScanNet-style pinhole fallback
```

Depth PNG values are interpreted as millimeters and converted to meters. Image names with pure numeric stems are sorted numerically; all other names are sorted lexicographically.

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
- `pipeline.output.pcd_max_points=10000000`: cap saved point cloud points.
- `pipeline.output.pcd_sample_ratio=0.015`: per-frame stride sampling before the global point cap.

Saved artifacts:

- `pose/color.npz`: camera-to-world pose per selected frame.
- `depth/color.zip`: per-frame sensor depth after any camera normalization, as float16 NumPy `.npy` entries.
- `intrinsics/color.json`: one shared original-resolution downstream pinhole intrinsics record.
- `pcd/color_backproject.ply`: direct backprojected point cloud, if enabled.
- `pcd/color_tsdf.ply`: native TSDF-fused sampled point cloud, if enabled.

## ScanNet Benchmark

```bash
python3 scripts/scannet_vipe_bench_evaluator.py \
  --scenes scene0000_00 scene0011_00 scene0378_00 \
  --work-dir ./workspace/evaluation_scannet_vipe_external_depth \
  --input-root /robodata/smodak/repos/ovo/data/input/ScanNet \
  --raw-root /robodata/smodak/datasets/scannet_v2/scans \
  --max-frames -1 \
  --num-fusion-workers 16 \
  streams.fps=30
```

The benchmark adapter runs ViPE, writes a lightweight local manifest pointing at native ViPE artifacts, and computes pose/reconstruction metrics for native TSDF and direct-backproject reconstruction with the local ScanNet evaluator in `vipe/bench/scannet.py`.
