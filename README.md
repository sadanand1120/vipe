# ViPE: Canonical RGB-D Scene Fork

This fork keeps one supported runtime path: a canonical ViPE RGB-D scene directory. ViPE estimates poses with the DROID/ViPE SLAM stack, uses the provided metric depth as the dense depth source, and writes pose plus native TSDF point-cloud artifacts. Dataset-specific cleanup, synchronization, and rectification happen before runtime in `scripts/data_extract/`.

## Installation

Manual setup:

```bash
conda create -n vipe-manual -c nvidia/label/cuda-12.8.0 -c conda-forge python=3.10 pip cuda-nvcc eigen zlib libcusparse-dev libcublas-dev libcusolver-dev -y
conda activate vipe-manual
pip3 install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip3 install --no-build-isolation -e .
```

The editable install builds the native ViPE extension, including the TSDF code path. The default CUDA arch list covers sm75, sm86, sm87, and sm90/PTX; rerun `pip3 install --no-build-isolation -e .` after changing `setup.py` or `csrc/`.

## Input Layout

`--input-dir` must point to the canonical scene root:

```text
<scene>/metadata.json
<scene>/color/000000.png
<scene>/depth/000000.png
<scene>/intrinsic/intrinsic_color.json
```

`metadata.json` is the source of truth for frame order. Each frame record names `color_file` and `depth_file`, and ScanNet benchmark scenes also include `pose_file`. Color is RGB8 PNG on disk, depth is `uint16` PNG in millimeters, and intrinsics are undistorted pinhole only. Runtime does not accept sidecar TXT intrinsics, JPG image discovery, or runtime OpenCV distortion branches.

Dataset converters live under `scripts/data_extract/`:

```bash
python3 scripts/data_extract/scannet_to_vipe.py --scans-root /path/to/scannet/scans --output-root data/scannet --scenes scene0000_00 --frame-skip 1
python3 scripts/data_extract/rosbag_to_vipe.py /path/to/bag.mcap --output-dir data/kinect_rosbags/processed/bag_scene
```

## Standalone Run

```bash
python run.py \
  --input-dir /path/to/scene \
  --output-dir /path/to/output
```

Useful output knobs in `configs/default.yaml`:

- `pipeline.output.pcd_max_points=10000000`: cap saved TSDF point cloud points.
- `pipeline.output.pcd_tsdf_depth_trunc_m=5.0`: ignore sensor depth beyond 5 meters.
- `pipeline.output.pcd_tsdf_num_voxels_per_block_edge=16`: TSDF voxel block edge size.
- `pipeline.output.pcd_tsdf_depth_sampling_stride=4`: sample every fourth depth pixel when opening TSDF voxel blocks.
- `pipeline.output.pcd_tsdf_depth_filter=false`: optional high-gradient depth-boundary rejection, currently disabled.
- `pipeline.output.pcd_tsdf_bilinear_color=true`: bilinearly sample RGB during fusion.

Saved artifacts:

- `pose/<scene>.npz`: camera-to-world pose per selected frame.
- `pcd/<scene>_tsdf.ply`: native TSDF-fused sampled point cloud with true RGB, `nx/ny/nz` normals, and `normals_red/green/blue` normal colors for `quick-tools ply-viewer`.

## ScanNet Benchmark

```bash
python3 scripts/scannet_vipe_bench_evaluator.py \
  --scenes scene0000_00 scene0011_00 scene0378_00 \
  --work-dir ./workspace/evaluation_scannet_vipe_external_depth \
  --input-root data/scannet \
  --raw-root /robodata/smodak/datasets/scannet_v2/scans \
  --do-final-eval
```

The benchmark adapter assumes `--input-root/<scene>` is already canonical. It runs ViPE, writes a lightweight local manifest pointing at native ViPE artifacts, and computes pose plus `recon` metrics with the local ScanNet evaluator in `vipe/bench/scannet.py` when `--do-final-eval` is supplied. Without `--do-final-eval`, it stops after ViPE exports and incremental pose metrics, which is useful for fast build/debug loops. Reconstruction eval aligns the saved TSDF PLY with the first ViPE and ScanNet camera poses using SE3, then reports a separate scale diagnostic before computing geometry and render metrics.
For benchmark runs, `--work-dir` owns the outputs: ViPE artifacts are written under `<work-dir>/vipe_outputs/<scene>`, benchmark manifests/caches under `<work-dir>/model_results/...`, and metric JSONs under `<work-dir>/metric_results/...`.
If multiple GPUs are visible through `CUDA_VISIBLE_DEVICES`, the benchmark splits scene builds across them and also parallelizes final eval workers. Completed scene artifacts are reused on rerun, and failed scenes are recorded under `metric_results/failed_scenes/`.
Runtime knobs live in `configs/default.yaml`; ScanNet-specific benchmark knobs live in `configs/eval_scannet_config.yaml`. Dataset roots stay explicit CLI inputs via `--input-dir`, `--input-root`, and `--raw-root`.

## Eval Dashboard

Use the dashboard to compare two ScanNet eval workspaces:

```bash
python3 scripts/scannet_eval_dashboard.py --before-root workspace/evaluation_scannet_default_old --after-root workspace/evaluation_scannet_default_new --input-root data/scannet --host 127.0.0.1 --port 18799
```

The dashboard reads aggregate and incremental pose JSONs, marks filtered/unavailable/failed scenes, excludes unavailable scenes from means, and refreshes every 30 seconds.
