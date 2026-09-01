# ViPE: RGB-D Pose, Reconstruction, And 3D Distillation

ViPE consumes one canonical RGB-D scene directory, estimates camera poses with the DROID/ViPE SLAM stack, fuses the provided metric depth into a native TSDF surface, and writes pose and reconstruction artifacts. Its optional post-TSDF path distills overlapping class-agnostic 3D instance hypotheses from SAM1/SAM2 masks and attaches an open-vocabulary semantic descriptor to every finalized hypothesis. Dataset-specific cleanup, synchronization, and rectification happen before runtime in `scripts/data_extract/`.

## Installation

Manual setup:

```bash
conda create -n vipe-manual -c nvidia/label/cuda-12.8.0 -c conda-forge python=3.10 pip cuda-nvcc eigen zlib libcusparse-dev libcublas-dev libcusolver-dev -y
conda activate vipe-manual
pip3 install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip3 install --no-build-isolation -e .
```

The editable install builds the native ViPE extension, including the TSDF code path. The default CUDA arch list covers sm75, sm86, sm87, and sm90/PTX; rerun `pip3 install --no-build-isolation -e .` after changing `setup.py` or `csrc/`.

Install the 3D distillation dependencies into the same environment. This extra includes the pinned SAM implementations and FG-CLIP runtime dependencies; SAM2's CUDA extension is unnecessary because its associated postprocessing path is disabled.

```bash
SAM2_BUILD_CUDA=0 SAM2_BUILD_ALLOW_ERRORS=0 pip3 install --no-build-isolation -e '.[instance]'
```

Stage the SAM checkpoints once:

```bash
mkdir -p models && wget -O models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && wget -O models/sam2.1_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

Semantic descriptors use FG-CLIP at the exact model revision in `configs/default_instance.yaml`; first use downloads it to the Hugging Face cache. To prefetch it:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('qihoo360/fg-clip-large', revision='5a8f0f23b5a06dc92310e907599b2a0c2d58fe6f')"
```

## Input Layout

`--input-dir` must point to the canonical scene root:

```text
<scene>/metadata.json
<scene>/color/000000.png
<scene>/depth/000000.png
<scene>/intrinsic/intrinsic_color.json
```

The length of `metadata.json["frames"]` defines the sequence length and `metadata.json["fps"]` records its canonical rate. Runtime frames use contiguous six-digit indices: `color/<index>.png` and `depth/<index>.png`; benchmark poses use `pose/<index>.txt`. Color is RGB8 PNG, depth is `uint16` millimeters, and intrinsics are shared undistorted pinhole parameters. Extractors preserve aspect ratio, normalize the image long side to `--vipe-res` pixels (`1280` by default), and subsample to `--vipe-fps` (`5` by default). RGB, depth, poses, and intrinsics are transformed together before runtime.

Dataset converters live under `scripts/data_extract/`:

```bash
python3 scripts/data_extract/scannet_to_vipe.py --scans-root /path/to/scannet/scans --output-root data/scannet --scenes scene0000_00 --vipe-res 1280 --vipe-fps 5
python3 scripts/data_extract/replica_niceslam_to_vipe.py --niceslam-root /path/to/Replica --full-root /path/to/Replica_full --output-root data/replica --vipe-res 1280 --vipe-fps 5
python3 scripts/data_extract/rosbag_to_vipe.py /path/to/bag.mcap --output-dir data/kinect_rosbags/processed/bag_scene --vipe-res 1280 --vipe-fps 5
```

ScanNet and Replica are treated as nominal 30 Hz sources. Rosbag extraction uses synchronized message timestamps and rejects a requested rate above the measured source rate rather than duplicating frames.

## Standalone Run

```bash
python run.py \
  --input-dir /path/to/scene \
  --output-dir /path/to/output
```

Runtime has no raw-rate or temporal-subsampling path. SLAM, pose infill, TSDF fusion, and saved pose indices consume the complete contiguous canonical sequence produced by the extractor.

To run instance distillation synchronously after pose and TSDF output:

```bash
python run.py --input-dir /path/to/scene --output-dir /path/to/output --instance-config
```

`--instance-config` without a value loads `configs/default_instance.yaml`; an explicit YAML path may be supplied instead. Stages 6-11 select the overlapping instance hypotheses and Stage 12 distills their semantic descriptors synchronously before artifacts are written.

The Stage-12 frontier is under `pipeline.instance.features`:

```yaml
features:
  grid: 64                         # G x G dense feature map
  weight_a: 1.0                    # projective-incidence exponent
  weight_b: 1.0                    # inverse-depth exponent
  occlusion_tolerance_m: 0.05
  model_path: qihoo360/fg-clip-large
  revision: 5a8f0f23b5a06dc92310e907599b2a0c2d58fe6f
```

FG-CLIP produces 768-dimensional descriptors from its native `336 x 336` input (`24 x 24` patch grid) and projects only depth-consistent TSDF points from the motion-selected frame coreset. See [`ALGORITHM.md`](ALGORITHM.md) for the exact visibility, weighting, fusion, and overlap equations.

Useful output knobs in `configs/default.yaml`:

- `pipeline.output.pcd_tsdf_depth_trunc_m=5.0`: ignore sensor depth beyond 5 meters.
- `pipeline.output.pcd_tsdf_num_voxels_per_block_edge=8`: TSDF voxel block edge size.
- `pipeline.output.pcd_tsdf_depth_sampling_stride=8`: sample every eighth depth pixel when opening TSDF voxel blocks.

TSDF fusion uses the provided sensor depth directly and bilinearly samples RGB at projected subpixel coordinates.

Saved artifacts:

- `pose/<scene>.npz`: camera-to-world pose per canonical frame.
- `pcd/<scene>_tsdf.ply`: one native zero-surface point per occupied TSDF surface cell, with true RGB, `nx/ny/nz` normals, and `normals_red/green/blue` normal colors for `quick-tools ply-viewer`.

Instance-enabled runs additionally save:

- `instances/<scene>.npz`: authoritative native TSDF surface representatives, packed overlapping hypotheses, and one row-aligned float16 semantic descriptor per hypothesis.
- `instances/<scene>_summary.json`: resolved config, structural counts, feature coverage, descriptor metadata, and Stage-12 timing.
- `pcd/<scene>_instances.ply`: smallest-hypothesis-wins visualization with contrasting colors for touching instances; evaluation uses the NPZ.
- `pcd/<scene>_semantic_pca.ply`: deterministic PCA coloring of the overlap-averaged semantic field; gray points have no valid descriptor coverage.

The complete distillation algorithm and its explicit relation to ViPE Stages 1-5 are documented in [`ALGORITHM.md`](ALGORITHM.md).

## Instance And Semantic Evaluation

```bash
python3 scripts/replica_instance_bench_evaluator.py --scenes office0 office2 room0 --work-dir workspace/evaluation_replica_instance --input-root data/replica --raw-root /robodata/smodak/datasets/Replica_full --do-final-eval

python3 scripts/scannet_instance_bench_evaluator.py --scenes scene0011_00 --work-dir workspace/evaluation_scannet_instance --input-root data/scannet --raw-root /robodata/smodak/datasets/scannet_v2/scans --do-final-eval
```

Final evaluation aligns the predicted trajectory to GT with one SE3 transform, transfers GT object IDs to the predicted TSDF domain, and evaluates the overlapping fixed-`K=5` hypothesis soup. It reports AR over IoU thresholds `0.50:0.05:0.95`, R50/R75/R90, hypothesis and membership statistics, build timing, and peak VRAM.

Runtime parameters, including the semantic frontier, live in `configs/default_instance.yaml`; dataset-specific GT construction, label transfer, exclusions, and IoU settings live in `configs/eval_replica_instance_config.yaml` and `configs/eval_scannet_instance_config.yaml`.

`semantic_top1` is a GT-only point metric, not a runtime classification output. The evaluator derives the scene vocabulary from dataset annotations, reconstructs the overlap-averaged point descriptor only where valid hypotheses provide coverage, encodes each present class as `a photo of a {class_name}` with FG-CLIP, and chooses the maximum cosine-similarity class. Only points with both descriptor coverage and a mapped GT class are scored. Scene aggregation weights top-1 by `semantic_evaluated_points`; `semantic_field_coverage` is reported separately and averaged across scenes.

Evaluation writes two additional visualizations on the predicted TSDF domain. `pcd/<scene>_instances_gt.ply` shows transferred GT object IDs and preserves them in the `instance` property. `pcd/<scene>_instances_gtmatch.ply` retains the unique best predicted hypothesis for each GT instance when its IoU is at least `0.30`. GT labels and class names are never consumed by runtime distillation.

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
If multiple GPUs are visible through `CUDA_VISIBLE_DEVICES`, the benchmark splits scene builds across them and also parallelizes final eval workers. Completed scene artifacts are reused when their canonical metadata/intrinsics are unchanged, and failed scenes are recorded under `metric_results/failed_scenes/`.
Runtime knobs live in `configs/default.yaml`; ScanNet-specific benchmark knobs live in `configs/eval_scannet_config.yaml`. Dataset roots stay explicit CLI inputs via `--input-dir`, `--input-root`, and `--raw-root`.

## Eval Dashboard

Use the dashboard to compare two ScanNet eval workspaces:

```bash
python3 scripts/scannet_eval_dashboard.py --before-root workspace/evaluation_scannet_default_old --after-root workspace/evaluation_scannet_default_new --input-root data/scannet --host 127.0.0.1 --port 18799
```

The dashboard reads aggregate and incremental pose JSONs, marks filtered/unavailable/failed scenes, excludes unavailable scenes from means, and refreshes every 30 seconds.
