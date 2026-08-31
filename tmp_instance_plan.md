# Instance Distillation Integration Plan

## Scope And Fixed Decisions

- Extend the current ViPE runtime, not the stale vendored ViPE copy under `../instance_bench`.
- Keep ViPE Stages 1-5 unchanged: final poses, native TSDF reconstruction, and normal artifacts remain the base pipeline.
- Run class-agnostic instance distillation synchronously inside `VipePipeline.run()` after Stage 5. It is not a second command and does not consume GT.
- Port only the instance path. Do not port semantic feature distillation, VLM backbones, semantic metrics, viewers, annotation tools, lakeFS plumbing, prep tar plumbing, or experiment diagnostics.
- For initial parity, Stage 6 rebuilds the frontier's separate 2 cm backprojected occupancy cloud from sensor depth and final ViPE poses. Any later migration to ViPE's TSDF surface is a separate user-approved task; this implementation stops after occupancy-cloud parity.
- Replica GT labels and exclusions exist only in the benchmark evaluator. Runtime instance predictions must be identical whether or not GT assets exist.

## Source Of Truth And Expected Parity

The algorithmic source is `../instance_bench` commit `a887292`, specifically `mask_source.py`, `sam1_masks.py`, `sam2_track.py`, `project.py`, `lift.py`, `atoms.py`, and the reachable instance-selection path in `assoc_hier.py`.

The valid historical Replica anchors are fixed-`K=5` AR:

| Scene | AR | Hypotheses |
| --- | ---: | ---: |
| `office0` | 0.4581 | 337 |
| `office2` | 0.5850 | 366 |
| `room0` | 0.6241 | 323 |

The actual mean is 0.5557. The `default.yaml` max-over-K numbers and `data_analytics/frontier_out` CA-PQ files are stale and will not be used.

These anchors came from a different vendored ViPE, 30 Hz source sequences, and an RTX PRO 6000 environment. The integrated pipeline uses the current ViPE frontier and canonical data, currently 5 Hz. Therefore validation has two layers:

1. **Port parity:** fixed cloud/poses/images must produce matching intermediate counts and approximately matching final hypotheses/AR against `instance_bench`.
2. **Current end-to-end behavior:** run current ViPE plus instance stages on `office0`, `office2`, and `room0`; compare the mean and scene ordering to the historical anchors while treating environment/pose/input-rate differences as explicit confounders rather than port bugs.

## Runtime Architecture

### Configuration

- Keep `configs/default.yaml` as the only SLAM/TSDF config.
- Add `configs/default_instance.yaml` as a small instance-only overlay. Do not duplicate SLAM or TSDF values.
- Add `run.py --instance-config configs/default_instance.yaml`. Supplying it enables the post-Stage-5 path; omitting it preserves the current pose-plus-TSDF pipeline.
- The Replica instance benchmark always loads both `default.yaml` and `default_instance.yaml`.
- Add `configs/eval_replica_instance_config.yaml` for GT projection, AR thresholds, exclusions, and evaluator output names. No evaluation-only values enter runtime.

`default_instance.yaml` will carry the frozen structural values:

```yaml
pipeline:
  instance:
    cloud:
      voxel_m: 0.02
      depth_min_m: 0.1
      depth_max_m: 12.0
    images:
      long_side: 1024
    frames:
      move_cm: 8.0
      move_deg: 8.0
    masks:
      chunk_keyframes: 4
      seed_topk: 100
      sam1:
        model_path: models/sam_vit_h_4b8939.pth
        points_per_side: 48
        points_per_batch: 128
        pred_iou_thresh: 0.6
        stability_score_thresh: 0.9
        stability_score_offset: 1.0
        box_nms_thresh: 0.7
      sam2:
        model_path: models/sam2.1_hiera_small.pt
        model_config: configs/sam2.1/sam2.1_hiera_s.yaml
        threshold: -1.0
      stitch:
        min_iou: 0.8
        margin: 1.2
    lift:
      occlusion_tolerance_m: 0.05
      min_voxels: 5
    association:
      membership_budget: 5
      mustlink_min_affinity: 0.98
      mustlink_min_observations: 8
      atom_size_m: 0.03
      atom_normal_weight: 4.0
      atom_knn: 12
      atom_radius_cap_voxels: 2.5
      normal_radius_min_m: 0.06
      normal_radius_voxel_multiplier: 4
      normal_max_neighbors: 30
      candidate_min_voxels: 10
      eligible_min_voxels: 8
      eligible_fraction: 0.05
      retained_coverage: 0.5
      dedup_ratio: 0.95
      max_mask_crowd: 100
      coverage_alpha: 2.0
      purity_beta: 2.0
      endorsement_floor: 0.05
      hypothesis_price: 0.5
```

SAM1 is fixed to ViT-H and SAM2 to SAM2.1-small. Their checkpoint and SAM2 model-config paths live explicitly in `default_instance.yaml`; no model-path environment variables are introduced. Model choices, all-seeds-at-once behavior, GPU-resident state, disabled SAM2 postprocessing, zero crop layers, and zero SAM1 region postprocessing are single code paths rather than exposed dead branches.

### Pipeline Handoff

Refactor `VipePipeline.run()` to convert final trajectory and intrinsics to CPU once. Its current `SLAMOutput` return value is unused by both `run.py` and the benchmark, so remove that return contract. Stage 5 consumes the CPU arrays, then releases the SLAM output and CUDA cache before constructing SAM models. Stage 6 receives:

- canonical `FrameDir` for lazy RGB/depth replay;
- final contiguous camera-to-world poses;
- shared canonical pinhole intrinsics;
- the scene `ArtifactPath`;
- the instance config.

The instance invocation occurs only after Stage 5 returns successfully. No pose NPZ reread, subprocess, prep tar, lakeFS round trip, or full RGB-D RAM cache is needed.

### Ported Runtime Modules

Create a compact `vipe/instance/` package:

- `pipeline.py`: post-Stage-5 orchestration, temporary occupancy-cloud construction, motion coreset, artifact writing, and timing/count summary.
- `masks.py`: only the frozen SAM1-H AMG, SAM2.1-small chunk propagation, deterministic track IDs, and global 3D track linking. Merge the useful parts of `sam1_masks.py`, `sam2_track.py`, and `mask_source.py`; remove their global config imports and viewer-only RLE APIs.
- `lift.py`: deterministic elementwise projection, depth occlusion test, and mask-to-voxel lifting. Merge the small `project.py` and `lift.py` paths.
- `atoms.py`: normal estimation and normal-aware supervoxels. Require Open3D; do not retain the result-changing PCA fallback.
- `association.py`: GT-free stages for signatures, must-link contraction, hierarchy, evidence candidates, K-budgeted selection, and exchange.

Port `assoc_hier.py` by behavior, not by blind copy. Remove from runtime:

- `gtv` and `inst_metrics` parameters;
- mask-union and atom-oracle ceilings;
- dense per-GT tree diagnostics;
- viewer-only `hyp_stats` recomputation;
- precompact soup and `groups.npz` dumps;
- RSS/debug instrumentation that has no runtime consumer;
- global module-level config state and alternate/dead mechanisms.

Preserve ordering-sensitive behavior exactly: stable SAM top-k ordering, ascending frame/chunk processing, monotonically allocated track IDs, minimum-ID union-find representatives, elementwise projection arithmetic, stable atom ordering, and the current association tie-breaks.

### Image And Cloud Handling

- Stage 6 replays every canonical depth frame and unions `floor(world_xyz / 0.02)` voxel coordinates, returning voxel centers `(index + 0.5) * 0.02`. Valid depth is strictly between 0.1 m and 12 m for the Replica frontier.
- The cloud is in the final ViPE SLAM frame. GT alignment never enters runtime.
- Motion coreset selection runs on the final ViPE trajectory and keeps a frame after more than 8 cm translation or 8 degrees optical-axis rotation from the last kept frame.
- Instance masks/lifting use a 1024-pixel-long-side view. RGB uses PIL LANCZOS and temporary JPEG quality 95; depth uses nearest-neighbor; intrinsics scale per axis. Only selected chunks are materialized for SAM2 and each temporary chunk directory is deleted immediately.
- This 1024 contract is local to instance stages. It does not alter the 1280-pixel canonical ViPE input or Stage-5 TSDF.

## Runtime Artifacts

The actual prediction is overlapping, at most K=5 hypotheses per voxel. A single hard label PLY is only a visualization.

- `instances/<scene>.npz`: uncompressed `float32 points`, concatenated `int32 hypothesis_indices`, `int64 hypothesis_offsets`, and scalar `K`. This is the canonical prediction artifact.
- `instances/<scene>_summary.json`: resolved instance config, stage timings, frame/mask/track/atom/node/candidate/hypothesis counts, and artifact schema version. No GT or metric values.
- `pcd/<scene>_instances.ply`: temporary occupancy cloud colored by smallest-hypothesis-wins, with an integer instance field. Unclaimed voxels are gray.
- Existing `pose/<scene>.npz` and `pcd/<scene>_tsdf.ply` remain unchanged.

Do not save masks, SAM state, lifted groups, the full hierarchy, precompact hypotheses, GT labels, or semantic features by default.

## Replica Instance Evaluation

Add `vipe/bench/replica_instance.py` and `scripts/replica_instance_bench_evaluator.py`.

The script follows the existing benchmark contract:

- CLI: `--scenes`, `--work-dir`, `--input-root`, `--raw-root`, `--print-only`, `--do-final-eval`.
- Dynamic per-GPU scene queue; one scene process is launched only when a GPU worker claims it.
- Scene completion requires pose, TSDF, instance NPZ, instance summary, and instance PLY with matching canonical metadata/config fingerprints.
- Build timing/FPS and peak VRAM are recorded per scene.
- Final evaluation can be resumed from completed runtime artifacts without rerunning ViPE or instance distillation.

Evaluation for each Replica scene:

1. Build/cache the GT 2 cm occupancy cloud from canonical sensor depth and canonical GT poses.
2. Label that GT cloud from `habitat/mesh_semantic.ply`: triangle-centroid 5-NN majority vote, reject neighbors beyond 0.10 m, and map object ID 0 to background `-1`.
3. SE3-align predicted camera centers to canonical GT camera centers with Kabsch, transform the predicted occupancy cloud, then 1-NN transfer GT instance IDs from the GT cloud. Record ATE and transfer-distance percentiles as diagnostics.
4. Apply Replica exclusions by relabeling excluded IDs to `-1`. Excluded voxels remain in the cloud and therefore still penalize a hypothesis union, matching the frontier.
5. Compute each GT instance's best point-set IoU over all predicted hypotheses. Report AR over thresholds 0.50:0.05:0.95, R50/R75/R90, hypothesis count, and mean/max per-voxel membership.

Replica exclusions in `eval_replica_instance_config.yaml`:

```yaml
office0: [1, 13, 18, 26, 56, 65, 67]
office2: [5, 6, 7, 78, 83]
room0: [17, 26, 28, 29, 37, 38, 42, 52, 53, 62, 66, 82, 91]
```

Use canonical scene names only; the old `office0vipe` aliases were prep-tar names and are unnecessary here.

## Dependencies And Determinism

- Add `pillow`, `pycocotools`, Hydra/iopath requirements for SAM2, and pinned SAM1/SAM2 installations to the Dockerfile. Install the same additions into the local `humble` container's `vipe-manual` environment before local validation. SAM1 is already known at commit `6fdee8f2727f4506cfbbe553e23b895e27956588`; choose and record a fixed SAM2 commit before implementation validation because `instance_bench` installed SAM2 from unpinned HEAD.
- Add SAM checkpoint staging to the TACC image/workflow without baking dataset or output data into the image.
- Keep TF32 disabled, deterministic ViPE setup enabled, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` documented for the instance run.
- Require Open3D normal estimation. A fallback would silently change atom boundaries and AR.

## Validation Gates

1. Unit tests for cloud voxelization, image/intrinsics resize, pose coreset, projection/occlusion, lifting, track linking, hypothesis packing, AR, exclusions, and deterministic ordering.
2. Fixed-input differential tests against `../instance_bench` for cloud coordinates, kept frame IDs, mask/track counts, lifted voxel sets, atom/node/candidate counts, final hypothesis count, membership cap, and AR.
3. One-scene `office2` smoke inside `humble`/`vipe-manual`: verify Stage 5 artifacts are unchanged and Stage 6-12 artifacts are complete.
4. End-to-end `office0`, `office2`, `room0` Replica run with the new benchmark; report scene and mean AR/R50/R75/R90, hypotheses, runtime, and peak VRAM against the historical anchors with the known environment/input-rate caveat.
5. After parity is confirmed on the temporary occupancy cloud, stop and hand control back to the user. Do not migrate to the ViPE TSDF surface autonomously.

## Explicitly Not Ported

`distill_feats.py`, `feature_pool.py`, `feat_backbones.py`, semantic labels/metrics, all HTML viewers, annotation server, lakeFS clients/uploads, prep scripts/tars, shell runners, old vendored ViPE, `droid.pth`, data analytics, experiment flags, legacy CA-PQ code, GT-match PLYs, and toy/debug artifacts.
