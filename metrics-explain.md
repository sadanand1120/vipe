# Scene0000_00 Recon Unposed Backproject Metrics: Current Analyzer State

This file explains the current diagnostic analyzer run:

```text
script = scripts/analyze_scannet_recon_metric.py
scene = scene0000_00
mode = recon_unposed / unposed
method = backproject
workspaces = test_direct vs test_null
matching = nearest-neighbor in XYZ
gt_sample_points = 10,000,000
render_voxel_size = 0.001 m
render_num_images = 800 sampled RGB frames out of 5,578
render metric = project fused PLY into RGB views, then PSNR/SSIM
```

Concrete command:

```bash
python3 scripts/analyze_scannet_recon_metric.py \
  workspace/evaluation_scannet_vipe_dav3_test_direct \
  workspace/evaluation_scannet_vipe_dav3_test_null \
  --scene scene0000_00 \
  --mode unposed \
  --method backproject \
  --render-workers 24 \
  --render-num-images 800 \
  --out-subdir metric_debug_10m
```

Important scope:

```text
This is analyzer-only diagnostic logic.
It does not change the official DA3/ScanNet evaluator yet.
```

## Exact Inputs

For each workspace, the analyzer loads the fused prediction PLY:

```text
direct:
workspace/evaluation_scannet_vipe_dav3_test_direct/model_results/scannet/scene0000_00/unposed/exports/fuse/pcd_backproject.ply

null:
workspace/evaluation_scannet_vipe_dav3_test_null/model_results/scannet/scene0000_00/unposed/exports/fuse/pcd_backproject.ply
```

It also loads:

```text
GT mesh:
/robodata/smodak/datasets/scannet_v2/scans/scene0000_00/scene0000_00_vh_clean_2.ply

ScanNet RGB / pose / intrinsics:
/robodata/smodak/repos/ovo/data/input/ScanNet/scene0000_00/{color,pose,intrinsic}
```

The prediction PLYs are already the benchmark fused PLYs from the regenerated 10M run. The analyzer does not reread ViPE depth zips or rerun fusion for this diagnostic.

Point counts:

| Case | Pred Raw | Pred In AABB | Pred Rejected | Pred Eval | GT Sampled Raw | GT Eval |
|---|---:|---:|---:|---:|---:|---:|
| `direct` | 10,000,000 | 9,998,652 | 1,348 | 9,998,652 | 10,000,000 | 10,000,000 |
| `null` | 10,000,000 | 9,986,096 | 13,904 | 9,986,096 | 10,000,000 | 10,000,000 |

AABB rejection:

| Case | Rejected / Raw | Rejected % |
|---|---:|---:|
| `direct` | 1,348 / 10,000,000 | 0.0135% |
| `null` | 13,904 / 10,000,000 | 0.1390% |

Geometry evaluation uses the full AABB-cropped predicted cloud and the full 10M sampled GT cloud:

```text
direct:
P_eval = 9,998,652 points
G_eval = 10,000,000 points

null:
P_eval = 9,986,096 points
G_eval = 10,000,000 points
```

## Geometry Pipeline

The geometry pipeline is:

```text
GT mesh
  -> uniformly sample 10M points
  -> GT point cloud G_raw
  -> G_eval

prediction PLY
  -> predicted point cloud P_raw
  -> crop P_raw to GT AABB +/- 0.1m
  -> P_inside
  -> P_eval

P_eval, G_eval
  -> nearest-neighbor query P_eval -> G_eval
  -> nearest-neighbor query G_eval -> P_eval
  -> square distances
  -> acc_l2, comp_l2, overall_l2
```

The AABB crop is:

```text
inside(p) =
  min_bound_x - 0.1 <= p_x <= max_bound_x + 0.1
  min_bound_y - 0.1 <= p_y <= max_bound_y + 0.1
  min_bound_z - 0.1 <= p_z <= max_bound_z + 0.1
```

The only point-count reduction before nearest-neighbor evaluation is the AABB crop on predicted points.

## Distance Arrays

The analyzer creates two nearest-neighbor distance arrays:

```text
A = pred_to_gt_distances
length(A) = number of points in P_eval
A[i] = || p_i - NN_G(p_i) ||_2

C = gt_to_pred_distances
length(C) = number of points in G_eval
C[j] = || g_j - NN_P(g_j) ||_2
```

Meaning:

```text
A asks: "For each predicted point, how far is the closest GT surface point?"
C asks: "For each GT point, how far is the closest predicted point?"
```

Actual distance summaries before squaring:

| Case | mean(A) | p50(A) | p90(A) | p95(A) | p99(A) | max(A) |
|---|---:|---:|---:|---:|---:|---:|
| `direct` | 0.07667 m | 0.06235 m | 0.12940 m | 0.16599 m | 0.53799 m | 1.08526 m |
| `null` | 0.09081 m | 0.06165 m | 0.19893 m | 0.27577 m | 0.55313 m | 1.33888 m |

| Case | mean(C) | p50(C) | p90(C) | p95(C) | p99(C) | max(C) |
|---|---:|---:|---:|---:|---:|---:|
| `direct` | 0.04331 m | 0.03523 m | 0.09094 m | 0.11379 m | 0.15304 m | 0.27033 m |
| `null` | 0.02080 m | 0.01099 m | 0.04329 m | 0.07634 m | 0.16788 m | 0.41876 m |

Core story:

```text
direct has better predicted-point accuracy:
  mean(A) direct = 0.07667 m
  mean(A) null   = 0.09081 m

null has better GT-surface coverage:
  mean(C) direct = 0.04331 m
  mean(C) null   = 0.02080 m
```

This remains the same qualitative tradeoff:

```text
direct predicted points are cleaner.
null covers the GT surface more densely.
```

## L2 Accuracy / `acc_l2`

The analyzer defines accuracy as:

```text
A2[i] = A[i]^2
acc_l2 = mean(A2)
```

Actual:

```text
direct:
acc_l2 = mean(pred_to_gt_distances^2)
       = 0.0120058796 m^2

null:
acc_l2 = mean(pred_to_gt_distances^2)
       = 0.0185098085 m^2
```

Lower is better, so `direct` wins accuracy:

```text
0.0120058796 < 0.0185098085
```

Difference:

```text
null - direct
= 0.0185098085 - 0.0120058796
= 0.0065039289 m^2
```

Interpretation:

```text
Predicted points from direct are closer to the GT surface on average after L2 penalty.
The L2 penalty makes larger wrong predicted structures matter more than a plain L1 mean.
```

Squared-distance summaries:

| Case | mean(A²) | p50(A²) | p90(A²) | p95(A²) | p99(A²) | max(A²) |
|---|---:|---:|---:|---:|---:|---:|
| `direct` | 0.012006 | 0.003888 | 0.016745 | 0.027553 | 0.289437 | 1.177796 |
| `null` | 0.018510 | 0.003801 | 0.039573 | 0.076048 | 0.305952 | 1.792591 |

Notice the median squared distance is similar, but the high-percentile and max errors are much worse for `null`. That is exactly where L2 pushes the metric toward `direct`.

## L2 Completeness / `comp_l2`

The analyzer defines completeness as:

```text
C2[j] = C[j]^2
comp_l2 = mean(C2)
```

Actual:

```text
direct:
comp_l2 = mean(gt_to_pred_distances^2)
        = 0.0030599690 m^2

null:
comp_l2 = mean(gt_to_pred_distances^2)
        = 0.0014366791 m^2
```

Lower is better, so `null` wins completeness:

```text
0.0014366791 < 0.0030599690
```

Difference:

```text
direct - null
= 0.0030599690 - 0.0014366791
= 0.0016232899 m^2
```

Interpretation:

```text
GT points are closer to some null predicted point.
This means null has better GT-surface coverage, even though its predicted points are noisier.
```

Squared-distance summaries:

| Case | mean(C²) | p50(C²) | p90(C²) | p95(C²) | p99(C²) | max(C²) |
|---|---:|---:|---:|---:|---:|---:|
| `direct` | 0.003060 | 0.001241 | 0.008271 | 0.012948 | 0.023421 | 0.073080 |
| `null` | 0.001437 | 0.000121 | 0.001874 | 0.005828 | 0.028182 | 0.175358 |

`null` has much better median and p90 completeness. Its tail is worse than direct at max, but the average still favors `null`.

## L2 Overall / `overall_l2`

The analyzer defines:

```text
overall_l2 = (acc_l2 + comp_l2) / 2
```

Actual:

```text
direct:
overall_l2 = (0.0120058796 + 0.0030599690) / 2
           = 0.0075329243 m^2

null:
overall_l2 = (0.0185098085 + 0.0014366791) / 2
           = 0.0099732438 m^2
```

Lower is better, so `direct` wins overall:

```text
0.0075329243 < 0.0099732438
```

Difference:

```text
null - direct
= 0.0099732438 - 0.0075329243
= 0.0024403195 m^2
```

Why direct wins despite null having better completeness:

```text
direct accuracy advantage:
  0.0185098085 - 0.0120058796 = 0.0065039289 m^2

null completeness advantage:
  0.0030599690 - 0.0014366791 = 0.0016232899 m^2
```

The direct accuracy advantage is about:

```text
0.0065039289 / 0.0016232899 = 4.01x
```

larger than the null completeness advantage, so `overall_l2` prefers `direct`.

## Image Projection Metric

The image metric pipeline is:

```text
prediction PLY
  -> voxel_down_sample(0.001m) for rendering only
  -> voxel centers + averaged RGB colors
  -> sample 800 ScanNet RGB frames with RNG seed 42
  -> for each sampled frame:
       world voxel centers -> camera coordinates
       camera coordinates -> pixel coordinates
       splat 0.1cm voxel footprint into image
       z-buffer closest voxel per pixel
       compare rendered RGB image to actual ScanNet RGB image
  -> mean PSNR over 800 frames
  -> mean SSIM over 800 frames
```

This 0.1cm voxelization is only for the image projection render. It is not used by the geometry nearest-neighbor metric.

Render input counts:

| Case | Raw PLY Points | 0.1cm Render Voxels | Rendered Images | Scene Images | Workers |
|---|---:|---:|---:|---:|---:|
| `direct` | 10,000,000 | 9,985,350 | 800 | 5,578 | 24 |
| `null` | 10,000,000 | 9,995,287 | 800 | 5,578 | 24 |

The projection uses ScanNet camera poses/intrinsics:

```text
X_cam = R_world_to_cam * X_world + t_world_to_cam

u = fx * X_cam_x / X_cam_z + cx
v = fy * X_cam_y / X_cam_z + cy
```

Only positive-depth voxels can render:

```text
X_cam_z > 0
```

Each 0.1cm voxel gets an approximate projected radius:

```text
radius_px = ceil(0.5 * voxel_size * max(fx, fy) / X_cam_z)
```

Then the radius is capped by `--render-radius-cap` to avoid huge splats from very near voxels.

Z-buffer:

```text
zbuf[pixel] = minimum projected voxel depth touching that pixel
render[pixel] = RGB color of the voxel that achieved zbuf[pixel]
```

Pixels not hit by any voxel stay black and count against PSNR/SSIM. That is intentional: holes in the projected point cloud are a visible reconstruction failure.

## PSNR

For one rendered frame:

```text
R = rendered RGB image, values in [0, 1]
I = actual ScanNet RGB image, values in [0, 1]
H = image height
W = image width
```

Mean squared RGB error:

```text
MSE = (1 / (3 * H * W)) * sum_y sum_x sum_c (R[y, x, c] - I[y, x, c])^2
```

PSNR:

```text
PSNR = 20 * log10(1 / sqrt(MSE))
```

Higher is better.

Actual 800-image mean:

```text
direct:
PSNR = 13.5626 dB

null:
PSNR = 13.0590 dB
```

So `direct` wins PSNR:

```text
13.5626 dB > 13.0590 dB
```

Difference:

```text
13.5626 - 13.0590 = 0.5036 dB
```

Concrete toy PSNR example:

```text
R[0] = [0.8, 0.7, 0.6]
I[0] = [0.7, 0.7, 0.6]

R[1] = [0.0, 0.0, 0.0]
I[1] = [0.2, 0.1, 0.0]
```

Squared errors:

```text
pixel 0: [(0.1)^2, (0.0)^2, (0.0)^2] = [0.01, 0.00, 0.00]
pixel 1: [(-0.2)^2, (-0.1)^2, (0.0)^2] = [0.04, 0.01, 0.00]
```

MSE:

```text
MSE = (0.01 + 0.00 + 0.00 + 0.04 + 0.01 + 0.00) / 6
    = 0.01
```

PSNR:

```text
PSNR = 20 * log10(1 / sqrt(0.01))
     = 20 * log10(10)
     = 20 dB
```

## SSIM

SSIM compares local image structure rather than only per-pixel squared error.

For each RGB channel:

```text
SSIM(x, y) =
  ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2))
  /
  ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))
```

Where:

```text
x = rendered channel
y = actual RGB channel
mu_x, mu_y = local Gaussian-window means
sigma_x^2, sigma_y^2 = local Gaussian-window variances
sigma_xy = local Gaussian-window covariance
C1 = 0.01^2
C2 = 0.03^2
```

The analyzer averages spatially and across RGB:

```text
SSIM_frame = mean([SSIM_R, SSIM_G, SSIM_B])
SSIM = mean(SSIM_frame over sampled frames)
```

Higher is better.

Actual 800-image mean:

```text
direct:
SSIM = 0.1858

null:
SSIM = 0.1512
```

So `direct` wins SSIM:

```text
0.1858 > 0.1512
```

Difference:

```text
0.1858 - 0.1512 = 0.0346
```

Interpretation:

```text
PSNR says direct has lower RGB image error.
SSIM says direct also preserves more image structure under projection.
```

## Final Summary

Current analyzer metrics:

| Metric | `direct` | `null` | Better |
|---|---:|---:|---|
| `acc_l2` | 0.012006 | 0.018510 | `direct` |
| `comp_l2` | 0.003060 | 0.001437 | `null` |
| `overall_l2` | 0.007533 | 0.009973 | `direct` |
| `PSNR`, 800 images | 13.5626 dB | 13.0590 dB | `direct` |
| `SSIM`, 800 images | 0.1858 | 0.1512 | `direct` |

Concrete interpretation:

```text
direct is cleaner by L2 geometry overall.
direct is also better by projected RGB PSNR/SSIM.

null still covers the GT surface more densely in the GT->prediction direction,
which is why comp_l2 is better for null.

But null's predicted points are much noisier/farther from GT:
  acc_l2 null = 0.018510
  acc_l2 direct = 0.012006

That accuracy gap dominates the completeness advantage under L2.
```

## PLY Outputs

The analyzer writes:

```text
pred_eval.ply = prediction after AABB crop
gt_eval.ply   = sampled GT mesh
```

It also writes continuous error-colored views:

```text
pred_accuracy_colored.ply = P_eval colored by A = pred_to_gt distance
gt_completion_colored.ply = G_eval colored by C = gt_to_pred distance
```

Output paths:

```text
direct:
workspace/evaluation_scannet_vipe_dav3_test_direct/<out-subdir>/scene0000_00/unposed/backproject/pred_eval.ply
workspace/evaluation_scannet_vipe_dav3_test_direct/<out-subdir>/scene0000_00/unposed/backproject/gt_eval.ply
workspace/evaluation_scannet_vipe_dav3_test_direct/<out-subdir>/scene0000_00/unposed/backproject/pred_accuracy_colored.ply
workspace/evaluation_scannet_vipe_dav3_test_direct/<out-subdir>/scene0000_00/unposed/backproject/gt_completion_colored.ply
workspace/evaluation_scannet_vipe_dav3_test_direct/<out-subdir>/scene0000_00/unposed/backproject/summary.json

null:
workspace/evaluation_scannet_vipe_dav3_test_null/<out-subdir>/scene0000_00/unposed/backproject/pred_eval.ply
workspace/evaluation_scannet_vipe_dav3_test_null/<out-subdir>/scene0000_00/unposed/backproject/gt_eval.ply
workspace/evaluation_scannet_vipe_dav3_test_null/<out-subdir>/scene0000_00/unposed/backproject/pred_accuracy_colored.ply
workspace/evaluation_scannet_vipe_dav3_test_null/<out-subdir>/scene0000_00/unposed/backproject/gt_completion_colored.ply
workspace/evaluation_scannet_vipe_dav3_test_null/<out-subdir>/scene0000_00/unposed/backproject/summary.json
```
