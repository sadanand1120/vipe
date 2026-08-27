# Current ViPE vs. `instance_bench` ViPE

Assumption: `instance_bench` runs its vendored ViPE TSDF path with `pcd_enable=true`. Its separate backprojection cloud builder is excluded from this comparison.

## Algorithm Differences

| Aspect | Current ViPE | `instance_bench` ViPE with TSDF enabled | CHOICE2 |
|---|---|---|---|
| BA objective | Dense flow + sensor-disparity prior + backend point-to-plane sensor-depth geometry + active adjacent-pose prior | Dense flow + sensor-disparity prior; no point-to-plane term; pose prior disabled | `instance_bench` |
| Pose infilling | Reuses DROID feature maps computed during pass 1 | Recomputes DROID feature maps during pass 2; intended pose computation is otherwise equivalent | Current ViPE: faster by avoiding duplicate feature encoding; assumes sufficient VRAM |
| Buffer management | Keeps the original 1024-slot allocation | Uses 2048 slots during the frontend, then trims unused capacity before backend BA | Current ViPE |
| Depth preprocessing before TSDF | Rejects invalid depth and depth beyond the configured truncation distance | Additionally removes pixels whose relative depth gradient exceeds `0.1`, reducing flying surfaces around depth discontinuities | Current ViPE |
| RGB fusion | Nearest-pixel RGB sampling | Bilinear RGB sampling at the projected subpixel coordinate | `instance_bench` |
| PLY writing | Direct C++ tensor-to-PLY writer | Converts extracted tensors to NumPy and writes the same XYZ/RGB/normal fields in Python; mainly a performance difference | Current ViPE: same output without NumPy copies |

## Configuration Differences

Settings with equal values are omitted.

| Parameter | Current ViPE | `instance_bench` ViPE | CHOICE2 |
|---|---:|---:|---|
| Graph buffer | 1024 | 2048 | 1024 |
| Forced keyframe gap | 16 frames | 8 frames | 8 frames |
| Warmup optimization | `2 updates x 2 BA iters = 4` cycles | `8 x 3 = 24` cycles | `8 x 3 = 24` cycles |
| Per-keyframe frontend maximum | `(3 + 2) x 2 = 10` cycles | `(4 + 2) x 3 = 18` cycles | `(4 + 2) x 3 = 18` cycles |
| Backend optimization | `17 x 5 = 85` cycles | `31 x 8 = 248` cycles | `31 x 8 = 248` cycles |
| Infill optimization per chunk | `4 x 2 = 8` cycles | `10 x 3 = 30` cycles; inner count is the code default | `10 x 3 = 30` cycles |
| Sensor-disparity regularization | `0.009` | `0.001` | `0.001` |
| Backend point-to-plane depth term | Weight `0.01`, 96 points/factor, 0.25 m residual cutoff | Absent | Absent |
| Pose prior | Weight `0.02`, active adjacent-pose prior | Weight `0.0`, disabled | Disabled (`0.0`) |
| TSDF depth truncation | 5 m | 15 m | 5 m |
| Voxels per sparse block edge | 8 | 16 | 16 |
| Depth-pixel stride for block allocation | 128 | 4 | 4 |
| Depth-edge filter | None | Enabled, relative-gradient threshold `0.1` | Disabled |
| RGB sampling | Nearest | Bilinear | Bilinear |

In short, the reconstruction mechanism is fundamentally the same under this assumption. The meaningful reconstruction differences are depth-edge filtering, RGB interpolation, depth range, and sparse-block allocation density. The pose trajectories can still differ substantially because the BA objectives, keyframe cadence, and optimization budgets differ.
