# Pipeline Comparison

## ViPE

```mermaid
flowchart TD
    A[Frame directory: ordered RGB frames] --> B[FrameDirStream]
    B --> C[GeoCalib intrinsics]
    B --> D[TrackAnything instance/sky masks]
    C --> E[SLAMSystem]
    D --> E

    E --> F{pipeline config}

    F -->|default.yaml| G[Keyframe metric depth: UniDepth-L]
    G --> H[ViPE SLAM optimization]
    H --> I[Pose trajectory: c2w per frame]
    H --> J[AdaptiveDepthProcessor: UniDepth-L + SVDA + SLAM map alignment]
    J --> K[Final metric depth per frame]

    F -->|dav3.yaml| L[Keyframe metric depth: DAV3]
    L --> M[ViPE SLAM optimization]
    M --> N[Pose trajectory: c2w per frame]
    M --> O[MultiviewDepthProcessor: DAV3-GIANT with ViPE poses/intrinsics]
    O --> P[Final DAV3 metric depth + optional confidence per frame]

    I --> Q[save_artifacts]
    K --> Q
    N --> Q
    P --> Q

    Q --> R[Terminal pose output: outputs/<scene>/pose/color.npz]
    Q --> S[Terminal pointcloud output: outputs/<scene>/pcd/color_backproject.ply]

    S --> T[Backproject aggregation: depth + intrinsics + c2w -> world points]
    P --> U[DAV3 confidence filter if available: conf >= mean(conf)*0.75, sample_ratio=0.015]
    U --> S
```

## Depth Anything 3 Streaming

```mermaid
flowchart TD
    A[Frame directory: ordered RGB frames] --> B[Chunk frames with overlap]
    B --> C[DepthAnything3 chunk inference]
    C --> D[Per-chunk predictions: depth, confidence, extrinsics, intrinsics]

    D --> E{use_gt_pose?}
    E -->|false| F[Use DA3 predicted chunk poses]
    E -->|true| G[Use ScanNet GT poses/intrinsics]

    F --> H[Overlap/loop alignment]
    H --> I[Estimate chunk-to-chunk Sim3 transforms]
    I --> J[Apply accumulated Sim3 to chunk point maps/poses]
    G --> J

    J --> K[Terminal pose output: camera_poses.txt]

    J --> L[Backproject each chunk: depth + intrinsics + extrinsics -> world points]
    L --> M[Confidence filter: conf >= mean(conf)*0.75]
    M --> N[Sample ratio: 0.015]
    N --> O[Per-chunk PLYs: pcd/*_pcd.ply]
    O --> P[Byte-merge chunk PLYs]
    P --> Q[Terminal pointcloud output: pcd/combined_pcd.ply]
```

## Key Differences

- ViPE pose comes from ViPE SLAM; DA3 streaming pose comes from DA3 chunk pose estimation plus chunk Sim3 alignment, unless `use_gt_pose` is enabled.
- ViPE `default.yaml` uses UniDepth-L for keyframe scale and adaptive UniDepth/SVDA post depth.
- ViPE `dav3.yaml` uses DAV3 for keyframe scale and multiview DAV3 post depth.
- Both ViPE `color_backproject.ply` and DA3 streaming `combined_pcd.ply` are backprojected point aggregations, not TSDF fusion.
- TSDF fusion is used by the benchmark evaluators for reconstruction metrics, not by these terminal backproject PLY outputs.
