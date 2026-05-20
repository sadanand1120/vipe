# Full Computation Walkthrough For The Current Standalone ViPE Run

This document explains the current standalone ViPE path started by `run.py`.

The standalone run is:

```bash
export NUMEXPR_MAX_THREADS=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES='2'

python run.py \
  streams.base_path=/robodata/smodak/repos/ovo/data/input/ScanNet/scene0000_00/color \
  streams.fps=30 \
  pipeline.output.path=/robodata/smodak/repos/vipe/outputs/scene00_dav3tsdf \
  pipeline.output.save_artifacts=true \
  pipeline.output.pcd_fusion_mode=both
```

## High-Level Runtime Flow

The current standalone run is easiest to read as five computation stages. Each stage hands one explicit object/state bundle to the next stage. There is no iterative feedback from a later stage back into an earlier stage.

The implementation details below follow those five stages directly. Subsections are kept only where they separate materially different computation, not just because the code has another helper class.

```mermaid
flowchart LR
    S1[Stage 1: initialization and stream setup] -->|pinhole-normalized frame stream plus one GeoCalib intrinsics tensor| S2[Stage 2: SLAM pass 1 frontend loop]
    S2 -->|keyframe buffer: poses, DROID features, disparities, depth anchors| S3[Stage 3: backend global BA over keyframes]
    S3 -->|refined keyframe poses/disparities in GraphBuffer| S4[Stage 4: SLAM pass 2 pose infill loop]
    S4 -->|SLAMOutput: full trajectory, recovered intrinsics, keyframe indices| S5[Stage 5: replay, final depth, artifact/PCD writing]
```

The execution grain is:

| Stage | What it computes |
| --- | --- |
| Stage 1: initialization and stream setup | Creates one `FrameDir`, optionally enables per-frame ScanNet sensor-depth and/or sensor-intrinsics loading, then hands downstream code a pinhole stream plus one pinhole intrinsics vector. Intrinsics come either from loaded RGB/color calibration or from GeoCalib. No full-sequence cache is created. |
| Stage 2: SLAM pass 1 frontend loop | Loops over frames once, attaches the shared intrinsics, resizes each frame for SLAM, runs DROID motion-filter features on every frame, stores accepted keyframes with DROID feature/context tensors and the configured keyframe depth anchor, and runs incremental frontend BA as keyframes arrive. |
| Stage 3: backend global BA | Runs once over the complete pass-1 keyframe set with `steps=backend_iters`. It builds one fresh non-incremental factor graph and optimizes all keyframes as a sequence-level solve. |
| Stage 4: SLAM pass 2 pose infill loop | Loops over every frame again, appends frames in chunks, initializes non-keyframe poses from neighboring keyframes, optimizes those appended poses motion-only, and returns one pose per original frame plus the selected-frame indices of SLAM keyframes. |
| Stage 5: final depth and outputs | Re-reads original-resolution frames, assigns final SLAM pose/intrinsics, computes the configured final dense depth path, writes artifacts, and writes the configured PCD export or exports. |

## Stage 1: Initialization And Stream Setup

### Runtime Construction

Source files:

| File | Role |
| --- | --- |
| `run.py` | Hydra entrypoint, direct frame-dir source construction, and pipeline launch |
| `configs/default.yaml` | Single config file |
| `vipe/utils/logging.py` | ViPE logger setup shared by `run.py` and benchmark scripts |
| `vipe/pipeline.py` | `VipePipeline` |
| `vipe/streams/base.py` | `FrameDir`, `FrameData`, `FrameStream` |

Stage context: this is the sequence-once setup part of Stage 1. It constructs config-backed Python objects only. It does not read the whole image sequence, does not run GeoCalib yet, and does not run SLAM. If direct sensor intrinsics are enabled, `FrameDir` loads only the single calibration file during construction. The handoff is one raw `FrameDir` source plus one `VipePipeline`; camera-model normalization happens when `pipeline.run(...)` starts.

#### Diagram

```mermaid
flowchart TD
    A[Shell command] --> B[Environment variables]
    A --> C[Hydra CLI overrides]
    C --> D[configs/default.yaml]
    D --> E[DictConfig args]
    E --> F[FrameDir args.streams base_path/fps/start/end/skip]
    E --> K[streams.input_camera_model]
    E --> G[VipePipeline args.pipeline]
    F --> H[frame_stream]
    G --> I[pipeline]
    K --> J[pipeline.run frame_stream plus input_camera_model]
    H --> J
    I --> J
```

#### Computation

`run.py` is decorated with:

```python
@hydra.main(version_base=None, config_path="configs", config_name="default")
```

Hydra loads `configs/default.yaml`, then applies CLI overrides such as:

```text
streams.base_path=/robodata/smodak/repos/ovo/data/input/ScanNet/scene0000_00/color
streams.fps=30
pipeline.output.path=/robodata/smodak/repos/vipe/outputs/scene00_dav3tsdf
pipeline.output.save_artifacts=true
pipeline.output.pcd_fusion_mode=both
```

The shell variables do not change Python control flow directly:

| Variable | Effect |
| --- | --- |
| `CUDA_VISIBLE_DEVICES='2'` | Makes physical GPU 2 appear as CUDA device 0 inside PyTorch/Open3D/DA3 calls. |
| `NUMEXPR_MAX_THREADS=16` | Controls NumExpr thread count if NumExpr is imported. |
| `OMP_NUM_THREADS=16` | Controls OpenMP thread count for libraries that obey it. |
| `MKL_NUM_THREADS=16` | Controls MKL thread count for libraries that obey it. |

`run.py` configures logging, constructs the pipeline, and validates the depth mode before it asks `FrameDir` whether sensor depth should be loaded:

```python
logger = configure_logging()
pipeline = VipePipeline(
    slam=args.pipeline.slam,
    depth=args.pipeline.depth,
    output=args.pipeline.output,
)
frame_stream = FrameDir(
    path=args.streams.base_path,
    fps=args.streams.fps,
    frame_start=args.streams.frame_start,
    frame_end=args.streams.frame_end,
    frame_skip=args.streams.frame_skip,
    load_sensor_depth=args.pipeline.depth.use_gt_sens_depths is not None,
    load_sensor_intrinsics=args.streams.use_gt_intrinsics,
)
```

`pipeline.depth.use_gt_sens_depths` controls whether `FrameDir` also reads ScanNet sensor depth from the sibling `depth/` directory:

| Value | FrameDir behavior | Later depth behavior |
| --- | --- | --- |
| `null` | RGB only. `FrameData.sensor_depth=None`. | Use DAV3 normally. |
| `scale` | RGB plus sensor depth. | Run DAV3, then multiply DAV3 depth by one least-squares scalar to match sensor depth. |
| `direct` | RGB plus sensor depth. | Skip the relevant DAV3 depth model and use sensor depth directly, with invalid sensor pixels masked out of BA priors. |

`streams.use_gt_intrinsics` controls whether `FrameDir` also reads RGB/color camera calibration from the sibling `intrinsic/` directory:

| Value | FrameDir behavior | Stage 1 camera behavior |
| --- | --- | --- |
| `false` | Does not load an external camera file. | Use the GeoCalib path controlled by `streams.input_camera_model`. |
| `true` | Load `<scene>/intrinsic/intrinsic_color.json` if present, otherwise `<scene>/intrinsic/intrinsic_color.txt`. | Use the loaded camera as the source of truth. If it has OpenCV distortion metadata, undistort the stream once into a pinhole image plane. |

The TXT path is ScanNet-style pinhole-only metadata. The JSON path can carry width, height, camera matrix, projection matrix, distortion model, and distortion coefficients. JSON currently supports OpenCV `plumb_bob` and `rational_polynomial` distortion, which covers ROS `sensor_msgs/CameraInfo` exports such as `data/rosbags/distilled_bag/intrinsic/intrinsic_color.json`.

`streams.input_camera_model` controls how Stage 1 interprets the incoming RGB frames before downstream pinhole SLAM:

| Value | Stage 1 behavior |
| --- | --- |
| `pinhole` | Current default. Estimate one shared pinhole FOV from three sampled frames and pass raw frames downstream. |
| `radial`, `simple_radial`, `simple_divisional`, `simple_mei` | Use GeoCalib distorted weights to estimate that distorted input camera from three sampled frames, average the estimated camera parameters, wrap the source in a pinhole-normalizing stream, and pass undistorted pinhole frames downstream. |

`radial` is the upstream GeoCalib two-parameter radial model with `k1,k2`. `simple_radial`, `simple_divisional`, and `simple_mei` use one distortion parameter.

`configure_logging()` owns the `vipe` logger tree. It clears duplicate handlers on the top-level `vipe` logger, installs one tqdm-compatible INFO handler, and re-enables `vipe.*` child loggers so module logs from `vipe.slam.system`, `vipe.slam.components.buffer`, `vipe.pipeline`, and artifact saving all propagate consistently in both `run.py` and the ScanNet benchmark adapter.

The pipeline constructor creates the output root:

```python
self.slam_cfg = slam
self.depth_cfg = depth
self.out_cfg = output
_validate_gt_sens_depth_mode(self.depth_cfg.use_gt_sens_depths)
self.out_path = Path(self.out_cfg.path)
self.out_path.mkdir(exist_ok=True, parents=True)
```

With the command above:

```text
self.out_path = /robodata/smodak/repos/vipe/outputs/scene00_dav3tsdf
```

Finally:

```python
logger.info(f"Processing {frame_stream.name()}")
pipeline.run(
    frame_stream,
    input_camera_model=args.streams.input_camera_model,
    use_gt_intrinsics=args.streams.use_gt_intrinsics,
)
logger.info(f"Finished processing {frame_stream.name()}")
```

#### Branches

| Branch | Current command outcome |
| --- | --- |
| `save_artifacts` | `true`, so artifacts are written. |
| `pcd_fusion_mode` | `both` for the command shown. This is also the default, so both PCD exports are written. |

### Frame Directory Source And Frame Ordering

Source files:

| File | Role |
| --- | --- |
| `vipe/streams/base.py` | `FrameDir`, `FrameData`, `FrameStream` |
| `vipe/utils/misc.py` | Numeric-vs-lexicographic image sorting |

Stage context: this is the frame-source part of Stage 1. The file list and image size are computed once in `FrameDir.__init__`. Actual pixel tensors are produced lazily each time the stream is iterated, so later stages can re-read the same directory without caching a full frame list in memory.

#### Diagram

```mermaid
flowchart TD
    A[base_path] --> B[Path exists check]
    B --> C[glob jpg jpeg png bmp tiff tif both cases]
    C --> D{Are all stems pure digits?}
    D -->|yes| E[sort by int(stem), then str(path)]
    D -->|no| F[sort lexicographically by str(path)]
    E --> G[frame_files]
    F --> G
    G --> H[read first frame with cv2.imread]
    H --> I[record H, W, fps, start, end, step]
    I --> J[__getitem__ or __iter__ reads selected frames]
    J --> K[BGR to RGB]
    K --> L[float32 0..1 RGB tensor on CUDA]
    J --> N{load_sensor_depth?}
    N -->|yes| O[Read sibling depth png, resize nearest to RGB, convert mm to meters]
    N -->|no| P[sensor_depth None]
    O --> M[FrameData(raw_frame_idx, rgb, sensor_depth)]
    P --> M
```

#### Computation

`run.py` calls:

```python
FrameDir(
    path=args.streams.base_path,
    fps=args.streams.fps,
    frame_start=args.streams.frame_start,
    frame_end=args.streams.frame_end,
    frame_skip=args.streams.frame_skip,
    load_sensor_depth=args.pipeline.depth.use_gt_sens_depths is not None,
)
```

`FrameDir.__init__`:

1. Checks `path.is_dir()`.
2. Collects image files with extensions:
   ```python
   [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
   ```
   and uppercase variants.
3. Calls:
   ```python
   self.frame_files = sort_image_sequence(set(self.frame_files))
   ```
4. Reads the first frame with `cv2.imread` to set `_height` and `_width`.
5. Sets:
   ```python
   self.start = frame_start
   self.end = len(self.frame_files) if frame_end == -1 else min(frame_end, len(self.frame_files))
   self.step = frame_skip
   self._fps = fps / self.step
   ```

The current image sorting rule is:

```python
def sort_image_sequence(paths):
    paths = list(paths)
    if paths and all(Path(path).stem.isdigit() for path in paths):
        return sorted(paths, key=lambda path: (int(Path(path).stem), str(path)))
    return sorted(paths, key=str)
```

That means:

| File names | Sort order |
| --- | --- |
| `0.png`, `1.png`, `2.png`, `10.png` | `0`, `1`, `2`, `10` because every stem is numeric |
| `frame1.png`, `frame10.png`, `frame2.png` | lexicographic: `frame1`, `frame10`, `frame2` |
| mixed `0.png`, `frame1.png` | lexicographic because not every stem is numeric |

`FrameDir` supports both iteration and random access. Random access uses selected-frame indices, not raw sorted-file indices:

```python
frame_idx = self.start + index * self.step
frame_path = self.frame_files[frame_idx]
```

So `frame_stream[0]` means "the first selected frame", and `FrameData.raw_frame_idx` stores the actual index in the sorted file list after applying `frame_start` and `frame_skip`.

Iteration is re-iterable and simply yields `self[0]`, `self[1]`, ...:

```python
for frame_idx in range(len(self)):
    yield self[frame_idx]
```

For each selected frame:

```python
frame = cv2.imread(str(frame_path))              # H,W,3 BGR uint8 CPU
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # H,W,3 RGB uint8 CPU
frame_rgb = torch.as_tensor(frame).float().cuda() / 255.0

sensor_depth = None
if self.load_sensor_depth:
    depth_path = self.path.parent / "depth" / f"{frame_path.stem}.png"
    raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw_depth.shape[:2] != frame.shape[:2]:
        raw_depth = cv2.resize(raw_depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    sensor = raw_depth.astype(np.float32) / 1000.0
    sensor[~np.isfinite(sensor)] = 0.0
    sensor[sensor <= 0.0] = 0.0
    sensor_depth = torch.as_tensor(sensor).float().cuda()

return FrameData(raw_frame_idx=frame_idx, rgb=frame_rgb, sensor_depth=sensor_depth)
```

The resulting `FrameData` initially has:

```python
FrameData(
    raw_frame_idx=<integer index in sorted file list>,
    rgb=<CUDA tensor, shape H x W x 3, float32, range 0..1>,
    pose=None,
    camera_type=None,
    intrinsics=None,
    metric_depth=None,
    sensor_depth=<CUDA tensor H x W in meters if enabled, otherwise None>,
    depth_confidence=None,
)
```

Sensor depth loading is intentionally tied to the current ScanNet directory convention:

```text
<scene>/color/<frame_id>.jpg
<scene>/depth/<frame_id>.png
```

Depth PNG values are treated as millimeters and converted to meters. Nonfinite and nonpositive values become zero. If depth resolution differs from RGB resolution, it is resized to the RGB shape with nearest-neighbor interpolation so zero invalid pixels stay invalid.

#### Branches

| Branch | Behavior |
| --- | --- |
| `frame_end == -1` | Uses all files through the end. |
| `frame_skip > 1` | Reads every `frame_skip`-th image and divides FPS by `frame_skip`. |
| `pipeline.depth.use_gt_sens_depths == null` | `load_sensor_depth=False`; no sensor depth file is opened. |
| `pipeline.depth.use_gt_sens_depths in {"scale", "direct"}` | `load_sensor_depth=True`; missing sibling depth PNG raises immediately. |
| Image read fails | Raises `ValueError`. |

### Camera Initialization And Pinhole Normalization

Source files:

| File | Role |
| --- | --- |
| `vipe/pipeline.py` | External-camera initialization, GeoCalib initialization, distorted-camera estimation, pinhole-normalizing stream wrappers, `VipePipeline.run` |
| `vipe/streams/base.py` | `FrameDir` calibration-file loading and `SensorCamera` metadata |
| `vipe/priors/geocalib/*` | GeoCalib model and LM optimizer |

Stage context: this completes Stage 1. It produces the pinhole camera representation used by every later stage. If `streams.use_gt_intrinsics=true`, ViPE loads RGB/color intrinsics from disk and skips GeoCalib. If that loaded camera has OpenCV distortion metadata, it builds an OpenCV undistortion map and wraps the source in `OpenCVPinholeNormalizedFrameStream`. Otherwise GeoCalib samples three frames once. If `streams.input_camera_model=pinhole`, GeoCalib estimates one shared vertical FOV and converts that FOV into one raw-resolution pinhole intrinsics tensor. If the GeoCalib input model is distorted, it estimates the configured distorted camera, builds a pinhole undistortion map, and wraps the original source in `PinholeNormalizedFrameStream`. Any undistortion wrapper also produces `FrameData.image_valid_mask`, where false means the output pixel had no valid source RGB sample. The one-time handoff to Stage 2 is always pinhole: a frame stream that yields pinhole-normalized images plus one CUDA tensor `[fx, fy, cx, cy]`.

#### Diagram

```mermaid
flowchart TD
    A[Raw FrameDir] --> AA{use_gt_intrinsics=true?}
    AA -->|yes| AB[Load SensorCamera from intrinsic_color.json or intrinsic_color.txt]
    AB --> AC{loaded camera has distortion?}
    AC -->|yes| AD[Build OpenCV undistortion grid]
    AD --> AE[OpenCVPinholeNormalizedFrameStream]
    AC -->|no| AF[Raw stream already pinhole]
    AB --> AG[Use loaded output_K as fx,fy,cx,cy]
    AA -->|no| B[Compute sample indices 0,gap,2gap]
    B --> C[Read only sampled frames from FrameDir]
    C --> D{streams.input_camera_model}
    D -->|pinhole| E[GeoCalib pinhole shared intrinsics]
    E --> F[Convert vfov to fx,fy,cx,cy]
    D -->|distorted model| G[GeoCalib distorted per-sample camera estimates]
    G --> H[Average camera parameters]
    H --> I[Build pinhole undistortion grid]
    I --> J[PinholeNormalizedFrameStream]
    H --> K[Use camera.pinhole f,c as fx,fy,cx,cy]
    F --> L[Pinhole intrinsics tensor]
    K --> L
    AG --> L
    A --> M[Raw stream when already pinhole]
    M --> N[SLAM receives pinhole stream and intrinsics]
    AF --> N
    AE --> N
    J --> N
    L --> N
    N --> O[SLAM attaches intrinsics and PINHOLE per frame]
```

#### Pipeline Construction

`VipePipeline.run(frame_stream, input_camera_model, use_gt_intrinsics)` starts by normalizing the input camera and computing downstream pinhole intrinsics:

```python
frame_stream, intrinsics = self._initialize(frame_stream, input_camera_model, use_gt_intrinsics)
artifact_path = io.ArtifactPath(self.out_path, frame_stream.name())
slam_output = self._run_slam(frame_stream, intrinsics)
```

`artifact_path` only records where outputs for this frame stream will be written. The camera initialization returns a pinhole intrinsics tensor and either the original stream or a lightweight wrapper; it does not store every frame. GeoCalib is only imported and executed when direct sensor intrinsics are not enabled.

#### Direct Sensor Intrinsics

When `streams.use_gt_intrinsics=true`, `FrameDir` loads one color-camera file from the sibling `intrinsic/` directory. The JSON file is preferred because it can carry distortion metadata:

```text
<scene>/intrinsic/intrinsic_color.json
<scene>/intrinsic/intrinsic_color.txt
```

For ScanNet TXT, the file is a `4x4` matrix. ViPE reads `K = matrix[:3, :3]` and uses:

```python
intrinsics = torch.as_tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]).float().cuda()
```

TXT has no distortion coefficients, so the original `FrameDir` is already the downstream pinhole stream.

For JSON, ViPE reads `camera_matrix` as the raw input camera matrix, `projection_matrix` as the output pinhole camera matrix when distortion is present, and `distortion_coefficients` plus `distortion_model`. Supported OpenCV distortion models are `plumb_bob` and `rational_polynomial`. If distortion coefficients are present and nonzero, Stage 1 builds a single map:

```python
mapx, mapy = cv2.initUndistortRectifyMap(
    input_K,
    distortion_coefficients,
    np.eye(3, dtype=np.float32),
    output_K,
    (width, height),
    cv2.CV_32FC1,
)
```

That map is converted to a CUDA `grid_sample` grid. `OpenCVPinholeNormalizedFrameStream` then remaps RGB with bilinear sampling and remaps loaded `sensor_depth` with nearest-neighbor sampling. It also records which output pixels had source coordinates inside the original image as `image_valid_mask`; pixels outside the source image are zeroed in RGB/depth and marked invalid. The intrinsics handed to SLAM are from `output_K`.

When SLAM later iterates `frame_stream`, the yielded frames are already pinhole-compatible. `SLAMSystem.run` attaches the same tensor and `CameraType.PINHOLE` to each yielded `FrameData` before resizing it for SLAM.

#### GeoCalib Sampling

Stage 1 computes sample indices:

```python
gap_frame = int(gap_sec * frame_stream.fps())
gap_frame = min(gap_frame, (len(frame_stream) - 1) // 2)
sample_frame_inds = [0, gap_frame, gap_frame * 2]
```

Default `gap_sec` is `1.0`.

For a real ScanNet run at `fps=30` with 5578 frames:

```text
gap_frame = min(30, (5578 - 1) // 2) = 30
sample_frame_inds = [0, 30, 60]
```

For a concrete 10-frame, 2-FPS example:

```text
gap_frame = min(int(1.0 * 2), (10 - 1)//2) = min(2, 4) = 2
sample_frame_inds = [0, 2, 4]
```

For the default pinhole path, Stage 1 then:

1. Constructs `GeoCalib(weights="pinhole").cuda()`.
2. Iterates the `FrameDir` until the last sample index and stores the needed samples:
   ```python
   sample_frame_set = set(sample_frame_inds)
   sample_by_idx = {}
   for frame_idx, frame in enumerate(frame_stream):
       if frame_idx in sample_frame_set:
           sample_by_idx[frame_idx] = frame.rgb.moveaxis(-1, 0)
       if frame_idx >= sample_frame_inds[-1]:
           break
   sample_frames = torch.stack([sample_by_idx[i] for i in sample_frame_inds])
   ```
   Shape is `(3, 3, H0, W0)`.
3. Calls:
   ```python
   res = model.calibrate(sample_frames, shared_intrinsics=True)
   ```
4. Reads the shared vertical FOV:
   ```python
   fov_y = res["camera"].vfov[0].item()
   ```

GeoCalib internal sequence:

```mermaid
flowchart TD
    A[3 RGB sample frames C,H,W in range 0..1] --> B[ImagePreprocessor resize short side to 320, edge divisible by 32]
    B --> C[MSCAN high-level features]
    B --> D[LowLevelEncoder features]
    C --> E[PerspectiveDecoder]
    D --> E
    E --> F[up_field, up_confidence]
    E --> G[latitude_field, latitude_confidence]
    F --> H[LMOptimizer]
    G --> H
    H --> I[Initialize camera from 0.7 * max(h,w) focal]
    I --> J[Iteratively optimize gravity and focal]
    J --> K[Return camera vfov]
```

The calibration model predicts dense perspective fields:

| Field | Shape after postprocess | Meaning |
| --- | --- | --- |
| `up_field` | `(B, 2, H, W)` | 2D image-space up direction |
| `up_confidence` | `(B, H, W)` | Confidence for up field |
| `latitude_field` | `(B, 1, H, W)` | Predicted latitude angle |
| `latitude_confidence` | `(B, H, W)` | Confidence for latitude |

`LMOptimizer` creates an initial pinhole camera and gravity, computes analytic perspective fields for that camera, and minimizes residuals between predicted fields and analytic fields. It uses Huber-weighted residuals and Levenberg-Marquardt updates. With `shared_intrinsics=True`, all 3 sample frames share the same optimized focal length, but each frame can have its own gravity estimate.

After GeoCalib returns, ViPE converts vertical FOV to one pinhole intrinsics vector at the original input image size:

```python
frame_height, frame_width = frame_stream.frame_size()
fx = fy = frame_height / (2 * np.tan(fov_y / 2))
intrinsics = torch.as_tensor([fx, fy, frame_width / 2, frame_height / 2]).float().cuda()
```

Important: this code uses image height to compute both `fx` and `fy` from vertical FOV. `cx` and `cy` are image center coordinates.

For distorted input camera models:

1. Constructs `GeoCalib(weights="distorted").cuda()`.
2. Runs one GeoCalib calibration per sampled frame because the local GeoCalib optimizer supports shared intrinsics only for pinhole.
3. Averages the resulting camera parameter vectors.
4. Builds a remap grid from output pinhole pixels to input distorted pixels:
   ```text
   output pinhole pixel -> pinhole bearing -> distorted camera projection -> input distorted pixel
   ```
5. Wraps the original source in `PinholeNormalizedFrameStream`.

`PinholeNormalizedFrameStream` remaps RGB with bilinear sampling and remaps loaded `sensor_depth` with nearest-neighbor sampling. Like the OpenCV wrapper, it computes `image_valid_mask` from the undistortion grid and zeroes pixels that map outside the source image. It preserves the original frame count, FPS, selected-frame indices, and artifact name. Downstream SLAM, DAV3, backprojection, TSDF, and benchmark code still see only pinhole frames and `[fx, fy, cx, cy]`.

#### Handoff To SLAM

The original `FrameDir` preserves the stream metadata:

| Method | Returned value |
| --- | --- |
| `frame_size()` | original `FrameDir.frame_size()` |
| `fps()` | original effective FPS |
| `name()` | original stream name, such as `color` |
| `__len__()` | original selected frame count |

Every yielded `FrameData` still has original-resolution RGB. If the input was distorted, this RGB has been undistorted into the pinhole image plane first. Stage 2 adds:

```python
frame_data.intrinsics = shared_intrinsics  # CUDA tensor [fx, fy, cx, cy]
frame_data.camera_type = CameraType.PINHOLE
```

SLAM then performs its own resizing in Stage 2. That separation matters: GeoCalib/pinhole-normalization is computed at the raw frame size; SLAM scales and crops both RGB and intrinsics afterward.

## Stage 2: SLAM Pass 1 Frontend Loop

### High-Level Pseudocode

```python
def stage_2_slam_pass_1_frontend(frame_stream, intrinsics):
    resizer = StandardResizeFrameProcessor()
    total_n_frames = len(frame_stream)

    frame_size = resizer.update_frame_size(frame_stream.frame_size())
    config.height = frame_size[0]
    config.width = frame_size[1]
    config.camera_type = PINHOLE

    droid_net = DroidNet().cuda().eval()
    buffer = GraphBuffer(height=config.height, width=config.width, ...)
    motion_filter = MotionFilter(droid_net, thresh=config.filter_thresh)
    frontend = SLAMFrontend(droid_net, buffer, config)
    metric_depth_model = DepthAnything3Model(configured_keyframe_model)

    for frame_idx, frame_data in enumerate(frame_stream):
        frame_data.intrinsics = intrinsics
        frame_data.camera_type = PINHOLE
        frame_data = resizer(frame_data)
        images = frame_data.rgb.permute(2, 0, 1)[None]  # BCHW

        motion_result = motion_filter.check(images)
        is_last_frame = frame_idx == total_n_frames - 1

        if motion_result.is_keyframe or is_last_frame:
            kf_idx = buffer.n_frames
            buffer.tstamp[kf_idx] = frame_idx
            buffer.images[kf_idx] = images[0]
            buffer.fmaps[kf_idx] = motion_result.fmap[0]
            net, inp = motion_result.net, motion_result.inp
            if net is None or inp is None:
                net, inp = droid_net.encode_context(images)
            buffer.nets[kf_idx], buffer.inps[kf_idx] = net[0], inp[0]

            if kf_idx == 0:
                buffer.intrinsics = frame_data.intrinsics

            buffer.n_frames += 1
            buffer.update_disps_sens(metric_depth_model, frame_idx=kf_idx)

        if not frontend.is_initialized and buffer.n_frames == config.warmup:
            frontend.t1 = buffer.n_frames
            frontend.graph.add_neighborhood_factors(0, frontend.t1, r=1)
            for _ in range(config.frontend_init_updates):
                frontend.graph.update(
                    t0=1,
                    itrs=config.frontend_ba_iters,
                )

            frontend.initialize_next_pose_and_disparity()
            frontend.is_initialized = True

        elif frontend.is_initialized and frontend.t1 < buffer.n_frames:
            frontend.graph.rm_factors(frontend.graph.age > config.frontend_max_age)
            frontend.graph.add_proximity_factors(...)

            for _ in range(config.frontend_update_iters1):
                frontend.graph.update(itrs=config.frontend_ba_iters)

            if second_newest_keyframe_is_too_close():
                frontend.graph.rm_second_newest_keyframe(...)
            else:
                for _ in range(config.frontend_update_iters2):
                    frontend.graph.update(itrs=config.frontend_ba_iters)

            frontend.initialize_next_pose_and_disparity()

    return buffer  # keyframe poses/disparities/features/depth anchors for Stage 3
```

### SLAM Setup: Resize, Graph Buffer, And Models

Source files:

| File | Role |
| --- | --- |
| `vipe/slam/system.py` | `SLAMSystem`, resize processor, two SLAM passes |
| `vipe/slam/components/buffer.py` | Persistent keyframe graph state |
| `vipe/slam/networks/droid_net.py` | DROID feature/context/update networks |
| `vipe/priors/depth/dav3.py` | Configured depth-anchor model used on SLAM keyframes |

Stage context: this is the sequence-once setup for Stage 2. It computes the resized SLAM frame size from the raw `FrameDir`, builds the DROID network, allocates the keyframe `GraphBuffer`, and prepares the configured keyframe depth-anchor provider. The handoff is an empty but fully allocated SLAM state ready for the pass-1 frame loop.

#### Diagram

```mermaid
flowchart TD
    A[FrameDir plus shared intrinsics tensor] --> B[SLAMSystem.run]
    B --> C[StandardResizeFrameProcessor]
    C --> D[Compute resized frame size]
    D --> E[Update config: height, width, camera_type]
    E --> F[_build_components]
    F --> F1[DroidNet]
    F --> F2[GraphBuffer tensors]
    F --> F3[MotionFilter]
    F --> F4[SLAMFrontend]
    F --> F5[SLAMBackend]
    F --> F6[InnerFiller]
    F --> F7[Configured keyframe depth-anchor provider]
```

#### Standard Resize And Crop

SLAM never works at the original frame size directly. It creates one resize processor:

```python
resizer = StandardResizeFrameProcessor()
frame_size = resizer.update_frame_size(frame_stream.frame_size())
```

During each SLAM pass, every raw `FrameData` is assigned the shared intrinsics and then resized/cropped inline:

```python
frame_data.intrinsics = intrinsics
frame_data.camera_type = CameraType.PINHOLE
frame_data = resizer(frame_data)
```

For each frame, `StandardResizeFrameProcessor._compute_frame_size_crop` computes:

```python
h0, w0 = previous_frame_size
scale_factor = sqrt((384 * 512) / (h0 * w0))
h1 = int(h0 * scale_factor)
w1 = int(w0 * scale_factor)

crop_h = h1 % 8
crop_w = w1 % 8
crop_top = crop_h // 2
crop_bottom = crop_h - crop_top
crop_left = crop_w // 2
crop_right = crop_w - crop_left

self.fac_x = w0 / w1
self.fac_y = h0 / h1
self.scx = crop_left
self.scy = crop_top
```

Then `__call__` does:

```python
frame_data = frame_data.resize((h1, w1))
frame_data = frame_data.crop(top=crop_top, bottom=crop_bottom, left=crop_left, right=crop_right)
```

`FrameData.resize` changes:

| Field | Resize rule |
| --- | --- |
| `rgb` | bilinear interpolation |
| `metric_depth` | bilinear interpolation if present |
| `sensor_depth` | nearest-neighbor interpolation if present, so invalid zero pixels remain invalid |
| `image_valid_mask` | nearest-neighbor interpolation if present |
| `depth_confidence` | bilinear interpolation if present |
| `intrinsics` | `fx,cx` scaled by `w1/w0`, `fy,cy` scaled by `h1/h0` |

`FrameData.crop` changes:

| Field | Crop rule |
| --- | --- |
| `rgb`, `metric_depth`, `sensor_depth`, `image_valid_mask`, `depth_confidence` | slice `[top:bottom, left:right]` |
| `intrinsics` | `cx -= left`, `cy -= top` |

The resized dimensions must be divisible by 8 because DROID features and disparities live at `1/8` resolution.

#### GraphBuffer State

After resize, `SLAMSystem.run` sets config values:

```python
self.config.update({
    "height": frame_size[0],
    "width": frame_size[1],
    "camera_type": camera_type,
})
```

For the current standalone path:

```text
camera_type = PINHOLE
```

`GraphBuffer` allocates fixed-size CUDA tensors:

| Buffer | Example shape | Meaning |
| --- | --- | --- |
| `tstamp` | `(1024,)` | raw frame index for each keyframe/buffer frame |
| `images` | `(1024,3,384,512)` float16 | resized RGB images |
| `poses` | `(1024,7)` | SE3 world-to-camera pose as translation + quaternion |
| `intrinsics` | `(4,)` | resized pinhole intrinsics |
| `disps` | `(1024,48,64)` | optimized inverse depth/disparity |
| `disps_sens` | `(1024,48,64)` | sampled inverse-depth anchor |
| `disps_sens_weight` | `(1024,48,64)` | per-anchor BA weight, zero for invalid DAV3/sensor/image-mask pixels |
| `fmaps` | `(1024,128,48,64)` half | DROID feature maps |
| `nets` | `(1024,128,48,64)` half | DROID GRU hidden state |
| `inps` | `(1024,128,48,64)` half | DROID context input |

`poses` are initialized to identity:

```text
[tx, ty, tz, qx, qy, qz, qw] = [0, 0, 0, 0, 0, 0, 1]
```

`disps` is initialized to `init_disp`, default `1.0`. That corresponds to initial depth `1 / 1.0 = 1.0 m`.

#### DROID Network Setup

`DroidNet()` contains:

| Module | Purpose |
| --- | --- |
| `fnet = BasicEncoder(output_dim=128, norm_fn="instance")` | Feature maps for correlation |
| `cnet = BasicEncoder(output_dim=256, norm_fn="none")` | Context split into hidden state and input |
| `update = UpdateModule()` | RAFT/DROID-style recurrent update that predicts coordinate deltas, weights, and damping |

`load_weights` checks:

```python
ckpt_path = Path(torch.hub.get_dir()) / "droid_slam" / "droid.pth"
```

If missing, it downloads the DROID checkpoint, strips `module.` prefixes, adapts the 4-channel original output heads down to 2 channels, loads weights, and sets eval mode.

#### Keyframe Depth Anchor Provider

`SLAMSystem._build_components` always creates:

```python
self.metric_depth = DepthAnything3Model(
    self.keyframe_depth_model,
    self.use_gt_sens_depths,
)
```

For the default `pipeline.depth.use_gt_sens_depths=null`, this is `DepthAnything3Model`, using:

```python
DepthAnything3.from_pretrained(pipeline.depth.keyframe_model)
```

This keyframe model is separate from the final post-processing multiview DAV3 model. It is used to anchor SLAM keyframe inverse depths during bundle adjustment.

For the sensor-depth modes:

| Mode | Keyframe depth model behavior |
| --- | --- |
| `null` | Run `pipeline.depth.keyframe_model` normally and use its metric depth as the BA depth anchor. |
| `scale` | Run the same DAV3 keyframe model, then multiply its depth by one scalar fitted to the loaded sensor depth for that same resized/cropped keyframe. |
| `direct` | Do not load or run the DAV3 keyframe model. Use loaded sensor depth directly and pass a valid-pixel mask into the BA depth-anchor term. |

### Pass 1 Loop, Motion Filtering, Keyframes, And Frontend BA

Source files:

| File | Role |
| --- | --- |
| `vipe/slam/system.py` | Pass 1 loop, `_rgb_bchw`, `_add_frontend_keyframe` |
| `vipe/slam/components/motion_filter.py` | Keyframe decision |
| `vipe/slam/components/frontend.py` | Warmup, incremental graph updates, keyframe pruning |
| `vipe/slam/components/factor_graph.py` | Correlation, update operator, graph edges |
| `vipe/slam/components/buffer.py` | Dense BA call and frame-distance metric |
| `vipe/slam/ba/*` | Sparse solver and residual terms |
| `vipe/slam/maths/geom.py` | Inverse projection, projection, SE3 transforms |

Stage context: this is the per-frame loop of Stage 2. It consumes the original `FrameDir` once, attaches the shared intrinsics, and resizes each frame to SLAM resolution. Every frame is converted to BCHW and runs the motion filter. The motion filter computes DROID features for every frame for keyframe selection, while accepted keyframes are stored in `GraphBuffer` with their DROID feature/context tensors plus the configured keyframe depth anchors. Frontend BA runs incrementally as keyframes arrive. The handoff to Stage 3 is the keyframe buffer with frontend-optimized poses and disparities.

#### Diagram

```mermaid
flowchart TD
    A[SLAM Pass 1 frame loop] --> B[_rgb_bchw]
    B --> C[images B,C,H,W]
    C --> D[MotionFilter.check]
    D --> E{First frame or dense motion > threshold or last frame?}
    E -->|yes| F[_add_frontend_keyframe]
    E -->|no| G[Do not add to GraphBuffer]
    F --> H[Store image, features, context, intrinsics, sampled inverse-depth anchor]
    G --> I[frontend.run]
    H --> I
    I --> J{Frontend initialized?}
    J -->|no and n_frames == warmup| K[Initialize graph with neighborhood factors]
    J -->|yes and new keyframe exists| L[Incremental graph update]
    J -->|otherwise| M[No frontend operation]
    K --> N[Graph update and BA]
    L --> N
    N --> O{Prune second-newest keyframe if too close}
    O -->|distance < keyframe_thresh| P[Remove keyframe]
    O -->|distance >= keyframe_thresh| Q[Keep keyframe]
```

#### Per-Frame BCHW Conversion

Each pass-1 iteration receives one `FrameData` from:

```python
for frame_idx, frame_data in enumerate(frame_stream):
```

`_rgb_bchw(frame_data)`:

```python
images = frame_data.rgb.permute(2, 0, 1)[None]
return images
```

Example output shapes:

```python
images.shape == (1, 3, 384, 512)
```

The batch dimension is `1` because this runtime path has one camera and one frame directory.

This helper only changes tensor layout and adds a batch dimension. DROID feature computation happens inside `MotionFilter.check(images)`. Accepted keyframes reuse the motion-filter feature map instead of recomputing it during buffer storage.

#### MotionFilter Keyframe Decision

`MotionFilter.check(images)` runs on every frame.

First frame branch:

```python
gmap = self.net.encode_features(images)
if not self.initialized:
    net, inp = self.net.encode_context(images)
    self.f_net, self.f_inp, self.f_fmap = net, inp, gmap
    self.current_frame_idx = 0
    self.last_kf_frame_idx = 0
    self.initialized = True
    return MotionFilterResult(True, gmap, net, inp)
```

So the first frame is always a keyframe.

Later-frame branch:

1. Encode current feature map:
   ```python
   gmap = self.net.encode_features(images)
   ```
   Example shape: `(1,128,48,64)`.
2. Create low-res coordinate grid:
   ```python
   coords0.shape == (1,1,48,64,2)
   ```
   Pixel `(x=10,y=5)` has coordinate `[10,5]`.
3. Build correlation between last keyframe feature map and current feature map:
   ```python
   corr = CorrBlock(self.f_fmap[None], gmap[None])(coords0)
   ```
4. Run one learned DROID update:
   ```python
   _, delta, weight = self.net.update.forward(self.f_net[None], self.f_inp[None], corr)
   ```
   `delta.shape == (1,1,48,64,2)`.
5. Compute per-pixel flow magnitude:
   ```python
   dense_flow = delta.norm(dim=-1)[0]  # shape (1,48,64)
   ```
6. Average flow magnitude over all low-res pixels:
   ```python
   dense_motion_score = dense_flow.mean([1, 2]).item()
   ```
7. Return a `MotionFilterResult` with `is_keyframe=True` if:
   ```python
   dense_motion_score > self.thresh
   ```
   Default threshold is `2.4`.

`SLAMSystem.run` also forces the last frame to be a keyframe:

```python
motion_result = self.motion_filter.check(images)
if motion_result.is_keyframe or frame_idx == total_n_frames - 1:
    self._add_frontend_keyframe(...)
```

#### Keyframe Addition

When a frame is accepted as a keyframe, `_add_frontend_keyframe(...)` runs.

Let:

```python
kf_idx = self.buffer.n_frames
```

The function stores:

```python
self.buffer.tstamp[kf_idx] = frame_idx
self.buffer.images[kf_idx] = images[0]
self.buffer.fmaps[kf_idx] = motion_result.fmap[0]
net, inp = motion_result.net, motion_result.inp
if net is None or inp is None:
    net, inp = self.droid_net.encode_context(images)
self.buffer.nets[kf_idx], self.buffer.inps[kf_idx] = net[0], inp[0]
```

For normal accepted keyframes, `MotionFilter.check` has already computed both `fmap` and context tensors, so `_store_buffer_frame` reuses them. For the last frame forced into the keyframe set even when its motion score is below threshold, the feature map is still reused and only the missing context tensors are computed.

```python
if kf_idx == 0:
    self.buffer.intrinsics = frame_data.intrinsics
```

Current initialized frames do not have output `metric_depth` yet. The keyframe depth-anchor path below fills `disps_sens` from the configured source.

Then Stage 2 fills the keyframe depth anchor:

```python
self.buffer.update_disps_sens(self.metric_depth, frame_idx=kf_idx, frame_data=frame_data)
```

`GraphBuffer.update_disps_sens` creates:

```python
DepthEstimationInput(
    rgb=self.images[frame_idx].moveaxis(0, -1).float(),  # H,W,3
    intrinsics=self.intrinsics,
    sensor_depth=frame_data.sensor_depth,
    image_valid_mask=frame_data.image_valid_mask,
    camera_type=self.camera_type,
)
```

`DepthAnything3Model.estimate` has three exact branches controlled by `pipeline.depth.use_gt_sens_depths`.

Default branch, `null`:

1. Converts RGB to uint8 numpy list.
2. Runs the configured single-image metric depth model, default `depth-anything/DA3METRIC-LARGE`:
   ```python
   self.model.inference(rgb_images, process_res_method="upper_bound_resize", process_res=504)
   ```
3. Computes focal scaling:
   ```python
   dav3_camera_focal = focal_length / max(width, height) * 504
   dav3_metric_depth = dav3_result.depth * dav3_camera_focal / 300.0
   ```
4. Zeroes sky depth from DA3 sky mask.
5. Interpolates to the SLAM image size.
6. Builds a depth-valid mask from positive finite DAV3 depth, intersected with `frame_data.image_valid_mask` when camera normalization created one.

Scale branch, `scale`:

1. Runs the same DAV3 branch above.
2. Uses the resized/cropped `frame_data.sensor_depth` from the same keyframe.
3. Computes one scalar:
   ```math
   s^\* = \frac{\sum_{u \in \mathcal{V}} D_{\text{dav3}}(u) D_{\text{sens}}(u)}
                {\sum_{u \in \mathcal{V}} D_{\text{dav3}}(u)^2}
   ```
   where:
   ```math
   \mathcal{V} = \{u \mid D_{\text{dav3}}(u) > 0,\; D_{\text{sens}}(u) > 0,\;
   D_{\text{dav3}}(u), D_{\text{sens}}(u) \text{ finite},\; M_{\text{img}}(u)=1\}
   ```
   where `M_img` is the image-valid mask; if no undistortion wrapper is active, it is effectively all ones.
4. Replaces:
   ```math
   D_{\text{keyframe}}(u) = s^\* D_{\text{dav3}}(u)
   ```
5. Sets invalid image/depth pixels to zero and returns the same valid mask to BA.

Direct branch, `direct`:

1. Does not load or run DAV3.
2. Uses the resized/cropped `frame_data.sensor_depth` tensor directly.
3. Builds:
   ```python
   valid = torch.isfinite(sensor_depth) & (sensor_depth > 0.0) & image_valid_mask
   metric_depth = torch.where(valid, sensor_depth, torch.zeros_like(sensor_depth))
   valid_mask = valid.float()
   ```
4. Returns `valid_mask` with the depth result so invalid sensor pixels and undistort-invalid image pixels do not become zero-depth or zero-disparity targets in BA.

Back in `GraphBuffer.update_disps_sens`, ViPE converts metric depth to inverse depth at the DROID grid sample positions:

```python
disp_sens = metric_depth[3::8, 3::8]
disp_sens = torch.where(disp_sens > 0, disp_sens.reciprocal(), disp_sens)
self.disps_sens[frame_idx] = disp_sens
if result.valid_mask is None:
    self.disps_sens_weight[frame_idx] = 1.0
else:
    self.disps_sens_weight[frame_idx] = result.valid_mask[3::8, 3::8].float()
```

With the current DAV3 depth model, `valid_mask` is returned in all three depth modes. The fallback only preserves the old all-ones behavior if a future depth model omits masks.

For a `384 x 512` SLAM image, `3::8` gives `48` rows and `64` columns. It samples pixel centers offset by 3 at each 8-pixel block.

The BA sensor-depth regularizer is:

```math
E_{\text{sens}} = \alpha \sum_k w_k \left(d_k - d_{\text{sens},k}\right)^2
```

where `w_k=0` at invalid depth pixels and at pixels marked invalid by camera undistortion. In `direct` mode this also masks invalid sensor-depth pixels. Therefore invalid sensor-depth zeros, DA3 sky zeros, and undistort padding zeros do not push optimized disparities toward zero.

Finally:

```python
self.buffer.n_frames += 1
```

#### Frontend Initialization

`self.frontend.run()` is called every frame, but it only acts when enough keyframes are present or a new keyframe arrives after initialization.

Initialization branch:

```python
if not self.is_initialized and self.video.n_frames == self.warmup:
    self.__initialize()
```

Default `warmup=8`.

`__initialize`:

1. Sets:
   ```python
   self.t1 = self.video.n_frames
   ```
2. Adds neighborhood factors:
   ```python
   self.graph.add_neighborhood_factors(0, self.t1, r=1)
   ```
   The reduced path always uses adjacent sequential initialization, so `r=1`.
3. Runs `frontend_init_updates` graph updates, default `8`:
   ```python
   for _ in range(self.frontend_init_updates):
       self.graph.update(t0=1, itrs=self.frontend_ba_iters)
   ```
4. Initialize the next pose by constant velocity from the latest optimized keyframe poses:
   ```python
   self.__init_pose()
   ```
5. Set next disparity initial value to mean of recent disparities:
   ```python
   self.video.disps[self.t1] = self.video.disps[self.t1 - 4:self.t1].mean()
   ```
6. Set `self.is_initialized=True`.
7. Remove factors older than `warmup - 4`:
   ```python
   self.graph.rm_factors(self.graph.ii < self.warmup - 4)
   ```

#### Frontend Incremental Update

After initialization, when a new keyframe exists:

```python
elif self.is_initialized and self.t1 < self.video.n_frames:
    self.__update()
```

`__update`:

1. Increments optimized keyframe count:
   ```python
   self.t1 += 1
   ```
2. Drops old active factors whose age is too high:
   ```python
   self.graph.rm_factors(self.graph.age > self.frontend_max_age)
   ```
3. Adds proximity factors:
   ```python
   self.graph.add_proximity_factors(
       self.t1 - 5,
       max(self.t1 - self.frontend_window, 0),
       rad=self.frontend_radius,
       nms=self.frontend_nms,
       thresh=self.frontend_thresh,
       beta=self.beta,
       remove=True,
   )
   ```
4. Runs `frontend_update_iters1` graph updates:
   ```python
   for _ in range(self.frontend_update_iters1):  # default 4
       self.graph.update(itrs=self.frontend_ba_iters)
   ```
5. Computes dense-disparity frame distance between the second-newest and third-newest keyframes:
   ```python
   d = self.video.frame_distance_dense_disp(t1-3, t1-2, beta=self.beta, bidirectional=True)
   ```
6. If `d.max() < keyframe_thresh`, remove the second-newest keyframe:
   ```python
   self.graph.rm_second_newest_keyframe(self.t1 - 2)
   self.t1 -= 1
   ```
7. Else run `frontend_update_iters2` more graph updates:
   ```python
   for _ in range(self.frontend_update_iters2):  # default 2
       self.graph.update(itrs=self.frontend_ba_iters)
   ```
8. Seed the next buffer slot pose and disparity for the next possible keyframe.

#### Factor Graph Update

`FactorGraph.update` is the central DROID-style learned optimization step.

For all active graph edges `(i,j)`:

1. Reproject current dense disparity from source frame `i` into target frame `j`:
   ```python
   coords1, _ = self.buffer.reproject_dense_disp(self.ii, self.jj)
   coords1.shape == (num_edges, 48, 64, 2)
   ```
2. Build motion features:
   ```python
   motn = torch.cat([coords1 - coords0, self.target - coords1], dim=-1)
   ```
   Per low-res pixel, this is:
   ```text
   [current_rigid_flow_x, current_rigid_flow_y, previous_residual_x, previous_residual_y]
   ```
3. Sample correlation features at `coords1`:
   ```python
   corr = self.corr(coords1)
   ```
4. Run learned update:
   ```python
   self.f_net, delta, weight, damping, _ = self.net.update.forward(...)
   ```
   Outputs:
   | Output | Shape | Meaning |
   | --- | --- | --- |
   | `delta` | `(1,E,48,64,2)` | learned correction to projected coordinates |
   | `weight` | `(1,E,48,64,2)` | per-residual confidence in x and y |
   | `damping` | `(1,unique_dense_disps,48,64)` | dense disparity damping |
5. Set new target coordinates:
   ```python
   self.target = coords1 + delta
   self.weight = weight
   self.damping[di] = damping
   ```
6. Run dense bundle adjustment:
   ```python
   self.buffer.bundle_adjustment(...)
   ```

#### Dense Bundle Adjustment Terms

`GraphBuffer.bundle_adjustment` builds a `Solver` with two term families in the current reduced path:

| Term | Active when | Residual |
| --- | --- | --- |
| `DenseDepthFlowTerm` | Always for graph edges | projected coordinate minus learned target coordinate |
| `DispSensRegularizationTerm` | When `disps_sens` exists for a keyframe | optimized disparity minus sampled inverse-depth anchor |

Let:

$$
\begin{aligned}
P_i &\in SE(3) && \text{world-to-camera pose for frame } i, \\
K &= (f_x, f_y, c_x, c_y) && \text{low-resolution pinhole intrinsics}, \\
d_i(\mathbf{p}) &> 0 && \text{optimized inverse depth / disparity at low-res pixel } \mathbf{p}, \\
\hat{\mathbf{p}}_{ij}(\mathbf{p}) &\in \mathbb{R}^2 && \text{learned DROID target coordinate in frame } j, \\
\mathbf{w}_{ij}(\mathbf{p}) &\in \mathbb{R}^2 && \text{learned x/y residual weights.}
\end{aligned}
$$

For low-resolution pixel $\mathbf{p}=(u,v)$ in source frame $i$, inverse projection is:

$$
z_i(\mathbf{p}) = \frac{1}{d_i(\mathbf{p})}
$$

$$
\Pi_K^{-1}(\mathbf{p}, d_i) =
\begin{bmatrix}
(u-c_x)z_i/f_x \\
(v-c_y)z_i/f_y \\
z_i \\
1
\end{bmatrix}.
$$

The source camera point is moved from camera $i$ to camera $j$ with the current poses:

$$
\mathbf{X}_{j}(\mathbf{p}) =
P_j P_i^{-1}\Pi_K^{-1}(\mathbf{p}, d_i).
$$

The pinhole projection is:

$$
\Pi_K(\mathbf{X}) =
\begin{bmatrix}
f_x X/Z + c_x \\
f_y Y/Z + c_y
\end{bmatrix}.
$$

So the dense flow residual for edge $(i,j)$ and pixel $\mathbf{p}$ is:

$$
\mathbf{r}^{\text{flow}}_{ij,\mathbf{p}} =
\Pi_K\!\left(P_j P_i^{-1}\Pi_K^{-1}(\mathbf{p}, d_i)\right)
- \hat{\mathbf{p}}_{ij}(\mathbf{p}).
$$

The solver uses componentwise weights. In code, the learned weights are additionally scaled by `weight_dense_disp = 0.001`, and invalid projections get zero weight:

$$
\tilde{\mathbf{w}}_{ij,\mathbf{p}}
= 10^{-3}\,\mathbf{w}_{ij,\mathbf{p}}\odot \mathbf{m}_{ij,\mathbf{p}},
$$

where $\mathbf{m}_{ij,\mathbf{p}}\in\{0,1\}^2$ is the valid-projection mask broadcast over x/y components.

The depth-anchor disparity regularizer uses:

$$
r^{\text{sens}}_{i,\mathbf{p}} =
d_i(\mathbf{p}) - d^{\text{sens}}_i(\mathbf{p}).
$$

It also has a per-pixel scalar weight:

$$
w^{\text{sens}}_{i,\mathbf{p}} \in \{0,1\}.
$$

For `null` and `scale`, this weight is `1` everywhere. For `direct`, it is `1` where loaded sensor depth is valid and `0` where loaded sensor depth is invalid.

With `dense_disp_alpha = 0.001`, the BA objective for one solve is:

$$
\min_{\{P_i\},\{d_i\}}
\sum_{(i,j)\in\mathcal{E}}
\sum_{\mathbf{p}\in\Omega}
\left[
\tilde{w}^{x}_{ij,\mathbf{p}}\left(r^{\text{flow},x}_{ij,\mathbf{p}}\right)^2
+
\tilde{w}^{y}_{ij,\mathbf{p}}\left(r^{\text{flow},y}_{ij,\mathbf{p}}\right)^2
\right]
+
\alpha
\sum_{i\in\mathcal{S}}
\sum_{\mathbf{p}\in\Omega}
w^{\text{sens}}_{i,\mathbf{p}}
\left(d_i(\mathbf{p}) - d^{\text{sens}}_i(\mathbf{p})\right)^2.
$$

$K$ is fixed during this minimization. When `motion_only=True`, dense disparity is also fixed, so the solve only updates poses.

Each nonlinear iteration linearizes residuals around the current state:

$$
\mathbf{r}(\mathbf{x}+\Delta\mathbf{x})
\approx
\mathbf{r}(\mathbf{x}) + J\Delta\mathbf{x}.
$$

The normal equations assembled by the solver are:

$$
\left(J^\top W J + \Lambda\right)\Delta\mathbf{x}
=
-J^\top W\mathbf{r}.
$$

Here $W$ is the diagonal matrix represented by the per-residual weights, and $\Lambda$ is the damping added by `set_damping`.

Dense disparities are marked as marginalized. Splitting the state update into regular variables $\Delta\mathbf{x}_r$ and marginalized dense-disparity variables $\Delta\mathbf{x}_m$ gives:

$$
\begin{bmatrix}
H_{rr} & H_{rm} \\
H_{mr} & H_{mm}
\end{bmatrix}
\begin{bmatrix}
\Delta\mathbf{x}_r \\
\Delta\mathbf{x}_m
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{b}_r \\
\mathbf{b}_m
\end{bmatrix}.
$$

The Schur-complement reduced system solved first is:

$$
\left(H_{rr} - H_{rm}H_{mm}^{-1}H_{mr}\right)\Delta\mathbf{x}_r
=
\mathbf{b}_r - H_{rm}H_{mm}^{-1}\mathbf{b}_m.
$$

Then dense-disparity updates are recovered by:

$$
\Delta\mathbf{x}_m =
H_{mm}^{-1}\left(\mathbf{b}_m - H_{mr}\Delta\mathbf{x}_r\right).
$$

The solver:

1. Evaluates residuals and Jacobians.
2. Builds sparse damped normal equations.
3. Applies damping.
4. Marginalizes dense disparity with Schur complement.
5. Solves the reduced sparse system with SciPy sparse `spsolve`.
6. Applies updates using retractions:
   | Variable | Retraction |
   | --- | --- |
   | `pose` | SE3 pose retraction |
   | `dense_disp` | dense disparity additive/retracted update |

Intrinsics remain the GeoCalib estimate at resized scale during SLAM.

## Stage 3: Backend Global BA Over Keyframes

### High-Level Pseudocode

```python
def stage_3_backend_global_ba(buffer):
    t = buffer.n_frames  # number of pass-1 keyframes

    graph = FactorGraph(
        net=droid_net,
        buffer=buffer,
        max_factors=config.backend_max_factors_per_keyframe * t,
        incremental=False,
    )

    graph.add_proximity_factors(
        rad=config.backend_radius,
        nms=config.backend_nms,
        thresh=config.backend_thresh,
        beta=config.beta,
    )

    if len(graph.ii) == 0:
        buffer.disps[0] = where(buffer.disps_sens[0] > 0, buffer.disps_sens[0], buffer.disps[0])
        return buffer

    corr_op = AltCorrBlock(buffer.fmaps[None])

    for step in range(config.backend_iters):
        coords1, _ = buffer.reproject_dense_disp(graph.ii, graph.jj)
        motn = build_motion_features(coords1, graph.coords0, graph.target)

        for source_chunk in chunks_by_source_keyframe(graph.ii, chunk_size=8):
            iis = graph.ii[source_chunk]
            jjs = graph.jj[source_chunk]

            corr = corr_op(coords1[:, source_chunk], iis, jjs)
            net, delta, weight, damping, _ = droid_net.update(
                graph.f_net[:, source_chunk],
                buffer.inps[iis][None],
                corr,
                motn[:, source_chunk],
                ix=unique_source_inverse_indices(iis),
            )

            graph.f_net[:, source_chunk] = net
            graph.target[:, source_chunk] = coords1[:, source_chunk] + delta.float()
            graph.weight[:, source_chunk] = weight.float()
            graph.damping[unique(iis)] = damping

        buffer.bundle_adjustment(
            target=flatten_hw(graph.target),
            weight=flatten_hw(graph.weight),
            disp_damping=graph.damping,
            ii=graph.ii,
            jj=graph.jj,
            t0=1,
            t1=t,
            n_iters=config.backend_ba_iters,
            pose_damping=1e-5,
            pose_ep=1e-2,
            motion_only=False,
            verbose=True,
        )

    return buffer  # same GraphBuffer, globally refined in-place
```

### Backend Graph And Batch Updates

Source files:

| File | Role |
| --- | --- |
| `vipe/slam/components/backend.py` | Global BA over keyframes |
| `vipe/slam/components/factor_graph.py` | Batch graph update with low-memory correlation |
| `vipe/slam/components/buffer.py` | Global frame-distance graph state |

Stage context: this is Stage 3. It is a sequence-level solve over the pass-1 keyframes. It does not loop over raw frames. The handoff into this stage is the complete keyframe buffer from Stage 2. The handoff out is the same keyframe buffer with globally refined keyframe poses/disparities.

#### Diagram

```mermaid
flowchart TD
    A[Pass 1 complete with keyframes] --> B[backend.run steps=backend_iters]
    B --> C[Build non-incremental FactorGraph]
    C --> D[Add backend proximity factors]
    D --> E{Any graph edges?}
    E -->|yes| F[update_batch with AltCorrBlock]
    E -->|no| G[Use sensor depth for single keyframe]
    F --> H[Dense BA over all keyframes]
    H --> K[Refined keyframe poses and disparities in GraphBuffer]
```

#### Backend Graph Creation

After pass 1:

```python
self.backend.run(self.config.backend_iters)
```

Default:

```text
backend_iters = 31
```

`SLAMBackend.run` creates a fresh non-incremental `FactorGraph`:

```python
graph = FactorGraph(
    self.net,
    self.video,
    self.device,
    max_factors=self.args.backend_max_factors_per_keyframe * t,
    incremental=False,
)
```

The backend adds denser proximity factors:

```python
graph.add_proximity_factors(
    rad=self.args.backend_radius,
    nms=self.args.backend_nms,
    thresh=self.args.backend_thresh,
    beta=self.args.beta,
)
```

Defaults:

```text
backend_radius = 2
backend_nms = 3
backend_thresh = 22.0
beta = 0.3
```

`add_proximity_factors` adds:

1. Local neighborhood edges around each keyframe.
2. Additional edges whose frame-distance score is below threshold.
3. Bidirectional pairs for accepted edges.

#### Backend Batch Update

If the graph has edges, the backend calls `_iterate`.

`_iterate(graph, steps)`:

```python
graph.update_batch(
    itrs=self.args.backend_ba_iters,
    steps=steps,
    batch_size=self.args.backend_batch_size,
    solver_verbose=True,
)
```

The backend does not recompute keyframe depth anchors. Sampled sensor/DAV3 disparities are already stored in `buffer.disps_sens` when each pass-1 keyframe is added, and `GraphBuffer.bundle_adjustment` uses those stored values through `DispSensRegularizationTerm`.

`FactorGraph.update_batch` uses `AltCorrBlock` instead of materializing all correlation volumes:

```python
corr_op = AltCorrBlock(self.buffer.fmaps[None])
```

For each batch step:

1. Reproject current dense disparities to get `coords1`.
2. Build motion features.
3. Process graph edges in source-index chunks of size `backend_batch_size`.
4. Sample alt correlation only for the chunk.
5. Run the DROID update module.
6. Update target coordinates, weights, and damping.
7. Run dense BA over all keyframes.

## Stage 4: SLAM Pass 2 Pose Infill

### High-Level Pseudocode

```python
def stage_4_slam_pass_2_pose_infill(refined_keyframe_buffer, frame_stream, intrinsics, resizer):
    inner_filler.start_after_keyframes(refined_keyframe_buffer.n_frames)
    keyframe_count = refined_keyframe_buffer.n_frames
    keyframe_indices = refined_keyframe_buffer.tstamp[:keyframe_count].cpu().tolist()
    total_n_frames = len(frame_stream)

    for frame_idx, frame_data in enumerate(frame_stream):
        frame_data.intrinsics = intrinsics
        frame_data.camera_type = PINHOLE
        frame_data = resizer(frame_data)
        images = frame_data.rgb.permute(2, 0, 1)[None]  # BCHW

        append_idx = refined_keyframe_buffer.n_frames
        refined_keyframe_buffer.tstamp[append_idx] = frame_idx
        refined_keyframe_buffer.images[append_idx] = images[0]
        refined_keyframe_buffer.fmaps[append_idx] = droid_net.encode_features(images)[0]
        net, inp = droid_net.encode_context(images)
        refined_keyframe_buffer.nets[append_idx], refined_keyframe_buffer.inps[append_idx] = net[0], inp[0]
        refined_keyframe_buffer.n_frames += 1

        is_chunk_ready = refined_keyframe_buffer.n_frames - keyframe_count >= config.infill_chunk_size
        is_last_frame = frame_idx == total_n_frames - 1

        if is_chunk_ready or is_last_frame:
            fill_start = keyframe_count
            fill_end = refined_keyframe_buffer.n_frames
            fill_inds = arange(fill_start, fill_end)

            for fill_idx in fill_inds:
                left_kf, right_kf = nearest_keyframes_by_timestamp(fill_idx)
                refined_keyframe_buffer.poses[fill_idx] = se3_constant_velocity_interpolate(
                    refined_keyframe_buffer.poses[left_kf],
                    refined_keyframe_buffer.poses[right_kf],
                    refined_keyframe_buffer.tstamp[left_kf],
                    refined_keyframe_buffer.tstamp[right_kf],
                    refined_keyframe_buffer.tstamp[fill_idx],
                )

            graph = FactorGraph(droid_net, refined_keyframe_buffer, max_factors=-1, incremental=True)
            graph.add_factors(left_keyframe_indices(fill_inds), fill_inds)
            graph.add_factors(right_keyframe_indices(fill_inds), fill_inds)

            for _ in range(10):
                graph.update(
                    t0=fill_start,
                    t1=fill_end,
                    motion_only=True,  # optimize appended poses, keep dense disparities fixed
                )

            inner_filler.filled_poses.append(SE3(refined_keyframe_buffer.poses[fill_start:fill_end].clone()))
            refined_keyframe_buffer.n_frames = keyframe_count  # discard appended frames from keyframe graph

    filled_w2c_poses = inner_filler.get_result().poses
    original_intrinsics = resizer.recover_intrinsics(refined_keyframe_buffer.intrinsics)

    return SLAMOutput(
        trajectory=filled_w2c_poses.inv(),  # camera-to-world pose for every raw frame
        intrinsics=original_intrinsics,
        keyframe_indices=keyframe_indices,
    )
```

### Pass 2 Loop And Non-Keyframe Pose Infill

Source files:

| File | Role |
| --- | --- |
| `vipe/slam/system.py` | Pass 2 loop |
| `vipe/slam/components/inner_filler.py` | Pose interpolation and optimization for every original frame |
| `vipe/slam/components/factor_graph.py` | Infill graph updates |
| `vipe/slam/components/buffer.py` | Shared keyframe/infill state table mutated by infill BA |
| `vipe/slam/interface.py` | `SLAMOutput` |

Stage context: this is Stage 4. It loops over the original sequence again; its primary purpose is to produce poses for frames that were not retained as pass-1 keyframes. The handoff into this stage is the globally refined keyframe buffer from Stage 3. The handoff out is `SLAMOutput`: camera-to-world pose for every original frame, recovered original-resolution intrinsics, and selected-frame indices of the SLAM keyframes.

#### Diagram

```mermaid
flowchart TD
    A[Backend keyframe graph complete] --> B[inner_filler.start_after_keyframes number_of_keyframes]
    B --> C[SLAM Pass 2 over every frame]
    C --> D[_add_infill_frame for every frame]
    D --> E{Chunk has infill_chunk_size frames or this is last frame?}
    E -->|no| C
    E -->|yes| F[InnerFiller.fill_pending_chunk]
    F --> G[Find nearest left and right keyframes by timestamp]
    G --> H[Initialize pose by SE3 constant-velocity interpolation]
    H --> I[Build factors keyframe->infilled frame]
    I --> J[Run 10 motion-only graph updates]
    J --> K[Append filled poses]
    K --> L[Reset buffer.n_frames back to keyframe count]
    L --> M[get_result returns poses for all N frames]
    M --> N[Recover original-resolution intrinsics]
    N --> O[Return SLAMOutput]
```

#### Computation

After backend, SLAM has optimized only keyframes. Non-keyframe poses are filled in pass 2.

The code sets:

```python
self.inner_filler.start_after_keyframes(self.buffer.n_frames)
```

At this moment, `self.buffer.n_frames` is the number of optimized keyframes.

Then pass 2 loops over every original frame again:

```python
for frame_idx, frame_data in enumerate(frame_stream):
    images = self._rgb_bchw(frame_data)
    self._add_infill_frame(frame_idx, images, frame_data)
    if self.inner_filler.chunk_ready() or frame_idx == total_n_frames - 1:
        self.inner_filler.fill_pending_chunk()
```

`_add_infill_frame(...)` stores every frame after the keyframes in the same `GraphBuffer`, but it does not run `update_disps_sens`; keyframe depth anchors were only inserted during `_add_frontend_keyframe(...)` in pass 1.

`InnerFiller.chunk_ready()` returns:

```python
self.video.n_frames - self.start_idx >= self.args.infill_chunk_size
```

Default:

```text
infill_chunk_size = 16
```

So for long scenes, it fills in chunks of 16 appended frames. For a concrete 10-frame sequence, it fills once at the last frame.

#### Pose Initialization Inside `fill_pending_chunk`

Let:

```python
m_tstamp = self.video.tstamp[self.start_idx:total_frames]  # timestamps for frames to fill
n_tstamp = self.video.tstamp[:self.start_idx]              # keyframe timestamps
```

For every infill timestamp, it finds left and right keyframes:

```python
t0 = torch.searchsorted(n_tstamp, m_tstamp, right=True) - 1
t1 = torch.where(t0 < self.start_idx - 1, t0 + 1, t0)
```

So:

| Case | `t0` | `t1` |
| --- | --- | --- |
| timestamp before the last keyframe | nearest left keyframe | next keyframe |
| timestamp at or after last keyframe | last keyframe | same last keyframe |

Then it does SE3 constant-velocity interpolation:

```python
d_time = n_tstamp[t1] - n_tstamp[t0] + 1e-3
n_pose = SE3(self.video.poses[:self.start_idx])
d_pose = n_pose[t1] * n_pose[t0].inv()
vel = d_pose.log() / d_time.unsqueeze(-1)
w = vel * (m_tstamp - n_tstamp[t0]).unsqueeze(-1)
m_pose = SE3.exp(w) * n_pose[t0]
self.video.poses[self.start_idx:total_frames] = m_pose.data
```

These poses are world-to-camera because `GraphBuffer.poses` stores inverse/camera poses.

#### Motion-Only Infill

Current `InnerFiller.fill_pending_chunk` only optimizes poses for appended non-keyframe frames. It does not update their dense disparities.

#### Infill Graph Optimization

`InnerFiller.fill_pending_chunk` builds a temporary incremental `FactorGraph`:

```python
graph = FactorGraph(..., max_factors=-1, incremental=True)
```

For each infill frame, it adds:

```python
graph.add_factors(t0, infill_inds)
graph.add_factors(t1, infill_inds)
```

That means each infilled frame is constrained by reprojection factors from its left and right neighboring keyframes.

Then:

```python
for _ in range(10):
    graph.update(
        self.start_idx,
        total_frames,
        motion_only=True,
    )
```

Because `motion_only=True`, BA fixes dense disparity and optimizes poses only.

After optimization:

```python
current_poses = SE3(self.video.poses[self.start_idx:total_frames].clone())
self.filled_poses.append(current_poses)
self.video.n_frames = self.start_idx
```

Resetting `video.n_frames` discards appended infill frames from the keyframe graph buffer, while preserving the filled poses in `self.filled_poses`.

#### Return From SLAM

After pass 2:

```python
filled_return = self.inner_filler.get_result()
```

`filled_return.poses` has one world-to-camera pose for every original frame.

`SLAMSystem.run` then recovers original-resolution intrinsics and returns:

```python
SLAMOutput(
    trajectory=filled_return.poses.inv(),  # camera-to-world
    intrinsics=original_intrinsics,
    keyframe_indices=keyframe_indices,
)
```

`original_intrinsics` are recovered from resized/cropped SLAM intrinsics:

```python
new_intrinsics = after_intrinsics.clone()
new_intrinsics[2] += crop_left
new_intrinsics[3] += crop_top
new_intrinsics[0:4:2] *= fac_x
new_intrinsics[1:4:2] *= fac_y
```

For a concrete no-crop example:

```text
after_intrinsics = [332.5536, 332.5536, 256.0, 192.0]
fac_x = 32 / 512 = 0.0625
fac_y = 24 / 384 = 0.0625
original_intrinsics = [20.7846, 20.7846, 16.0, 12.0]
```

## Stage 5: Final Dense Depth And Outputs

### Final Dense Depth Pass

Source files:

| File | Role |
| --- | --- |
| `vipe/pipeline.py` | `DAV3DepthEstimator`, final Stage 5 call site |
| `vipe/streams/base.py` | `FrameData.dav3_conditions` |

Stage context: this starts Stage 5. `run.py` has already received `SLAMOutput` from Stage 4. The pipeline now prepares final dense depth by re-reading original frames, assigning final camera-to-world poses, assigning recovered raw-resolution SLAM intrinsics, setting the camera type to pinhole, and yielding frames with final dense depth. With the default depth mode this is DAV3 posed multi-frame inference. With sensor-depth modes it can additionally scale DAV3 depth to sensor depth or bypass DAV3 and use sensor depth directly. If camera normalization created `image_valid_mask`, final depth is zeroed outside that mask before depth artifacts, backprojection, and TSDF fusion consume it.

#### Re-Reading Original Frames

After SLAM returns, `VipePipeline._save_outputs` builds the final output iterator:

```python
final_frames = self._run_final_depth(frame_stream, slam_output)
```

`frame_stream` here is the original `FrameDir`, not the resized SLAM frame stream. Iterating or indexing it reads original-resolution RGB frames from disk again.

`DAV3DepthEstimator._attach_slam_output` attaches final SLAM geometry by selected frame index:

```python
frame = frame_stream[frame_idx]
frame.pose = slam_output.trajectory[frame_idx]
frame.intrinsics = slam_output.intrinsics
frame.camera_type = CameraType.PINHOLE
return frame
```

So after attachment, frame `i` has:

```python
FrameData(
    raw_frame_idx=i,
    rgb=original_resolution_rgb,
    pose=slam_output.trajectory[i],       # camera-to-world
    intrinsics=slam_output.intrinsics,    # recovered raw-resolution [fx, fy, cx, cy]
    camera_type=CameraType.PINHOLE,
    sensor_depth=loaded_original_resolution_sensor_depth_if_enabled,
    image_valid_mask=undistort_valid_mask_if_enabled,
)
```

`slam_output.trajectory` is a length-`N` batch of camera-to-world poses. `slam_output.intrinsics` is one recovered original-resolution pinhole intrinsic vector with shape `(4,)`, so the same tensor is assigned to every frame.

`slam_output.keyframe_indices` is a plain Python list of selected-frame indices for the optimized SLAM keyframes. It is not a dense map or point cloud; it only lets Stage 5 re-read nearby original-resolution keyframe images as extra DAV3 context.

The important consequence is that artifact writing receives original-resolution frames, not the resized SLAM frames. SLAM’s resized/cropped intrinsics are recovered back to original size before assignment, so final depth and final PCD construction operate in the original input image coordinate system.

#### Final Depth Mode Branches

`DAV3DepthEstimator` is configured with:

```python
DAV3DepthEstimator(
    model_name=pipeline.depth.final_model,
    window_size=pipeline.depth.window_size,
    overlap_size=pipeline.depth.overlap_size,
    use_gt_sens_depths=pipeline.depth.use_gt_sens_depths,
)
```

The branch table is:

| Mode | Stage 5 computation |
| --- | --- |
| `null` | Load `pipeline.depth.final_model`, run DAV3 posed sliding-window inference, yield DAV3 depth. |
| `scale` | Load and run the same DAV3 path, then multiply each window's DAV3 depth by one scalar fitted to the loaded sensor depths for that window before overlap blending. |
| `direct` | Do not load or run final DAV3. Re-read each original frame, attach pose/intrinsics, and set `metric_depth` to loaded sensor depth with invalid pixels zeroed. `depth_confidence` is the valid-pixel mask. |

#### Sliding Window DAV3 Inference (`null` and `scale`)

```mermaid
flowchart TD
    A[Frame loop 0..N-1] --> B[Attach final pose/intrinsics to original frame]
    B --> C[Append to current sliding window]
    C --> D{Window full or last frame?}
    D -->|no| A
    D -->|yes| E[Probe left/right SLAM keyframes near window frames]
    E --> F[Re-read context keyframes not already in window]
    F --> G[Build DAV3 image, extrinsic, intrinsic lists]
    G --> H[Configured DepthAnything3 final posed multi-frame inference]
    H --> I[Keep only current-window depth/confidence outputs]
    I --> J[Interpolate depth/confidence to original frame size]
    J --> S{use_gt_sens_depths == scale?}
    S -->|yes| U[Fit one scalar to current-window sensor depths and scale DAV3 depth]
    S -->|no| K{Trailing overlap exists?}
    U --> K
    K -->|yes| L[Linearly blend overlap depths/confidences]
    K -->|no| M[Use window depths]
    L --> N[Yield non-overlap frames]
    M --> N
    N --> O[Save trailing frames for next overlap]
```

#### Processor State

Constructor state:

```python
self.window_size = pipeline.depth.window_size
self.overlap_size = pipeline.depth.overlap_size
```

Default config:

```text
window_size = 10
overlap_size = 3
```

It loads:

```python
DepthAnything3.from_pretrained(pipeline.depth.final_model)
```

This model is loaded for `null` and `scale`. In `direct`, the constructor intentionally does not load DAV3 because the final dense depth is the loaded sensor depth.

This is the final post-SLAM DAV3 depth model. It is different from the SLAM keyframe single-image metric depth model:

| Use | Model |
| --- | --- |
| SLAM keyframe inverse-depth anchor | `pipeline.depth.keyframe_model`, default `depth-anything/DA3METRIC-LARGE` |
| Final depth for every frame | `pipeline.depth.final_model`, default `depth-anything/DA3-GIANT-1.1` |

#### Sliding Window State

At the start of `estimate(frame_stream, slam_output)`, the estimator uses the frame count and the keyframe index list from `SLAMOutput`:

```python
n_frames = len(frame_stream)
keyframe_indices = slam_output.keyframe_indices
```

The main frame loop maintains:

```python
current_sliding_window: list[FrameData]
current_sliding_window_idx: list[int]
trailing_depth: torch.Tensor | None
trailing_confidence: torch.Tensor | None
```

Each frame is appended until:

```python
len(current_sliding_window) == window_size or is_last_frame
```

For each ready window:

1. Probe neighboring SLAM keyframes for every frame index in the current window:
   ```python
   context_indices = sorted({
       keyframe_idx
       for i in current_sliding_window_idx
       for keyframe_idx in _probe_keyframe_indices(slam_output.keyframe_indices, i)
   })
   ```
2. Remove context frames already present in the current window:
   ```python
   context_indices = [i for i in context_indices if i not in current_sliding_window_idx]
   ```
3. Convert the current window frames to DAV3 conditions:
   ```python
   sw_images, sw_exts, sw_ints = zip(*[frame.dav3_conditions() for frame in current_sliding_window])
   ```
4. Re-read each context keyframe from the original `FrameDir`, attach its final pose/intrinsics, and convert it to DAV3 conditions.
5. Run final DAV3 on current window frames plus context keyframes:
   ```python
   dav3_api.inference(
       list(sw_images + ctx_images),
       extrinsics=np.stack(sw_exts + ctx_exts, axis=0),
       intrinsics=np.stack(sw_ints + ctx_ints, axis=0),
       process_res_method="lower_bound_resize",
   )
   ```
6. Keep only the first `len(sw_images)` outputs, because appended keyframes are context only:
   ```python
   sw_depth = torch.from_numpy(dav3_inference_result.depth[:len(sw_images)]).float().cuda()
   ```
7. Interpolate depth to the original frame size:
   ```python
   sw_depth = interpolate(sw_depth[:, None], frame.size(), mode="bilinear")[:,0]
   ```
8. Do the same slicing, conversion, and interpolation for confidence if DAV3 returns confidence.
9. If `pipeline.depth.use_gt_sens_depths=scale`, stack the same current-window sensor depths and fit one scalar:
   ```math
   s^\* = \frac{\sum_{f,u \in \mathcal{V}} D_{\text{dav3}}(f,u) D_{\text{sens}}(f,u)}
                {\sum_{f,u \in \mathcal{V}} D_{\text{dav3}}(f,u)^2}
   ```
   Then:
   ```math
   D_{\text{window}}(f,u) = s^\* D_{\text{dav3}}(f,u)
   ```
   This scalar is per DAV3 sliding-window inference, not per pixel. It is applied before overlap blending.

`FrameData.dav3_conditions()` returns:

```python
dav3_rgb = (self.rgb.cpu().numpy() * 255).astype(np.uint8)
dav3_ext = self.pose.inv().matrix().cpu().numpy()  # world-to-camera
dav3_int = [[fx,0,cx],[0,fy,cy],[0,0,1]]
```

Important: ViPE frame pose is camera-to-world. DAV3 receives extrinsics as world-to-camera.

#### Direct Sensor Final Depth (`direct`)

When:

```text
pipeline.depth.use_gt_sens_depths=direct
```

Stage 5 does not build sliding windows, does not probe keyframe context, and does not call DAV3. The final depth iterator is simply:

```python
for frame_idx in range(len(frame_stream)):
    frame = _attach_slam_output(frame_stream, frame_idx, slam_output)
    sensor_depth = frame.sensor_depth.float()
    valid = torch.isfinite(sensor_depth) & (sensor_depth > 0.0) & image_valid_mask
    frame.metric_depth = torch.where(valid, sensor_depth, torch.zeros_like(sensor_depth))
    frame.depth_confidence = valid.float()
    yield frame
```

So the downstream artifact writer and PCD code still consume `FrameData.metric_depth`; only the source of that tensor changes.

In default and `scale` modes, DAV3 still receives the pinhole-normalized RGB image, but its predicted depth is post-masked by `image_valid_mask`. In `scale` mode the least-squares scalar is also computed only over pixels where DAV3 depth, sensor depth, and `image_valid_mask` are all valid. This prevents DAV3 hallucinations on undistort padding pixels from entering saved depth, backproject PCD, TSDF, or BA depth priors.

#### Overlap Blending

If the current window has trailing depth from the previous window:

```python
n_interp_frames = len(trailing_depth)
alpha = torch.linspace(0, 1, n_interp_frames + 2)[1:-1][:, None, None]
sw_depth[:n_interp_frames] = trailing_depth * (1 - alpha) + sw_depth[:n_interp_frames] * alpha
```

For default `overlap_size=3`, alpha values are:

```text
torch.linspace(0,1,5)[1:-1] = [0.25, 0.50, 0.75]
```

So overlap frame depths are blended:

| Overlap position | Output depth |
| --- | --- |
| first overlap | `0.75 * old + 0.25 * new` |
| second overlap | `0.50 * old + 0.50 * new` |
| third overlap | `0.25 * old + 0.75 * new` |

Number of yielded frames:

```python
n_frames_to_yield = window_size - overlap_size if not is_last_frame else len(current_sliding_window)
```

Default normal window yields `10 - 3 = 7` frames and keeps 3 trailing frames for the next window. Last window yields everything.

### Artifact Saving And PCD Fusion

Source files:

| File | Role |
| --- | --- |
| `vipe/pipeline.py` | Calls `io.save_artifacts` |
| `vipe/utils/io.py` | Saves pose, per-frame final depth, one shared intrinsics JSON, and configured PCD exports |

Stage context: this is the persistence part of Stage 5. It consumes the final dense-depth frame iterator once. It records poses, writes each final dense depth frame, records the single shared recovered intrinsics vector once, updates the configured PCD fusion path or paths online during the loop, and then performs the final PCD extraction/write step. It does not write an RGB video. The saved pose/depth/intrinsics artifacts are also the disk contract referenced by the ScanNet benchmark manifest.

#### Diagram

```mermaid
flowchart TD
    A[Final frame iterator] --> B{save_artifacts?}
    B -->|yes| C[save_artifacts]
    B -->|no| D[Skip artifact save]
    C --> E[Iterate final frames with final dense depth]
    E --> F[Write one depth NPY into depth zip]
    E --> G[Collect pose matrices]
    E --> H[Record first intrinsics vector]
    E --> J{pcd_fusion_mode}
    J -->|backproject| L[Append sampled world points to temporary PLY body]
    J -->|tsdf| M[Integrate frame into Open3D TSDF volume]
    J -->|both| R[Do both updates for the same frame]
    L --> N[Write color_backproject.ply]
    M --> O[Extract TSDF mesh and sample max_points]
    O --> P[Write color_tsdf.ply]
    C --> Q[Write pose npz and intrinsics json]
    D --> T[Done]
    Q --> T
```

#### Artifact Paths

For input base path:

```text
/robodata/smodak/repos/ovo/data/input/ScanNet/scene0000_00/color
```

`FrameDir.name()` is the directory basename:

```text
artifact_name = "color"
```

With output:

```text
/robodata/smodak/repos/vipe/outputs/scene00_dav3tsdf
```

Artifacts are:

| Artifact | Path |
| --- | --- |
| Pose npz | `outputs/scene00_dav3tsdf/pose/color.npz` |
| Depth zip | `outputs/scene00_dav3tsdf/depth/color.zip` |
| Intrinsics JSON | `outputs/scene00_dav3tsdf/intrinsics/color.json` |
| Backproject PCD | `outputs/scene00_dav3tsdf/pcd/color_backproject.ply` |
| TSDF PCD | `outputs/scene00_dav3tsdf/pcd/color_tsdf.ply` |

With the default `pcd_fusion_mode=both`, both PCDs are written. With `backproject` or `tsdf`, only the selected branch is written.

#### `save_artifacts` Streaming Pass

`save_artifacts` is designed to avoid keeping the whole final sequence in RAM.

It initializes:

```python
pose_list = []
intrinsics = None
intrinsics_frame_size = None
```

It also opens the depth zip before iterating final frames:

```python
with zipfile.ZipFile(out_path.depth_path, "w", compression=zipfile.ZIP_STORED) as depth_zip:
    ...
```

For PCD:

```python
if pcd_fusion_mode not in {"backproject", "tsdf", "both"}:
    raise ValueError(...)

write_backproject = pcd_fusion_mode in {"backproject", "both"}
write_tsdf = pcd_fusion_mode in {"tsdf", "both"}
pcd_body_file = tempfile.TemporaryFile() if write_backproject else None
pcd_vertex_count = 0
max_points_per_frame = ceil(max_pcd_points / max(n_frames, 1))
tsdf_volume = _make_tsdf_volume(...) if write_tsdf else None
```

With `max_pcd_points=8,000,000` and a real 5578-frame scene:

```text
max_points_per_frame = ceil(8,000,000 / 5578) = 1435
```

With a concrete 10-frame sequence:

```text
max_points_per_frame = ceil(8,000,000 / 10) = 800,000
```

During iteration, for each final frame:

1. Write the final dense depth to the depth zip as `000000.npy`, `000001.npy`, and so on.
2. If pose exists, append `(frame_idx, pose.matrix())`.
3. If intrinsics have not been recorded yet, store the first intrinsics tensor plus the frame size.
4. Update the selected PCD fusion branch or branches using the same in-memory final dense depth.

The depth write is:

```python
depth = frame_data.metric_depth.detach().cpu().numpy().astype(np.float16)
buffer = BytesIO()
np.save(buffer, depth, allow_pickle=False)
depth_zip.writestr(f"{frame_idx:06d}.npy", buffer.getvalue())
```

So `depth/color.zip` is a zip container of NumPy arrays, not EXR images. Each `.npy` stores shape and dtype metadata. The stored dtype is `float16` for compactness; the benchmark evaluator reads the same `float16` frames directly through the ViPE manifest path.

At the end:

1. Save pose npz:
   ```python
   np.savez(path, data=pose_data, inds=pose_inds)
   ```
2. Save one intrinsics JSON:
   ```python
   {
     "camera_model": "pinhole",
     "width": W,
     "height": H,
     "params": [fx, fy, cx, cy],
     "fx": fx,
     "fy": fy,
     "cx": cx,
     "cy": cy
   }
   ```
3. Write the selected PCD export or exports.

#### Backproject PCD Mode

This branch runs when:

```text
pipeline.output.pcd_fusion_mode=backproject or pipeline.output.pcd_fusion_mode=both
```

For each frame, `_backproject_vertices` requires:

```python
frame_data.metric_depth is not None
frame_data.pose is not None
frame_data.intrinsics is not None
max_points_per_frame > 0
```

It computes:

```python
depth = frame_data.metric_depth.detach().cpu().numpy()
valid = np.isfinite(depth) & (depth > 0.0)
if frame_data.image_valid_mask is not None:
    valid &= frame_data.image_valid_mask.detach().cpu().numpy().astype(bool)
```

If depth confidence exists:

```python
confidence = frame_data.depth_confidence.detach().cpu().numpy()
valid &= (confidence >= mean(confidence) * conf_threshold_coef) & (confidence > 1e-5)
```

Default:

```text
conf_threshold_coef = 0.75
sample_ratio = 0.015
```

Sampling rule:

| Confidence exists? | Initial sample count |
| --- | --- |
| yes | `int(num_valid * sample_ratio)` |
| no | `num_valid` |

Then:

```python
sample_count = min(sample_count, max_points_per_frame)
```

If sampling is needed:

| Confidence exists? | Sampling method |
| --- | --- |
| yes | random without replacement with seed `frame_data.raw_frame_idx` |
| no | deterministic striding |

Backprojection formula:

```python
ys, xs = np.divmod(valid_flat, width)
zs = depth.ravel()[valid_flat]
fx, fy, cx, cy = intrinsics

points_cam[:,0] = (xs - cx) * zs / fx
points_cam[:,1] = (ys - cy) * zs / fy
points_cam[:,2] = zs
points_cam[:,3] = 1.0

points_world = (pose_c2w @ points_cam.T).T[:, :3]
```

Colors come from the same RGB pixels:

```python
colors = frame_data.rgb.reshape(-1,3)[valid_flat] * 255
```

The temporary binary vertex body is written as:

```text
x float32, y float32, z float32, red uint8, green uint8, blue uint8
```

At the end, `_write_backproject_pcd` writes the PLY header and copies the temporary body to `pcd/color_backproject.ply`.

#### TSDF PCD Mode

This branch runs when:

```text
pipeline.output.pcd_fusion_mode=tsdf or pipeline.output.pcd_fusion_mode=both
```

It creates:

```python
o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=pcd_tsdf_voxel_length,
    sdf_trunc=pcd_tsdf_sdf_trunc,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)
```

Defaults:

```text
voxel_length = 0.02 m
sdf_trunc = 0.15 m
depth_trunc = 5.0 m
```

For every final frame, `_integrate_tsdf_frame`:

1. Checks depth, pose, and intrinsics exist.
2. Converts invalid depth to zero:
   ```python
   depth[~np.isfinite(depth)] = 0.0
   depth[depth <= 0.0] = 0.0
   if frame_data.image_valid_mask is not None:
       image_valid_mask = frame_data.image_valid_mask.detach().cpu().numpy().astype(bool)
       depth[~image_valid_mask] = 0.0
   ```
3. Converts RGB to uint8.
4. Creates Open3D camera intrinsics:
   ```python
   PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
   ```
5. Creates RGBD image:
   ```python
   RGBDImage.create_from_color_and_depth(
       color,
       depth,
       depth_scale=1.0,
       depth_trunc=5.0,
       convert_rgb_to_intensity=False,
   )
   ```
6. Converts pose to world-to-camera:
   ```python
   w2c = frame_data.pose.inv().matrix().cpu().numpy()
   ```
7. Integrates:
   ```python
   volume.integrate(rgbd, intrinsics, w2c)
   ```

Conceptually, the TSDF volume stores a sparse voxel grid. Each voxel stores a truncated signed distance to the closest observed surface plus an integration weight and color. Every depth frame updates voxels near that observed surface. Free space and far-away volume are not represented as a full dense infinite grid; Open3D `ScalableTSDFVolume` allocates blocks as needed.

After all frames:

```python
mesh = volume.extract_triangle_mesh()
pcd = mesh.sample_points_uniformly(number_of_points=max_points)
o3d.io.write_point_cloud(str(out_path.tsdf_pcd_path), pcd, write_ascii=False)
```

So `pcd/color_tsdf.ply` is not raw backprojected pixels. It is uniformly sampled points on the extracted TSDF mesh. The requested sample count is `max_points`, default `8,000,000`.

## Runtime Branch Values In The Shown Command

| Branch | Current command value | Effect |
| --- | --- | --- |
| Frame input type | frame directory | Reads sorted images directly from `streams.base_path`. |
| Input camera model | `streams.input_camera_model=pinhole` by default | If set to a supported distorted model, Stage 1 undistorts to a pinhole stream before downstream SLAM/DAV3/PCD. |
| Downstream camera type | `pinhole` | SLAM, DAV3 conditioning, backprojection, TSDF, and benchmark artifacts all remain pinhole. |
| SLAM pose source | estimated from the frame sequence | SLAM starts from internal identity and constant-velocity initialization, then optimizes poses with frontend/backend BA. |
| Sensor-depth mode | `pipeline.depth.use_gt_sens_depths=null` by default | No sensor depth is loaded; keyframe and final depth use DAV3 normally. |
| Sensor-intrinsics mode | `streams.use_gt_intrinsics=false` by default | GeoCalib estimates intrinsics. If set to `true`, ViPE loads `intrinsic_color.json` if present, otherwise `intrinsic_color.txt`, and uses that camera instead of GeoCalib. |
| SLAM keyframe depth anchor | `pipeline.depth.keyframe_model`, default DAV3 `DA3METRIC-LARGE` | In default mode, DAV3 metric depth regularizes keyframe disparities. In `scale`/`direct`, loaded sensor depth scales or replaces this anchor. |
| Final dense depth | `pipeline.depth.final_model`, default DAV3 `DA3-GIANT-1.1` | In default mode, final dense depth comes from DAV3 posed multi-frame inference. In `scale`/`direct`, loaded sensor depth scales or replaces this final depth. |
| `save_artifacts` | `true` in your command | Pose, depth zip, intrinsics JSON, and configured PCD artifacts are written. |
| `pcd_fusion_mode` | `both` in your command and default config | Both `pcd/color_backproject.ply` and `pcd/color_tsdf.ply` are written. |

## Practical Interpretation Of The Final Results

| Artifact | What it represents |
| --- | --- |
| `pose/color.npz` | ViPE-estimated camera-to-world trajectory for every selected input frame. |
| `depth/color.zip` | Per-frame final dense depth, stored as `float16` NumPy `.npy` entries inside a zip. Default source is DAV3; sensor-depth modes can scale or replace it. |
| `intrinsics/color.json` | One recovered original-resolution pinhole intrinsic vector and image size. |
| `pcd/color_tsdf.ply` | Surface point cloud sampled from TSDF fusion of final dense depth plus ViPE poses/intrinsics. Produced when `pcd_fusion_mode=tsdf` or `both`. |
| `pcd/color_backproject.ply` | Direct sampled pixel backprojection of final dense depth plus ViPE poses/intrinsics. Produced when `pcd_fusion_mode=backproject` or `both`. |

## ScanNet Benchmark Adapter And Reconstruction Eval

The standalone run writes ViPE-native artifacts. The ScanNet benchmark script now keeps those artifacts native and writes a small DA3-side manifest instead of repacking everything into a huge `results.npz`.

```mermaid
flowchart TD
    A[ViPE artifacts under pipeline.output.path] --> B[pose/color.npz]
    A --> C[depth/color.zip]
    A --> D[intrinsics/color.json]
    B --> E[scannet_vipe_bench_evaluator.py]
    C --> E
    D --> E
    E --> F[exports/vipe_manifest.json]
    E --> G[exports/gt_meta.npz]
    F --> H[DA3 ScanNet direct ViPE loader]
    G --> H
    H --> I[Pose eval]
    H --> J[TSDF reconstruction eval]
    H --> K[Backproject reconstruction eval]
```

Normal benchmark mode:

```text
run ViPE -> write vipe_manifest.json -> write gt_meta.npz -> call DA3 evaluator
```

`--eval-only` mode:

```text
reuse existing ViPE artifacts -> write vipe_manifest.json -> write gt_meta.npz -> call DA3 evaluator
```

So `--eval-only` skips only the ViPE inference run. It still refreshes the lightweight benchmark manifest and GT metadata, but it no longer rebuilds a duplicated `results.npz`.

The ViPE manifest is written at:

```text
workspace/.../model_results/scannet/<scene>/{unposed,posed}/exports/vipe_manifest.json
```

It contains:

| Key | Meaning |
| --- | --- |
| `format` | Manifest version, currently `vipe_artifacts_v1`. |
| `scene` | ScanNet scene name. |
| `artifact_name` | Frame stream artifact basename, usually `color`. |
| `vipe_output_dir` | Root of the native ViPE artifacts for this scene. |
| `pose_path` | Native ViPE pose artifact, e.g. `pose/color.npz`. |
| `depth_path` | Native ViPE final dense depth artifact, e.g. `depth/color.zip`. |
| `intrinsics_path` | Native ViPE intrinsics artifact, e.g. `intrinsics/color.json`. |
| `frame_indices` | Exact ViPE frame indices included in the benchmark subset. |

For ScanNet, the DA3 evaluator asks the dataset for its prediction artifact path, and the current ScanNet dataset returns `exports/vipe_manifest.json`. The ViPE benchmark path does not write, read, hardlink, or delete `exports/mini_npz/results.npz`.

The direct ViPE loader reconstructs the same logical arrays the DA3 evaluator used before, but on demand from native artifacts:

| Logical field | Direct source |
| --- | --- |
| Predicted depth | `depth/color.zip`, reading `000000.npy`, `000001.npy`, etc. from `frame_indices`. |
| Predicted extrinsics | `pose/color.npz`, converting saved ViPE camera-to-world poses to world-to-camera matrices. |
| Predicted intrinsics | `intrinsics/color.json`, expanded to a `3x3` pinhole matrix per benchmark frame. |
| GT metadata | `exports/gt_meta.npz`, containing the sampled ScanNet GT extrinsics, intrinsics, and image file list. |

No fake `conf=np.ones(...)` array is produced anymore because ScanNet reconstruction metrics do not use confidence. Pose eval also reads predicted extrinsics through the same dataset loader, so it no longer requires a result NPZ.

The evaluator then runs two reconstruction methods for each reconstruction mode:

| Metric group | Reconstruction method | Output PLY |
| --- | --- | --- |
| `scannet_recon_unposed_tsdf` | TSDF fusion after evaluator unposed prep | `exports/fuse/pcd_tsdf.ply` |
| `scannet_recon_unposed_backproject` | Direct backprojection after evaluator unposed prep | `exports/fuse/pcd_backproject.ply` |
| `scannet_recon_posed_tsdf` | TSDF fusion after evaluator posed prep | `exports/fuse/pcd_tsdf.ply` |
| `scannet_recon_posed_backproject` | Direct backprojection after evaluator posed prep | `exports/fuse/pcd_backproject.ply` |

Those PLY paths are relative to each mode-specific export directory, so unposed and posed outputs live under separate benchmark directories.

The two evaluator reconstruction methods share the same prepared inputs for a given mode. For `recon_unposed`, evaluator prep aligns predicted ViPE poses to GT with Umeyama/RANSAC and scales depth by the recovered Sim3 scale. For `recon_posed`, evaluator prep uses GT poses and GT intrinsics while still scaling predicted depth by the same alignment scale. Both methods resize depth to original RGB size, apply the ScanNet GT valid-depth mask, enforce `max_depth=5.0`, and compare to the GT mesh after the same AABB crop, voxel downsample, and distance-threshold metric logic.

This means the standalone `pcd/color_tsdf.ply` and `pcd/color_backproject.ply` are inspection/use artifacts built directly during ViPE artifact saving, while benchmark `pcd_tsdf.ply` and `pcd_backproject.ply` are evaluator-controlled reconstructions built from the same native ViPE pose/depth/intrinsics artifacts through `vipe_manifest.json`.

The key computational distinction is:

```text
SLAM estimates poses using DROID-style dense reprojection factors and the configured keyframe depth regularization.
Final dense depth is computed afterward from the configured final depth path using the solved poses and intrinsics.
Final saved depth and PCDs are built from that final depth plus those solved poses and intrinsics.
```
