# GraphBuffer, FactorGraph, Frontend Graph, Backend Graph

This explains the concrete objects in the current reduced ViPE SLAM path.

## The Short Version

There is one conceptual optimization problem:

```text
keyframe poses + keyframe dense disparities
-> constrained by pairwise reprojection factors and external sensor-depth anchor factors
-> optimized by bundle adjustment
```

In code, that conceptual problem is split across these objects:

| Object | Code class | What it owns |
| --- | --- | --- |
| `GraphBuffer` | `vipe/slam/components/buffer.py::GraphBuffer` | The persistent per-frame/keyframe state: poses, disparities, images, DROID features, timestamps, intrinsics. This is the node/state table. |
| `FactorGraph` | `vipe/slam/components/factor_graph.py::FactorGraph` | The active edge list and learned per-edge state: `(ii,jj)` pairs, DROID correlation/update state, target flow, weights, factor ages. This is the edge/factor manager. |
| `frontend.graph` | `SLAMFrontend.graph` | A persistent incremental `FactorGraph` used during pass-1 as new keyframes arrive. |
| backend graph | local variable `graph` inside `SLAMBackend.run` | A fresh non-incremental `FactorGraph` built once for global BA after pass-1 keyframes are known. |
| actual BA solver | `GraphBuffer.bundle_adjustment(...)` -> `Solver.run_inplace(...)` | Builds solver terms from the current `FactorGraph` tensors and mutates `GraphBuffer.poses` and `GraphBuffer.disps`. |

So your mental model is close:

```text
GraphBuffer ~= nodes/state variables
FactorGraph ~= edges/factors plus DROID learned edge machinery
```

But `FactorGraph` does not itself solve the optimization. It prepares learned targets/weights for edges, then calls:

```python
self.buffer.bundle_adjustment(...)
```

That is the method that constructs the actual solver terms and runs BA.

## What Is A Factor?

A factor is one constraint term in an optimization problem.

Toy state:

```text
node 0 pose: P0
node 1 pose: P1
node 0 dense disparity: D0
```

A pairwise visual factor from frame `0` to frame `1` says:

```text
Take every low-res pixel in frame 0.
Use D0 and P0/P1 to project that pixel into frame 1.
The projected coordinate should match a target coordinate predicted by DROID.
```

For one pixel:

```text
predicted projection: [10.2, 5.8]
DROID target:         [11.0, 6.0]
residual:             [-0.8, -0.2]
```

BA changes `P0`, `P1`, and `D0` to reduce many such residuals.

A depth-anchor factor says:

```text
optimized disparity D0 should stay close to the sampled anchor disparity D0_sens
```

For one low-res pixel:

```text
optimized disparity: 0.50
anchor disparity:    0.45
residual:            0.05
```

So in this code there are two solver-level factor families:

| Solver term | Class | Optimizes against |
| --- | --- | --- |
| Pairwise reprojection / dense flow | `DenseDepthFlowTerm` | DROID-predicted target coordinates between frame pairs. |
| Depth anchor | `DispSensRegularizationTerm` | `GraphBuffer.disps_sens`, the inverse-depth map sampled from external sensor depth, plus `GraphBuffer.disps_sens_weight` for invalid sensor or undistort-invalid pixels. |

## GraphBuffer: Persistent Node/State Table

Class:

```python
vipe/slam/components/buffer.py::GraphBuffer
```

Created once in:

```python
SLAMSystem._build_components()
```

with:

```python
self.buffer = GraphBuffer(
    height=config.height,
    width=config.width,
    buffer_size=config.buffer,
    init_disp=config.init_disp,
    ba_config=config.ba,
    camera_type=config.camera_type,
)
```

### What It Stores

`GraphBuffer` is a fixed-size tensor table. Slot `k` means "the `k`-th buffered SLAM frame/keyframe", not necessarily raw input frame `k`.

| Field | Shape / type | Meaning |
| --- | --- | --- |
| `n_frames` | int | Number of active slots currently filled. |
| `height`, `width` | int | SLAM resized image size, divisible by 8. |
| `tstamp` | `(buffer_size,)` int | Original stream frame index for each slot. |
| `images` | `(buffer_size,3,H,W)` float16 | Resized RGB image for each slot. |
| `poses` | `(buffer_size,7)` float32 | World-to-camera pose for each slot, stored as SE3 data. |
| `intrinsics` | `(4,)` float32 | Shared resized pinhole intrinsics. |
| `disps` | `(buffer_size,H/8,W/8)` float32 | Optimized dense inverse depth/disparity. |
| `disps_sens` | `(buffer_size,H/8,W/8)` float32 | Keyframe depth anchor converted to inverse depth. |
| `disps_sens_weight` | `(buffer_size,H/8,W/8)` float32 | Depth-anchor BA weight. It is one at valid external sensor-depth samples and zero at invalid depth or undistort-invalid pixels. |
| `fmaps` | `(buffer_size,128,H/8,W/8)` float16 | DROID feature maps used for correlations. |
| `nets` | `(buffer_size,128,H/8,W/8)` float16 | DROID recurrent hidden state. |
| `inps` | `(buffer_size,128,H/8,W/8)` float16 | DROID context input. |

`GraphBuffer` is where optimization variables live:

```text
poses -> optimized by BA
disps -> optimized by BA
```

Everything else supports feature matching, depth anchoring, or keyframe/infill bookkeeping.

`GraphBuffer` remains an internal SLAM workspace. The final output handoff is `SLAMOutput`, which contains the full-frame camera-to-world trajectory, recovered original-resolution intrinsics, and the selected-frame indices of optimized SLAM keyframes. The current output stage uses the trajectory and intrinsics to replay external sensor depth into pose/depth/PCD artifacts.

### How Slots Get Filled

During SLAM pass 1, keyframes are added through:

```python
SLAMSystem._add_frontend_keyframe(...)
```

which calls:

```python
kf_idx = self._store_buffer_frame(...)
self.buffer.update_disps_sens(frame_idx=kf_idx, frame_data=frame_data)
```

`_store_buffer_frame` writes:

```python
buffer.tstamp[kf_idx] = raw frame index
buffer.images[kf_idx] = resized image
buffer.fmaps[kf_idx] = DROID feature map
buffer.nets[kf_idx], buffer.inps[kf_idx] = DROID context tensors
buffer.intrinsics = shared intrinsics, only for first slot
buffer.n_frames += 1
```

For the current standalone run, poses start from the buffer identity initialization and are then updated by frontend/backend BA plus constant-velocity pose initialization for newly allocated slots.

Then `update_disps_sens` samples the external sensor depth into the low-resolution inverse-depth anchor:

```python
metric_depth = frame_data.sensor_depth.float()
valid = isfinite(metric_depth) & (metric_depth > 0)
if frame_data.image_valid_mask is not None:
    valid &= frame_data.image_valid_mask
metric_depth = where(valid, metric_depth, 0)

disp_sens = metric_depth[3::8, 3::8]
disp_sens = torch.where(disp_sens > 0, disp_sens.reciprocal(), disp_sens)
self.disps_sens[frame_idx] = disp_sens
self.disps_sens_weight[frame_idx] = valid[3::8, 3::8].float()
```

That means:

```text
sensor depth map at image resolution
-> sample every 8 pixels starting at offset 3
-> convert depth to inverse depth
-> store in disps_sens
-> store matching BA weights in disps_sens_weight
```

### Important GraphBuffer Methods

#### `remove_second_newest(ix)`

Used by frontend keyframe pruning.

It assumes:

```python
ix == self.n_frames - 2
```

Then slot `ix` is overwritten by slot `ix+1`, and `n_frames` decrements.

Toy state:

```text
slots: 0, 1, 2, 3
n_frames = 4
remove_second_newest(2)
```

After:

```text
slot 2 now contains old slot 3
n_frames = 3
active slots: 0, 1, 2
```

`FactorGraph.rm_second_newest_keyframe` also updates edge indices to stay consistent.

#### `update_disps_sens(frame_idx, frame_data)`

Reads the already-loaded external sensor depth and stores the depth prior as inverse depth.

This does not optimize anything. It only writes `disps_sens` and `disps_sens_weight`.

#### `reproject_dense_disp(ii, jj)`

Given frame-pair arrays `ii` and `jj`, project every low-res pixel from source frame `ii[k]` into target frame `jj[k]`.

Inputs:

```text
poses
disps
intrinsics
edge list ii,jj
```

Output:

```text
coords: projected target-frame coordinates, shape (num_edges,H/8,W/8,2)
valid_mask: which projections are valid
```

This is used by `FactorGraph` to know what the current geometry predicts.

#### `frame_distance_dense_disp(ii, jj, beta, bidirectional)`

Computes a geometry-based distance between two frames using current poses and disparities.

This is not the final metric. It is used to decide which frame pairs should get edges and whether a keyframe is redundant.

#### `bundle_adjustment(...)`

This is the actual BA entrypoint.

Called by:

```python
FactorGraph.update(...)
FactorGraph.update_batch(...)
```

It receives:

```text
target: DROID target coordinates for each edge and pixel
weight: DROID confidence/weights for each edge and pixel
ii, jj: edge source/target frame indices
disp_damping: DROID-predicted damping for disparity variables
t0, t1: pose optimization window bounds
n_iters: number of solver iterations
motion_only: whether to freeze dense_disp
```

Inside, it creates:

```python
solver = Solver(...)
solver.add_term(DenseDepthFlowTerm(...))
```

If `motion_only=False`, it also adds:

```python
DispSensRegularizationTerm(...)
```

Then it repeatedly calls:

```python
solver.run_inplace({
    "pose": SE3(self.poses),
    "dense_disp": disps_flattened,
})
```

`solver.run_inplace` mutates:

```text
self.poses
self.disps
```

through SE3 and dense-disparity retractions.

## FactorGraph: Edge List + Learned Edge State

Class:

```python
vipe/slam/components/factor_graph.py::FactorGraph
```

Created in two places:

```python
SLAMFrontend.__init__:
    self.graph = FactorGraph(..., max_factors=frontend_max_factors, incremental=True)

SLAMBackend.run:
    graph = FactorGraph(..., max_factors=backend_max_factors_per_keyframe*t, incremental=False)
```

`FactorGraph` owns the graph edges. An edge is a directed pair:

```text
source frame i -> target frame j
```

stored in:

```python
self.ii  # source indices
self.jj  # target indices
```

Example:

```python
self.ii = [0, 1, 1, 2]
self.jj = [1, 0, 2, 1]
```

This means active directed edges:

```text
0 -> 1
1 -> 0
1 -> 2
2 -> 1
```

### What It Stores

| Field | Meaning |
| --- | --- |
| `buffer` | The shared `GraphBuffer` whose states are optimized. |
| `ii`, `jj` | Active directed edge list. |
| `age` | Number of frontend updates since each edge was added. |
| `target` | Learned target coordinates for each edge and low-res pixel. |
| `weight` | Learned residual weights for each edge and low-res pixel. |
| `corr` | Incremental correlation object for active edges, frontend only. |
| `f_net` | DROID recurrent hidden state for each active edge/source frame. |
| `inp` | DROID context input for each active edge/source frame, frontend only. |
| `damping` | Per-frame disparity damping predicted by DROID update module. |
| `coords0` | Static low-res coordinate grid. |

`FactorGraph` does not own poses or disparities. It reads/writes them through `GraphBuffer`.

### `add_factors(ii, jj, remove=False)`

Adds directed edges.

Steps:

1. Convert inputs to tensors.
2. Remove repeats already present in the active edge set.
3. If frontend graph would exceed `max_factors` and `remove=True`, remove old factors.
4. If `incremental=True`, precompute/append correlation blocks:
   ```python
   fmap1 = buffer.fmaps[ii]
   fmap2 = buffer.fmaps[jj]
   corr = CorrBlock(fmap1, fmap2)
   ```
5. Compute initial target as current geometric projection:
   ```python
   target, _ = buffer.reproject_dense_disp(ii, jj)
   weight = zeros_like(target)
   ```
6. Append `ii`, `jj`, `age=0`, DROID net state, target, and weight.

Initial `target` is not a measurement yet. It starts as the current projection. The DROID update network later modifies it into a learned target.

### `rm_factors(mask)`

Removes active edges and their associated correlation/state/target/weight tensors.

### `rm_second_newest_keyframe(ix)`

Calls:

```python
buffer.remove_second_newest(ix)
```

and also removes or reindexes all edges touching shifted keyframe indices.

This keeps the `GraphBuffer` slot table and `FactorGraph` edge indices consistent.

### `update(...)`: Frontend Incremental Update

Used only when:

```python
incremental=True
```

This is the frontend update loop.

Core sequence:

```python
coords1 = buffer.reproject_dense_disp(self.ii, self.jj)
motn = concat(coords1 - coords0, target - coords1)
corr = self.corr(coords1)
self.f_net, delta, weight, damping, _ = net.update.forward(...)
self.target = coords1 + delta
self.weight = weight
self.damping[di] = damping
buffer.bundle_adjustment(...)
self.age += 1
```

Meaning:

1. Project current geometry across every active edge.
2. Feed correlation + motion features to DROID update network.
3. DROID predicts:
   ```text
   delta: how target coordinates should move
   weight: confidence/weight per residual
   damping: solver damping for disparity variables
   ```
4. Convert predicted target/weight into BA inputs.
5. Call `GraphBuffer.bundle_adjustment`.

So the exact frontend BA call path is:

```text
SLAMFrontend.run
-> SLAMFrontend.__initialize or __update
-> frontend.graph.update(...)
-> GraphBuffer.bundle_adjustment(...)
-> Solver.run_inplace(...)
```

### `update_batch(...)`: Backend Global Update

Used by backend with:

```python
incremental=False
```

It is similar to `update`, but it recomputes correlations in batches using:

```python
AltCorrBlock(self.buffer.fmaps[None])
```

and loops:

```python
for _ in range(steps):
    update DROID targets/weights/damping for all edges
    buffer.bundle_adjustment(...)
```

The exact backend BA call path is:

```text
SLAMBackend.run
-> graph = FactorGraph(..., incremental=False)
-> graph.add_proximity_factors(...)
-> graph.update_batch(itrs=backend_ba_iters, steps=backend_iters, batch_size=backend_batch_size)
-> GraphBuffer.bundle_adjustment(...)
-> Solver.run_inplace(...)
```

In current config:

```yaml
backend_iters: 31
backend_ba_iters: 8
backend_batch_size: 8
```

So backend executes 31 outer update steps. Each step calls:

```python
buffer.bundle_adjustment(..., n_iters=backend_ba_iters, ...)
```

That means each backend step runs 8 internal solver iterations after updating DROID targets/weights.

## Frontend Graph Versus Backend Graph

`frontend.graph` is a persistent object:

```python
self.graph = FactorGraph(..., max_factors=frontend_max_factors, incremental=True)
```

It lives across pass-1. As new keyframes arrive, it adds/removes edges and carries recurrent DROID state for those edges.

The backend graph is a temporary local object:

```python
graph = FactorGraph(..., max_factors=backend_max_factors_per_keyframe*t, incremental=False)
```

It is created fresh once after pass-1 ends. It sees the final keyframe set in `GraphBuffer`, builds denser proximity edges, optimizes globally, then is discarded.

Both graphs point to the same `GraphBuffer`.

That means:

```text
frontend.graph optimizes GraphBuffer
backend graph optimizes the same GraphBuffer
```

The backend does not copy keyframes into a new state. It modifies the same `buffer.poses` and `buffer.disps` produced by frontend.

## Factor Types In This Code

There are two levels of "factor type" naming:

1. Graph-construction factors: how frame-pair edges are selected.
2. Solver factors: mathematical residual terms inside BA.

### Graph-Construction Factor Types

These decide which directed frame pairs `(i,j)` become edges in a `FactorGraph`.

#### Neighborhood Factors

Method:

```python
FactorGraph.add_neighborhood_factors(t0, t1, r)
```

It creates all directed edges where:

```python
0 < abs(i - j) <= r
```

for frames in `[t0, t1)`.

Toy example:

```text
frames: 0,1,2,3
r = 1
```

Edges:

```text
0 -> 1
1 -> 0
1 -> 2
2 -> 1
2 -> 3
3 -> 2
```

With `r=2`, it also includes edges like:

```text
0 -> 2
2 -> 0
1 -> 3
3 -> 1
```

Current use:

```python
SLAMFrontend.__initialize:
    graph.add_neighborhood_factors(0, self.t1, r=1)
```

So frontend initialization uses only adjacent keyframe edges.

#### Proximity Factors

Method:

```python
FactorGraph.add_proximity_factors(t0, t1, rad, nms, beta, thresh, remove)
```

This uses current geometry to choose frame pairs whose views are close/overlapping.

High-level steps:

1. Build candidate pairs from:
   ```python
   ix = arange(t0, t)
   jx = arange(t1, t)
   ```
2. Compute distance:
   ```python
   d = buffer.frame_distance_dense_disp(ii, jj, beta)
   ```
3. Suppress pairs already active.
4. Reject near-diagonal pairs:
   ```python
   d[(ii - rad < jj) | (d > thresh)] = inf
   ```
5. Always add local bidirectional edges:
   ```python
   for i in range(t0, t):
       for j in range(max(i - rad - 1, 0), i):
           add i->j and j->i
   ```
6. Add additional low-distance pairs sorted by distance, with NMS suppression.

Current frontend use:

```python
SLAMFrontend.__update:
    graph.add_proximity_factors(
        t0=self.t1 - 5,
        t1=max(self.t1 - frontend_window, 0),
        rad=frontend_radius,
        nms=frontend_nms,
        thresh=frontend_thresh,
        beta=beta,
        remove=True,
    )
```

Current backend use:

```python
SLAMBackend.run:
    graph.add_proximity_factors(
        rad=backend_radius,
        nms=backend_nms,
        thresh=backend_thresh,
        beta=beta,
    )
```

Current config:

```yaml
frontend_radius: 2
frontend_nms: 1
frontend_thresh: 16.0
frontend_window: 25

backend_radius: 2
backend_nms: 3
backend_thresh: 22.0
```

So:

```text
frontend proximity = recent/local, capped by frontend_max_factors, incremental
backend proximity = denser whole-keyframe-set graph, capped by backend_max_factors_per_keyframe*t
```

### Solver Factor Types

These are the actual residual terms optimized in `GraphBuffer.bundle_adjustment`.

#### DenseDepthFlowTerm

Class:

```python
vipe/slam/ba/terms.py::DenseDepthFlowTerm
```

It receives:

```text
pose_i_inds = ii
pose_j_inds = jj
dense_disp_i_inds = ii
target = DROID target coordinates
weight = DROID weights
intrinsics = fixed resized intrinsics
```

For each directed edge `i -> j` and each low-res pixel `p`:

```text
use pose_i, pose_j, dense_disp_i[p], intrinsics
-> project p from frame i into frame j
-> compare projected coordinate to DROID target coordinate
```

Residual:

```text
projected_coord(i -> j, p) - target(i -> j, p)
```

Variables touched:

```text
pose_i
pose_j
dense_disp_i
```

This is the main visual BA term.

#### DispSensRegularizationTerm

Class:

```python
vipe/slam/ba/terms.py::DispSensRegularizationTerm
```

It is added only when:

```python
not motion_only
and disps_sens for that frame has nonzero sum
```

Residual:

```text
dense_disp[i] - disps_sens[i]
```

Weight:

```yaml
pipeline.slam.ba.dense_disp_alpha: 0.001
```

Per-pixel sensor weight:

```text
disps_sens_weight[i]
```

The effective scalar weight for a low-res pixel is `dense_disp_alpha * disps_sens_weight`.

Variables touched:

```text
dense_disp only
```

This is the keyframe depth prior. It does not directly optimize pose, but by constraining disparity scale it indirectly anchors pose translation scale through the reprojection terms. Invalid sensor pixels have zero regularization weight and therefore do not pull disparity toward zero.

## Toy Example: Three Keyframes

Assume `GraphBuffer` currently has three keyframes:

```text
n_frames = 3
slots = 0,1,2
tstamp = [0, 8, 17]
poses = [P0, P1, P2]
disps = [D0, D1, D2]
disps_sens = [S0, S1, S2]
```

Assume the low-res grid has only 2 pixels to keep the example small:

```text
D0 = [0.50, 0.40]
D1 = [0.55, 0.45]
D2 = [0.60, 0.50]

S0 = [0.48, 0.41]
S1 = [0.57, 0.44]
S2 = [0.61, 0.49]
```

### Add Neighborhood Factors

Call:

```python
graph.add_neighborhood_factors(0, 3, r=1)
```

Edges become:

```text
ii = [0,1,1,2]
jj = [1,0,2,1]
```

Meaning:

```text
0 -> 1
1 -> 0
1 -> 2
2 -> 1
```

For each edge, `add_factors` initializes:

```text
target = current projection from current poses/disps
weight = 0
age = 0
```

### First FactorGraph Update

Call:

```python
graph.update(t0=1, itrs=frontend_ba_iters)
```

For edge `0 -> 1`, suppose current projection for pixel 0 is:

```text
coords1 = [10.0, 5.0]
```

DROID update predicts:

```text
delta = [0.8, 0.2]
weight = [0.7, 0.7]
```

Then:

```text
target = coords1 + delta = [10.8, 5.2]
```

The BA residual for that pixel is:

```text
project(P0, P1, D0[pixel0]) - [10.8, 5.2]
```

At the same time, the depth-anchor residual for frame 0 is:

```text
D0 - S0 = [0.50 - 0.48, 0.40 - 0.41] = [0.02, -0.01]
```

and for frame 1:

```text
D1 - S1 = [0.55 - 0.57, 0.45 - 0.44] = [-0.02, 0.01]
```

`GraphBuffer.bundle_adjustment` builds all such residuals and calls `Solver.run_inplace`.

After the solve, `GraphBuffer` may become:

```text
poses = [P0 fixed, P1 adjusted, P2 adjusted]
disps = [
  [0.49, 0.405],
  [0.56, 0.445],
  [0.605, 0.495],
]
```

This update is in-place. `FactorGraph` still has the same edge list unless factors are later added/removed.

## Which Factors Are Used Where Currently?

### Frontend initialization

Code:

```python
SLAMFrontend.__initialize()
```

The reduced path always does:

```python
graph.add_neighborhood_factors(0, warmup, r=1)
for _ in range(frontend_init_updates):
    graph.update(t0=1, itrs=frontend_ba_iters)
```

Solver-level terms during each `graph.update`:

```text
DenseDepthFlowTerm
DispSensRegularizationTerm
```

because `motion_only=False` by default.

### Frontend incremental updates

Code:

```python
SLAMFrontend.__update()
```

It does:

```python
graph.rm_factors(age > frontend_max_age)
graph.add_proximity_factors(...)
for _ in range(frontend_update_iters1):
    graph.update(itrs=frontend_ba_iters)
```

Then it may prune the second-newest keyframe. If it does not prune:

```python
for _ in range(frontend_update_iters2):
    graph.update(itrs=frontend_ba_iters)
```

Graph-construction factors:

```text
proximity factors
```

Solver-level terms:

```text
DenseDepthFlowTerm
DispSensRegularizationTerm
```

### Backend global BA

Code:

```python
SLAMBackend.run(steps=backend_iters)
```

It creates a fresh graph:

```python
graph = FactorGraph(..., incremental=False)
```

Then:

```python
graph.add_proximity_factors(
    rad=backend_radius,
    nms=backend_nms,
    thresh=backend_thresh,
    beta=beta,
)
graph.update_batch(
    itrs=backend_ba_iters,
    steps=backend_iters,
    batch_size=backend_batch_size,
)
```

Graph-construction factors:

```text
proximity factors over the full keyframe set
```

Solver-level terms:

```text
DenseDepthFlowTerm
DispSensRegularizationTerm
```

Backend is not a continuation of `frontend.graph`. It is a new `FactorGraph` object operating on the same `GraphBuffer`.

## Final Call Path Summary

Frontend:

```text
SLAMSystem.run pass 1
-> keyframe accepted
-> SLAMSystem._add_frontend_keyframe
-> GraphBuffer slot filled
-> GraphBuffer.disps_sens filled from external sensor depth
-> SLAMFrontend.run
-> frontend.graph.add_neighborhood_factors or add_proximity_factors
-> frontend.graph.update
-> GraphBuffer.bundle_adjustment
-> Solver.run_inplace
-> GraphBuffer.poses/disps mutated
```

Backend:

```text
SLAMSystem.run after pass 1
-> SLAMBackend.run
-> local FactorGraph created
-> local graph.add_proximity_factors
-> local graph.update_batch
-> GraphBuffer.bundle_adjustment
-> Solver.run_inplace
-> same GraphBuffer.poses/disps mutated
```

One sentence version:

```text
GraphBuffer stores the variables; FactorGraph chooses and maintains pairwise constraints; FactorGraph.update/update_batch asks DROID to refine those constraints; GraphBuffer.bundle_adjustment builds solver terms and runs the actual optimizer that changes poses and disparities.
```
