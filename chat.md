### Frontend: when it happens

Triggered during **SLAM pass 1**, every time pass-1 loop processes a frame:

```python
if frame is keyframe or last frame:
    add keyframe to GraphBuffer

frontend.run()
```

`frontend.run()` does real work only:
- once when `GraphBuffer.n_frames == warmup`
- later whenever a new keyframe has been added

Current values:
- `warmup = 8` config
- `frontend_update_iters1 = 4` config
- `frontend_update_iters2 = 2` config
- `frontend_ba_iters = 3` config
- `frontend_max_age = 25` config
- `frontend_max_factors = 48` config

### Frontend init pseudocode

Runs **once** when 8 keyframes have accumulated.

```python
# event: first time buffer.n_frames == warmup=8
frontend.t1 = buffer.n_frames  # 8

frontend.graph.add_neighborhood_factors(0, 8, r=1)  # hardcoded adjacent sequential initialization

repeat frontend_init_updates=8 times:  # config
    frontend.graph.update(t0=1, itrs=frontend_ba_iters)
```

Inside each `graph.update(...)`:

```python
# edge list ii,jj stays fixed for this update call

coords1 = reproject using current GraphBuffer.poses/disps
corr = DROID correlation at coords1
delta, weight, damping = DROID update network(...)

FactorGraph.target = coords1 + delta      # updated once per graph.update call
FactorGraph.weight = weight               # updated once per graph.update call
FactorGraph.damping = damping             # updated once per graph.update call

GraphBuffer.bundle_adjustment(..., n_iters=frontend_ba_iters)
```

Inside `bundle_adjustment(n_iters=frontend_ba_iters)`:

```python
# target/weight/damping are fixed during these inner solver iterations

repeat frontend_ba_iters=3 times:  # config
    build/use same solver terms:
        DenseDepthFlowTerm
        DispSensRegularizationTerm
    Solver.run_inplace(...)
    update GraphBuffer.poses
    update GraphBuffer.disps
```

So frontend init totals:

```text
8 outer graph.update calls
each has 1 DROID target/weight refresh
each has 3 inner BA solver iterations
=> 24 solver iterations total
```

### Frontend incremental pseudocode

Runs **multiple times** through pass 1: once per newly accepted keyframe after initialization.

```python
# event: frontend initialized and buffer.n_frames > frontend.t1

frontend.t1 += 1

if graph already has corr:
    remove active factors with age > frontend_max_age=25  # config

frontend.graph.add_proximity_factors(...)
```

Then:

```python
repeat frontend_update_iters1=4 times:  # config
    frontend.graph.update(itrs=frontend_ba_iters)
```

Each `graph.update`:
- keeps current edge list fixed during that call
- refreshes DROID `target/weight/damping` once
- runs `bundle_adjustment(n_iters=frontend_ba_iters)`

Then keyframe pruning check:

```python
d = buffer.frame_distance_dense_disp(t1-3, t1-2)

if d < keyframe_thresh=4.0 config:
    graph.rm_second_newest_keyframe(...)
    buffer slot removed
    graph edge indices adjusted
else:
    repeat frontend_update_iters2=2 times:  # config
        frontend.graph.update(itrs=frontend_ba_iters)
```

So per kept keyframe:

```text
4 + 2 = 6 outer graph.update calls
each has 1 DROID target/weight refresh
each has 3 inner BA solver iterations
=> 18 solver iterations per kept-keyframe frontend update
```

Per pruned keyframe:

```text
4 outer graph.update calls
=> 12 solver iterations
then keyframe removed
```

### What changes in frontend

Fixed during whole frontend lifetime:
```text
GraphBuffer object identity
frontend.graph object identity
frontend.graph max_factors=frontend_max_factors=48
DROID network weights
intrinsics
```

Changes when keyframe is added:
```text
GraphBuffer.n_frames
GraphBuffer.tstamp/images/fmaps/nets/inps
GraphBuffer.disps_sens
```

Changes when factors are added/removed:
```text
FactorGraph.ii/jj
FactorGraph.age
FactorGraph.corr/f_net/inp
FactorGraph.target/weight
```

Changes once per `graph.update` outer step:
```text
FactorGraph.target
FactorGraph.weight
FactorGraph.damping
FactorGraph.f_net
```

Changes during inner BA solver iterations:
```text
GraphBuffer.poses
GraphBuffer.disps
```

---

### Backend: when it happens

Triggered **once** after pass 1 completes.

```python
# event: after all pass-1 keyframes are in GraphBuffer
backend.run(steps=backend_iters)
```

Current values:
- `backend_iters = 31` config
- `backend_ba_iters = 8` config
- `backend_max_factors_per_keyframe = 16` config, so `max_factors = 16 * t`
- `backend_batch_size = 8` config
- `incremental = False` hardcoded for backend graph

### Backend pseudocode

```python
# event: after SLAM pass 1
t = buffer.n_frames  # number of keyframes

graph = FactorGraph(
    net=droid_net,
    buffer=same GraphBuffer,
    max_factors=backend_max_factors_per_keyframe*t,
    incremental=False,
)

graph.add_proximity_factors(
    rad=backend_radius=2 config,
    nms=backend_nms=3 config,
    thresh=backend_thresh=22.0 config,
    beta=beta=0.3 config,
)
```

Important:

```text
backend graph object is new
backend edge list is built once
backend graph is discarded after backend.run finishes
same GraphBuffer is mutated
```

Then:

```python
graph.update_batch(itrs=backend_ba_iters, steps=backend_iters, batch_size=backend_batch_size)
```

Inside `update_batch`:

```python
corr_op = AltCorrBlock(buffer.fmaps)

repeat steps=31 times:  # config backend_iters
    coords1 = reproject using current GraphBuffer.poses/disps

    for edges in source-index batches of backend_batch_size=8:  # config
        corr = corr_op(...)
        net, delta, weight, damping = DROID update network(...)

        FactorGraph.target[edges] = coords1 + delta
        FactorGraph.weight[edges] = weight
        FactorGraph.damping[frames] = damping

    GraphBuffer.bundle_adjustment(..., n_iters=backend_ba_iters)
```

Inside each backend `bundle_adjustment(n_iters=backend_ba_iters)`:

```python
# edge list fixed
# target/weight/damping fixed for these 8 solver iterations

repeat 8 times:
    DenseDepthFlowTerm
    DispSensRegularizationTerm
    Solver.run_inplace(...)
    update GraphBuffer.poses
    update GraphBuffer.disps
```

So backend totals:

```text
31 outer update_batch steps
each has 1 full DROID target/weight refresh over all backend edges
each has 8 inner BA solver iterations
=> 248 solver iterations total
```

### What changes in backend

Fixed for whole backend run:
```text
backend FactorGraph edge list ii/jj
GraphBuffer object identity
DROID network weights
intrinsics
```

Changes once per backend outer step:
```text
FactorGraph.target
FactorGraph.weight
FactorGraph.damping
FactorGraph.f_net
```

Changes during inner BA solver iterations:
```text
GraphBuffer.poses
GraphBuffer.disps
```

Does not change in backend:
```text
GraphBuffer.n_frames
GraphBuffer.images
GraphBuffer.fmaps
GraphBuffer.nets/inps
GraphBuffer.disps_sens
FactorGraph.ii/jj edge set
```

### After backend

Pass 2 appends original frames in infill chunks, initializes their poses from neighboring keyframes, and runs motion-only graph updates for those appended poses. The final SLAM handoff is:

```text
SLAMOutput.trajectory  # camera-to-world pose for every input frame
SLAMOutput.intrinsics  # one recovered original-resolution pinhole intrinsics vector
SLAMOutput.keyframe_indices  # selected-frame indices of optimized SLAM keyframes
```

Final DAV3 depth uses `keyframe_indices` only to re-read neighboring keyframes as context frames. No `FactorGraph` or dense keyframe-map object is passed to the output stage.

### One-line mental model

```text
Outer graph update step = refresh learned targets/weights using DROID, then call BA.
Inner BA solver iterations = optimize GraphBuffer.poses/disps against fixed targets/weights.
```
