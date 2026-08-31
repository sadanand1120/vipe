# ViPE Instance Distillation

This document specifies the class-agnostic 3D instance pipeline that runs inside ViPE after the existing pose and reconstruction pipeline. ViPE Stages 1-5 are documented in [`fullexplain.md`](fullexplain.md). The instance path starts only after Stage 5 has successfully written the final pose and TSDF artifacts.

The instance algorithm consumes the native TSDF surface, RGB, sensor depth, shared pinhole intrinsics, and final ViPE camera-to-world poses. It does not consume GT labels or semantic features. Its output is an overlapping, K-capped set of 3D instance hypotheses.

```mermaid
flowchart LR
    V[ViPE Stages 1-5<br/>poses and TSDF] --> S6[Stage 6<br/>TSDF surface and frame coreset]
    S6 --> S7[Stage 7<br/>normal-aware atoms]
    S7 --> S8[Stage 8<br/>masks, lift, and track linking]
    S8 --> S9[Stage 9<br/>affinity and hierarchy]
    S9 --> S10[Stage 10<br/>evidence candidates]
    S10 --> S11[Stage 11<br/>K-capped selection and exchange]
    S11 --> O[Instance NPZ and visualization PLY]
```

## Representation

The algorithm uses four nested representations:

| Level | Symbol | Meaning |
| --- | --- | --- |
| Surface point | `i` | One native TSDF zero-surface sample retained as the instance prediction domain. |
| Atom | `a` | A small connected, normal-aware surface patch containing surface points. Atoms partition the surface. |
| Node | `R` | A union of atoms retained from the agglomeration hierarchy. Nodes are nested or disjoint. |
| Hypothesis | `h` | A selected node or a linked union of nodes representing one possible 3D instance. Hypotheses may overlap. |

For frame `f` and lifted mask `gm`:

```text
V_f       visible TSDF surface points in frame f
q_gm      points claimed by mask gm; q_gm is a subset of V_f
N_a,f     number of points from atom a visible in frame f
I_a,gm    number of points from atom a claimed by mask gm
```

The final output is a hypothesis soup with at most `K=5` hypotheses covering any atom. It is not a hard partition. The PLY visualization resolves overlap by assigning each surface point to the smallest selected hypothesis that contains it.

## Stage 6: TSDF Surface And Frame Coreset

Stage 5 extracts the fused TSDF zero surface once. The native extension writes the full RGB+normal PLY in bounded chunks while simultaneously retaining the compact surface domain needed by Stage 6. Python receives only those retained point/normal pairs, not the dense output sample tensors. There is no PLY reread and no second all-frame backprojection.

The saved PLY contains a dense deterministic area sample. Running association over all ten million output samples would make point count depend on an output-density knob rather than TSDF resolution. During that same native extraction stream, Stage 5 retains one original TSDF sample per native TSDF-sized cell for Stage 6. For TSDF voxel edge `s`, sample `p_i`, and cell index `c_i`:

```math
c_i = \left\lfloor p_i/s \right\rfloor,
\qquad
i_c = \underset{i:\,c_i=c}{\arg\min}\;
\left\|p_i-(c+\tfrac{1}{2})s\right\|_2^2.
```

Equal-distance ties select the lower original extraction index. The retained point is an actual zero-surface sample, not a cell centroid. Cell keys are sorted deterministically, so every hypothesis index has a stable surface-point meaning. The cell edge comes from `pipeline.output.pcd_tsdf_voxel_edge_m`; it is not duplicated in the instance config.

The frame coreset walks final poses in sequence order. Frame `f` is retained when, relative to the last retained frame, either:

```math
\lVert t_f-t_r\rVert_2 > 0.08\text{ m}
```

or the optical-axis angle exceeds `8 degrees`. The retained frame becomes the next reference. No RGB or depth is loaded during this selection.

Instance images use a separate 1024-pixel-long-side view. RGB is aspect-preserving LANCZOS-resized; depth is nearest-neighbor resized; intrinsics are scaled independently in x and y. This does not alter the canonical ViPE input or TSDF.

## Stage 7: Normal-Aware Atoms

Use the native TSDF-gradient normal carried by each retained surface sample. Build a symmetric 12-neighbor point graph, capped at `2.5` TSDF voxel edges. Edge cost is:

```math
w(i,j)=\lVert p_i-p_j\rVert_2\left(1+4\left(1-|n_i^Tn_j|\right)\right).
```

Seed one surface point in each occupied 3 cm cell. Multi-source Dijkstra assigns every point to its lowest-cost seed, producing an atom partition. Atom adjacency consists of point-graph edges crossing between atoms.

## Stage 8: Masks, Lift, And Global Track Linking

Retained frames are processed in consecutive chunks of four. Mask generation and lifting are interleaved one chunk at a time.

### Stage 8.1: SAM1 AMG Seeds

SAM1 ViT-H automatic mask generation runs on the first frame of each chunk with:

- a `48 x 48` query grid;
- predicted-IoU threshold `0.6`;
- stability threshold `0.9` with offset `1.0`;
- box NMS threshold `0.7`;
- no crop pyramid and no mask-region postprocessing.

Survivors are stably sorted by predicted IoU and capped at 100. Each seed receives a monotonically allocated track ID. IDs are never reused across chunks.

### Stage 8.2: SAM2 Propagation

SAM2.1-small receives every seed in one inference state and propagates them forward through the remaining chunk frames. The mask-logit threshold is `-1.0`; postprocessing and CPU offload are disabled.

The seed frame keeps the original SAM1 masks. Later frames carry the same track IDs and inherited SAM1 scores. A track may disappear and later reappear within its chunk. Track IDs never cross chunk boundaries at this stage.

Once a chunk has been lifted, its decoded masks and SAM state are released.

### Stage 8.3: Visibility And Lift

For surface point `X_w`, final pose `T_{c2w}`, and frame depth map `D_f`, compute camera coordinates and projection:

```math
X_c = R_{c2w}^{T}(X_w-t_{c2w}),
\qquad
(u,v)=\left(f_x X_c^x/X_c^z+c_x,\ f_y X_c^y/X_c^z+c_y\right).
```

A projected point is visible only when it is in front of the camera, inside the image, has valid measured depth, and:

```math
\left|X_c^z-D_f(u,v)\right| \le 0.05\text{ m}.
```

Each 2D mask selects from those visible surface-point indices, producing `q_gm`. Lifted masks with fewer than five points are discarded. The lift immediately counts visible and claimed points per atom and stores the nonzero counts in two atom-major CSR tables:

```text
leafI(atom, mask)  = I_a,gm
leafN(atom, frame) = N_a,f
```

No dense `number_of_atoms x number_of_masks` membership matrix exists. Frame provenance and chunk-local track IDs remain attached to the sparse mask records.

### Stage 8.4: Global Track Linking

After every retained frame is lifted, each chunk-local track is represented by the union of all point sets claimed by that track. Candidate track pairs must share points and come from different chunks.

For every ordered chunk pair, a track pair is accepted only when:

- 3D set IoU is at least `0.8`;
- each track is the other's best partner in the opposite chunk;
- the accepted IoU exceeds both runner-up IoUs by a factor of at least `1.2`.

Accepted pairs are processed in descending IoU with union-find. A connected component may contain at most one track from each chunk, preventing same-frame part/whole seeds from collapsing. The minimum track ID becomes the deterministic global ID. Only one exact union of claimed point indices is retained per chunk-local track for this linking operation; it is released immediately after global IDs are resolved.

## Stage 9: Affinity And Hierarchy

For atom `a`, mask `gm`, and the mask's frame `f(gm)`, define visible claim fraction:

```math
x_{a,gm}=I_{a,gm}/N_{a,f(gm)}.
```

Within each atom/frame block, normalize mask claims and weight weakly claimed atoms by the square root of the maximum claim:

```math
\hat{x}_{a,gm}=
\frac{x_{a,gm}}{\lVert x_{a,\cdot}\rVert_2}
\sqrt{\max_{gm}x_{a,gm}}.
```

For adjacent atoms `a,b`:

```math
\operatorname{num}(a,b)=\sum_{gm}\hat{x}_{a,gm}\hat{x}_{b,gm},
```

```math
W(a,b)=\#\{f:\ a,b\text{ co-visible and at least one is masked}\},
\qquad
A(a,b)=\operatorname{num}(a,b)/W(a,b).
```

A co-visible frame masking only one atom contributes an opportunity but no agreement.

Before agglomeration, adjacent atoms with `A >= 0.98` over at least eight observations are contracted into meta-atoms. Agglomeration then repeatedly merges the adjacent region pair with the strongest pooled border evidence under a scene-fitted Beta prior. Every child and merged parent remains a candidate node.

Nested candidates with IoU at least `0.95` are collapsed to the largest representative, creating an epsilon-cover of the hierarchy without turning it into a partition.

## Stage 10: Evidence Candidates

### Stage 10.1: Per-Node Evidence

A post-order pass sums visibility and mask intersections from atoms to every node. A frame can judge node `R` only when:

```math
N_{R,f}\ge\max(8,\ 0.05|R|).
```

For each eligible frame and mask:

```math
\operatorname{IoU}(R,gm)=
\frac{I_{R,gm}}{|q_{gm}|+N_{R,f}-I_{R,gm}},
```

```math
\operatorname{cov}(R,gm)=I_{R,gm}/N_{R,f},
\qquad
\operatorname{pur}(R,gm)=I_{R,gm}/|q_{gm}|.
```

Nodes smaller than ten points are not candidates. Masks with coverage at least `0.5` enter the evidence-edge stage.

### Stage 10.2: Evidence-Edge Clustering

For two co-eligible, non-nested nodes, a frame provides graded join evidence when one mask covers both. The union endorsement is:

```math
\operatorname{cov}_u^2\operatorname{pur}_u^2\operatorname{IoU}_u.
```

A co-eligible frame with no co-covering mask is a miss. For total join weight `k` over `n` opportunities, edge cost is the same-vs-different log-likelihood ratio:

```math
k\ln\frac{p_{same}}{p_{diff}}+(n-k)\ln\frac{1-p_{same}}{1-p_{diff}}.
```

`p_same` and `p_diff` are fit per scene by an unlabeled two-component binomial-mixture EM. Positive pair edges and positive greedy contractions become additional union candidates. A separate track channel proposes cross-viewpoint unions when one global track covers different nodes in different frames. Track proposals add candidates but no negative evidence.

## Stage 11: K-Capped Selection And Exchange

Selection chooses hypotheses by maximizing:

```math
F(S)=\sum_e w_e\max_{h\in S}\phi_h(e)-0.5|S|
\quad\text{subject to at most }K=5\text{ selected hypotheses per atom}.
```

For a mask endorsement:

```math
\phi_h(e)=\operatorname{cov}(h,e)^2\operatorname{pur}(h,e)^2
\operatorname{IoU}(h,e).
```

Weights below `0.05` are omitted as a compute guard. Clients are global tracks with equal weight. A track's value for a hypothesis is the mean over its masks in eligible frames; an eligible frame with no endorsement contributes zero.

Lazy greedy selection uses hypothesis size only as a tie-break and stops when marginal gain falls below the `0.5` hypothesis price. It also enforces at most one variant of a node and the per-atom K cap.

Finally, if two or more selected hypotheses lie inside one tree ancestor, replace the group with that ancestor only when total evidence does not decrease and the K cap remains satisfied:

```math
\operatorname{loss}=\sum_g\left[
\operatorname{best}_g(S)-
\max\left(\operatorname{best}_g(S\setminus G),\phi_{ancestor}(g)\right)
\right].
```

Accept only `loss <= 0`. Process larger ancestors first and repeat to a fixed point for at most three passes.

## Runtime Artifacts

Runtime writes no GT or semantic data:

```text
pose/<scene>.npz                 final ViPE camera-to-world trajectory
pcd/<scene>_tsdf.ply            native ViPE RGB+normal TSDF surface
instances/<scene>.npz           TSDF surface points plus packed overlapping hypotheses
instances/<scene>_summary.json  resolved config, timings, and structural counts
pcd/<scene>_instances.ply       smallest-hypothesis-wins visualization
```

The instance NPZ is authoritative. It stores retained TSDF surface points, concatenated hypothesis point indices, offsets, K, the `tsdf_surface` domain tag, and the TSDF voxel edge. The visualization PLY is not used by the algorithm or metric. Its colors are assigned so hypotheses that touch on the output surface are visually distinct.

## Replica Evaluation Boundary

Replica evaluation is separate from runtime. It builds a GT 2 cm cloud from canonical depth and GT poses, labels it from `mesh_semantic.ply`, aligns the predicted ViPE trajectory to GT by first solving one global SE3 transform, and transfers GT instance IDs to the predicted TSDF surface by nearest neighbor.

Known annotation-debris IDs are relabeled to background before scoring:

```text
office0: 1, 13, 18, 26, 56, 65, 67
office2: 5, 6, 7, 78, 83
room0:   17, 26, 28, 29, 37, 38, 42, 52, 53, 62, 66, 82, 91
```

Excluded labels remain represented on the prediction surface as background, so they still penalize a predicted hypothesis that includes them.

For each GT instance `g`, evaluation computes the best point-set IoU over all hypotheses. Average Recall is the mean recall over IoU thresholds `0.50, 0.55, ..., 0.95`. R50, R75, R90, hypothesis count, and mean/max point membership are reported alongside AR.

Evaluation also writes `pcd/<scene>_instances_gtmatch.ply`. It keeps the unique best predicted hypothesis for every GT instance whose best IoU is at least `0.30`, then applies the same smallest-hypothesis-wins visualization and spatially contrasting colors. This is a benchmark visualization only: GT labels select hypotheses after runtime has completed and never enter instance distillation.

## Determinism And Lifetime

Ordering is part of the algorithm: frames and chunks are ascending, SAM top-k sorting is stable, track IDs are monotonic, union-find keeps the minimum representative, and projection uses explicit elementwise arithmetic. TF32 remains disabled and deterministic ViPE settings remain active.

Stage 5 releases SLAM GPU state before SAM models are built. Native TSDF extraction streams the dense PLY in bounded chunks and returns only the compact Stage-6 surface domain. Stage 8 retains one mask chunk at a time and converts each lifted mask directly into sparse atom counts. Mask tensors, SAM2 state, temporary JPEGs, exact track unions, sparse evidence, and hierarchy tables are released after their last consumer. Only the retained TSDF surface and final packed hypotheses survive to instance artifact writing.
