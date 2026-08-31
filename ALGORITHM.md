# ViPE Instance Distillation

This document specifies the class-agnostic 3D instance pipeline that runs inside ViPE after the existing pose and reconstruction pipeline. ViPE Stages 1-5 are documented in [`fullexplain.md`](fullexplain.md). The instance path starts only after Stage 5 has successfully written the final pose and TSDF artifacts.

The instance algorithm consumes RGB, sensor depth, shared pinhole intrinsics, and final ViPE camera-to-world poses. It does not consume GT labels or semantic features. Its output is an overlapping, K-capped set of 3D instance hypotheses.

```mermaid
flowchart LR
    V[ViPE Stages 1-5<br/>poses and TSDF] --> S6[Stage 6<br/>instance cloud and frame coreset]
    S6 --> S7[Stage 7<br/>SAM1 seeds and SAM2 propagation]
    S7 --> S8[Stage 8<br/>3D lift and track linking]
    S8 --> S9[Stage 9<br/>normal-aware atoms]
    S9 --> S10[Stage 10<br/>affinity and hierarchy]
    S10 --> S11[Stage 11<br/>evidence candidates]
    S11 --> S12[Stage 12<br/>K-capped selection and exchange]
    S12 --> O[Instance NPZ and visualization PLY]
```

## Representation

The algorithm uses four nested representations:

| Level | Symbol | Meaning |
| --- | --- | --- |
| Voxel | `i` | One point in the temporary 2 cm occupancy cloud; this is the instance prediction domain. |
| Atom | `a` | A small connected, normal-aware surface patch containing whole voxels. Atoms partition the cloud. |
| Node | `R` | A union of atoms retained from the agglomeration hierarchy. Nodes are nested or disjoint. |
| Hypothesis | `h` | A selected node or a linked union of nodes representing one possible 3D instance. Hypotheses may overlap. |

For frame `f` and lifted mask `gm`:

```text
V_f       visible cloud voxels in frame f
q_gm      voxels claimed by mask gm; q_gm is a subset of V_f
N_a,f     number of voxels from atom a visible in frame f
I_a,gm    number of voxels from atom a claimed by mask gm
```

The final output is a hypothesis soup with at most `K=5` hypotheses covering any atom. It is not a hard partition. The PLY visualization resolves overlap by assigning each voxel to the smallest selected hypothesis that contains it.

## Stage 6: Instance Cloud And Frame Coreset

Stage 6 runs after ViPE Stage 5 and uses the final ViPE trajectory. It replays canonical sensor depth and backprojects every valid depth pixel:

```math
X_c = d\begin{bmatrix}(u-c_x)/f_x \\ (v-c_y)/f_y \\ 1\end{bmatrix},
\qquad
X_w = R_{c2w}X_c+t_{c2w}.
```

Only finite depths strictly between `0.1 m` and `12 m` enter the Replica frontier cloud. World points are quantized into 2 cm cells:

```math
v = \left\lfloor X_w / 0.02 \right\rfloor,
\qquad
p_v = (v+0.5)\,0.02.
```

The union of occupied cells is the instance cloud. It is separate from ViPE's TSDF surface during initial parity validation.

The frame coreset walks final poses in sequence order. Frame `f` is retained when, relative to the last retained frame, either:

```math
\lVert t_f-t_r\rVert_2 > 0.08\text{ m}
```

or the optical-axis angle exceeds `8 degrees`. The retained frame becomes the next reference. No RGB or depth is loaded during this selection.

Instance images use a separate 1024-pixel-long-side view. RGB is aspect-preserving LANCZOS-resized; depth is nearest-neighbor resized; intrinsics are scaled independently in x and y. This does not alter the canonical ViPE input or TSDF.

## Stage 7: 2D Mask Generation And Propagation

Retained frames are processed in consecutive chunks of four.

### Stage 7.1: SAM1 AMG Seeds

SAM1 ViT-H automatic mask generation runs on the first frame of each chunk with:

- a `48 x 48` query grid;
- predicted-IoU threshold `0.6`;
- stability threshold `0.9` with offset `1.0`;
- box NMS threshold `0.7`;
- no crop pyramid and no mask-region postprocessing.

Survivors are stably sorted by predicted IoU and capped at 100. Each seed receives a monotonically allocated track ID. IDs are never reused across chunks.

### Stage 7.2: SAM2 Propagation

SAM2.1-small receives every seed in one inference state and propagates them forward through the remaining chunk frames. The mask-logit threshold is `-1.0`; postprocessing and CPU offload are disabled.

The seed frame keeps the original SAM1 masks. Later frames carry the same track IDs and inherited SAM1 scores. A track may disappear and later reappear within its chunk. Track IDs never cross chunk boundaries at this stage.

Mask generation and Stage 8 lifting are interleaved one chunk at a time. Once a chunk has been lifted, its decoded masks and SAM state are released.

## Stage 8: Lift And Global Track Linking

### Stage 8.1: Visibility And Lift

For cloud point `X_w`, final pose `T_{c2w}`, and frame depth map `D_f`, compute camera coordinates and projection:

```math
X_c = R_{c2w}^{T}(X_w-t_{c2w}),
\qquad
(u,v)=\left(f_x X_c^x/X_c^z+c_x,\ f_y X_c^y/X_c^z+c_y\right).
```

A projected voxel is visible only when it is in front of the camera, inside the image, has valid measured depth, and:

```math
\left|X_c^z-D_f(u,v)\right| \le 0.05\text{ m}.
```

Each 2D mask selects from those visible voxel indices, producing `q_gm`. Lifted masks with fewer than five voxels are discarded. Frame provenance and chunk-local track IDs remain attached.

### Stage 8.2: Global Track Linking

After every retained frame is lifted, each chunk-local track is represented by the union of all voxel sets claimed by that track. Candidate track pairs must share voxels and come from different chunks.

For every ordered chunk pair, a track pair is accepted only when:

- 3D set IoU is at least `0.8`;
- each track is the other's best partner in the opposite chunk;
- the accepted IoU exceeds both runner-up IoUs by a factor of at least `1.2`.

Accepted pairs are processed in descending IoU with union-find. A connected component may contain at most one track from each chunk, preventing same-frame part/whole seeds from collapsing. The minimum track ID becomes the deterministic global ID.

## Stage 9: Normal-Aware Atoms

Estimate cloud normals with Open3D using radius:

```math
r_n=\max(0.06\text{ m},\ 4\cdot0.02\text{ m})
```

and at most 30 neighbors. Build a symmetric 12-neighbor voxel graph, capped at `2.5` voxel units. Edge cost is:

```math
w(i,j)=\lVert p_i-p_j\rVert_2\left(1+4\left(1-|n_i^Tn_j|\right)\right).
```

Seed one voxel in each occupied 3 cm cell. Multi-source Dijkstra assigns every voxel to its lowest-cost seed, producing an atom partition. Atom adjacency consists of voxel-graph edges crossing between atoms.

## Stage 10: Affinity And Hierarchy

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

## Stage 11: Evidence Candidates

### Stage 11.1: Per-Node Evidence

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

Nodes smaller than ten voxels are not candidates. Masks with coverage at least `0.5` enter the evidence-edge stage.

### Stage 11.2: Evidence-Edge Clustering

For two co-eligible, non-nested nodes, a frame provides graded join evidence when one mask covers both. The union endorsement is:

```math
\operatorname{cov}_u^2\operatorname{pur}_u^2\operatorname{IoU}_u.
```

A co-eligible frame with no co-covering mask is a miss. For total join weight `k` over `n` opportunities, edge cost is the same-vs-different log-likelihood ratio:

```math
k\ln\frac{p_{same}}{p_{diff}}+(n-k)\ln\frac{1-p_{same}}{1-p_{diff}}.
```

`p_same` and `p_diff` are fit per scene by an unlabeled two-component binomial-mixture EM. Positive pair edges and positive greedy contractions become additional union candidates. A separate track channel proposes cross-viewpoint unions when one global track covers different nodes in different frames. Track proposals add candidates but no negative evidence.

## Stage 12: K-Capped Selection And Exchange

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
instances/<scene>.npz           occupancy points plus packed overlapping hypotheses
instances/<scene>_summary.json  resolved config, timings, and structural counts
pcd/<scene>_instances.ply       smallest-hypothesis-wins visualization
```

The instance NPZ is authoritative. It stores cloud points, concatenated hypothesis voxel indices, offsets, and K. The visualization PLY is not used by the algorithm or metric.

## Replica Evaluation Boundary

Replica evaluation is separate from runtime. It builds a GT 2 cm cloud from canonical depth and GT poses, labels it from `mesh_semantic.ply`, aligns the predicted ViPE trajectory to GT by first solving one global SE3 transform, and transfers GT instance IDs to the predicted occupancy cloud by nearest neighbor.

Known annotation-debris IDs are relabeled to background before scoring:

```text
office0: 1, 13, 18, 26, 56, 65, 67
office2: 5, 6, 7, 78, 83
room0:   17, 26, 28, 29, 37, 38, 42, 52, 53, 62, 66, 82, 91
```

Excluded voxels remain in the cloud, so they still penalize a predicted hypothesis that includes them.

For each GT instance `g`, evaluation computes the best point-set IoU over all hypotheses. Average Recall is the mean recall over IoU thresholds `0.50, 0.55, ..., 0.95`. R50, R75, R90, hypothesis count, and mean/max voxel membership are reported alongside AR.

## Determinism And Lifetime

Ordering is part of the algorithm: frames and chunks are ascending, SAM top-k sorting is stable, track IDs are monotonic, union-find keeps the minimum representative, and projection uses explicit elementwise arithmetic. TF32 remains disabled and deterministic ViPE settings remain active.

Stage 5 releases SLAM GPU state before SAM models are built. Stage 7 retains one mask chunk at a time. Mask tensors, SAM2 state, temporary JPEGs, lifted intermediates, and hierarchy tables are released immediately after their last consumer. Only final packed hypotheses and the temporary occupancy cloud survive to artifact writing.

The occupancy cloud is the initial parity scaffold. Any later migration to the native ViPE TSDF surface is a separate user-approved task, not part of this implementation.
