"""GT-free association from lifted masks to overlapping 3D hypotheses.

The numerical path follows the frozen ``instance_bench@a887292`` association core:

   7 atoms      normal-aware over-segmentation (3cm seeds) + adjacency graph
   9 tree       per-atom signatures -> affinity -> border-mean agglomeration; EVERY node is
                retained, then >=95%-nested chains collapse to their largest member (an
                epsilon-cover of the laminar family at the metric's own top IoU threshold)
  10 evidence   per-node evidence tables (fr_stats / cov_stats / D / vis_bins), then evidence-edge
                clustering over them: shared covering masks become signed join/miss edges, every
                positive edge and every GAEC contraction becomes a union candidate
  11 select     the objective: F(S) = sum_e max_h phi_h(e) - lam*|S| s.t. per-atom <= K, with
                graded phi (cov^alpha * pur^beta * IoU) and clients = global tracks; then the
                zero-loss ancestor swap, the exchange move on the greedy cover"""
import time
from collections import defaultdict

import numpy as np

from vipe.instance import atoms as ATOMS


def _ranges(starts, lens):
    """Concatenated [start, start+len) index ranges, vectorised."""
    keep = lens > 0
    starts, lens = starts[keep], lens[keep]
    tot = int(lens.sum())
    if tot == 0:
        return np.empty(0, np.int64)
    return np.arange(tot, dtype=np.int64) + np.repeat(
        starts - np.concatenate(([0], np.cumsum(lens[:-1]))), lens)


def build_tables(T, log=print):
    """Stage 10's view of the sparse leaf tables; this is a relabeling, not a rebuild."""
    return {"rI": (T["leafI_ptr"], T["leafI_gm"], T["leafI_c"]),
            "rN": (T["leafN_ptr"], T["leafN_f"], T["leafN_c"]),
            "gm_frame": np.asarray(T["gm_frame"], np.int64),
            }


def _merge_sparse_counts(k1, v1, k2, v2):
    """Add two sorted sparse count vectors without allocating a dense key space."""
    if not len(k1):
        return k2, np.asarray(v2, np.float64)
    if not len(k2):
        return k1, np.asarray(v1, np.float64)
    keys = np.union1d(k1, k2)
    values = np.zeros(len(keys), np.float64)
    values[np.searchsorted(keys, k1)] = v1
    values[np.searchsorted(keys, k2)] += v2
    return keys, values


def _tree_evidence(T, tables, targets):
    """Yield exact evidence for target tree nodes using one bounded-memory post-order pass."""
    targets = {int(node) for node in targets}
    if not targets:
        return

    na = T["na"]
    children = T["children"]
    offI, gI, vI = tables["rI"]
    offN, fN, vN = tables["rN"]

    needed = np.zeros(T["n_nodes"], bool)
    stack = list(targets)
    while stack:
        node = stack.pop()
        if needed[node]:
            continue
        needed[node] = True
        if node >= na:
            stack.extend(children[node])

    raw = {}
    for node in T["order_nodes"]:
        if not needed[node]:
            continue
        if node < na:
            evidence = (
                gI[offI[node]:offI[node + 1]],
                np.asarray(vI[offI[node]:offI[node + 1]], np.float64),
                fN[offN[node]:offN[node + 1]],
                np.asarray(vN[offN[node]:offN[node + 1]], np.float64),
            )
        else:
            left, right = children[node]
            left_evidence = raw.pop(left)
            right_evidence = raw.pop(right)
            evidence = (
                *_merge_sparse_counts(*left_evidence[:2], *right_evidence[:2]),
                *_merge_sparse_counts(*left_evidence[2:], *right_evidence[2:]),
            )
        raw[node] = evidence
        if node in targets:
            yield node, evidence if len(evidence[0]) else None
        parent = T["parent_of"].get(node)
        if parent is None or not needed[parent]:
            raw.pop(node)


def frame_coreset_poses(frames, c2w_of, move_cm, move_deg, log=print):
    """Stage 6 frame coreset: walk the trajectory in order and keep a frame once
    the camera has moved > move_cm or turned > move_deg since the last KEPT one, which then becomes the
    reference. Pose-only, so it runs before the lift.

    The sampling rate follows the trajectory's own speed: a slow pan or a dwell yields few frames, a
    fast sweep yields many. Nothing is fixed per unit of trajectory the way a per-bin quota is."""
    C = np.array([np.asarray(c2w_of(f), np.float32)[:3, 3] for f in frames], np.float64)
    Z = np.array([np.asarray(c2w_of(f), np.float32)[:3, 2] for f in frames], np.float64)
    cos_thr = np.cos(np.deg2rad(move_deg)); d_thr = move_cm / 100.0
    keep = [0]; ref = 0
    for i in range(1, len(frames)):
        if np.linalg.norm(C[i] - C[ref]) > d_thr or float(Z[i] @ Z[ref]) < cos_thr:
            keep.append(i); ref = i
    log(f"  [hier] PRE-LIFT frame coreset: {len(frames)} -> {len(keep)} frames "
        f"(>{move_cm} cm or >{move_deg}deg since the last kept)")
    return [frames[i] for i in keep]


def build_atom_graph(pts, normals, config, ds, log=print):
    """Stage 7: build normal-aware atoms and their adjacency graph.
    knn/rcap_mult shape the voxel kNN graph that BOTH the Dijkstra grouping and the atom adjacency
    derive from. `acnt` = per-edge crossing-voxel-edge count, a contact-area proxy, logged only."""
    t = time.time()
    atom_cm = float(config["atom_size_m"]) * 100.0
    lam = float(config["atom_normal_weight"])
    k = int(config["atom_knn"])
    rcap_mult = float(config["atom_radius_cap_voxels"])
    atom_of, edges = ATOMS.build_atoms(
        pts,
        normals,
        seed_resolution=atom_cm / 100.0,
        k=k,
        normal_weight=lam,
        radius_cap_voxels=rcap_mult,
        voxel_size=ds,
    )
    av = ATOMS.atom_voxels(atom_of)
    aa, ab, acnt = ATOMS.atom_adjacency(atom_of, edges)
    log(f"  [hier] {len(av)} atoms (~{atom_cm}cm lam{lam} k{k} rcap{rcap_mult}), "
        f"{len(aa)} adj-edges (mean cross {acnt.mean():.1f}) ({time.time()-t:.0f}s)")
    return {"atom_of": atom_of, "av": av, "aa": aa, "ab": ab, "acnt": acnt,
            "asize": np.array([a.size for a in av], np.int64)}


def build_tree(evidence, A, mustlink=None, log=print):
    """Stage 9: consume accumulated adjacency affinity and agglomerate atoms.
    A is the atom-graph output. Border weighting is count (each adjacent atom
    pair = 1; area/sqrt weights measured worse). Linkage is weighted mean over the border: dyadic
    UPGMA was measured worse (its extents kill R@0.9). Returns T: atoms + mask tables + tree.

    Every kept frame counts once. Stage 6 keeps a frame only after real camera motion, so there are no
    near-duplicate views left to down-weight -- "distinct viewpoints" is just "distinct frames"."""
    atom_of, av, na = A["atom_of"], A["av"], len(A["av"])
    gm_frame = np.asarray(evidence["gm_frame"], np.int64)
    gm_size = np.asarray(evidence["gm_size"], np.int64)
    leafI_ptr = evidence["leafI_ptr"]
    leafI_gm = evidence["leafI_gm"]
    leafI_c = evidence["leafI_c"]
    leafN_ptr = evidence["leafN_ptr"]
    leafN_f = evidence["leafN_f"]
    leafN_c = evidence["leafN_c"]
    log(f"  [hier] {len(gm_frame)} masks | leaf sums: {len(leafI_gm)} I-rows "
        f"{len(leafN_f)} N-rows")

    # Stage 8 accumulates each frame's normalised signatures into adjacency-edge statistics, then
    # discards the frame signature. Raw leaf counts above remain available to Stages 10-11.
    t = time.time()
    edge_index = evidence.pop("affinity_edge")
    enum_ = evidence.pop("affinity_num")
    eW_ = evidence.pop("affinity_weight")
    ea, eb = A["aa"][edge_index], A["ab"][edge_index]
    eaff = enum_ / eW_
    log(f"  [hier] {len(eaff)} accumulated positive-affinity edges ({time.time()-t:.0f}s)")

    # ---- MUST-LINK contraction: atoms that the masks can NEVER tell apart become one meta-atom.
    #      P(inseparable | num, W) from a two-component soft-binomial mixture over the edges (EM,
    #      no labels); contraction merges only what no cut, endorsement or selection could ever
    #      separate, so it sheds candidate mass without touching the evidence-achievable ceiling.
    if mustlink is not None and float(mustlink["min_aff"]) > 0:
        t = time.time()
        _ma, _wm = float(mustlink["min_aff"]), float(mustlink["min_w"])
        ml = (enum_ / eW_ >= _ma) & (eW_ >= _wm)
        # union-find over must-link edges -> meta-atom ids
        par = np.arange(na, dtype=np.int64)

        def find(x):
            root = x
            while par[root] != root:
                root = par[root]
            while par[x] != root:
                par[x], x = root, par[x]
            return root
        for a2, b2 in zip(ea[ml].tolist(), eb[ml].tolist()):
            ra, rb = find(a2), find(b2)
            if ra != rb:
                par[max(ra, rb)] = min(ra, rb)
        roots = np.array([find(a2) for a2 in range(na)], np.int64)
        _u, m_of = np.unique(roots, return_inverse=True)
        nm = len(_u)
        log(f"  [hier] must-link: {int(ml.sum())} edges contracted "
            f"-> {na} atoms => {nm} meta-atoms ({time.time()-t:.0f}s)")
        if nm < na:
            # rebuild everything downstream consumes, at meta granularity
            atom_of = m_of[atom_of]
            av_new = [[] for _ in range(nm)]
            for a2 in range(na):
                av_new[m_of[a2]].append(av[a2])
            av = [np.sort(np.concatenate(vs)).astype(np.int32) for vs in av_new]

            def _remap_csr(ptr, key, val):
                rows = m_of[np.repeat(np.arange(na), np.diff(ptr))]
                o2 = np.lexsort((key, rows))
                rows, key, val = rows[o2], key[o2], val[o2].astype(np.float64)
                newk = np.r_[True, (rows[1:] != rows[:-1]) | (key[1:] != key[:-1])]
                gidx = np.cumsum(newk) - 1
                vals = np.bincount(gidx, weights=val).astype(np.float32)
                rows, key = rows[newk], key[newk]
                nptr = np.zeros(nm + 1, np.int64)
                np.cumsum(np.bincount(rows, minlength=nm), out=nptr[1:])
                return nptr, key, vals
            leafI_ptr, leafI_gm, leafI_c = _remap_csr(leafI_ptr, leafI_gm, leafI_c)
            leafN_ptr, leafN_f, leafN_c = _remap_csr(leafN_ptr, leafN_f, leafN_c)
            # aggregate affinity edges by meta pair (internal edges drop)
            ma, mb = m_of[ea], m_of[eb]
            keep = ma != mb
            lo2 = np.minimum(ma[keep], mb[keep]); hi2 = np.maximum(ma[keep], mb[keep])
            ekey = lo2 * nm + hi2
            if len(ekey):
                o2 = np.argsort(ekey, kind="stable")
                ekey, en, ew = ekey[o2], enum_[keep][o2], eW_[keep][o2]
                newk = np.r_[True, ekey[1:] != ekey[:-1]]
                gidx = np.cumsum(newk) - 1
                enum_ = np.bincount(gidx, weights=en)
                eW_ = np.bincount(gidx, weights=ew)
                ea = (ekey[newk] // nm).astype(np.int64); eb = (ekey[newk] % nm).astype(np.int64)
                eaff = enum_ / eW_
            else:
                ea = eb = np.empty(0, np.int64)
                enum_ = eW_ = eaff = np.empty(0, np.float64)
            na = nm
            A = dict(A, atom_of=atom_of, av=av, asize=np.array([len(v) for v in av], np.int64))
            log(f"  [hier] must-link rebuild: {len(ea)} meta edges ({time.time()-t:.0f}s)")

    # ---- agglomeration: merge gain = border-weighted MEAN affinity, every node retained ----
    t = time.time()
    import heapq
    # border affinity POOLS the underlying evidence under a Beta prior (method-of-moments fit over
    # the scene's own per-edge distribution): a one-edge contact shrinks toward the prior while a
    # long, well-observed border is trusted at its full weight.
    mu = float(eaff.mean()); var = float(eaff.var()) + 1e-12
    k0 = max(mu * (1 - mu) / var - 1.0, 1e-3)
    _a0, _b0 = mu * k0, (1 - mu) * k0
    log(f"  [hier] border prior: a0={_a0:.3f} b0={_b0:.3f} (mu={mu:.3f}, k0={k0:.2f})")
    adj = defaultdict(dict)                                       # region -> {sum num, sum W} per nbr
    for i_e in range(len(ea)):
        a, b = int(ea[i_e]), int(eb[i_e])
        e = adj[a].get(b)
        if e is None:
            e = [0.0, 0.0]; adj[a][b] = e; adj[b][a] = e          # shared list (symmetric)
        e[0] += float(enum_[i_e]); e[1] += float(eW_[i_e])
    node_size = list(A["asize"])                                  # voxels per node
    children = {}; parent_of = {}
    nxt = na
    alive = set(range(na))

    def gain(s, c):
        return (_a0 + s) / (_a0 + _b0 + c)

    # Heap discipline for HUB nodes and big scenes:
    #  - small-into-large: a merge folds the SMALLER adjacency dict into the larger one in place,
    #    so total fold work is O(E log V) instead of quadratic when a large region grows edge by
    #    edge. The surviving dict keeps its physical slot; label[] maps slot -> current node id,
    #    so neighbours' back-edges never need re-keying.
    #  - staleness is detected by gain mismatch (edge counts are folded in place, both directions
    #    share one list): a popped entry whose gain no longer matches re-queues at the current
    #    gain, O(1). A dead endpoint resolves to its live successor (lazy union) and re-pushes.
    #  - after a merge only the folded (changed) edges are re-pushed, capped at PUSH_CAP by gain;
    #    untouched edges keep their still-valid heap entries. No candidacy is ever lost.
    PUSH_CAP = 64
    merged_into = {}                                              # dead slot -> surviving slot

    def _resolve(x):
        while x in merged_into:
            x = merged_into[x]
        return x

    label = list(range(na))                                       # physical slot -> current node id
    heap = [(-gain(*adj[a][b]), a, b) for a in adj for b in adj[a] if a < b]
    heapq.heapify(heap)
    log(f"  [hier] agglomerate: start {len(heap)} edges, {len(alive)} nodes")
    _m = 0; _tlast = time.time()
    while heap:
        neg, pa, pb = heapq.heappop(heap)
        _m += 1
        if _m % 50000 == 0 and time.time() - _tlast >= 60:
            _tlast = time.time()
            log(f"  [hier] agglomerate: {len(children)} merges, {_m} pops, heap {len(heap)}, "
                f"{len(alive)} live ({_tlast-t:.0f}s)")
        da = adj.get(pa); e = da.get(pb) if da is not None else None
        if e is None:                                             # dead endpoint: re-push live successor edge
            ra, rb = _resolve(pa), _resolve(pb)
            if ra != rb:
                dra = adj.get(ra); e2 = dra.get(rb) if dra is not None else None
                if e2 is not None:
                    heapq.heappush(heap, (-gain(e2[0], e2[1]), min(ra, rb), max(ra, rb)))
            continue
        g = gain(e[0], e[1])
        if -neg != g:                                             # counts folded since push: re-queue current
            heapq.heappush(heap, (-g, pa, pb))
            continue
        if g <= 0:
            break
        a, b = label[pa], label[pb]
        p = nxt; nxt += 1; children[p] = (a, b); parent_of[a] = p; parent_of[b] = p
        node_size.append(node_size[a] + node_size[b])
        alive.discard(a); alive.discard(b); alive.add(p)
        big, small = (pa, pb) if len(adj[pa]) >= len(adj[pb]) else (pb, pa)
        nbb = adj[big]; nbs = adj.pop(small)
        del nbb[small]
        merged_into[small] = big
        label[big] = p
        changed = []
        for q, e2 in nbs.items():
            if q == big:
                continue
            dq = adj[q]; del dq[small]
            ee = nbb.get(q)
            if ee is None:
                nbb[q] = e2; dq[big] = e2                          # shared list re-pointed, both directions
                changed.append((q, e2))
            else:
                ee[0] += e2[0]; ee[1] += e2[1]                     # in-place fold updates both directions
                changed.append((q, ee))
        if len(changed) > PUSH_CAP:
            changed.sort(key=lambda kv: -gain(kv[1][0], kv[1][1]))
            del changed[PUSH_CAP:]
        for q, ee in changed:
            heapq.heappush(heap, (-gain(ee[0], ee[1]), min(big, q), max(big, q)))
    n_nodes = nxt
    log(f"  [hier] hierarchy: {n_nodes} nodes ({n_nodes-na} merges), {len(alive)} roots "
        f"({time.time()-t:.0f}s)")

    # ---- DFS leaf-ordering (node = contiguous leaf range) + post-order ----
    node_lo = np.zeros(n_nodes, np.int64); node_hi = np.zeros(n_nodes, np.int64)
    leaf_order = np.empty(na, np.int64); pos = [0]
    order_nodes = []
    for root in sorted(alive):
        stack = [(root, False)]
        while stack:
            nd, done = stack.pop()
            if nd < na:
                node_lo[nd] = node_hi[nd] = pos[0]; leaf_order[pos[0]] = nd; pos[0] += 1; order_nodes.append(nd)
                continue
            if done:
                L, R = children[nd]; node_lo[nd] = min(node_lo[L], node_lo[R]); node_hi[nd] = max(node_hi[L], node_hi[R])
                order_nodes.append(nd)
            else:
                stack.append((nd, True)); L, R = children[nd]; stack.append((R, False)); stack.append((L, False))

    return {"na": na, "atom_of": atom_of, "av": av,
            "gm_frame": gm_frame, "gm_size": gm_size,
            "leafI_ptr": leafI_ptr, "leafI_gm": leafI_gm, "leafI_c": leafI_c,
            "leafN_ptr": leafN_ptr, "leafN_f": leafN_f, "leafN_c": leafN_c,
            "children": children, "parent_of": parent_of, "node_size": node_size, "n_nodes": n_nodes,
            "node_lo": node_lo, "node_hi": node_hi, "leaf_order": leaf_order,
            "order_nodes": order_nodes}


def node_atoms(T, nd):
    return T["leaf_order"][T["node_lo"][nd]:T["node_hi"][nd] + 1]


def atoms_vox(T, nd):
    A = node_atoms(T, nd)
    return np.concatenate([T["av"][a] for a in A]) if len(A) else np.empty(0, np.int32)


def verify_stats(T, nvox, vf, log=print):
    """Stage 10.1: build evidence tables. Returns V: bottom-up frame sums (post-order; if freed on
    consumption, Nf kept for the contrastive parent lookup), then per candidate node cache the
    per-ELIGIBLE-FRAME stats of its best-IoU mask (frame, bin, iou, cov, pur) + D (mean over bins of
    bin-max IoU) + the DISAMBIGUATING bin set (parent remainder visible) + the cov_stats rows Stage 10.2
    links on."""
    t = time.time()
    node_size, children, order_nodes = T["node_size"], T["children"], T["order_nodes"]
    gm_frame, gm_size = np.asarray(T["gm_frame"], np.int64), np.asarray(T["gm_size"], np.float64)
    na = T["na"]
    min_vox, e_vox, e_frac = vf["min_vox"], vf["elig_min_vox"], vf["elig_frac"]
    cov_retain = vf["cov_retain"]
    lNp, lNf, lNc = T["leafN_ptr"], np.asarray(T["leafN_f"], np.int32), np.asarray(T["leafN_c"], np.float32)
    lIp, lIg, lIc = T["leafI_ptr"], np.asarray(T["leafI_gm"], np.int32), np.asarray(T["leafI_c"], np.float32)

    def _merge(k1, v1, k2, v2):
        """Sum two sorted (key, count) arrays. Counts are integers in float64, so the sum is exact
        and order-free -- bitwise identical to the Counter it replaces, at a fraction of the RAM."""
        k = np.concatenate([k1, k2]); v = np.concatenate([v1, v2])
        o = np.argsort(k, kind="stable"); k = k[o]; v = v[o]
        new = np.r_[True, k[1:] != k[:-1]]
        return k[new], np.bincount(np.cumsum(new) - 1, weights=v).astype(np.float32)

    Nf_of = {}; If_of = {}
    fr_stats = {}; D_of = {}; vis_bins = {}; cov_stats = {}
    for _vi, nd in enumerate(order_nodes):
        if _vi % 200000 == 0:
            log(f"  [hier] verify {_vi}/{len(order_nodes)} nodes (scored {len(fr_stats)})")
        if nd < na:
            kN, vN = lNf[lNp[nd]:lNp[nd + 1]], lNc[lNp[nd]:lNp[nd + 1]]
            kI, vI = lIg[lIp[nd]:lIp[nd + 1]], lIc[lIp[nd]:lIp[nd + 1]]
        else:
            L, R = children[nd]
            if L not in Nf_of or R not in Nf_of:
                continue
            kN, vN = _merge(*Nf_of[L], *Nf_of[R])
            kI, vI = _merge(*If_of[L], *If_of[R])
            del If_of[L], If_of[R], Nf_of[L], Nf_of[R]
        Nf_of[nd] = (kN, vN); If_of[nd] = (kI, vI)
        Rsz = node_size[nd]
        if Rsz < min_vox or not len(kI):
            continue
        elig_thr = max(e_vox, e_frac * Rsz)
        fI = gm_frame[kI]                             # a mask's frame is always in kN (visibility
        Nhf = vN[np.searchsorted(kN, fI)]             # is a superset of maskedness)
        ok = Nhf >= elig_thr
        if not ok.any():
            continue
        kIe, fIe = kI[ok], fI[ok]
        vIe = vI[ok].astype(np.float64); Ne = Nhf[ok].astype(np.float64)   # exact ints: upcast
        iou = vIe / (gm_size[kIe] + Ne - vIe)
        o = np.lexsort((-iou, fIe))                   # per-frame best; ties -> lowest mask id
        first = np.r_[True, fIe[o][1:] != fIe[o][:-1]]
        sel = o[first]
        arr = np.stack([fIe[sel].astype(np.float64), iou[sel],
                        vIe[sel] / Ne[sel], vIe[sel] / gm_size[kIe[sel]]], 1)
        D_of[nd] = float(arr[:, 1].mean())            # one row per frame -> plain mean
        fr_stats[nd] = arr
        cm = vIe >= cov_retain * Ne                   # mask COVERS enough of the node's visible part
        if cm.any():
            cov_stats[nd] = np.stack([fIe[cm].astype(np.float64), kIe[cm].astype(np.float64),
                                      vIe[cm] / Ne[cm], vIe[cm] / gm_size[kIe[cm]], Ne[cm]], 1)
        # visibility denominator: FRAMES where enough of the node is visible (masked or not).
        vis_bins[nd] = int((vN >= elig_thr).sum())
    log(f"  [hier] verify-stats: {len(fr_stats)} candidate nodes ({time.time()-t:.0f}s)")
    V = {"fr_stats": fr_stats, "D": D_of, "vis_bins": vis_bins, "cov_stats": cov_stats}
    return V


def make_pool(scored, T, dedup_ratio):
    """Stage 9: chain-collapse near-duplicate nested candidates (tree nodes are nested-or-disjoint,
    so all near-duplicates are ancestor chains and nested IoU == size ratio). An epsilon-cover of the
    candidate family at the metric's own top IoU threshold; the representative is the LARGEST
    member -- the dedup victims are fuller extents, and no mask-agreement score can rank extent
    within a >=95%-nested chain."""
    node_size, parent_of = T["node_size"], T["parent_of"]
    scored = set(scored)
    dpar = {nd: nd for nd in scored}

    def dfind(x):
        while dpar[x] != x:
            dpar[x] = dpar[dpar[x]]; x = dpar[x]
        return x
    for nd in scored:
        p = parent_of.get(nd)
        while p is not None and p not in scored:
            p = parent_of.get(p)                                  # nearest SCORED ancestor
        if p is not None and node_size[nd] / node_size[p] >= dedup_ratio:
            ra, rb = dfind(nd), dfind(p)
            if ra != rb:
                dpar[ra] = rb
    grp = defaultdict(list)
    for nd in scored:
        grp[dfind(nd)].append(nd)
    return [max(g, key=lambda n: node_size[n]) for g in grp.values()]


def evidence_link_candidates(V, pool, T, lk, sel, gm_gid, max_vox, mech_sink, members_of,
                             log=print):
    """Stage 10.2: evidence-edge correlation clustering over the pool.

    A part's per-frame BEST mask is its own part mask, so part<->whole relations are invisible to
    best-mask statistics; the signal is a mask that covers a large fraction of SEVERAL nodes' visible
    extents in one frame (V.cov_stats). This stage turns that signal into signed edges and clusters:

      JOIN        in a co-eligible frame (both nodes visible enough to judge), a mask covering both
                  endorses their union, weighted exactly as stage 8 weights any endorsement:
                  cov_u^alpha * pur_u^beta * IoU_u, with the union's stats exact because non-nested
                  tree nodes are disjoint (I and N add). One vote per frame: the frame's best.
      MISS        a co-eligible frame with NO co-covering mask, at weight 1. A part being masked
                  ALONE is not evidence against the union -- the generator masks every granularity
                  by design -- so the negative evidence is missed opportunity, the same denominator
                  the stage-6 affinity uses.

    Per pair: A = join / (join + miss), W = join + miss, cost = W * (logit(clip(A, eps)) - beta) --
    a signed-cost correlation clustering.

    A SECOND, track-level channel adds what no single-frame mechanism can see: CROSS-VIEWPOINT
    identity. A track that covers node i well in some frame and node j well in another PROPOSES
    their union even if no frame ever shows both. It proposes only -- it never contributes to edge
    costs, because there is no honest negative evidence at track level (a part-track staying a
    part-track is the generator's granularity design, not evidence of separateness). The proposed
    pairs enter the candidate family and stage 8's priced objective arbitrates them.
    Every positive edge is emitted as a pair candidate, and greedy additive contraction (GAEC)
    emits every >=3-member contraction on top. Nothing is ranked here: stage 8 decides.

    Data layout is flat numpy row tables + per-node eligibility BITSETS (a scene like mercedes has
    a ~1M-node pool; per-node dicts are gigabytes of object overhead). The pair loop is bounded by
    structure, not by luck: purities of disjoint nodes against ONE mask sum to <= 1, and a pair
    needs (p1+p2)^beta >= w_floor, so each (frame, mask) admits at most a few dozen valid pairs.
    max_crowd caps the per-(frame,mask) inversion."""
    node_lo, node_hi = T["node_lo"], T["node_hi"]
    nn = T["n_nodes"]; nsz = T["node_size"]
    alpha, beta_w, w_floor = float(sel["alpha"]), float(sel["beta"]), float(sel["w_floor"])
    max_crowd = int(lk["max_crowd"])
    t0 = time.time()

    def nested(a, b):
        return (node_lo[a] <= node_lo[b] and node_hi[b] <= node_hi[a]) or \
               (node_lo[b] <= node_lo[a] and node_hi[a] <= node_hi[b])

    # ---- flat row table of cov_stats over pool nodes + eligibility bitsets ----
    pool = [nd for nd in pool]
    r_nd, r_blk = [], []
    nfmax = 0
    elig_list = {}
    for nd in pool:
        arr = V["fr_stats"].get(nd)
        if arr is not None and len(arr):
            ef = np.unique(arr[:, 0]).astype(np.int64)
            elig_list[nd] = ef
            if len(ef):
                nfmax = max(nfmax, int(ef[-1]))
        cs = V["cov_stats"].get(nd)
        if cs is None:
            continue
        r_nd.append(np.full(len(cs), nd, np.int64)); r_blk.append(cs)
    if not r_blk:
        log("  [link] no covering-mask rows in the pool"); return []
    r_nd = np.concatenate(r_nd)
    R = np.concatenate(r_blk)                        # columns: f, gm, cov, pur, Nhf
    r_f = R[:, 0].astype(np.int64); r_gm = R[:, 1].astype(np.int64)
    r_I = (R[:, 2] * R[:, 4]).astype(np.float64); r_pur = R[:, 3]; r_N = R[:, 4]
    del r_blk, R
    words = (nfmax >> 6) + 1
    E = {}                                           # nd -> packed eligibility bitset (uint64)
    for nd, ef in elig_list.items():
        b = np.zeros(words, np.uint64)
        np.bitwise_or.at(b, ef >> 6, np.uint64(1) << (ef & 63).astype(np.uint64))
        E[nd] = b
    elig_list.clear()
    log(f"  [link] rows: {len(r_nd)} covering-mask rows over {len(pool)} pool nodes, "
        f"{words * 8}B/bitset ({time.time()-t0:.0f}s)")

    # ---- join votes, grouped by (frame, mask) ----
    key = r_f * (r_gm.max() + 1) + r_gm
    order = np.argsort(key, kind="stable")
    kb = key[order]
    starts = np.flatnonzero(np.r_[True, kb[1:] != kb[:-1]])
    ends = np.r_[starts[1:], len(kb)]
    pu_min = w_floor ** (1.0 / beta_w)               # a pair needs p1+p2 >= this (cu, iou <= 1)
    pa, pb, pf, pw = [], [], [], []                  # pair vote rows
    _tl = time.time(); ngrp = 0
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e - s < 2:
            continue
        ngrp += 1
        if time.time() - _tl > 60:
            _tl = time.time()
            log(f"  [link] join scan {ngrp} multi-node (frame,mask) groups, "
                f"{len(pa)} pair votes ({time.time()-t0:.0f}s)")
        idx = order[s:e]
        pur = r_pur[idx]
        o2 = np.argsort(-pur, kind="stable")[:max_crowd]
        idx = idx[o2]; pur = pur[o2]
        nds = r_nd[idx]; I = r_I[idx]; N = r_N[idx]
        f = int(r_f[idx[0]])
        g = len(idx)
        for i2 in range(g - 1):
            if pur[i2] + pur[i2 + 1] < pu_min:
                break                                # sorted desc: no remaining pair can clear it
            for j2 in range(i2 + 1, g):
                puu = pur[i2] + pur[j2]
                if puu < pu_min:
                    break
                a, b = int(nds[i2]), int(nds[j2])
                if nested(a, b):
                    continue
                cu = (I[i2] + I[j2]) / (N[i2] + N[j2])
                w = cu ** alpha * min(puu, 1.0) ** beta_w
                if w < w_floor:
                    continue
                gsz = I[i2] / max(pur[i2], 1e-9)     # |q_gm|, identical from either member
                iou_u = (I[i2] + I[j2]) / (gsz + (N[i2] + N[j2]) - (I[i2] + I[j2]))
                if a > b:
                    a, b = b, a
                pa.append(a); pb.append(b); pf.append(f); pw.append(w * iou_u)
    n_occ = len(starts)
    del order, kb, starts, ends
    log(f"  [link] {n_occ} (frame,mask) occurrences -> {len(pa)} frame pair votes "
        f"({time.time()-t0:.0f}s)")

    # ---- aggregate: per (pair, frame) keep the best vote, then per pair sum + frame count ----
    POP = np.array([bin(x).count("1") for x in range(256)], np.uint8)
    edge_cost = {}
    if pa:
        pa = np.asarray(pa, np.int64); pb = np.asarray(pb, np.int64)
        pf = np.asarray(pf, np.int64); pw = np.asarray(pw, np.float64)
        pk = pa * (nn + 1) + pb
        o = np.lexsort((-pw, pf, pk))                # per (pair, frame): best vote first
        pk, pf, pw, pa, pb = pk[o], pf[o], pw[o], pa[o], pb[o]
        first = np.r_[True, (pk[1:] != pk[:-1]) | (pf[1:] != pf[:-1])]
        pk, pw, pa, pb = pk[first], pw[first], pa[first], pb[first]
        pstart = np.flatnonzero(np.r_[True, pk[1:] != pk[:-1]])
        pend = np.r_[pstart[1:], len(pk)]
        ua, ub = pa[pstart], pb[pstart]
        jw = np.add.reduceat(pw, pstart)
        njf = (pend - pstart)
        log(f"  [link] {len(ua)} frame-channel pairs ({time.time()-t0:.0f}s)")
        # co-eligible frame counts via bitset intersections
        n_co = np.empty(len(ua), np.int64)
        _tl = time.time()
        for i2 in range(len(ua)):
            if time.time() - _tl > 60:
                _tl = time.time()
                log(f"  [link] co-eligibility {i2}/{len(ua)} ({time.time()-t0:.0f}s)")
            ea, eb = E.get(int(ua[i2])), E.get(int(ub[i2]))
            n_co[i2] = int(POP[(ea & eb).view(np.uint8)].sum()) \
                if ea is not None and eb is not None else 0
        miss = np.maximum(n_co - njf, 0)
        # edge cost = the calibrated same-vs-different log-likelihood ratio: a two-component
        # binomial mixture over the pairs' own (joins, opportunities), fit by EM -- no labels, no
        # hand prior; the mixture estimates the scene's join rates for one-object and two-object
        # pairs and every pair is scored under both.
        k_, n_ = jw, jw + miss
        p_s, p_d, pi_s = 0.5, 0.02, 0.1
        for _ in range(50):
            ls = np.log(pi_s) + k_ * np.log(p_s) + (n_ - k_) * np.log(1 - p_s)
            ld = np.log(1 - pi_s) + k_ * np.log(p_d) + (n_ - k_) * np.log(1 - p_d)
            g_ = 1.0 / (1.0 + np.exp(np.clip(ld - ls, -50, 50)))
            pi_s = float(g_.mean())
            p_s = float(np.clip((g_ * k_).sum() / max((g_ * n_).sum(), 1e-9), 1e-3, 1 - 1e-3))
            p_d = float(np.clip(((1 - g_) * k_).sum() / max(((1 - g_) * n_).sum(), 1e-9),
                                1e-4, 1 - 1e-4))
        if p_s < p_d:
            p_s, p_d = p_d, p_s
        log(f"  [link] join LLR: p_same={p_s:.3f} p_diff={p_d:.4f} pi={pi_s:.3f}")
        cf = k_ * np.log(p_s / p_d) + (n_ - k_) * np.log((1 - p_s) / (1 - p_d))
        for i2 in range(len(ua)):
            edge_cost[(int(ua[i2]), int(ub[i2]))] = float(cf[i2])

    # ---- TRACK channel: cross-viewpoint union PROPOSALS (no cost contribution) ----
    gm_gid = np.asarray(gm_gid, np.int64)
    # side strength s(node, track) = best graded endorsement of the node by any of the track's masks
    s_gid = gm_gid[r_gm]
    val = (r_I / r_N) ** alpha * np.minimum(r_pur, 1.0) ** beta_w \
        * (r_I / np.maximum(r_I / np.maximum(r_pur, 1e-9) + r_N - r_I, 1e-9))
    okg = (s_gid >= 0) & (val >= w_floor)
    sn, sg, sv = r_nd[okg], s_gid[okg], val[okg]
    del r_nd, r_f, r_gm, r_I, r_pur, r_N, s_gid, val, okg
    o = np.lexsort((-sv, sn, sg))                    # per (track, node): best strength first
    sn, sg, sv = sn[o], sg[o], sv[o]
    if len(sn):
        first = np.r_[True, (sg[1:] != sg[:-1]) | (sn[1:] != sn[:-1])]
        sn, sg, sv = sn[first], sg[first], sv[first]
    gstart = np.flatnonzero(np.r_[True, sg[1:] != sg[:-1]])
    gend = np.r_[gstart[1:], len(sg)]
    tprop = set()
    _tl = time.time()
    for gi, ge in zip(gstart.tolist(), gend.tolist()):
        if time.time() - _tl > 60:
            _tl = time.time()
            log(f"  [link] track proposals {gi}/{len(sn)} rows ({time.time()-t0:.0f}s)")
        nds2 = sn[gi:ge]
        if len(nds2) < 2:
            continue
        cap = min(len(nds2), max_crowd)              # strength-sorted within the track already
        for i2 in range(cap - 1):
            for j2 in range(i2 + 1, cap):
                a, b = int(nds2[i2]), int(nds2[j2])
                if nested(a, b):
                    continue
                tprop.add((min(a, b), max(a, b)))
    tprop -= set(edge_cost)                          # frame-channel pairs already carry their cost
    log(f"  [link] track channel: +{len(tprop)} cross-viewpoint pair proposals ({time.time()-t0:.0f}s)")
    E.clear()

    ua = np.array([k[0] for k in edge_cost], np.int64)
    ub = np.array([k[1] for k in edge_cost], np.int64)
    cost = np.array(list(edge_cost.values()), np.float64)
    npos = int((cost > 0).sum())
    log(f"  [link] {len(ua)} evidence edges ({npos} positive) ({time.time()-t0:.0f}s)")

    out = []; sid = nn + 1000000
    # every positive edge is itself a candidate: stage 8 arbitrates instead of GAEC's merge order
    # deciding which pairs ever exist. Track proposals join them (candidates only, no cost).
    for a, b in ([(int(ua[i2]), int(ub[i2])) for i2 in np.flatnonzero(cost > 0).tolist()]
                 + sorted(tprop)):
        if int(nsz[a]) + int(nsz[b]) > max_vox:
            continue
        ats = np.concatenate([node_atoms(T, a), node_atoms(T, b)])
        out.append((sid, ats, int(nsz[a] + nsz[b])))
        mech_sink[sid] = "link"; members_of[sid] = (a, b)
        sid += 1
    log(f"  [link] pair emission: +{len(out)} (positive edges + track proposals) ({time.time()-t0:.0f}s)")

    # ---- GAEC: greedy additive contraction; every >=3-member contraction is a candidate ----
    from heapq import heapify, heappop, heappush
    comp = {}; nbr = defaultdict(dict)
    for i2 in range(len(ua)):
        a, b, c = int(ua[i2]), int(ub[i2]), float(cost[i2])
        comp[a] = (a,); comp[b] = (b,)
        nbr[a][b] = c; nbr[b][a] = c
    h = [(-float(cost[i2]), int(ua[i2]), int(ub[i2])) for i2 in np.flatnonzero(cost > 0).tolist()]
    heapify(h)
    nmax = 0; n0 = len(out); _tl = time.time()
    while h:
        negc, a, b = heappop(h)
        if time.time() - _tl > 60:
            _tl = time.time()
            log(f"  [link] GAEC {len(out) - n0} contractions ({time.time()-t0:.0f}s)")
        if a not in comp or b not in comp:           # a contracted-away root: stale
            continue
        c = nbr[a].get(b)
        if c is None or c <= 0 or abs(-negc - c) > 1e-9:
            continue                                 # cost changed since push: stale
        if any(nested(x, y) for x in comp[a] for y in comp[b]) \
                or sum(int(nsz[m]) for m in comp[a] + comp[b]) > max_vox:
            del nbr[a][b], nbr[b][a]                 # never mergeable: retire the edge
            continue
        if len(comp[b]) > len(comp[a]):
            a, b = b, a                              # contract the smaller into the larger
        comp[a] = comp[a] + comp[b]
        del nbr[a][b]
        for y, cc in nbr.pop(b).items():
            if y == a:
                continue
            c2 = nbr[a].get(y, 0.0) + cc
            nbr[a][y] = c2; nbr[y][a] = c2
            nbr[y].pop(b, None)
            if c2 > 0:
                heappush(h, (-c2, a, y))
        del comp[b]
        mem = comp[a]
        nmax = max(nmax, len(mem))
        if len(mem) > 2:                             # 2-member contractions = the edge, emitted above
            ats = np.concatenate([node_atoms(T, m) for m in mem])
            out.append((sid, ats, int(sum(nsz[m] for m in mem))))
            mech_sink[sid] = "link"; members_of[sid] = tuple(mem)
            sid += 1
    log(f"  [link] GAEC: {len(out) - n0} contraction candidates (largest {nmax} members) "
        f"({time.time()-t0:.0f}s)")
    return out


def compact_output(T, phi_by_node, phi_of_tree_nodes, vox_by_k, selected_by_k, KS, log=print):
    """Stage 11: zero-loss ancestor replacement, per budget K: the exchange move.

    Where >=2 surviving hypotheses sit inside one tree ancestor,
    swap the group for the ancestor iff the evidence cover does NOT drop and the per-atom <=K cap
    still holds. Strictly zero-loss: a swap that would lose any mask's best phi is rejected.
    Iterated to a fixed point (<=3 passes). A non-laminar hypothesis has no parent, so its chain
    starts at the tightest tree node spanning its atom block.

    This is the exchange move on stage 11's greedy cover: greedy can fragment an object across
    children when the parent alone would cover the same evidence. Also returns the per-hypothesis
    `phi_by_node` carries per-mask evidence for selected candidates; `phi_of_tree_nodes` scores
    ancestors that the replacement pass needs and selection never evaluated."""
    from collections import defaultdict as _dd
    na = T["na"]; nn = T["n_nodes"]
    atom_of = T["atom_of"]; leaf_order = T["leaf_order"]
    node_lo, node_hi, node_size = T["node_lo"], T["node_hi"], T["node_size"]
    nvox = len(atom_of)
    par = np.full(nn, -1, np.int64)
    for _p in range(na, nn):
        _L, _R = T["children"][_p]
        par[_L] = _p; par[_R] = _p
    posmap = np.empty(na, np.int64); posmap[leaf_order] = np.arange(na)
    _cc = {}

    def _contain(pl, ph):
        """Smallest tree node whose atom block spans [pl, ph] -- the tightest laminar container of a
        non-laminar hypothesis, which is where its ancestor chain has to start."""
        key = (pl, ph)
        nd = _cc.get(key)
        if nd is None:
            nd = int(leaf_order[pl])
            while nd >= 0 and not (node_lo[nd] <= pl and node_hi[nd] >= ph):
                nd = int(par[nd])
            _cc[key] = nd
        return nd

    def _targets(S):
        """Ancestors containing at least two current hypotheses; other swaps are impossible."""
        anc = set()
        for h in S:
            p = int(par[h["id"]]) if h["id"] < nn else _contain(h["pl"], h["ph"])
            while p >= 0:
                anc.add(p)
                p = int(par[p])
        if not anc:
            return set()
        pl = np.fromiter((h["pl"] for h in S), np.int64, len(S))
        ph = np.fromiter((h["ph"] for h in S), np.int64, len(S))
        order = np.argsort(pl, kind="stable")
        pls, phs = pl[order], ph[order]
        targets = set()
        for p in anc:
            start = int(np.searchsorted(pls, node_lo[p]))
            if np.count_nonzero(phs[start:] <= node_hi[p]) >= 2:
                targets.add(p)
        return targets

    prepared = {}
    all_targets = set()
    for k in KS:
        hyps = []
        for (nd, vox) in selected_by_k[k]:
            nd = int(nd)
            A = np.unique(atom_of[vox])
            hyps.append({"id": nd, "A": A, "vox": vox, "phi": phi_by_node[nd],
                         "synth": nd >= nn, "pl": int(posmap[A].min()),
                         "ph": int(posmap[A].max())})
        prepared[k] = hyps
        all_targets.update(_targets(hyps))

    ancestor_phi = {node: phi_by_node[node] for node in all_targets if node in phi_by_node}
    ancestor_phi.update(phi_of_tree_nodes(all_targets - ancestor_phi.keys()))

    vb2, sb2 = {}, {}
    stats = {}
    for k in KS:
        hyps = prepared[k]
        S = list(hyps)                          # lam-priced selection already did the filtering
        # ---- zero-loss ancestor replacement (exchange move on the selected cover) ----
        cov = np.zeros(na, np.int32)
        for h in S:
            cov[h["A"]] += 1
        n_rep = 0
        b1v = {}; b1o = {}; b2v = {}; b2o = {}

        def _top2(gs, inv_):
            """(best, owner, second-best, owner) per mask over the CURRENT S, so a swap's loss can be
            evaluated without rescanning every hypothesis per mask."""
            for g in gs:
                v1 = v2 = 0.0; o1 = o2 = None
                for h2 in inv_.get(g, ()):
                    v = h2["phi"].get(g, 0.0)
                    if v > v1:
                        v2, o2 = v1, o1; v1, o1 = v, id(h2)
                    elif v > v2:
                        v2, o2 = v, id(h2)
                b1v[g] = v1; b1o[g] = o1; b2v[g] = v2; b2o[g] = o2

        for _pass in range(3):
            inv = _dd(list)
            for h in S:
                for g in h["phi"]:
                    inv[g].append(h)
            b1v.clear(); b1o.clear(); b2v.clear(); b2o.clear()
            _top2(inv.keys(), inv)
            changed = False; _dirty = True
            _anc_sorted = sorted(_targets(S), key=lambda x: -node_size[x])
            _t_rep = time.time(); _t_log = _t_rep
            for _ai, p in enumerate(_anc_sorted):
                if time.time() - _t_log > 60:                  # phi gathers make this the silent block
                    _t_log = time.time()
                    log(f"  [compact] K{k} replace pass {_pass+1}: {_ai}/{len(_anc_sorted)} ancestors, "
                        f"{n_rep} folded ({time.time()-_t_rep:.0f}s)")
                lo, hi = node_lo[p], node_hi[p]
                if _dirty:                                    # the pl-sorted index, rebuilt on change
                    _pl = np.fromiter((h["pl"] for h in S), np.int64, len(S))
                    _ph = np.fromiter((h["ph"] for h in S), np.int64, len(S))
                    _ord = np.argsort(_pl, kind="stable")
                    _pls, _phs = _pl[_ord], _ph[_ord]
                    _dirty = False
                _i0 = int(np.searchsorted(_pls, lo))
                _cand = _ord[_i0:][_phs[_i0:] <= hi]
                if len(_cand) < 2:
                    continue
                Dp = [S[i] for i in _cand.tolist()]
                pA = leaf_order[lo:hi + 1]
                loc = np.zeros(hi - lo + 1, np.int32)         # local counts, no n_atoms alloc
                for h in Dp:
                    loc[posmap[h["A"]] - lo] += 1
                if (cov[pA] - loc + 1 > k).any():             # the swap would breach the K cap
                    continue
                php = ancestor_phi[p]
                dset = {id(h) for h in Dp}
                affected = set(php)
                for h in Dp:
                    affected.update(h["phi"])
                loss = 0.0
                for g in affected:
                    old = b1v.get(g, 0.0); o1 = b1o.get(g)
                    if o1 is None or o1 not in dset:
                        surv = old
                    elif b2o.get(g) is None or b2o[g] not in dset:
                        surv = b2v.get(g, 0.0)
                    else:                                     # rare: both leaders are being replaced
                        surv = max((h2["phi"].get(g, 0.0) for h2 in inv.get(g, [])
                                    if id(h2) not in dset), default=0.0)
                    loss += old - max(surv, php.get(g, 0.0))          # every mask counts once
                if loss > 1e-12:                              # zero-loss only
                    continue
                newh = {"id": p, "A": pA, "vox": None, "phi": php,
                        "synth": False, "pl": int(lo), "ph": int(hi)}
                S = [h for h in S if id(h) not in dset] + [newh]
                cov[pA] = cov[pA] - loc + 1
                _dirty = True
                for g in php:
                    inv[g].append(newh)
                for h in Dp:
                    for g in h["phi"]:
                        inv[g] = [h2 for h2 in inv[g] if id(h2) not in dset]
                _top2(affected, inv)                          # refresh only the touched masks
                n_rep += len(Dp) - 1
                changed = True
            if not changed:
                break
        av = T["av"]
        for h in S:                                           # replacements carry no voxels yet
            if h["vox"] is None:
                h["vox"] = np.unique(np.concatenate([av[a] for a in h["A"]])) if len(h["A"]) \
                    else np.zeros(0, np.int32)
        vb2[k] = [h["vox"] for h in S]
        sb2[k] = [(h["id"], v) for h, v in zip(S, vb2[k])]
        stats[k] = {"n0": len(hyps), "n": len(S), "replaced": n_rep}
    log(f"  [compact] K counts {[stats[k]['n0'] for k in KS]} -> {[stats[k]['n'] for k in KS]} "
        f"(replace {[stats[k]['replaced'] for k in KS]})")
    return vb2, sb2, stats


def cover_select(flat, T, tables, KS, vf, sel, gm_gid, members_of, log=print):
    """Stage 11: lambda-priced greedy max-weighted evidence-cover selection of the objective

        F(S) = sum_e w_e * max_{h in S} phi_h(e)  -  lam * |S|,   s.t. per-atom <= K

    maximised by lazy greedy (submodular, so cached gains only shrink), Q as tie-break, stopping
    when the marginal gain falls below `lam` -- the price of emitting a hypothesis. There is no
    order beyond the objective: what greedy declines to pay for is not emitted.

    phi_h(e) is GRADED: each mask endorses h with weight cov^alpha * pur^beta (a Beta-likelihood
    reading of truncation and leak noise) times its IoU with h -- no hard tau gates. A client's
    value is the mean over its track's masks in h's ELIGIBLE frames, so a frame that sees h but has
    no endorsing mask pulls the mean down (the miss penalty).

    Clients are global tracks, not per-frame masks: the many frames of one track would otherwise
    act as that many independent clients, so a long-lived track would outweigh a singleton object.
    Masks with no global id stay singleton clients."""
    from heapq import heapify, heappop, heappush
    _rg = _ranges
    alpha, beta = float(sel["alpha"]), float(sel["beta"])
    w_floor, lam = float(sel["w_floor"]), float(sel["lam"])
    e_vox, e_frac = vf["elig_min_vox"], vf["elig_frac"]
    tb = tables
    gm_frame = tb["gm_frame"]
    na = T["na"]; nn = T["n_nodes"]
    node_lo, node_hi, leaf_order = T["node_lo"], T["node_hi"], T["leaf_order"]
    atom_of = T["atom_of"]
    ngm, nf = len(gm_frame), int(gm_frame.max()) + 1
    offI, gI, vI = tb["rI"]                                    # CSR, already atom-major
    offN, fN, vN = tb["rN"]
    gm_vsz = np.bincount(gI, weights=vI, minlength=ngm)
    _key2cid = {}
    cid_of = np.empty(ngm, np.int64)
    for g in range(ngm):
        k = int(gm_gid[g]) if gm_gid[g] >= 0 else -1 - g
        c = _key2cid.get(k)
        if c is None:
            c = len(_key2cid); _key2cid[k] = c
        cid_of[g] = c
    ngm_eff = len(_key2cid)
    w_cid = np.ones(ngm_eff, np.float64)                       # every client counts once
    log(f"  [cover] clients: {ngm} masks -> {ngm_eff} track clients")
    # per-client SORTED frame array, for the miss-penalty denominator (built once)
    _ord_c = np.argsort(cid_of, kind="stable")
    _cnt_c = np.bincount(cid_of, minlength=ngm_eff)
    _off_c = np.r_[0, np.cumsum(_cnt_c)]
    cli_frames = [np.sort(gm_frame[_ord_c[_off_c[c]:_off_c[c + 1]]]) for c in range(ngm_eff)]

    def evidence_of(A):
        """Gather one candidate's evidence from the leaf tables: (masks, intersection counts) and
        (frames, visible counts), both sparse and ascending. This is the expensive part -- the row
        gather is proportional to every leaf row under the candidate's atoms."""
        ii = _rg(offI[A], offI[A + 1] - offI[A])
        if not len(ii):
            return None
        ni = _rg(offN[A], offN[A + 1] - offN[A])
        Nf = np.bincount(fN[ni], weights=vN[ni], minlength=nf)
        Isum = np.bincount(gI[ii], weights=vI[ii], minlength=ngm)
        gms = np.nonzero(Isum)[0]
        frs = np.nonzero(Nf)[0]
        return gms, Isum[gms], frs, Nf[frs]

    def phi_from_evidence(ev, sz):
        """GRADED endorsement over one candidate's evidence -> ({mask: weighted IoU}, elig frames).
        weight = cov^alpha * pur^beta; only eligibility (can this frame judge the candidate at all)
        and the numeric w_floor cut anything."""
        if ev is None:
            return {}, np.empty(0, np.int64)
        gms, I2, frs, Nv = ev
        thr = max(float(e_vox), e_frac * sz)
        elig = frs[Nv >= thr].astype(np.int64)
        NfI = Nv[np.searchsorted(frs, gm_frame[gms])]
        ok = NfI >= thr
        gms, NfI, I2 = gms[ok], NfI[ok], I2[ok]
        w = (I2 / NfI) ** alpha * (I2 / gm_vsz[gms]) ** beta
        m = w >= w_floor
        gms, I2, NfI, w = gms[m], I2[m], NfI[m], w[m]
        return dict(zip(gms.tolist(), (w * I2 / (gm_vsz[gms] + NfI - I2)).tolist())), elig

    _ev_cache = {}

    def evidence_of_node(nd):
        """Evidence for a POOL NODE, cached -- link members are pool nodes and each is a member of
        many unions, so this is gathered once and reused."""
        ev = _ev_cache.get(nd)
        if ev is None:
            ev = evidence_of(leaf_order[node_lo[nd]:node_hi[nd] + 1])
            _ev_cache[nd] = ev
        return ev

    _pn_cache = {}

    def phi_of_tree_nodes(nodes):
        """Score exchange ancestors from one bottom-up sparse evidence pass."""
        nodes = [int(node) for node in nodes]
        missing = [node for node in nodes if node not in _pn_cache]
        for node, evidence in _tree_evidence(T, tb, missing):
            _pn_cache[node] = phi_from_evidence(evidence, int(T["node_size"][node]))[0]
        return {node: _pn_cache[node] for node in nodes}

    def evidence_of_union(members):
        """Evidence for a union, from its members' cached evidence. Members are mutually non-nested
        tree nodes, so their atom sets -- and therefore their voxel sets -- are DISJOINT: per-mask
        intersections and per-frame visible counts simply ADD. Summing integer counts in float64 is
        exact and order-independent, so this is bitwise identical to gathering the union's own rows,
        at a fraction of the cost (the gather is ~31x larger than the sparse vectors it produces)."""
        evs = [e for e in (evidence_of_node(m) for m in members) if e is not None]
        if not evs:
            return None
        if len(evs) == 1:
            return evs[0]
        g = np.concatenate([e[0] for e in evs]); gi = np.concatenate([e[1] for e in evs])
        f = np.concatenate([e[2] for e in evs]); fi = np.concatenate([e[3] for e in evs])
        gu, ginv = np.unique(g, return_inverse=True)            # ascending, like np.nonzero
        fu, finv = np.unique(f, return_inverse=True)
        return gu, np.bincount(ginv, weights=gi), fu, np.bincount(finv, weights=fi)

    def to_clients(phi, elig):
        """Per-mask phi -> per-client mean over the track's masks in the candidate's ELIGIBLE
        frames. An eligible appearance with no endorsement contributes 0 to the sum but 1 to the
        denominator -- the miss penalty. Masks arrive in ascending id order, fixing the float
        summation order."""
        agg = {}                                               # client -> phi sum
        for g, v in phi.items():
            c = int(cid_of[g])
            agg[c] = agg.get(c, 0.0) + v
        out = {}
        for c, sv in agg.items():
            fr2 = cli_frames[c]
            den = int(np.searchsorted(fr2, elig, "right").sum()
                      - np.searchsorted(fr2, elig, "left").sum())
            out[c] = sv / max(den, 1)
        return out

    t0 = time.time()
    phis = []                                                  # client-aggregated, for the greedy
    phi_mask = []                                              # per-mask, forwarded to replacement
    for _ci, (nd, ats, vsz) in enumerate(flat):
        if nd < nn:                                            # tree node
            _ev = evidence_of_node(nd)
        elif nd in members_of:                                 # link union: reuse its members
            _ev = evidence_of_union(members_of[nd])
        else:                                                  # arbitrary atom set (defensive)
            _ev = evidence_of(np.sort(ats))
        _pm, _el = phi_from_evidence(_ev, vsz)
        phi_mask.append(_pm); phis.append(to_clients(_pm, _el))
        if _ci and _ci % 20000 == 0:
            log(f"  [cover] phi {_ci}/{len(flat)} ({time.time()-t0:.0f}s)")
    log(f"  [cover] phi for {len(flat)} candidates ({time.time()-t0:.0f}s, "
        f"evidence cached for {len(_ev_cache)} nodes)")
    _ev_cache.clear()

    best = np.zeros(ngm_eff)                                   # current best phi per CLIENT over sel
    # coverage is counted per ATOM, not per voxel: every hypothesis is a union of whole atoms, so
    # all voxels of an atom carry the same count and the <=K predicate is identical -- over an array
    # 2-4x smaller, and without materialising a voxel array per candidate.
    cov = np.zeros(T["na"], np.int32); used = set()
    sel_ats = []; sel_nd = []
    cover_ids = set()                                          # phase-1 (cover-pick) node ids
    vox_by_k = {}; selected_by_k = {}; _vcache = []; av = T["av"]

    _w = w_cid
    def gain(i):
        return sum(_w[g] * (v - best[g]) for g, v in phis[i].items() if v > best[g])

    for K in KS:
        h = [(-gain(i), i) for i in range(len(flat)) if flat[i][0] not in used and phis[i]]
        heapify(h)
        npick = 0
        while h:
            negg, i = heappop(h)
            if -negg < lam:                                    # marginal gain below the price: stop
                break
            g = gain(i)                                        # stale? gains only shrink
            if h and g < -h[0][0] - 1e-12:
                if g >= lam:
                    heappush(h, (-g, i))
                continue
            nd, ats, vsz = flat[i]
            if nd in used or not vsz or (cov[ats] >= K).any():
                continue                                       # inadmissible at this K; retry next K
            cov[ats] += 1; used.add(nd); cover_ids.add(nd)
            sel_ats.append(ats); sel_nd.append(nd); npick += 1
            for gm2, v in phis[i].items():
                if v > best[gm2]:
                    best[gm2] = v
        F = float((w_cid * best).sum())
        log(f"  [cover] K={K}: +{npick} picks (stop: gain < lam={lam}) -> {len(sel_ats)} total | "
            f"F={F:.1f} F-lam|S|={F - lam * len(sel_ats):.1f}")
        # materialise voxels ONCE per selected hypothesis (cached across K, since selection is
        # monotone: the K-1 picks are a prefix of the K picks)
        for j in range(len(_vcache), len(sel_ats)):
            _vcache.append(np.unique(np.concatenate([av[a] for a in sel_ats[j]]))
                           if len(sel_ats[j]) else np.empty(0, np.int32))
        vox_by_k[K] = list(_vcache)
        selected_by_k[K] = [(sel_nd[j], _vcache[j]) for j in range(len(sel_ats))]
    phi_by_node = {flat[i][0]: phi_mask[i] for i in range(len(flat)) if flat[i][0] in used}
    return vox_by_k, selected_by_k, cover_ids, phi_by_node, phi_of_tree_nodes



def _config_for_core(config):
    """Translate the public flat config into the source algorithm's internal groups."""
    return {
        "mustlink": {
            "min_aff": float(config["mustlink_min_affinity"]),
            "min_w": int(config["mustlink_min_observations"]),
        },
        "verify": {
            "min_vox": int(config["candidate_min_voxels"]),
            "elig_min_vox": int(config["eligible_min_voxels"]),
            "elig_frac": float(config["eligible_fraction"]),
            "cov_retain": float(config["retained_coverage"]),
        },
        "dedup": {"ratio": float(config["dedup_ratio"])},
        "link": {"max_crowd": int(config["max_mask_crowd"])},
        "select": {
            "alpha": float(config["coverage_alpha"]),
            "beta": float(config["purity_beta"]),
            "w_floor": float(config["endorsement_floor"]),
            "lam": float(config["hypothesis_price"]),
        },
    }


def associate(atom_graph, evidence, point_count, config, atom_seconds=0.0, log=print):
    """Run hierarchy, evidence association, and fixed-K selection on prebuilt lifted evidence."""
    started = time.time()
    cfg = _config_for_core(config)
    budget = int(config["membership_budget"])
    timings = {"atoms_s": round(atom_seconds, 3)}

    tick = time.time()
    tree = build_tree(evidence, atom_graph, mustlink=cfg["mustlink"], log=log)
    timings["hierarchy_s"] = round(time.time() - tick, 3)

    tick = time.time()
    verified = verify_stats(tree, point_count, cfg["verify"], log=log)
    timings["evidence_s"] = round(time.time() - tick, 3)

    pool = make_pool(verified["fr_stats"].keys(), tree, cfg["dedup"]["ratio"])
    node_size = tree["node_size"]
    node_lo, node_hi, leaf_order = tree["node_lo"], tree["node_hi"], tree["leaf_order"]
    candidates = [
        (node, leaf_order[node_lo[node] : node_hi[node] + 1], int(node_size[node]))
        for node in pool
        if node_size[node] >= cfg["verify"]["min_vox"]
    ]
    global_track_ids = np.asarray(evidence["global_track_ids"], np.int64)
    if len(global_track_ids) != len(tree["gm_frame"]):
        raise ValueError("Global track IDs do not match the hierarchy mask enumeration")

    tick = time.time()
    mechanisms = {}
    members = {}
    candidates += evidence_link_candidates(
        verified,
        pool,
        tree,
        cfg["link"],
        cfg["select"],
        global_track_ids,
        point_count,
        mechanisms,
        members,
        log=log,
    )
    candidates.sort(key=lambda item: -item[2])
    tables = build_tables(tree, log=log)
    voxels_by_k, selected_by_k, _, phi_by_node, phi_for_tree_node = cover_select(
        candidates,
        tree,
        tables,
        [budget],
        cfg["verify"],
        cfg["select"],
        global_track_ids,
        members,
        log=log,
    )
    voxels_by_k, _, compact_stats = compact_output(
        tree,
        phi_by_node,
        phi_for_tree_node,
        voxels_by_k,
        selected_by_k,
        [budget],
        log=log,
    )
    timings["selection_s"] = round(time.time() - tick, 3)
    timings["total_s"] = round(atom_seconds + time.time() - started, 3)

    hypotheses = [np.asarray(voxels, np.int32) for voxels in voxels_by_k[budget]]
    return {
        "hypotheses": hypotheses,
        "timings": timings,
        "counts": {
            "atoms": int(tree["na"]),
            "nodes": int(tree["n_nodes"]),
            "evidence_candidates": int(len(verified["fr_stats"])),
            "pool": int(len(pool)),
            "selection_candidates": int(len(candidates)),
            "hypotheses": int(len(hypotheses)),
            "exchange_replacements": int(compact_stats[budget]["replaced"]),
        },
    }
