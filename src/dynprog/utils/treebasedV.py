import numpy as np

def _outer_nd(vectors):
    """
    N-ary outer product: given a list of 1D arrays [v0, v1, ..., v{D-1}],
    return an array with shape (len(v0), len(v1), ..., len(v{D-1}))
    whose entries are the product of all coordinates.
    """
    # Start with scalar 1.0 and expand dimension by dimension
    out = np.array(1.0, dtype=float)
    for v in vectors:
        out = np.multiply.outer(out, np.asarray(v, dtype=float))
    return out

def _fmt_bytes(n):
    for u in ("B","KiB","MiB","GiB","TiB"):
        if n < 1024: return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PiB"

def compute_tv_from_tree(G, DFA, L, *, max_elements=50_000_000, max_bytes=None):
    """
    Build satProb 'tv' from the DFA tree and per-letter labeling L for ANY number of dimensions.

    Inputs:
      - G.V[d] has shape (num_nodes, N_d) for each dimension d
      - L[d] has shape (n_letters, N_d) for each dimension d
      - DFA.S, DFA.F, DFA.S0, DFA.trans, DFA.act as usual

    Returns:
      - tv: an N-dimensional ndarray with shape (N_0, N_1, ..., N_{D-1})
      - node_outer_max: dict mapping node index -> max(node_outer) for nodes
        that actually contribute non-zero probability
    """
    q0 = int(DFA.S0[0])
    nQ = len(DFA.S)
    D  = len(G.V)  # number of dimensions

    # sizes per dimension
    Ns = [G.V[d].shape[1] for d in range(D)]
    tv_shape = tuple(Ns)

    # ---- guard BEFORE any large allocations (incl. mask_q) ----
    elem_size = np.dtype(float).itemsize  # usually 8 bytes
    total_elems = int(np.prod(Ns, dtype=np.int64))
    est_bytes_one = total_elems * elem_size
    # working set ~ (|S| + 1) arrays of tv_shape (mask_q for each q + tv)
    est_bytes_work = est_bytes_one * (nQ + 1)

    if max_bytes is None:
        limit_bytes = max_elements * elem_size
    else:
        limit_bytes = int(max_bytes)

    if est_bytes_one > limit_bytes or est_bytes_work > max(2 * limit_bytes, limit_bytes):
        raise MemoryError(
            "compute_tv_from_tree: arrays too large to build safely.\n"
            f"  shape={tv_shape} → {total_elems:,} elems\n"
            f"  one array ≈ {_fmt_bytes(est_bytes_one)}; "
            f"  working set ≈ {_fmt_bytes(est_bytes_work)} (≈(|S|+1) arrays)\n"
            f"  limit ≈ {_fmt_bytes(limit_bytes)} "
            f"(tune with max_elements or max_bytes)"
        )

    # ---- build per-destination-state mask from q0 over all letters ----
    mask_q = [np.zeros(tv_shape, dtype=float) for _ in range(nQ)]
    trans = np.asarray(DFA.trans, dtype=int)
    n_letters = trans.shape[1]

    for l in range(n_letters):
        qdst = int(trans[q0, l])
        # N-ary outer over all dimensions of the label row L[d][l, :]
        label_vecs = [L[d][l, :] for d in range(D)]
        mask_q[qdst] += _outer_nd(label_vecs)

    # start with immediate acceptance (q0 -> F)
    tv = mask_q[int(DFA.F)].copy()

    # ---- accumulate contributions from nodes grouped by their DFA state ----
    non_final_states = [int(q) for q in DFA.S if int(q) != int(DFA.F)]

    nodes_with_prob = []
    nodes_no_prob = []
    node_outer_max = {}  # standalone container for visualization

    # IMPORTANT: loop only over non_final_states to keep tv identical to original
    for q in non_final_states:
        Mq = mask_q[q]
        # for each node n that carries DFA state q
        for n in G.Q.get(q, []):
            node_vecs = [G.V[d][n, :] for d in range(D)]
            node_outer = _outer_nd(node_vecs)
            contrib = Mq * node_outer

            if np.any(contrib != 0.0):
                nodes_with_prob.append(n)
                # store max(node_outer) (keep the largest if ever seen multiple times)
                m = float(node_outer.max())
                prev = node_outer_max.get(n, 0.0)
                if m > prev:
                    node_outer_max[n] = m
            else:
                nodes_no_prob.append(n)

            # tv accumulation is exactly as in the original version
            tv += contrib

    # ---- grouped diagnostics at the end (optional) ----
    if nodes_with_prob:
        uniq = sorted(set(nodes_with_prob))
        print("probability lying in nodes :" + ",".join(str(i) for i in uniq))
    else:
        print("probability lying in nodes :(none)")

    if nodes_no_prob:
        uniq0 = sorted(set(nodes_no_prob))
        print("no probability at all lying in nodes:" + ",".join(str(i) for i in uniq0))
    else:
        print("no probability at all lying in nodes:(none)")

    return tv, node_outer_max

# compute interested_

def _centers_from_sysAbs(sysAbs, Ns):
    """
    sysAbs[d].hx in your printout looks like [array([...])].
    This returns centers[d] as a 1D array of length N_d.
    """
    D = len(Ns)
    centers = []
    for d in range(D):
        hx = sysAbs[d].hx
        if isinstance(hx, (list, tuple)):
            # common: [array([...])]
            if len(hx) == 1:
                x = np.asarray(hx[0]).ravel()
            else:
                # pick the entry that matches N_d if possible
                cand = [np.asarray(a).ravel() for a in hx]
                x = next((c for c in cand if c.size == Ns[d]), cand[0])
        else:
            x = np.asarray(hx).ravel()
        if x.size != Ns[d]:
            raise ValueError(f"sysAbs[{d}].hx centers length {x.size} != N_{d}={Ns[d]}")
        centers.append(x)
    return centers

def tv_interested_region_from_tree(
    G, DFA, L, sysAbs, region, *,
    inclusive=True,
    max_elements=50_000_000,
    max_bytes=None,
    return_index_sets=True,
    return_node_outer_max=True,
    verbose=False
):
    """
    Region-only TV with semantics:
      - pick letter l for each composed state (via L)
      - qdst = trans[q0,l]
      - add sum_{n in G.Q[qdst]} prod_d G.V[d][n, idx_d]

    Inputs:
      - G.V[d]: (num_nodes, N_d)
      - L[d]:   (n_letters, N_d)  (typically 0/1)
      - sysAbs[d].hx: centers for dimension d (your case: [array([...])])
      - region[d] = (low, high) in center coordinates
    """
    q0 = int(DFA.S0[0])
    trans = np.asarray(DFA.trans, dtype=int)
    n_letters = trans.shape[1]

    D = len(G.V)
    if len(region) != D:
        raise ValueError(f"region must have length D={D}, got {len(region)}")
    if len(sysAbs) < D:
        raise ValueError(f"sysAbs must have at least D={D} entries")

    Ns = [G.V[d].shape[1] for d in range(D)]
    centers = _centers_from_sysAbs(sysAbs, Ns)

    # --- indices inside region (center-point test) ---
    idx_sets = []
    for d in range(D):
        lo, hi = region[d]
        x = centers[d]
        if inclusive:
            idx = np.where((x >= lo) & (x <= hi))[0]
        else:
            idx = np.where((x > lo) & (x < hi))[0]
        idx_sets.append(idx)

    # empty region -> empty tv
    if any(len(idx) == 0 for idx in idx_sets):
        tv_empty = np.zeros(tuple(len(idx) for idx in idx_sets), dtype=float)
        out = (tv_empty,)
        if return_index_sets:
            out += (idx_sets,)
        if return_node_outer_max:
            out += ({},)
        return out if len(out) > 1 else out[0]

    tv_shape = tuple(len(idx) for idx in idx_sets)

    # --- memory guard ---
    elem_size = np.dtype(float).itemsize
    total_elems = int(np.prod(tv_shape, dtype=np.int64))
    est_one = total_elems * elem_size

    # cache node-sums for each qdst that can occur from q0 (only those with nodes matter)
    qdsts = np.unique(trans[q0, :]).astype(int)
    qdsts = [int(q) for q in qdsts if len(G.Q.get(int(q), [])) > 0]
    K = len(qdsts)

    est_work = est_one * (1 + K)  # tv + node_sum arrays
    limit_bytes = int(max_bytes) if max_bytes is not None else int(max_elements * elem_size)

    if est_one > limit_bytes or est_work > max(2 * limit_bytes, limit_bytes):
        raise MemoryError(
            "tv_interested_region_from_tree: region arrays too large to build safely.\n"
            f"  region_shape={tv_shape} → {total_elems:,} elems\n"
            f"  one array ≈ {_fmt_bytes(est_one)}; working set ≈ {_fmt_bytes(est_work)} (tv + {K} node_sum)\n"
            f"  limit ≈ {_fmt_bytes(limit_bytes)}"
        )

    # --- precompute node_sum[q] over the region: sum_n outer(G.V at node n) ---
    node_sum = {}
    node_outer_max = {} if return_node_outer_max else None

    for q in qdsts:
        acc = np.zeros(tv_shape, dtype=float)
        for n in G.Q.get(q, []):
            node_vecs = [G.V[d][n, idx_sets[d]] for d in range(D)]
            node_outer = _outer_nd(node_vecs)
            acc += node_outer
            if return_node_outer_max:
                m = float(node_outer.max())
                prev = node_outer_max.get(n, 0.0)
                if m > prev:
                    node_outer_max[n] = m
        node_sum[q] = acc

    # --- accumulate tv over letters, routing to destination qdst ---
    tv = np.zeros(tv_shape, dtype=float)

    for l in range(n_letters):
        qdst = int(trans[q0, l])
        acc = node_sum.get(qdst, None)
        if acc is None:
            continue  # no nodes for that destination => contributes nothing

        label_vecs = [L[d][l, idx_sets[d]] for d in range(D)]
        letter_mask = _outer_nd(label_vecs)  # (region tensor); typically 0/1
        tv += letter_mask * acc

    if verbose:
        print(f"region idx sizes: {[len(i) for i in idx_sets]}")
        print(f"cached qdsts with nodes: {sorted(node_sum.keys())}")

    out = (tv,)
    if return_index_sets:
        out += (idx_sets,)
    if return_node_outer_max:
        out += (node_outer_max,)
    return out if len(out) > 1 else out[0]
