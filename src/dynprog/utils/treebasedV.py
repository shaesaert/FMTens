"""
Satisfaction-tensor (TV) computation from DFA-tree value functions.

Two public entry points:

    - :func:`compute_tv_from_tree`: builds the full N-D satisfaction
      tensor by accumulating contributions from every DFA-tree node.
    - :func:`tv_interested_region_from_tree`: same idea but restricted
      to a hyperrectangular region in state space, avoiding the full
      tensor allocation.

Both functions include a pre-allocation memory guard that raises
:class:`MemoryError` (with a helpful estimate) before attempting to
allocate tensors that would exceed the requested limit. Caller can
tune the limit via ``max_elements`` or ``max_bytes``.

The input ``G`` is a DFA-tree object exposing:
    - ``G.V[d]`` of shape ``(num_nodes, N_d)`` -- per-dimension value
      functions, indexed by node and grid axis
    - ``G.Q[q]`` -> iterable of tree-node indices for DFA state ``q``

The labeling ``L`` is a list of arrays ``L[d]`` of shape
``(n_letters, N_d)`` giving per-letter, per-axis membership.
"""

import numpy as np

from src.dynprog.utils.delta_corr_apos import _outer_nd


def _fmt_bytes(n):
    """Format a byte count using binary units (B, KiB, MiB, ...)."""
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PiB"


def compute_tv_from_tree(
    G, DFA, L,
    *,
    max_elements: int = 50_000_000,
    max_bytes=None,
    verbose: bool = False,
):
    """
    Build the N-D satisfaction tensor ``tv`` from the DFA tree.

    Parameters
    ----------
    G : DFA-tree object
        Must expose ``V[d]`` of shape ``(num_nodes, N_d)`` and
        ``Q[q]`` -> tree-node indices.
    DFA : DFA object
        Must expose ``S``, ``F``, ``S0``, ``trans``.
    L : list of np.ndarray
        Per-dimension labeling, ``L[d]`` of shape ``(n_letters, N_d)``.
    max_elements : int, optional
        Maximum total elements per tensor. Default ``50_000_000``.
    max_bytes : int, optional
        Maximum bytes per tensor; overrides ``max_elements`` if given.
    verbose : bool, optional
        If True, print per-node probability diagnostics at the end.
        Default ``False``.

    Returns
    -------
    tv : np.ndarray
        Satisfaction tensor of shape ``(N_0, N_1, ..., N_{D-1})``.
    node_outer_max : dict
        Mapping ``node_index -> max(node_outer)`` for nodes that
        contributed non-zero probability. Used by visualisation.

    Raises
    ------
    MemoryError
        If the tensor or working set would exceed the requested limit.
    """
    q0 = int(DFA.S0[0])
    nQ = len(DFA.S)
    D  = len(G.V)
    Ns = [G.V[d].shape[1] for d in range(D)]
    tv_shape = tuple(Ns)

    # === Memory guard (before any large allocations) ======================
    elem_size = np.dtype(float).itemsize
    total_elems = int(np.prod(Ns, dtype=np.int64))
    est_bytes_one = total_elems * elem_size
    est_bytes_work = est_bytes_one * (nQ + 1)  # mask_q for each q + tv

    if max_bytes is None:
        limit_bytes = max_elements * elem_size
    else:
        limit_bytes = int(max_bytes)

    if est_bytes_one > limit_bytes or est_bytes_work > max(2 * limit_bytes, limit_bytes):
        raise MemoryError(
            "compute_tv_from_tree: arrays too large to build safely.\n"
            f"  shape={tv_shape} → {total_elems:,} elems\n"
            f"  one array ≈ {_fmt_bytes(est_bytes_one)};"
            f"  working set ≈ {_fmt_bytes(est_bytes_work)} (≈(|S|+1) arrays)\n"
            f"  limit ≈ {_fmt_bytes(limit_bytes)}"
            f" (tune with max_elements or max_bytes)"
        )

    # === Build per-destination-state masks from q0 ========================
    mask_q = [np.zeros(tv_shape, dtype=float) for _ in range(nQ)]
    trans = np.asarray(DFA.trans, dtype=int)
    n_letters = trans.shape[1]

    for l in range(n_letters):
        qdst = int(trans[q0, l])
        label_vecs = [L[d][l, :] for d in range(D)]
        mask_q[qdst] += _outer_nd(label_vecs)

    # Start with immediate acceptance (q0 -> F).
    tv = mask_q[int(DFA.F)].copy()

    # === Accumulate node contributions, grouped by DFA state ==============
    non_final_states = [int(q) for q in DFA.S if int(q) != int(DFA.F)]
    nodes_with_prob = []
    nodes_no_prob = []
    node_outer_max = {}

    for q in non_final_states:
        Mq = mask_q[q]
        for n in G.Q.get(q, []):
            node_vecs = [G.V[d][n, :] for d in range(D)]
            node_outer = _outer_nd(node_vecs)
            contrib = Mq * node_outer

            if np.any(contrib != 0.0):
                nodes_with_prob.append(n)
                m = float(node_outer.max())
                if m > node_outer_max.get(n, 0.0):
                    node_outer_max[n] = m
            else:
                nodes_no_prob.append(n)

            tv += contrib

    # === Optional diagnostics =============================================
    if verbose:
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


def _centers_from_sysAbs(sysAbs, Ns):
    """
    Extract per-dimension centre arrays from ``sysAbs``.

    Each ``sysAbs[d].hx`` may be a 1D array, a list, or a tuple. For
    list/tuple inputs, picks the entry whose length matches ``Ns[d]``;
    otherwise raises.

    Returns a list of length ``D``, each entry a 1D array of length
    ``N_d``.
    """
    D = len(Ns)
    centers = []
    for d in range(D):
        hx = sysAbs[d].hx
        if isinstance(hx, (list, tuple)):
            if len(hx) == 1:
                x = np.asarray(hx[0]).ravel()
            else:
                cand = [np.asarray(a).ravel() for a in hx]
                x = next((c for c in cand if c.size == Ns[d]), cand[0])
        else:
            x = np.asarray(hx).ravel()
        if x.size != Ns[d]:
            raise ValueError(f"sysAbs[{d}].hx centers length {x.size} != N_{d}={Ns[d]}")
        centers.append(x)
    return centers


def tv_interested_region_from_tree(
    G, DFA, L, sysAbs, region,
    *,
    inclusive: bool = True,
    max_elements: int = 50_000_000,
    max_bytes=None,
    return_index_sets: bool = True,
    return_node_outer_max: bool = True,
    verbose: bool = False,
):
    """
    Satisfaction tensor restricted to a hyperrectangular region.

    Computes the same TV as :func:`compute_tv_from_tree` but only over
    the grid points inside ``region``, avoiding allocation of the full
    state-space tensor.

    For each letter ``l``::

        qdst = trans[q0, l]
        tv  += letter_mask(l) * sum_{n in G.Q[qdst]}
                                    outer_d G.V[d][n, idx_d]

    where ``idx_d`` are the grid indices in dimension ``d`` whose centre
    points lie in ``region[d]``.

    Parameters
    ----------
    G : DFA-tree object
        Same requirements as :func:`compute_tv_from_tree`.
    DFA : DFA object
        Must expose ``S0`` and ``trans``.
    L : list of np.ndarray
        Per-dimension labeling.
    sysAbs : list
        Per-dimension abstraction objects, each exposing ``hx`` (centre
        points).
    region : sequence of (float, float)
        Per-dimension ``(low, high)`` bounds.
    inclusive : bool, optional
        Closed intervals ``[low, high]`` if True, open ``(low, high)``
        otherwise. Default ``True``.
    max_elements : int, optional
        Maximum total elements per tensor. Default ``50_000_000``.
    max_bytes : int, optional
        Maximum bytes per tensor; overrides ``max_elements`` if given.
    return_index_sets : bool, optional
        If True, also return the per-dimension grid indices selected.
        Default ``True``.
    return_node_outer_max : bool, optional
        If True, also return the ``node_outer_max`` diagnostic dict.
        Default ``True``.
    verbose : bool, optional
        If True, print region/destination diagnostics. Default ``False``.

    Returns
    -------
    tv : np.ndarray
        Satisfaction tensor over the region.
    idx_sets : list of np.ndarray, optional
        Per-dimension grid indices selected.
    node_outer_max : dict, optional
        Mapping node -> max(node_outer).

    Raises
    ------
    ValueError
        If ``region`` or ``sysAbs`` lengths don't match the tree.
    MemoryError
        If region arrays would exceed the requested limit.
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

    # === Indices inside region (centre-point test) ========================
    idx_sets = []
    for d in range(D):
        lo, hi = region[d]
        x = centers[d]
        if inclusive:
            idx = np.where((x >= lo) & (x <= hi))[0]
        else:
            idx = np.where((x > lo) & (x < hi))[0]
        idx_sets.append(idx)

    # Empty region -> empty tv (short-circuit).
    if any(len(idx) == 0 for idx in idx_sets):
        tv_empty = np.zeros(tuple(len(idx) for idx in idx_sets), dtype=float)
        out = (tv_empty,)
        if return_index_sets:
            out += (idx_sets,)
        if return_node_outer_max:
            out += ({},)
        return out if len(out) > 1 else out[0]

    tv_shape = tuple(len(idx) for idx in idx_sets)

    # === Memory guard =====================================================
    elem_size = np.dtype(float).itemsize
    total_elems = int(np.prod(tv_shape, dtype=np.int64))
    est_one = total_elems * elem_size

    # Cache node-sums for each qdst reachable from q0 (only those with nodes).
    qdsts = np.unique(trans[q0, :]).astype(int)
    qdsts = [int(q) for q in qdsts if len(G.Q.get(int(q), [])) > 0]
    K = len(qdsts)

    est_work = est_one * (1 + K)  # tv + K node_sum arrays
    limit_bytes = int(max_bytes) if max_bytes is not None else int(max_elements * elem_size)

    if est_one > limit_bytes or est_work > max(2 * limit_bytes, limit_bytes):
        raise MemoryError(
            "tv_interested_region_from_tree: region arrays too large to build safely.\n"
            f"  region_shape={tv_shape} → {total_elems:,} elems\n"
            f"  one array ≈ {_fmt_bytes(est_one)};"
            f" working set ≈ {_fmt_bytes(est_work)} (tv + {K} node_sum)\n"
            f"  limit ≈ {_fmt_bytes(limit_bytes)}"
        )

    # === Precompute node_sum[q] over the region ===========================
    # node_sum[q] = sum_{n in G.Q[q]} outer_d G.V[d][n, idx_sets[d]]
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
                if m > node_outer_max.get(n, 0.0):
                    node_outer_max[n] = m
        node_sum[q] = acc

    # === Accumulate tv over letters, routing to destination qdst ==========
    tv = np.zeros(tv_shape, dtype=float)
    for l in range(n_letters):
        qdst = int(trans[q0, l])
        acc = node_sum.get(qdst)
        if acc is None:
            continue
        label_vecs = [L[d][l, idx_sets[d]] for d in range(D)]
        letter_mask = _outer_nd(label_vecs)
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