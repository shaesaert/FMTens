import numpy as np

def _outer_nd(vectors):
    """
    N-ary outer product: given [v0, v1, ..., v_{D-1}] (1D arrays),
    return an array with shape (len(v0), len(v1), ..., len(v_{D-1}))
    whose entries are the product over all coordinates.
    """
    out = np.array(1.0, dtype=float)
    for v in vectors:
        out = np.multiply.outer(out, np.asarray(v, dtype=float))
    return out

def apply_delta_correction_apos(tv, G, DFA, L, sysAbs, delta_sys, T, dims=None):
    """
    N-D a-posteriori delta correction (faithful N-D tensors).

    Inputs
    ------
    tv       : N-D ndarray, shape (N_0, N_1, ..., N_{D-1})
               The uncorrected satisfaction tensor (already N-D).
    G        : DFATree (with G.V[d] shape (num_nodes, N_d))
    DFA, L   : standard DFA and per-dimension labeling (L[d] shape (n_letters, N_d))
    sysAbs   : dict of abstractions, used to size delta vectors
    delta_sys: list/tuple (aligned to sorted(sysAbs.keys())) OR dict {dim_key: scalar}
               Per-dimension delta scalars (e.g., 0.01 per dim)
    T        : int, horizon for the correction
    dims     : optional iterable of dim keys (order). If None, sorted(sysAbs.keys()).

    Returns
    -------
    tv_corr  : N-D ndarray (same shape as tv), clamped to [0, 1]
    """
    # ----- dimension/order bookkeeping -----
    keys = sorted(sysAbs.keys()) if dims is None else list(dims)
    D = len(keys)
    Ns = [G.V[k].shape[1] for k in keys]
    tv_shape = tuple(Ns)

    # ----- build per-destination-state masks (full N-D) from q0 over letters -----
    q0 = int(DFA.S0[0])
    F  = int(DFA.F)
    nQ = len(DFA.S)
    trans = np.asarray(DFA.trans, dtype=int)
    n_letters = trans.shape[1]

    mask_q = [np.zeros(tv_shape, dtype=float) for _ in range(nQ)]
    for l in range(n_letters):
        qdst = int(trans[q0, l])
        # N-D outer over label rows across all dims
        label_vecs = [L[k][l, :] for k in keys]
        mask_q[qdst] += _outer_nd(label_vecs)

    # ----- N-D delta tensor: 1 - ⊗_d (1 - delta_d) -----
    if isinstance(delta_sys, dict):
        delta_scalars = [float(delta_sys[k]) for k in keys]
    else:
        # assume iterable aligned with 'keys'
        if len(delta_sys) != D:
            raise ValueError(f"delta_sys length {len(delta_sys)} != #dims {D}")
        delta_scalars = [float(x) for x in delta_sys]

    delta_vecs = [np.full(sysAbs[k].N, delta_scalars[i], dtype=float)
                  for i, k in enumerate(keys)]
    one_minus_delta_nd = _outer_nd([1.0 - dv for dv in delta_vecs])   # N-D
    delta_nd = 1.0 - one_minus_delta_nd                               # N-D

    # ----- per-destination-state deltas -----
    non_final_states = [int(q) for q in DFA.S if int(q) != F]
    delta_nd_Q = {q: (delta_nd * mask_q[q]) for q in non_final_states}

    # ----- weighted vtens per state (must be N-D) -----
    # Expectation: this utility returns an N-D array per q with shape == tv_shape.
    # (If your current function is 2D, adapt it to produce N-D or write an N-D variant.)
    from src.dynprog.utils.delta_corr_apos import weighted_vtens_sum_per_state
    SumNodes_Q = weighted_vtens_sum_per_state(G, T=T, non_final_states=non_final_states)
    # SumNodes_Q should be a dict: {q: N-D ndarray with shape tv_shape}

    # ----- assemble correction -----
    delta_correction = sum(
        (-T * delta_nd_Q[q] + delta_nd_Q[q] * SumNodes_Q[q]) for q in non_final_states
    )

    # ----- clamp & return -----
    return np.clip(tv + delta_correction, 0.0, 1.0)

