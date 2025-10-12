import numpy as np

def _level_indices(G, level: int, bucket='both'):
    """
    Fetch node indices at a given depth `level` from G.Dl.
    Dl[depth][0] -> qf nodes, Dl[depth][1] -> non-final (q0,q1,...) nodes.
    bucket: 0, 1, or 'both'
    """
    if not hasattr(G, "Dl") or level < 0 or level >= len(G.Dl):
        return []
    if bucket == 'both':
        return list(G.Dl[level][0]) + list(G.Dl[level][1])
    if bucket in (0, 1):
        return list(G.Dl[level][bucket])
    raise ValueError("bucket must be 0, 1, or 'both'")

def _outer_nd(vectors):
    """
    N-ary outer product of 1D arrays: returns an N-D array whose shape is
    tuple(len(v) for v in vectors) and whose entries are the product of all coords.
    """
    out = np.array(1.0, dtype=float)
    for v in vectors:
        out = np.multiply.outer(out, np.asarray(v, dtype=float))
    return out

def _vtens_from_indices(G, idxs, dims=None):
    """
    N-D version of vtens over a set of node indices:
        sum_{n in idxs}  ⊗_{d in dims} G.V[d][n, :]

    Returns an N-D ndarray with shape (N_d0, N_d1, ..., N_d{D-1}).
    """
    if not idxs:
        # build an all-zeros tensor with the correct shape
        if dims is None:
            dims = range(len(G.V))
        shape = tuple(G.V[d].shape[1] for d in dims)
        return np.zeros(shape, dtype=float)

    if dims is None:
        dims = range(len(G.V))

    # accumulate N-D sum of outers
    acc = None
    for n in idxs:
        vecs = [G.V[d][n, :] for d in dims]  # 1D arrays per dim
        tens = _outer_nd(vecs)               # N-D
        acc = tens if acc is None else (acc + tens)
    return acc

def weighted_vtens_sum_per_state(G, T: int, non_final_states, dims=None):

    # dimension order (defaults to all dims, in order 0..D-1)
    if dims is None:
        dims = tuple(range(len(G.V)))
    else:
        dims = tuple(dims)

    # shape along chosen dims
    shape = tuple(G.V[d].shape[1] for d in dims)

    nQ = len(G.DFA.S)
    # init result containers
    SumNodes_Q = [np.zeros(shape, dtype=float) for _ in range(nQ)]

    # pre-build sets for fast intersection
    Q_sets = {int(q): set(G.Q.get(int(q), [])) for q in G.DFA.S}

    up_to = min(T, len(getattr(G, 'Dl', [])))
    for i in range(up_to):
        level_nodes = set(_level_indices(G, i, bucket='both'))
        if not level_nodes:
            continue
        w = (T - i)
        # accumulate per state at this level
        for q in non_final_states:
            q = int(q)
            idxs_q_level = list(level_nodes & Q_sets.get(q, set()))
            if not idxs_q_level:
                continue
            SumNodes_Q[q] += w * _vtens_from_indices(G, idxs_q_level, dims=dims)

    return SumNodes_Q
