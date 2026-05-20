"""
A-posteriori (APoS) delta correction for tree-based value iteration.

Helpers to compute the per-DFA-state weighted sum of value-function
tensor products over nodes of the DFA tree, used as the correction
term in the APoS variant of dual-tree dynamic programming.

The single public entry point is :func:`weighted_vtens_sum_per_state`,
which computes, for each DFA state ``q`` in ``non_final_states``::

    SumNodes_Q[q] = sum_{level=0..T-1}  (T - level) *
                    sum_{n in nodes at level, n.Q == q}
                        outer_product_{d in dims}  G.V[d][n, :]

The input ``G`` is a DFA-tree object that must expose:
    - ``G.DFA``  : underlying DFA, with state list ``DFA.S``
    - ``G.Dl``   : per-depth node buckets, ``Dl[depth][0|1]``
                   (bucket 0 = accepting/qf nodes, bucket 1 = non-final)
    - ``G.V``    : list of per-dimension value-function arrays,
                   indexed as ``V[d][node, axis_index]``
    - ``G.Q``    : mapping DFA state -> set of tree-node indices
"""

import numpy as np


def _level_indices(G, level: int, bucket="both"):
    """
    Fetch node indices at a given depth ``level`` from ``G.Dl``.

    Parameters
    ----------
    G : DFA-tree object
        Must expose ``Dl``.
    level : int
        Depth to query. Out-of-range levels return an empty list.
    bucket : {0, 1, "both"}, optional
        Which bucket to read: 0 = accepting (``q_f``) nodes,
        1 = non-final nodes, ``"both"`` = concatenation of 0 then 1.
        Default ``"both"``.
    """
    if not hasattr(G, "Dl") or level < 0 or level >= len(G.Dl):
        return []
    if bucket == "both":
        return list(G.Dl[level][0]) + list(G.Dl[level][1])
    if bucket in (0, 1):
        return list(G.Dl[level][bucket])
    raise ValueError("bucket must be 0, 1, or 'both'")


def _outer_nd(vectors):
    """
    N-ary outer product of 1D arrays.

    Returns an N-D array of shape ``tuple(len(v) for v in vectors)``
    whose entries are the product of all coordinates.
    """
    out = np.array(1.0, dtype=float)
    for v in vectors:
        out = np.multiply.outer(out, np.asarray(v, dtype=float))
    return out


def _vtens_from_indices(G, idxs, dims=None):
    """
    N-D vtens over a set of node indices.

    Computes::

        sum_{n in idxs}  ⊗_{d in dims}  G.V[d][n, :]

    Parameters
    ----------
    G : DFA-tree object
        Must expose ``V``.
    idxs : iterable of int
        Node indices to sum over.
    dims : iterable of int, optional
        Dimensions to take the outer product over. Defaults to all
        dimensions of ``G.V`` in their natural order.

    Returns
    -------
    np.ndarray
        N-D tensor of shape ``(G.V[d0].shape[1], G.V[d1].shape[1], ...)``.
        Zero tensor if ``idxs`` is empty.
    """
    if dims is None:
        dims = range(len(G.V))

    if not idxs:
        shape = tuple(G.V[d].shape[1] for d in dims)
        return np.zeros(shape, dtype=float)

    acc = None
    for n in idxs:
        vecs = [G.V[d][n, :] for d in dims]
        tens = _outer_nd(vecs)
        acc = tens if acc is None else (acc + tens)
    return acc


def weighted_vtens_sum_per_state(G, T: int, non_final_states, dims=None):
    """
    Compute the per-DFA-state weighted vtens sum for the APoS correction.

    For each DFA state ``q`` in ``non_final_states``::

        SumNodes_Q[q] = sum_{level=0..T-1}  (T - level) *
                        sum_{n in nodes at level, n.Q == q}
                            outer_product_{d in dims}  G.V[d][n, :]

    Parameters
    ----------
    G : DFA-tree object
        Must expose ``DFA.S`` (DFA state list), ``Dl`` (per-depth node
        buckets), ``V`` (per-dimension value arrays), and ``Q``
        (mapping DFA state -> set of tree-node indices).
    T : int
        Horizon length. Levels deeper than ``T`` are not summed; the
        weight at level ``i`` is ``T - i``, decreasing with depth.
    non_final_states : iterable of int
        DFA states to compute the per-state sum for.
    dims : iterable of int, optional
        Dimensions to take the outer product over. Defaults to all
        dimensions of ``G.V``.

    Returns
    -------
    list of np.ndarray
        ``SumNodes_Q``, length ``len(G.DFA.S)``. Entries for DFA states
        not in ``non_final_states`` (and for empty levels) remain zero.
    """
    if dims is None:
        dims = tuple(range(len(G.V)))
    else:
        dims = tuple(dims)

    shape = tuple(G.V[d].shape[1] for d in dims)
    nQ = len(G.DFA.S)
    SumNodes_Q = [np.zeros(shape, dtype=float) for _ in range(nQ)]

    # Pre-build sets for fast intersection.
    Q_sets = {int(q): set(G.Q.get(int(q), [])) for q in G.DFA.S}

    up_to = min(T, len(getattr(G, "Dl", [])))
    for i in range(up_to):
        level_nodes = set(_level_indices(G, i, bucket="both"))
        if not level_nodes:
            continue
        w = T - i  # weight at depth i (decreasing with depth)
        for q in non_final_states:
            q = int(q)
            idxs_q_level = list(level_nodes & Q_sets.get(q, set()))
            if not idxs_q_level:
                continue
            SumNodes_Q[q] += w * _vtens_from_indices(G, idxs_q_level, dims=dims)

    return SumNodes_Q