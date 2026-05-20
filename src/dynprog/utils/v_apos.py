"""
A-posteriori (APoS) delta correction for the satisfaction tensor.

Applies an N-dimensional delta correction to an uncorrected satisfaction
tensor ``tv`` produced by tree-based value iteration. The correction
accounts for the abstraction error per dimension, propagated through the
DFA tree over a horizon ``T``.

The single public entry point is :func:`apply_delta_correction_apos`.

Open question
-------------
The current correction does not mask ``delta_nd`` by per-destination-state
reachability from ``q0`` (the ``mask_q`` tensor). An alternative form,
which DOES apply that mask, is preserved in the comments and as a
computed but unused tensor. It is currently unclear which form is
correct; resolution requires a numerical comparison against published
APoS results (e.g. paper Fig. 4 curves). See the comment block in
:func:`apply_delta_correction_apos` for details.
"""

import numpy as np

from src.dynprog.utils.delta_corr_apos import (
    _outer_nd,
    weighted_vtens_sum_per_state,
)


def apply_delta_correction_apos(tv, G, DFA, L, sysAbs, delta_sys, T, dims=None):
    """
    N-D a-posteriori delta correction.

    Parameters
    ----------
    tv : np.ndarray
        Uncorrected satisfaction tensor of shape
        ``(N_0, N_1, ..., N_{D-1})``.
    G : DFATree
        With ``G.V[d]`` of shape ``(num_nodes, N_d)``.
    DFA : DFA object
        Used for state list, accepting state, and transition table.
    L : list of np.ndarray
        Per-dimension labeling, ``L[d]`` of shape ``(n_letters, N_d)``.
    sysAbs : dict
        Per-dimension abstraction objects. Currently used only to obtain
        the dimension keys when ``dims`` is None.
    delta_sys : dict or sequence of float
        Per-dimension delta scalars. If a dict, indexed by the same keys
        as ``sysAbs``. If a sequence, aligned to the chosen ``dims`` /
        ``sorted(sysAbs.keys())``.
    T : int
        Horizon for the correction.
    dims : iterable, optional
        Dimension keys (and their order). Defaults to
        ``sorted(sysAbs.keys())``.

    Returns
    -------
    np.ndarray
        Corrected satisfaction tensor, same shape as ``tv``,
        clamped to ``[0, 1]``.
    """
    # === Dimension/order bookkeeping ======================================
    keys = sorted(sysAbs.keys()) if dims is None else list(dims)
    D = len(keys)
    Ns = [G.V[k].shape[1] for k in keys]
    tv_shape = tuple(Ns)

    # === Per-destination-state reachability mask from q0 ==================
    # mask_q[q] is the N-D tensor:
    #     sum_{l : trans(q0, l) = q}  outer_d L[d][l, :]
    # i.e. the membership of each abstract state in a letter that
    # transitions q0 -> q.
    #
    # Currently this is computed but NOT used in the final correction
    # (see the OPEN QUESTION block below).
    q0 = int(DFA.S0[0])
    F  = int(DFA.F)
    nQ = len(DFA.S)
    trans = np.asarray(DFA.trans, dtype=int)
    n_letters = trans.shape[1]

    mask_q = [np.zeros(tv_shape, dtype=float) for _ in range(nQ)]
    for l in range(n_letters):
        qdst = int(trans[q0, l])
        label_vecs = [L[k][l, :] for k in keys]
        mask_q[qdst] += _outer_nd(label_vecs)

    # === N-D delta tensor: 1 - prod_d (1 - delta_d) =======================
    if isinstance(delta_sys, dict):
        delta_scalars = [float(delta_sys[k]) for k in keys]
    else:
        if len(delta_sys) != D:
            raise ValueError(f"delta_sys length {len(delta_sys)} != #dims {D}")
        delta_scalars = [float(x) for x in delta_sys]

    one_minus_delta_nd = _outer_nd([1.0 - dv for dv in delta_scalars])
    delta_nd = 1.0 - one_minus_delta_nd

    # === Weighted vtens per state =========================================
    non_final_states = [int(q) for q in DFA.S if int(q) != F]
    SumNodes_Q = weighted_vtens_sum_per_state(
        G, T=T, non_final_states=non_final_states
    )
    # SumNodes_Q is a list of N-D ndarrays, length nQ, indexed by DFA state.

    # === Assemble correction ==============================================
    #
    # OPEN QUESTION (preserved from earlier iterations of this code):
    # Two candidate corrections exist. The current implementation uses
    # variant (A); variant (B), which masks the delta tensor by mask_q[q],
    # is preserved here so it can be compared numerically against the
    # paper's APoS results.
    #
    #   (A) unmasked  (currently active):
    #       delta_correction = sum_{q non-final}
    #           (-T * delta_nd + delta_nd * SumNodes_Q[q])
    #
    #   (B) masked    (currently inactive):
    #       delta_nd_Q[q] = delta_nd * mask_q[q]
    #       delta_correction = sum_{q non-final}
    #           (-T * delta_nd_Q[q] + delta_nd_Q[q] * SumNodes_Q[q])
    #
    # To switch, replace the loop body below with the variant-(B) version.
    delta_correction = sum(
        (-T * delta_nd + delta_nd * SumNodes_Q[q])
        for q in non_final_states
    )

    return np.clip(tv + delta_correction, 0.0, 1.0)