"""
Policy composition with block-column transition matrices.

The single public entry point is :func:`Pc`, which composes a policy
``pol`` with a block-column transition matrix or
:class:`TransitionProbability2D`, collapsing the action-letter dimension.

Mathematically, given a transition matrix in block-column form
``P_full`` of shape ``(n, n * nu)`` and a policy ``pol`` of shape
``(n, nu)`` mapping each state to a distribution over input letters,
``Pc`` returns the closed-loop transition matrix of shape ``(n, n)``::

    Pc(P_full, pol)[i, j] = sum_k P_full[i, k * n + j] * pol[i, k]

The function preserves the input type:
    - ``np.ndarray``                -> ``np.ndarray``
    - :class:`TransitionProbability2D` -> a new
      :class:`TransitionProbability2D` with the same grid shape ``l`` and
      separable kernels ``Pi``, but with ``P_det`` collapsed to ``(n, n)``.
"""

from typing import Union

import numpy as np
from scipy import sparse

from .tensor_transition_probability_2d import TransitionProbability2D

Matrix = Union[np.ndarray, sparse.spmatrix]

def Pc(P_flat, pol):
    """
    Compose a policy with the abstraction's transition operator,
    collapsing the input-letter dimension.

        Pc(P_flat, pol)[i, j] = sum_k 1[shift(j, k) == i] * pol[j, k]
    """
    if not sparse.issparse(pol):
        pol = sparse.csr_matrix(pol)
    n, nu = pol.shape

    # === 2D wrapper path =============================================
    if isinstance(P_flat, TransitionProbability2D):
        K0, K1 = P_flat.Pi[0], P_flat.Pi[1]
        l      = (P_flat.l1, P_flat.l2)

        # Gather-based projection: works directly from target_idx, no
        # (n, n*nu) intermediate.
        if P_flat._mode == "gather":
            target_idx = P_flat.target_idx                       # (n, nu)
            pol_dense  = pol.toarray()                           # (n, nu)

            # For each (source j, input k): add pol[j, k] to P_comp[target_idx[j, k], j].
            # Sentinel target_idx == n is excluded by the validity mask.
            sources = np.tile(np.arange(n, dtype=np.int64), nu)              # (n*nu,)
            targets = target_idx.ravel(order="F").astype(np.int64)           # (n*nu,)
            weights = pol_dense.ravel(order="F")                             # (n*nu,)

            valid = targets < n
            P_comp = sparse.csr_matrix(
                (weights[valid], (targets[valid], sources[valid])),
                shape=(n, n),
            )
            return TransitionProbability2D(l=l, P_det=P_comp, Pi0=K0, Pi1=K1)

        # Legacy sparse / dense P_det path (post-Pc operators land here
        # if they're ever projected again, and old call sites still work).
        P_full = P_flat.P_det
        v      = np.asarray(pol.toarray()).ravel(order="F")
        if sparse.issparse(P_full):
            Plarge = P_full @ sparse.diags(v)
            blocks = Plarge.shape[1] // n
            P_comp = sparse.csr_matrix((n, n))
            for i in range(blocks):
                P_comp += Plarge[:, i * n:(i + 1) * n]
        else:
            P_full = np.asarray(P_full, dtype=float)
            Plarge = P_full * v
            blocks = Plarge.shape[1] // n
            P_comp = Plarge.reshape(n, n, blocks, order="F").sum(axis=2)
        return TransitionProbability2D(l=l, P_det=P_comp, Pi0=K0, Pi1=K1)

    # === 1D dense path: return (n, n) ndarray ========================
    P_full = np.asarray(P_flat, dtype=float)
    v      = np.asarray(pol.toarray()).ravel(order="F")
    Plarge = P_full * v
    blocks = Plarge.shape[1] // n
    return Plarge.reshape(n, n, blocks, order="F").sum(axis=2)

#
# def Pc(P_flat: Union[np.ndarray, TransitionProbability2D], pol: Matrix):
#     """
#     Compose a policy with a block-column transition matrix.
#
#     Parameters
#     ----------
#     P_flat : np.ndarray or TransitionProbability2D
#         Transition object to be composed with the policy. Two forms
#         are accepted:
#
#             - ``np.ndarray`` of shape ``(n, n * nu)``: dense
#               block-column transition matrix. Returned collapsed to
#               ``(n, n)``.
#             - :class:`TransitionProbability2D` with ``P_det`` of shape
#               ``(n, n * nu)``: returned as a NEW
#               :class:`TransitionProbability2D` with the same ``l`` and
#               separable kernels ``Pi``, but with ``P_det`` collapsed
#               to ``(n, n)``.
#
#     pol : np.ndarray or scipy.sparse matrix
#         Policy of shape ``(n, nu)``. Each row ``pol[i, :]`` is a
#         distribution over the ``nu`` input letters at state ``i``.
#         Coerced to CSR sparse internally.
#
#     Returns
#     -------
#     np.ndarray or TransitionProbability2D
#         Same type as ``P_flat``, with the action-letter dimension
#         collapsed. The input ``P_flat`` is never mutated.
#     """
#     # Normalise the policy to CSR and pack it as a length-(n*nu) vector
#     # in Fortran (column-major) order: [pol[:, 0]; pol[:, 1]; ...].
#     if not sparse.issparse(pol):
#         pol = sparse.csr_matrix(pol)
#     n, nu = pol.shape
#     v = np.asarray(pol.toarray()).ravel(order="F")   # (n * nu,)
#
#     # === 2D wrapper path: return a NEW TransitionProbability2D ============
#     if hasattr(P_flat, "P_det"):
#         P_full = P_flat.P_det                         # (n, n*nu)
#         K0, K1 = P_flat.Pi[0], P_flat.Pi[1]
#         l = (P_flat.l1, P_flat.l2)
#
#         if sparse.issparse(P_full):
#             Plarge = P_full @ sparse.diags(v)         # (n, n*nu) sparse
#             blocks = Plarge.shape[1] // n
#             P_comp = sparse.csr_matrix((n, n))
#             for i in range(blocks):
#                 P_comp += Plarge[:, i * n:(i + 1) * n]   # sum n-wide blocks
#         else:
#             P_full = np.asarray(P_full, dtype=float)
#             Plarge = P_full * v                       # column scaling
#             blocks = Plarge.shape[1] // n
#             P_comp = Plarge.reshape(n, n, blocks, order="F").sum(axis=2)
#
#         return TransitionProbability2D(l=l, P_det=P_comp, Pi0=K0, Pi1=K1)
#
#     # === 1D dense path: return (n, n) ndarray =============================
#     P_full = np.asarray(P_flat, dtype=float)          # (n, n*nu)
#     Plarge = P_full * v
#     blocks = Plarge.shape[1] // n
#     return Plarge.reshape(n, n, blocks, order="F").sum(axis=2)