"""
Transition matrix builder for the MDP abstraction.

Constructs the transition matrix of an abstract MDP from a continuous-state
linear system. Two construction paths are dispatched on ``mode``:

    - ``mode='1d'``: direct construction along a single state axis.
      Computes per-cell Gaussian cumulants and returns the transition
      tensor in either flat ``(N, N*M)`` or block ``(N, N, M)`` form.
    - ``mode='2d'``: 2D separable SVD construction. SVD-decomposes the
      noise covariance into per-dimension Toeplitz kernels plus a sparse
      deterministic-shift operator, packaged as a
      :class:`TransitionProbability2D` together with auxiliary data.

The ``'nd'`` mode is reserved for higher-rank tensor decompositions and
is not yet implemented.

The public entry point :func:`transition_matrix_nd_separable_impl` is
called by :class:`MDPModel.transition_matrix_nd_separable` in
:mod:`src.models.mdpmodel`, via :mod:`abstraction_factory`.
"""

from __future__ import annotations

import warnings
from typing import List, Literal

import numpy as np
import polytope as pc
from polytope import qhull
from scipy.linalg import toeplitz
from scipy.sparse import coo_matrix
from scipy.stats import norm

from .abstraction_tensor_helper import (
    linear_image_vertices,
    ff2n,
    combvec_first_fast,
    f_det,
    repmat_vector_col,
)
from .tensor_transition_probability_2d import TransitionProbability2D

Array = np.ndarray


# === public dispatch =======================================================

def transition_matrix_nd_separable_impl(
    sys,
    X_axes: List[Array],
    U_points: Array,
    X_poly,
    tol: float = 1e-19,
    renormalize: bool = True,
    return_flat: bool = True,
    mode: Literal["1d", "2d", "nd"] = "1d",
):
    """
    Build the abstraction transition matrix for ``sys``.

    Two construction paths, dispatched on ``mode``:

        - ``mode='1d'``: 1D direct construction; returns a tensor.
        - ``mode='2d'``: 2D separable SVD construction; returns a 7-tuple
          ``(tP_2d, hz, XhatSpace, beta, sys, ZhatSpace, Uz)``.

    Parameters
    ----------
    sys : LinModel-like
        Linear system with attributes ``A``, ``B``, ``Bw``, ``mu``,
        ``sigma``, and ``X`` (a Polytope).
    X_axes : list of np.ndarray
        Per-dimension axes, each shaped ``(1, m_i)``, as returned by
        :func:`abstraction_grid_helper.make_uniform_grid`.
    U_points : np.ndarray
        Input letter grid, shape ``(n_u, M)``.
    X_poly : polytope.Polytope
        State-space polytope. Accepted by both paths but currently
        unused; kept for signature compatibility.
    tol : float, optional
        Threshold below which Gaussian cumulant differences are zeroed
        to keep the transition matrix sparse. Default ``1e-19``.
    renormalize : bool, optional
        Currently unused; kept for signature compatibility with older
        callers. Default ``True``.
    return_flat : bool, optional
        Used only by ``mode='1d'``. If True, the transition tensor is
        returned flattened to ``(N, N*M)``; otherwise as block form
        ``(N, N, M)``. Default ``True``.
    mode : {'1d', '2d', 'nd'}, optional
        Construction path. Default ``'1d'``.

    Returns
    -------
    ``mode='1d'`` : np.ndarray
        Transition tensor, flat ``(N, N*M)`` or block ``(N, N, M)``
        depending on ``return_flat``.
    ``mode='2d'`` : tuple
        ``(tP_2d, hz, XhatSpace, beta, sys, ZhatSpace, Uz)``:
            - ``tP_2d`` : :class:`TransitionProbability2D` operator
            - ``hz`` : list of per-dimension axes on the rotated grid
            - ``XhatSpace`` : centred grid in the original frame
            - ``beta`` : per-cell box vertices mapped through ``Uz``
            - ``sys`` : the input system (echoed back)
            - ``ZhatSpace`` : centred grid in the rotated frame
            - ``Uz`` : left singular vectors of the noise covariance

    Raises
    ------
    NotImplementedError
        If ``mode='nd'``.
    ValueError
        If ``mode`` is none of the recognised values.
    """
    if mode == "1d":
        return _transition_matrix_1d(
            sys, X_axes, U_points, tol=tol, return_flat=return_flat
        )
    if mode == "2d":
        return _transition_matrix_2d_separable(
            sys, X_axes, U_points, tol=tol
        )
    if mode == "nd":
        raise NotImplementedError(
            "mode='nd' is not implemented yet (tensor path)."
        )
    raise ValueError(f"mode must be '1d', '2d', or 'nd'; got {mode!r}.")


# === 1D direct path ========================================================

def _transition_matrix_1d(
    sys,
    X_axes: List[Array],
    U_points: Array,
    *,
    tol: float,
    return_flat: bool,
):
    """
    1D direct transition tensor construction.

    For each state grid point and each input letter, computes the Gaussian
    CDF on the cell-edge grid and differences to get per-cell transition
    probabilities. Returns the tensor in flat ``(N, N*M)`` or block
    ``(N, N, M)`` form.
    """
    A = np.asarray(sys.A, dtype=float)
    B = np.asarray(sys.B, dtype=float)
    Bw = np.asarray(sys.Bw, dtype=float)
    mu_z = np.asarray(Bw * sys.mu, dtype=float)
    sigma_z = np.asarray(Bw * sys.sigma * Bw.T, dtype=float)

    # X_axes is a list of (1, N) arrays; grab the single axis.
    hx = np.asarray(X_axes, dtype=float)[0]
    hx_arr = hx.ravel()

    # Cell-edge grid: midpoints between successive states, plus half-cell
    # extensions at either end.
    mids = 0.5 * (hx_arr[:-1] + hx_arr[1:])
    left = hx_arr[0] - 0.5 * (hx_arr[1] - hx_arr[0])
    right = hx_arr[-1] + 0.5 * (hx_arr[-1] - hx_arr[-2])
    mx = np.concatenate(([left], mids, [right])).reshape(1, -1)  # (1, N+1)

    XhatSpace = hx
    N = hx.shape[1]
    M = U_points.shape[1]
    P_blocks = np.zeros((N, N, M), dtype=float)

    for k in range(M):
        for idx in range(N):
            xhat = XhatSpace[:, idx]
            cdf_vals = norm.cdf(
                mx,
                A[0] * xhat + B[0] * U_points[:, k] + mu_z[0],
                sigma_z[0],
            )
            cp = np.diff(cdf_vals)
            cp[cp < tol] = 0.0
            P_blocks[idx, :, k] = cp

    # Transpose each action block to match the downstream convention.
    P_blocks_t = np.transpose(P_blocks, (1, 0, 2))

    if return_flat:
        P_flat = np.empty((N, N * M), dtype=P_blocks_t.dtype)
        for k in range(M):
            P_flat[:, k * N:(k + 1) * N] = P_blocks_t[:, :, k]
        return P_flat
    return P_blocks_t


# === 2D separable path =====================================================
def _transition_matrix_2d_separable(
    sys,
    X_axes: List[Array],
    U_points: Array,
    *,
    tol: float,
):
    """
    2D separable SVD construction.

    Steps:
        1. Diagonalise the noise covariance via SVD.
        2. Rotate the state polytope into the noise-eigenframe.
        3. Build a cell-centred grid in the rotated frame.
        4. Build a dense int32 deterministic-shift lookup over
           (state, input). ``target_idx[i, u]`` is the grid cell that
           source cell ``i`` lands in under input ``u``; the sentinel
           value ``n = l1*l2`` marks "image fell outside the rotated
           state polytope" and is resolved to a zero contribution by
           :class:`TransitionProbability2D.mtimes`.
        5. Build per-axis Toeplitz stochastic kernels (zero-mean Gaussian).
        6. Pack into a :class:`TransitionProbability2D` plus auxiliary
           data needed by the abstraction layer.

    Currently hardcoded for ``dim=2``: the index linearisation step uses
    exactly two grid indices.
    """
    dim_i = len(X_axes)
    Bw      = np.asarray(sys.Bw,    dtype=float)
    Sigma_w = np.asarray(sys.sigma, dtype=float)
    mu_w    = np.asarray(sys.mu,    dtype=float).reshape(-1, 1)

    # 1) Diagonalise the noise covariance: Sigma = Uz @ diag(sigma_z) @ Vh.
    Sigma = Bw @ Sigma_w @ Bw.T
    Uz, sigma_z, _ = np.linalg.svd(Sigma, full_matrices=False)
    mu_z = Uz.T @ Bw @ mu_w

    if np.sum(np.abs(mu_z)) > 0:
        warnings.warn(
            "Implementation assumes zero-mean noise; mu_z is non-zero, "
            "the separable decomposition may give incorrect results.",
            UserWarning,
        )

    # 2) Rotate the state polytope into the noise-eigenframe and bound it.
    V_z   = np.asarray(pc.extreme(sys.X))
    V_z2  = (Uz.T @ V_z.T).T
    Z     = pc.qhull(V_z2)
    Ver_Z = np.asarray(pc.extreme(Z))
    Zl    = Ver_Z.min(axis=0)
    Zu    = Ver_Z.max(axis=0)

    # 3) Per-dimension cell-centred grid in the rotated frame.
    l        = np.array([ax.shape[1] for ax in X_axes], dtype=int)
    gridSize = (Zu - Zl) / l
    hz = [
        (Zl[d] + (0.5 + np.arange(l[d])) * gridSize[d]).reshape(1, -1)
        for d in range(dim_i)
    ]
    ZhatSpace = combvec_first_fast(*hz)
    XhatSpace = Uz @ ZhatSpace

    # 4) Deterministic-shift operator as a dense (n, M) int32 lookup.
    #
    # target_idx[i, u] is the linearised grid cell that source cell i
    # lands in under input u. The sentinel value == nXhatSpace marks
    # "image fell outside the rotated state polytope"; mtimes resolves
    # it to a zero contribution by gathering against a 0.0 sink
    # appended to the smoothed value vector.
    #
    # This replaces a sparse (n, n*M) CSR matrix whose construction
    # peaked at O(n*M) memory twice over due to the np.hstack
    # accumulator + COO->CSR conversion. The dense int32 lookup is
    # ~3x smaller and has no transient construction peak.
    nXhatSpace = XhatSpace.shape[1]
    nUhat      = U_points.shape[1]
    target_idx = np.full((nXhatSpace, nUhat), nXhatSpace, dtype=np.int32)

    diag_gridSize_inv = np.diag(1.0 / gridSize)
    Zl_T              = Zl[:, None]
    l_minus_1         = (l - 1)[:, None]

    for k in range(nUhat):
        z_n = Uz.T @ f_det(
            sys, XhatSpace,
            repmat_vector_col(U_points[:, k], n_cols=nXhatSpace),
        )
        z_n_ind = np.floor(diag_gridSize_inv @ (z_n - Zl_T)).astype(np.int32)
        mask    = np.all((z_n_ind >= 0) & (z_n_ind <= l_minus_1), axis=0)
        if not np.any(mask):
            continue

        zi0  = z_n_ind[0, mask][None, :]
        zi1  = z_n_ind[1, mask][None, :]
        lin0 = np.ravel_multi_index((zi0, zi1), dims=tuple(l), order="F").ravel()
        target_idx[mask, k] = lin0.astype(np.int32)

    # 5) Per-dimension stochastic kernels (Toeplitz, zero-mean Gaussian).
    mx2 = [
        ((np.arange(int(li) + 1) - 0.5) * float(gs)).reshape(1, -1)
        for gs, li in zip(gridSize, l)
    ]
    P = []
    for d in range(dim_i):
        edges = np.asarray(mx2[d]).ravel()
        mu    = float(np.asarray(mu_z).ravel()[d])
        sig   = float(np.asarray(sigma_z).ravel()[d])
        cp    = np.diff(norm.cdf(edges, loc=mu, scale=sig))
        cp[cp < tol] = 0.0
        P.append(toeplitz(cp))

    # 6) Pack into a 2D transition object and compute the per-cell box.
    # tP_2d = TransitionProbability2D(l, target_idx, P[0], P[1])
    tP_2d = TransitionProbability2D(l, target_idx=target_idx, Pi0=P[0], Pi1=P[1])

    ff2n_comp_T = (ff2n(dim_i) - 0.5).T
    poly_v      = (np.diag(2 * gridSize) @ ff2n_comp_T).T
    poly_comp   = qhull(np.asarray(poly_v))
    beta        = linear_image_vertices(poly_comp, Uz)

    return tP_2d, hz, XhatSpace, beta, sys, ZhatSpace, Uz
# def _transition_matrix_2d_separable(
#     sys,
#     X_axes: List[Array],
#     U_points: Array,
#     *,
#     tol: float,
# ):
#     """
#     2D separable SVD construction.
#
#     Steps:
#         1. Diagonalise the noise covariance via SVD.
#         2. Rotate the state polytope into the noise-eigenframe.
#         3. Build a cell-centred grid in the rotated frame.
#         4. Build a sparse deterministic-shift operator over (state, input).
#         5. Build per-axis Toeplitz stochastic kernels (zero-mean Gaussian).
#         6. Pack into a TransitionProbability2D plus auxiliary data.
#
#     Currently hardcoded for ``dim=2``: the index linearisation step uses
#     exactly two grid indices.
#     """
#     dim_i = len(X_axes)
#     Bw      = np.asarray(sys.Bw,    dtype=float)
#     Sigma_w = np.asarray(sys.sigma, dtype=float)
#     mu_w    = np.asarray(sys.mu,    dtype=float).reshape(-1, 1)
#
#     # 1) Diagonalise the noise covariance: Sigma = Uz @ diag(sigma_z) @ Vh.
#     Sigma = Bw @ Sigma_w @ Bw.T
#     Uz, sigma_z, _ = np.linalg.svd(Sigma, full_matrices=False)
#     mu_z = Uz.T @ Bw @ mu_w
#
#     if np.sum(np.abs(mu_z)) > 0:
#         warnings.warn(
#             "Implementation assumes zero-mean noise; mu_z is non-zero, "
#             "the separable decomposition may give incorrect results.",
#             UserWarning,
#         )
#
#     # 2) Rotate the state polytope into the noise-eigenframe and bound it.
#     V_z   = np.asarray(pc.extreme(sys.X))
#     V_z2  = (Uz.T @ V_z.T).T
#     Z     = pc.qhull(V_z2)
#     Ver_Z = np.asarray(pc.extreme(Z))
#     Zl    = Ver_Z.min(axis=0)
#     Zu    = Ver_Z.max(axis=0)
#
#     # 3) Per-dimension cell-centred grid in the rotated frame.
#     l        = np.array([ax.shape[1] for ax in X_axes], dtype=int)
#     gridSize = (Zu - Zl) / l
#     hz = [
#         (Zl[d] + (0.5 + np.arange(l[d])) * gridSize[d]).reshape(1, -1)
#         for d in range(dim_i)
#     ]
#     ZhatSpace = combvec_first_fast(*hz)
#     XhatSpace = Uz @ ZhatSpace
#
#     # 4) Deterministic-shift operator (sparse).
#     #
#     # For each (state, input) the deterministic image f_det(x, u) lands in
#     # a single grid cell, so the resulting (N, N*M) matrix has at most
#     # nU*nX nonzeros, all equal to one. Pre-allocate the row/column index
#     # buffers at that maximum, fill in place, and trim at the end. This
#     # replaces the previous np.hstack-in-loop pattern, which had peak
#     # memory ~2x the final size and total cost O(nU^2 * nX).
#     nXhatSpace = XhatSpace.shape[1]
#     nUhat      = U_points.shape[1]
#     max_nnz    = nUhat * nXhatSpace
#
#     rows = np.empty(max_nnz, dtype=np.int32)
#     cols = np.empty(max_nnz, dtype=np.int32)
#     off  = 0
#
#     index_total       = np.arange(nXhatSpace, dtype=np.int32)
#     diag_gridSize_inv = np.diag(1.0 / gridSize)
#     Zl_T              = Zl[:, None]
#     l_minus_1         = (l - 1)[:, None]
#
#     for k in range(nUhat):
#         z_n = Uz.T @ f_det(
#             sys, XhatSpace,
#             repmat_vector_col(U_points[:, k], n_cols=nXhatSpace),
#         )
#         z_n_ind = np.floor(diag_gridSize_inv @ (z_n - Zl_T)).astype(np.int32)
#         mask    = np.all((z_n_ind >= 0) & (z_n_ind <= l_minus_1), axis=0)
#
#         n_k = int(mask.sum())
#         if n_k == 0:
#             continue
#
#         zi0  = z_n_ind[0, mask][None, :]
#         zi1  = z_n_ind[1, mask][None, :]
#         lin0 = np.ravel_multi_index((zi0, zi1), dims=tuple(l), order="F").ravel()
#
#         rows[off:off + n_k] = lin0
#         cols[off:off + n_k] = index_total[mask] + np.int32(k * nXhatSpace)
#         off += n_k
#
#     rows = rows[:off]
#     cols = cols[:off]
#     data = np.ones(off, dtype=np.float32)
#     P_det = coo_matrix(
#         (data, (rows, cols)), shape=(nXhatSpace, nXhatSpace * nUhat)
#     ).tocsr()
#
#     # 5) Per-dimension stochastic kernels (Toeplitz, zero-mean Gaussian).
#     mx2 = [
#         ((np.arange(int(li) + 1) - 0.5) * float(gs)).reshape(1, -1)
#         for gs, li in zip(gridSize, l)
#     ]
#     P = []
#     for d in range(dim_i):
#         edges = np.asarray(mx2[d]).ravel()
#         mu    = float(np.asarray(mu_z).ravel()[d])
#         sig   = float(np.asarray(sigma_z).ravel()[d])
#         cp    = np.diff(norm.cdf(edges, loc=mu, scale=sig))
#         cp[cp < tol] = 0.0
#         P.append(toeplitz(cp))
#
#     # 6) Pack into a 2D transition object and compute the per-cell box.
#     tP_2d = TransitionProbability2D(l, P_det, P[0], P[1])
#
#     ff2n_comp_T = (ff2n(dim_i) - 0.5).T
#     poly_v      = (np.diag(2 * gridSize) @ ff2n_comp_T).T
#     poly_comp   = qhull(np.asarray(poly_v))
#     beta        = linear_image_vertices(poly_comp, Uz)
#
#     return tP_2d, hz, XhatSpace, beta, sys, ZhatSpace, Uz
