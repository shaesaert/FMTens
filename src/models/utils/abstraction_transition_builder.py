from __future__ import annotations
from typing import List, Literal

import numpy as np
import polytope as pc
from polytope import qhull
import warnings
from scipy.sparse import coo_matrix
from scipy.stats import norm
from scipy.linalg import toeplitz

from .abstraction_tensor_helper import (
    linear_image_vertices,
    ff2n,
    combvec_first_fast,
    f_det,
    repmat_vector_col,
)

Array = np.ndarray

def transition_matrix_nd_separable_impl(
    sys,
    X_axes: List[Array],
    U_points: Array,
    X_poly,
    tol: float = 1e-19,
    renormalize: bool = True,      # kept for signature compatibility
    return_flat: bool = True,      # kept for signature compatibility
    mode: Literal['1d', '2d', 'nd'] = '1d',
):
    """
    Moved implementation of the transition builder.
    API-compatible with MDPModel.transition_matrix_nd_separable.
    """

    # ---------- 1D path ----------
    if mode == '1d':
        A = np.asarray(sys.A, dtype=float)
        B = np.asarray(sys.B, dtype=float)
        Bw = np.asarray(sys.Bw, dtype=float)
        mu_z = np.asarray(Bw * sys.mu, dtype=float)
        sigma_z = np.asarray(Bw * sys.sigma * Bw.T, dtype=float)

        # X_axes is a list of 1×N arrays; grab the single axis
        hx = np.asarray(X_axes, dtype=float)[0]
        hx_arr = np.asarray(hx, dtype=float).ravel()

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
                # single dimension
                cdf_vals = norm.cdf(
                    mx,
                    A[0] * xhat + B[0] * U_points[:, k] + mu_z[0],
                    sigma_z[0],
                )
                cpdiff = np.diff(cdf_vals)
                cpdiff[cpdiff < tol] = 0.0
                P_blocks[idx, :, k] = cpdiff

        # transpose each action block (to match original)
        P_blocks_t = np.transpose(P_blocks, (1, 0, 2))  # (N,N,M)

        if return_flat:
            P_flat = np.empty((N, N * M), dtype=P_blocks_t.dtype)
            for k in range(M):
                P_flat[:, k * N:(k + 1) * N] = P_blocks_t[:, :, k]
            return P_flat
        return P_blocks_t

    # ---------- ≥2D tensor path ----------
    dim_i = len(X_axes)
    Bw = np.asarray(sys.Bw, dtype=float)
    Sigma_w = np.asarray(sys.sigma, dtype=float)
    mu_w = np.asarray(sys.mu, dtype=float).reshape(-1, 1)

    Sigma = Bw @ Sigma_w @ Bw.T
    Uz, s, Vh = np.linalg.svd(Sigma, full_matrices=False)
    Sz = np.diag(s)
    sigma_z = np.diag(Sz)
    mu_z = Uz.T @ Bw @ mu_w

    if np.sum(np.abs(mu_z)) > 0:
        warnings.warn("Implementation does not hold for mu not equal to zero", UserWarning)

    V_z = np.asarray(pc.extreme(sys.X))
    V_z2 = (Uz.T @ V_z.T).T
    Z = pc.qhull(V_z2)

    Ver_Z = np.asarray(pc.extreme(Z))
    Zl = Ver_Z.min(axis=0)
    Zu = Ver_Z.max(axis=0)

    l = np.array([ax.shape[1] for ax in X_axes], dtype=int)
    gridSize = (Zu - Zl) / l
    hz = [(Zl[d] + (0.5 + np.arange(l[d])) * gridSize[d]).reshape(1, -1) for d in range(dim_i)]
    ZhatSpace = combvec_first_fast(*hz)
    XhatSpace = Uz @ ZhatSpace

    # Deterministic shift operator
    nXhatSpace = XhatSpace.shape[1]
    nUhat = U_points.shape[1]
    i_indices = np.empty((1, 0), dtype=int)
    j_indices = np.empty((1, 0), dtype=int)
    index_total = np.arange(ZhatSpace.shape[1], dtype=int)[None, :]

    gridSize_inv = 1.0 / gridSize
    diag_gridSize_inv = np.diag(gridSize_inv)
    Zl_T = Zl[:, None]
    for k in range(nUhat):
        z_n = Uz.T @ f_det(sys, XhatSpace, repmat_vector_col(U_points[:, k], n_cols=nXhatSpace))
        diff_zn = z_n - Zl_T
        z_n_ind = np.floor(diag_gridSize_inv @ diff_zn).astype(int)
        valid = (z_n_ind >= 0) & (z_n_ind <= (l - 1)[:, None])
        mask = np.all(valid, axis=0)

        zi0 = z_n_ind[0, mask][None, :]
        zi1 = z_n_ind[1, mask][None, :]
        lin0 = np.ravel_multi_index((zi0, zi1), dims=tuple(l), order='F')

        i_indices = np.hstack((i_indices, lin0))
        cols = index_total.ravel()[mask] + k * nXhatSpace
        j_indices = np.hstack((j_indices, cols.reshape(1, -1)))

    rows = np.asarray(i_indices, dtype=int).ravel()
    cols = np.asarray(j_indices, dtype=int).ravel()
    data = np.ones(rows.size, dtype=float)
    P_det = coo_matrix((data, (rows, cols)), shape=(nXhatSpace, nXhatSpace * nUhat)).tocsr()

    # Stochastic per-dim kernels
    mx2 = [((np.arange(int(li) + 1) - 0.5) * float(gs)).reshape(1, -1) for gs, li in zip(gridSize, l)]
    P = []
    for d in range(dim_i):
        edges = np.asarray(mx2[d]).ravel()
        mu = float(np.asarray(mu_z).ravel()[d])
        sig = float(np.asarray(sigma_z).ravel()[d])
        cp = np.diff(norm.cdf(edges, loc=mu, scale=sig))
        cp[cp < tol] = 0.0
        P_d = toeplitz(cp)
        P.append(P_d)

    # Build 2D transition object
    from .tensor_transition_probability_2d import TransitionProbability2D
    if mode == '2d':
        tP_2d = TransitionProbability2D(l, P_det, P[0], P[1])
    else:
        raise NotImplementedError("compute_P='n>2 d' not implemented yet (tensor path).")

    ff2n_comp_T = (ff2n(dim_i) - 0.5).T
    poly_v = (np.diag(2 * gridSize) @ ff2n_comp_T).T
    poly_comp = qhull(np.asarray(poly_v))
    beta = linear_image_vertices(poly_comp, Uz)

    return tP_2d, hz, XhatSpace, beta, sys, ZhatSpace, Uz
