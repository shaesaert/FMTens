# -*- coding: utf-8 -*-
# Separable Gaussian transition on a grid (nD), robust to U_points shapes.

import numpy as np
from math import sqrt
import polytope as pc
from scipy.special import erf as _erf


def _normal_cdf(x, m, s):
    """Gaussian CDF with mean m and std s; supports NumPy broadcasting."""
    z = (x - m) / (s * np.sqrt(2.0))
    return 0.5 * (1.0 + _erf(z))


def _bbox_from_poly(P):
    """Return per-dimension bounding box [lower, upper] computed from polytope vertices."""
    V = pc.extreme(P)                              # list of vertices
    V = np.array([np.array(v).ravel() for v in V]) # (nv, n)
    lower = V.min(axis=0)
    upper = V.max(axis=0)
    return lower, upper


def _edges_from_axes(X_axes, X_poly):
    """
    From per-dimension axis arrays -> per-dimension edge arrays of length (l_d + 1).
    Edges use domain bounds at both ends and midpoints in between.
    If an axis has a single point, create a tiny 2-edge bin around it.
    """
    lower, upper = _bbox_from_poly(X_poly)
    edges = []
    for d, ax in enumerate(X_axes):
        ax = np.asarray(ax, dtype=float).ravel()
        L, U = float(lower[d]), float(upper[d])
        if ax.size == 1:
            eps = 1e-12
            e = np.array([ax[0] - eps, ax[0] + eps], dtype=float)
        else:
            mid = 0.5 * (ax[:-1] + ax[1:])
            e = np.concatenate([[L], mid, [U]]).astype(float)
        edges.append(e)
    return edges


def _diag_std_from_noise(Bw, Sigma_w):
    """
    Std of x^+ contributed by noise: sqrt(diag(Bw * Sigma_w * Bw^T)).
    Treat dimensions independently (off-diagonals are ignored in the final result).
    """
    Sigma_x = Bw @ Sigma_w @ Bw.T
    var = np.clip(np.diag(Sigma_x), a_min=0.0, a_max=None)
    std = np.sqrt(var)
    return std  # (n,)


def transition_matrix_nd_separable(
    sys, X_axes, U_points, X_poly, tol=1e-15, renormalize=True, return_flat=True
):
    """
    Build an n-dimensional (dimension-wise independent) Gaussian-noise transition matrix.
    This matches the MATLAB cp/Kronecker implementation you used.

    Parameters
    ----------
    sys : LinModel
        Contains A(n,n), B(n,m), Bw(n,r), mu(r,1), sigma(r,r).
    X_axes : list[np.ndarray]
        Length n; one 1D array per state dimension (grid axis). Typically cell centers.
    U_points : array-like
        Discrete input set. Accepted shapes: (M,), (1, M), (M, 1), (M, m), etc.
        Will be reshaped to (M, m) internally.
    X_poly : polytope.Polytope
        State domain polytope used to construct bin edges.
    tol : float
        Entries smaller than this are set to 0.
    renormalize : bool
        If True, renormalize each (row, action) distribution to sum to 1
        after discarding probability mass outside the domain.
    return_flat : bool
        If True, return shape (N, N*M); else return (N, N, M).

    Returns
    -------
    P : np.ndarray
        - If return_flat=True : shape (N, N*M); the k-th action occupies columns [k*N : (k+1)*N)
        - If return_flat=False: shape (N, N, M)
    """
    # --- unpack and shapes ---
    A = np.asarray(sys.A, dtype=float)
    B = np.asarray(sys.B, dtype=float)
    Bw = np.asarray(sys.Bw, dtype=float)
    mu_w = np.asarray(sys.mu, dtype=float).reshape(-1, 1)    # (r,1)
    Sigma_w = np.asarray(sys.sigma, dtype=float)             # (r,r)

    n = A.shape[0]                          # state dimension
    X_axes = [np.asarray(ax, dtype=float).ravel() for ax in X_axes]
    l = [ax.size for ax in X_axes]          # bins per dimension
    N = int(np.prod(l))                     # total bins / states

    # --- robust U_points shape handling ---
    m = B.shape[1]                          # input dimension
    U_points = np.asarray(U_points, dtype=float)
    if U_points.ndim == 1:
        U_points = U_points.reshape(-1, m)  # (M, m)
    else:
        if U_points.shape[1] != m:
            if U_points.shape[0] == m:
                U_points = U_points.T       # (M, m)
            else:
                U_points = U_points.reshape(-1, m)
    M = U_points.shape[0]                   # number of inputs

    # --- edges per dimension ---
    edges = _edges_from_axes(X_axes, X_poly)   # list length n, each (l_d+1,)

    # --- noise std per dimension ---
    std = _diag_std_from_noise(Bw, Sigma_w)    # (n,)
    if np.any(std == 0.0):
        raise ValueError("Some dimensions have zero noise std; handle degenerate dims separately.")

    # --- build XhatSpace: (N, n) ---
    meshes = np.meshgrid(*X_axes, indexing="ij")
    XhatSpace = np.stack([m_.reshape(-1) for m_ in meshes], axis=1)

    # allocate
    if return_flat:
        P = np.zeros((N, N * M), dtype=float)
    else:
        P = np.zeros((N, N, M), dtype=float)

    # precompute terms independent of x_i
    Bu = U_points @ B.T             # (M, n)
    w_mean = (Bw @ mu_w).ravel()    # (n,)

    # main loops
    for i in range(N):
        xi = XhatSpace[i, :]                  # (n,)
        Axi = (A @ xi).ravel()                # (n,)
        means_all = Bu + Axi + w_mean         # (M, n)

        for k in range(M):
            m_vec = means_all[k, :]           # (n,)
            probs_per_dim = []
            for d in range(n):
                e = edges[d]
                cdf_r = _normal_cdf(e[1:], m_vec[d], std[d])
                cdf_l = _normal_cdf(e[:-1], m_vec[d], std[d])
                cp = (cdf_r - cdf_l)
                cp[cp < tol] = 0.0
                probs_per_dim.append(cp)

            # Kronecker product across dimensions -> length N
            pij = probs_per_dim[0]
            for d in range(1, n):
                pij = np.kron(pij, probs_per_dim[d])

            if renormalize:
                s = pij.sum() + 1e-15
                pij = pij / s

            if return_flat:
                P[i, k * N : (k + 1) * N] = pij
            else:
                P[i, :, k] = pij

    return P
