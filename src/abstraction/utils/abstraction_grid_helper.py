"""
Grid utilities for axis-aligned uniform grids over bounded polytopes.

Used by the abstraction layer to enumerate state-space grid points within
a polytopic safe set. Supports linear (centers / endpoints) and log-spaced
placements; the log case handles strictly positive, strictly negative, and
zero-crossing intervals.

The public entry point is :func:`make_uniform_grid`. The returned axes and
points use the MATLAB-style "row = dimension, column = grid index"
convention to match the rest of the FMTens pipeline.
"""

from typing import List, Tuple, Union

import numpy as np
import polytope as pc

Array = np.ndarray


# === bounding-box and point-filtering helpers ==============================

def _bbox_from_polytope(poly_or_bounds) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return per-dimension (lower, upper) bounds for a polytope or bound spec.

    Accepts:
        - :class:`polytope.Polytope`  (queried via :func:`polytope.extreme`)
        - ``(lower, upper)`` tuple or list of per-dimension bounds
        - 1D ndarray of length 2 -- interpreted as scalar 1D bounds
        - 2 x n ndarray             -- row 0 lower, row 1 upper
    """
    if poly_or_bounds is None:
        raise ValueError("Expected a Polytope or (lower, upper) bounds, got None.")

    if isinstance(poly_or_bounds, (tuple, list)):
        lower = np.asarray(poly_or_bounds[0], dtype=float).ravel()
        upper = np.asarray(poly_or_bounds[1], dtype=float).ravel()
        return lower, upper

    arr = np.asarray(poly_or_bounds)

    # 1D array with two entries: scalar bounds for 1D
    if arr.ndim == 1 and arr.size == 2:
        return np.array([float(arr[0])]), np.array([float(arr[1])])

    # 2 x n array: row 0 lower, row 1 upper
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr[0].astype(float).ravel(), arr[1].astype(float).ravel()

    # Fall through: polytope-like object
    V = np.asarray([np.asarray(v).ravel() for v in pc.extreme(poly_or_bounds)])
    return V.min(axis=0), V.max(axis=0)


def _filter_points_in_poly(P, pts: Array, tol: float = 1e-9) -> Array:
    """
    Keep only the points inside polytope P (via the half-space form
    ``A x <= b``) or within axis-aligned bounds. P may be any input
    accepted by :func:`_bbox_from_polytope`.
    """
    if hasattr(P, "A") and hasattr(P, "b"):
        A = np.asarray(P.A, dtype=float)
        b = np.asarray(P.b, dtype=float).reshape(-1, 1)
        ok = (A @ pts.T <= b + tol).all(axis=0)
        return pts[ok]

    lower, upper = _bbox_from_polytope(P)
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    ok = np.all((pts >= lower) & (pts <= upper + tol), axis=1)
    return pts[ok]


# === log-spaced axis helper ================================================

def _log_spaced_axis(L: float, U: float, m: int) -> np.ndarray:
    """
    Build an m-point log-spaced axis over [L, U].

    Three cases by sign of the interval:
        - L, U > 0  : standard positive logspace.
        - L, U < 0  : logspace on magnitudes, then negate and sort ascending.
        - L < 0 < U : symmetric log density on each side of zero; no exact
                      zero is included. Points are split roughly half-half,
                      with at least one point per side.
    """
    if not (U > L):
        raise ValueError(f"Degenerate bounds [{L}, {U}].")

    if L > 0 and U > 0:
        return np.logspace(np.log10(L), np.log10(U), m)

    if L < 0 and U < 0:
        mags = np.logspace(np.log10(abs(U)), np.log10(abs(L)), m)
        return np.sort(-mags)

    # Range crosses zero: split between negative and positive sides.
    tiny = 1e-12
    eps = max(
        tiny,
        min(abs(L) if L != 0 else np.inf, abs(U) if U != 0 else np.inf) * 1e-9,
    )

    m_pos = max(1, int(np.ceil(m / 2)))
    m_neg = max(1, m - m_pos)

    start_pos = max(eps, min(U, 1.0) * eps)
    pos = np.logspace(np.log10(start_pos), np.log10(U), m_pos)

    start_neg_mag = max(eps, min(abs(L), 1.0) * eps)
    neg_mag = np.logspace(np.log10(start_neg_mag), np.log10(abs(L)), m_neg)
    neg = -neg_mag[::-1]

    return np.concatenate([neg, pos])


# === uniform grid (public API) =============================================

def make_uniform_grid(
    U_poly,
    grid_counts: Union[int, List[int], Tuple[int, ...]],
    *,
    filter_inside: bool = True,
    placement: str = "centers",
) -> Tuple[List[Array], Array]:
    """
    Build an axis-aligned uniform grid over a bounded polytope.

    Parameters
    ----------
    U_poly : polytope.Polytope or bound spec
        Bounded region to grid. Accepts any form recognised by
        :func:`_bbox_from_polytope`.
    grid_counts : int or iterable of ints
        Number of points per dimension. A scalar is allowed only for 1D.
    filter_inside : bool, optional
        Keep only the grid points lying inside the polytope (or bounds).
        Defaults to True.
    placement : {'centers', 'endpoints', 'log'}, optional
        Strategy for placing points on each axis:
            - 'centers'   : cell-centre placement; m points uniformly spaced
                            in [L + step/2, U - step/2] with step = (U-L)/m.
            - 'endpoints' : m points including both endpoints (np.linspace).
            - 'log'       : log-spaced endpoints; see :func:`_log_spaced_axis`.
        Defaults to 'centers'.

    Returns
    -------
    axes : list of np.ndarray
        Per-dimension axes of shape ``(1, m_i)`` -- row vectors, in
        MATLAB-style "row = dimension, column = grid index" form.
    points : np.ndarray
        Cartesian product grid of shape ``(n, N)`` -- same convention,
        rows are dimensions and columns are grid points. Filtered to
        ``U_poly`` if ``filter_inside`` is True.
    """
    lower, upper = _bbox_from_polytope(U_poly)
    n = lower.size

    if np.isscalar(grid_counts):
        if n != 1:
            raise ValueError(
                f"grid_counts is scalar but polytope is {n}D; pass a length-{n} iterable."
            )
        counts = [int(grid_counts)]
    else:
        counts = [int(c) for c in list(grid_counts)]
        if len(counts) != n:
            raise ValueError(
                f"grid_counts length {len(counts)} must match dimension {n}."
            )

    # Per-axis 1D arrays in the chosen placement
    axes_1d: List[np.ndarray] = []
    for i in range(n):
        L, U, m = float(lower[i]), float(upper[i]), counts[i]
        if placement == "centers":
            step = (U - L) / m
            ax = L + step * (0.5 + np.arange(m))
        elif placement == "endpoints":
            ax = np.linspace(L, U, m)
        elif placement == "log":
            ax = _log_spaced_axis(L, U, m)
        else:
            raise ValueError(
                f"placement must be 'centers', 'endpoints', or 'log'; got '{placement}'."
            )
        axes_1d.append(np.asarray(ax, dtype=float))

    # Cartesian product -> (N, n)
    if n == 1:
        points = axes_1d[0].reshape(-1, 1)
    else:
        meshes = np.meshgrid(*axes_1d, indexing="ij")
        points = np.stack([m.reshape(-1) for m in meshes], axis=1)

    if filter_inside:
        points = _filter_points_in_poly(U_poly, points)

    # Switch to MATLAB-style convention: rows = dimensions, columns = grid points
    points = points.T
    axes = [ax.reshape(1, -1) for ax in axes_1d]

    return axes, points