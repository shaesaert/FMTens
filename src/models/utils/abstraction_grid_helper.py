# =====================
# Grid utilities
# =====================
import numpy as np
import polytope as pc
from typing import List, Tuple, Union

Array = np.ndarray

def _bbox_from_polytope(poly_or_bounds) -> Tuple[np.ndarray, np.ndarray]:
    if poly_or_bounds is None:
        raise ValueError("Expected a Polytope or (lower, upper) bounds, got None.")

    # tuple/list of bounds
    if isinstance(poly_or_bounds, (tuple, list)):
        lower = np.asarray(poly_or_bounds[0], dtype=float).ravel()
        upper = np.asarray(poly_or_bounds[1], dtype=float).ravel()
        return lower, upper

    arr = np.asarray(poly_or_bounds)

    # NEW: 1-D array with two entries -> treat as scalar bounds for 1-D
    if arr.ndim == 1 and arr.size == 2:
        lower = np.array([float(arr[0])])
        upper = np.array([float(arr[1])])
        return lower, upper

    # 2×n array -> row 0 lower, row 1 upper
    if arr.ndim == 2 and arr.shape[0] == 2:
        lower = np.asarray(arr[0], dtype=float).ravel()
        upper = np.asarray(arr[1], dtype=float).ravel()
        return lower, upper

    # Otherwise assume a polytope object
    V = np.asarray([np.asarray(v).ravel() for v in pc.extreme(poly_or_bounds)])
    lower = V.min(axis=0)
    upper = V.max(axis=0)
    return lower, upper



def _filter_points_in_poly(P, pts: Array, tol: float = 1e-9) -> Array:
    """Keep only points inside polytope P (A x <= b) or within axis-aligned bounds."""
    # Polytope-like objects have A, b
    if hasattr(P, "A") and hasattr(P, "b"):
        A = np.asarray(P.A, dtype=float)
        b = np.asarray(P.b, dtype=float).reshape(-1, 1)
        ok = (A @ pts.T <= b + tol).all(axis=0)
        return pts[ok]

    # Otherwise treat P as bounds (tuple/list/array)
    lower, upper = _bbox_from_polytope(P)  # returns (n,) arrays
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()

    # pts is (N, n); works for n=1 too
    ok = np.all((pts >= lower) & (pts <= upper + tol), axis=1)
    return pts[ok]



def make_uniform_grid(U_poly,
                      grid_counts: Union[int, List[int], Tuple[int, ...]],
                      *,
                      filter_inside: bool = True,
                      placement: str = 'centers') -> Tuple[List[Array], Array]:
    """
    Build an axis-aligned uniform grid over a bounded polytope.

    Parameters
    ----------
    U_poly : polytope.Polytope (bounded)
    grid_counts : int or iterable of ints (per dimension)
    filter_inside : keep only points inside polytope
    placement : {'centers','endpoints','log'}
                'log' creates log-spaced *endpoints* (requires strictly positive bounds).

    Returns
    -------
    axes : list[np.ndarray] — per-dimension 1D arrays
    points : np.ndarray — (N, n) Cartesian product (filtered if requested)
    """
    lower, upper = _bbox_from_polytope(U_poly)
    n = lower.size

    # normalize grid counts
    if np.isscalar(grid_counts):
        if n != 1:
            raise ValueError(
                f"grid_counts is scalar but polytope is {n}D; pass a length-{n} iterable."
            )
        counts = [int(grid_counts)]
    else:
        counts = [int(c) for c in list(grid_counts)]
        if len(counts) != n:
            raise ValueError(f"grid_counts length {len(counts)} must match dimension {n}.")

    # per-axis arrays
    axes = []
    for i in range(n):
        L, U, m = float(lower[i]), float(upper[i]), counts[i]
        if placement == 'endpoints':
            ax = np.linspace(L, U, m)
        elif placement == 'centers':
            step = (U - L) / m
            ax = L + step * (0.5 + np.arange(m))
        elif placement == 'log':
            # Log-like spacing for any sign pattern:
            # - L>0,U>0: standard logspace(L..U)
            # - L<0,U<0: logspace on |bounds| then negate
            # - L<0<U:   symmetric log density around 0 (no exact 0 included)

            # guard for degenerate interval
            if not (U > L):
                raise ValueError(f"Degenerate bounds [{L}, {U}] on dim {i}.")

            tiny = 1e-12
            # scale epsilon to the interval size so we don't collapse points
            eps = max(tiny, min(abs(L) if L != 0 else np.inf, abs(U) if U != 0 else np.inf) * 1e-9)

            if L > 0 and U > 0:
                # standard positive logspace, include endpoints
                ax = np.logspace(np.log10(L), np.log10(U), m)

            elif U < 0 and L < 0:
                # fully negative range: build on magnitudes and negate
                # use endpoints |U| (closest to 0) .. |L| (largest magnitude), then negate & sort ascending
                mags = np.logspace(np.log10(abs(U)), np.log10(abs(L)), m)
                ax = -mags
                ax = np.sort(ax)  # from most negative (L) to least negative (U)

            else:
                # range crosses zero: split points between negative and positive sides
                # allocate roughly half/half, ensuring both sides get at least 1 point
                m_pos = max(1, int(np.ceil(m / 2)))
                m_neg = max(1, m - m_pos)

                # positive side: from small epsilon up to U (avoid exact 0)
                # ensure start < stop in logspace
                start_pos = max(eps, min(U, 1.0) * eps)
                pos = np.logspace(np.log10(start_pos), np.log10(U), m_pos)

                # negative side: from L up to small magnitude (avoid exact 0)
                # build magnitudes then negate and put in ascending order
                start_neg_mag = max(eps, min(abs(L), 1.0) * eps)
                neg_mag = np.logspace(np.log10(start_neg_mag), np.log10(abs(L)), m_neg)
                neg = -neg_mag[::-1]  # from L (most negative) towards -small

                ax = np.concatenate([neg, pos])

            ax = np.asarray(ax, dtype=float)

        else:
            raise ValueError("placement must be 'centers' or 'endpoints'")
        axes.append(ax)

    # Cartesian product grid -> (N, n)
    if n == 1:
        points = axes[0].reshape(-1, 1)
    else:
        meshes = np.meshgrid(*axes, indexing="ij")
        points = np.stack([m.reshape(-1) for m in meshes], axis=1)

    if filter_inside:
        points = _filter_points_in_poly(U_poly, points)

    # row: dim_x, column: grid number, matching matlab style
    points = points.T
    for i, ax in enumerate(axes):
        axes[i] = np.asarray(ax).reshape(1, -1)  # shape: (len(ax), 1)

    return axes, points