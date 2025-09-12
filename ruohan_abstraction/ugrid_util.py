import numpy as np
import polytope as pc

def _bbox_from_polytope(U_poly):
    """Return axis-aligned bounding box [lower, upper] from a bounded polytope via its vertices."""
    V = pc.extreme(U_poly)                          # list of vertices
    V = np.array([np.array(v).ravel() for v in V])  # (k, n)
    lower = V.min(axis=0)
    upper = V.max(axis=0)
    return lower, upper

def _filter_points_in_poly(P, pts, tol=1e-9):
    """Keep only points inside polytope P using H-representation A x <= b (+ tolerance)."""
    A = np.asarray(P.A, dtype=float)                # (m, n)
    b = np.asarray(P.b, dtype=float).reshape(-1, 1) # (m, 1)
    ok = (A @ pts.T <= b + tol).all(axis=0)         # (N,)
    return pts[ok]

def make_uniform_grid(U_poly, grid_counts, filter_inside=True, placement='centers'):
    """
    Build an axis-aligned uniform grid over a bounded polytope.

    Parameters
    ----------
    U_poly : polytope.Polytope
        Must be bounded.
    grid_counts : int or sequence of ints
        1D: pass an int; nD: pass an iterable of length n (e.g., [m1, m2, ...]).
    filter_inside : bool
        If True, keep only points that are inside U_poly.
    placement : {'centers', 'endpoints'}
        'centers'   → use cell centers (MATLAB: xl+gs/2:gs:xu)
        'endpoints' → use endpoints (np.linspace)

    Returns
    -------
    axes : list[np.ndarray]
        Per-dimension 1D grid arrays (either centers or endpoints, per `placement`).
    points : np.ndarray
        Shape (N, n). Cartesian product of the axes. If filter_inside=True,
        only points inside U_poly are returned.
    """
    # 1) bounding box
    lower, upper = _bbox_from_polytope(U_poly)
    n = lower.size

    # 2) normalize grid counts
    if np.isscalar(grid_counts):
        if n != 1:
            raise ValueError(f"grid_counts is scalar but U_poly is {n}D; "
                             f"pass a length-{n} iterable instead.")
        counts = [int(grid_counts)]
    else:
        counts = [int(c) for c in list(grid_counts)]
        if len(counts) != n:
            raise ValueError(f"grid_counts length {len(counts)} must match dimension {n}.")

    # 3) build per-axis arrays
    axes = []
    for i in range(n):
        L, U, m = float(lower[i]), float(upper[i]), counts[i]
        if placement == 'endpoints':
            ax = np.linspace(L, U, m)
        elif placement == 'centers':
            step = (U - L) / m
            ax = L + step * (0.5 + np.arange(m))
        else:
            raise ValueError("placement must be 'centers' or 'endpoints'")
        axes.append(ax)

    # 4) Cartesian product grid -> (N, n)
    if n == 1:
        points = axes[0].reshape(-1, 1)
    else:
        meshes = np.meshgrid(*axes, indexing="ij")
        points = np.stack([m.reshape(-1) for m in meshes], axis=1)

    # 5) optionally filter to points inside the polytope
    if filter_inside:
        points = _filter_points_in_poly(U_poly, points)

    return axes, points
