"""
Tensor and vector helpers for the abstraction layer.

Small utilities used during transition tensor construction and reachable-
set computation. Matches the MATLAB-style "row = dimension, column = grid
index" convention used by :mod:`abstraction_grid_helper`.
"""

from itertools import product

import numpy as np
import polytope as pc

Array = np.ndarray


def linear_image_vertices(P, M: Array):
    """
    Apply a linear map ``M`` to polytope ``P`` and return the convex hull
    of the image.

    Computed by enumerating the vertices of ``P``, mapping each through
    ``M``, and taking the convex hull. Sound only for bounded polytopes.
    """
    V = np.array([np.asarray(v).ravel() for v in pc.extreme(P)])
    V_img = (M @ V.T).T
    return pc.qhull(V_img)


def ff2n(dim: int) -> Array:
    """
    MATLAB-style ``ff2n``: full-factorial design over ``dim`` binary factors.

    Returns a ``(2**dim, dim)`` array of all 0/1 combinations -- i.e. the
    vertices of the unit hypercube indexed in lexicographic order.
    """
    return np.array(list(product([0, 1], repeat=dim)), dtype=int)


def combvec_first_fast(*axes: Array) -> Array:
    """
    MATLAB-style Cartesian product of 1D axes; first axis varies fastest.

    Returns an ``(n, N)`` array whose columns enumerate the Cartesian
    product of the given axes, with the first axis cycling fastest.
    Matches MATLAB's ``combvec`` ordering, in contrast to
    :func:`abstraction_shape_helper.cartesian_from_axes`, which returns
    the transposed shape ``(N, n)`` with the last axis varying fastest.
    """
    axes_arr = [np.asarray(a).ravel() for a in axes]
    grids = np.meshgrid(*axes_arr, indexing="ij")
    return np.vstack([g.reshape(-1, order="F") for g in grids])


def f_det(sys, x: Array, u: Array) -> Array:
    """
    Deterministic next-state map of a linear system: ``A x + B u``.

    ``sys`` must expose attributes ``A`` and ``B`` (e.g., a ``LinModel``).
    """
    return sys.A @ x + sys.B @ u


def repmat_vector_col(vec: Array, n_cols: int) -> Array:
    """
    MATLAB-style ``repmat`` for a column vector: tile ``vec`` across
    ``n_cols`` columns.

    Parameters
    ----------
    vec : np.ndarray
        Shape ``(m,)`` or ``(m, 1)``. Other shapes raise.
    n_cols : int
        Number of times to replicate the column.

    Returns
    -------
    np.ndarray
        Shape ``(m, n_cols)``. A contiguous copy, not a broadcast view.
    """
    v = np.asarray(vec)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    elif v.ndim == 2 and v.shape[1] != 1:
        raise ValueError("vec must be a single column: (m,) or (m, 1).")
    m = v.shape[0]
    return np.broadcast_to(v, (m, n_cols)).copy()