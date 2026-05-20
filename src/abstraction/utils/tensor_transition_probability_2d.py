"""
2D transition probability operator for separable stochastic kernels.

The class :class:`TransitionProbability2D` wraps a transition that is
separable along two axes: it reshapes a flat state vector ``V`` of length
``n = l1 * l2`` to an ``(l1, l2)`` grid, applies stochastic kernels
``Pi0`` and ``Pi1`` from each side, flattens back, and applies a
deterministic projection ``P_det``. Used by the tree-based value
iteration to apply the stochastic part of the transition operator
efficiently.

Usage::

    P = TransitionProbability2D(l, P_det, Pi0, Pi1)
    Vnext = V @ P.stoch                # right-hand matmul; returns (n,)

The ``.stoch`` endpoint is the only live operator. Bare ``V @ P`` and
``P @ V`` are intentionally disabled to keep the algorithmic intent
explicit at every call site.

MATLAB equivalent for the operator (kept as algorithmic reference for
readers porting from SysCoRe)::

    VI  = vec(Pi{1}' * reshape(V, [l1 l2]) * Pi{2})
    out = VI' * P_det
"""

from __future__ import annotations

import numpy as np

# SciPy is optional; fall back gracefully if absent.
try:
    from scipy.sparse import isspmatrix  # type: ignore
except Exception:  # pragma: no cover
    def isspmatrix(_):  # noqa: N802
        return False


class _StochView:
    """
    Right-hand matmul view onto a :class:`TransitionProbability2D`.

    Exists so callers can write ``V @ P.stoch`` rather than ``V @ P``;
    this makes the algorithmic intent (stochastic kernel application)
    explicit at the call site. Bare ``V @ P`` is forbidden by
    :class:`TransitionProbability2D` itself.
    """

    __array_priority__ = 10_000

    def __init__(self, parent: "TransitionProbability2D"):
        self.parent = parent

    def __rmatmul__(self, V):
        return self.parent.mtimes(V)

    def __matmul__(self, _):
        return NotImplemented


class TransitionProbability2D:
    """
    Separable 2D stochastic transition probability operator.

    Parameters
    ----------
    l : tuple of int
        Grid shape ``(l1, l2)``. Their product equals the flat state
        dimension ``n``.
    P_det : np.ndarray
        ``(n, n)`` deterministic projection matrix applied after the
        separable stochastic part.
    Pi0 : np.ndarray
        ``(l1, l1)`` stochastic kernel along the first axis.
    Pi1 : np.ndarray
        ``(l2, l2)`` stochastic kernel along the second axis.

    Attributes
    ----------
    l1, l2 : int
        Per-axis grid sizes.
    l : np.ndarray
        Convenience alias: 2-element int array ``[l1, l2]``.
    n : int
        Flat state dimension, ``l1 * l2``.
    dim : int
        Number of grid dimensions (always 2 for this class).
    P_det : np.ndarray or scipy.sparse
        Stored as-is.
    Pi : list
        ``[Pi0, Pi1]`` per-axis kernels.
    stoch : _StochView
        Right-hand matmul endpoint. ``V @ P.stoch`` applies the
        separable 2D stochastic operator and returns a flat ``(n,)``
        result.

    Notes
    -----
    Bare ``V @ P`` and ``P @ V`` are intentionally disabled; use
    ``V @ P.stoch`` to make the operation explicit at the call site.
    """

    # Force numpy's @ to defer to __rmatmul__ rather than broadcasting
    # against the class internals.
    __array_priority__ = 10_000

    def __init__(self, l, P_det, Pi0=None, Pi1=None):
        self.l1, self.l2 = int(l[0]), int(l[1])
        self.l = np.array([self.l1, self.l2], dtype=int)
        self.dim = 2
        self.n = self.l1 * self.l2
        self.P_det = P_det
        self.Pi = [Pi0, Pi1]

        # Operator endpoint (right-hand matmul).
        self.stoch = _StochView(self)

    def mtimes(self, V):
        """
        Apply the separable 2D stochastic operator to ``V``.

            VI  = vec(Pi0.T @ reshape(V, (l1, l2)) @ Pi1)
            out = VI @ P_det

        Used internally by ``_StochView.__rmatmul__``. Named after the
        MATLAB SysCoRe equivalent.

        Parameters
        ----------
        V : np.ndarray
            Flat state vector of length ``n = l1 * l2``.

        Returns
        -------
        np.ndarray
            Flat ``(n,)`` result.
        """
        l1, l2 = self.l1, self.l2
        K0, K1 = self.Pi

        V_grid = np.asarray(V, dtype=float).reshape(l1, l2, order="F")
        VI_grid = (K0.T @ V_grid) @ K1
        VI = VI_grid.reshape(-1, order="F").reshape(1, -1)

        out = VI @ self.P_det
        if isspmatrix(out):
            out = out.toarray()
        return np.asarray(out).ravel(order="F")

    # Bare V @ P and P @ V are forbidden — callers must use V @ P.stoch.
    def __rmatmul__(self, V):
        return NotImplemented

    def __matmul__(self, _):
        return NotImplemented