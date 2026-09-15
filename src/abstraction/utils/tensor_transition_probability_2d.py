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

# remove the _LegacyShapeProxy class — no longer needed


class TransitionProbability2D:
    """
    Separable 2D stochastic transition operator with two supported
    deterministic-shift representations:

      - ``target_idx``: dense ``(n, M)`` int32 lookup, used for the raw
        abstraction. Memory-efficient; ``mtimes`` applies it as a gather.
      - ``P_det``: ``(n, M*n)`` or ``(n, n)`` matrix, used after policy
        projection collapses the input axis. ``mtimes`` applies it as a
        standard matmul.

    Exactly one of the two must be provided at construction. In
    ``target_idx`` mode the legacy ``(n, n*M)`` sparse ``P_det`` is built
    lazily on first attribute access and cached on the instance; code
    that wants to preserve the abstraction-side memory savings should
    use ``target_idx`` directly via :meth:`gather_at` rather than
    accessing ``P_det``.
    """

    __array_priority__ = 10_000

    def __init__(self, l, P_det=None, Pi0=None, Pi1=None, *, target_idx=None):
        if (P_det is None) == (target_idx is None):
            raise ValueError("Provide exactly one of `P_det` or `target_idx`.")

        self.l1, self.l2 = int(l[0]), int(l[1])
        self.l   = np.array([self.l1, self.l2], dtype=int)
        self.dim = 2
        self.n   = self.l1 * self.l2

        if target_idx is not None:
            self.target_idx = np.asarray(target_idx, dtype=np.int32)
            self.n_inputs   = int(self.target_idx.shape[1])
            self._P_det     = None              # lazy
            self._mode      = "gather"
        else:
            self.target_idx = None
            self._P_det     = P_det
            cols = P_det.shape[1]
            self.n_inputs   = 1 if cols == self.n else cols // self.n
            self._mode      = "matmul"

        self.Pi    = [Pi0, Pi1]
        self.stoch = _StochView(self)

    @property
    def P_det(self):
        """
        Legacy ``(n, n*M)`` sparse deterministic-shift matrix.

        In ``gather`` mode this is materialised on first access from
        ``target_idx`` and cached. Doing so allocates roughly
        ``12 * n * M`` bytes; at large grids the gather-based code paths
        (:meth:`mtimes`, :meth:`gather_at`, :func:`Pc` on a gather-mode
        operator) avoid this allocation entirely.
        """
        if self._P_det is None and self.target_idx is not None:
            self._P_det = self._build_legacy_sparse()
        return self._P_det

    @P_det.setter
    def P_det(self, value):
        self._P_det = value

    def _build_legacy_sparse(self):
        """Materialise the (n, n*M) sparse P_det from target_idx."""
        from scipy import sparse
        n, M = self.n, self.n_inputs
        sources = np.tile(np.arange(n, dtype=np.int64), M)            # (n*M,)
        ks      = np.repeat(np.arange(M, dtype=np.int64), n)          # (n*M,)
        targets = self.target_idx.ravel(order="F").astype(np.int64)   # (n*M,)
        valid   = targets < n
        rows    = targets[valid]
        cols    = ks[valid] * n + sources[valid]
        data    = np.ones(int(valid.sum()), dtype=np.float64)
        return sparse.coo_matrix(
            (data, (rows, cols)), shape=(n, n * M)
        ).tocsr()

    def smooth(self, V):
        K0, K1  = self.Pi
        V_grid  = np.asarray(V, dtype=float).reshape(self.l1, self.l2, order="F")
        VI_grid = (K0.T @ V_grid) @ K1
        return VI_grid.reshape(-1, order="F")

    def mtimes(self, V):
        VI = self.smooth(V)

        if self._mode == "gather":
            VI_with_sink = np.empty(self.n + 1, dtype=VI.dtype)
            VI_with_sink[: self.n] = VI
            VI_with_sink[self.n]   = 0.0
            return VI_with_sink[self.target_idx].reshape(-1, order="F")

        out = VI.reshape(1, -1) @ self._P_det
        if hasattr(out, "toarray"):
            out = out.toarray()
        return np.asarray(out).ravel(order="F")

    def gather_at(self, VI, u):
        if self._mode != "gather":
            raise RuntimeError("gather_at requires the target_idx representation")
        idx   = self.target_idx[:, u]
        valid = idx < self.n
        out   = np.zeros(self.n, dtype=VI.dtype)
        out[valid] = VI[idx[valid]]
        return out

    def __rmatmul__(self, V): return NotImplemented
    def __matmul__(self, _):  return NotImplemented

# version 1
# class _LegacyShapeProxy:
#     """Carries the legacy (n, n*M) shape for code that derives M from
#     P_det.shape[1] // n. No data; do not matmul against it."""
#     def __init__(self, n, n_inputs):
#         self.shape = (n, n * n_inputs)
#
#
# class TransitionProbability2D:
#     """
#     Separable 2D stochastic transition operator with two supported
#     deterministic-shift representations:
#
#       - ``target_idx``: dense ``(n, M)`` int32 lookup, used for the raw
#         abstraction. Memory-efficient; ``mtimes`` applies it as a gather.
#       - ``P_det``: ``(n, M*n)`` or ``(n, n)`` matrix (sparse or dense),
#         used after policy projection collapses the input axis. ``mtimes``
#         applies it as a standard matmul.
#
#     Exactly one of the two must be provided. The class duck-types as a
#     ``TransitionProbability2D`` to existing callers regardless of which
#     one was used at construction.
#     """
#
#     __array_priority__ = 10_000
#
#     def __init__(self, l, P_det=None, Pi0=None, Pi1=None, *, target_idx=None):
#         if (P_det is None) == (target_idx is None):
#             raise ValueError(
#                 "Provide exactly one of `P_det` or `target_idx`."
#             )
#
#         self.l1, self.l2 = int(l[0]), int(l[1])
#         self.l   = np.array([self.l1, self.l2], dtype=int)
#         self.dim = 2
#         self.n   = self.l1 * self.l2
#
#         if target_idx is not None:
#             # Raw-abstraction representation.
#             self.target_idx = np.asarray(target_idx, dtype=np.int32)
#             self.n_inputs   = int(self.target_idx.shape[1])
#             self.P_det      = _LegacyShapeProxy(self.n, self.n_inputs)
#             self._mode      = "gather"
#         else:
#             # Closed-loop or legacy sparse representation.
#             self.target_idx = None
#             self.P_det      = P_det
#             cols            = P_det.shape[1]
#             self.n_inputs   = 1 if cols == self.n else cols // self.n
#             self._mode      = "matmul"
#
#         self.Pi    = [Pi0, Pi1]
#         self.stoch = _StochView(self)
#
#     def smooth(self, V):
#         """
#         Apply the per-axis Toeplitz stochastic kernels to ``V``.
#
#             VI = vec(Pi0.T @ reshape(V, (l1, l2)) @ Pi1)
#
#         Returns a flat ``(n,)`` vector. Shared between both ``_mode`` branches.
#         """
#         K0, K1  = self.Pi
#         V_grid  = np.asarray(V, dtype=float).reshape(self.l1, self.l2, order="F")
#         VI_grid = (K0.T @ V_grid) @ K1
#         return VI_grid.reshape(-1, order="F")
#
#     def mtimes(self, V):
#         """
#         Apply the operator to ``V``. Output shape depends on the
#         deterministic-shift representation:
#
#             - ``gather``  (target_idx): ``(n * M,)``, F-ordered over
#               ``(source_cell, input)``.
#             - ``matmul``  (P_det):      ``(n * M,)`` if ``P_det`` is
#               ``(n, n*M)``, or ``(n,)`` if it is ``(n, n)``.
#         """
#         VI = self.smooth(V)
#
#         if self._mode == "gather":
#             VI_with_sink = np.empty(self.n + 1, dtype=VI.dtype)
#             VI_with_sink[: self.n] = VI
#             VI_with_sink[self.n]   = 0.0
#             return VI_with_sink[self.target_idx].reshape(-1, order="F")
#
#         out = VI.reshape(1, -1) @ self.P_det
#         if hasattr(out, "toarray"):
#             out = out.toarray()
#         return np.asarray(out).ravel(order="F")
#
#     def gather_at(self, VI, u):
#         """
#         Per-input gather for a smoothed ``VI`` vector. Only available
#         when the operator is in ``gather`` mode (raw abstraction).
#         """
#         if self._mode != "gather":
#             raise RuntimeError("gather_at requires the target_idx representation")
#         idx   = self.target_idx[:, u]
#         valid = idx < self.n
#         out   = np.zeros(self.n, dtype=VI.dtype)
#         out[valid] = VI[idx[valid]]
#         return out
#
#     def __rmatmul__(self, V): return NotImplemented
#     def __matmul__(self, _):  return NotImplemented


# version 0
# class TransitionProbability2D:
#     """
#     Separable 2D stochastic transition probability operator.
#
#     Parameters
#     ----------
#     l : tuple of int
#         Grid shape ``(l1, l2)``. Their product equals the flat state
#         dimension ``n``.
#     P_det : np.ndarray
#         ``(n, n)`` deterministic projection matrix applied after the
#         separable stochastic part.
#     Pi0 : np.ndarray
#         ``(l1, l1)`` stochastic kernel along the first axis.
#     Pi1 : np.ndarray
#         ``(l2, l2)`` stochastic kernel along the second axis.
#
#     Attributes
#     ----------
#     l1, l2 : int
#         Per-axis grid sizes.
#     l : np.ndarray
#         Convenience alias: 2-element int array ``[l1, l2]``.
#     n : int
#         Flat state dimension, ``l1 * l2``.
#     dim : int
#         Number of grid dimensions (always 2 for this class).
#     P_det : np.ndarray or scipy.sparse
#         Stored as-is.
#     Pi : list
#         ``[Pi0, Pi1]`` per-axis kernels.
#     stoch : _StochView
#         Right-hand matmul endpoint. ``V @ P.stoch`` applies the
#         separable 2D stochastic operator and returns a flat ``(n,)``
#         result.
#
#     Notes
#     -----
#     Bare ``V @ P`` and ``P @ V`` are intentionally disabled; use
#     ``V @ P.stoch`` to make the operation explicit at the call site.
#     """
#
#     # Force numpy's @ to defer to __rmatmul__ rather than broadcasting
#     # against the class internals.
#     __array_priority__ = 10_000
#
#     def __init__(self, l, P_det, Pi0=None, Pi1=None):
#         self.l1, self.l2 = int(l[0]), int(l[1])
#         self.l = np.array([self.l1, self.l2], dtype=int)
#         self.dim = 2
#         self.n = self.l1 * self.l2
#         self.P_det = P_det
#         self.Pi = [Pi0, Pi1]
#
#         # Operator endpoint (right-hand matmul).
#         self.stoch = _StochView(self)
#
#     def mtimes(self, V):
#         """
#         Apply the separable 2D stochastic operator to ``V``.
#
#             VI  = vec(Pi0.T @ reshape(V, (l1, l2)) @ Pi1)
#             out = VI @ P_det
#
#         Used internally by ``_StochView.__rmatmul__``. Named after the
#         MATLAB SysCoRe equivalent.
#
#         Parameters
#         ----------
#         V : np.ndarray
#             Flat state vector of length ``n = l1 * l2``.
#
#         Returns
#         -------
#         np.ndarray
#             Flat ``(n,)`` result.
#         """
#         l1, l2 = self.l1, self.l2
#         K0, K1 = self.Pi
#
#         V_grid = np.asarray(V, dtype=float).reshape(l1, l2, order="F")
#         VI_grid = (K0.T @ V_grid) @ K1
#         VI = VI_grid.reshape(-1, order="F").reshape(1, -1)
#
#         out = VI @ self.P_det
#         if isspmatrix(out):
#             out = out.toarray()
#         return np.asarray(out).ravel(order="F")
#
#     # Bare V @ P and P @ V are forbidden — callers must use V @ P.stoch.
#     def __rmatmul__(self, V):
#         return NotImplemented
#
#     def __matmul__(self, _):
#         return NotImplemented