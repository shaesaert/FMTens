# tensor_transition_probability_2d.py
from __future__ import annotations
import numpy as np

# SciPy is optional. If absent, fall back gracefully.
try:
    from scipy.sparse import isspmatrix  # type: ignore
except Exception:  # pragma: no cover
    def isspmatrix(_):  # noqa: N802
        return False


class _PView:
    """Lightweight view that enables right-hand @ and delegates to parent."""
    __array_priority__ = 10_000

    def __init__(self, parent, mode: str):
        self.parent = parent
        self.mode = mode

    def __rmatmul__(self, V):
        if self.mode == "det":
            return self.parent.dettimes(V)
        if self.mode == "stoch":
            return self.parent.mtimes(V)
        raise ValueError("unknown mode")

    # Disallow P.view @ V
    def __matmul__(self, _):
        return NotImplemented


class TransitionProbability2D:
    """
    One data class with two operator endpoints:
      - V @ P.det   -> dettimes (deterministic collapse; returns (n, a))
      - V @ P.stoch -> mtimes   (separable 2D; expects P_det (n, n); returns (n,))

    MATLAB equivalents:
      dettimes: V_n = reshape(V * P_det, n, a)
      mtimes:   VI = vec(Pi{1}' * reshape(V, [l1 l2]) * Pi{2});
                V' * = VI' * P_det   (here returned as a flat length-n vector)
    """

    __array_priority__ = 10_000

    def __init__(self, l, P_det, Pi0=None, Pi1=None):
        self.l1, self.l2 = int(l[0]), int(l[1])
        self.n = self.l1 * self.l2
        self.P_det = P_det          # For mtimes, treat as (n, n)
        self.Pi = [Pi0, Pi1]

        # For dettimes, if P_det is (n, n*a), this infers 'a' (a==1 if P_det is (n, n))
        self.a = self.P_det.shape[1] // self.n

        # Named operator endpoints
        self.det = _PView(self, "det")
        self.stoch = _PView(self, "stoch")

    # -------- MATLAB-equivalent cores --------

    # def dettimes(self, V):
    #     """
    #     Deterministic multiply:
    #         V_n = reshape(V * P_det, n, a)
    #     Works whether P_det is (n, n*a) or (n, n) (then a==1).
    #     """
    #     Vrow = np.asarray(V, float).reshape(1, -1, order="F")  # (1, n)
    #     out = Vrow @ self.P_det                                # (1, n*a)
    #     if isspmatrix(out):
    #         out = out.toarray()
    #     return np.asarray(out).reshape(self.n, self.a, order="F")

    def mtimes(self, V):
        """
        Stochastic/separable multiply (2D):
            VI = vec(Pi0' * reshape(V, [l1 l2]) * Pi1)
            out = VI' * P_det
        Expects P_det to be (n, n). Returns a flat vector of length n.
        """
        l1, l2 = self.l1, self.l2
        K0, K1 = self.Pi

        Vgrid = np.asarray(V, float).reshape(l1, l2, order="F")  # l1 x l2
        VI_grid = (K0.T @ Vgrid) @ K1                            # l1 x l2
        VI = VI_grid.reshape(-1, order="F").reshape(1, -1)       # 1 x n

        out = VI @ self.P_det                                    # 1 x n
        if isspmatrix(out):
            out = out.toarray()
        return np.asarray(out).ravel(order="F")                  # (n,)

    # Disallow bare V @ P and P @ V to force explicit .det or .stoch usage
    def __rmatmul__(self, V):
        return NotImplemented

    def __matmul__(self, _):
        return NotImplemented
