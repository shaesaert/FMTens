# TensorTransitionProbability_2D.py
from __future__ import annotations
from typing import Sequence, Union
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix

Array = np.ndarray

class TensorTransitionProbability_2D:
    """
    2D analogue of the MATLAB class.

    - dettimes(V):  V(1×n) * P_det  -> reshape(n, a)
    - mtimes(V):    VI = vec(Pi[0]' * reshape(V,[l1 l2]) * Pi[1]);
                    V_n = reshape(VI' * P_det, [n a])

    l = (l1,l2), n = l1*l2; P_det: (n × n*a); Pi[0]: (l1×l1), Pi[1]: (l2×l2)
    Uses Fortran order to match MATLAB vec/reshape.
    """

    def __init__(self,
                 l: Sequence[int],
                 P_det: Union[Array, csr_matrix],
                 Pi0: Union[Array, csr_matrix],
                 Pi1: Union[Array, csr_matrix]) -> None:
        if len(l) != 2:
            raise ValueError("TensorTransitionProbability_2D requires l with exactly 2 entries.")
        self.l1, self.l2 = int(l[0]), int(l[1])
        self.n = self.l1 * self.l2

        # P_det as CSR
        if isspmatrix(P_det):
            self.P_det = P_det.tocsr().astype(float, copy=False)
        else:
            self.P_det = csr_matrix(np.asarray(P_det, dtype=float))
        if self.P_det.shape[0] != self.n or (self.P_det.shape[1] % self.n) != 0:
            raise ValueError("P_det must be (n × n*a) with n = l1*l2.")
        self.a = self.P_det.shape[1] // self.n  # number of actions

        # store per-dimension kernels in a list Pi[0], Pi[1]
        K0 = Pi0.tocsr().astype(float, copy=False) if isspmatrix(Pi0) else np.asarray(Pi0, dtype=float)
        K1 = Pi1.tocsr().astype(float, copy=False) if isspmatrix(Pi1) else np.asarray(Pi1, dtype=float)
        if K0.shape != (self.l1, self.l1):
            raise ValueError(f"Pi[0] must be ({self.l1},{self.l1}), got {K0.shape}.")
        if K1.shape != (self.l2, self.l2):
            raise ValueError(f"Pi[1] must be ({self.l2},{self.l2}), got {K1.shape}.")
        self.Pi = [K0, K1]

    def dettimes(self, V: Array) -> Array:
        """V: (n,) or (1,n) → (n, a), equivalent to reshape(V*P_det,n,a)."""
        Vrow = np.asarray(V, dtype=float).reshape(1, -1, order='F')
        out = Vrow @ self.P_det               # (1, n*a)
        return np.asarray(out).reshape(self.n, self.a, order='F')

    def mtimes(self, V: Array) -> Array:
        """
        2D separable multiply:
            VI = vec(Pi[0]' * reshape(V,[l1 l2]) * Pi[1])
            V_n = reshape(VI' * P_det, [n a])
        """
        Vgrid = np.asarray(V, dtype=float).reshape(self.l1, self.l2, order='F')
        K0, K1 = self.Pi[0], self.Pi[1]

        tmp = (K0.T @ Vgrid) if isspmatrix(K0) else (K0.T @ Vgrid)
        VI_grid = (tmp @ K1) if isspmatrix(K1) else (tmp @ K1)

        VI = VI_grid.reshape(-1, order='F').reshape(1, -1)  # (1, n)
        out = VI @ self.P_det                                # (1, n*a)
        return np.asarray(out).reshape(self.n, self.a, order='F')
