# pc_utils.py
from typing import Union
import numpy as np
from scipy import sparse

# import your wrapper
from src.models.utils.tensor_transition_probability_2d import TransitionProbability2D


Matrix = Union[np.ndarray, sparse.spmatrix]

def Pc(P_flat: Union[np.ndarray, TransitionProbability2D], pol: Matrix):
    """
    Compose policy with a block-column transition.

    If P_flat is:
      • TransitionProbability2D with P_det (n, n*nu) and Pi[0], Pi[1]:
          - returns a NEW TransitionProbability2D with SAME l, Pi
            but P_det collapsed to (n, n).
      • ndarray (n, n*nu): returns collapsed (n, n) ndarray.

    Never mutates the input P_flat.
    """
    # normalize policy -> CSR (N, nu)
    if not sparse.issparse(pol):
        pol = sparse.csr_matrix(pol)
    N, nu = pol.shape

    # policy vector in Fortran order: [π(:,1); π(:,2); ...]  (N*nu,)
    v = np.asarray(pol.toarray()).ravel(order="F")

    # ---- 2D / wrapper path: build a NEW wrapper with collapsed P_det ----
    if hasattr(P_flat, "P_det"):
        P_full = P_flat.P_det                       # (n, n*nu) sparse or dense
        K0, K1 = P_flat.Pi[0], P_flat.Pi[1]
        l = (P_flat.l1, P_flat.l2)
        n = P_flat.n

        if sparse.issparse(P_full):
            Plarge = P_full @ sparse.diags(v)       # (n, n*nu) sparse
            blocks = Plarge.shape[1] // n
            P_comp = sparse.csr_matrix((n, n))
            for i in range(blocks):
                P_comp += Plarge[:, i*n:(i+1)*n]    # sum N-wide blocks -> (n, n)
        else:
            P_full = np.asarray(P_full, float)
            Plarge = P_full * v                     # column scaling
            blocks = Plarge.shape[1] // N
            P_comp = Plarge.reshape(N, N, blocks, order="F").sum(axis=2)  # (n, n)

        # return a NEW wrapper with (n,n) P_det and same Pi
        return TransitionProbability2D(l=l, P_det=P_comp, Pi0=K0, Pi1=K1)

    # ---- 1D dense path: return (N,N) ndarray ----
    P_full = np.asarray(P_flat, float)              # (N, N*nu)
    Plarge = P_full * v
    blocks = Plarge.shape[1] // N
    return Plarge.reshape(N, N, blocks, order="F").sum(axis=2)
