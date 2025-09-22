# pc_utils.py
from typing import Tuple
import numpy as np
from scipy import sparse

def Pc(P_flat, pol) -> np.ndarray:
    """
    Compute the compressed (controlled) transition matrix Pcomp from:
      - P_flat: an (N, N*nu) array of transitions concatenated by action
      - pol:    an (N, nu) policy matrix (ideally one-hot), typically CSR

    This version mirrors the original in-place logic you were using:
      1) Create a length-(N*nu) selector vector v_den from the one-hot policy.
      2) Multiply column-wise: Plarge = P_flat * v_den.
      3) Sum selected N-column blocks back into an (N, N) matrix.

    Returns
    -------
    Pcomp : np.ndarray of shape (N, N)
    """
    Pprob = np.asarray(P_flat, float)  # (N, N*nu)
    m, n = pol.shape                   # m = N, n = nu

    coo = pol.tocoo()
    # k indexes which columns (within the N*nu flattened-by-block structure) are picked
    k = coo.row + coo.col * m
    # Build a 1 x (N*nu) sparse row vector with ones at those positions
    v = sparse.csr_matrix((coo.data, (np.zeros_like(k), k)), shape=(1, m * n))
    v_den = np.asarray(v.toarray()).ravel()

    # Column-wise Hadamard (broadcasted) selection of P_flat columns
    Plarge = Pprob * v_den  # still (N, N*nu)

    # Fold N-column blocks back into an (N, N) matrix
    r, c = Plarge.shape
    cr = c // r
    Pcomp = np.zeros((r, r), dtype=Plarge.dtype)


    for i in range(cr):  # i = 0..cr-1
        Pcomp += Plarge[:, i * r: (i + 1) * r]

    return Pcomp
