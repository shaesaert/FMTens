# pc_utils.py
from typing import Union
import numpy as np
from scipy import sparse

def Pc(P_flat: Union[np.ndarray, object], pol) -> np.ndarray:
    """
    If P_flat is:
      • ndarray (N, N*nu): use your original 1D logic (unchanged).
      • object with .P_det (sparse (N, N*nu)): select columns sparsely, then fold.
    Returns Pcomp (N, N).
    """

    # --- normalize policy to sparse (works for 1D and 2D) ---
    if not sparse.issparse(pol):
        pol = sparse.csr_matrix(pol)
    N, nu = pol.shape

    # Build selector v_den (length N*nu) from one-hot/weighted policy
    coo = pol.tocoo()
    k = coo.row + coo.col * N
    v = sparse.csr_matrix((coo.data, (np.zeros_like(k), k)), shape=(1, N * nu))
    v_den = np.asarray(v.toarray()).ravel()

    # =========================
    # 2D/tensor case (new only)
    # =========================
    if hasattr(P_flat, "P_det"):
        Psrc = P_flat.P_det  # expected sparse (N, N*nu)
        if not sparse.issparse(Psrc):
            Psrc = sparse.csr_matrix(Psrc)

        # select columns via sparse diagonal (keeps sparsity)
        D = sparse.diags(v_den)           # (N*nu, N*nu)
        Plarge = Psrc @ D                 # (N, N*nu) sparse

        # fold N-column blocks into (N, N)
        _, C = Plarge.shape
        blocks = C // N
        Pcomp = np.zeros((N, N), dtype=float)
        for i in range(blocks):
            Pcomp += Plarge[:, i * N:(i + 1) * N].toarray()
        return Pcomp

    # =========================
    # 1D dense case (unchanged)
    # =========================
    Pprob = np.asarray(P_flat, float)     # (N, N*nu)

    # Column-wise Hadamard selection
    Plarge = Pprob * v_den                # (N, N*nu)

    # Fold N-column blocks back into (N, N)
    r, c = Plarge.shape
    cr = c // r
    Pcomp = np.zeros((r, r), dtype=Plarge.dtype)
    for i in range(cr):
        Pcomp += Plarge[:, i * r: (i + 1) * r]

    return Pcomp
