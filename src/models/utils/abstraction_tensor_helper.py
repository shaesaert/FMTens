import numpy as np
import polytope as pc
from itertools import product

def linear_image_vertices(P, M):
    V = np.array([np.asarray(v).ravel() for v in pc.extreme(P)])
    Vimg = (M @ V.T).T
    return pc.qhull(Vimg)

def ff2n(dim: int) -> np.ndarray:
    return np.array(list(product([0, 1], repeat=dim)), dtype=int)

def combvec_first_fast(*axes):
    axes = [np.asarray(a).ravel() for a in axes]
    grids = np.meshgrid(*axes, indexing='ij')
    return np.vstack([g.reshape(-1, order='F') for g in grids])

def f_det(sys, x, u):
    return sys.A @ x + sys.B @ u

def repmat_vector_col(vec: np.ndarray, n_cols: int) -> np.ndarray:
    v = np.asarray(vec)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    elif v.ndim == 2 and v.shape[1] != 1:
        raise ValueError("vec must be a single column: (m,) or (m,1).")
    m = v.shape[0]
    return np.broadcast_to(v, (m, n_cols)).copy()
