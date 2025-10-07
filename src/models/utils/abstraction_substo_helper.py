import numpy as np
Array = np.ndarray

def _sub_stochasticize_block(Bu: Array, eps: float = 1e-12,
                             target: float = 0.999, mode: str = "cap") -> Array:
    Bu = np.asarray(Bu, dtype=float).copy()
    Bu[np.isnan(Bu)] = 0.0
    rsum = Bu.sum(axis=1, keepdims=True)

    z = rsum <= eps
    if np.any(z):
        idx = np.where(z[:, 0])[0]
        Bu[idx, :] = 0.0
        Bu[idx, idx] = target
        rsum = Bu.sum(axis=1, keepdims=True)

    nz = ~z[:, 0]
    if np.any(nz):
        if mode == "set":
            Bu[nz, :] *= (target / rsum[nz])
        else:
            scale = np.minimum(1.0, target / rsum[nz])
            Bu[nz, :] *= scale
    return Bu
