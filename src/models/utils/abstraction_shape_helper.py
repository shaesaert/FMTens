# ==========================================
# Shape/helpers (MATLAB SysCoRe-compatible memory layout helpers)
# ==========================================
import numpy as np
from typing import List

Array = np.ndarray

def _cartesian_from_axes(axes: List[Array]) -> Array:
    """Build (N, n) array from a list of 1D axes using meshgrid (indexing='ij')."""
    if len(axes) == 1:
        return np.asarray(axes[0], dtype=float).reshape(-1, 1)
    meshes = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.reshape(-1) for m in meshes], axis=1)
    return pts  # (N, n)


def _flat_from_blocks(P_blocks: Array) -> Array:
    """Convert (N, N, M) -> (N, N*M)."""
    N, _, M = P_blocks.shape
    return P_blocks.reshape(N, N * M, order="F")


def _blocks_from_flat(P_flat: np.ndarray, N: int) -> np.ndarray:
    """Convert (N, N*M) -> (N, N, M) using Fortran (column-major) layout."""
    if P_flat.shape[1] % N != 0:
        raise ValueError("P_flat second dimension is not a multiple of N.")
    M = P_flat.shape[1] // N
    return P_flat.reshape(N, N, M, order="F")