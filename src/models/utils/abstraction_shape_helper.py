"""
Memory-layout helpers for the abstraction layer.

Small set of conversions between block- and flat-stacked representations of
transition tensors, and a meshgrid-based Cartesian product over a list of
1D axes. The block/flat conversions use Fortran (column-major) ordering to
stay compatible with the MATLAB SysCoRe reference implementation.

Shape conventions used elsewhere in the pipeline:
    P_blocks : (N, N, M)   one (N, N) transition matrix per input letter
    P_flat   : (N, N*M)    horizontal concatenation in column-major order

These helpers are imported by :mod:`mdpmodel`.
"""

from typing import List

import numpy as np

Array = np.ndarray


def cartesian_from_axes(axes: List[Array]) -> Array:
    """
    Build an (N, n) Cartesian-product array from a list of n 1D axes
    using ``np.meshgrid(..., indexing='ij')``.
    """
    if len(axes) == 1:
        return np.asarray(axes[0], dtype=float).reshape(-1, 1)
    meshes = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.reshape(-1) for m in meshes], axis=1)


def flat_from_blocks(P_blocks: Array) -> Array:
    """
    Convert a block-stacked transition tensor ``(N, N, M)`` to the
    column-major flat form ``(N, N*M)``.
    """
    N, _, M = P_blocks.shape
    return P_blocks.reshape(N, N * M, order="F")


def blocks_from_flat(P_flat: Array, N: int) -> Array:
    """
    Inverse of :func:`flat_from_blocks`: convert a flat ``(N, N*M)``
    transition matrix back into the block-stacked form ``(N, N, M)``.
    """
    if P_flat.shape[1] % N != 0:
        raise ValueError("P_flat second dimension is not a multiple of N.")
    M = P_flat.shape[1] // N
    return P_flat.reshape(N, N, M, order="F")