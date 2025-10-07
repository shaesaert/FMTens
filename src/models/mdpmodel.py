"""
MDPModel — discrete MDP abstraction built from a continuous system.
See docs/models/MDPModel.md for usage and details.
"""
# ===============================
# Imports & basic type alias
# ===============================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Literal

import numpy as np
from .utils.abstraction_shape_helper import (
    _cartesian_from_axes, _flat_from_blocks, _blocks_from_flat
)

from .utils.abstraction_factory import mdp_from_system as _mdp_from_system

Array = np.ndarray

# ===============================
# MDPModel
# ===============================
@dataclass
class MDPModel:
    # ---- public fields ----
    type: str = "MDP abstract"
    states: Optional[Array] = None      # (N, n) grid points
    outputs: Optional[Array] = None     # optional: C @ states^T (computed if orig is given)
    inputs: Optional[Array] = None      # (M, m) discrete input set (e.g., uhat)
    dim: Optional[int] = None           # n (state dimension)
    orig: Optional[object] = None       # original continuous model (e.g., LinModel)
    P: Optional[Array] = None           # transitions: public view as (N, N*M)
    labels: Optional[list] = None
    beta: Optional[Array] = None
    hx: Optional[List[Array]] = None    # per-dimension axes
    zstates: Optional[Array] = None
    l: Optional[Tuple[int, ...]] = None # bins per dimension
    Partition: Optional[object] = None
    outputmap: Optional[object] = None
    Pxix: Optional[object] = None

    # ---- internal cached formats ----
    _P_flat: Optional[Array] = None     # (N, N*M)
    _P_blocks: Optional[Array] = None   # (N, N, M)

    # =============================
    # Constructors / initializers
    # =============================
    def __init__(
            self,
            P: Array,
            hx: List[Array],
            states: Optional[Array] = None,
            beta: Optional[Array] = None,
            orig: Optional[object] = None,
            inputs: Optional[Array] = None,
            labels: Optional[list] = None,
    ):
        # -------------------------
        # grid / sizes
        # -------------------------
        self.hx = [np.asarray(ax, dtype=float).ravel() for ax in hx]
        self.l = tuple(len(ax) for ax in self.hx)  # bins per dim
        self.dim = len(self.hx)

        # states
        if states is None:
            self.states = _cartesian_from_axes(self.hx)  # (N, n)
        else:
            self.states = np.asarray(states, dtype=float).reshape(-1, self.dim)

        # inputs
        if inputs is not None:
            U = np.asarray(inputs, dtype=float)
            if U.ndim == 1:
                U = U.reshape(-1, 1)
            self.inputs = U  # (M, m)
        else:
            self.inputs = None

        # -------------------------
        # transitions
        # -------------------------
        # prepare fields so they exist in all cases
        self._P_flat = None  # dense (N, N*M)
        self._P_blocks = None  # dense (N, N, M)
        self.P_det = None  # present when operator is provided
        self.Pi = None  # list of per-dimension kernels when operator
        self.P = None  # public alias: dense array OR operator

        # Case A: classic numeric (1D) — easy behaviour
        if isinstance(P, np.ndarray):
            P_arr = np.asarray(P, dtype=float)
            if P_arr.ndim == 2:
                self._P_flat = P_arr
                N = self.states.shape[0]
                self._P_blocks = _blocks_from_flat(P_arr, N)
            elif P_arr.ndim == 3:
                self._P_blocks = P_arr
                self._P_flat = _flat_from_blocks(P_arr)
            else:
                raise ValueError("Dense P must be shape (N, N*M) or (N, N, M).")
            self.P = self._P_flat  # keep old public alias for 1D/dense

        # Case B: 2D tensor operator object (duck-typed: must expose P_det, dim, l, Pi)
        elif all(hasattr(P, attr) for attr in ("P_det", "dim", "l", "Pi")):
            # store operator directly
            self.P = P
            # mirror helpful fields for convenience
            self.P_det = P.P_det
            try:
                self.l = tuple(int(v) for v in np.asarray(P.l).ravel())
            except Exception:
                # fall back to l derived from hx if operator.l is not cleanly coercible
                pass
            self.Pi = list(P.Pi)

        else:
            raise TypeError(
                "P must be either a dense numpy array with shape (N, N*M)/(N, N, M) "
                "or a tensor_transition_probability_2d-like object exposing P_det, dim, l, Pi."
            )

        # -------------------------
        # other props
        # -------------------------
        self.beta = beta
        self.orig = orig
        self.labels = labels

        # outputs = C @ states if available
        if self.orig is not None and hasattr(self.orig, "C"):
            C = np.asarray(self.orig.C, dtype=float)
            self.outputs = (C @ self.states.T).T
        else:
            self.outputs = None

    # # --------- classmethod: build from continuous system ---------
    @classmethod
    def from_system(cls, *args, **kwargs):
        return _mdp_from_system(cls, *args, **kwargs)

    # =====================
    # Properties
    # =====================
    @property
    def N(self) -> int:
        return self.states.shape[0]

    @property
    def M(self) -> int:
        return 0 if self._P_blocks is None else self._P_blocks.shape[2]

    @property
    def P_blocks(self) -> Array:
        return self._P_blocks

    @property
    def P_flat(self) -> Array:
        return self._P_flat

