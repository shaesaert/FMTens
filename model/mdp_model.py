# -*- coding: utf-8 -*-
"""
MDPModel: discrete MDP abstraction over a gridded continuous system.

- Accepts two transition-matrix layouts:
  * flat:   (N, N*M)      # the k-th action occupies columns [k*N : (k+1)*N)
  * blocks: (N, N, M)     # the 3rd axis indexes action k
- hx: list of 1D grid axes (e.g., [ax_x, ax_y, ...])
- states: if omitted, built from hx as the Cartesian product with shape (N, n)

Common attributes/methods:
- mdp.P_flat / mdp.P_blocks / mdp.P (= P_flat)
- mdp.block(k) / mdp.block_cols(k) / mdp.replace_block(k, Bk)
- mdp.check_rowsum()
- mdp.sim(x, u): sample the next grid state according to transition probabilities
- mdp.row_block(i, k): return the (N,) probability vector for row i under action k
- mdp.state_index_from_x(x) / mdp.input_index_from_u(u)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

Array = np.ndarray


def _cartesian_from_axes(axes: List[Array]) -> Array:
    """Build (N, n) array from a list of 1D axes using meshgrid (indexing='ij')."""
    if len(axes) == 1:
        return np.asarray(axes[0], dtype=float).reshape(-1, 1)
    meshes = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.reshape(-1) for m in meshes], axis=1)
    return pts  # (N, n)


def _flat_from_blocks(P_blocks: Array) -> Array:
    """Convert (N, N, M) to (N, N*M)."""
    N, _, M = P_blocks.shape
    return P_blocks.reshape(N, N * M, order="C")


def _blocks_from_flat(P_flat: Array, N: int) -> Array:
    """Convert (N, N*M) to (N, N, M)."""
    if P_flat.shape[1] % N != 0:
        raise ValueError("P_flat second dimension is not a multiple of N.")
    M = P_flat.shape[1] // N
    return P_flat.reshape(N, N, M, order="C")


@dataclass
class MDPModel:
    # ---- public fields (mirroring your MATLAB class) ----
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
        # store basics
        self.hx = [np.asarray(ax, dtype=float).ravel() for ax in hx]
        self.l = tuple(len(ax) for ax in self.hx)
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

        # transition formats
        P = np.asarray(P, dtype=float)
        if P.ndim == 2:
            self._P_flat = P
            N = self.states.shape[0]
            self._P_blocks = _blocks_from_flat(P, N)
        elif P.ndim == 3:
            self._P_blocks = P
            self._P_flat = _flat_from_blocks(P)
        else:
            raise ValueError("P must be (N, N*M) or (N, N, M).")
        self.P = self._P_flat  # public alias

        # other props
        self.beta = beta
        self.orig = orig
        self.labels = labels

        # outputs = C @ states' if possible
        if self.orig is not None and hasattr(self.orig, "C"):
            C = np.asarray(self.orig.C, dtype=float)
            # states: (N, n) -> (n, N) then C*(n, N) -> (p, N) -> (N, p)
            self.outputs = (C @ self.states.T).T
        else:
            self.outputs = None

    # -------------- properties --------------

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

    # -------------- dynamics wrappers --------------

    def f_det(self, x: Array, u: Array) -> Array:
        """Deterministic next state via original model (if provided)."""
        if self.orig is None or not hasattr(self.orig, "f_det"):
            raise RuntimeError("orig model with f_det(x,u) is required.")
        return self.orig.f_det(np.asarray(x, dtype=float), np.asarray(u, dtype=float))

    # -------------- indexing: continuous x/u -> discrete indices --------------

    def state_multi_idx_from_x(self, x: Array) -> Tuple[int, ...]:
        """Map a continuous state x (n,) to per-dimension grid indices via nearest neighbor."""
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.dim:
            raise ValueError(f"x has dim {x.size}, expected {self.dim}")
        idxs = []
        for d, ax in enumerate(self.hx):
            j = int(np.argmin(np.abs(ax - x[d])))
            idxs.append(j)
        return tuple(idxs)

    def state_index_from_x(self, x: Array) -> int:
        """Map x to a flat index in [0, N-1]."""
        multi = self.state_multi_idx_from_x(x)
        return int(np.ravel_multi_index(multi, self.l, order="C"))

    def input_index_from_u(self, u: Array) -> int:
        """Map a continuous input u (m,) to the nearest discrete input index."""
        if self.inputs is None:
            raise RuntimeError("inputs (discrete set) not provided.")
        u = np.asarray(u, dtype=float).ravel()
        m = self.inputs.shape[1]
        if u.size != m:
            raise ValueError(f"u has dim {u.size}, expected {m}")
        dists = np.linalg.norm(self.inputs - u[None, :], axis=1)
        return int(np.argmin(dists))

    def idx_to_state(self, i: int) -> Array:
        """Return the grid-state vector for flat index i."""
        return self.states[int(i), :]

    # -------------- abstract simulation --------------

    def row_block(self, i: int, k: int) -> Array:
        """Return the (N,) transition probability vector for row i under action k."""
        return self._P_blocks[int(i), :, int(k)]

    def sim_index(self, i: int, k: int, rng: Optional[np.random.Generator] = None) -> int:
        """Sample next-state index j ~ P(j | i, u_k)."""
        rng = rng or np.random.default_rng()
        p = self.row_block(i, k)
        s = p.sum()
        if s <= 0:
            return int(i)  # no outgoing probability; stay
        p = p / s
        j = rng.choice(self.N, p=p)
        return int(j)

    def sim(
        self,
        x: Union[int, Array],
        u: Union[int, Array],
        return_index: bool = False,
        rng: Optional[np.random.Generator] = None
    ) -> Union[Array, Tuple[Array, int]]:
        """
        One abstract step:
        - if x is an array -> quantize to state index; if x is int -> treat as index
        - if u is an array -> map to input index;  if u is int -> treat as index
        Returns the next grid-state vector (and optionally its index).
        """
        # current state index
        if isinstance(x, (int, np.integer)):
            i = int(x)
        else:
            i = self.state_index_from_x(np.asarray(x, dtype=float))

        # input index
        if isinstance(u, (int, np.integer)):
            k = int(u)
        else:
            k = self.input_index_from_u(np.asarray(u, dtype=float))

        j = self.sim_index(i, k, rng=rng)
        x_next = self.idx_to_state(j)
        return (x_next, j) if return_index else x_next

    # ===== convenience for flat/blocks forms =====
    def to_flat(self) -> Array:
        """Return P in flat form (N, N*M)."""
        return self._P_flat

    def to_blocks(self) -> Array:
        """Return P in block form (N, N, M)."""
        return self._P_blocks

    def block(self, k: int) -> Array:
        """Get the k-th action block B_k (N, N) from P."""
        return self._P_blocks[:, :, int(k)]

    def block_cols(self, k: int) -> slice:
        """Column slice [k*N : (k+1)*N] for the k-th action in flat form."""
        N = self.N
        return slice(k * N, (k + 1) * N)

    def replace_block(self, k: int, Bk: Array) -> None:
        """Replace the k-th action block with Bk (N, N) and update both views."""
        Bk = np.asarray(Bk, dtype=float)
        if Bk.shape != (self.N, self.N):
            raise ValueError(f"Bk must be ({self.N}, {self.N})")
        self._P_blocks[:, :, k] = Bk
        self._P_flat[:, self.block_cols(k)] = Bk

    def check_rowsum(self, atol: float = 1e-12) -> Tuple[bool, float]:
        """
        Check that every action block is (approximately) row-stochastic; return (ok, max_err).
        ok: True if every row sum is within `atol` of 1
        max_err: the maximum absolute deviation |row_sum - 1| across all blocks
        """
        if self.M == 0:
            return True, 0.0
        s = self._P_blocks.sum(axis=1)  # (N, M)
        max_err = float(np.max(np.abs(s - 1.0)))
        return (max_err <= atol, max_err)
