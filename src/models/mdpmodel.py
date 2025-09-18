# -*- coding: utf-8 -*-
"""
MDPModel (merged): discrete MDP abstraction that can **self-build** its grids and
transition matrices from a continuous system, so your main can be as simple as:

    sysAbs = {
        0: MDPModel.from_system(sysLTI[0], nx=1000, nu=5, placement='centers',
                                tol=1e-15, renormalize=True, contract_sum=1.0),
        1: MDPModel.from_system(sysLTI[1], nx=1000, nu=5, placement='centers',
                                tol=1e-15, renormalize=True, contract_sum=1.0),
    }

This file **merges** utilities previously in `ugrid_util.py` and `Psas.py` into the
class, so you don't need to import them separately.

Highlights
---------
- `MDPModel.from_system(...)` builds X-grid and U-grid and computes P internally.
- `make_uniform_grid` and the separable Gaussian transition builder are included
  as static methods on the class.
- Supports both flat `(N, N*M)` and block `(N, N, M)` forms internally.
- Optional sub-stochastic row capping via `contract_sum` (like your renorm step).

Dependencies: `numpy`, `polytope`, `scipy.special.erf`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import polytope as pc
from scipy.special import erf as _erf

Array = np.ndarray


# =====================
# Helper math utilities
# =====================

def _normal_cdf(x: Array, m: float, s: float) -> Array:
    """Gaussian CDF with mean m and std s; supports NumPy broadcasting."""
    z = (x - m) / (s * np.sqrt(2.0))
    return 0.5 * (1.0 + _erf(z))


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
    return P_blocks.reshape(N, N * M, order="C")


def _blocks_from_flat(P_flat: Array, N: int) -> Array:
    """Convert (N, N*M) -> (N, N, M)."""
    if P_flat.shape[1] % N != 0:
        raise ValueError("P_flat second dimension is not a multiple of N.")
    M = P_flat.shape[1] // N
    return P_flat.reshape(N, N, M, order="C")


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
            self.outputs = (C @ self.states.T).T
        else:
            self.outputs = None

    # --------- classmethod: build from continuous system ---------
    @classmethod
    def from_system(
            cls,
            orig: object,
            nx: Union[int, List[int], Tuple[int, ...]],
            nu: Union[int, List[int], Tuple[int, ...]],
            *,
            placement: str = 'centers',
            filter_inside: bool = True,
            tol: float = 1e-15,
            renormalize: bool = True,
            return_flat: bool = True,
            contract_sum: Optional[float] = None,
            contract_mode: str = 'cap',
            X_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # NEW
            U_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # NEW
    ) -> 'MDPModel':

        """Convenience constructor: auto-build X-grid, U-grid, and P from `orig`.

        Parameters
        ----------
        orig : continuous model with fields A, B, Bw, mu, sigma, X(polytope), U(polytope)
        nx   : int or sequence — grid counts per state dimension
        nu   : int or sequence — grid counts per input dimension
        placement : {'centers','endpoints'} — grid placement
        filter_inside : keep only points inside the polytopes
        tol : small probability cutoff inside transition builder
        renormalize : per (row,action) distribution normalization
        return_flat : build P in flat form (N, N*M)
        contract_sum : if given, post-process each action block with sub-stochastic
                        capping to make per-row sum <= contract_sum (e.g., 0.999 or 1.0)
        contract_mode: 'cap' (shrink only) or 'set' (scale to exactly target)
        """
        # 1) X-grid and U-grid
        X_poly_or_bounds = X_bounds if X_bounds is not None else orig.X
        U_poly_or_bounds = U_bounds if U_bounds is not None else orig.U

        X_axes, X_points = cls.make_uniform_grid(X_poly_or_bounds, grid_counts=nx,
                                                 filter_inside=filter_inside,
                                                 placement=placement)
        U_axes, U_points = cls.make_uniform_grid(U_poly_or_bounds, grid_counts=nu,
                                                 filter_inside=filter_inside,
                                                 placement=placement)

        # U_points shape -> (M, m)
        if U_points.ndim == 1:
            U_points = U_points.reshape(-1, 1)

        # 2) Transition matrix (flat by default)
        P = cls.transition_matrix_nd_separable(
            sys=orig,
            X_axes=X_axes,
            U_points=U_points,
            X_poly=orig.X,
            tol=tol,
            renormalize=renormalize,
            return_flat=return_flat,
        )

        # 3) Build the model
        mdp = cls(P=P, hx=X_axes, states=X_points, orig=orig, inputs=U_points)

        # 4) Optional sub-stochastic row capping (per action block)
        if contract_sum is not None:
            mdp.apply_substochastic(target=contract_sum, mode=contract_mode)

        return mdp

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

    # =====================
    # Grid utilities (merged)
    # =====================
    @staticmethod
    def _bbox_from_polytope(poly_or_bounds) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return [lower, upper] bounds.
        Accepts:
          - polytope.Polytope
          - (lower, upper) tuple/list/array, each (n,)
          - array with shape (2, n) where row 0 = lower, row 1 = upper
        """
        if poly_or_bounds is None:
            raise ValueError("Expected a Polytope or (lower, upper) bounds, got None.")

        # tuple/list/array of bounds
        if isinstance(poly_or_bounds, (tuple, list)):
            lower = np.asarray(poly_or_bounds[0], dtype=float).ravel()
            upper = np.asarray(poly_or_bounds[1], dtype=float).ravel()
            return lower, upper

        arr = np.asarray(poly_or_bounds)
        if arr.ndim == 2 and arr.shape[0] == 2:
            lower = np.asarray(arr[0], dtype=float).ravel()
            upper = np.asarray(arr[1], dtype=float).ravel()
            return lower, upper

        # assume polytope object
        V = np.asarray([np.asarray(v).ravel() for v in pc.extreme(poly_or_bounds)])
        lower = V.min(axis=0)
        upper = V.max(axis=0)
        return lower, upper

    @staticmethod
    def _filter_points_in_poly(P, pts: Array, tol: float = 1e-9) -> Array:
        """Keep only points inside polytope P using A x <= b (+ tolerance)."""
        A = np.asarray(P.A, dtype=float)
        b = np.asarray(P.b, dtype=float).reshape(-1, 1)
        ok = (A @ pts.T <= b + tol).all(axis=0)
        return pts[ok]

    @staticmethod
    def make_uniform_grid(U_poly,
                          grid_counts: Union[int, List[int], Tuple[int, ...]],
                          *,
                          filter_inside: bool = True,
                          placement: str = 'centers') -> Tuple[List[Array], Array]:
        """
        Build an axis-aligned uniform grid over a bounded polytope.

        Parameters
        ----------
        U_poly : polytope.Polytope (bounded)
        grid_counts : int or iterable of ints (per dimension)
        filter_inside : keep only points inside polytope
        placement : 'centers' (cell centers) or 'endpoints' (linspace)

        Returns
        -------
        axes : list[np.ndarray] — per-dimension 1D arrays
        points : np.ndarray — (N, n) Cartesian product (filtered if requested)
        """
        lower, upper = MDPModel._bbox_from_polytope(U_poly)
        n = lower.size

        # normalize grid counts
        if np.isscalar(grid_counts):
            if n != 1:
                raise ValueError(
                    f"grid_counts is scalar but polytope is {n}D; pass a length-{n} iterable."
                )
            counts = [int(grid_counts)]
        else:
            counts = [int(c) for c in list(grid_counts)]
            if len(counts) != n:
                raise ValueError(f"grid_counts length {len(counts)} must match dimension {n}.")

        # per-axis arrays
        axes = []
        for i in range(n):
            L, U, m = float(lower[i]), float(upper[i]), counts[i]
            if placement == 'endpoints':
                ax = np.linspace(L, U, m)
            elif placement == 'centers':
                step = (U - L) / m
                ax = L + step * (0.5 + np.arange(m))
            else:
                raise ValueError("placement must be 'centers' or 'endpoints'")
            axes.append(ax)

        # Cartesian product grid -> (N, n)
        if n == 1:
            points = axes[0].reshape(-1, 1)
        else:
            meshes = np.meshgrid(*axes, indexing="ij")
            points = np.stack([m.reshape(-1) for m in meshes], axis=1)

        if filter_inside:
            points = MDPModel._filter_points_in_poly(U_poly, points)

        return axes, points

    # ==========================================
    # Transition builder (merged from Psas.py)
    # ==========================================
    @staticmethod
    def _edges_from_axes(X_axes: List[Array], X_poly) -> List[Array]:
        """Per-dimension bin edges using domain bounds at ends and midpoints inside."""
        lower, upper = MDPModel._bbox_from_polytope(X_poly)
        edges = []
        for d, ax in enumerate(X_axes):
            ax = np.asarray(ax, dtype=float).ravel()
            L, U = float(lower[d]), float(upper[d])
            if ax.size == 1:
                eps = 1e-12
                e = np.array([ax[0] - eps, ax[0] + eps], dtype=float)
            else:
                mid = 0.5 * (ax[:-1] + ax[1:])
                e = np.concatenate([[L], mid, [U]]).astype(float)
            edges.append(e)
        return edges

    @staticmethod
    def _diag_std_from_noise(Bw: Array, Sigma_w: Array) -> Array:
        """Std of x^+ from noise: sqrt(diag(Bw * Sigma_w * Bw^T)) (per-dim)."""
        Sigma_x = Bw @ Sigma_w @ Bw.T
        var = np.clip(np.diag(Sigma_x), a_min=0.0, a_max=None)
        std = np.sqrt(var)
        return std

    @staticmethod
    def transition_matrix_nd_separable(
        sys,
        X_axes: List[Array],
        U_points: Array,
        X_poly,
        tol: float = 1e-15,
        renormalize: bool = True,
        return_flat: bool = True,
    ) -> Array:
        """
        Build an n-D (dimension-wise independent) Gaussian-noise transition matrix.
        Matches MATLAB cp/Kronecker logic used in your repo.

        Returns P:
          - (N, N*M) if return_flat, otherwise (N, N, M)
        """
        A = np.asarray(sys.A, dtype=float)
        B = np.asarray(sys.B, dtype=float)
        Bw = np.asarray(sys.Bw, dtype=float)
        mu_w = np.asarray(sys.mu, dtype=float).reshape(-1, 1)
        Sigma_w = np.asarray(sys.sigma, dtype=float)

        n = A.shape[0]
        X_axes = [np.asarray(ax, dtype=float).ravel() for ax in X_axes]
        l = [ax.size for ax in X_axes]
        N = int(np.prod(l))

        m = B.shape[1]
        U_points = np.asarray(U_points, dtype=float)
        if U_points.ndim == 1:
            U_points = U_points.reshape(-1, m)
        else:
            if U_points.shape[1] != m:
                if U_points.shape[0] == m:
                    U_points = U_points.T
                else:
                    U_points = U_points.reshape(-1, m)
        M = U_points.shape[0]

        edges = MDPModel._edges_from_axes(X_axes, X_poly)
        std = MDPModel._diag_std_from_noise(Bw, Sigma_w)
        if np.any(std == 0.0):
            raise ValueError("Some dimensions have zero noise std; handle degenerate dims separately.")

        meshes = np.meshgrid(*X_axes, indexing="ij")
        XhatSpace = np.stack([m_.reshape(-1) for m_ in meshes], axis=1)

        P = np.zeros((N, N * M), dtype=float) if return_flat else np.zeros((N, N, M), dtype=float)

        Bu = U_points @ B.T             # (M, n)
        w_mean = (Bw @ mu_w).ravel()    # (n,)

        for i in range(N):
            xi = XhatSpace[i, :]                  # (n,)
            Axi = (A @ xi).ravel()                # (n,)
            means_all = Bu + Axi + w_mean         # (M, n)

            for k in range(M):
                m_vec = means_all[k, :]           # (n,)
                probs_per_dim = []
                for d in range(n):
                    e = edges[d]
                    cdf_r = _normal_cdf(e[1:], m_vec[d], std[d])
                    cdf_l = _normal_cdf(e[:-1], m_vec[d], std[d])
                    cp = (cdf_r - cdf_l)
                    cp[cp < tol] = 0.0
                    probs_per_dim.append(cp)

                # Kronecker product across dimensions -> length N
                pij = probs_per_dim[0]
                for d in range(1, n):
                    pij = np.kron(pij, probs_per_dim[d])

                if renormalize:
                    s = pij.sum() + 1e-15
                    pij = pij / s

                if return_flat:
                    P[i, k * N : (k + 1) * N] = pij
                else:
                    P[i, :, k] = pij

        return P

    # ======================================
    # Sub-stochastic row processing (optional)
    # ======================================
    @staticmethod
    def _sub_stochasticize_block(Bu: Array, eps: float = 1e-12,
                                 target: float = 0.999, mode: str = "cap") -> Array:
        """
        Process an (N×N) block to satisfy a target row-sum:
          - Replace NaNs with 0
          - All-zero rows → self-loop with `target`
          - Nonzero rows:
              mode="set": scale each row to sum exactly `target`
              mode="cap": if row-sum > target, scale down; otherwise keep
        """
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
            else:  # cap
                scale = np.minimum(1.0, target / rsum[nz])
                Bu[nz, :] *= scale
        return Bu

    def apply_substochastic(self, target: float = 0.999, mode: str = 'cap') -> None:
        """Apply sub-stochastic row adjustment to every action block in-place."""
        if self.M == 0:
            return
        N = self.N
        blocks = []
        for u in range(self.M):
            Bu = self._P_blocks[:, :, u]
            blocks.append(MDPModel._sub_stochasticize_block(Bu, target=target, mode=mode))
        P_new = np.stack(blocks, axis=2)  # (N, N, M)
        self._P_blocks = P_new
        self._P_flat = _flat_from_blocks(P_new)
        self.P = self._P_flat

    # =====================
    # Dynamics wrappers / indexing
    # =====================
    def f_det(self, x: Array, u: Array) -> Array:
        """Deterministic next state via original model (if provided)."""
        if self.orig is None or not hasattr(self.orig, "f_det"):
            raise RuntimeError("orig model with f_det(x,u) is required.")
        return self.orig.f_det(np.asarray(x, dtype=float), np.asarray(u, dtype=float))

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
        if isinstance(x, (int, np.integer)):
            i = int(x)
        else:
            i = self.state_index_from_x(np.asarray(x, dtype=float))

        if isinstance(u, (int, np.integer)):
            k = int(u)
        else:
            k = self.input_index_from_u(np.asarray(u, dtype=float))

        j = self.sim_index(i, k, rng=rng)
        x_next = self.idx_to_state(j)
        return (x_next, j) if return_index else x_next

    # ===== convenience =====
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
        """Check every action block is approximately row-stochastic; return (ok, max_err)."""
        if self.M == 0:
            return True, 0.0
        s = self._P_blocks.sum(axis=1)  # (N, M)
        max_err = float(np.max(np.abs(s - 1.0)))
        return (max_err <= atol, max_err)
