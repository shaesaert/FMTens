# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
MDPModel
========
A lightweight, self-contained abstraction of a **discrete** Markov Decision Process
that can be *built directly from a continuous system*. The class constructs
state and input grids and computes the transition probabilities internally,
so the caller only specifies grid sizes and a few options. This keeps your
application code short and readable while remaining explicit about numerical
choices.

What this module provides
-------------------------
- `MDPModel.from_system(...)`: end-to-end constructor that:
  1) builds uniform grids for the state space `X` and input space `U`,
  2) computes the transition matrix `P` using separable Gaussian kernels,
  3) optionally caps/renormalizes rows to be (sub-)stochastic.
- Static helpers for gridding and tensorized transitions:
  `make_uniform_grid`, `_cartesian_from_axes`, `_flat_from_blocks`,
  `_blocks_from_flat`, and `transition_matrix_nd_separable(...)`.
- Optional sub-stochastic row capping via `apply_substochastic(target, mode)`
  (useful when you want ≤1 mass per row after truncation).

Typical use
-----------
If you have a continuous linear system with fields
`A, B, Bw, mu, sigma, X (polytope), U (polytope), C`, you can do:

    sysAbs = {
        0: MDPModel.from_system(sysLTI[0], nx=1000, nu=5,
                                placement='centers',
                                tol=1e-15,
                                renormalize=False,
                                contract_sum=1.0),
        1: MDPModel.from_system(sysLTI[1], nx=1000, nu=5,
                                placement='centers',
                                tol=1e-15,
                                renormalize=False,
                                contract_sum=1.0),
    }

By default, `compute_P='1d'` builds a flat `(N, N*M)` matrix. You may also
request a tensor/operator path with `compute_P='2d'` for 2-D systems, which
exposes per-dimension kernels and a sparse deterministic shift operator.

Key options (high level)
------------------------
- `nx`, `nu`: grid counts per state/input dimension (int or sequence).
- `placement`: `'centers' | 'endpoints'` (and `u_placement` override for inputs);
  a `'log'` option is supported for inputs in special cases.
- `filter_inside`: keep only grid points inside the given polytope/bounds.
- `tol`: probability mass below this threshold is truncated to zero.
- `renormalize`: per (row,action) normalization after truncation.
- `contract_sum`: cap or set each row-sum to a target (e.g., `1.0` or `0.999`).
- `return_flat`: choose flat `(N, N*M)` vs. block `(N, N, M)` storage.
- `compute_P`: `'1d' | '2d'` (tensor path for 2-D; `'nd'` placeholder).

Shapes & conventions
--------------------
- State grid `states`: `(N, n)` with Fortran-style (column-major) reshaping for
  consistent block layout.
- Inputs `inputs`: `(M, m)`; `M` discrete inputs of dimension `m`.
- Transitions:
  - Flat view: `P_flat` with shape `(N, N*M)`, action blocks concatenated in
    **Fortran** order.
  - Block view: `P_blocks` with shape `(N, N, M)`, where `P_blocks[:, :, u]`
    is the `(N×N)` kernel for the `u`-th input.
- Outputs `outputs`: computed as `C @ states.T` when `orig.C` is available.

When to use this abstraction
----------------------------
- You need a reproducible, explicit discretization of a stochastic LTI system.
- You want to carry both a friendly flat matrix (for legacy MATLAB/NumPy code)
  and a tensor/operator form (for structure-exploiting solvers).
- You need fine control over row sums (renormalization vs. sub-stochastic capping).

Assumptions & scope
-------------------
- The transition builder uses **separable Gaussian** noise injected through `Bw`,
  with mean `mu` and covariance `sigma` (see code for details). Non-Gaussian,
  non-separable kernels are out of scope here.
- The `'2d'` tensor path targets 2-D systems; `'nd'` is a reserved placeholder.
- Polytopes are expected to be bounded; bounds tuples/arrays are also supported.

Dependencies
------------
`numpy`, `polytope`, `scipy.special.erf`, and selected utilities from SciPy
(sparse matrices, statistics, linear algebra).

"""

# ===============================
# Imports & basic type alias
# ===============================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Literal

import numpy as np
import polytope as pc
from polytope import qhull
import warnings

from scipy._lib.array_api_compat import torch
from scipy.sparse import coo_matrix
from itertools import product

from fontTools.misc.plistlib import end_array
from scipy.special import erf as _erf  # (kept as imported in your file)

Array = np.ndarray

# ==========================================
# Shape/helpers (MATLAB SysCoRe-compatible memory layout helpers)
# ==========================================
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

    # --------- classmethod: build from continuous system ---------
    @classmethod
    def from_system(
            cls,
            orig: object,
            nx: Union[int, List[int], Tuple[int, ...]],
            nu: Union[int, List[int], Tuple[int, ...]],
            *,
            placement: str = 'centers',
            u_placement: Optional[str] = None,
            filter_inside: bool = True,
            tol: float = 1e-15,
            renormalize: bool = False,
            return_flat: bool = True,
            contract_sum: Optional[float] = None,
            contract_mode: str = 'cap',
            X_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # NEW
            U_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # NEW
            compute_P: Literal['1d', '2d','nd'] = '1d',
    ) -> 'MDPModel':

        """Convenience constructor: auto-build X-grid, U-grid, and P from `orig`.

        Parameters
        ----------
        orig : continuous model with fields A, B, Bw, mu, sigma, X(polytope), U(polytope)
        nx   : int or sequence — grid counts per state dimension
        nu   : int or sequence — grid counts per input dimension
        placement : {'centers','endpoints'} — grid placement
        u_placement: {'centers','endpoints','log'} or None
                     If provided, overrides `placement` for *inputs* only.
                     'log' creates log-spaced endpoints (requires positive bounds).
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

        # use u_placement if given; else fall back to placement
        up = u_placement if u_placement is not None else placement

        X_axes, X_points = cls.make_uniform_grid(X_poly_or_bounds, grid_counts=nx,
                                                 filter_inside=filter_inside,
                                                 placement=placement)
        U_axes, U_points = cls.make_uniform_grid(U_poly_or_bounds, grid_counts=nu,
                                                 filter_inside=filter_inside,
                                                 placement=up)

        # U_points shape -> (M, m)
        if U_points.ndim == 1:
            U_points = U_points.reshape(-1, 1)

        # 2) Transition matrix (if by default: compute one flattened matrix, works for 1d case)
        if compute_P == '1d':
            P = cls.transition_matrix_nd_separable(
                sys=orig,
                X_axes=X_axes,
                U_points=U_points,
                X_poly=orig.X,
                tol=tol,
                renormalize=renormalize,
                return_flat=return_flat,
                mode=compute_P
                )
            mdp = cls(P=P, hx=X_axes, states=X_points, orig=orig, inputs=U_points)
            mdp.outputs = mdp.orig.C @ X_points

        elif compute_P == '2d': # if indicated '2d': tensor computation
            (tP_2d, hz, XhatSpace, beta, sys_ret, ZhatSpace, Uz)= cls.transition_matrix_nd_separable(
                sys=orig,
                X_axes=X_axes,
                U_points=U_points,
                X_poly=orig.X,
                tol=tol,
                renormalize=renormalize,
                return_flat=return_flat,
                mode=compute_P
                )
            tP_2d.dim = 2
            tP_2d.l = np.array([tP_2d.l1, tP_2d.l2], dtype=int)
            mdp = cls(P=tP_2d, hx=hz, states=XhatSpace, beta=beta, orig=sys_ret, inputs=U_points)
            mdp.zstates = ZhatSpace  # transformed grid in Z-space
            mdp.Uz = Uz
            mdp.outputs = mdp.orig.C @ XhatSpace
            mdp.outputmap = mdp.orig.C @ Uz

        elif compute_P == 'nd':
            raise NotImplementedError("compute_P='n>2 d' not implemented yet (tensor path).")

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
    # Tensor asbtraction utilities
    # =====================
    import numpy as np
    from typing import Optional, Sequence

    @staticmethod
    def linear_image_vertices(P, M):
        # P: polytope.Polytope with H-rep but we use vertices
        V = np.array([np.asarray(v).ravel() for v in pc.extreme(P)])  # (Nv, n)
        Vimg = (M @ V.T).T  # map vertices
        return pc.qhull(Vimg)

    @staticmethod
    def ff2n(dim: int) -> np.ndarray:
        """Python equivalent of MATLAB ff2n(dim): 2^dim-by-dim matrix with levels {0,1}."""
        return np.array(list(product([0, 1], repeat=dim)), dtype=int)

    @staticmethod
    def combvec_first_fast(*axes):
        """
        Cartesian product columns with FIRST axis varying fastest.
        Returns array of shape (k, N), where k=len(axes), N=∏ len(axes[i]).
        """
        # make sure each axis is a 1-D numpy array
        axes = [np.asarray(a).ravel() for a in axes]
        # meshgrid with 'ij' indexing gives a grid per axis
        grids = np.meshgrid(*axes, indexing='ij')
        # reshape each grid column-wise (Fortran order) so the first axis cycles fastest
        return np.vstack([g.reshape(-1, order='F') for g in grids])

    @staticmethod
    def f_det(sys,x,u):
        x_n = sys.A @ x + sys.B @ u
        return x_n

    @staticmethod
    def repmat_vector_col(vec: np.ndarray, n_cols: int) -> np.ndarray:
            """
            Take ONE column vector (shape (m,) or (m,1)) and replicate it across columns.
            Returns an array of shape (m, n_cols).
            """
            v = np.asarray(vec)
            if v.ndim == 1:
                v = v.reshape(-1, 1)  # (m,1)
            elif v.ndim == 2 and v.shape[1] != 1:
                raise ValueError("vec must be a single column: (m,) or (m,1).")
            m = v.shape[0]
            return np.broadcast_to(v, (m, n_cols)).copy()


    # =====================
    # Grid utilities
    # =====================
    @staticmethod
    def _bbox_from_polytope(poly_or_bounds) -> Tuple[np.ndarray, np.ndarray]:
        if poly_or_bounds is None:
            raise ValueError("Expected a Polytope or (lower, upper) bounds, got None.")

        # tuple/list of bounds
        if isinstance(poly_or_bounds, (tuple, list)):
            lower = np.asarray(poly_or_bounds[0], dtype=float).ravel()
            upper = np.asarray(poly_or_bounds[1], dtype=float).ravel()
            return lower, upper

        arr = np.asarray(poly_or_bounds)

        # NEW: 1-D array with two entries -> treat as scalar bounds for 1-D
        if arr.ndim == 1 and arr.size == 2:
            lower = np.array([float(arr[0])])
            upper = np.array([float(arr[1])])
            return lower, upper

        # 2×n array -> row 0 lower, row 1 upper
        if arr.ndim == 2 and arr.shape[0] == 2:
            lower = np.asarray(arr[0], dtype=float).ravel()
            upper = np.asarray(arr[1], dtype=float).ravel()
            return lower, upper

        # Otherwise assume a polytope object
        V = np.asarray([np.asarray(v).ravel() for v in pc.extreme(poly_or_bounds)])
        lower = V.min(axis=0)
        upper = V.max(axis=0)
        return lower, upper

    @staticmethod
    def _filter_points_in_poly(P, pts: Array, tol: float = 1e-9) -> Array:
        """Keep only points inside polytope P (A x <= b) or within axis-aligned bounds."""
        # Polytope-like objects have A, b
        if hasattr(P, "A") and hasattr(P, "b"):
            A = np.asarray(P.A, dtype=float)
            b = np.asarray(P.b, dtype=float).reshape(-1, 1)
            ok = (A @ pts.T <= b + tol).all(axis=0)
            return pts[ok]

        # Otherwise treat P as bounds (tuple/list/array)
        lower, upper = MDPModel._bbox_from_polytope(P)  # returns (n,) arrays
        lower = np.asarray(lower, dtype=float).ravel()
        upper = np.asarray(upper, dtype=float).ravel()

        # pts is (N, n); works for n=1 too
        ok = np.all((pts >= lower) & (pts <= upper + tol), axis=1)
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
        placement : {'centers','endpoints','log'}
                    'log' creates log-spaced *endpoints* (requires strictly positive bounds).

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
            elif placement == 'log':
                # Log-like spacing for any sign pattern:
                # - L>0,U>0: standard logspace(L..U)
                # - L<0,U<0: logspace on |bounds| then negate
                # - L<0<U:   symmetric log density around 0 (no exact 0 included)

                # guard for degenerate interval
                if not (U > L):
                    raise ValueError(f"Degenerate bounds [{L}, {U}] on dim {i}.")

                tiny = 1e-12
                # scale epsilon to the interval size so we don't collapse points
                eps = max(tiny, min(abs(L) if L != 0 else np.inf, abs(U) if U != 0 else np.inf) * 1e-9)

                if L > 0 and U > 0:
                    # standard positive logspace, include endpoints
                    ax = np.logspace(np.log10(L), np.log10(U), m)

                elif U < 0 and L < 0:
                    # fully negative range: build on magnitudes and negate
                    # use endpoints |U| (closest to 0) .. |L| (largest magnitude), then negate & sort ascending
                    mags = np.logspace(np.log10(abs(U)), np.log10(abs(L)), m)
                    ax = -mags
                    ax = np.sort(ax)  # from most negative (L) to least negative (U)

                else:
                    # range crosses zero: split points between negative and positive sides
                    # allocate roughly half/half, ensuring both sides get at least 1 point
                    m_pos = max(1, int(np.ceil(m / 2)))
                    m_neg = max(1, m - m_pos)

                    # positive side: from small epsilon up to U (avoid exact 0)
                    # ensure start < stop in logspace
                    start_pos = max(eps, min(U, 1.0) * eps)
                    pos = np.logspace(np.log10(start_pos), np.log10(U), m_pos)

                    # negative side: from L up to small magnitude (avoid exact 0)
                    # build magnitudes then negate and put in ascending order
                    start_neg_mag = max(eps, min(abs(L), 1.0) * eps)
                    neg_mag = np.logspace(np.log10(start_neg_mag), np.log10(abs(L)), m_neg)
                    neg = -neg_mag[::-1]  # from L (most negative) towards -small

                    ax = np.concatenate([neg, pos])

                ax = np.asarray(ax, dtype=float)

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

        # row: dim_x, column: grid number, matching matlab style
        points = points.T
        for i, ax in enumerate(axes):
            axes[i] = np.asarray(ax).reshape(1, -1)   # shape: (len(ax), 1)

        return axes, points

    # ---------- transition builder ----------
    @staticmethod
    def transition_matrix_nd_separable(
            sys,
            X_axes: List[Array],
            U_points: Array,
            X_poly,
            tol: float = 1e-19,
            renormalize: bool = True,
            return_flat: bool = True,
            mode: Literal['1d', '2d', 'nd'] = '1d',
    ) -> Array:

        import numpy as _np


        A = _np.asarray(sys.A, dtype=float)
        B = _np.asarray(sys.B, dtype=float)     # (n, m)
        Bw = _np.asarray(sys.Bw, dtype=float)
        mu_w = _np.asarray(sys.mu, dtype=float).reshape(-1, 1)
        Sigma_w = _np.asarray(sys.sigma, dtype=float)


        if mode == '1d':
            # transform state space
            sigma_z = _np.asarray(sys.Bw * Sigma_w * sys.Bw.T, dtype=float)
            mu_z = _np.asarray(sys.Bw * sys.mu, dtype=float)
            hx = _np.asarray(X_axes, dtype=float)[0] # state points, each grid -> column
            hx_arr = np.asarray(hx, dtype=float).ravel()
            # compute uniform grids
            mids = 0.5 * (hx_arr[:-1] + hx_arr[1:])  # (N-1,)
            left = hx_arr[0] - 0.5 * (hx_arr[1] - hx_arr[0])
            right = hx_arr[-1] + 0.5 * (hx_arr[-1] - hx_arr[-2])
            #compute mx as the bounds of hx  N+1
            mxT = np.concatenate(([left], mids, [right]))  # (N+1,)
            mx = mxT.reshape(1,-1)

            XhatSpace = hx

            N = hx.shape[1]
            M = U_points.shape[1]
            #initialize P_blocks : (N,N,M) N is number of grid points of state, M is .. of  input
            P_blocks = np.zeros((N, N, M), dtype=float)
            dim_i = 1

            from scipy.stats import norm

            for k in range(1, M+1):  # # loop layer (1): i1 = 1:length(uhat)=M; use k-1 for indexing
                for index in range(1, N+1):  # loop layer (2): i2 = 1:length(XhatSpace)=N; use index-1 for indexing
                    xhat = XhatSpace[:,index-1] # "xhat = XhatSpace[:,i2]"
                    pij_ = np.array([1.0], dtype=float)
                    for d_index in range(dim_i): # loop layer (3): i3 = 1:dim_agent=1; use d_index for indexing
                        cdf_vals = norm.cdf(mx,A[d_index]*xhat+B[d_index]*U_points[:,k-1]+mu_z[d_index],sigma_z[d_index])
                        cpdiff = np.diff(cdf_vals)
                        cpdiff[cpdiff<tol] = 0
                        pij_ = np.kron(cpdiff,pij_)
                    P_blocks[index-1,:,k-1] = pij_

            P_blocks_t = np.zeros((N, N, M), dtype=float)
            for k in range(1, M+1):
                P_blocks_t[:,:,k-1] = P_blocks[:,:,k-1].T

            if return_flat:
                N, _, M = P_blocks_t.shape
                P_flat = np.empty((N, N * M), dtype=P_blocks_t.dtype)

                for k in range(M):
                    P_flat[:, k * N: (k + 1) * N] = P_blocks_t[:, :, k]
                return P_flat
            else:
                return P_blocks_t
        else:
            # ------ transform state space ------
            dim_i = len(X_axes)
            Sigma = Bw @ Sigma_w @ Bw.T  # (2,2)

            # SVD: NumPy returns U, singular_vals, Vh  (Vh = V^T)
            Uz, s, Vh = np.linalg.svd(Sigma, full_matrices=False)
            Sz =  np.diag(s)
            sigma_z = np.diag(Sz)
            mu_z = Uz.T @ Bw @ mu_w

            if np.sum(np.abs(mu_z)) > 0:
                warnings.warn("Implementation does not hold for mu not equal to zero", UserWarning)

            V_z = np.asarray(pc.extreme(sys.X))  # (n_vert, 2)
            V_z2 = (Uz.T @ V_z.T).T  # map each vertex
            Z = pc.qhull(V_z2)

            Ver_Z= np.asarray(pc.extreme(Z))  # (n_vertices, n_dim)
            Zl = Ver_Z.min(axis=0)  # per-dimension mins
            Zu = Ver_Z.max(axis=0)

            # TODO(optional)@Ruohan: add nonuniform state gridding

            # ---- compute uniform grid ----
            l = np.array([ax.shape[1] for ax in X_axes], dtype=int)
            gridSize = (Zu - Zl) / l
            # select representative points in each in dim_i
            hz = [(Zl[d] + (0.5 + np.arange(l[d])) * gridSize[d]).reshape(1, -1) for d in range(dim_i)]
            #hz = np.array([np.asarray(ax).ravel() for ax in X_axes], dtype=object)
            # compute cartesian product of all states in a desired axis varying rate: h[0] vary first, h[1] next,..
            ZhatSpace = MDPModel.combvec_first_fast(*hz)
            XhatSpace = Uz @ ZhatSpace

            # --- precompute once before the loop ---
            Dinv = np.diag(1.0 / gridSize)  # inverse cell sizes in Z
            Zl_T = Zl[:, None]  # (2,1)
            eps = 1e-12 * float(np.max(gridSize))  # edge epsilon
            l_tuple = tuple(l)  # for ravel_multi_index

            #compute deterministic probability matrix P_det (sparse)
            nXhatSpace = XhatSpace.shape[1]
            nUhat = U_points.shape[1]
            index_total = np.arange(ZhatSpace.shape[1], dtype=int)[None, :]
            i_indices = np.empty((1, 0), dtype=int)  # 1×0 row
            j_indices = np.empty((1, 0), dtype=int)

            for k in range(nUhat):
                z_n = Uz.T @ MDPModel.f_det(sys, XhatSpace, MDPModel.repmat_vector_col(U_points[:,k], n_cols=nXhatSpace) )
                gridSize_inv = 1.0 / gridSize
                diag_gridSize_inv = np.diag(gridSize_inv)
                Zl_T = Zl[:, None]
                diff_zn = z_n - Zl_T
                z_n_ind = np.zeros((dim_i,1),dtype=int) + np.floor(diag_gridSize_inv @ diff_zn)
                l_T = l[:, None]
                valid = (z_n_ind >= 0) & (z_n_ind <= (l - 1)[:, None])
                indices = np.all(valid, axis=0)
                indices = indices.reshape(1, -1)
                mask = np.ravel(indices).astype(bool)  # shape (N,)
                zi_indices_2T = [z_n_ind[d, mask] for d in range(dim_i)]
                zi_indices = [np.asarray(zi, dtype=np.int64).reshape(1, -1) for zi in zi_indices_2T]
                lin0 = np.ravel_multi_index((zi_indices[0], zi_indices[1]),dims=tuple(l), order='F')
                i_indices = np.hstack((i_indices, lin0))
                cols = index_total.ravel()[np.ravel(indices).astype(bool)]+ k * nXhatSpace
                cols_row = cols.reshape(1, -1)
                j_indices = np.hstack((j_indices, cols_row))



            rows = np.asarray(i_indices, dtype=int).ravel()
            cols = np.asarray(j_indices, dtype=int).ravel()

            data = np.ones(rows.size, dtype=float)
            P_det = coo_matrix((data, (rows, cols)), shape=(nXhatSpace, nXhatSpace * nUhat)).tocsr()


            #compute stochastic probability matrix Pstoch
            mx2 = [((np.arange(int(li) + 1) - 0.5) * float(gs)).reshape(1, -1) for gs, li in zip(gridSize, l)]
            from scipy.stats import norm
            from scipy.linalg import toeplitz
            P = []  # Python list = MATLAB cell
            for d in range(dim_i):
                edges = np.asarray(mx2[d]).ravel()  # (L_d+1,)
                mu = float(np.asarray(mu_z).ravel()[d])
                sig = float(np.asarray(sigma_z).ravel()[d])

                cdf = norm.cdf(edges, loc=mu, scale=sig)  # (L_d+1,)
                cp = np.diff(cdf)  # (L_d,)

                cp[cp < tol] = 0.0

                P_d = toeplitz(cp)  # (L_d, L_d)
                P.append(P_d)

            # TODO@Ruohan for n>2 d

            from .utils.tensor_transition_probability_2d import TransitionProbability2D
            if mode == '2d':
                tP_2d = TransitionProbability2D(l, P_det, P[0], P[1])
            else:
                raise NotImplementedError("compute_P='n>2 d' not implemented yet (tensor path).")

            ff2n_comp_T = (MDPModel.ff2n(dim_i)-0.5).T
            poly_v = (np.diag(2*gridSize) @ ff2n_comp_T ).T
            p_V = np.asarray(poly_v)  # shape (4, 2) vertices
            poly_comp = qhull(p_V)
            beta = MDPModel.linear_image_vertices(poly_comp, Uz)
            return tP_2d, hz, XhatSpace, beta, sys, ZhatSpace, Uz


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

