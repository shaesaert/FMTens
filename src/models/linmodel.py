"""
LinModel: discrete-time LTI system with additive Gaussian noise.

    x(k+1) = A x(k) + B u(k) + Bw w(k)
    y(k)   = C x(k) + D u(k)

where w(k) ~ N(mu, sigma). The class also accepts bounded-support noise
(via ``wsupport``) and deterministic systems (no noise at all).

Used as the continuous-time model passed to :class:`MDPModel.from_system`
for abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

Array = np.ndarray


def _as_col(a, n_expected: int | None = None) -> np.ndarray:
    """
    Return ``a`` as a column vector ``(n, 1)``.

    If ``a`` is a row vector ``(1, n_expected)``, it is transposed.
    1D inputs are reshaped to column form. Other 2D shapes are passed
    through unchanged.
    """
    arr = np.array(a, dtype=float, copy=False)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        if arr.shape[0] == 1 and (n_expected is None or arr.shape[1] == n_expected):
            return arr.T
        return arr
    return arr.reshape(-1, 1)


@dataclass
class LinModel:
    """
    Discrete-time linear system with additive noise.

        x(k+1) = A x(k) + B u(k) + Bw w(k)
        y(k)   = C x(k) + D u(k)

    Parameters
    ----------
    A, B, C, D, Bw : np.ndarray
        State, input, output, feedthrough, and noise matrices. Coerced
        to 2D float arrays in ``__post_init__``.
    mu : np.ndarray, optional
        Mean of the noise ``w``. If None and ``wsupport`` is also None,
        the system is treated as deterministic.
    sigma : np.ndarray, optional
        Covariance of ``w``. Defaults to identity if ``mu`` is given
        but ``sigma`` is omitted.
    wsupport : np.ndarray, optional
        Bounded support for ``w``, shape ``(n_w, 2)`` with columns
        ``(lower, upper)``. Defaults to ``[-inf, +inf]`` per dimension.
    dim : int, optional
        State dimension. Inferred from ``Bw.shape[0]`` if not given;
        must match ``A.shape[0]``.

    Additional fields
    -----------------
    type : str
        Model-type tag; defaults to ``"LTI"``.
    X, U : polytope.Polytope, optional
        State and input polytopes used by the abstraction layer.
    regions, AP : list, optional
        Atomic propositions and their corresponding regions.
    MOR, P, Q, original :
        Model-order-reduction placeholders.
    KKfilter, InitState, Xdare, K, Cobs :
        Kalman-filter extension placeholders.
    """

    # === required matrices ================================================
    A: Array
    B: Array
    C: Array
    D: Array
    Bw: Array

    # === noise parameters =================================================
    mu: Optional[Array] = None
    sigma: Optional[Array] = None
    wsupport: Optional[Array] = None
    dim: Optional[int] = None

    # === additional attributes (placeholders for extensions) ==============
    type: str = "LTI"
    X: Any = None
    U: Any = None
    regions: Optional[List[Any]] = field(default_factory=list)
    AP: Optional[List[Any]] = field(default_factory=list)

    MOR: bool = False
    P: Optional[Array] = None
    Q: Optional[Array] = None
    original: Optional["LinModel"] = None

    KKfilter: bool = False
    InitState: Optional[Tuple[Array, Array]] = None
    Xdare: Optional[Array] = None
    K: Optional[Array] = None
    Cobs: Optional[Array] = None

    def __post_init__(self):
        # Coerce all matrices to 2D float arrays.
        self.A = np.atleast_2d(np.array(self.A, dtype=float))
        self.B = np.atleast_2d(np.array(self.B, dtype=float))
        self.C = np.atleast_2d(np.array(self.C, dtype=float))
        self.D = np.atleast_2d(np.array(self.D, dtype=float))
        self.Bw = np.atleast_2d(np.array(self.Bw, dtype=float))

        # State-dimension check.
        n_x = self.A.shape[0]
        if self.dim is None:
            self.dim = int(self.Bw.shape[0])  # MATLAB convention: dim = size(Bw, 1)
        assert self.dim == n_x, "dim must match state dimension (rows of A)."

        # Noise initialisation, three cases:
        #   1) mu given (with optional sigma, wsupport)
        #   2) wsupport given but no mu  (set-based bounds, no stochastic noise)
        #   3) neither given             (fully deterministic system)
        if self.mu is not None:
            self.mu = np.array(self.mu, dtype=float).reshape(-1, 1)
            n_w = self.mu.shape[0]
            if self.sigma is None:
                self.sigma = np.eye(n_w)
            else:
                self.sigma = np.array(self.sigma, dtype=float)
                assert self.sigma.shape == (n_w, n_w), "sigma must be (n_w, n_w)."

            if self.wsupport is None:
                self.wsupport = np.hstack([
                    -np.inf * np.ones((n_w, 1)),
                    +np.inf * np.ones((n_w, 1)),
                ])
            else:
                self.wsupport = self._normalize_wsupport(self.wsupport, n_w)

        elif self.wsupport is not None:
            n_w = self.Bw.shape[1]
            self.wsupport = self._normalize_wsupport(self.wsupport, n_w)

        else:
            n_w = self.Bw.shape[1]
            self.mu = np.zeros((n_w, 1))
            self.sigma = np.zeros((n_w, n_w))
            self.wsupport = np.hstack([
                -np.inf * np.ones((n_w, 1)),
                +np.inf * np.ones((n_w, 1)),
            ])

    @staticmethod
    def _normalize_wsupport(wsupport: Array, n_w: int) -> Array:
        """
        Normalise ``wsupport`` to shape ``(n_w, 2)``.

        If a single row is given, it is broadcast across all ``n_w``
        noise dimensions. Raises on shape mismatch.
        """
        W = np.array(wsupport, dtype=float)
        if W.ndim == 1:
            W = W.reshape(1, -1)
        assert W.shape[1] == 2, "wsupport must have two columns"
        if W.shape[0] == 1 and n_w != 1:
            W = np.repeat(W, n_w, axis=0)
        assert W.shape[0] == n_w, "wsupport rows must match noise dimension n_w"
        return W

    # === system dynamics =================================================

    def f_det(self, x: Array, u: Array) -> Array:
        """
        Deterministic next-state map: ``A x + B u``.

        ``x`` and ``u`` may be row or column vectors; they are
        normalised to columns internally.
        """
        x = _as_col(x, n_expected=self.A.shape[0])
        u = _as_col(u, n_expected=self.B.shape[1])
        return self.A @ x + self.B @ u

    def f_stoch(self, x: Array, u: Array, w: Optional[Array] = None):
        """
        Stochastic next-state map with noise sample ``w``.

        If ``w`` is None, a sample is drawn from ``N(mu, sigma)``. If
        both are unset (deterministic system), zero noise is used.
        Returns the tuple ``(x_next, w)`` so callers can record the
        sampled noise.
        """
        x = _as_col(x, n_expected=self.A.shape[0])
        u = _as_col(u, n_expected=self.B.shape[1])

        if w is None:
            if self.sigma is None or self.mu is None:
                w = np.zeros((self.Bw.shape[1], 1))
            else:
                w = np.random.multivariate_normal(
                    mean=self.mu.ravel(),
                    cov=self.sigma,
                    size=1,
                ).reshape(-1, 1)
        else:
            w = _as_col(w, n_expected=self.Bw.shape[1])

        x_next = self.A @ x + self.B @ u + self.Bw @ w
        return x_next, w

    def output(self, x: Array, u: Array) -> Array:
        """Output map: ``C x + D u``."""
        x = _as_col(x, n_expected=self.A.shape[0])
        u = _as_col(u, n_expected=self.B.shape[1])
        return self.C @ x + self.D @ u