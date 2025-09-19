# -*- coding: utf-8 -*-
# linmodel.py
# Definition of LinModel class in Python

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any
import numpy as np


Array = np.ndarray


# linmodel.py (add near the top-level helpers)
def _as_col(a, n_expected: int | None = None) -> np.ndarray:
    """Return a as a column vector (n,1). If it's (1,n_expected), transpose."""
    arr = np.array(a, dtype=float, copy=False)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        # if given as a row (1, n_expected), flip to (n_expected, 1)
        if arr.shape[0] == 1 and (n_expected is None or arr.shape[1] == n_expected):
            return arr.T
        return arr
    # Fallback: flatten to column
    return arr.reshape(-1, 1)


@dataclass
class LinModel:
    """
    Python version of LinModel: LTI system with additive Gaussian noise

      x(k+1) = A x(k) + B u(k) + Bw w(k)
      y(k)   = C x(k) + D u(k)

    where w(k) ~ N(mu, sigma)
    """

    # --- required matrices ---
    A: Array
    B: Array
    C: Array
    D: Array
    Bw: Array

    # --- noise parameters ---
    mu: Optional[Array] = None
    sigma: Optional[Array] = None
    wsupport: Optional[Array] = None
    dim: Optional[int] = None

    # --- additional attributes (placeholders for extension) ---
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
        # Convert to numpy arrays
        self.A = np.atleast_2d(np.array(self.A, dtype=float))
        self.B = np.atleast_2d(np.array(self.B, dtype=float))
        self.C = np.atleast_2d(np.array(self.C, dtype=float))
        self.D = np.atleast_2d(np.array(self.D, dtype=float))
        self.Bw = np.atleast_2d(np.array(self.Bw, dtype=float))

        n_x = self.A.shape[0]
        if self.dim is None:
            self.dim = int(self.Bw.shape[0])  # consistent with MATLAB: dim = size(Bw,1)
        assert self.dim == n_x, "dim must match state dimension (rows of A)."

        # Handle noise parameters
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
                    +np.inf * np.ones((n_w, 1))
                ])
            else:
                self.wsupport = self._normalize_wsupport(self.wsupport, n_w)

        elif self.wsupport is not None:
            n_w = self.Bw.shape[1]
            self.wsupport = self._normalize_wsupport(self.wsupport, n_w)
        else:
            # default: deterministic (w=0)
            self.mu = np.zeros((self.Bw.shape[1], 1))
            self.sigma = np.zeros((self.Bw.shape[1], self.Bw.shape[1]))
            self.wsupport = np.hstack([
                -np.inf * np.ones((self.Bw.shape[1], 1)),
                +np.inf * np.ones((self.Bw.shape[1], 1))
            ])

    @staticmethod
    def _normalize_wsupport(wsupport: Array, n_w: int) -> Array:
        W = np.array(wsupport, dtype=float)
        if W.ndim == 1:
            W = W.reshape(1, -1)
        assert W.shape[1] == 2, "wsupport must have two columns"
        if W.shape[0] == 1 and n_w != 1:
            W = np.repeat(W, n_w, axis=0)
        assert W.shape[0] == n_w, "wsupport rows must match noise dimension n_w"
        return W

    # ---------- system dynamics ----------
    def f_det(self, x: Array, u: Array) -> Array:
        x = _as_col(x, n_expected=self.A.shape[0])
        u = _as_col(u, n_expected=self.B.shape[1])
        return self.A @ x + self.B @ u

    def f_stoch(self, x: Array, u: Array, w: Optional[Array] = None):
        x = _as_col(x, n_expected=self.A.shape[0])
        u = _as_col(u, n_expected=self.B.shape[1])

        if w is None:
            if self.sigma is None or self.mu is None:
                w = np.zeros((self.Bw.shape[1], 1))
            else:
                w = np.random.multivariate_normal(
                    mean=self.mu.ravel(),
                    cov=self.sigma,
                    size=1
                ).reshape(-1, 1)
        else:
            w = _as_col(w, n_expected=self.Bw.shape[1])

        x_next = self.A @ x + self.B @ u + self.Bw @ w
        return x_next, w

    def output(self, x: Array, u: Array) -> Array:
        x = _as_col(x, n_expected=self.A.shape[0])
        u = _as_col(u, n_expected=self.B.shape[1])
        return self.C @ x + self.D @ u

