"""
MDPModel — discrete MDP abstraction built from a continuous system.

A finite-state MDP obtained by gridding a continuous-state linear
system (:class:`LinModel`) and building a transition matrix between
grid cells under each input letter. The MDP is the substrate on
which the tree-based value iteration (:mod:`src.dynprog`) runs.

Two construction paths:
    - Direct: pass ``P`` and ``hx`` to the constructor.
    - Via :meth:`MDPModel.from_system`: auto-build the grid, transition
      matrix, and outputs from a :class:`LinModel`. Dispatches on
      ``compute_P`` (``'1d'`` or ``'2d'``) to either a dense transition
      tensor or a separable :class:`TransitionProbability2D` operator.

Transition matrix storage:
    Dense path (1D): the matrix is stored in two cached forms,
    ``_P_flat`` of shape ``(N, N*M)`` and ``_P_blocks`` of shape
    ``(N, N, M)``, accessible read-only via :attr:`P_flat` and
    :attr:`P_blocks`. The public :attr:`P` aliases ``_P_flat``.

    Operator path (2D): :attr:`P` holds the
    :class:`TransitionProbability2D` instance itself; ``P_flat`` /
    ``P_blocks`` are ``None``, and :attr:`P_det` / :attr:`Pi` mirror
    the operator's components.

Helper namespace: the underlying construction utilities are also
exposed as static methods on the class -- :meth:`MDPModel.make_uniform_grid`
and :meth:`MDPModel.transition_matrix_nd_separable` -- so callers can
use ``MDPModel.X(...)`` namespacing alongside the free-function form
``from .utils.X import X``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .utils.abstraction_shape_helper import (
    cartesian_from_axes, flat_from_blocks, blocks_from_flat,
)
from src.abstraction.utils.abstraction_grid_helper import (
    make_uniform_grid as _make_uniform_grid,
)
from src.abstraction.utils.abstraction_transition_builder import (
    transition_matrix_nd_separable_impl as _tmat_impl,
)
from .utils.abstraction_factory import mdp_from_system as _mdp_from_system

Array = np.ndarray


@dataclass(init=False)
class MDPModel:
    """
    Discrete MDP abstraction of a continuous-state linear system.

    See module docstring for algorithmic context and construction paths.

    Attributes
    ----------
    type : str
        Tag string; defaults to ``"MDP abstract"``.
    states : np.ndarray
        Grid points of shape ``(N, n)``.
    outputs : np.ndarray, optional
        ``C @ states.T`` if ``orig`` is given and has a ``C`` matrix.
    inputs : np.ndarray, optional
        Discrete input set of shape ``(M, m)``.
    dim : int
        State dimension ``n``.
    orig : object, optional
        Original continuous model (e.g. :class:`LinModel`).
    P : np.ndarray or TransitionProbability2D
        Transition matrix: dense ``(N, N*M)`` array (1D path) or
        :class:`TransitionProbability2D` operator (2D path).
    P_det, Pi : optional
        Operator components, populated when ``P`` is a
        :class:`TransitionProbability2D`.
    hx : list of np.ndarray
        Per-dimension grid axes (1D arrays).
    l : tuple of int
        Bins per dimension.
    beta : optional
        Per-cell box vertices in the rotated frame (2D path).
    zstates : optional
        Centred grid in the rotated frame (2D path).
    Uz : optional
        Left singular vectors of the noise covariance (2D path).
    outputmap : optional
        Per-input output map (2D path).
    labels, Partition, Pxix :
        Reserved for downstream extensions.
    """

    # === public fields =====================================================
    type: str = "MDP abstract"
    states: Optional[Array] = None
    outputs: Optional[Array] = None
    inputs: Optional[Array] = None
    dim: Optional[int] = None
    orig: Optional[object] = None
    P: Optional[object] = None
    labels: Optional[list] = None
    beta: Optional[Array] = None
    hx: Optional[List[Array]] = None
    zstates: Optional[Array] = None
    Uz: Optional[Array] = None
    outputmap: Optional[object] = None
    l: Optional[Tuple[int, ...]] = None
    Partition: Optional[object] = None
    Pxix: Optional[object] = None
    P_det: Optional[object] = None
    Pi: Optional[list] = None

    # === internal cached formats (dense path only) =========================
    _P_flat: Optional[Array] = None
    _P_blocks: Optional[Array] = None

    # === constructor ========================================================
    def __init__(
        self,
        P,
        hx: List[Array],
        states: Optional[Array] = None,
        beta: Optional[Array] = None,
        orig: Optional[object] = None,
        inputs: Optional[Array] = None,
        labels: Optional[list] = None,
    ):
        """
        Build an MDPModel from a transition matrix and per-dimension axes.

        Parameters
        ----------
        P : np.ndarray or TransitionProbability2D
            Transition matrix in one of three forms:
                - dense ``(N, N*M)`` 2D ndarray
                - dense ``(N, N, M)`` 3D ndarray
                - :class:`TransitionProbability2D` operator (must
                  expose ``P_det``, ``dim``, ``l``, ``Pi``).
        hx : list of np.ndarray
            Per-dimension grid axes (1D arrays).
        states : np.ndarray, optional
            Grid points ``(N, n)``. Defaults to the Cartesian product
            of ``hx``.
        beta, orig, inputs, labels :
            Auxiliary fields; see class docstring.
        """
        # === grid / sizes =================================================
        self.hx = [np.asarray(ax, dtype=float).ravel() for ax in hx]
        self.l = tuple(len(ax) for ax in self.hx)
        self.dim = len(self.hx)

        if states is None:
            self.states = cartesian_from_axes(self.hx)
        else:
            self.states = np.asarray(states, dtype=float).reshape(-1, self.dim)

        # === inputs =======================================================
        if inputs is not None:
            U = np.asarray(inputs, dtype=float)
            if U.ndim == 1:
                U = U.reshape(-1, 1)
            self.inputs = U
        else:
            self.inputs = None

        # === transitions ==================================================
        self._P_flat = None
        self._P_blocks = None
        self.P_det = None
        self.Pi = None
        self.P = None

        if isinstance(P, np.ndarray):
            # Dense path: store both flat and block forms.
            P_arr = np.asarray(P, dtype=float)
            if P_arr.ndim == 2:
                self._P_flat = P_arr
                N = self.states.shape[0]
                self._P_blocks = blocks_from_flat(P_arr, N)
            elif P_arr.ndim == 3:
                self._P_blocks = P_arr
                self._P_flat = flat_from_blocks(P_arr)
            else:
                raise ValueError(
                    "Dense P must be shape (N, N*M) or (N, N, M); "
                    f"got {P_arr.shape}."
                )
            self.P = self._P_flat
        elif all(hasattr(P, a) for a in ("P_det", "dim", "l", "Pi")):
            # Operator path: store reference plus mirrored convenience attrs.
            self.P = P
            self.P_det = P.P_det
            try:
                self.l = tuple(int(v) for v in np.asarray(P.l).ravel())
            except Exception:
                # Keep l derived from hx if operator.l isn't cleanly coercible.
                pass
            self.Pi = list(P.Pi)
        else:
            raise TypeError(
                "P must be either a dense numpy array (N, N*M) / (N, N, M) "
                "or a TransitionProbability2D-like object exposing "
                "P_det, dim, l, Pi."
            )

        # === other fields =================================================
        self.beta = beta
        self.orig = orig
        self.labels = labels

        # outputs = C @ states  (if orig has C)
        if self.orig is not None and hasattr(self.orig, "C"):
            C = np.asarray(self.orig.C, dtype=float)
            self.outputs = (C @ self.states.T).T
        else:
            self.outputs = None

    # === alternative constructor ==========================================

    @classmethod
    def from_system(cls, *args, **kwargs):
        """
        Build an MDPModel by gridding a continuous-state linear system.

        Thin wrapper around :func:`abstraction_factory.mdp_from_system`;
        see that function for the full parameter list.
        """
        return _mdp_from_system(cls, *args, **kwargs)

    # === namespace delegators =============================================
    # Static-method aliases so callers can use ``MDPModel.X(...)`` alongside
    # ``from .utils.X import X``. Useful as a single-import namespace.
    make_uniform_grid = staticmethod(_make_uniform_grid)
    transition_matrix_nd_separable = staticmethod(_tmat_impl)

    # === properties =======================================================
    @property
    def N(self) -> int:
        """Number of states."""
        return self.states.shape[0]

    @property
    def M(self) -> int:
        """
        Number of input letters.

        Inferred from ``_P_blocks.shape[2]`` (dense path) or from
        ``P.P_det.shape[1] // P.P_det.shape[0]`` (operator path).
        Returns ``0`` if neither is available.
        """
        if self._P_blocks is not None:
            return self._P_blocks.shape[2]
        if self.P is not None and hasattr(self.P, "P_det"):
            n_rows = self.P.P_det.shape[0]
            n_cols = self.P.P_det.shape[1]
            return int(n_cols // n_rows)
        return 0

    @property
    def P_blocks(self) -> Optional[Array]:
        """Block-stacked transition tensor ``(N, N, M)``; None for the operator path."""
        return self._P_blocks

    @property
    def P_flat(self) -> Optional[Array]:
        """Flat block-column transition matrix ``(N, N*M)``; None for the operator path."""
        return self._P_flat