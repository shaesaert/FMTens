"""
Factory for constructing an :class:`MDPModel` from a continuous-state
linear system.

The public entry point is :func:`mdp_from_system`, intended to be bound
as a classmethod on :class:`MDPModel` (e.g. ``MDPModel.from_system``).
It auto-builds the state grid, input grid, and transition matrix from a
continuous model and returns a fully-initialised :class:`MDPModel`.

Two construction paths are dispatched on ``compute_P``:
    - ``'1d'`` : builds a flat transition matrix.
    - ``'2d'`` : builds a :class:`TransitionProbability2D` operator together
      with auxiliary fields (rotated grid, output map, etc.).

The ``'nd'`` mode is reserved for higher-dimensional decompositions and
currently raises :exc:`NotImplementedError`.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from src.abstraction.utils.abstraction_grid_helper import make_uniform_grid
from src.abstraction.utils.abstraction_transition_builder import transition_matrix_nd_separable_impl


def mdp_from_system(
    cls,
    orig: object,
    nx: Union[int, List[int], Tuple[int, ...]],
    nu: Union[int, List[int], Tuple[int, ...]],
    *,
    placement: str = "centers",
    u_placement: Optional[str] = None,
    filter_inside: bool = True,
    tol: float = 1e-15,
    renormalize: bool = False,
    return_flat: bool = True,
    contract_sum: Optional[float] = None,
    contract_mode: str = "cap",
    X_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    U_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    compute_P: Literal["1d", "2d", "nd"] = "1d",
) -> "MDPModel":
    """
    Build an :class:`MDPModel` from a continuous-state linear system.

    Auto-constructs the state-space grid, input grid, and transition
    matrix from ``orig``, then wraps everything in an :class:`MDPModel`
    instance.

    Parameters
    ----------
    orig : object
        Continuous-state linear model with attributes ``A``, ``B``,
        ``Bw``, ``mu``, ``sigma``, ``X`` (state polytope), ``U`` (input
        polytope), and ``C`` (output map).
    nx : int or sequence of int
        Grid counts per state dimension. A scalar is allowed only for 1D.
    nu : int or sequence of int
        Grid counts per input dimension. A scalar is allowed only for 1D.
    placement : {"centers", "endpoints"}, optional
        Grid placement for both states and inputs. Default ``"centers"``.
    u_placement : str, optional
        If provided, overrides ``placement`` for inputs only. Accepts
        ``"centers"``, ``"endpoints"``, or ``"log"``.
    filter_inside : bool, optional
        Keep only grid points lying inside the bounding polytope. Default
        ``True``.
    tol : float, optional
        Small-probability cutoff used by the transition builder. Default
        ``1e-15``.
    renormalize : bool, optional
        Passed through to the transition builder; currently unused there.
        Default ``False``.
    return_flat : bool, optional
        Used by the 1D path. If True, the transition tensor is returned
        in flat ``(N, N*M)`` form; otherwise as block ``(N, N, M)``.
        Default ``True``.
    contract_sum : float, optional
        Currently unused. Was previously used for sub-stochastic capping
        of per-action transition blocks; the helper has been removed.
        Kept for signature compatibility.
    contract_mode : {"cap", "set"}, optional
        Currently unused; see ``contract_sum``. Default ``"cap"``.
    X_bounds : tuple of np.ndarray, optional
        ``(lower, upper)`` bounds for the state grid, overriding
        ``orig.X``. If None, uses ``orig.X``.
    U_bounds : tuple of np.ndarray, optional
        ``(lower, upper)`` bounds for the input grid, overriding
        ``orig.U``. If None, uses ``orig.U``.
    compute_P : {"1d", "2d", "nd"}, optional
        Transition matrix construction mode. Default ``"1d"``. See
        :func:`transition_matrix_nd_separable_impl` for details.

    Returns
    -------
    MDPModel
        A fully-initialised MDP abstraction. The 1D path populates
        ``P``, ``hx``, ``states``, ``orig``, ``inputs``, ``outputs``.
        The 2D path additionally populates ``zstates``, ``Uz``,
        ``outputmap``, and sets ``P.dim`` and ``P.l``.

    Raises
    ------
    NotImplementedError
        If ``compute_P='nd'``.
    ValueError
        If ``compute_P`` is none of the recognised values.
    """
    # === 1) Build state and input grids ====================================
    X_region = X_bounds if X_bounds is not None else orig.X
    U_region = U_bounds if U_bounds is not None else orig.U
    up = u_placement if u_placement is not None else placement

    X_axes, X_points = make_uniform_grid(
        X_region,
        grid_counts=nx,
        filter_inside=filter_inside,
        placement=placement,
    )
    U_axes, U_points = make_uniform_grid(
        U_region,
        grid_counts=nu,
        filter_inside=filter_inside,
        placement=up,
    )

    # Ensure U_points is 2D: (n_u, M).
    if U_points.ndim == 1:
        U_points = U_points.reshape(-1, 1)

    # === 2) Build transition matrix and wrap in MDPModel ===================
    if compute_P == "1d":
        P = transition_matrix_nd_separable_impl(
            sys=orig,
            X_axes=X_axes,
            U_points=U_points,
            X_poly=orig.X,
            tol=tol,
            renormalize=renormalize,
            return_flat=return_flat,
            mode=compute_P,
        )
        mdp = cls(P=P, hx=X_axes, states=X_points, orig=orig, inputs=U_points)
        if hasattr(mdp.orig, "C"):
            mdp.outputs = mdp.orig.C @ X_points
        return mdp

    if compute_P == "2d":
        (tP_2d, hz, XhatSpace, beta, sys_ret, ZhatSpace, Uz) = (
            transition_matrix_nd_separable_impl(
                sys=orig,
                X_axes=X_axes,
                U_points=U_points,
                X_poly=orig.X,
                tol=tol,
                renormalize=renormalize,
                return_flat=return_flat,
                mode=compute_P,
            )
        )

        mdp = cls(
            P=tP_2d, hx=hz, states=XhatSpace, beta=beta,
            orig=sys_ret, inputs=U_points,
        )
        mdp.zstates = ZhatSpace
        mdp.Uz = Uz
        if hasattr(mdp.orig, "C"):
            mdp.outputs = mdp.orig.C @ XhatSpace
            mdp.outputmap = mdp.orig.C @ Uz
        return mdp

    if compute_P == "nd":
        raise NotImplementedError(
            "compute_P='nd' is not implemented yet (tensor path)."
        )

    raise ValueError(
        f"compute_P must be '1d', '2d', or 'nd'; got {compute_P!r}."
    )