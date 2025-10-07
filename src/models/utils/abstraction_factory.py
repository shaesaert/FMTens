from __future__ import annotations
from typing import Union, List, Tuple, Optional, Literal
import numpy as np

from .abstraction_transition_builder import transition_matrix_nd_separable_impl
from .abstraction_grid_helper import make_uniform_grid

def mdp_from_system(cls,
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

        X_axes, X_points = make_uniform_grid(X_poly_or_bounds, grid_counts=nx,
                                                 filter_inside=filter_inside,
                                                 placement=placement)
        U_axes, U_points = make_uniform_grid(U_poly_or_bounds, grid_counts=nu,
                                                 filter_inside=filter_inside,
                                                 placement=up)

        # U_points shape -> (M, m)
        if U_points.ndim == 1:
            U_points = U_points.reshape(-1, 1)

        # 2) Transition matrix (if by default: compute one flattened matrix, works for 1d case)
        if compute_P == '1d':
            P = transition_matrix_nd_separable_impl(
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
            (tP_2d, hz, XhatSpace, beta, sys_ret, ZhatSpace, Uz)= transition_matrix_nd_separable_impl(
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



        return mdp

