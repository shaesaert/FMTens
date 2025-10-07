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