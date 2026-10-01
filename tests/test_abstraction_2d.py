# tests/test_abstraction_2d.py
"""
Invariant tests for the 2D system abstraction (compute_P='2d'),
parametrised over three input configurations:

    1d_input              : B (2, 1),  U 1D,  nu = NU                  → NU_total = NU
    2d_input              : B (2, 2),  U 2D,  nu = [NU, NU]            → NU_total = NU·NU
    2d_input_asymmetric   : B (2, 2),  U 2D,  nu = [NU, NU+2]          → NU_total = NU·(NU+2)

Layout (confirmed empirically):
    State index F-order        :  i_source = ip + iv · Np
    P_det column F-order       :  c        = i_source + iu_flat · N
    Input flat-index C-order   :  iu_flat  = iu_0 · NU_1 + iu_1        (multi-D input)
    sysAbs[d].inputs.shape     :  (input_dim, NU_total)

The 2D MDP abstraction stores its transition kernel in factored form:
    sysAbs[d].P.Pi[0]   : (Np, Np)        — position noise kernel
    sysAbs[d].P.Pi[1]   : (Nv, Nv)        — velocity noise kernel
    sysAbs[d].P.P_det   : (N, N·NU_total) — sparse indicator of A x + B u shift
    sysAbs[d].P.stoch   : V @ ·           — composed operator endpoint,
                                            returns shape (N · NU_total,)

The composed operator is:
    V @ P.stoch  ==  vec(Pi[0].T · reshape(V, (Np, Nv), order='F') · Pi[1]) · P_det

Tests:
    test_grid_topology                  : N == Np · Nv per agent; P.l matches.
    test_kernels_column_stochastic      : Pi[0], Pi[1] column-stochastic in
                                          the interior; boundary columns
                                          retain exactly the in-box Gaussian
                                          mass for the configured std.
    test_P_det_indicator_structure      : (N, N · NU_total) sparse, every
                                          column has at most one nonzero of
                                          value 1.
    test_full_kernel_row_sums           : V_ones @ P.stoch ∈ [0, 1]; zeros
                                          coincide with out-of-box P_det
                                          columns; interior dominates.
    test_labels_position_only           : Position-only APs are constant
                                          along the velocity axis under
                                          F-order reshape.
    test_single_source_is_outer_product : Noise-smearing of a single-cell
                                          indicator equals Pi[0][ip,:] ⊗
                                          Pi[1][iv,:] to fp precision.
    test_modal_successor                : For interior (ip, iv, iu), the
                                          unique nonzero row of P_det's
                                          (i_source + iu · N) column sits at
                                          the cell containing A·x + B·u.
    test_position_shifts_with_velocity  : Holding (ip, iu) fixed and
                                          increasing iv by Δiv, the modal
                                          successor position shifts by
                                          T · Δv to within one cell.

Run from the repo root:
    pytest tests/test_abstraction_2d.py -v
"""

from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import polytope as pc
import pytest
from scipy.stats import norm

from src.models.linmodel               import LinModel
from src.pipeline                      import build_regions_from_cfg, prepare_pipeline
from src.specifications.translate      import translate
from src.specifications.utils.dfa_tool import dfa_manipulation


# --------------------------------------------------------------------------
# Test configuration
# --------------------------------------------------------------------------
NP, NV = 20, 10
NU     = 5
SAMP_T = 0.5
NOISE  = 0.5

POSITION_LO, POSITION_HI = -20.0, 5.0
VELOCITY_LO, VELOCITY_HI =  -5.0, 5.0
INPUT_LO,    INPUT_HI    =  -2.0, 2.0


def _retention(lo, hi, n, std):
    """Per cell i of a uniform grid of n cells on [lo, hi]: the mass that a
    Gaussian centred on the cell centre keeps inside [lo, hi]. This is the
    exact column sum of the per-axis noise kernel (Gaussian gridding at cell
    centres). With std = NOISE = 0.5 the outermost cells retain 0.8944
    (position, h = 1.25) and 0.8413 (velocity, h = 1.0); the next cells in
    retain 1 - 8.8e-5 and 1 - 1.3e-3. Before the noise-scale fix (dcf8cdd)
    the builder received the variance 0.25 as std, the edge cells retained
    0.994 / 0.977 and everything inside was 1 to 1e-9, which is why the old
    round tolerances (5 % at the edge, 1e-9 inside) used to pass."""
    h = (hi - lo) / n
    c = lo + h * (np.arange(n) + 0.5)
    return norm.cdf((hi - c) / std) - norm.cdf((lo - c) / std)


RET = [_retention(POSITION_LO, POSITION_HI, NP, NOISE),   # axis 0: position, (NP,)
       _retention(VELOCITY_LO, VELOCITY_HI, NV, NOISE)]   # axis 1: velocity, (NV,)
RETAIN = [RET[0][0], RET[1][0]]                            # edge cells


# --------------------------------------------------------------------------
# Layout helpers (F-order state index, F-order P_det column index,
# half-open cell-boundary tie-breaking for cell lookup)
# --------------------------------------------------------------------------
def _state_index(ip, iv, Np):
    """Flat state index: i = ip + iv · Np."""
    return int(ip + iv * Np)


def _Pdet_column(i_source, iu_flat, N):
    """Flat P_det column index: c = i_source + iu_flat · N."""
    return int(i_source + iu_flat * N)


def _cell_of(val, centres):
    """
    Cell index containing `val` under half-open [lo + i·dx, lo + (i+1)·dx)
    cell intervals, with lo = centres[0] - dx/2. Matches P_det's tie-breaking
    convention at exact cell boundaries (e.g. v = 0 with centres at ±0.5
    lands in the upper cell, not the lower one as argmin-of-distance would).
    """
    centres = np.asarray(centres)
    dx = float(centres[1] - centres[0])
    lo = float(centres[0]) - dx / 2.0
    i  = int(np.floor((val - lo) / dx))
    return max(0, min(centres.size - 1, i))


# --------------------------------------------------------------------------
# Parametrised fixture: 1D-input, symmetric 2D-input, asymmetric 2D-input
# --------------------------------------------------------------------------
@pytest.fixture(
    scope="module",
    params=["1d_input", "2d_input", "2d_input_asymmetric"],
)
def sys2d(request):
    A = np.array([[1.0, SAMP_T], [0.0, 1.0]])

    if request.param == "1d_input":
        B        = np.array([[0.0], [SAMP_T]])
        U        = pc.box2poly([[INPUT_LO, INPUT_HI]])
        D        = np.array([[0.0]])
        nu_arg   = NU
        NU_total = NU
    elif request.param == "2d_input":
        B        = SAMP_T * np.eye(2)
        U        = pc.box2poly([[INPUT_LO, INPUT_HI],
                                 [INPUT_LO, INPUT_HI]])
        D        = np.zeros((1, 2))
        nu_arg   = [NU, NU]
        NU_total = NU * NU
    else:  # "2d_input_asymmetric"
        B        = SAMP_T * np.eye(2)
        U        = pc.box2poly([[INPUT_LO, INPUT_HI],
                                 [INPUT_LO, INPUT_HI]])
        D        = np.zeros((1, 2))
        nu_arg   = [NU, NU + 2]
        NU_total = NU * (NU + 2)

    C  = np.array([[1.0, 0.0]])
    Bw = NOISE * np.eye(2)
    mu = np.array([[0.0], [0.0]])
    sigma = np.eye(2)

    region_intervals = {
        "p1": (  0.0,   5.0),
        "p2": ( -5.0,   0.0),
        "p3": (-20.0, -15.0),
    }
    S_1D = np.array([[1.0], [-1.0]])
    regions_b = {k: np.array([hi, -lo]) for k, (lo, hi) in region_intervals.items()}
    regions   = build_regions_from_cfg(S_1D, regions_b)
    ap_by_agent = {0: ["p1", "p2"], 1: ["p3"]}

    sysLTI = {}
    for k in [0, 1]:
        sysLTI[k] = LinModel(A, B, C, D, Bw, mu=mu, sigma=sigma)
        sysLTI[k].X = pc.box2poly([[POSITION_LO, POSITION_HI],
                                    [VELOCITY_LO, VELOCITY_HI]])
        sysLTI[k].U = U
        sysLTI[k].regions = [regions[ap] for ap in ap_by_agent[k]]
        sysLTI[k].AP      = list(ap_by_agent[k])

    DFA = translate("((!p2 | !p3) U p1)")
    DFA, letters = dfa_manipulation(
        DFA, index_base=0, remove_qf_self_loop=True,
        replacements={"!p1 & !p2": "p3&!p1&!p2", "!p1 & !p3": "!p3&!p1"},
        desired_order=["p1", "p3&!p1&!p2", "!p3&!p1"],
    )

    abs_cfg = SimpleNamespace(
        nx=[NP, NV], nu=nu_arg,
        placement="centers", u_placement="endpoints",
        tol=1e-19, contract_sum=None, compute_P="2d",
    )

    sysAbs, L, L_list, pol, rho, nx_list = prepare_pipeline(
        sysLTI=sysLTI, DFA=DFA, letters=letters, abs_cfg=abs_cfg, eps_val=0.0,
    )

    return SimpleNamespace(
        sysAbs    = sysAbs,
        L         = L,
        letters   = letters,
        Np        = NP,
        Nv        = NV,
        NU        = NU,
        NU_total  = NU_total,
        N         = NP * NV,
        samp_T    = SAMP_T,
        A         = A,
        B         = B,
        input_dim = B.shape[1],
        param     = request.param,
        agents    = [0, 1],
        dp_cell   = (POSITION_HI - POSITION_LO) / NP,
        dv_cell   = (VELOCITY_HI - VELOCITY_LO) / NV,
    )


# --------------------------------------------------------------------------
# 1.  Grid topology
# --------------------------------------------------------------------------
def test_grid_topology(sys2d):
    for d in sys2d.agents:
        N_d = int(sys2d.sysAbs[d].N)
        assert N_d == sys2d.N, (
            f"[{sys2d.param}] agent {d}: |X̂| = {N_d}, "
            f"expected Np·Nv = {sys2d.N}"
        )
        assert tuple(sys2d.sysAbs[d].P.l) == (sys2d.Np, sys2d.Nv), (
            f"[{sys2d.param}] agent {d}: P.l = {tuple(sys2d.sysAbs[d].P.l)}, "
            f"expected ({sys2d.Np}, {sys2d.Nv})"
        )


# --------------------------------------------------------------------------
# 2.  Per-axis kernels: column-stochastic in the interior; bounded leak at
#     the two boundary columns of each axis retain exactly the Gaussian
#     mass that stays in the box (RETAIN[k]).
# --------------------------------------------------------------------------
def test_kernels_column_stochastic(sys2d):
    for d in sys2d.agents:
        for k, Pi in enumerate(sys2d.sysAbs[d].P.Pi):
            col_sums = np.asarray(Pi).sum(axis=0)

            assert (col_sums <= 1.0 + 1e-9).all(), (
                f"[{sys2d.param}] agent {d}, Pi[{k}]: "
                f"col sum > 1 (max {col_sums.max():.9f})"
            )
            assert np.allclose(col_sums, RET[k], atol=1e-9), (
                f"[{sys2d.param}] agent {d}, Pi[{k}]: column sums differ from "
                f"the exact in-box Gaussian mass (max |diff| "
                f"{np.abs(col_sums - RET[k]).max():.3e}; edge expected "
                f"{RET[k][0]:.6f}, got {col_sums[0]:.6f}; std {NOISE})"
            )


# --------------------------------------------------------------------------
# 3.  P_det is a (N, N · NU_total) sparse 0-1 indicator
# --------------------------------------------------------------------------
def test_P_det_indicator_structure(sys2d):
    for d in sys2d.agents:
        Pd = sys2d.sysAbs[d].P.P_det
        assert Pd.shape == (sys2d.N, sys2d.N * sys2d.NU_total), (
            f"[{sys2d.param}] agent {d}: P_det shape {Pd.shape}, "
            f"expected ({sys2d.N}, {sys2d.N * sys2d.NU_total})"
        )

        col_nnz  = np.asarray((Pd != 0).sum(axis=0)).ravel()
        col_sums = np.asarray(Pd.sum(axis=0)).ravel()

        assert set(col_nnz.tolist()).issubset({0, 1}), (
            f"[{sys2d.param}] agent {d}: P_det columns have nnz ∉ {{0, 1}}: "
            f"{sorted(set(col_nnz.tolist()))}"
        )

        nonzero_vals = col_sums[col_nnz == 1]
        assert np.allclose(nonzero_vals, 1.0, atol=1e-12), (
            f"[{sys2d.param}] agent {d}: P_det nonzero entries are not all 1 "
            f"(range [{nonzero_vals.min()}, {nonzero_vals.max()}])"
        )

        frac_inbox = (col_nnz == 1).mean()
        assert frac_inbox > 0.7, (
            f"[{sys2d.param}] agent {d}: only {frac_inbox*100:.1f}% of "
            f"(state, input) pairs land in-box — far too much leakage"
        )


# --------------------------------------------------------------------------
# 4.  V_ones @ P.stoch lies in [0, 1]; zeros match out-of-box P_det columns;
#     the interior dominates.
# --------------------------------------------------------------------------
def test_full_kernel_row_sums(sys2d):
    for d in sys2d.agents:
        P   = sys2d.sysAbs[d].P
        out = np.ones(sys2d.N) @ P.stoch

        assert out.shape == (sys2d.N * sys2d.NU_total,), (
            f"[{sys2d.param}] agent {d}: V @ P.stoch shape {out.shape}, "
            f"expected ({sys2d.N * sys2d.NU_total},)"
        )
        assert (out >= -1e-12).all(), (
            f"[{sys2d.param}] agent {d}: negative row sum entries"
        )
        assert (out <=  1.0 + 1e-9).all(), (
            f"[{sys2d.param}] agent {d}: row sum > 1 (max {out.max():.9f})"
        )

        col_nnz    = np.asarray((P.P_det != 0).sum(axis=0)).ravel()
        out_of_box = (col_nnz == 0)

        assert np.allclose(out[out_of_box], 0.0, atol=1e-12), (
            f"[{sys2d.param}] agent {d}: out-of-box P_det columns produced "
            f"nonzero row sums"
        )

        # In-box pairs: the row sum is the product of the two per-axis
        # retentions at the modal successor cell (A x + B u lands there and the
        # noise kernels are applied around it).
        Pd_csc  = P.P_det.tocsc()
        inbox   = np.flatnonzero(~out_of_box)
        targets = np.array([Pd_csc.indices[Pd_csc.indptr[c]] for c in inbox])
        ip_t, iv_t = targets % sys2d.Np, targets // sys2d.Np
        expected = RET[0][ip_t] * RET[1][iv_t]
        assert np.allclose(out[inbox], expected, atol=1e-9), (
            f"[{sys2d.param}] agent {d}: in-box row sums differ from the "
            f"product of per-axis retentions at the modal successor "
            f"(max |diff| {np.abs(out[inbox] - expected).max():.3e}, "
            f"min expected {expected.min():.6f})"
        )
        frac_unity = float(np.isclose(out[inbox], 1.0, atol=1e-9).mean())
        assert frac_unity > 0.3, (
            f"[{sys2d.param}] agent {d}: only {frac_unity*100:.1f}% of "
            f"in-box pairs have row sum exactly 1"
        )


# --------------------------------------------------------------------------
# 5.  Labels are velocity-invariant for position-only APs (F-order reshape)
# --------------------------------------------------------------------------
def test_labels_position_only(sys2d):
    for d in sys2d.agents:
        L_d = np.asarray(sys2d.L[d])
        for l_idx, letter in enumerate(sys2d.letters):
            grid = L_d[l_idx, :].reshape(sys2d.Np, sys2d.Nv, order="F")
            max_diff = float(np.abs(grid - grid[:, 0:1]).max())
            assert max_diff == 0.0, (
                f"[{sys2d.param}] agent {d}, letter '{letter}' "
                f"velocity-dependent under F-order reshape "
                f"(max diff = {max_diff:.3g})"
            )


# --------------------------------------------------------------------------
# 6.  Single-source noise spread equals Pi[0][ip, :] ⊗ Pi[1][iv, :]
# --------------------------------------------------------------------------
def test_single_source_is_outer_product(sys2d):
    P   = sys2d.sysAbs[0].P
    Pi0 = np.asarray(P.Pi[0])
    Pi1 = np.asarray(P.Pi[1])

    rng = np.random.default_rng(0)
    for _ in range(4):
        ip = int(rng.integers(sys2d.Np // 4, 3 * sys2d.Np // 4))
        iv = int(rng.integers(sys2d.Nv // 4, 3 * sys2d.Nv // 4))

        V_grid = np.zeros((sys2d.Np, sys2d.Nv))
        V_grid[ip, iv] = 1.0

        VI_grid  = (Pi0.T @ V_grid) @ Pi1
        expected = Pi0[ip, :][:, None] * Pi1[iv, :][None, :]

        err = float(np.abs(VI_grid - expected).max())
        assert err < 1e-12, (
            f"[{sys2d.param}] single-source noise spread is not "
            f"Pi[0][{ip}, :] ⊗ Pi[1][{iv}, :] (max err = {err:.3e})"
        )


# --------------------------------------------------------------------------
# 7.  Modal successor matches the deterministic shift A x + B u
#     Uses the vector form so it covers both 1D and 2D input uniformly.
# --------------------------------------------------------------------------
def test_modal_successor(sys2d):
    Pd          = sys2d.sysAbs[0].P.P_det.toarray()
    pos_centres = np.asarray(sys2d.sysAbs[0].hx[0])
    vel_centres = np.asarray(sys2d.sysAbs[0].hx[1])
    inputs      = np.asarray(sys2d.sysAbs[0].inputs)

    assert inputs.shape == (sys2d.input_dim, sys2d.NU_total), (
        f"[{sys2d.param}] inputs.shape {inputs.shape} != "
        f"({sys2d.input_dim}, {sys2d.NU_total})"
    )

    Np, Nv, NU_total, N = sys2d.Np, sys2d.Nv, sys2d.NU_total, sys2d.N
    A, B = sys2d.A, sys2d.B

    rng      = np.random.default_rng(0)
    misses   = []
    n_trials = 24

    for _ in range(n_trials):
        ip = int(rng.integers(Np // 3, 2 * Np // 3))
        iv = int(rng.integers(Nv // 3, 2 * Nv // 3))
        iu = int(rng.integers(NU_total))

        i_source = _state_index(ip, iv, Np)
        c        = _Pdet_column(i_source, iu, N)

        x     = np.array([pos_centres[ip], vel_centres[iv]])
        u     = inputs[:, iu]                                # (input_dim,)
        x_pre = A @ x + B @ u                                # (2,)
        p_pre, v_pre = float(x_pre[0]), float(x_pre[1])

        ip_t = _cell_of(p_pre, pos_centres)
        iv_t = _cell_of(v_pre, vel_centres)
        i_t  = _state_index(ip_t, iv_t, Np)

        nz = np.where(Pd[:, c] > 0.5)[0]
        if len(nz) != 1 or int(nz[0]) != i_t:
            misses.append((ip, iv, iu, p_pre, v_pre, ip_t, iv_t, i_t, nz.tolist()))

    if misses:
        msg = "\n".join(
            f"  src=(ip={m[0]},iv={m[1]}) iu={m[2]} "
            f"shift=({m[3]:+.3f},{m[4]:+.3f}) → "
            f"predicted (ip_t={m[5]},iv_t={m[6]}, i_t={m[7]}), "
            f"actual nz={m[8]}"
            for m in misses
        )
        raise AssertionError(
            f"[{sys2d.param}] modal-successor mismatch in "
            f"{len(misses)}/{n_trials} interior trials:\n{msg}"
        )


# --------------------------------------------------------------------------
# 8.  Position shifts with velocity by T · Δv
#     Hold (ip, iu) fixed and vary iv. Position shift due to A·x is T·Δv;
#     B·u is independent of iv. So Δp_modal == T·Δv to within one cell,
#     regardless of input dimension.
# --------------------------------------------------------------------------
def test_position_shifts_with_velocity(sys2d):
    Pd          = sys2d.sysAbs[0].P.P_det.toarray()
    pos_centres = np.asarray(sys2d.sysAbs[0].hx[0])
    vel_centres = np.asarray(sys2d.sysAbs[0].hx[1])
    Np, Nv, NU_total, N, T = (
        sys2d.Np, sys2d.Nv, sys2d.NU_total, sys2d.N, sys2d.samp_T
    )
    dp = float(pos_centres[1] - pos_centres[0])

    ip    = Np // 2
    iu    = NU_total // 2
    iv_lo = Nv // 4
    iv_hi = 3 * Nv // 4

    def modal_ip(iv):
        c  = _Pdet_column(_state_index(ip, iv, Np), iu, N)
        nz = np.where(Pd[:, c] > 0.5)[0]
        assert len(nz) == 1, (
            f"[{sys2d.param}] P_det column {c} (ip={ip}, iv={iv}, iu={iu}) "
            f"does not have a unique nonzero row (nz={nz.tolist()})"
        )
        return int(nz[0]) % Np

    ip_lo = modal_ip(iv_lo)
    ip_hi = modal_ip(iv_hi)

    expected_shift = T * (vel_centres[iv_hi] - vel_centres[iv_lo])
    actual_shift   = pos_centres[ip_hi] - pos_centres[ip_lo]

    assert abs(actual_shift - expected_shift) <= dp, (
        f"[{sys2d.param}] position shift mismatch: "
        f"expected T·Δv = {expected_shift:.3f}, "
        f"observed Δp_modal = {actual_shift:.3f}, "
        f"cell width = {dp:.3f}"
    )