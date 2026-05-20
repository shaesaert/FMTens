# paper_artifacts/arxiv_2511_06873/fig3_RA2D.py
"""
Reproduces Fig. 3 of arxiv:2511.06873.

Plots two curves vs iteration count T:
    Optimal Robust-tree Value functions
    Optimal A-posteriori corrected Value functions

The two curves come from two runs of the same 2D reach-avoid sweep with
different (rt / apos) settings. Data is cached as .npz files under
`data/`; on first run any missing curve is computed and saved, then
plotted. Subsequent runs load from cache and plot instantly. Delete
`data/` to force regeneration.

Run from the repository root:
    python -m paper_artifacts.arxiv_2511_06873.fig3_RA2D
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Dict, List, Sequence

import numpy as np
import polytope as pc
import matplotlib.ticker as mticker

from src.models.linmodel               import LinModel
from src.models.mdpmodel               import MDPModel
from src.specifications.translate      import translate
from src.specifications.utils.dfa_tool import dfa_manipulation
from src.abstraction.utils.labeling    import dim_label
from src.dynprog.dfa_tree_r1           import DFATree
from src.dynprog.utils.treebasedV      import compute_tv_from_tree
from src.dynprog.utils.v_apos          import apply_delta_correction_apos


# ===========================================================================
# Paper parameters — published in arxiv:2511.06873
# ===========================================================================
NX             = 1000
NU             = 10
TMAX           = 80
SIGMA          = (1.0, 1.0)
EPS            = 0.1
PRUNE_TOL      = 1e-15
ROWS           = (0, 799)
COLS           = (600, 699)
DELTA_RT       = (0.002, 0.002)
DELTA_APOS_POL = (0.002, 0.002)
DELTA_CORR     = (0.002, 0.002)

HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


# ===========================================================================
# Sweep result type
# ===========================================================================
@dataclass
class _Result:
    mode:  str
    Ts:    np.ndarray
    curve: np.ndarray
    meta:  Dict = field(default_factory=dict)


# ===========================================================================
# Sweep — system, abstraction, tree iteration
# ===========================================================================
def _build_system_and_spec(sigma_vals: Sequence[float]):
    """Build the 2D reach-avoid system + DFA used in Fig. 3."""
    A  = {0: np.array([[0.9]]), 1: np.array([[0.9]])}
    B  = {0: np.array([[0.5]]), 1: np.array([[0.5]])}
    C  = {0: np.array([[1.0]]), 1: np.array([[1.0]])}
    Dm = {0: np.array([[0.0]]), 1: np.array([[0.0]])}
    Bw = {0: np.array([[0.5]]), 1: np.array([[0.5]])}
    mu = {0: np.array([0.0]),   1: np.array([0.0])}
    sigma = {0: float(sigma_vals[0]) * np.eye(1),
             1: float(sigma_vals[1]) * np.eye(1)}

    sysLTI = {
        0: LinModel(A[0], B[0], C[0], Dm[0], Bw[0], mu=mu[0], sigma=sigma[0]),
        1: LinModel(A[1], B[1], C[1], Dm[1], Bw[1], mu=mu[1], sigma=sigma[1]),
    }

    DFA = translate('( (!p2 | !p3 ) U p1)')
    DFA, letters = dfa_manipulation(
        DFA,
        index_base          = 0,
        ensure_transitions  = True,
        remove_qf_self_loop = True,
        replacements        = {"!p1 & !p2": "p3&!p1&!p2", "!p1 & !p3": "!p3&!p1"},
        desired_order       = ['p1', 'p3&!p1&!p2', '!p3&!p1'],
        verbose             = False,
    )

    s   = np.array([[1], [-1]])
    bx  = np.array([5, 20])
    bu  = np.array([5, 5])
    bp1 = np.array([5, 0])      # p1: [0, 5]
    bp2 = np.array([0, 5])      # p2: [-5, 0]
    bp3 = np.array([-15, 20])   # p3: [-20, -15]
    P1, P2, P3 = pc.Polytope(s, bp1), pc.Polytope(s, bp2), pc.Polytope(s, bp3)

    for i in (0, 1):
        sysLTI[i].X = pc.Polytope(s, bx)
        sysLTI[i].U = pc.Polytope(s, bu)
    sysLTI[0].regions = [P1, P2]
    sysLTI[1].regions = [P3]
    sysLTI[0].AP = ['p1', 'p2']
    sysLTI[1].AP = ['p3']

    return sysLTI, DFA, letters


def _build_abs_and_label(sysLTI, letters):
    sysAbs = {
        i: MDPModel.from_system(
            sysLTI[i], nx=NX, nu=NU,
            placement='centers', u_placement='endpoints',
            tol=1e-19, contract_sum=None, compute_P='1d',
        )
        for i in (0, 1)
    }
    L = dim_label(sysAbs, sysLTI, letters,
                  eps=EPS, visualize=False, outdir=None, prefix="L_eps")
    return sysAbs, L


def _delta_vecs(scalars: Sequence[float], sysAbs) -> List[np.ndarray]:
    keys = sorted(sysAbs.keys())
    return [np.full(sysAbs[k].N, float(scalars[i]), dtype=float)
            for i, k in enumerate(keys)]


def _region_mean(tv: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> float:
    if tv.ndim != 2:
        raise ValueError(f"expected 2D tv; got shape {tv.shape}")
    nrows, ncols = tv.shape
    rr0 = max(0, min(nrows - 1, r0))
    rr1 = max(0, min(nrows - 1, r1))
    cc0 = max(0, min(ncols - 1, c0))
    cc1 = max(0, min(ncols - 1, c1))
    if rr1 < rr0 or cc1 < cc0:
        return 0.0
    return float(np.mean(tv[rr0:rr1 + 1, cc0:cc1 + 1]))


def _run_sweep(mode: str) -> _Result:
    """
    Run one 2D reach-avoid sweep over T = 0..TMAX.
    mode: 'rt' (robust-tree) or 'apos' (a-posteriori corrected).
    """
    if mode not in ("rt", "apos"):
        raise ValueError(f"mode must be 'rt' or 'apos'; got {mode!r}")

    sysLTI, DFA, letters = _build_system_and_spec(SIGMA)
    sysAbs, L            = _build_abs_and_label(sysLTI, letters)

    dims    = sorted(sysAbs.keys())
    nx_list = [sysAbs[d].N for d in dims]
    nu_list = [sysAbs[d].M for d in dims]
    L_list  = [L[d] for d in dims]
    nQ      = len(DFA.S)

    pol = [
        [np.full((nx_list[d], nu_list[d]), 1.0 / nu_list[d], dtype=float)
         for d in range(len(dims))]
        for _ in range(nQ)
    ]
    rho = [np.full(nx_list[d], 1.0 / nx_list[d], dtype=float) for d in range(len(dims))]

    if mode == "rt":
        delta_VI_vecs  = _delta_vecs(DELTA_RT, sysAbs)
        delta_pol_vecs = _delta_vecs(DELTA_RT, sysAbs)
        corr_delta     = None
    else:
        delta_VI_vecs  = [np.zeros(sysAbs[d].N, dtype=float) for d in dims]
        delta_pol_vecs = _delta_vecs(DELTA_APOS_POL, sysAbs)
        corr_delta     = tuple(DELTA_CORR)

    G = DFATree(
        DFA, sysAbs, pol, nx_list, L_list,
        delta_VI=delta_VI_vecs,
        delta_pol=delta_pol_vecs,
        pol_mode=mode,
        VI_mode=mode,
    )
    G.initiate()

    r0, r1 = ROWS
    c0, c1 = COLS
    Ts:    List[int]   = list(range(TMAX + 1))
    curve: List[float] = []

    tv, _ = compute_tv_from_tree(G, DFA, L)
    if mode == "apos":
        tv = apply_delta_correction_apos(tv, G, DFA, L, sysAbs, delta_sys=corr_delta, T=0)
    curve.append(_region_mean(tv, r0, r1, c0, c1))
    print(f"T=0: mean({mode})={curve[-1]:.6g}")

    for t in range(1, TMAX + 1):
        print(f"=== Iteration {t}/{TMAX} (mode={mode}) ===")
        if mode == "apos":
            G.set_iter(t)
        G.maxpolicy(rho)
        G.update_tree()
        G.prune(PRUNE_TOL, "leafs")
        G.grow()

        tv, _ = compute_tv_from_tree(G, DFA, L)
        if mode == "apos":
            tv = apply_delta_correction_apos(tv, G, DFA, L, sysAbs, delta_sys=corr_delta, T=t + 1)
        curve.append(_region_mean(tv, r0, r1, c0, c1))
        print(f"T={t}: mean({mode})={curve[-1]:.6g}")

    meta = dict(
        mode      = mode,
        rows      = list(ROWS), cols = list(COLS),
        sigma     = list(SIGMA),
        nx        = NX, nu = NU, eps = EPS,
        prune_tol = PRUNE_TOL,
        delta_rt       = list(DELTA_RT)       if mode == "rt"   else None,
        delta_apos_pol = list(DELTA_APOS_POL) if mode == "apos" else None,
        delta_corr     = list(DELTA_CORR)     if mode == "apos" else None,
    )

    return _Result(
        mode  = mode,
        Ts    = np.asarray(Ts,    dtype=int),
        curve = np.asarray(curve, dtype=float),
        meta  = meta,
    )


# ===========================================================================
# Cache
# ===========================================================================
def _cache_path(mode: str) -> Path:
    if mode == "rt":   return DATA_DIR / "fig3_rt.npz"
    if mode == "apos": return DATA_DIR / "fig3_apos.npz"
    raise ValueError(f"bad mode: {mode!r}")


def _load_or_compute(mode: str) -> np.ndarray:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(mode)

    if path.exists():
        print(f"[fig3] loading cached {path.name}")
        return np.asarray(np.load(path)["curve"], dtype=float)

    print(f"[fig3] no cached {path.name}; running sweep ...")
    result = _run_sweep(mode)
    np.savez(path,
             Ts    = result.Ts,
             curve = result.curve,
             meta  = np.array(repr(result.meta), dtype=str))
    print(f"[fig3] saved {path.name}")
    return result.curve


# ===========================================================================
# Plotting
# ===========================================================================
def _apply_paper_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "text.usetex":        True,
        "font.family":        "serif",
        "font.size":          16,
        "axes.titlesize":     18,
        "axes.labelsize":     18,
        "xtick.labelsize":    14,
        "ytick.labelsize":    14,
        "legend.fontsize":    14,
        "font.weight":        "bold",
        "axes.titleweight":   "bold",
        "axes.labelweight":   "bold",
    })


def main() -> int:
    _apply_paper_style()
    import matplotlib.pyplot as plt

    curve_rt   = _load_or_compute("rt")
    curve_apos = _load_or_compute("apos")

    T_rt   = np.arange(len(curve_rt))
    T_apos = np.arange(len(curve_apos))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(T_rt,   curve_rt,
            label=r"$\textbf{Optimal Robust-tree Value functions}$",
            marker="^")
    ax.plot(T_apos, curve_apos,
            label=r"$\textbf{Optimal A-posteriori corrected Value functions}$",
            marker="s")

    ax.legend(
        prop           = {"size": 18, "weight": "bold"},
        loc            = "lower right",
        bbox_to_anchor = (0.98, 0.02),
        bbox_transform = ax.transAxes,
        ncol           = 1,
        frameon        = True,  framealpha = 0.9,
        borderpad      = 0.3,   labelspacing = 0.25,
        handlelength   = 2.0,   handletextpad = 0.5,
        markerscale    = 0.9,
    )

    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter(r"\textbf{{{x:g}}}"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(r"\textbf{{{x:g}}}"))
    ax.tick_params(axis="both", labelsize=20, length=8, width=2, pad=8)
    ax.minorticks_off()
    ax.margins(x=0.08)
    ax.set_xlabel(r"\textbf{Specification Horizon}")
    for sp in ax.spines.values():
        sp.set_linewidth(2)
    ax.set_title("")
    ax.grid(False)
    fig.tight_layout()

    fig.savefig(HERE / "fig3.png", dpi=300, bbox_inches="tight")
    fig.savefig(HERE / "fig3.eps", format="eps", bbox_inches="tight")
    print("[fig3] wrote fig3.png, fig3.eps")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())