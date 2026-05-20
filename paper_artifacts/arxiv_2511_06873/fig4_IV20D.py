# paper_artifacts/arxiv_2511_06873/fig4_IV20D.py
"""
Reproduces Fig. 4 of arxiv:2511.06873.

Sweeps the invariance case study over both specification horizon and
dimension, then plots RT vs APoS value functions vs specification horizon,
for each DIM in DIMS.

Per-dim data is cached under `data/` as `fig4_D{DIM}.npz`. First run computes
and saves; subsequent runs load from cache. Delete `data/` to regenerate.

Run from the repository root:
    python -m paper_artifacts.arxiv_2511_06873.fig4_IV20D
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from itertools   import product
from pathlib     import Path
from types       import SimpleNamespace
from typing      import List, Optional

import numpy as np
import polytope as pc
import matplotlib.ticker as mticker

from src.models.linmodel               import LinModel
from src.specifications.translate      import translate
from src.specifications.utils.dfa_tool import dfa_manipulation
from src.pipeline                      import prepare_pipeline
from src.dynprog.dfa_tree_r1           import DFATree


# ===========================================================================
# Paper parameters — published in arxiv:2511.06873
# ===========================================================================
DIMS           = (4, 20)
T_SPECS        = tuple(range(1, 8))
ANCHOR_CENTERS = (40, 45, 50, 55, 59)

BX      = (10.0, 10.0)     # X = [-10, 10]
BU      = ( 2.0,  2.0)     # U = [-2, 2]
BP_SAFE = ( 5.0,  5.0)     # safe region = [-5, 5]

A_DYN   = 0.9
B_DYN   = 0.5
BW_DYN  = 0.5

EPS_VAL = 0.1
DELTA_I = 0.0717

ABS_CFG = SimpleNamespace(
    nx           = 100,
    nu           = 10,
    placement    = "centers",
    u_placement  = "endpoints",
    tol          = 1e-19,
    contract_sum = None,
    compute_P    = "1d",
)
SPEC_MANIP = SimpleNamespace(
    index_base          = 0,
    ensure_transitions  = True,
    remove_qf_self_loop = True,
    verbose             = False,
    replacements        = None,
    desired_order       = None,
)

PREFER_BACKEND = "QtAgg"

HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


# ===========================================================================
# Case-specific helpers (inlined from IFAC26_IV_setup.py)
# ===========================================================================
def _make_sysLTI_and_safeset(DIM: int):
    """DIM identical 1D agents; each agent has one AP 'p1{i}' over [-5, 5]."""
    s  = np.array([[1.0], [-1.0]])
    bx = np.array(BX, dtype=float)
    bu = np.array(BU, dtype=float)
    P_safe = pc.Polytope(s, np.array(BP_SAFE, dtype=float))

    A     = np.array([[A_DYN]],  dtype=float)
    B     = np.array([[B_DYN]],  dtype=float)
    C     = np.array([[1.0]],    dtype=float)
    Dm    = np.array([[0.0]],    dtype=float)
    Bw    = np.array([[BW_DYN]], dtype=float)
    mu    = np.array([[0.0]],    dtype=float)
    sigma = np.array([[1.0]],    dtype=float)

    sysLTI = {}
    for i in range(DIM):
        sys = LinModel(A, B, C, Dm, Bw, mu=mu, sigma=sigma)
        sys.X       = pc.Polytope(s, bx)
        sys.U       = pc.Polytope(s, bu)
        sys.AP      = [f"p1{i}"]
        sys.regions = [P_safe]
        sysLTI[i]   = sys

    safeset = " & ".join(f"p1{i}" for i in range(DIM))
    return sysLTI, safeset, s


def _make_formula(safeset: str, t_spec: int) -> str:
    """Sparse-horizon finite formula: k in {0, t_spec-1, t_spec} -> X^k(safeset)."""
    ks = {0, int(t_spec)}
    if t_spec >= 2:
        ks.add(int(t_spec) - 1)
    return " & ".join(f"{'X' * k}({safeset})" for k in sorted(ks))


def _build_dfa_from_formula(formula: str):
    DFA = translate(formula)
    DFA, letters = dfa_manipulation(
        DFA,
        index_base          = SPEC_MANIP.index_base,
        ensure_transitions  = SPEC_MANIP.ensure_transitions,
        remove_qf_self_loop = SPEC_MANIP.remove_qf_self_loop,
        replacements        = SPEC_MANIP.replacements,
        desired_order       = SPEC_MANIP.desired_order,
        verbose             = SPEC_MANIP.verbose,
    )
    return DFA, letters


# ===========================================================================
# Sweep primitives (inlined from sweep_eval.py, with FHT rename)
# ===========================================================================
def _nodes_with_q(G: DFATree, q: int) -> List[int]:
    return list(G.Q.get(int(q), []))


def _derive_qbar_for_anchor(DFA, L_list, idx_vec) -> Optional[List[int]]:
    q0    = int(np.asarray(DFA.S0).ravel()[0])
    Trans = np.asarray(DFA.trans, dtype=int)
    qbar: List[int] = []
    for d in range(len(L_list)):
        col = np.asarray(L_list[d][:, idx_vec[d]], dtype=bool)
        if not np.any(col):
            return None
        l = int(np.where(col)[0][0])
        qbar.append(int(Trans[q0, l]))
    return qbar


def _robust_value_at_anchor(G: DFATree, qbar_vec, idx_vec) -> float:
    D = G.dim
    assert len(qbar_vec) == D and len(idx_vec) == D
    per_dim = [_nodes_with_q(G, int(qbar_vec[d])) for d in range(D)]
    if any(len(lst) == 0 for lst in per_dim):
        return 0.0
    total = 0.0
    for combo in product(*per_dim):
        term = 1.0
        for d, n in enumerate(combo):
            term *= float(G.V[d][n, idx_vec[d]])
            if term == 0.0:
                break
        total += term
    return float(total)


def _first_hitting_time_term(G: DFATree, T_build: int, qbar: int, idx_x: int) -> float:
    """Exact first-hitting-time term used in the paper's APoS correction."""
    if "Dl" not in G.tree.graph:
        G._recompute_levels()
    Dl = G.tree.graph["Dl"]

    interested = set(G.Q.get(int(qbar), []))
    if not interested or T_build <= 0:
        return 0.0

    D = len(G.V)

    def node_val(z: int) -> float:
        prodv = 1.0
        for d in range(D):
            v = float(G.V[d][z, idx_x])
            if v <= 0.0:
                return 0.0
            prodv *= v
        return prodv

    total = 0.0
    for h in range(0, int(T_build) + 1):
        for npred in range(0, h):
            nodes_at_npred = Dl[npred][1] if npred < len(Dl) else []
            for z in nodes_at_npred:
                if z in interested:
                    total += node_val(z)
    return float(total)


def _build_tree_no_prune(*, DFA, sysAbs, pol, rho, nx_list, L_list,
                          delta_VI_vecs, delta_pol_vecs, pol_mode, VI_mode,
                          T_build) -> DFATree:
    G = DFATree(
        DFA, sysAbs, pol, nx_list, L_list,
        delta_VI=delta_VI_vecs,
        delta_pol=delta_pol_vecs,
        pol_mode=pol_mode,
        VI_mode=VI_mode,
    ).initiate()

    for it in range(1, int(T_build) + 1):
        if hasattr(G, "set_iter"):
            try:
                G.set_iter(it)
            except Exception:
                pass
        G.maxpolicy(rho)
        G.update_tree()
        G.grow()
    return G


@dataclass
class _SweepResult:
    DIM:            int
    t_specs:        np.ndarray
    anchor_centers: np.ndarray
    V_rt_avg:       np.ndarray
    V_apos_avg:     np.ndarray
    FHT_avg:        np.ndarray
    T_builds:       np.ndarray


def _run_dim_sweep(DIM: int) -> _SweepResult:
    """
    Run the IV invariance sweep for one DIM.
    APoS correction expression V_apos_corr = max(0, V_apos_base - T_build*Delta + Delta*FHT)
    is baked in here, matching arxiv:2511.06873.
    """
    sysLTI, safeset, _s = _make_sysLTI_and_safeset(DIM)
    anchor_idx_list     = [[int(k)] * DIM for k in ANCHOR_CENTERS]
    Delta               = 1.0 - (1.0 - DELTA_I) ** DIM

    rt_av:   List[float] = []
    apos_av: List[float] = []
    fht_av:  List[float] = []
    tbuild:  List[int]   = []

    for t_spec in T_SPECS:
        formula      = _make_formula(safeset, int(t_spec))
        DFA, letters = _build_dfa_from_formula(formula)

        sysAbs, L, L_list, pol, rho, nx_list = prepare_pipeline(
            sysLTI  = sysLTI,
            DFA     = DFA,
            letters = letters,
            abs_cfg = ABS_CFG,
            eps_val = EPS_VAL,
        )

        for d in range(DIM):
            for idx_vec in anchor_idx_list:
                if idx_vec[d] >= nx_list[d]:
                    raise ValueError(
                        f"[D={DIM}] anchor idx {idx_vec[d]} out of range "
                        f"for dim {d} (nx={nx_list[d]})."
                    )

        T_build = int(t_spec)
        tbuild.append(T_build)

        delta_VI_rt    = [np.full(sysAbs[d].N, DELTA_I, dtype=float) for d in range(DIM)]
        delta_pol_rt   = [np.full(sysAbs[d].N, DELTA_I, dtype=float) for d in range(DIM)]
        delta_VI_apos  = [np.zeros(sysAbs[d].N,         dtype=float) for d in range(DIM)]
        delta_pol_apos = [np.full(sysAbs[d].N, DELTA_I, dtype=float) for d in range(DIM)]

        G_rt = _build_tree_no_prune(
            DFA=DFA, sysAbs=sysAbs, pol=pol, rho=rho, nx_list=nx_list, L_list=L_list,
            delta_VI_vecs=delta_VI_rt, delta_pol_vecs=delta_pol_rt,
            pol_mode="rt", VI_mode="rt", T_build=T_build,
        )
        G_apos = _build_tree_no_prune(
            DFA=DFA, sysAbs=sysAbs, pol=pol, rho=rho, nx_list=nx_list, L_list=L_list,
            delta_VI_vecs=delta_VI_apos, delta_pol_vecs=delta_pol_apos,
            pol_mode="apos", VI_mode="apos", T_build=T_build,
        )

        vals_rt:        List[float] = []
        vals_apos_corr: List[float] = []
        vals_fht:       List[float] = []

        for idx_vec in anchor_idx_list:
            qbar = _derive_qbar_for_anchor(DFA, L_list, idx_vec)
            if qbar is None:
                continue

            V_rt        = _robust_value_at_anchor(G_rt,   qbar, idx_vec)
            V_apos_base = _robust_value_at_anchor(G_apos, qbar, idx_vec)
            FHT         = _first_hitting_time_term(
                G_apos, T_build, qbar=int(qbar[0]), idx_x=int(idx_vec[0])
            )

            V_apos_corr = max(0.0, V_apos_base - T_build * Delta + Delta * FHT)
            vals_rt.append(V_rt)
            vals_apos_corr.append(V_apos_corr)
            vals_fht.append(FHT)

        rt_av.append(  float(np.mean(vals_rt))        if vals_rt        else np.nan)
        apos_av.append(float(np.mean(vals_apos_corr)) if vals_apos_corr else np.nan)
        fht_av.append( float(np.mean(vals_fht))       if vals_fht       else np.nan)

    return _SweepResult(
        DIM            = int(DIM),
        t_specs        = np.asarray(T_SPECS,        dtype=int),
        anchor_centers = np.asarray(ANCHOR_CENTERS, dtype=int),
        V_rt_avg       = np.asarray(rt_av,   dtype=float),
        V_apos_avg     = np.asarray(apos_av, dtype=float),
        FHT_avg        = np.asarray(fht_av,  dtype=float),
        T_builds       = np.asarray(tbuild,  dtype=int),
    )


# ===========================================================================
# Cache
# ===========================================================================
def _cache_path(DIM: int) -> Path:
    return DATA_DIR / f"fig4_D{DIM}.npz"


def _load_or_compute(DIM: int) -> _SweepResult:
    path = _cache_path(DIM)
    if path.exists():
        print(f"[fig4] loading cached {path.name}")
        d = np.load(path)
        return _SweepResult(
            DIM            = int(DIM),
            t_specs        = np.asarray(d["t_specs"],        dtype=int),
            anchor_centers = np.asarray(d["anchor_centers"], dtype=int),
            V_rt_avg       = np.asarray(d["V_rt_avg"],       dtype=float),
            V_apos_avg     = np.asarray(d["V_apos_avg"],     dtype=float),
            FHT_avg        = np.asarray(d["FHT_avg"],        dtype=float),
            T_builds       = np.asarray(d["T_builds"],       dtype=int),
        )

    print(f"[fig4] no cache for D={DIM}; running sweep ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    res = _run_dim_sweep(DIM)
    np.savez(path,
             t_specs        = res.t_specs,
             anchor_centers = res.anchor_centers,
             V_rt_avg       = res.V_rt_avg,
             V_apos_avg     = res.V_apos_avg,
             FHT_avg        = res.FHT_avg,
             T_builds       = res.T_builds)
    print(f"[fig4] saved {path.name}")
    return res


# ===========================================================================
# Plotting
# ===========================================================================
def _apply_paper_style(prefer_backend: str = PREFER_BACKEND) -> None:
    import matplotlib as mpl
    try:
        mpl.use(prefer_backend, force=True)
    except Exception:
        mpl.use("MacOSX" if platform.system() == "Darwin" else "TkAgg", force=True)

    mpl.rcParams["text.usetex"]         = True
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{bm}"
    mpl.rcParams.update({
        "font.family":      "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size":        16,
        "axes.titlesize":   18,
        "axes.labelsize":   18,
        "xtick.labelsize":  14,
        "ytick.labelsize":  14,
        "legend.fontsize":  14,
        "font.weight":      "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })


def _plot_results(results: List[_SweepResult]) -> None:
    import matplotlib.pyplot as plt

    markers = [("s", "o"), ("^", "v"), ("D", "x"), ("P", "*")]

    fig, ax = plt.subplots(figsize=(12, 3))
    for i, res in enumerate(results):
        m_rt, m_ap = markers[i % len(markers)]
        N = int(res.DIM)
        ax.plot(res.t_specs, res.V_rt_avg,
                marker=m_rt, linewidth=2.0,
                label=rf"$\textbf{{Optimal RT Value functions}} \ (N={N})$")
        ax.plot(res.t_specs, res.V_apos_avg,
                marker=m_ap, linewidth=2.0, linestyle="--",
                label=rf"$\textbf{{Optimal APoS Value functions}} \ (N={N})$")

    ax.legend(
        prop={"size": 19, "weight": "bold"},
        loc="upper right", bbox_to_anchor=(0.98, 0.98),
        bbox_transform=ax.transAxes,
        ncol=1, frameon=True, framealpha=0.9,
        borderpad=0.3, labelspacing=0.25,
        handlelength=2.0, handletextpad=0.5, markerscale=0.9,
    )
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter(r"\textbf{{{x:g}}}"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(r"\textbf{{{x:g}}}"))
    ax.set_yticks([0.0, 0.25, 0.5, 0.75])
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(axis="both", labelsize=18, length=8, width=2, pad=8)
    ax.minorticks_off()
    ax.margins(x=0.08)
    ax.set_xlabel(r"\textbf{Specification Horizon}")
    for sp in ax.spines.values():
        sp.set_linewidth(2)
    ax.grid(False)
    fig.tight_layout()

    fig.savefig(HERE / "fig4.png", dpi=300, bbox_inches="tight")
    fig.savefig(HERE / "fig4.eps", format="eps", bbox_inches="tight")
    print("[fig4] wrote fig4.png, fig4.eps")

    plt.show(block=True)


def main() -> int:
    _apply_paper_style()
    results = [_load_or_compute(DIM) for DIM in DIMS]
    _plot_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())