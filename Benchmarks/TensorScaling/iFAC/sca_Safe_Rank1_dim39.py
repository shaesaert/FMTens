# compare_multi_spec_dims_3to6_sharedpolicy_and_apos_styled.py
# For N=3..6:
#   (A) Robust-vs-Nominal with SHARED policy: Gd.maxpolicy -> copy Pxx to G0.
#   (B) A-posteriori correction on an INDEPENDENT nominal tree G0a.
#
# Plots (styled like your APoS script):
#   1) || tv_Δ - tv_0 ||_∞  (shared policy)
#   2) || tv_before - tv_after ||_∞  (a-posteriori)

import argparse
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{bm}'
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "text.usetex": True,
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})
mpl.use("QtAgg")

import matplotlib.pyplot as plt
import polytope as pc
from importlib import reload

# --- optional dev reload ---
def dev_reload():
    import src.models.linmodel as LinModel_mod
    import src.models.mdpmodel as mdpmodel_mod
    import src.abstraction.utils.labeling as dim_label_mod
    import src.dynprog.dfa_tree_r1 as DFATree_mod
    reload(LinModel_mod); reload(mdpmodel_mod); reload(dim_label_mod); reload(DFATree_mod)

# ---- project imports ----
from src.models.linmodel import LinModel
from src.models.mdpmodel import MDPModel
from src.specifications.translate import translate
from src.specifications.utils.dfa_tool import dfa_manipulation
from src.abstraction.utils.labeling import dim_label
from src.dynprog.dfa_tree_r1 import DFATree
from src.dynprog.utils.treebasedV import compute_tv_from_tree
from src.dynprog.utils.v_apos import apply_delta_correction_apos
from src.abstraction.utils.pc_utils import Pc
from src.config.delta_ui import parse_delta_list as parse_list

# ---------- small helpers ----------
def ask(prompt: str, default: str = "") -> str:
    try:
        ans = input(f"{prompt} ").strip()
    except EOFError:
        ans = ""
    return ans if ans else default

def ask_yes_no(prompt: str, default: bool) -> bool:
    tag = "[Y/n]" if default else "[y/N]"
    ans = ask(f"{prompt} {tag}:", "Y" if default else "N").lower()
    return ans in ("y", "yes") if ans else default

def parse_Ts(s: str, default: str = "1:100:1"):
    """
    Parse Ts string as either:
      - 'start:end' (inclusive), or 'start:end:step'
      - CSV list like '1,5,10'
    """
    s = (s or "").strip()
    if not s:
        s = default
    if ":" in s:
        parts = [int(x.strip()) for x in s.split(":")]
        if len(parts) == 2:
            start, end = parts; step = 1
        elif len(parts) == 3:
            start, end, step = parts
        else:
            raise ValueError("Ts format must be 'start:end[:step]' or CSV.")
        if step == 0:
            raise ValueError("Ts step cannot be 0.")
        if end < start and step > 0:
            raise ValueError("Ts end must be >= start when step>0.")
        return list(range(start, end + 1, step))
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty Ts list.")
    return vals

def _compute_tv_safe(G, DFA, L, max_elements=50_000_000):
    try:
        return compute_tv_from_tree(G, DFA, L, max_elements=max_elements)
    except TypeError:
        return compute_tv_from_tree(G, DFA, L)

def copy_Pxx(G_src: DFATree, G_dst: DFATree):
    """Copy decision kernels Pxx from src to dst (no shared refs required here)."""
    if G_src.DFA.S != G_dst.DFA.S or G_src.dim != G_dst.dim:
        raise ValueError("G_src and G_dst must have same DFA and dimension count.")
    for q in G_src.DFA.S:
        for d in range(G_src.dim):
            G_dst.Pxx[q][d] = G_src.Pxx[q][d]

# ---------- build system/spec for arbitrary dim ----------
def build_system_and_spec(dim: int, sigma_vals):
    """
    Build 'dim' independent 1D subsystems and a time-staggered safety spec:
      safeset = p10 & p11 & ... & p1{dim-1}
      formula = (safeset) & X(safeset) & X^2(safeset) & ... & X^{dim-2}(safeset)
    """
    A  = {i: np.array([[1.0]]) for i in range(dim)}
    B  = {i: np.array([[1.0]]) for i in range(dim)}
    C  = {i: np.array([[1.0]]) for i in range(dim)}
    Dm = {i: np.array([[0.0]]) for i in range(dim)}
    Bw = {i: np.array([[1.0]]) for i in range(dim)}
    mu = {i: np.array([0.0])  for i in range(dim)}
    sigma = {i: float(sigma_vals[i]) * np.eye(1) for i in range(dim)}

    s_mat = np.array([[1], [-1]])
    bx = np.array([10, 10])
    bu = np.array([2, 2])

    bp1 = np.array([5, 5])
    P1 = pc.Polytope(s_mat, bp1)

    sysLTI = {}
    for i in range(dim):
        sys = LinModel(A[i], B[i], C[i], Dm[i], Bw[i], mu=mu[i], sigma=sigma[i])
        sys.X = pc.Polytope(s_mat, bx)
        sys.U = pc.Polytope(s_mat, bu)
        sys.AP = [f"p1{i}"]
        sys.regions = [P1]
        sysLTI[i] = sys

    safeset = " & ".join(f"p1{i}" for i in range(dim))
    terms = []
    for k in range(dim - 1):  # 0..dim-2
        Xk = "X" * k
        terms.append(f"{Xk}({safeset})" if Xk else f"({safeset})")
    formula = " & ".join(terms) if terms else f"({safeset})"

    DFA = translate(formula)
    DFA, letters = dfa_manipulation(
        DFA, index_base=0, ensure_transitions=True, remove_qf_self_loop=True, verbose=True
    )
    return sysLTI, DFA, letters

def build_abs_label_policy(sysLTI, DFA, letters, nx: int, nu: int):
    dim = len(sysLTI)
    sysAbs = {
        i: MDPModel.from_system(sysLTI[i], nx=nx, nu=nu, placement='centers',
                                u_placement='endpoints', tol=1e-19,
                                contract_sum=None, compute_P='1d')
        for i in range(dim)
    }
    # avoid LaTeX inside the labeling utility
    with mpl.rc_context({'text.usetex': False}):
        L = dim_label(sysAbs, sysLTI, letters, visualize=False)

    dims_sorted = sorted(sysAbs.keys())
    nxv  = [sysAbs[d].N for d in dims_sorted]
    nuv  = [sysAbs[d].M for d in dims_sorted]
    nQ   = len(DFA.S)
    pol = [[np.full((nxv[d], nuv[d]), 1.0/nuv[d], dtype=float) for d in range(len(dims_sorted))]
           for _ in range(nQ)]
    rho = [np.full(nxv[d], 1.0/nxv[d], dtype=float) for d in range(len(dims_sorted))]
    return sysAbs, L, pol, rho, dims_sorted

# ---------- one curve for a given dim ----------
def run_one_dim_curve(dim, Ts_list, nx, nu, sigma_scalar, delta_scalar,
                      use_maxpolicy=True, prune_tol=0.005, max_elements=50_000_000):
    """
    Returns:
      robust_vs_normal:  || tv_d - tv_0_shared ||_inf (G0 shares policy from Gd)
      apos_vs_normal:    || tv0a_before - tv0a_after ||_inf (independent nominal tree)
    """
    sigma_vals = [float(sigma_scalar)] * dim
    delta_sys  = [float(delta_scalar)] * dim

    sysLTI, DFA, letters = build_system_and_spec(dim, sigma_vals)
    sysAbs, L, pol, rho, dims_sorted = build_abs_label_policy(sysLTI, DFA, letters, nx=nx, nu=nu)

    nx_list = [sysAbs[k].N for k in dims_sorted]
    L_list  = [L[k]       for k in dims_sorted]

    delta_zero_vecs = [np.zeros(sysAbs[k].N, dtype=float) for k in dims_sorted]
    delta_vecs      = [np.full(sysAbs[k].N, delta_sys[i], dtype=float) for i, k in enumerate(dims_sorted)]

    Gd         = DFATree(DFA, sysAbs, pol, nx_list, L_list, delta=delta_vecs)
    G0_shared  = DFATree(DFA, sysAbs, pol, nx_list, L_list, delta=delta_zero_vecs)
    # independent nominal for APoS
    pol_a = [[p.copy() for p in row] for row in pol]
    G0a        = DFATree(DFA, sysAbs, pol_a, nx_list, L_list, delta=delta_zero_vecs)

    Gd.initiate(); G0_shared.initiate(); G0a.initiate()

    robust_vs_normal = []
    apos_vs_normal   = []

    for T_run in Ts_list:
        # Shared policy (opt on Gd, copy to G0_shared)
        if use_maxpolicy:
            Gd.maxpolicy(rho)
        Gd.update_tree()
        if prune_tol and prune_tol > 0:
            Gd.prune(prune_tol, 'leafs')
        Gd.grow()

        copy_Pxx(Gd, G0_shared)
        G0_shared.update_tree()
        if prune_tol and prune_tol > 0:
            G0_shared.prune(prune_tol, 'leafs')
        G0_shared.grow()

        tvd        = _compute_tv_safe(Gd, DFA, L, max_elements=max_elements)
        tv0_shared = _compute_tv_safe(G0_shared, DFA, L, max_elements=max_elements)
        robust_vs_normal.append(float(np.max(np.abs(tvd - tv0_shared))))

        # Independent nominal for APoS
        if use_maxpolicy:
            G0a.maxpolicy(rho)
        G0a.update_tree()
        if prune_tol and prune_tol > 0:
            G0a.prune(prune_tol, 'leafs')
        G0a.grow()

        tv0a_before = _compute_tv_safe(G0a, DFA, L, max_elements=max_elements)
        tv0a_after  = apply_delta_correction_apos(
            tv0a_before, G0a, DFA, L, sysAbs, delta_sys=delta_sys, T=T_run
        )
        apos_vs_normal.append(float(np.max(np.abs(tv0a_before - tv0a_after))))

    return np.array(robust_vs_normal), np.array(apos_vs_normal)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Dims N=3..6: robust-vs-normal (shared policy) and a-posteriori (independent nominal) with styled plots.")
    ap.add_argument("--Ts", type=str, default="", help="T sweep: 'start:end[:step]' or CSV (default prompt).")
    ap.add_argument("--nx", type=int, default=None, help="States per dimension (default prompt).")
    ap.add_argument("--nu", type=int, default=5, help="Actions per dimension (default 5).")
    ap.add_argument("--delta", type=str, default="", help="Δ scalar (default prompt).")
    ap.add_argument("--sigma", type=str, default="1.0", help="Σ scalar (same for all dims; default 1.0).")
    ap.add_argument("--prune-tol", type=float, default=0.005, help="Pruning tolerance for leafs (default 0.005).")
    ap.add_argument("--apply-maxpolicy", action="store_true", help="Skip prompt and apply maxpolicy.")
    ap.add_argument("--dev-reload", action="store_true")
    ap.add_argument("--out-prefix", type=str, default="multi_spec_shared_apos",
                    help="Filename prefix for saved figures (EPS/PNG).")
    args, _ = ap.parse_known_args()

    if args.dev_reload:
        dev_reload()

    Ts_str = args.Ts or ask("Enter T sweep (e.g. '1:100:1' or '1,5,10') [default 1:100:1]:", "1:100:1")
    Ts_list = parse_Ts(Ts_str, default="1:100:1")

    if args.nx is None:
        nx_str = ask("Enter nx (states per dimension) [default 10]:", "10")
        nx = int(nx_str)
    else:
        nx = int(args.nx)

    if args.delta:
        delta_scalar = float(parse_list(args.delta, 1, default=0.001)[0])
    else:
        delta_str = ask("Enter delta (single scalar for all dims) [default 0.001]:", "0.001")
        delta_scalar = float(parse_list(delta_str, 1, default=0.001)[0])

    use_maxpolicy = args.apply_maxpolicy or ask_yes_no("Apply policy improvement (maxpolicy) each iteration?", True)

    nu = int(args.nu)
    sigma_scalar = float(parse_list(args.sigma, 1, default=1.0)[0])

    print(f"\nConfig → dims=3..6, Ts={Ts_list}, nx={nx}, nu={nu}, "
          f"sigma={sigma_scalar}, delta={delta_scalar}, "
          f"apply_maxpolicy={use_maxpolicy}, prune_tol={args.prune_tol}")

    dims_to_run = list(range(3, 7))
    robust_curves = {}
    apos_curves   = {}

    for d in dims_to_run:
        print(f"\n=== Running dim={d} ===")
        r, a = run_one_dim_curve(d, Ts_list, nx, nu,
                                 sigma_scalar, delta_scalar,
                                 use_maxpolicy=use_maxpolicy,
                                 prune_tol=args.prune_tol,
                                 max_elements=50_000_000)
        robust_curves[d] = r
        apos_curves[d]   = a

    # --- Styled Plot: Robustness-aware (shared policy) vs Normal ---
    fig, ax = plt.subplots(figsize=(12, 3))
    for d in dims_to_run:
        ax.plot(Ts_list, robust_curves[d], label=fr'\textbf{{N={d}}}', marker='s', linewidth=2)
    import matplotlib.ticker as mticker
    ax.set_xlim(min(Ts_list), max(Ts_list))
    # ax.xaxis.set_major_locator(mticker.MultipleLocator(1))  # <- integer ticks every 1
    ax.xaxis.set_minor_locator(mticker.NullLocator())  # no minor ticks
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
    ax.tick_params(axis="both", labelsize=20, length=8, width=2, pad=8)
    ax.minorticks_off()
    ax.tick_params(axis="both", which="minor", length=5, width=1.6)
    ax.margins(x=0.08)
    ax.set_xlabel(r'\textbf{Specification Horizon}')
    ax.set_ylabel(
        r'$\boldsymbol{\lVert}\mathbf{V}_{\delta=0}-\mathbf{V}_{\delta_i=0.001}^{robusttree}\boldsymbol{\rVert}_{\infty}$',
        fontsize=22
    )
    #ax.set_ylabel(r'$\boldsymbol{\lVert}\mathbf{V}_{\Delta}-\mathbf{V}_{0}\boldsymbol{\rVert}_{\infty}$', fontsize=22)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.tick_params(axis="both", which="major", width=2, length=8)
    for s in ax.spines.values():
        s.set_linewidth(2)
    ax.set_title('')
    ax.grid(False)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_robust_vs_normal_shared.eps", format="eps", bbox_inches="tight")
    fig.savefig(f"{args.out_prefix}_robust_vs_normal_shared.png", dpi=300, bbox_inches="tight")

    # --- Styled Plot: A-posteriori vs Normal (independent nominal) ---
    fig, ax = plt.subplots(figsize=(12, 3))
    for d in dims_to_run:
        ax.plot(Ts_list, apos_curves[d], label=fr'\textbf{{N={d}}}', marker='o', linewidth=2)
    import matplotlib.ticker as mticker
    ax.set_xlim(min(Ts_list), max(Ts_list))
     # <- integer ticks every 1
    ax.xaxis.set_minor_locator(mticker.NullLocator())  # no minor ticks
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
    ax.tick_params(axis="both", labelsize=20, length=8, width=2, pad=8)
    ax.minorticks_off()
    ax.tick_params(axis="both", which="minor", length=5, width=1.6)
    ax.margins(x=0.08)
    ax.set_xlabel(r'\textbf{Specification Horizon}')
    ax.set_ylabel(
        r'$\boldsymbol{\lVert}\mathbf{V}_{\delta=0}-\mathbf{V}_{\delta_i=0.001}^{apos}\boldsymbol{\rVert}_{\infty}$',
        fontsize=22
    )
    #ax.set_ylabel(r'$\boldsymbol{\lVert}\mathbf{V}_{\Delta}-\mathbf{V}_{0}\boldsymbol{\rVert}_{\infty}$', fontsize=22)
    #ax.set_ylabel(r'$\boldsymbol{\lVert}\mathbf{V}_{\text{before}}-\mathbf{V}_{\text{after}}\boldsymbol{\rVert}_{\infty}$', fontsize=22)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.tick_params(axis="both", which="major", width=2, length=8)
    for s in ax.spines.values():
        s.set_linewidth(2)
    ax.set_title('')
    ax.grid(False)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_apos_vs_normal.eps", format="eps", bbox_inches="tight")
    fig.savefig(f"{args.out_prefix}_apos_vs_normal.png", dpi=300, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    raise SystemExit(main())
