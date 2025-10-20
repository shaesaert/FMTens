# isolate_Gd_drives_G0_copyPxx_with_APoS.py
# Keep your Gd/G0 pipeline unchanged.
# Add an independent nominal tree G0a that we optimize on its own and evaluate a-posteriori corrections vs T.

import argparse
import os
import copy
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

# --- project imports ---
from src.models.linmodel import LinModel
from src.models.mdpmodel import MDPModel
from src.specifications.translate import translate
from src.specifications.utils.dfa_tool import dfa_manipulation
from src.abstraction.utils.labeling import dim_label
from src.dynprog.dfa_tree_r1 import DFATree
from src.dynprog.utils.treebasedV import compute_tv_from_tree
from src.dynprog.utils.v_apos import apply_delta_correction_apos
from src.config.delta_ui import parse_delta_list as parse_list
from src.vis.plot_tv import plotV_rank1

# ------------ tiny I/O ------------
def ask(prompt: str, default: str = "") -> str:
    try:
        ans = input(f"{prompt} ").strip()
    except EOFError:
        ans = ""
    return ans if ans else default

# ------------ build system + spec ------------
def build_system_and_spec(sigma_vals):
    # two 1D subsystems
    A = {0: np.array([[0.9]]), 1: np.array([[0.9]])}
    B = {0: np.array([[0.5]]), 1: np.array([[0.5]])}
    C = {0: np.array([[1.0]]), 1: np.array([[1.0]])}
    Dm = {0: np.array([[0.0]]), 1: np.array([[0.0]])}
    Bw = {0: np.array([[0.5]]), 1: np.array([[0.5]])}
    mu = {0: np.array([0.0]), 1: np.array([0.0])}
    sigma = {
        0: float(sigma_vals[0]) * np.eye(1),
        1: float(sigma_vals[1]) * np.eye(1),
    }
    sysLTI = {
        0: LinModel(A[0], B[0], C[0], Dm[0], Bw[0], mu=mu[0], sigma=sigma[0]),
        1: LinModel(A[1], B[1], C[1], Dm[1], Bw[1], mu=mu[1], sigma=sigma[1]),
    }

    # DFA: ( (!p2 | !p3) U p1 )
    DFA = translate('( (!p2 | !p3 ) U p1)')
    DFA, letters = dfa_manipulation(
        DFA,
        index_base=0, ensure_transitions=True, remove_qf_self_loop=True,
        replacements={"!p1 & !p2": "p3&!p1&!p2", "!p1 & !p3": "!p3&!p1"},
        desired_order=['p1', 'p3&!p1&!p2', '!p3&!p1'],
        verbose=True
    )

    # AP geometry
    s  = np.array([[1], [-1]])
    bx = np.array([5, 20])    # state box
    bu = np.array([5, 5])     # input box
    bp1 = np.array([5, 0])    # p1: [0, 5]
    bp2 = np.array([0, 5])    # p2: [-5, 0]
    bp3 = np.array([-15, 20]) # p3: [-20, -15]
    P1 = pc.Polytope(s, bp1); P2 = pc.Polytope(s, bp2); P3 = pc.Polytope(s, bp3)

    for i in [0, 1]:
        sysLTI[i].X = pc.Polytope(s, bx)
        sysLTI[i].U = pc.Polytope(s, bu)
    sysLTI[0].regions = [P1, P2]; sysLTI[1].regions = [P3]
    sysLTI[0].AP = ['p1', 'p2'];  sysLTI[1].AP = ['p3']

    return sysLTI, DFA, letters

def build_abs_and_label(sysLTI, DFA, letters, nx_val: int, nu_val: int):
    sysAbs = {
        0: MDPModel.from_system(sysLTI[0], nx=nx_val, nu=nu_val,
                                placement='centers', u_placement='endpoints',
                                tol=1e-19, contract_sum=None, compute_P='1d'),
        1: MDPModel.from_system(sysLTI[1], nx=nx_val, nu=nu_val,
                                placement='centers', u_placement='endpoints',
                                tol=1e-19, contract_sum=None, compute_P='1d'),
    }
    # labeling visual: avoid LaTeX
    with mpl.rc_context({'text.usetex': False}):
        L = dim_label(sysAbs, sysLTI, letters, visualize=True)

    dims = sorted(sysAbs.keys())
    nx = [sysAbs[d].N for d in dims]
    nu = [sysAbs[d].M for d in dims]
    nQ = len(DFA.S)

    pol = [[np.full((nx[d], nu[d]), 1.0/nu[d], dtype=float) for d in range(len(dims))]
           for _ in range(nQ)]
    rho = [np.full(nx[d], 1.0/nx[d], dtype=float) for d in range(len(dims))]
    return sysAbs, L, pol, rho

# ------------ Pxx copier ------------
def copy_Pxx(G_src: DFATree, G_dst: DFATree):
    """
    Deep-copy transition operators Pxx from G_src to G_dst, for all DFA states and dims.
    This evaluates G_dst under the same decision kernels without sharing references.
    """
    if G_src.DFA.S != G_dst.DFA.S or G_src.dim != G_dst.dim:
        raise ValueError("G_src and G_dst must have same DFA and dimension count.")
    for q in G_src.DFA.S:
        for d in range(G_src.dim):
            G_dst.Pxx[q][d] = copy.deepcopy(G_src.Pxx[q][d])

# ------------ main ------------
def main():
    ap = argparse.ArgumentParser(description="Optimize Gd; copy its Pxx to G0 (nominal). Also run an independent G0a for a-posteriori correction curve.")
    ap.add_argument("--nx", type=int, default=None, help="grid states per dim (default ask → 1000)")
    ap.add_argument("--nu", type=int, default=None, help="grid actions per dim (default ask → 10)")
    ap.add_argument("--sigma", type=str, default="", help="per-dim sigma scalars; single or CSV (default 1.0)")
    ap.add_argument("--delta", type=str, default="", help="per-dim Δ; single or CSV (default 0.001)")
    ap.add_argument("--T", type=int, default=None, help="horizon (iterations) (default ask → 20)")
    ap.add_argument("--prune-tol", type=float, default=1e-15, help="pruning tolerance for leafs")
    ap.add_argument("--plot", action="store_true", help="plot tvd and tv0_shared at the end")
    ap.add_argument("--out-prefix", type=str, default="outputs/fig",
                    help="Prefix for output filenames (directory + base name, no extension).")
    ap.add_argument("--dev-reload", action="store_true")
    args, _ = ap.parse_known_args()

    if args.dev_reload:
        dev_reload()

    # ensure output directory exists
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    nx_val = args.nx if args.nx is not None else int(ask("Enter state grid size nx [default 1000]:", "1000"))
    nu_val = args.nu if args.nu is not None else int(ask("Enter action grid size nu [default 10]:", "10"))
    T      = args.T  if args.T  is not None else int(ask("Enter horizon T [default 20]:", "20"))
    if nx_val < 1 or nu_val < 1 or T < 1:
        raise ValueError("nx, nu, T must be positive integers.")

    # sigmas & deltas
    D_dims = 2
    if args.sigma:
        sigma_vals = parse_list(args.sigma, D_dims, default=1.0)
    else:
        sigma_vals = parse_list(ask(f"Enter Sigma (single or {D_dims}-CSV) [default 1.0]:", "1.0"),
                                D_dims, default=1.0)
    if args.delta:
        delta_sys = parse_list(args.delta, D_dims, default=0.001)
    else:
        delta_sys = parse_list(ask(f"Enter Δ (single or {D_dims}-CSV) [default 0.001]:", "0.001"),
                               D_dims, default=0.001)

    # build system/spec
    sysLTI, DFA, letters = build_system_and_spec(sigma_vals)
    sysAbs, L, pol, rho  = build_abs_and_label(sysLTI, DFA, letters, nx_val=nx_val, nu_val=nu_val)

    # build trees: (1) your Gd/G0 pipeline (unchanged)
    nx_list = [sysAbs[k].N for k in sorted(sysAbs.keys())]
    L_list  = [L[k]       for k in sorted(sysAbs.keys())]
    delta_zero_vecs = [np.zeros(sysAbs[k].N, dtype=float) for k in sorted(sysAbs.keys())]
    delta_vecs      = [np.full(sysAbs[k].N, delta_sys[i], dtype=float)
                       for i, k in enumerate(sorted(sysAbs.keys()))]

    G0 = DFATree(DFA, sysAbs, pol, nx_list, L_list, delta=delta_zero_vecs)  # nominal (shared Pxx from Gd)
    Gd = DFATree(DFA, sysAbs, pol, nx_list, L_list, delta=delta_vecs)       # robust
    G0.initiate(); Gd.initiate()

    # (2) independent nominal tree G0a for a-posteriori curve
    pol_a = [[p.copy() for p in row] for row in pol]  # shallow copy is fine for dense numpy
    G0a = DFATree(DFA, sysAbs, pol_a, nx_list, L_list, delta=delta_zero_vecs)
    G0a.initiate()

    print(f"\nConfig → nx={nx_val}, nu={nu_val}, sigma={sigma_vals}, delta={delta_sys}, T={T}, prune_tol={args.prune_tol}")

    # storage for curves
    Ts_list = list(range(1, T + 1))
    diff_robust_vs_normal = []  # || tvd_shared - tv0_shared ||_inf
    diff_apos_vs_normal   = []  # || tv0a - tv_apos ||_inf

    # --- iterations ---
    for it in Ts_list:
        print(f"\n=== Iteration {it}/{T} ===")

        # 1) Optimize & advance robust tree (Gd)
        Gd.maxpolicy(rho)                 # ← improves Gd.Pxx
        Gd.update_tree()
        Gd.prune(args.prune_tol, 'leafs')
        Gd.grow()

        # 2) Copy Gd's kernels to G0, then evaluate & advance G0
        copy_Pxx(Gd, G0)                  # ← evaluate G0 under Gd's policy (no shared refs)
        G0.update_tree()
        G0.prune(args.prune_tol, 'leafs')
        G0.grow()

        # 3) Independent nominal run G0a: improve on its own, then advance
        G0a.maxpolicy(rho)
        G0a.update_tree()
        G0a.prune(args.prune_tol, 'leafs')
        G0a.grow()

        # 4) Build tvs and gaps
        tvd_shared = compute_tv_from_tree(Gd, DFA, L)
        tv0_shared = compute_tv_from_tree(G0, DFA, L)
        gap_robust = float(np.max(np.abs(tvd_shared - tv0_shared)))
        diff_robust_vs_normal.append(gap_robust)

        # a-posteriori on G0a for the current horizon it
        tv0a = compute_tv_from_tree(G0a, DFA, L)
        tv_apos = apply_delta_correction_apos(
            tv0a, G0a, DFA, L, sysAbs, delta_sys=delta_sys, T=it
        )
        gap_apos = float(np.max(np.abs(tv0a - tv_apos)))
        diff_apos_vs_normal.append(gap_apos)

        print(f"  || tvd_shared - tv0_shared ||_inf : {gap_robust:.6g}")
        print(f"  || tv0a - tv_apos ||_inf (APoS) : {gap_apos:.6g}")

    # --- Plot with your styling ---
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(Ts_list, diff_robust_vs_normal, label=r'\textbf{Robustness-aware tree}', marker='s')
    ax.plot(Ts_list, diff_apos_vs_normal,   label=r'\textbf{A-posteriori correction}', marker='o')
    ax.legend(fontsize=18, handlelength=3, handletextpad=0.6, borderaxespad=0.8)

    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))

    ax.tick_params(axis="both", labelsize=20, length=8, width=2, pad=8)
    ax.minorticks_off()
    ax.tick_params(axis="both", which="minor", length=5, width=1.6)
    ax.margins(x=0.08)
    ax.set_xlabel(r'\textbf{Specification Horizon}')
    ax.set_ylabel(
        r'$\boldsymbol{\lVert}\mathbf{V}_{\delta=0}-\mathbf{V}_{\delta_i=0.001}\boldsymbol{\rVert}_{\infty}$',
        fontsize=22
    )
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.tick_params(axis="both", which="major", width=2, length=8)
    for s in ax.spines.values():
        s.set_linewidth(2)
    ax.set_title('')
    ax.grid(False)
    ax.legend(ncol=2)
    fig.tight_layout()

    fig.savefig(f"{args.out_prefix}_2d_robust_and_apos_vs_normal.eps", format="eps", bbox_inches="tight")
    fig.savefig(f"{args.out_prefix}_2d_robust_and_apos_vs_normal.png", dpi=300, bbox_inches="tight")
    plt.show()

    # optional: final tv visuals if requested
    if args.plot:
        tvd_final  = compute_tv_from_tree(Gd, DFA, L)
        tv0_finalS = compute_tv_from_tree(G0, DFA, L)
        plotV_rank1(sysAbs, tvd_final)
        plt.tight_layout(); plt.show()
        plotV_rank1(sysAbs, tv0_finalS)
        plt.tight_layout(); plt.show()

    # optional: save raw data
    np.savez(f"{args.out_prefix}_curves.npz",
             Ts=np.array(Ts_list),
             robust_gap=np.array(diff_robust_vs_normal),
             apos_gap=np.array(diff_apos_vs_normal))

    return 0

# optimal policy
# plots error correction
if __name__ == "__main__":
    raise SystemExit(main())
