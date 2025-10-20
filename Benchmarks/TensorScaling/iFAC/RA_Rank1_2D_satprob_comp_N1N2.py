# isolate_Gd_drives_G0_copyPxx.py
# Optimize only Gd; copy Gd.Pxx to G0.Pxx each iteration; update G0 accordingly (no clobber back to Gd).
# User picks TWO iterations (e.g., 56 and 57) to snapshot tv for Gd and G0, and plot tv0 - tvd for each.
# Plots have no titles or colorbars; axes are labeled x_1(0), x_2(0) in bold with bold tick labels.

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
    "font.size": 14,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
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
from src.config.delta_ui import parse_delta_list as parse_list
from src.vis.plot_tv import plotV_rank1

# ------------ small helpers ------------
def remove_titles(fig=None):
    """Remove suptitle and all axes titles from the current (or given) figure."""
    import matplotlib.pyplot as plt
    fig = fig or plt.gcf()
    try:
        fig._suptitle = None
    except Exception:
        pass
    for ax in fig.axes:
        try:
            ax.set_title("")
        except Exception:
            pass

def style_axes_bold_labels_ticks(fig=None):
    """Bold axis labels x_1(0), x_2(0) and tick labels for all axes in the figure."""
    import matplotlib.pyplot as plt
    fig = fig or plt.gcf()
    for ax in fig.axes:
        ax.set_xlabel(r'$\boldsymbol{x_1(0)}$')
        ax.set_ylabel(r'$\boldsymbol{x_2(0)}$')
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
        ax.tick_params(axis="both", which="both", width=1.6, length=6)

def ask(prompt: str, default: str = "") -> str:
    try:
        ans = input(f"{prompt} ").strip()
    except EOFError:
        ans = ""
    return ans if ans else default

def parse_two_iters(s: str, default_a=56, default_b=57):
    """
    Parse 'a,b' into two positive ints. If empty, returns defaults.
    """
    s = (s or "").strip()
    if not s:
        return default_a, default_b
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("Please provide exactly two integers separated by a comma, e.g., '56,57'.")
    a, b = int(parts[0]), int(parts[1])
    if a < 1 or b < 1:
        raise ValueError("Iteration indices must be >= 1.")
    return a, b

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
    # the labeling visual uses simple text; avoid LaTeX there
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
    """Deep-copy transition operators Pxx from G_src to G_dst for all DFA states and dims."""
    if G_src.DFA.S != G_dst.DFA.S or G_src.dim != G_dst.dim:
        raise ValueError("G_src and G_dst must have same DFA and dimension count.")
    for q in G_src.DFA.S:
        for d in range(G_src.dim):
            G_dst.Pxx[q][d] = copy.deepcopy(G_src.Pxx[q][d])

# ------------ main ------------
def main():
    ap = argparse.ArgumentParser(description="Optimize only Gd; copy Gd.Pxx to G0 each iteration; update G0. Snapshot two user-chosen iterations.")
    ap.add_argument("--nx", type=int, default=None, help="grid states per dim (default ask → 1000)")
    ap.add_argument("--nu", type=int, default=None, help="grid actions per dim (default ask → 10)")
    ap.add_argument("--sigma", type=str, default="", help="per-dim sigma scalars; single or CSV (default 1.0)")
    ap.add_argument("--delta", type=str, default="", help="per-dim Δ; single or CSV (default 0.001)")
    ap.add_argument("--T", type=int, default=None, help="horizon (iterations) (default ask → 20)")
    ap.add_argument("--snap", type=str, default="", help="two iterations as 'i,j' to snapshot (default prompt → 56,57)")
    ap.add_argument("--prune-tol", type=float, default=1e-15, help="pruning tolerance for leafs")
    ap.add_argument("--plot", action="store_true", help="also plot final tvd via plotV_rank1 (stripped)")
    ap.add_argument("--dev-reload", action="store_true")
    args, _ = ap.parse_known_args()

    if args.dev_reload:
        dev_reload()

    nx_val = args.nx if args.nx is not None else int(ask("Enter state grid size nx [default 1000]:", "1000"))
    nu_val = args.nu if args.nu is not None else int(ask("Enter action grid size nu [default 10]:", "10"))
    T      = args.T  if args.T  is not None else int(ask("Enter horizon T [default 20]:", "20"))
    if nx_val < 1 or nu_val < 1 or T < 1:
        raise ValueError("nx, nu, T must be positive integers.")

    # choose the two snapshot iterations
    if args.snap:
        itA, itB = parse_two_iters(args.snap)
    else:
        itA, itB = parse_two_iters(ask("Enter two iterations to snapshot (e.g., '56,57') [default 56,57]:", "56,57"))
    if itA > T or itB > T:
        raise ValueError(f"Snapshot iterations must be ≤ T. Got ({itA},{itB}) with T={T}.")

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

    # build trees
    nx_list = [sysAbs[k].N for k in sorted(sysAbs.keys())]
    L_list  = [L[k]       for k in sorted(sysAbs.keys())]
    delta_zero_vecs = [np.zeros(sysAbs[k].N, dtype=float) for k in sorted(sysAbs.keys())]
    delta_vecs      = [np.full(sysAbs[k].N, delta_sys[i], dtype=float)
                       for i, k in enumerate(sorted(sysAbs.keys()))]

    G0 = DFATree(DFA, sysAbs, pol, nx_list, L_list, delta=delta_zero_vecs)  # nominal
    Gd = DFATree(DFA, sysAbs, pol, nx_list, L_list, delta=delta_vecs)       # robust
    G0.initiate(); Gd.initiate()

    print(f"\nConfig → nx={nx_val}, nu={nu_val}, sigma={sigma_vals}, delta={delta_sys}, T={T}, prune_tol={args.prune_tol}")
    print(f"Snapshots at iterations: {itA} and {itB}")

    # holders for the two snapshot iterations
    tvd_A = tvd_B = tv0_A = tv0_B = None

    for it in range(1, T + 1):
        print(f"\n=== Iteration {it}/{T} ===")

        # Optimize & advance robust tree (Gd)
        Gd.maxpolicy(rho)
        Gd.update_tree()
        Gd.prune(args.prune_tol, 'leafs')
        Gd.grow()

        # Copy Gd's kernels to G0, then evaluate & advance G0
        copy_Pxx(Gd, G0)
        G0.update_tree()
        G0.prune(args.prune_tol, 'leafs')
        G0.grow()

        # Capture at user-chosen iterations
        if it == itA:
            tvd_A = compute_tv_from_tree(Gd, DFA, L)
            tv0_A = compute_tv_from_tree(G0, DFA, L)
        if it == itB:
            tvd_B = compute_tv_from_tree(Gd, DFA, L)
            tv0_B = compute_tv_from_tree(G0, DFA, L)

    # final values (optional)
    tvd_final  = compute_tv_from_tree(Gd, DFA, L)
    tv0_shared = compute_tv_from_tree(G0, DFA, L)

    print("\nDone. Shapes:")
    print("  tvd_final  :", [v.shape for v in tvd_final])
    print("  tv0_shared :", [v.shape for v in tv0_shared])

    # ---- plotting snapshots (no titles, no colorbars; bold labels & ticks) ----
    def _plot_tv(tv):
        plotV_rank1(sysAbs, tv)
        remove_titles()
        # best-effort removal of colorbars if the plotting util added any
        for ax in list(plt.gcf().axes):
            if hasattr(ax, 'collections') and any(getattr(c, 'colorbar', None) for c in ax.collections):
                # try to remove attached colorbar axes, if present
                pass  # plotV_rank1 typically draws imshow without explicit colorbar; keep as-is
        style_axes_bold_labels_ticks()
        plt.show()

    # (A) Gd and G0 at iteration itA
    if tvd_A is not None:
        _plot_tv(tvd_A)
    if tv0_A is not None:
        _plot_tv(tv0_A)

    # (B) Gd and G0 at iteration itB
    if tvd_B is not None:
        _plot_tv(tvd_B)
    if tv0_B is not None:
        _plot_tv(tv0_B)

    # Difference maps (optional; uncomment if you want visual deltas)
    if (tvd_A is not None) and (tv0_A is not None):
        _plot_tv(tv0_A - tvd_A)
    if (tvd_B is not None) and (tv0_B is not None):
        _plot_tv(tv0_B - tvd_B)

    if args.plot:
        plotV_rank1(sysAbs, tvd_final); remove_titles(); style_axes_bold_labels_ticks(); plt.show()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
