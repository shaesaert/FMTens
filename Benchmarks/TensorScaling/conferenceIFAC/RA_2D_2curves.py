# compare_rt_apos_two_curves.py
# Build two trees:
#   • G_rt   : VI_mode=rt,   pol_mode=rt,   Δ_VI=δ_rt,        Δ_pol=δ_rt
#   • G_apos : VI_mode=apos, pol_mode=apos, Δ_VI=0,           Δ_pol=δ_apos_pol
#
# For t = 0..Tmax:
#   - Compute tv for each tree
#   - A-posteriori correct tv_apos with δ_corr → tv_apos_corr
#   - Region means: mean_rt (corrected by construction), mean_apos_corr (a-posteriori corrected)
# Plot ONLY THESE TWO curves.

import argparse
import os
import numpy as np
import matplotlib as mpl

# --- Matplotlib config (must be before pyplot import) ---
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
try:
    mpl.use("QtAgg")
except Exception:
    mpl.use("Agg")

import matplotlib.pyplot as plt
import polytope as pc

# --- project imports ---
from src.models.linmodel import LinModel
from src.models.mdpmodel import MDPModel
from src.specifications.translate import translate
from src.specifications.utils.dfa_tool import dfa_manipulation
from src.abstraction.utils.labeling import dim_label_eps
from src.dynprog.dfa_tree_r1 import DFATree
from src.dynprog.utils.treebasedV import compute_tv_from_tree
from src.dynprog.utils.v_apos import apply_delta_correction_apos
from src.config.delta_ui import parse_delta_list as parse_list

# -------- I/O helpers --------
def ask(prompt: str, default: str = "") -> str:
    try:
        ans = input(f"{prompt} ").strip()
    except EOFError:
        ans = ""
    return ans if ans else default

def region_mean(tv: np.ndarray, r0: int, r1_incl: int, c0: int, c1_incl: int) -> float:
    """Mean over tv[r0:r1_incl, c0:c1_incl] (safe-clipped)."""
    if tv.ndim != 2:
        raise ValueError(f"Expected 2D tv; got shape {tv.shape}")
    nrows, ncols = tv.shape
    rr0 = max(0, min(nrows-1, r0))
    rr1 = max(0, min(nrows-1, r1_incl))
    cc0 = max(0, min(ncols-1, c0))
    cc1 = max(0, min(ncols-1, c1_incl))
    if rr1 < rr0 or cc1 < cc0:
        return 0.0
    sub = tv[rr0:rr1+1, cc0:c1_incl+1]
    return float(np.mean(sub))

def make_delta_vectors_from_scalars(scalars, sysAbs) -> list:
    """Per-dim scalars -> per-state vectors (length N_d)."""
    out = []
    for i, k in enumerate(sorted(sysAbs.keys())):
        N = sysAbs[k].N
        out.append(np.full(N, float(scalars[i]), dtype=float))
    return out

# -------- system + spec --------
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
        verbose=False
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
    eps_val = 0.1  # robustness margin

    L = dim_label_eps(
        sysAbs,
        sysLTI,
        letters,
        eps=eps_val,
        visualize=False,
        outdir=None,
        prefix="L_eps"
    )

    dims = sorted(sysAbs.keys())
    nx = [sysAbs[d].N for d in dims]
    nu = [sysAbs[d].M for d in dims]
    nQ = len(DFA.S)

    pol = [[np.full((nx[d], nu[d]), 1.0/nu[d], dtype=float) for d in range(len(dims))]
           for _ in range(nQ)]
    rho = [np.full(nx[d], 1.0/nx[d], dtype=float) for d in range(len(dims))]
    return sysAbs, L, pol, rho

# -------- main --------
def main():
    ap = argparse.ArgumentParser(
        description="RT/APoS — compare only corrected satProb (region means) vs horizon."
    )
    ap.add_argument("--nx", type=int, default=None, help="grid states per dim (default prompt → 1000)")
    ap.add_argument("--nu", type=int, default=None, help="grid actions per dim (default prompt → 10)")
    ap.add_argument("--Tmax", type=int, default=None, help="max horizon (default prompt → 60)")
    ap.add_argument("--sigma", type=str, default="", help="per-dim sigma scalars; CSV or single (default 1.0)")

    # deltas
    ap.add_argument("--delta-rt",       type=str, default="", help="per-dim δ used for BOTH Δ_VI and Δ_pol in G_rt")
    ap.add_argument("--delta-apos-pol", type=str, default="", help="per-dim δ used as Δ_pol in G_apos (Δ_VI stays 0)")
    ap.add_argument("--delta-corr",     type=str, default="", help="per-dim δ for a-posteriori correction of tv_apos")

    ap.add_argument("--prune-tol", type=float, default=1e-15, help="pruning tolerance for leafs")
    ap.add_argument("--rows", type=str, default="0,798", help="row range inclusive as 'r0,r1' (default '0,798')")
    ap.add_argument("--cols", type=str, default="599,699", help="col range inclusive as 'c0,c1' (default '599,699')")
    ap.add_argument("--out-prefix", type=str, default="outputs/rt_apos_two_curves",
                    help="output prefix for files")
    args, _ = ap.parse_known_args()

    nx_val = args.nx if args.nx is not None else int(ask("Enter nx [1000]:", "1000"))
    nu_val = args.nu if args.nu is not None else int(ask("Enter nu [10]:", "10"))
    Tmax   = args.Tmax if args.Tmax is not None else int(ask("Enter Tmax [60]:", "60"))
    if nx_val < 1 or nu_val < 1 or Tmax < 0:
        raise ValueError("nx, nu >=1 and Tmax >= 0 required.")

    r0, r1 = [int(x.strip()) for x in args.rows.split(",")]
    c0, c1 = [int(x.strip()) for x in args.cols.split(",")]

    D_dims = 2
    if args.sigma:
        sigma_vals = parse_list(args.sigma, D_dims, default=1.0)
    else:
        sigma_vals = parse_list(ask(f"Enter Sigma (single or {D_dims}-CSV) [1.0]:", "1.0"),
                                D_dims, default=1.0)

    # ---- deltas ----
    delta_rt_scalars       = parse_list(args.delta_rt       or ask(f"Enter δ_rt (single or {D_dims}-CSV) [0.0084]:", "0.0084"),
                                        D_dims, default=0.0084)
    delta_apos_pol_scalars = parse_list(args.delta_apos_pol or ask(f"Enter δ_apos_pol (single or {D_dims}-CSV) [0.0084]:", "0.0084"),
                                        D_dims, default=0.0084)
    delta_corr_scalars     = parse_list(args.delta_corr     or ask(f"Enter δ_corr (single or {D_dims}-CSV) [0.0084]:", "0.0084"),
                                        D_dims, default=0.0084)

    # Build system/spec and abstraction
    sysLTI, DFA, letters = build_system_and_spec(sigma_vals)
    sysAbs, L, pol, rho  = build_abs_and_label(sysLTI, DFA, letters, nx_val=nx_val, nu_val=nu_val)

    dims_sorted = sorted(sysAbs.keys())
    nx_list = [sysAbs[k].N for k in dims_sorted]
    L_list  = [L[k]       for k in dims_sorted]

    # Δ vectors
    delta_zero_vecs   = [np.zeros(sysAbs[k].N, dtype=float) for k in dims_sorted]
    delta_rt_vecs     = make_delta_vectors_from_scalars(delta_rt_scalars,       sysAbs)
    delta_apos_polvec = make_delta_vectors_from_scalars(delta_apos_pol_scalars, sysAbs)

    # --- Trees ---
    G_rt = DFATree(
        DFA, sysAbs, pol, nx_list, L_list,
        delta_VI=delta_rt_vecs,
        delta_pol=delta_rt_vecs,
        pol_mode='rt',
        VI_mode='rt',
    ).initiate()

    G_apos = DFATree(
        DFA, sysAbs, pol, nx_list, L_list,
        delta_VI=delta_zero_vecs,
        delta_pol=delta_apos_polvec,
        pol_mode='apos',
        VI_mode='apos',
    ).initiate()

    # Curves
    Ts = list(range(0, Tmax + 1))
    mean_rt_curve        = []
    mean_apos_corr_curve = []

    # ---- T = 0 ----
    tv_rt   = compute_tv_from_tree(G_rt,   DFA, L)
    tv_apos = compute_tv_from_tree(G_apos, DFA, L)
    tv_apos_corr = apply_delta_correction_apos(tv_apos, G_apos, DFA, L, sysAbs,
                                               delta_sys=delta_corr_scalars, T=0)

    mean_rt_curve.append(region_mean(tv_rt, r0, r1, c0, c1))
    mean_apos_corr_curve.append(region_mean(tv_apos_corr, r0, r1, c0, c1))

    print(f"T=0: mean(RT)={mean_rt_curve[-1]:.6g}, "
          f"mean(APoS-corr)={mean_apos_corr_curve[-1]:.6g}")

    # ---- Iterate T = 1..Tmax ----
    for t in range(1, Tmax + 1):
        print(f"\n=== Iteration {t}/{Tmax} ===")

        # RT improve + advance
        G_rt.maxpolicy(rho)
        G_rt.update_tree()
        G_rt.prune(args.prune_tol, 'leafs')
        G_rt.grow()

        # APoS improve + advance
        G_apos.set_iter(t)
        G_apos.maxpolicy(rho)
        G_apos.update_tree()
        G_apos.prune(args.prune_tol, 'leafs')
        G_apos.grow()

        # Recompute tvs after this horizon
        tv_rt   = compute_tv_from_tree(G_rt,   DFA, L)
        tv_apos = compute_tv_from_tree(G_apos, DFA, L)
        tv_apos_corr = apply_delta_correction_apos(tv_apos, G_apos, DFA, L, sysAbs,
                                                   delta_sys=delta_corr_scalars, T=t)

        mean_rt_curve.append(region_mean(tv_rt, r0, r1, c0, c1))
        mean_apos_corr_curve.append(region_mean(tv_apos_corr, r0, r1, c0, c1))

        print(f"T={t}: mean(RT)={mean_rt_curve[-1]:.6g}, "
              f"mean(APoS-corr)={mean_apos_corr_curve[-1]:.6g}")

    # Save results (only two curves)
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        f"{args.out_prefix}_curves.npz",
        Ts=np.asarray(Ts),
        mean_rt=np.asarray(mean_rt_curve),
        mean_apos_corr=np.asarray(mean_apos_corr_curve),
        rows=np.array([r0, r1], dtype=int),
        cols=np.array([c0, c1], dtype=int),
        delta_rt=np.asarray(delta_rt_scalars, dtype=float),
        delta_apos_pol=np.asarray(delta_apos_pol_scalars, dtype=float),
        delta_corr=np.asarray(delta_corr_scalars, dtype=float),
    )

    # ---- Plot (two curves) ----
    import matplotlib.ticker as mticker
    fig, ax = plt.subplots(figsize=(12, 3))

    lbl_rt_mean        = r'$\textbf{Optimal Robust-tree Value functions}$  '
    lbl_apos_corr_mean = r'$\textbf{Optimal A-posteriori corrected Value functions}$'

    ax.plot(Ts, mean_rt_curve,        label=lbl_rt_mean,        marker='s')
    ax.plot(Ts, mean_apos_corr_curve, label=lbl_apos_corr_mean, marker='o')

    # Legend: bottom right, one column
    leg = ax.legend(
        prop={'size': 18, 'weight': 'bold'},
        loc='lower right',  # anchor = legend's lower-right corner
        bbox_to_anchor=(0.98, 0.02),  # near bottom-right *inside* the axes
        bbox_transform=ax.transAxes,  # interpret the anchor in axes coords
        ncol=1,
        frameon=True, framealpha=0.9,
        borderpad=0.3, labelspacing=0.25,
        handlelength=2.0, handletextpad=0.5, markerscale=0.9
    )

    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
    ax.tick_params(axis="both", labelsize=20, length=8, width=2, pad=8)
    ax.minorticks_off()
    ax.tick_params(axis="both", which="minor", length=5, width=1.6)
    ax.margins(x=0.08)

    ax.set_xlabel(r'\textbf{Specification Horizon}')
    # ax.set_ylabel(r'\textbf{Corrected satProb (region mean)}')
    for s in ax.spines.values():
        s.set_linewidth(2)
    ax.set_title('')
    ax.grid(False)
    fig.tight_layout()

    fig.savefig(f"{args.out_prefix}_two_curves.eps", format="eps", bbox_inches="tight")
    fig.savefig(f"{args.out_prefix}_two_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nSaved curves to {args.out_prefix}_curves.npz and plot to {args.out_prefix}_two_curves.png")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
