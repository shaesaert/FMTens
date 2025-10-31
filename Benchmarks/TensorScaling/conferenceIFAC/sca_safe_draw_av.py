# ---------- choose a GUI backend BEFORE importing pyplot ----------
import platform
import matplotlib as mpl
try:
    mpl.use("QtAgg", force=True)          # prefer Qt for interactive
except Exception:
    if platform.system() == "Darwin":
        mpl.use("MacOSX", force=True)     # macOS native
    else:
        mpl.use("TkAgg", force=True)      # fallback

import matplotlib.pyplot as plt
print("Matplotlib backend:", mpl.get_backend())

# ---- TeX styling (comment these two if no LaTeX installed) ----
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{bm}'

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
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

# ---- std + project imports ----
from importlib import reload
import numpy as np
import polytope as pc
from typing import Optional
from itertools import product
import matplotlib.ticker as mticker

# Optional hot-reloads (comment out if not needed)
import src.models.linmodel as LinModel_mod
reload(LinModel_mod)
import src.models.mdpmodel as mdpmodel_mod
reload(mdpmodel_mod)
import src.abstraction.utils.labeling as dim_label_mod
reload(dim_label_mod)
import src.dynprog.dfa_tree_r1 as DFATree_mod
reload(DFATree_mod)

from src.models.linmodel import LinModel
from src.models.mdpmodel import MDPModel
from src.specifications.translate import translate
from src.specifications.utils.dfa_tool import dfa_manipulation
from src.abstraction.utils.labeling import dim_label_eps
from src.dynprog.dfa_tree_r1 import DFATree

# -----------------------
# Small helpers
# -----------------------
def nodes_with_q(G: DFATree, q: int):
    """Tree node ids whose DFA mode == q (via G.Q)."""
    return list(G.Q.get(int(q), []))

def robust_value_at_anchor_mul_then_sum(G: DFATree, qbar_vec, idx_vec):
    """
    Anchor value at idx_vec:
      sum_{(n_0,...,n_{D-1})}  prod_d V^{(d)}[n_d, idx_vec[d]],
    where n_d ranges over nodes_with_q(G, qbar_vec[d]).
    """
    D = G.dim
    assert len(qbar_vec) == D and len(idx_vec) == D

    per_dim = [nodes_with_q(G, int(qbar_vec[d])) for d in range(D)]
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

def fht_term(G: DFATree, T: int, qbar_vec, idx_vec) -> float:
    """
    Triple sum as exat: no algebraic shortcut

      for h = 0 .. T_build (= T)
        for npred = 0 .. h-1
          sum over nodes z that:
             - are at depth == npred  (accessed via Dl[npred][1])
             - are labeled qbar       (z in G.Q[qbar])
          of  Π_d G.V[d][z, idx_x]

    Notes:
      - qbar_vec and idx_vec are assumed to have identical entries (use qbar, idx_x).
    """
    # unify qbar and idx_x
    qbar = int(qbar_vec[0] if hasattr(qbar_vec, "__len__") else qbar_vec)
    idx_x = int(idx_vec[0]  if hasattr(idx_vec, "__len__")  else idx_vec)

    # ensure Dl is present
    if 'Dl' not in G.tree.graph:
        G._recompute_levels()
    Dl = G.tree.graph['Dl']

    # interested set: nodes with DFA label qbar
    interested = set(G.Q.get(qbar, []))
    if not interested or T <= 0:
        return 0.0

    D = len(G.V)

    def node_val(z: int) -> float:
        prod = 1.0
        for d in range(D):
            v = float(G.V[d][z, idx_x])
            if v <= 0.0:
                return 0.0
            prod *= v
        return prod

    total = 0.0
    T_build = int(T)

    # ---- exact three-layer loops ----
    for h in range(0, T_build + 1):          # outer: h = 0 .. T_build
        for npred in range(0, h):            # second: npred = 0 .. h-1
            # nodes at this predecessor count (depth), non-final bucket as requested
            nodes_at_npred = Dl[npred][1] if npred < len(Dl) else []
            # restrict to nodes labeled qbar
            nodes = [z for z in nodes_at_npred if z in interested]
            # inner: sum products over dims
            for z in nodes:
                total += node_val(z)

    return float(total)


def make_uniform_pol_and_rho(DFA, sysAbs, L):
    dims = sorted(sysAbs.keys())
    nx_list = [sysAbs[d].N for d in dims]
    nu_list = [sysAbs[d].M for d in dims]
    nQ = len(DFA.S)
    pol = [
        [np.full((nx_list[d], nu_list[d]), 1.0 / nu_list[d], dtype=float) for d in range(len(dims))]
        for _ in range(nQ)
    ]
    rho = [np.full(nx_list[d], 1.0 / nx_list[d], dtype=float) for d in range(len(dims))]
    L_list = [L[d] for d in dims]
    return pol, rho, nx_list, L_list

# -----------------------
# One full sweep for a given dimension D
# -----------------------
def sweep_for_dim(DIM: int,
                  nx_per_dim: int = 100,
                  nu_per_dim: int = 10,
                  anchor_centers = (40, 45, 50, 55, 59),
                  t_specs = None):
    if t_specs is None:
        t_specs = list(range(1, 8))   # inclusive 4..20

    # Deltas depend on DIM
    delta_i = 0.0717
    Delta   = 1.0 - (1.0 - delta_i) ** DIM

    # Build continuous systems and safeset for this DIM
    s = np.array([[1], [-1]])
    bx = np.array([10, 10])  # X = [-10, 10]
    bu = np.array([2,  2])   # U = [-2,  2]
    bp1 = np.array([5, 5])
    P1 = pc.Polytope(s, bp1)

    # LTI params (identical per dim)
    A = {i: np.array([[0.9]]) for i in range(DIM)}
    B = {i: np.array([[0.5]]) for i in range(DIM)}
    C = {i: np.array([[1.0]]) for i in range(DIM)}
    Dmat = {i: np.array([[0.0]]) for i in range(DIM)}
    Bw = {i: np.array([[0.5]]) for i in range(DIM)}
    mu = {i: np.array([[0.0]]) for i in range(DIM)}
    sigma = {i: np.array([[1.0]]) for i in range(DIM)}

    sysLTI: dict[int, Optional[np.ndarray]] = {}
    safeset = ' '
    AP = set()
    for i in range(DIM):
        sys = LinModel(A[i], B[i], C[i], Dmat[i], Bw[i], mu=mu[i], sigma=sigma[i])
        sys.X = pc.Polytope(s, bx)
        sys.U = pc.Polytope(s, bu)
        sys.AP = [f'p1{i}']
        sys.regions = [P1]
        sysLTI[i] = sys
        safeset += f' & p1{i}'
        AP.update(sys.AP)
    safeset = safeset[3:]
    print(f"[D={DIM}] Safeset: {safeset}")

    # Anchor sets (same index across all dims)
    anchor_idx_list = [[k]*DIM for k in anchor_centers]

    rt_av_vals   = []  # average per t_spec
    apos_av_vals = []
    rt_per_anchor   = []
    apos_per_anchor = []

    # Collect FHT averages and the T_build sequence for plotting
    fht_avg_vals = []
    tbuild_vals  = []

    # for t_spec in t_specs:
    #     print("\n" + "="*70)
    #     print(f"[D={DIM}] Building for t_spec = {t_spec}")
    #     print("="*70)
    #
    #     terms = [f"{'X' * k}({safeset})" for k in range(t_spec)]
    #     formula = " & ".join(terms)
    for t_spec in t_specs:
        print("\n" + "=" * 70)
        print(f"[D={DIM}] Building for t_spec = {t_spec}")
        print("=" * 70)

        ks = {0, t_spec}
        if t_spec >= 2:
            ks.add(t_spec - 1)

        terms = [f"{'X' * k}({safeset})" for k in sorted(ks)]
        formula = " & ".join(terms)

        print(f"[D={DIM}] LTL Formula (t_spec={t_spec}): {formula}")


        DFA = translate(formula)
        DFA, letters = dfa_manipulation(
            DFA,
            index_base=0,
            ensure_transitions=True,
            remove_qf_self_loop=True,
            verbose=False
        )

        # Abstraction for this DIM
        sysAbs = {
            i: MDPModel.from_system(sysLTI[i], nx=nx_per_dim, nu=nu_per_dim,
                                    placement='centers', u_placement='endpoints',
                                    tol=1e-19, contract_sum=None, compute_P='1d')
            for i in range(DIM)
        }

        # Robust labeling
        eps_val = 0.1
        L = dim_label_eps(sysAbs, sysLTI, letters, eps=eps_val, visualize=False, outdir=None, prefix="L_eps")

        # Policy/rho lists
        pol, rho, nx_list, L_list = make_uniform_pol_and_rho(DFA, sysAbs, L)

        # bounds sanity for anchors
        for d in range(DIM):
            for idx_set in anchor_idx_list:
                if idx_set[d] >= nx_list[d]:
                    raise ValueError(f"[D={DIM}] anchor idx {idx_set[d]} out of range for dim {d} (nx={nx_list[d]}).")

        # Initial DFA state and transition table
        q0 = int(np.asarray(DFA.S0).ravel()[0])
        Trans = np.asarray(DFA.trans, dtype=int)

        # Build trees (simple: no prune)
        # T_build = max(1, t_spec - 3)
        T_build = t_spec

        # RT deltas
        delta_VI_vecs_rt  = [np.full(sysAbs[d].N, delta_i, dtype=float) for d in range(DIM)]
        delta_pol_vecs_rt = [np.full(sysAbs[d].N, delta_i, dtype=float) for d in range(DIM)]
        # APoS deltas
        delta_VI_vecs_apos  = [np.zeros(sysAbs[d].N, dtype=float) for d in range(DIM)]
        delta_pol_vecs_apos = [np.full(sysAbs[d].N, delta_i, dtype=float) for d in range(DIM)]

        # ---- RT ----
        G_rt = DFATree(
            DFA, sysAbs, pol, nx_list, L_list,
            delta_VI=delta_VI_vecs_rt, delta_pol=delta_pol_vecs_rt,
            pol_mode='rt', VI_mode='rt'
        ).initiate()
        for it in range(1, T_build + 1):
            G_rt.maxpolicy(rho)
            G_rt.update_tree()
            G_rt.grow()

        # ---- APoS ----
        G_apos = DFATree(
            DFA, sysAbs, pol, nx_list, L_list,
            delta_VI=delta_VI_vecs_apos, delta_pol=delta_pol_vecs_apos,
            pol_mode='apos', VI_mode='apos'
        ).initiate()
        for it in range(1, T_build + 1):
            # if your DFATree has set_iter (for zeta scaling), call safely:
            if hasattr(G_apos, "set_iter"):
                try:
                    G_apos.set_iter(it)
                except Exception:
                    pass
            G_apos.set_iter(it)
            G_apos.maxpolicy(rho)
            G_apos.update_tree()
            G_apos.grow()

        # ---- Evaluate per-anchor, then average ----
        vals_rt   = []
        vals_apos = []
        term_fhts = []

        for idx_vec in anchor_idx_list:
            # derive qbar for this anchor (letter active at each dim's index)
            qbar = []
            valid = True
            for d in range(DIM):
                col = np.asarray(L_list[d][:, idx_vec[d]], dtype=bool)
                if not np.any(col):
                    print(f"[D={DIM} t_spec={t_spec}] WARNING: no active letter at idx={idx_vec[d]} for dim {d}; skip anchor.")
                    valid = False
                    break
                l = int(np.where(col)[0][0])
                qbar.append(int(Trans[q0, l]))
            if not valid:
                continue

            # RT anchor value
            V_rt_k = robust_value_at_anchor_mul_then_sum(G_rt, qbar, idx_vec)
            vals_rt.append(V_rt_k)

            # APoS corrected value
            V_apos_base_k = robust_value_at_anchor_mul_then_sum(G_apos, qbar, idx_vec)
            TermFHT_k     = fht_term(G_apos, T_build, qbar, idx_vec)
            term_fhts.append(TermFHT_k)
            expr_k        = V_apos_base_k - T_build * (1.0 - (1.0 - delta_i)**DIM) + (1.0 - (1.0 - delta_i)**DIM) * TermFHT_k
            V_apos_k      = max(0.0, expr_k)
            vals_apos.append(V_apos_k)

            print(f"[D={DIM} t_spec={t_spec}] idx={idx_vec[0]}  RT={V_rt_k:.6e}  APoS={V_apos_k:.6e}")

        if len(vals_rt) == 0:
            rt_av = np.nan
            apos_av = np.nan
            fht_av = np.nan
            print(f"[D={DIM} t_spec={t_spec}] No valid anchors; averages are NaN.")
        else:
            rt_av = float(np.mean(vals_rt))
            apos_av = float(np.mean(vals_apos))
            fht_av = float(np.mean(term_fhts))
            print(f"[D={DIM} t_spec={t_spec}] AVG over anchors: RT_AV={rt_av:.6e}  APoS_AV={apos_av:.6e}")

        rt_av_vals.append(rt_av)
        apos_av_vals.append(apos_av)
        fht_avg_vals.append(fht_av)
        tbuild_vals.append(T_build)

    # pack results
    rt_av_vals   = np.asarray(rt_av_vals, float)
    apos_av_vals = np.asarray(apos_av_vals, float)
    fht_avg_vals = np.asarray(fht_avg_vals, float)
    tbuild_vals  = np.asarray(tbuild_vals, int)

    # save per-dimension result
    np.savez(
        f"anchor_values_vs_tspec_multi_anchors_D{DIM}.npz",
        t_specs=np.asarray(t_specs),
        anchor_centers=np.asarray(anchor_centers, dtype=int),
        V_rt_avg=rt_av_vals,
        V_apos_avg=apos_av_vals,
        FHT_avg=fht_avg_vals,
        T_builds=tbuild_vals
    )
    print(f"[D={DIM}] Saved data to anchor_values_vs_tspec_multi_anchors_D{DIM}.npz")

    return dict(
        DIM=DIM, t_specs=np.asarray(t_specs),
        V_rt_avg=rt_av_vals, V_apos_avg=apos_av_vals,
        FHT_avg=fht_avg_vals, T_builds=tbuild_vals
    )

# -----------------------
# Run for D=4 and D=20
# -----------------------
t_specs = list(range(1, 8))  # inclusive 4..20
anchor_centers = (40, 45, 50, 55, 59)

res4  = sweep_for_dim(4,  nx_per_dim=100, nu_per_dim=10, anchor_centers=anchor_centers, t_specs=t_specs)
res20 = sweep_for_dim(20, nx_per_dim=100, nu_per_dim=10, anchor_centers=anchor_centers, t_specs=t_specs)

# -----------------------
# Save combined
# -----------------------
np.savez(
    "anchor_values_vs_tspec_multi_anchors_D4_D20.npz",
    t_specs=res4["t_specs"],
    anchor_centers=np.asarray(anchor_centers, dtype=int),
    V_rt_avg_D4=res4["V_rt_avg"],
    V_apos_avg_D4=res4["V_apos_avg"],
    V_rt_avg_D20=res20["V_rt_avg"],
    V_apos_avg_D20=res20["V_apos_avg"],
    FHT_avg_D4=res4["FHT_avg"],
    FHT_avg_D20=res20["FHT_avg"],
    T_builds=res4["T_builds"]        # same schedule for both Ds
)
print("Saved combined data to anchor_values_vs_tspec_multi_anchors_D4_D20.npz")

# -----------------------
# Plot 1: Four curves (same as before)
# -----------------------
plt.ion()
fig1, ax1 = plt.subplots(figsize=(12, 3))

ax1.plot(res4["t_specs"],  res4["V_rt_avg"],    marker='s', linewidth=2.0, label=r'$\textbf{Optimal RT Value functions} (N=4)$')
ax1.plot(res4["t_specs"],  res4["V_apos_avg"],  marker='o', linewidth=2.0, linestyle='--', label=r'$\textbf{Optimal APoS Value functions} (N=4)$')
ax1.plot(res20["t_specs"], res20["V_rt_avg"],   marker='^', linewidth=2.0, label=r'$\textbf{Optimal RT Value functions} (N=20)$')
ax1.plot(res20["t_specs"], res20["V_apos_avg"], marker='v', linewidth=2.0, linestyle='--', label=r'$\textbf{Optimal APoS Value functions} (N=20)$')

leg1 = ax1.legend(
    prop={'size': 16, 'weight': 'bold'},
    loc='upper right', bbox_to_anchor=(0.98, 0.98),
    bbox_transform=ax1.transAxes,
    ncol=1, frameon=True, framealpha=0.9,
    borderpad=0.3, labelspacing=0.25,
    handlelength=2.0, handletextpad=0.5, markerscale=0.9
)

ax1.xaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.tick_params(axis="both", labelsize=18, length=8, width=2, pad=8)
ax1.minorticks_off()
ax1.tick_params(axis="both", which="minor", length=5, width=1.6)
ax1.margins(x=0.08)
ax1.set_xlabel(r'\textbf{Specification Horizon}')
for s in ax1.spines.values():
    s.set_linewidth(2)
ax1.grid(False)

fig1.tight_layout()
fig1.savefig("anchor_values_vs_tspec_multi_anchors_D4_D20.png", dpi=300, bbox_inches="tight")

# -----------------------
# Plot 2 (what you asked for): FHT_term vs T_build
# -----------------------
fig2, ax2 = plt.subplots(figsize=(12, 3))

ax2.plot(res4["T_builds"],  res4["FHT_avg"],  marker='o', linewidth=2.2, label=r'$\textbf{FHT term} \ (N=4)$')
ax2.plot(res20["T_builds"], res20["FHT_avg"], marker='s', linewidth=2.2, label=r'$\textbf{FHT term} \ (N=20)$')

leg2 = ax2.legend(
    prop={'size': 16, 'weight': 'bold'},
    loc='upper left', bbox_to_anchor=(0.02, 0.98),
    bbox_transform=ax2.transAxes,
    ncol=1, frameon=True, framealpha=0.9,
    borderpad=0.3, labelspacing=0.25,
    handlelength=2.0, handletextpad=0.5, markerscale=0.9
)

ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.tick_params(axis="both", labelsize=18, length=8, width=2, pad=8)
ax2.minorticks_off()
ax2.tick_params(axis="both", which="minor", length=5, width=1.6)
ax2.margins(x=0.08)
ax2.set_xlabel(r'\textbf{$T_{\text{build}}$}')
ax2.set_ylabel(r'\textbf{FHT term (avg over anchors)}')
for s in ax2.spines.values():
    s.set_linewidth(2)
ax2.grid(False)

fig2.tight_layout()
fig2.savefig("fht_vs_tbuild_D4_D20.png", dpi=300, bbox_inches="tight")

plt.show(block=True)   # keep interactive windows open
