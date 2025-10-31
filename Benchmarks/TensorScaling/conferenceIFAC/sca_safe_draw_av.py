# # -----------------------------------------
# # Evaluate average anchor values AFTER all iterations
# # (per t_spec, average over 5 anchors: 40,45,50,55,59 across all dimensions)
# # -----------------------------------------
#
# # ---------- choose a GUI backend BEFORE importing pyplot ----------
# import platform
# import matplotlib as mpl
# try:
#     mpl.use("QtAgg", force=True)          # prefer Qt for interactive
# except Exception:
#     if platform.system() == "Darwin":
#         mpl.use("MacOSX", force=True)     # macOS native
#     else:
#         mpl.use("TkAgg", force=True)      # fallback
#
# import matplotlib.pyplot as plt
# print("Matplotlib backend:", mpl.get_backend())
#
# # ---- TeX styling (comment these two if no LaTeX installed) ----
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{bm}'
#
# mpl.rcParams.update({
#     "font.family": "DejaVu Sans",
#     "mathtext.fontset": "dejavusans",
#     "font.size": 16,
#     "axes.titlesize": 18,
#     "axes.labelsize": 18,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14,
#     "legend.fontsize": 14,
#     "font.weight": "bold",
#     "axes.titleweight": "bold",
#     "axes.labelweight": "bold",
# })
#
# # ---- std + project imports ----
# from importlib import reload
# import numpy as np
# import polytope as pc
# from typing import Optional
# from itertools import product
# import matplotlib.ticker as mticker
#
# # Optional hot-reloads (comment out if not needed)
# import src.models.linmodel as LinModel_mod
# reload(LinModel_mod)
# import src.models.mdpmodel as mdpmodel_mod
# reload(mdpmodel_mod)
# import src.abstraction.utils.labeling as dim_label_mod
# reload(dim_label_mod)
# import src.dynprog.dfa_tree_r1 as DFATree_mod
# reload(DFATree_mod)
#
# from src.models.linmodel import LinModel
# from src.models.mdpmodel import MDPModel
# from src.specifications.translate import translate
# from src.specifications.utils.dfa_tool import dfa_manipulation
# from src.abstraction.utils.labeling import dim_label_eps
# from src.dynprog.dfa_tree_r1 import DFATree
#
# # -----------------------
# # Small helpers
# # -----------------------
# def nodes_with_q(G: DFATree, q: int):
#     """Tree node ids whose DFA mode == q (via G.Q)."""
#     return list(G.Q.get(int(q), []))
#
# def nodes_with_q_and_predcount(G: DFATree, q: int, ns: int):
#     """
#     Nodes in DFA mode q that have exactly ns DFA-predecessors.
#     We use out-degree as that count.
#     """
#     q = int(q)
#     return [n for n in G.Q.get(q, []) if G.tree.out_degree(n) == ns]
#
# def robust_value_at_anchor_mul_then_sum(G: DFATree, qbar_vec, idx_vec):
#     """
#     Anchor value at idx_vec:
#       sum_{(n_0,...,n_{D-1})}  prod_d V^{(d)}[n_d, idx_vec[d]],
#     where n_d ranges over nodes_with_q(G, qbar_vec[d]).
#     """
#     D = G.dim
#     assert len(qbar_vec) == D and len(idx_vec) == D
#
#     per_dim = [nodes_with_q(G, int(qbar_vec[d])) for d in range(D)]
#     if any(len(lst) == 0 for lst in per_dim):
#         return 0.0
#
#     total = 0.0
#     for combo in product(*per_dim):
#         term = 1.0
#         for d, n in enumerate(combo):
#             term *= float(G.V[d][n, idx_vec[d]])
#             if term == 0.0:
#                 break
#         total += term
#     return float(total)
#
# def fht_term(G: DFATree, T: int, qbar_vec, idx_vec):
#     """
#     A-posteriori first-hitting-time term at idx_vec, evaluated after T iterations:
#
#       sum_{ns=1}^{T-1} (T-ns) *
#         [ sum_{(n_0,...,n_{D-1})} prod_d V[d][n_d, idx_vec[d]] ],
#
#     where n_d ranges over nodes_with_q_and_predcount(G, qbar_vec[d], ns).
#     """
#     D = len(G.V)
#     total = 0.0
#     for ns in range(1, T):
#         weight = (T - ns)
#
#         node_sets = []
#         for d in range(D):
#             nds = nodes_with_q_and_predcount(G, int(qbar_vec[d]), ns)
#             if not nds:
#                 node_sets = []
#                 break
#             node_sets.append(nds)
#         if not node_sets:
#             continue
#
#         s_ns = 0.0
#         for combo in product(*node_sets):
#             prod_val = 1.0
#             for d, n_d in enumerate(combo):
#                 prod_val *= float(G.V[d][n_d, idx_vec[d]])
#                 if prod_val == 0.0:
#                     break
#             s_ns += prod_val
#
#         total += weight * s_ns
#
#     return float(total)
#
# # -----------------------
# # Base system (shared across runs)
# # -----------------------
# dim = 20              # choose from {3,4,5,6,7,8,9,10}
# nx_per_dim = 100         # set to 1000 for your large run
# nu_per_dim = 10
# bx = np.array([10, 10])  # X = [-10, 10]
# bu = np.array([2,  2])   # U = [-2,  2]
#
# # One safe AP per dimension: p1_i := [-5, 5]
# s = np.array([[1], [-1]])
# bp1 = np.array([5, 5])
# P1 = pc.Polytope(s, bp1)
#
# # LTI params (identical per dim)
# A = {i: np.array([[0.9]]) for i in range(dim)}
# B = {i: np.array([[0.5]]) for i in range(dim)}
# C = {i: np.array([[1.0]]) for i in range(dim)}
# Dmat = {i: np.array([[0.0]]) for i in range(dim)}
# Bw = {i: np.array([[0.5]]) for i in range(dim)}
# mu = {i: np.array([[0.0]]) for i in range(dim)}
# sigma = {i: np.array([[1.0]]) for i in range(dim)}
#
# # Build continuous systems
# sysLTI: dict[int, Optional[np.ndarray]] = {}
# AP = set()
# act = {}
# safeset = ' '
# for i in range(dim):
#     sys = LinModel(A[i], B[i], C[i], Dmat[i], Bw[i], mu=mu[i], sigma=sigma[i])
#     sys.X = pc.Polytope(s, bx)
#     sys.U = pc.Polytope(s, bu)
#     sys.AP = [f'p1{i}']
#     sys.regions = [P1]
#     sysLTI[i] = sys
#
#     act[i] = [' ', f'p1{i}']
#     safeset += f' & p1{i}'
#     AP.update(sys.AP)
#
# safeset = safeset[3:]
# print(f"Safeset: {safeset}")
#
# # Uniform policy/rho templates (rebuilt per run after we know DFA size)
# def make_uniform_pol_and_rho(DFA, sysAbs, L):
#     dims = sorted(sysAbs.keys())
#     nx_list = [sysAbs[d].N for d in dims]
#     nu_list = [sysAbs[d].M for d in dims]
#     nQ = len(DFA.S)
#     pol = [
#         [np.full((nx_list[d], nu_list[d]), 1.0 / nu_list[d], dtype=float) for d in range(len(dims))]
#         for _ in range(nQ)
#     ]
#     rho = [np.full(nx_list[d], 1.0 / nx_list[d], dtype=float) for d in range(len(dims))]
#     L_list = [L[d] for d in dims]
#     return pol, rho, nx_list, L_list
#
# # Deltas
# delta_i = 0.0717
# Delta = 1.0 - (1.0 - delta_i) ** dim
#
# # ---- Anchor centers to sample (same index across all dims) ----
# anchor_centers = [40, 45, 50, 55, 59]  # 5 anchors
# anchor_idx_list = [[k]*dim for k in anchor_centers]
#
# # -----------------------
# # Sweep t_spec = 4..20
# # -----------------------
# t_specs = list(range(4, 20))
# rt_av_vals   = []  # average over anchors
# apos_av_vals = []  # average over anchors
#
# # also keep per-anchor values for debugging/saving
# rt_per_anchor   = []  # list of arrays (len=5) per t_spec
# apos_per_anchor = []
#
# for t_spec in t_specs:
#     print("\n" + "="*70)
#     print(f"Building for t_spec = {t_spec}")
#     print("="*70)
#
#     # -----------------------
#     # Spec and DFA for this t_spec
#     # -----------------------
#     formula = ' '
#     for t_index in range(1, t_spec - 1):
#         X_prefix = 'X' * (t_index - 1)
#         formula += f' & {X_prefix}({safeset})'
#     formula = formula[3:]
#     print(f"LTL Formula: {formula}")
#
#     DFA = translate(formula)
#     DFA, letters = dfa_manipulation(
#         DFA,
#         index_base=0,
#         ensure_transitions=True,
#         remove_qf_self_loop=True,
#         verbose=False
#     )
#
#     # -----------------------
#     # Abstraction (reuse parameters)
#     # -----------------------
#     sysAbs = {
#         i: MDPModel.from_system(sysLTI[i], nx=nx_per_dim, nu=nu_per_dim,
#                                 placement='centers', u_placement='endpoints',
#                                 tol=1e-19, contract_sum=None, compute_P='1d')
#         for i in range(dim)
#     }
#
#     # Robust labeling (depends on DFA letters)
#     eps_val = 0.1
#     L = dim_label_eps(sysAbs, sysLTI, letters, eps=eps_val, visualize=False, outdir=None, prefix="L_eps")
#
#     # Uniform policy/rho for this DFA/abstraction
#     pol, rho, nx_list, L_list = make_uniform_pol_and_rho(DFA, sysAbs, L)
#
#     # bounds sanity for anchors
#     for d in range(dim):
#         for idx_set in anchor_idx_list:
#             if idx_set[d] >= nx_list[d]:
#                 raise ValueError(f"anchor idx {idx_set[d]} out of range for dim {d} (nx={nx_list[d]}).")
#
#     # Initial DFA state from S0
#     q0 = int(np.asarray(DFA.S0).ravel()[0])
#     Trans = np.asarray(DFA.trans, dtype=int)
#
#     # -----------------------
#     # Build trees for this t_spec (no pruning). Keep it simple.
#     # Depth chosen relative to t_spec:
#     # -----------------------
#     T_build = max(1, t_spec - 3)
#
#     # RT deltas: Δ_VI = 0.0717, Δ_pol = 0.0717
#     delta_VI_vecs_rt  = [np.full(sysAbs[d].N, delta_i, dtype=float) for d in range(dim)]
#     delta_pol_vecs_rt = [np.full(sysAbs[d].N, delta_i, dtype=float) for d in range(dim)]
#
#     # APoS deltas: Δ_VI = 0.0, Δ_pol = 0.0717
#     delta_VI_vecs_apos  = [np.zeros(sysAbs[d].N, dtype=float) for d in range(dim)]
#     delta_pol_vecs_apos = [np.full(sysAbs[d].N, delta_i, dtype=float) for d in range(dim)]
#
#     # ---- RT ----
#     G_rt = DFATree(
#         DFA, sysAbs, pol, nx_list, L_list,
#         delta_VI=delta_VI_vecs_rt, delta_pol=delta_pol_vecs_rt,
#         pol_mode='rt', VI_mode='rt'
#     ).initiate()
#     for it in range(1, T_build + 1):
#         G_rt.maxpolicy(rho)
#         G_rt.update_tree()
#         G_rt.grow()
#
#     # ---- APoS ----
#     G_apos = DFATree(
#         DFA, sysAbs, pol, nx_list, L_list,
#         delta_VI=delta_VI_vecs_apos, delta_pol=delta_pol_vecs_apos,
#         pol_mode='apos', VI_mode='apos'
#     ).initiate()
#     for it in range(1, T_build + 1):
#         G_apos.maxpolicy(rho)
#         G_apos.update_tree()
#         G_apos.grow()
#
#     # -----------------------
#     # Evaluate per-anchor, then average
#     # -----------------------
#     vals_rt   = []
#     vals_apos = []
#     for idx_vec in anchor_idx_list:
#         # derive qbar for this anchor (letter active at each dim's index)
#         qbar = []
#         valid = True
#         for d in range(dim):
#             col = np.asarray(L_list[d][:, idx_vec[d]], dtype=bool)
#             if not np.any(col):
#                 print(f"[t_spec={t_spec}] WARNING: no active letter at idx={idx_vec[d]} for dim {d}; skipping this anchor.")
#                 valid = False
#                 break
#             l = int(np.where(col)[0][0])
#             qbar.append(int(Trans[q0, l]))
#         if not valid:
#             continue
#
#         # RT value at this anchor
#         V_rt_k = robust_value_at_anchor_mul_then_sum(G_rt, qbar, idx_vec)
#         vals_rt.append(V_rt_k)
#
#         # APoS corrected value at this anchor
#         V_apos_base_k = robust_value_at_anchor_mul_then_sum(G_apos, qbar, idx_vec)
#         TermFHT_k     = fht_term(G_apos, T_build, qbar, idx_vec)
#         expr_k        = V_apos_base_k - T_build * Delta + Delta * TermFHT_k
#         V_apos_k      = max(0.0, expr_k)  # floor at 0
#         vals_apos.append(V_apos_k)
#
#         print(f"[t_spec={t_spec}] idx={idx_vec[0]}  RT={V_rt_k:.6e}  APoS={V_apos_k:.6e}")
#
#     # average across anchors (if any valid)
#     if len(vals_rt) == 0:
#         rt_av = np.nan
#         apos_av = np.nan
#         print(f"[t_spec={t_spec}] No valid anchors; averages are NaN.")
#     else:
#         rt_av = float(np.mean(vals_rt))
#         apos_av = float(np.mean(vals_apos))
#         print(f"[t_spec={t_spec}] AVG over anchors: RT_AV={rt_av:.6e}  APoS_AV={apos_av:.6e}")
#
#     rt_av_vals.append(rt_av)
#     apos_av_vals.append(apos_av)
#     rt_per_anchor.append(np.array(vals_rt, dtype=float))
#     apos_per_anchor.append(np.array(vals_apos, dtype=float))
#
# # -----------------------
# # Save and plot (interactive window; blocking show)
# # -----------------------
# rt_av_vals   = np.asarray(rt_av_vals, float)
# apos_av_vals = np.asarray(apos_av_vals, float)
#
# # pack per-anchor into uniform shape (pad with NaN if some anchors skipped)
# max_k = len(anchor_centers)
# rt_mat = np.full((len(t_specs), max_k), np.nan, dtype=float)
# ap_mat = np.full((len(t_specs), max_k), np.nan, dtype=float)
# for i, (rrow, arow) in enumerate(zip(rt_per_anchor, apos_per_anchor)):
#     k = min(max_k, len(rrow))
#     if k > 0:
#         rt_mat[i, :k] = rrow[:k]
#         ap_mat[i, :k] = arow[:k]
#
# np.savez(
#     "anchor_values_vs_tspec_multi_anchors.npz",
#     t_specs=np.asarray(t_specs),
#     anchor_centers=np.asarray(anchor_centers, dtype=int),
#     V_rt_avg=rt_av_vals,
#     V_apos_avg=apos_av_vals,
#     V_rt_per_anchor=rt_mat,
#     V_apos_per_anchor=ap_mat
# )
# print("Saved data to anchor_values_vs_tspec_multi_anchors.npz")
#
# fig, ax = plt.subplots(figsize=(12, 3))
#
# lbl_rt   = r'$\textbf{Optimal Robust-tree Value functions}$'
# lbl_apos = r'$\textbf{Optimal A-posteriori corrected Value functions}$'
#
# ax.plot(t_specs, rt_av_vals,   marker='s', linewidth=2.0, label=lbl_rt)
# ax.plot(t_specs, apos_av_vals, marker='o', linewidth=2.0, label=lbl_apos)
#
# # Legend: bottom-right, inside axes
# leg = ax.legend(
#     prop={'size': 18, 'weight': 'bold'},
#     loc='upper right',
#     bbox_to_anchor=(0.98, 0.98),
#     bbox_transform=ax.transAxes,
#     ncol=1,
#     frameon=True, framealpha=0.9,
#     borderpad=0.3, labelspacing=0.25,
#     handlelength=2.0, handletextpad=0.5, markerscale=0.9
# )
#
# # Bold tick labels
# ax.xaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
# ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
# ax.tick_params(axis="both", labelsize=20, length=8, width=2, pad=8)
# ax.minorticks_off()
# ax.tick_params(axis="both", which="minor", length=5, width=1.6)
# ax.margins(x=0.08)
#
# ax.set_xlabel(r'\textbf{Specification Horizon}')
# # ax.set_ylabel(r'\textbf{Value at anchor (avg)}')
# for s in ax.spines.values():
#     s.set_linewidth(2)
# ax.grid(False)
# fig.tight_layout()
#
# fig.savefig("anchor_values_vs_tspec_multi_anchors.png", dpi=300, bbox_inches="tight")
# plt.show(block=True)   # keep interactive window open


# -----------------------------------------
# Average anchor values AFTER all iterations
# Four curves on one interactive plot:
#   - RT avg (dim=4),  APoS avg (dim=4)
#   - RT avg (dim=20), APoS avg (dim=20)
# -----------------------------------------

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

def nodes_with_q_and_predcount(G: DFATree, q: int, ns: int):
    """
    Nodes in DFA mode q that have exactly ns DFA-predecessors.
    We use out-degree as that count.
    """
    q = int(q)
    return [n for n in G.Q.get(q, []) if G.tree.out_degree(n) == ns]

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

def fht_term(G: DFATree, T: int, qbar_vec, idx_vec):
    """
    A-posteriori first-hitting-time term at idx_vec, evaluated after T iterations:

      sum_{ns=1}^{T-1} (T-ns) *
        [ sum_{(n_0,...,n_{D-1})} prod_d V[d][n_d, idx_vec[d]] ],

    where n_d ranges over nodes_with_q_and_predcount(G, qbar_vec[d], ns).
    """
    D = len(G.V)
    total = 0.0
    for ns in range(1, T):
        weight = (T - ns)

        node_sets = []
        for d in range(D):
            nds = nodes_with_q_and_predcount(G, int(qbar_vec[d]), ns)
            if not nds:
                node_sets = []
                break
            node_sets.append(nds)
        if not node_sets:
            continue

        s_ns = 0.0
        for combo in product(*node_sets):
            prod_val = 1.0
            for d, n_d in enumerate(combo):
                prod_val *= float(G.V[d][n_d, idx_vec[d]])
                if prod_val == 0.0:
                    break
            s_ns += prod_val

        total += weight * s_ns

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
        t_specs = list(range(4, 21))   # inclusive 4..20

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

    for t_spec in t_specs:
        print("\n" + "="*70)
        print(f"[D={DIM}] Building for t_spec = {t_spec}")
        print("="*70)

        # Spec/DFA for this t_spec   (formula: & X^k(safeset))
        formula = ' '
        for t_index in range(1, t_spec - 1):
            X_prefix = 'X' * (t_index - 1)
            formula += f' & {X_prefix}({safeset})'
        formula = formula[3:]
        print(f"[D={DIM}] LTL Formula: {formula}")

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
        T_build = max(1, t_spec - 3)

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
            G_apos.set_iter(it)
            G_apos.maxpolicy(rho)
            G_apos.update_tree()
            G_apos.grow()

        # ---- Evaluate per-anchor, then average ----
        vals_rt   = []
        vals_apos = []
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
            expr_k        = V_apos_base_k - T_build * Delta + Delta * TermFHT_k
            V_apos_k      = max(0.0, expr_k)  # floor at 0
            vals_apos.append(V_apos_k)

            print(f"[D={DIM} t_spec={t_spec}] idx={idx_vec[0]}  RT={V_rt_k:.6e}  APoS={V_apos_k:.6e}")

        if len(vals_rt) == 0:
            rt_av = np.nan
            apos_av = np.nan
            print(f"[D={DIM} t_spec={t_spec}] No valid anchors; averages are NaN.")
        else:
            rt_av = float(np.mean(vals_rt))
            apos_av = float(np.mean(vals_apos))
            print(f"[D={DIM} t_spec={t_spec}] AVG over anchors: RT_AV={rt_av:.6e}  APoS_AV={apos_av:.6e}")

        rt_av_vals.append(rt_av)
        apos_av_vals.append(apos_av)
        rt_per_anchor.append(np.array(vals_rt, dtype=float))
        apos_per_anchor.append(np.array(vals_apos, dtype=float))

    # pack results
    rt_av_vals   = np.asarray(rt_av_vals, float)
    apos_av_vals = np.asarray(apos_av_vals, float)
    max_k = len(anchor_centers)
    rt_mat = np.full((len(t_specs), max_k), np.nan, dtype=float)
    ap_mat = np.full((len(t_specs), max_k), np.nan, dtype=float)
    for i, (rrow, arow) in enumerate(zip(rt_per_anchor, apos_per_anchor)):
        k = min(max_k, len(rrow))
        if k > 0:
            rt_mat[i, :k] = rrow[:k]
            ap_mat[i, :k] = arow[:k]

    # save per-dimension result
    np.savez(
        f"anchor_values_vs_tspec_multi_anchors_D{DIM}.npz",
        t_specs=np.asarray(t_specs),
        anchor_centers=np.asarray(anchor_centers, dtype=int),
        V_rt_avg=rt_av_vals,
        V_apos_avg=apos_av_vals,
        V_rt_per_anchor=rt_mat,
        V_apos_per_anchor=ap_mat
    )
    print(f"[D={DIM}] Saved data to anchor_values_vs_tspec_multi_anchors_D{DIM}.npz")

    return dict(
        DIM=DIM, t_specs=np.asarray(t_specs),
        V_rt_avg=rt_av_vals, V_apos_avg=apos_av_vals
    )

# -----------------------
# Run for D=4 and D=20, then plot four curves
# -----------------------
t_specs = list(range(4, 21))  # inclusive 4..20
anchor_centers = (40, 45, 50, 55, 59)


res4  = sweep_for_dim(4,  nx_per_dim=100, nu_per_dim=10, anchor_centers=anchor_centers, t_specs=t_specs)
res20 = sweep_for_dim(20, nx_per_dim=100, nu_per_dim=10, anchor_centers=anchor_centers, t_specs=t_specs)

# -----------------------
# Save combined & plot (interactive window; blocking show)
# -----------------------
np.savez(
    "anchor_values_vs_tspec_multi_anchors_D4_D20.npz",
    t_specs=res4["t_specs"],
    anchor_centers=np.asarray(anchor_centers, dtype=int),
    V_rt_avg_D4=res4["V_rt_avg"],
    V_apos_avg_D4=res4["V_apos_avg"],
    V_rt_avg_D20=res20["V_rt_avg"],
    V_apos_avg_D20=res20["V_apos_avg"]
)
print("Saved combined data to anchor_values_vs_tspec_multi_anchors_D4_D20.npz")

plt.ion()
fig, ax = plt.subplots(figsize=(12, 3))

# Four curves: styles kept distinct
ax.plot(res4["t_specs"],  res4["V_rt_avg"],   marker='s', linewidth=2.0, label=r'$\textbf{Optimal RT Value functions} (N=4) $')
ax.plot(res4["t_specs"],  res4["V_apos_avg"], marker='o', linewidth=2.0, linestyle='--', label=r'$\textbf{Optimal APoS Value functions} (N=4) $')
ax.plot(res20["t_specs"], res20["V_rt_avg"],  marker='^', linewidth=2.0, label=r'$\textbf{Optimal RT Value functions} (N=20) $')
ax.plot(res20["t_specs"], res20["V_apos_avg"],marker='v', linewidth=2.0, linestyle='--', label=r'$\textbf{Optimal APoS Value functions} (N=20) $')

# Legend: upper-right inside axes
leg = ax.legend(
    prop={'size': 16, 'weight': 'bold'},
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),
    bbox_transform=ax.transAxes,
    ncol=1,
    frameon=True, framealpha=0.9,
    borderpad=0.3, labelspacing=0.25,
    handlelength=2.0, handletextpad=0.5, markerscale=0.9
)

# Bold tick labels
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.tick_params(axis="both", labelsize=18, length=8, width=2, pad=8)
ax.minorticks_off()
ax.tick_params(axis="both", which="minor", length=5, width=1.6)
ax.margins(x=0.08)

ax.set_xlabel(r'\textbf{Specification Horizon}')
# ax.set_ylabel(r'\textbf{Average Value at Anchors}')
for s in ax.spines.values():
    s.set_linewidth(2)
ax.grid(False)
fig.tight_layout()

fig.savefig("anchor_values_vs_tspec_multi_anchors_D4_D20.png", dpi=300, bbox_inches="tight")
plt.show(block=True)   # keep interactive window open
