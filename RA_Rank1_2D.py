# TODO @Ruohan: dynamic programming part wrong: summing up nodes exceed 1

import numpy as np
# import re
# import types
import polytope as pc
from importlib import reload

from ruohan_model.linmodel import LinModel
from ruohan_model.mdp_model import MDPModel

from ruohan_abstraction.ugrid_util import make_uniform_grid
from ruohan_abstraction.sastransition import transition_matrix_nd_separable
from ruohan_abstraction.dim_letter_label import label_axis_by_letters
from ruohan_abstraction.prep_label_1D import _intervals_from_polytope_1d
from ruohan_abstraction.renorm_scan import renorm_and_scan
from ruohan_abstraction.renorm_scan import _sub_stochasticize_block
from ruohan_abstraction.prep_xax import _nx_of_axes, _axis_from_hx
from ruohan_abstraction.vis_label import plot_binary_matrix

from src.specifications.translate import translate
from src.specifications.dfa_qf_simp import dfa_remove_qf_self_loops_inplace
from src.specifications.letter_consis_dfa import letters_from_dfa_consistent
from src.specifications.dfa_overlap import remove_overlap_inplace
from src.specifications.dfa_01base import normalize_dfa_to_1based_inplace
from src.specifications.dfa_trans_inplace import ensure_transitions_matrix_inplace

from ruohan_vis.vis_r1_2D_satProb import plot_q0_outer_sum
from ruohan_vis.vis_tree import summarize_tree, draw_tree

# # —— Latest DFATreeR1 implementation ——
# import src.dynprog.dfa_tree_r1_h as dfa_mod
# reload(dfa_mod)
# from src.dynprog.dfa_tree_r1_h import DFATreeR1

import src.dynprog.dfa_tree_r1_test as dfa_mod
reload(dfa_mod)
from src.dynprog.dfa_tree_r1_test import DFATree

# -----------------------
# Continuous system (2 dimensions, each 1D)
# -----------------------
A = {1: np.array([[0.9]]), 2: np.array([[0.9]])}
B = {1: np.array([[0.5]]), 2: np.array([[0.5]])}
C = {1: np.array([[1.0]]), 2: np.array([[1.0]])}
D = {1: np.array([[0.0]]), 2: np.array([[0.0]])}
Bw = {1: np.array([[1.0]]), 2: np.array([[1.0]])}
mu = {1: np.array([0.0]), 2: np.array([0.0])}
sigma = {1: np.eye(1), 2: np.eye(1)}

sysLTI = {
    1: LinModel(A[1], B[1], C[1], D[1], Bw[1], mu=mu[1], sigma=sigma[1]),
    2: LinModel(A[2], B[2], C[2], D[2], Bw[2], mu=mu[2], sigma=sigma[2]),
}

# Constraints: -20 <= x <= 5  (H-rep: s x <= b)
s  = np.array([[1], [-1]])
bx = np.array([5, 20])
bu = np.array([5, 5])

# AP regions (1D)
bp1 = np.array([5, 0])      # p1: [0, 5]
bp2 = np.array([0, 5])      # p2: [-5, 0]
bp3 = np.array([-15, 20])   # p3: [-20, -15]

P1 = pc.Polytope(s, bp1)
P2 = pc.Polytope(s, bp2)
P3 = pc.Polytope(s, bp3)

# Bind AP to system
for i in [1, 2]:
    sysLTI[i].X = pc.Polytope(s, bx)
    sysLTI[i].U = pc.Polytope(s, bu)

sysLTI[1].regions = [P1, P2]
sysLTI[2].regions = [P3]
sysLTI[1].AP = ['p1', 'p2']
sysLTI[2].AP = ['p3']

# -----------------------
# Grids (1000 states, 5 inputs)
# -----------------------
uax, uhat = {}, {}
uax[1], uhat[1] = make_uniform_grid(sysLTI[1].U, grid_counts=5,    filter_inside=True)
uax[2], uhat[2] = make_uniform_grid(sysLTI[2].U, grid_counts=5,    filter_inside=True)

xax, xhat = {}, {}
xax[1], xhat[1] = make_uniform_grid(sysLTI[1].X, grid_counts=1000, filter_inside=True)
xax[2], xhat[2] = make_uniform_grid(sysLTI[2].X, grid_counts=1000, filter_inside=True)

## use Prob computed from matlab
import scipy.io as sio

mat = sio.loadmat('sysAbs1_PProb.mat')
Pprob = mat['Pprob']  # numpy.ndarray, shape = (N, N*nu)
P = {}
P[1] = Pprob
P[2] = Pprob

# -----------------------
# Abstract transitions (flat)
# -----------------------
ContractSum = 1  # set to 1 if you want no contractivity for Psas (row sums == 1)

# P = {}
# P[1] = transition_matrix_nd_separable(
#     sys=sysLTI[1],
#     X_axes=xax[1],
#     U_points=uhat[1].ravel(),
#     X_poly=sysLTI[1].X,
#     tol=1e-15,
#     renormalize=True,
#     return_flat=True
# )
# P[2] = transition_matrix_nd_separable(
#     sys=sysLTI[2],
#     X_axes=xax[2],
#     U_points=uhat[2].ravel(),
#     X_poly=sysLTI[2].X,
#     tol=1e-15,
#     renormalize=True,
#     return_flat=True
# )

# Abstract MDP
sysAbs = {
    1: MDPModel(P=P[1], hx=xax[1], orig=sysLTI[1], inputs=uhat[1]),
    2: MDPModel(P=P[2], hx=xax[2], orig=sysLTI[2], inputs=uhat[2]),
}

# Replace NaN with 0 in P; ensure contractivity (row sums of Psas < 1) or not (=1)
renorm_and_scan([sysAbs[1], sysAbs[2]], names=["agent1", "agent2"], target=ContractSum, mode="cap")

# -----------------------
# Specification & letters
# -----------------------
DFA = translate('( (!p2 | !p3 ) U p1)')
# Remove qf->qf if necessary:
dfa_remove_qf_self_loops_inplace(DFA)

normalize_dfa_to_1based_inplace(DFA)
ensure_transitions_matrix_inplace(DFA)

# (Optional) quick check
print("trans shape =", np.asarray(DFA.trans).shape, " |S|=", len(DFA.S), " |act|=", len(DFA.act))
# e.g., find all edges that point to the accepting state (your DFATree uses this logic)
toF_rows, toF_cols = np.where(np.asarray(DFA.trans, dtype=int) == int(DFA.F))
print("#edges to F:", len(toF_rows))

# letters = letters_from_dfa_consistent(DFA)
# print("letters =", letters)

# remove_overlap_inplace(DFA, verbose=True)
#
# letters = letters_from_dfa_consistent(DFA)
# print("letters (disjoint) =", letters)

# ==== Fix overlapping letters (robust to spaces/parentheses) ====
import re
from collections import OrderedDict
from src.specifications.letter_consis_dfa import letters_from_dfa_consistent
from ruohan_abstraction.dim_letter_label import label_axis_by_letters
from ruohan_abstraction.prep_label_1D import _intervals_from_polytope_1d

def _canon_label(s: str) -> str:
    # unify: remove spaces/parentheses, and/AND -> &, not/NOT -> !
    s = (str(s)
         .replace("AND","&").replace("and","&")
         .replace("NOT","!").replace("not","!")
         .replace(" ", "").replace("(", "").replace(")", ""))
    if s in ("", "1"):  # true
        return "1"
    toks = []
    for t in s.split("&"):
        if not t:
            continue
        m = re.fullmatch(r'!?p\d+', t.lower())
        if m:
            toks.append(m.group(0))
    if not toks:
        return "1"
    # canonicalize order by AP index then negation flag
    def key(tok):
        neg = tok.startswith("!")
        idx = int(tok[2:] if neg else tok[1:])
        return (idx, 1 if neg else 0)
    toks = sorted(set(toks), key=key)
    return "&".join(toks)

def _rewrite_labels_inplace(DFA, replacements_pretty):
    """
    replacements_pretty: dict[str -> str]
      keys are patterns to match (any spacing/parentheses allowed),
      values are the 'pretty' strings you want to write back exactly.
    """
    # precompute canonical keys
    repl_canon = {_canon_label(k): v for k, v in replacements_pretty.items()}
    changed = 0

    def _maybe_rewrite_label(lbl: str) -> str:
        nonlocal changed
        c = _canon_label(lbl)
        if c in repl_canon:
            changed += 1
            return replacements_pretty[[k for k in replacements_pretty if _canon_label(k)==c][0]]
        return lbl

    # transitions style
    if hasattr(DFA, "transitions") and DFA.transitions is not None:
        new_tr = []
        for (u, v, data) in list(DFA.transitions):
            lab = data.get("label", data.get("condition", "1"))
            new_lab = _maybe_rewrite_label(lab)
            data = dict(data)
            data["label"] = new_lab
            data["condition"] = new_lab
            new_tr.append((u, v, data))
        DFA.transitions = new_tr

    # graph style
    if hasattr(DFA, "graph"):
        for (u, v, data) in list(DFA.graph.edges(data=True)):
            lab = data.get("label", data.get("condition", "1"))
            new_lab = _maybe_rewrite_label(lab)
            data["label"] = new_lab
            data["condition"] = new_lab

    # refresh DFA.act if present
    if hasattr(DFA, "act"):
        seen = OrderedDict()
        if hasattr(DFA, "transitions") and DFA.transitions is not None:
            for (_, _, data) in DFA.transitions:
                lab = data.get("label", data.get("condition", "1"))
                seen[_canon_label(lab)] = lab
        elif hasattr(DFA, "graph"):
            for (_, _, data) in DFA.graph.edges(data=True):
                lab = data.get("label", data.get("condition", "1"))
                seen[_canon_label(lab)] = lab
        DFA.act = list(seen.values())

    print(f"[rewrite] labels changed: {changed}")

# --- Your specific remap ---
remap = {
    "!p1 & !p2": "p3&!p1&!p2",
    "!p1 & !p3": "!p3&!p1",
}
_rewrite_labels_inplace(DFA, remap)

# Re-collect letters
letters_raw = letters_from_dfa_consistent(DFA)

# Force the desired display order
desired = ['p1', 'p3&!p1&!p2', '!p3&!p1']
# keep desired that actually appear, then append any remaining (if any)
letters = [s for s in desired if s in letters_raw] + [s for s in letters_raw if s not in desired]
print("letters (updated) =", letters)

# ===== Rebuild L1/L2 with the updated letters =====
xax1 = _axis_from_hx(sysAbs[1].hx)
xax2 = _axis_from_hx(sysAbs[2].hx)
aps_set_1 = set(sysLTI[1].AP)
aps_set_2 = set(sysLTI[2].AP)
regions_1 = {ap: _intervals_from_polytope_1d(reg) for ap, reg in zip(sysLTI[1].AP, sysLTI[1].regions)}
regions_2 = {ap: _intervals_from_polytope_1d(reg) for ap, reg in zip(sysLTI[2].AP, sysLTI[2].regions)}

L1 = np.asarray(label_axis_by_letters(xax1, letters, aps_set_1, regions_1), dtype=float)
L2 = np.asarray(label_axis_by_letters(xax2, letters, aps_set_2, regions_2), dtype=float)
assert L1.shape == (len(letters), len(xax1))
assert L2.shape == (len(letters), len(xax2))
print("L1/L2 rebuilt with updated letters.")

# # -----------------------
# # Key: Construction of L — consistent with RA_Rank1_2D.py
# #   For each dimension use label_axis_by_letters(axis, letters, aps_set, regions_by_dim)
# #   regions_by_dim should provide 1D interval lists [(lo, hi)]
# # -----------------------
# # 1) Extract 1D axes
# xax1 = _axis_from_hx(sysAbs[1].hx)
# xax2 = _axis_from_hx(sysAbs[2].hx)
#
# # 2) AP sets for each dimension
# aps_by_dim = {
#     1: set(sysLTI[1].AP),
#     2: set(sysLTI[2].AP),
# }
#
# # 3) Convert polytopes to 1D intervals for each dimension
# regions_by_dim = {
#     1: {ap: _intervals_from_polytope_1d(reg)
#         for ap, reg in zip(sysLTI[1].AP, sysLTI[1].regions)},
#     2: {ap: _intervals_from_polytope_1d(reg)
#         for ap, reg in zip(sysLTI[2].AP, sysLTI[2].regions)},
# }
#
# # 4) Generate L1/L2 (same as RA_Rank1_2D.py)
# L1 = np.asarray(label_axis_by_letters(xax1, letters, aps_by_dim[1], regions_by_dim[1]), dtype=float)
# L2 = np.asarray(label_axis_by_letters(xax2, letters, aps_by_dim[2], regions_by_dim[2]), dtype=float)
# assert L1.shape == (len(letters), len(xax1))
# assert L2.shape == (len(letters), len(xax2))

plot_binary_matrix(L1.astype(int),
                   title="L[1] (letters × x1-grid)",
                   row_labels=letters,
                   outfile="L1.png",
                   x_tick_step=1000)

plot_binary_matrix(L2.astype(int),
                   title="L[1] (letters × x2-grid)",
                   row_labels=letters,
                   outfile="L1.png",
                   x_tick_step=1000)

nx_list = [len(xax1), len(xax2)]

# -----------------------
# Policy & rho
# -----------------------
nu_list = [
    int(np.prod(np.shape(sysAbs[1].inputs))),
    int(np.prod(np.shape(sysAbs[2].inputs))),
]
pol = [
    [np.full((nx_list[d], nu_list[d]), 1.0/nu_list[d], dtype=float) for d in range(2)]
    for _ in range(len(DFA.S))
]
rho = [np.ones(n, dtype=float)/n for n in nx_list]

# -----------------------
# Build tree (if letter order mismatches, rebuild L)
# -----------------------
# DFATreeR1.stochasticize_block = staticmethod(
#     lambda Bu, eps=1e-12: _sub_stochasticize_block(Bu, eps=eps, target=ContractSum, mode="cap")
# )

# Build DFATree (you already have DFA, sysAbs, pol, nx_list, L)
tree = DFATree(DFA, [sysAbs[1], sysAbs[2]], pol, nx_list, [L1, L2]).initiate()

# tree = DFATreeR1(DFA, [sysAbs[1], sysAbs[2]], pol, nx_list, [L1, L2]).initiate()

draw_tree(tree, letters, label='idrole', show=True)
if letters != tree.letters:
    print("[warn] letters mismatch; rebuilding L with tree.letters")
    letters = tree.letters
    L1 = np.asarray(label_axis_by_letters(xax1, letters, aps_by_dim[1], regions_by_dim[1]), dtype=float)
    L2 = np.asarray(label_axis_by_letters(xax2, letters, aps_by_dim[2], regions_by_dim[2]), dtype=float)
    tree.L = [L1, L2]

# One iteration
# tree.maxpolicy(rho)
# tree.update_tree()
# tree.grow()

# 1) Run policy maximization as usual
# tree.maxpolicy(rho)

from scipy.io import loadmat
from scipy.sparse import csr_matrix, issparse
import numpy as np

d = loadmat('pol46_to_use.mat', squeeze_me=True)
P11 = d['pol46_dim1'].tocsr()
P12 = d['pol46_dim2'].tocsr()

q0 = 2
q_idx = list(tree.DFA.S).index(q0)   # usually 1

# 1) overwrite the policies for q0
# (left intentionally blank here if you want maxpolicy to set them)

# 2) force recompute of controlled transitions for those dims
tree.Pxx[q_idx][0] = None
tree.Pxx[q_idx][1] = None
# (optional: compute immediately)
tree.Pxx[q_idx][0] = tree.Pc(tree.sysAbs[0], P11)
tree.Pxx[q_idx][1] = tree.Pc(tree.sysAbs[1], P12)

# 3) sanity checks
def sparse_identical(A, B):
    A = A.tocsr(); B = B.tocsr()
    A.sum_duplicates(); B.sum_duplicates()
    A.sort_indices(); B.sort_indices()
    return (A.shape == B.shape) and ((A != B).nnz == 0)

# print("pol[q0,0]==P11 ?", sparse_identical(tree.pol[q_idx][0], P11))
# print("pol[q0,1]==P12 ?", sparse_identical(tree.pol[q_idx][1], P12))
# print("one-hot rows dim0?", np.all(tree.pol[q_idx][0].getnnz(axis=1)==1))
# print("one-hot rows dim1?", np.all(tree.pol[q_idx][1].getnnz(axis=1)==1))

# then proceed
tree.maxpolicy(rho)
tree.update_tree()
draw_tree(tree, letters, label='idrole', show=True)
tree.grow()
draw_tree(tree, letters, label='idrole', show=True)

# tree.maxpolicy(rho)
tree.update_tree()
draw_tree(tree, letters, label='idrole', show=True)
tree.grow()
draw_tree(tree, letters, label='idrole', show=True)

# for it in range(1):
#     print(f"=== Iter {it+1} ===")
#     tree.maxpolicy(rho)        # Pc has fallback randomization per row
#     tree.update_tree()
#     tree.grow()                # or tree.grow('number', k)
#     print(f"[iter {it+1}] #nodes={tree.tree.number_of_nodes()}  #leafs={len(tree.leafs)}")
#     summarize_tree(tree, letters)
#     tree.check_values(eps=1e-8, include_accepting=False)
#     draw_tree(tree, letters, label='idrole', show=True)

#agg, x1, x2, q0 = plot_q0_outer_sum(tree, sysAbs, title="satProb")
