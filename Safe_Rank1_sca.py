import numpy as np
import re
import types
import polytope as pc
from importlib import reload

from ruohan_model.linmodel import LinModel
from ruohan_model.mdp_model import MDPModel

from ruohan_abstraction.ugrid_util import make_uniform_grid
from ruohan_abstraction.sastransition import transition_matrix_nd_separable
from ruohan_abstraction.dim_letter_label import label_axis_by_letters
from ruohan_abstraction.prep_label_1D import _intervals_from_polytope_1d
from ruohan_abstraction.renorm_scan import renorm_and_scan

from src.specifications.translate import translate
from src.specifications.dfa_qf_simp import dfa_remove_qf_self_loops_inplace
from src.specifications.letter_consis_dfa import letters_from_dfa_consistent

from ruohan_vis.vis_r1_2D_satProb import plot_q0_outer_sum
from ruohan_vis.vis_tree import summarize_tree, draw_tree

# —— Latest DFATreeR1 implementation ——
import src.dynprog.dfa_tree_r1_h as dfa_mod
reload(dfa_mod)
from src.dynprog.dfa_tree_r1_h import DFATreeR1


# -----------------------
# Utilities: DFA / transition blocks / letters / L
# -----------------------
def dfa_remove_qf_self_loops_inplace(dfa):
    """Remove qf→qf edges in place; modify transitions/graph only, keep dfa.trans unchanged."""
    F = int(np.asarray(dfa.F).ravel()[0])
    if hasattr(dfa, "transitions") and dfa.transitions is not None:
        dfa.transitions = [(u, v, d) for (u, v, d) in list(dfa.transitions)
                           if not (int(u) == F and int(v) == F)]
    if hasattr(dfa, "graph"):
        to_remove = []
        for u, v in dfa.graph.edges():
            if int(u) == F and int(v) == F:
                to_remove.append((u, v))
        for u, v in to_remove:
            dfa.graph.remove_edge(u, v)

def _stochasticize_block(Bu, eps=1e-12):
    """Row-stochasticize an (N×N) transition block; clean NaNs; add self-loops for all-zero rows."""
    Bu = np.asarray(Bu, dtype=float).copy()
    Bu[np.isnan(Bu)] = 0.0
    rsum = Bu.sum(axis=1)
    zero_rows = rsum <= eps
    if np.any(zero_rows):
        Bu[zero_rows, :] = 0.0
        idx = np.where(zero_rows)[0]
        for i in idx:
            Bu[i, i if i < Bu.shape[1] else 0] = 1.0
        rsum = Bu.sum(axis=1)
    nz = rsum > eps
    Bu[nz, :] /= rsum[nz, None]
    return Bu

def renorm_sysAbs_inplace(sysAbs_d):
    """
    Row-stochasticize every action block and write back to sysAbs_d.P (if present);
    also rebind sysAbs_d.block() to directly return the repaired blocks.
    """
    N = sysAbs_d.block(0).shape[0]
    nu = int(np.prod(np.shape(getattr(sysAbs_d, "inputs", 1))))
    if nu < 1 and hasattr(sysAbs_d, "P"):
        nu = sysAbs_d.P.shape[1] // N
    assert nu >= 1, "Could not infer number of actions (nu)."

    blocks = [_stochasticize_block(sysAbs_d.block(u)) for u in range(nu)]
    P_new = np.hstack(blocks)

    if hasattr(sysAbs_d, "P"):
        sysAbs_d.P = P_new

    def _block(self, u, _blocks=blocks):
        return _blocks[u]
    sysAbs_d.block = types.MethodType(_block, sysAbs_d)

def scan_blocks(sysAbs_d, name=""):
    """Print per-action block row-sum min/max."""
    B0 = sysAbs_d.block(0)
    N = B0.shape[0]
    nu = int(np.prod(np.shape(getattr(sysAbs_d, "inputs", 1)))) or (
        (sysAbs_d.P.shape[1] // N) if hasattr(sysAbs_d, "P") else 1
    )
    print(f"[scan] {name}: N={N}, nu={nu}")
    for u in range(nu):
        Bu = sysAbs_d.block(u).astype(float)
        rsum = Bu.sum(axis=1)
        print(f"  u={u:2d} rowsum min/max = {rsum.min():.6f} / {rsum.max():.6f}  nan={np.isnan(Bu).any()}")

def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())

def letters_from_dfa_consistent(dfa):
    """Same letter collection logic as DFATreeR1: prefer transitions, then graph."""
    from collections import OrderedDict
    seen = OrderedDict()
    if hasattr(dfa, "transitions") and dfa.transitions is not None:
        for _, _, data in dfa.transitions:
            lab = _norm_label(data.get("label", data.get("condition", "1")))
            if lab not in seen:
                seen[lab] = True
    elif hasattr(dfa, "graph"):
        for _, _, data in dfa.graph.edges(data=True):
            lab = _norm_label(data.get("label", data.get("condition", "1")))
            if lab not in seen:
                seen[lab] = True
    else:
        seen["1"] = True
    return list(seen.keys())

def _axis_from_hx(hx):
    """hx may be np.array or [axis]; unify and return a 1D axis array."""
    return hx[0] if isinstance(hx, (list, tuple)) else hx

def _intervals_from_polytope_1d(Ppoly):
    """Ax<=b (1D) -> [(lo, hi)]"""
    A = np.asarray(Ppoly.A, dtype=float).reshape(-1, 1)
    b = np.asarray(Ppoly.b, dtype=float).ravel()
    lo, hi = -np.inf, +np.inf
    for ai, bi in zip(A[:, 0], b):
        if ai > 0:
            hi = min(hi, bi / ai)
        elif ai < 0:
            lo = max(lo, bi / ai)
        else:
            if bi < 0:
                raise ValueError("Infeasible 0*x <= b with b<0")
    return [(float(lo), float(hi))]


# -----------------------
# System (6 dimensions)
# -----------------------
D = 6  # number of dimensions
A = {i: np.array([[1.0]]) for i in range(1, D+1)}   # MATLAB-equivalent: A=1
B = {i: np.array([[1.0]]) for i in range(1, D+1)}   # B=1
C = {i: np.array([[1.0]]) for i in range(1, D+1)}   # C=1
Dmat = {i: np.array([[0.0]]) for i in range(1, D+1)}
Bw = {i: np.array([[1.0]]) for i in range(1, D+1)}
mu = {i: np.array([0.0]) for i in range(1, D+1)}
sigma = {i: np.eye(1) for i in range(1, D+1)}

sysLTI = {
    i: LinModel(A[i], B[i], C[i], Dmat[i], Bw[i], mu=mu[i], sigma=sigma[i])
    for i in range(1, D+1)
}

# Constraints: x ∈ [-10, 10], u ∈ [-2, 2] (same as MATLAB)
s = np.array([[1], [-1]])     # H-rep: s x <= b
bx = np.array([10, 10])       # -10 ≤ x ≤ 10
bu = np.array([2, 2])         # -2 ≤ u ≤ 2

# One AP per dimension: p1i, covering interval [-5, 5]
bp1 = np.array([5, 5])        # -5 ≤ x ≤ 5
P1 = pc.Polytope(s, bp1)

for i in range(1, D+1):
    sysLTI[i].X = pc.Polytope(s, bx)
    sysLTI[i].U = pc.Polytope(s, bu)
    sysLTI[i].regions = [P1]                  # single region
    sysLTI[i].AP = [f"p1{i}"]                 # one AP name per dimension

# -----------------------
# Grids (moderate size to avoid OOM)
# -----------------------
# MATLAB uses l{i}=100; 200 points/dim here is still feasible (reduce to 100 for speed)
STATE_GRID = 200
INPUT_GRID = 5

uax, uhat, xax, xhat = {}, {}, {}, {}
for i in range(1, D+1):
    uax[i], uhat[i] = make_uniform_grid(sysLTI[i].U, grid_counts=INPUT_GRID, filter_inside=True)
    xax[i], xhat[i] = make_uniform_grid(sysLTI[i].X, grid_counts=STATE_GRID, filter_inside=True)

# -----------------------
# Abstract transitions (flat)
# -----------------------
P = {}
for i in range(1, D+1):
    P[i] = transition_matrix_nd_separable(
        sys=sysLTI[i],
        X_axes=xax[i],
        U_points=uhat[i].ravel(),
        X_poly=sysLTI[i].X,
        tol=1e-15,
        renormalize=True,
        return_flat=True
    )

sysAbs = {
    i: MDPModel(P=P[i], hx=xax[i], orig=sysLTI[i], inputs=uhat[i])
    for i in range(1, D+1)
}

# —— Normalization check & fix ——
print("BEFORE renorm")
for i in range(1, D+1):
    scan_blocks(sysAbs[i], f"agent{i}")
for i in range(1, D+1):
    renorm_sysAbs_inplace(sysAbs[i])
print("AFTER renorm")
for i in range(1, D+1):
    scan_blocks(sysAbs[i], f"agent{i}")

# -----------------------
# Specification: formula =  (safeset) & X(safeset) & ... & X^5(safeset)
# safeset = p11 & p12 & ... & p16
# -----------------------
aps_all = [f"p1{i}" for i in range(1, D+1)]
safeset = " & ".join(aps_all)
terms = []
for k in range(D):  # 6 terms
    terms.append(("X" * k) + "(" + safeset + ")")
formula = " & ".join(terms)

# DFA = translate(formula, aps_all)
DFA = translate(formula)

# Optional: remove qf self-loops
dfa_remove_qf_self_loops_inplace(DFA)

# -----------------------
# letters and L matrices (fully consistent with DFATreeR1)
# -----------------------
letters = letters_from_dfa_consistent(DFA)
print("letters =", letters)

def _axis_from_hx(hx):
    return hx[0] if isinstance(hx, (list, tuple)) else hx

def _intervals_from_polytope_1d(Ppoly):
    A = np.asarray(Ppoly.A, dtype=float).reshape(-1, 1)
    b = np.asarray(Ppoly.b, dtype=float).ravel()
    lo, hi = -np.inf, +np.inf
    for ai, bi in zip(A[:, 0], b):
        if ai > 0:
            hi = min(hi, bi / ai)
        elif ai < 0:
            lo = max(lo, bi / ai)
        else:
            if bi < 0:
                raise ValueError("Infeasible 0*x <= b with b<0")
    return [(float(lo), float(hi))]

nx_list = []
L_list = []
for i in range(1, D+1):
    axis_i = _axis_from_hx(sysAbs[i].hx)
    aps_set_i = set(sysLTI[i].AP)                  # { 'p1i' }
    regions_i = {ap: _intervals_from_polytope_1d(reg)
                 for ap, reg in zip(sysLTI[i].AP, sysLTI[i].regions)}
    Li = np.asarray(label_axis_by_letters(axis_i, letters, aps_set_i, regions_i), dtype=float)
    assert Li.shape == (len(letters), len(axis_i)), f"L[{i}] shape mismatch"
    nx_list.append(len(axis_i))
    L_list.append(Li)

# -----------------------
# Policy & rho
# -----------------------
nu_list = [int(np.prod(np.shape(sysAbs[i].inputs))) for i in range(1, D+1)]
pol = [
    [np.full((nx_list[d], nu_list[d]), 1.0/nu_list[d], dtype=float) for d in range(D)]
    for _ in range(len(DFA.S))
]
rho = [np.ones(nx_list[d], dtype=float)/nx_list[d] for d in range(D)]

# -----------------------
# Build tree (if letters mismatch, rebuild L)
# -----------------------
sysAbs_list = [sysAbs[i] for i in range(1, D+1)]
tree = DFATreeR1(DFA, sysAbs_list, pol, nx_list, L_list).initiate()

if letters != tree.letters:
    print("[warn] letters mismatch; rebuilding L with tree.letters")
    letters = tree.letters
    L_list = []
    for i in range(1, D+1):
        axis_i = _axis_from_hx(sysAbs[i].hx)
        aps_set_i = set(sysLTI[i].AP)
        regions_i = {ap: _intervals_from_polytope_1d(reg)
                     for ap, reg in zip(sysLTI[i].AP, sysLTI[i].regions)}
        Li = np.asarray(label_axis_by_letters(axis_i, letters, aps_set_i, regions_i), dtype=float)
        L_list.append(Li)
    tree.L = L_list

# -----------------------
# Numerical checks (excluding accepting state)
# -----------------------
# Only check 0–1 bounds; skip high-dimensional outer-product diagnostics
draw_tree(tree, letters, label='idrole', show=True)
summarize_tree(tree, letters)
tree.check_values(eps=1e-8, include_accepting=False, check_outer=False)


# -----------------------
# Iterations (no visualization)
# -----------------------
for it in range(12):
    print(f"=== Iter {it+1} ===")
    tree.maxpolicy(rho)        # Pc has fallback row randomization
    tree.update_tree()
    tree.grow()                # or tree.grow('number', k)
    print(f"[iter {it+1}] #nodes={tree.tree.number_of_nodes()}  #leafs={len(tree.leafs)}")
    # Only check 0–1 bounds; skip high-dimensional outer-product diagnostics
    draw_tree(tree, letters, label='idrole', show=True)
    summarize_tree(tree, letters)
    tree.check_values(eps=1e-8, include_accepting=False, check_outer=False)

# (Optional) print memory usage
try:
    import psutil, os
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    print(f"[mem] RSS ~ {rss_mb:.2f} MB")
except Exception:
    pass

m = 1  # prevent some IDEs from exiting immediately
