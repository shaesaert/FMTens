#TODO@Ruohan: this 4D case study does not compute correct values for nodes in tree, need to be fixed


# RA_Rank1_2D_2x2D.py —— two 2D subsystems (p, v); APs depend only on position p;
# generic multi-dimensional L labeling; no visualization
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
from ruohan_abstraction.prep_label_mD import _build_L_for_subsystem
from ruohan_abstraction.renorm_scan import renorm_and_scan
from ruohan_abstraction.renorm_scan import _sub_stochasticize_block
from ruohan_abstraction.prep_xax import _nx_of_axes, _axis_from_hx

from src.specifications.translate import translate
from src.specifications.dfa_qf_simp import dfa_remove_qf_self_loops_inplace
from src.specifications.letter_consis_dfa import letters_from_dfa_consistent

from ruohan_vis.vis_r1_2D_satProb import plot_q0_outer_sum
from ruohan_vis.vis_tree import summarize_tree, draw_tree
from ruohan_vis.viz_pospos_outer_avg import plot_q0_outer_sum_posavg

# —— Latest DFATreeR1 implementation ——
import src.dynprog.dfa_tree_r1_h as dfa_mod
reload(dfa_mod)
from src.dynprog.dfa_tree_r1_h import DFATreeR1




# =============================
# Tunable grid resolution (start small, then increase)
# Set L_POS, L_VEL to 100,100 and LU=10 to match MATLAB l=[100,100], lu=10
# =============================
L_POS = 40
L_VEL = 10
LU    = 10




# =============================
# 1) Continuous systems: two 2D subsystems (consistent with MATLAB)
# =============================
samp_t = 0.5

A1 = np.array([[1.0, samp_t],
               [0.0, 1.0]])
B1 = np.array([[0.0],
               [samp_t]])
C1 = np.array([[1.0, 0.0]])  # observe position
D1 = np.array([[0.0]])
Bw1 = np.sqrt(0.25) * np.eye(2)
mu1 = np.array([0.0, 0.0])
sigma1 = np.eye(2)

A2, B2, C2, D2, Bw2, mu2, sigma2 = A1.copy(), B1.copy(), C1.copy(), D1.copy(), Bw1.copy(), mu1.copy(), sigma1.copy()

sysLTI = {
    1: LinModel(A1, B1, C1, D1, Bw1, mu=mu1, sigma=sigma1),
    2: LinModel(A2, B2, C2, D2, Bw2, mu=mu2, sigma=sigma2),
}

# State constraints: p∈[-20,5], v∈[-5,5]
pl, pu = -20.0, 5.0
vl, vu = -5.0, 5.0

Ax = np.array([[ 1, 0],
               [-1, 0],
               [ 0, 1],
               [ 0,-1]], dtype=float)
bx = np.array([pu, -pl, vu, -vl], dtype=float)

# Input constraints: a∈[-2,2]
al, au = -2.0, 2.0
Au = np.array([[ 1.0],
               [-1.0]])
bu = np.array([au, -al])

for i in (1, 2):
    sysLTI[i].X = pc.Polytope(Ax, bx)
    sysLTI[i].U = pc.Polytope(Au, bu)

# AP regions (vertical strips: constrain position p only; v spans full range)
# x-system (system 1): p1: [0,5], p2: [-5,0]
A_strip = Ax.copy()  # same 4 constraints (p bounds + v full range)
P1 = pc.Polytope(np.array([[ 1, 0],
                           [-1, 0],
                           [ 0, 1],
                           [ 0,-1]], float),
                 np.array([ 5.0,  0.0, vu, -vl], float))
P2 = pc.Polytope(np.array([[ 1, 0],
                           [-1, 0],
                           [ 0, 1],
                           [ 0,-1]], float),
                 np.array([ 0.0,  5.0, vu, -vl], float))

# y-system (system 2): p3: [-20,-15]
P3 = pc.Polytope(np.array([[ 1, 0],
                           [-1, 0],
                           [ 0, 1],
                           [ 0,-1]], float),
                 np.array([-15.0, 20.0, vu, -vl], float))

sysLTI[1].regions = [P1, P2]
sysLTI[1].AP      = ['p1', 'p2']
sysLTI[2].regions = [P3]
sysLTI[2].AP      = ['p3']

# =============================
# 2) Grids and abstraction
# =============================
# State grids: 2D (p, v)
x_axes = {}
u_axes = {}
u_points = {}

x_axes[1], _ = make_uniform_grid(sysLTI[1].X, grid_counts=(L_POS, L_VEL), filter_inside=True)
x_axes[2], _ = make_uniform_grid(sysLTI[2].X, grid_counts=(L_POS, L_VEL), filter_inside=True)

u_axes[1], u_points[1] = make_uniform_grid(sysLTI[1].U, grid_counts=LU, filter_inside=True)
u_axes[2], u_points[2] = make_uniform_grid(sysLTI[2].U, grid_counts=LU, filter_inside=True)

# Number of actions (counts input grid points only)
nu_list = []
for i in (1, 2):
    if isinstance(u_axes[i], (list, tuple)):
        nu_i = int(np.prod([len(ax) for ax in u_axes[i]]))
    else:
        nu_i = len(np.asarray(u_axes[i]).ravel())
    nu_list.append(nu_i)

# Transitions (flat)
P = {}
ContractSum = 0.98 # set to 1 if wants no contractivity for Psas
P[1] = transition_matrix_nd_separable(
    sys=sysLTI[1],
    X_axes=x_axes[1],
    U_points=u_points[1].ravel(),
    X_poly=sysLTI[1].X,
    tol=1e-15,
    renormalize=True,
    return_flat=True
)
P[2] = transition_matrix_nd_separable(
    sys=sysLTI[2],
    X_axes=x_axes[2],
    U_points=u_points[2].ravel(),
    X_poly=sysLTI[2].X,
    tol=1e-15,
    renormalize=True,
    return_flat=True
)

# Abstract MDP (set inputs to a 1D dummy vector of length=nu)
sysAbs = {
    1: MDPModel(P=P[1], hx=x_axes[1], orig=sysLTI[1], inputs=np.arange(nu_list[0])),
    2: MDPModel(P=P[2], hx=x_axes[2], orig=sysLTI[2], inputs=np.arange(nu_list[1])),
}

# replace NAN with 0 in P, ensure contractivity (row sum of Psas < 1) or not (=1)
renorm_and_scan([sysAbs[1], sysAbs[2]], names=["agent1", "agent2"], target=ContractSum, mode="cap")


# State counts nx


nx_list = [_nx_of_axes(x_axes[1]), _nx_of_axes(x_axes[2])]
print("nx_list =", nx_list, "nu_list =", nu_list)

# =============================
# 3) Specification (DFA) and letters/L
# =============================
# Replace with your own scLTL formula if desired; we reuse the previous example (with p1/p2/p3)
formula = '( (!p2 | !p3 ) U p1)'
DFA = translate(formula)
dfa_remove_qf_self_loops_inplace(DFA)

letters = letters_from_dfa_consistent(DFA)
print("letters =", letters)

# Global AP set (for missing APs → True)
all_aps = sorted(set(sysLTI[1].AP) | set(sysLTI[2].AP))

# L for each subsystem (computed in a unified multidimensional manner)
aps_by_dim = {1: list(sysLTI[1].AP), 2: list(sysLTI[2].AP)}
regions_by_dim = {
    1: {ap: sysLTI[1].regions[i] for i, ap in enumerate(sysLTI[1].AP)},
    2: {ap: sysLTI[2].regions[i] for i, ap in enumerate(sysLTI[2].AP)},
}
L_all = []
for d in (1, 2):
    Ld = _build_L_for_subsystem(
        x_axes[d], letters, aps_by_dim[d], regions_by_dim[d], all_aps
    )
    assert Ld.shape[1] == nx_list[d-1], f"L[{d}] shape {Ld.shape} != nx {nx_list[d-1]}"
    L_all.append(Ld)

# =============================
# 4) Policy / rho / tree
# =============================
pol = [
    [np.full((nx_list[d], nu_list[d]), 1.0/nu_list[d], dtype=float) for d in range(2)]
    for _ in range(len(DFA.S))
]
rho = [np.ones(n, dtype=float)/n for n in nx_list]

DFATreeR1.stochasticize_block = staticmethod(
    lambda Bu, eps=1e-12: _sub_stochasticize_block(Bu, eps=eps, target=ContractSum, mode="cap")
)
tree = DFATreeR1(DFA, [sysAbs[1], sysAbs[2]], pol, nx_list, L_all).initiate()
draw_tree(tree, letters, label='idrole', show=True)

# Align letters (if DFA internal order differs)
if letters != tree.letters:
    print("[warn] letters mismatch; rebuilding L with tree.letters")
    letters = tree.letters
    L_all = []
    for d in (1, 2):
        Ld = _build_L_for_subsystem(x_axes[d], letters, aps_by_dim[d], regions_by_dim[d], all_aps)
        L_all.append(Ld)
    tree.L = L_all


summarize_tree(tree, letters)
tree.check_values(eps=1e-8, include_accepting=False)

for it in range(8):
    print(f"=== Iter {it+1} ===")
    tree.maxpolicy(rho)
    tree.update_tree()
    tree.grow()
    print(f"[iter {it+1}] #nodes={tree.tree.number_of_nodes()}  #leafs={len(tree.leafs)}")
    draw_tree(tree, letters, label='idrole', show=True)
    summarize_tree(tree, letters)

    tree.check_values(eps=1e-8, include_accepting=False)

agg, p1, p2, q0 = plot_q0_outer_sum_posavg(tree, [sysAbs[1], sysAbs[2]],
                                           title="Position–Position (avg over velocity)")
m = 1  # prevent IDE from exiting immediately
