# -----------------------
# Load modified modules if needed
# -----------------------
from importlib import reload
import src.models.linmodel as LinModel_mod
reload(LinModel_mod)
import src.models.mdpmodel as mdpmodel_mod
reload(mdpmodel_mod)
import src.abstraction.utils.labeling as dim_label_mod
reload(dim_label_mod)
import src.dynprog.dfa_tree_r1 as DFATree_mod
reload(DFATree_mod)
# -----------------------

import numpy as np
import polytope as pc
# from pathlib import Path; import h5py
from src.models.linmodel import LinModel
from src.models.mdpmodel import MDPModel
from src.specifications.translate import translate
from src.specifications.utils.dfa_tool import dfa_manipulation
from src.abstraction.utils.labeling import dim_label
from src.dynprog.dfa_tree_r1 import DFATree
from src.vis.dfa_tree_viz import plot_tree_layered



# -----------------------
# Continuous system (2 dimensions, each 1D)
# -----------------------
A = {0: np.array([[0.9]]), 1: np.array([[0.9]])}
B = {0: np.array([[0.5]]), 1: np.array([[0.5]])}
C = {0: np.array([[1.0]]), 1: np.array([[1.0]])}
D = {0: np.array([[0.0]]), 1: np.array([[0.0]])}
Bw = {0: np.array([[0.5]]), 1: np.array([[0.5]])}
mu = {0: np.array([0.0]), 1: np.array([0.0])}
sigma = {0: np.eye(1), 1: np.eye(1)}

sysLTI = {
    0: LinModel(A[0], B[0], C[0], D[0], Bw[0], mu=mu[0], sigma=sigma[0]),
    1: LinModel(A[1], B[1], C[1], D[1], Bw[1], mu=mu[1], sigma=sigma[1]),
}

# -----------------------
# Specification
# -----------------------
DFA = translate('( (!p2 | !p3 ) U p1)')
# Optional manipulation of DFA
DFA, letters = dfa_manipulation(
    DFA,
    index_base = 0,                            # << -- ensure it's 0-based DFA state
    ensure_transitions=True,
    remove_qf_self_loop=True,                  # <- toggle
    replacements={                             # <- optional, don't pass if no needed
        "!p1 & !p2": "p3&!p1&!p2",
        "!p1 & !p3": "!p3&!p1",
    },
    desired_order=['p1', 'p3&!p1&!p2', '!p3&!p1'],  # <- optional
    verbose=True
)

# -----------------------
# APs and regions on original system
# -----------------------
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

# Bind APs to systems
for i in [0, 1]:
    sysLTI[i].X = pc.Polytope(s, bx)
    sysLTI[i].U = pc.Polytope(s, bu)

sysLTI[0].regions = [P1, P2]
sysLTI[1].regions = [P3]
sysLTI[0].AP = ['p1', 'p2']
sysLTI[1].AP = ['p3']

# -----------------------
# Abstraction
# -----------------------

sysAbs = {
    0: MDPModel.from_system(sysLTI[0], nx=1000, nu=5, placement='centers', u_placement='endpoints',
                            tol=1e-19, contract_sum=None, compute_P = '1d'),
    1: MDPModel.from_system(sysLTI[1], nx=1000, nu=5, placement='centers', u_placement='endpoints',
                            tol=1e-19, contract_sum=None, compute_P = '1d'),
}

# -----------------------
# Labeling
# -----------------------
L = dim_label(sysAbs, sysLTI, letters, visualize=True)

# -----------------------
# Policy (randomized uniform) & rho
# -----------------------
dims = sorted(sysAbs.keys())          # e.g. [0, 1, ...]
nx   = [sysAbs[d].N for d in dims]    # states per dim
nu   = [sysAbs[d].M for d in dims]    # actions per dim
nQ   = len(DFA.S)                     # #DFA states (0-based)

# randomized policy: for every DFA state q and dimension d,
# use a uniform distribution over actions at every state i

pol = [
    [np.full((nx[d], nu[d]), 1.0/nu[d], dtype=float) for d in range(len(dims))]
    for _ in range(nQ)
]

# sampling rho: uniform over states in each dimension
rho = [np.full(nx[d], 1.0/nx[d], dtype=float) for d in range(len(dims))]

nx_list = [sysAbs[k].N for k in sorted(sysAbs.keys())]
L_list  = [L[k]       for k in sorted(sysAbs.keys())]

# -----------------------
# Configure delta
# -----------------------
from src.config.delta_ui import interactive_delta
mode, delta_sys, delta_list_tree, apos, dims_sorted = interactive_delta(sysAbs, default=0.001)

# -----------------------
# Tree
# -----------------------
# 1) Initialization
G = DFATree(DFA, sysAbs, pol, nx_list, L_list, delta=delta_list_tree)
G.initiate()

# 2) Grow tree
T = 20

for it in range(1, T + 1):
    print(f"\n=== Iteration {it} ===")
    G.maxpolicy(rho)
    G.update_tree()
    G.prune(0.005,'leafs')
    G.grow()
    plot_tree_layered(G)

# 3) Compute lb(satProb) based on tree
from src.dynprog.utils.treebasedV import compute_tv_from_tree
tv = compute_tv_from_tree(G, DFA, L, max_elements=50_000_000)

if apos == 1:
    from src.dynprog.utils.v_apos import apply_delta_correction_apos
    tv = apply_delta_correction_apos(tv, G, DFA, L, sysAbs, delta_sys=delta_sys, T=T)

# -----------------------
# visualize satProb
# -----------------------
from src.vis.plot_tv import plotV_rank1
plotV_rank1(sysAbs, tv)
import matplotlib.pyplot as plt
plt.show()

exit = 1






