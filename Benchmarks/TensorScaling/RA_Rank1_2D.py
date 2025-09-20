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
from pathlib import Path; import h5py
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
                            tol=1e-19, renormalize=True, contract_sum=1.0),
    1: MDPModel.from_system(sysLTI[1], nx=1000, nu=5, placement='centers', u_placement='endpoints',
                            tol=1e-19, renormalize=True, contract_sum=1.0),
}

# -----------------------
# FIXME @Ruohan: sysAbs[d].P_flat is different (max dev: 0.0399 ) from matlab
# Transition probability matrix computed in mdpmodel NOT IDENTICAL to from MATLAB
# Load and impose Psas computed from MATLAB
# -----------------------
with h5py.File(Path(__file__).resolve().parents[2] / "mdata" / "mdp_probs.mat", "r") as f:
    P1_flat, P2_flat = np.array(f["P1_flat"]).T, np.array(f["P2_flat"]).T

# sysAbs[0].P = P1_flat
# sysAbs[1].P = P2_flat
# def overwrite_P_from_mat(mdp, P_flat_new):
#     P_flat_new = np.asarray(P_flat_new, dtype=float)
#     N = mdp.N
#     if P_flat_new.shape[0] != N or P_flat_new.shape[1] % N != 0:
#         raise ValueError(f"Bad shape {P_flat_new.shape}; expected (N, N*M) with N={N}")
#
#     # Install flat and rebuild blocks in Fortran order (MATLAB-compatible)
#     mdp._P_flat = P_flat_new.copy()                          # private
#     M = P_flat_new.shape[1] // N
#     mdp._P_blocks = P_flat_new.reshape(N, N, M, order="F")   # private
#     mdp.P = mdp._P_flat                                      # public alias
# overwrite_P_from_mat(sysAbs[0], P1_flat)
# overwrite_P_from_mat(sysAbs[1], P2_flat)


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

# -----------------------
# Tree
# -----------------------
# 1) Initialization
nx_list = [sysAbs[k].N for k in sorted(sysAbs.keys())]
L_list  = [L[k]       for k in sorted(sysAbs.keys())]
G = DFATree(DFA, sysAbs, pol, nx_list, L_list)
G.initiate()
plot_tree_layered(G)

# -----------------------
# FIXME 1)@Ruohan: G.maxpolicy(rho): machine precision difference (e-15) in Vxa[1] results in different policy for sys[1] (2nd dimension
# FIXME 2)@Ruohan: G.update_tree(): nonsmooth probabilities updated for dim 0, node.1 (second node), G.V[0] (1,:), only 0 or 1
# FIXME 3)@Ruohan: G.update_tree(): too low probabilities for dim1, node.1 (second node), G.V[1] (1,:)
# G.Pxx not updated identical compared to matlab
# Load and impose Pxx computed from matlab
# -----------------------
with h5py.File(Path(__file__).resolve().parents[2] / "mdata" / "dfa_pxx.mat", "r") as f: Pxx1, Pxx2 = np.array(
    f["Pxx1"]).T, np.array(f["Pxx2"]).T
USE_PXX_OVERRIDE = False  # set True to impose Pxx computed from MATLAB, set False to use Pxx computed based on DFATree

# 2) Tree.maxpolicy
G.maxpolicy(rho)

if USE_PXX_OVERRIDE:
    G.Pxx[1][0] = Pxx1
    G.Pxx[1][1] = Pxx2

G.update_tree()
plot_tree_layered(G)
G.grow()

exit = 1
# -----------------------
# Visualization of satProb
# -----------------------









