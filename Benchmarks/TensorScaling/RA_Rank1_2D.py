import numpy as np
import polytope as pc
from importlib import reload

from src.models.linmodel import LinModel
# USE FOLLOWING LINES IF CHANGES ARE MADE IN 'src.models.mdpmodel'
# import src.models.mdpmodel as MDPModel_mod
# reload(MDPModel_mod)
from src.models.mdpmodel import MDPModel
from src.specifications.translate import translate
from src.specifications.utils.dfa_tool import dfa_manipulation
# USE FOLLOWING LINES IF CHANGES ARE MADE IN 'src.abstraction.utils.labeling'
# import src.abstraction.utils.labeling as dim_label_mod
# reload(dim_label_mod)
from src.abstraction.utils.labeling import dim_label

#TODO@Ruohan:
# from src.dynprog import dfa_tree_r1

# -----------------------
# Continuous system (2 dimensions, each 1D)
# -----------------------
A = {0: np.array([[0.9]]), 1: np.array([[0.9]])}
B = {0: np.array([[0.5]]), 1: np.array([[0.5]])}
C = {0: np.array([[1.0]]), 1: np.array([[1.0]])}
D = {0: np.array([[0.0]]), 1: np.array([[0.0]])}
Bw = {0: np.array([[1.0]]), 1: np.array([[1.0]])}
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
    0: MDPModel.from_system(sysLTI[0], nx=1000, nu=5, placement='centers',
                            tol=1e-15, renormalize=True, contract_sum=1.0),
    1: MDPModel.from_system(sysLTI[1], nx=1000, nu=5, placement='centers',
                            tol=1e-15, renormalize=True, contract_sum=1.0),
}

# -----------------------
# Labeling
# -----------------------
L = dim_label(sysAbs, sysLTI, letters, visualize=True)

# -----------------------
# Tree
# -----------------------
# TODO@Ruohan: 1) initialize randomized policy                      DONE
#              2) uniform approximation weighting (for policy op)   DONE
#              3) initialize tree
#              4) begin of tree loop
#               4.a) optimize policy
#               4.b) update node value
#               4.c) grow new leafs (children of previous leafs), assign 0

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
# Visualization of satProb
# -----------------------









