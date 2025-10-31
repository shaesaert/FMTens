# -----------------------
# Load modified modules if needed
# -----------------------
from importlib import reload

from prompt_toolkit.contrib.telnet import TelnetServer

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
from typing import Optional
import psutil, os
from pathlib import Path; import h5py
from src.models.linmodel import LinModel
from src.models.mdpmodel import MDPModel
from src.specifications.translate import translate
from src.specifications.utils.dfa_tool import dfa_manipulation
from src.abstraction.utils.labeling import dim_label
from src.dynprog.dfa_tree_r1 import DFATree
from src.vis.dfa_tree_viz import plot_tree_layered

# -----------------------
# Continuous system (dim = optional dimensions, each 1D)
# -----------------------
dim = 6

# --- System ---
# Initialize system dynamics containers
sysLTI: dict[int, Optional[np.ndarray]] = {i: None for i in range(dim)}
A: dict[int, Optional[np.ndarray]] = {i: None for i in range(dim)}
B: dict[int, Optional[np.ndarray]] = {i: None for i in range(dim)}
C: dict[int, Optional[np.ndarray]] = {i: None for i in range(dim)}
D: dict[int, Optional[np.ndarray]] = {i: None for i in range(dim)}
Bw: dict[int, Optional[np.ndarray]] = {i: None for i in range(dim)}
mu: dict[int, Optional[np.ndarray]] = {i: None for i in range(dim)}
sigma: dict[int, Optional[np.ndarray]] = {i: None for i in range(dim)}

# # --- System ---
# State and input bounds
bx = np.array([10, 10])
bu = np.array([2, 2])

# --- Specification ---
# AP regions (1D)
s = np.array([[1], [-1]])
bp1 = np.array([5, 5])
bp2 = np.array([2, 2])
P1 = pc.Polytope(s, bp1)
P2 = pc.Polytope(s, bp2)

# --- Specification ---
safeset = ' '
targetset = ' '
AP = set()
act = {}

# --- System Initialization ---
print("Initializing system dynamics...")

for i in range(dim):
    # Homogeneous systems identical in this case
    A[i]= np.array([[1]])
    B[i] = np.array([[1]])
    C[i] = np.array([[1]])
    D[i] = np.array([[0]])
    Bw[i] = np.array([[1]])
    mu[i] = np.array([[0]])
    sigma[i] = np.array([[1]])

    sysLTI[i] = LinModel(A[i], B[i], C[i], D[i], Bw[i], mu = mu[i], sigma = sigma[i])
    sysLTI[i].X = pc.Polytope(s, bx)
    sysLTI[i].U = pc.Polytope(s, bu)

    # --- Bind APs to systems ---
    sysLTI[i].AP = [f'p1{i}',f'p2{i}']  #  0-based indexing p10 p11....
    sysLTI[i].regions = [P1,P2]
    act[i] = [' ', f'p1{i}',f'p2{i}']
    safeset = safeset + f' & p1{i}'
    targetset = targetset + f' | p2{i}'
    AP.update(sysLTI[i].AP)

    # Clean up safeset string
    if i == 0:
        safeset = safeset[3:]  # Remove initial ' & '
        targetset = targetset[3:]  # Remove initial ' | '


print(f"Safeset: {safeset}")
print(f"Targetset: {targetset}")
print(f"Atomic propositions: {AP}")

# --- DFA ---
formula = f" ({safeset.strip()}) U ( ({targetset.strip()}) & ({safeset.strip()}) )"
print(f"LTL Formula: {formula}")

# Translate the spec to a DFA
print("Translating specification to DFA...")
DFA = translate(formula)

# Optional manipulation of DFA
DFA, letters = dfa_manipulation(
    DFA,
    index_base = 0,                            # << -- ensure it's 0-based DFA state
    ensure_transitions=True,
    remove_qf_self_loop=True,                  # <- toggle
    verbose=True
)

# -----------------------
# Abstraction
# -----------------------
sysAbs: dict[int, Optional[np.ndarray]] = {i: None for i in range(dim)}
for i in range(dim):
    sysAbs[i] = MDPModel.from_system(sysLTI[i], nx=10, nu=5, placement='centers', u_placement='endpoints',
                            tol=1e-19, contract_sum=None, compute_P = '1d')

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

# 2) Grow tree
T = 6
for it in range(1, T + 1):
    print(f"\n=== Iteration {it} ===")
    G.maxpolicy(rho)
    G.update_tree()
    # G.prune(0.005,'leafs')
    G.grow()
    plot_tree_layered(G)

# Memory usage calculation
mb = sum(np.asarray(G.V[d]).nbytes for d in range(len(G.V))) / (1024**2)
print(f"Total G.V arrays: {mb:.2f} MB")

from src.dynprog.utils.treebasedV import compute_tv_from_tree
tv = compute_tv_from_tree(G, DFA, L, max_elements=50_000_000)

exit = 1