# -----------------------
# Load modified modules if needed
# -----------------------
from importlib import reload

# from prompt_toolkit.contrib.telnet import TelnetServer

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
from src.abstraction.utils.labeling import dim_label_eps
from src.dynprog.dfa_tree_r1 import DFATree
from src.dynprog.utils.treebasedV import compute_tv_from_tree
from src.vis.dfa_tree_viz import plot_tree_layered
from src.vis.dfa_tree_viz import plot_tree_layered_heatnode
import matplotlib.pyplot as plt

def make_delta_vectors_from_scalars(scalars, sysAbs) -> list:
    """Create per-dimension vectors from scalars."""
    out = []
    for i, k in enumerate(sorted(sysAbs.keys())):
        N = sysAbs[k].N
        out.append(np.full(N, float(scalars[i]), dtype=float))
    return out
# -----------------------
# Continuous system (dim = optional dimensions, each 1D)
# -----------------------
dim = 2

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
P1 = pc.Polytope(s, bp1)

# --- Specification ---
safeset = ' '
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
    sysLTI[i].AP = [f'p1{i}']  #  0-based indexing p10 p11....
    sysLTI[i].regions = [P1]
    act[i] = [' ', f'p1{i}']
    safeset = safeset + f' & p1{i}'
    AP.update(sysLTI[i].AP)

    # Clean up safeset stringf
    if i == 0:
        safeset = safeset[3:]  # Remove initial ' & '


print(f"Safeset: {safeset}")
print(f"Atomic propositions: {AP}")

# --- DFA ---
t_spec = 6
formula = ''

ks = range(0, t_spec + 1)
terms = [f"{'X' * k}({safeset})" for k in sorted(ks)]
formula = " & ".join(terms)

print(f"[D={dim}] LTL Formula (t_spec={t_spec}): {formula}")


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
eps_val = 0  # this is your robustness margin

L = dim_label_eps(
    sysAbs,
    sysLTI,
    letters,
    eps=eps_val,
    visualize=True,  # or True if you want plots
    outdir=None,  # or "some/folder" if visualize=True
    prefix="L_eps"
)

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


# -----------------------
# Tree
# -----------------------
# 1) Initialization
delta_VI_vecs = make_delta_vectors_from_scalars([0,0], sysAbs)
delta_pol_vecs = make_delta_vectors_from_scalars([0,0], sysAbs)

G = DFATree(
    DFA, sysAbs, pol, nx_list, L_list,
    delta_VI=delta_VI_vecs,
    delta_pol=delta_pol_vecs,
    pol_mode='rt',
    VI_mode='rt',
)
G.initiate()

# 2) Grow tree
for it in range(1, t_spec + 1):
    print(f"\n=== Iteration {it} ===")
    G.maxpolicy(rho)
    G.update_tree()
    G.prune(0.005,'leafs')
    G.grow()
    plot_tree_layered(G)

# -----------------------
# Compute tv and optional APoS correction
# -----------------------
# G.maxpolicy(rho)              # uses pol_mode & delta_pol internally
# G.update_tree()               # uses VI_mode & delta_VI internally
# G.prune(0.005, 'leafs')
# plot_tree_layered(G)

tv, node_outer_max = compute_tv_from_tree(G, DFA, L, max_elements=50_000_000)

plot_tree_layered_heatnode(G, DFA, node_outer_max=node_outer_max)

# if pol_mode == 'apos':
#     from src.dynprog.utils.v_apos import apply_delta_correction_apos
#     tv = apply_delta_correction_apos(tv, G, DFA, L, sysAbs,
#                                      delta_sys=apos_corr_scalars, T=T)


# -----------------------
# visualize satProb
# -----------------------

plt.show()

exit = 0




