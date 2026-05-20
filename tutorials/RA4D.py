
# Tutorials/RA4D.py
"""
2-agent reach-avoid, each agent 2D (position + velocity) — satisfaction
probability through end of tree iteration.

Status
------
WORK IN PROGRESS — not runnable. This tutorial depends on 2D system abstraction
(`compute_P="2d"` in `src/abstraction/`), which is not yet implemented.
Execution will fail in Stage 2 (prepare_pipeline -> MDPModel.from_system).

Stages run:
    1.1  Continuous LTI system + atomic propositions  (2 agents, each 2D)
    1.2  LTL specification compiled to a DFA          (no structural override
         needed: translation + label re-writes alone suffice)
    2    Abstraction + labeling under eps             (2D abstraction via
         compute_P='2d', nx is a per-dimension list)
    3    Tree-based value iteration

Stops at the end of the iteration. Satisfaction-probability evaluation (Stage 4)
and visualisation (Stage 5) are intentionally out of scope.

Run from the repo root:
    python -m Tutorials.RA4D

Prerequisite:
    prepare_pipeline must accept list-valued `nx` and `nu` (see the
    `_maybe_int` patch in src/pipeline.py).
"""

from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import polytope as pc

from src.models.linmodel import LinModel
from src.pipeline import (
    build_regions_from_cfg,
    prepare_pipeline, run_tree,
)
from src.specifications.translate      import translate
from src.specifications.utils.dfa_tool import dfa_manipulation


# ===========================================================================
# 1.  Algorithm knobs — tune these
# ===========================================================================
NX        = [100, 100]    # abstract states per dimension per agent  (2D agents)
NU        = 10            # abstract inputs per agent
EPS       = 0.0           # abstraction-labeling correction  (0 = non-robust)
T         = 10            # max tree-iteration depth
PRUNE_TOL = 5e-3          # prune nodes whose value < PRUNE_TOL

MODE      = "rt"          # "rt" or "apos"
DELTA     = 0.0           # no abstraction-error correction in this tutorial

SAMP_T    = 0.5           # sampling time for the discrete-time dynamics


# ===========================================================================
# 2.  System dynamics: 2 identical 2D agents (position + velocity)
# ===========================================================================
# Per-agent state-space model:
#     x(k+1) = A x(k) + B u(k) + Bw w(k),   w ~ N(mu, sigma)
# with  x = [position, velocity]^T,  u = [force]  (1D input).
A_mat  = np.array([[1.0, SAMP_T],
                   [0.0, 1.0   ]])
B_mat  = np.array([[0.0   ],
                   [SAMP_T]])
C_mat  = np.array([[1.0, 0.0]])
D_mat  = np.array([[0.0]])
Bw_mat = np.sqrt(0.25) * np.eye(2)     # per-state noise std = 0.5
mu_vec = np.array([[0.0], [0.0]])
sigma  = np.eye(2)

# State and input bounds (shared by both agents)
POSITION_LO, POSITION_HI = -20.0,  5.0
VELOCITY_LO, VELOCITY_HI =  -5.0,  5.0
INPUT_LO,    INPUT_HI    =  -2.0,  2.0


# ===========================================================================
# 3.  Regions and atomic propositions  (1D — defined on the position coordinate)
# ===========================================================================
# AP polytopes are 1D (position only); velocity is unconstrained for labeling.
S_1D = np.array([[1.0], [-1.0]])

def interval_to_b(lo: float, hi: float) -> np.ndarray:
    return np.array([hi, -lo], dtype=float)

region_intervals = {
    "p1": (  0.0,   5.0),     # goal      :  position ∈ [ 0,  5]
    "p2": ( -5.0,   0.0),     # agent 0   :  position ∈ [-5,  0]
    "p3": (-20.0, -15.0),     # agent 1   :  position ∈ [-20,-15]
}
regions_b = {name: interval_to_b(lo, hi) for name, (lo, hi) in region_intervals.items()}
regions   = build_regions_from_cfg(S_1D, regions_b)

# Which APs each agent observes
ap_by_agent = {
    0: ["p1", "p2"],
    1: ["p3"],
}


# ===========================================================================
# 4.  Build sysLTI  (Stage 1.1)
#
# Built manually here because X (2D box) and U (1D box) have different
# dimensions, so the generic build_system_with_ap (which uses one shared
# S matrix for both) does not apply.
# ===========================================================================
AGENTS = list(range(2))
sysLTI = {}
for k in AGENTS:
    sysLTI[k] = LinModel(A_mat, B_mat, C_mat, D_mat, Bw_mat,
                          mu=mu_vec, sigma=sigma)
    sysLTI[k].X = pc.box2poly([[POSITION_LO, POSITION_HI],
                               [VELOCITY_LO, VELOCITY_HI]])
    sysLTI[k].U = pc.box2poly([[INPUT_LO, INPUT_HI]])
    sysLTI[k].regions = [regions[ap] for ap in ap_by_agent[k]]
    sysLTI[k].AP      = list(ap_by_agent[k])


# ===========================================================================
# 5.  Specification → DFA  (Stage 1.2)
#
# Specification:  ((!p2 | !p3) U p1)
#   Until the goal p1 is reached, at least one of {!p2, !p3} holds.
# Translation is structurally adequate; only label re-writes and an ordered
# letter set are needed. No structural override.
# ===========================================================================
spec = "((!p2 | !p3) U p1)"

# Option A (Linux/macOS with Spot):
DFA = translate(spec)

# Option B (Windows or any Spot-less environment): Option B is not allowed for this spec

DFA, letters = dfa_manipulation(
    DFA,
    index_base          = 0,
    remove_qf_self_loop = True,
    replacements = {
        "!p1 & !p2": "p3&!p1&!p2",
        "!p1 & !p3": "!p3&!p1",
    },
    desired_order = ["p1", "p3&!p1&!p2", "!p3&!p1"],
)


# ===========================================================================
# 6.  Abstraction + labeling + uniform initial policy/rho  (Stage 2)
# ===========================================================================
abs_cfg = SimpleNamespace(
    nx           = NX,                # list [nx_position, nx_velocity]
    nu           = NU,
    placement    = "centers",
    u_placement  = "endpoints",
    tol          = 1e-19,
    contract_sum = None,
    compute_P    = "2d",              # 2D agents -> 2D transition computation
)

sysAbs, L, L_list, pol, rho, nx_list = prepare_pipeline(
    sysLTI  = sysLTI,
    DFA     = DFA,
    letters = letters,
    abs_cfg = abs_cfg,
    eps_val = EPS,
)


# ===========================================================================
# 7.  Tree-based value iteration  (Stage 3) — stop at end of iteration
# ===========================================================================
run_cfg = SimpleNamespace(
    T                 = T,
    prune_tol         = PRUNE_TOL,
    VI_mode           = MODE,
    pol_mode          = MODE,
    delta_VI_scalars  = DELTA,
    delta_pol_scalars = DELTA,
    do_progress_check = False,
)

G, _ = run_tree(
    DFA        = DFA,
    sysAbs     = sysAbs,
    sysLTI     = sysLTI,
    L          = L,
    L_list     = L_list,
    pol        = pol,
    rho        = rho,
    nx_list    = nx_list,
    run_cfg    = run_cfg,
    compute_tv = False,        # stop at the end of the tree iteration
)


# ===========================================================================
# 8.  Summary
# ===========================================================================
max_depth = max(G.depths.values()) if G.depths else 0
print("Tree iteration complete.")
print(f"  Tree nodes : {G.tree.number_of_nodes()}")
print(f"  Leaves     : {len(G.leafs)}")
print(f"  Max depth  : {max_depth}")