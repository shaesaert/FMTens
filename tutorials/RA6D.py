# Tutorials/RA6D.py
"""
6-agent reach-avoid — satisfaction probability through end of tree iteration.

Status
------
WORK IN PROGRESS — not yet validated. The Python translation pipeline has
not been compared end-to-end against the MATLAB reference; the DFA produced
by translate() + dfa_manipulation() may still differ from the MATLAB DFA.
Treat results as preliminary.

6 identical 1D agents, each driven by:
    x(k+1) = x(k) + u(k) + w(k),   w ~ N(0, 1)
with state bound x ∈ [-10, 10] and input bound u ∈ [-2, 2].

Specification: stay in the safe set (all agents in [-5, 5]) until some agent
reaches the target region [-2, 2] while all agents are still safe.

    safeset   = p10 & p11 & ... & p1<DIM-1>          (all agents in safe region)
    targetset = p20 | p21 | ... | p2<DIM-1>          (some agent in target region)
    spec      = safeset U (targetset & safeset)

Stages run:
    1.1  Continuous LTI system + atomic propositions  (DIM agents × 1D)
    1.2  LTL specification compiled to a DFA          (no structural override needed)
    2    Abstraction + labeling                       (1D abstraction per agent)
    3    Tree-based value iteration

Stops at the end of the iteration. Stage 4 (satisfaction probability) and
Stage 5 (visualisation) are intentionally out of scope.

Run from the repo root:
    python -m Tutorials.RA6D
"""

from __future__ import annotations
from types import SimpleNamespace
import numpy as np

from src.pipeline import (
    build_regions_from_cfg, build_system_with_ap,
    prepare_pipeline, run_tree,
)
from src.specifications.translate      import translate
from src.specifications.utils.dfa_tool import dfa_manipulation


# ===========================================================================
# 1.  Algorithm knobs — tune these
# ===========================================================================
DIM       = 6             # number of agents

NX        = 100            # abstract states per agent
NU        = 5             # abstract inputs per agent
EPS       = 0.0           # abstraction-labeling correction  (0 = non-robust)
T         = 6             # max tree-iteration depth
PRUNE_TOL = 5e-3          # prune nodes whose value < PRUNE_TOL

MODE      = "rt"          # "rt" or "apos"
DELTA     = 0.0           # no abstraction-error correction in this tutorial


# ===========================================================================
# 2.  System: DIM identical 1D agents
#
# Each agent follows a discrete-time random walk with control:
#     x(k+1) = x(k) + u(k) + w(k),   w ~ N(0, 1)
# ===========================================================================
# Polytope encoding:  S @ x <= b   with  S = [[+1], [-1]]   ->   b = [upper, -lower]
S = np.array([[1.0], [-1.0]])

def interval_to_b(lo: float, hi: float) -> np.ndarray:
    return np.array([hi, -lo], dtype=float)

# Per-agent dynamics (identical across all agents)
model = dict(
    A     = np.array([[1.0]]),
    B     = np.array([[1.0]]),
    C     = np.array([[1.0]]),
    D     = np.array([[0.0]]),
    Bw    = np.array([[1.0]]),
    mu    = np.array([0.0]),
    sigma = np.eye(1),
)

# State and input bounds, shared by all agents
STATE_LO, STATE_HI = -10.0, 10.0
INPUT_LO, INPUT_HI =  -2.0,  2.0

bx = interval_to_b(STATE_LO, STATE_HI)
bu = interval_to_b(INPUT_LO, INPUT_HI)

AGENTS = list(range(DIM))
models_by_agent = {k: model for k in AGENTS}
bx_by_agent     = {k: bx    for k in AGENTS}
bu_by_agent     = {k: bu    for k in AGENTS}
s_by_agent      = {k: S     for k in AGENTS}


# ===========================================================================
# 3.  Regions and atomic propositions
#
# All agents share the same safe and target region geometries. Each agent
# gets two distinct APs: "p1<i>" for the safe region and "p2<i>" for the
# target region.
# ===========================================================================
SAFE_LO,   SAFE_HI   = -5.0, 5.0      # safe region   for all agents
TARGET_LO, TARGET_HI = -2.0, 2.0      # target region for all agents

regions_b = {
    "safe":   interval_to_b(SAFE_LO,   SAFE_HI),
    "target": interval_to_b(TARGET_LO, TARGET_HI),
}
regions       = build_regions_from_cfg(S, regions_b)
safe_region   = regions["safe"]
target_region = regions["target"]

# Two APs per agent, aligned with the regions list below.
ap_by_agent      = {k: [f"p1{k}",   f"p2{k}"]        for k in AGENTS}
regions_by_agent = {k: [safe_region, target_region]  for k in AGENTS}


# ===========================================================================
# 4.  Build sysLTI  (Stage 1.1)
# ===========================================================================
sys_cfg = SimpleNamespace(models_by_agent=models_by_agent)

sysLTI = build_system_with_ap(
    sys_cfg          = sys_cfg,
    bx_by_agent      = bx_by_agent,
    bu_by_agent      = bu_by_agent,
    s_by_agent       = s_by_agent,
    regions_by_agent = regions_by_agent,
    ap_by_agent      = ap_by_agent,
)


# ===========================================================================
# 5.  Specification → DFA  (Stage 1.2)
#
# Build the reach-avoid LTL formula programmatically:
#     safeset   = p10 & p11 & ... & p1<DIM-1>
#     targetset = p20 | p21 | ... | p2<DIM-1>
#     spec      = safeset U (targetset & safeset)
#
# "Stay in the safe set until some agent reaches the target while all
# agents are still safe." No structural override needed.
# ===========================================================================
safeset   = " & ".join(f"p1{i}" for i in range(DIM))
targetset = " | ".join(f"p2{i}" for i in range(DIM))
spec      = f"({safeset}) U (({targetset}) & ({safeset}))"

print(f"[D={DIM}] LTL formula:")
print(f"  {spec}")

# Option A (Linux/macOS with Spot):
DFA = translate(spec)

# Option B (Windows or any Spot-less environment): Option B is not allowed for this spec

DFA, letters = dfa_manipulation(
    DFA,
    index_base               = 0,
    remove_qf_self_loop      = True,
    remove_sink_only_letters = True,    # match MATLAB: drop letters that always lead to sink
)


# ===========================================================================
# 6.  Abstraction + labeling + uniform initial policy/rho  (Stage 2)
# ===========================================================================
abs_cfg = SimpleNamespace(
    nx           = NX,
    nu           = NU,
    placement    = "centers",
    u_placement  = "endpoints",
    tol          = 1e-19,
    contract_sum = None,
    compute_P    = "1d",
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
    compute_tv = False,
)


# ===========================================================================
# 8.  Summary
# ===========================================================================
max_depth = max(G.depths.values()) if G.depths else 0
print("Tree iteration complete.")
print(f"  Tree nodes : {G.tree.number_of_nodes()}")
print(f"  Leaves     : {len(G.leafs)}")
print(f"  Max depth  : {max_depth}")