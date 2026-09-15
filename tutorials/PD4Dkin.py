# Tutorials/PD4Dkin.py
"""
2-agent synchronised reach-avoid on the plane — two holonomic vehicles
with planar kinematic dynamics over a 20x20 workspace divided into three
horizontal regions:

    top strip      [-10, 10] x [ 5, 10]    tracked as p1 (agent 0), p4 (agent 1)
    bot-left      [-10, -7] x [-10, -7]    tracked as p2 (agent 0) only
    bot-right     [  7, 10] x [-10, -7]    tracked as p5 (agent 1) only
    centre        [ -2,  2] x [ -2,  2]    tracked as p3 (agent 0), p6 (agent 1)

Each physical region carries one AP per agent (six AP names total) so the
per-agent AP sets stay disjoint and the per-agent labelling decomposition
remains exact (no AND-vs-OR mismatch on shared negated literals).

Per agent
---------
    State  : x = [x_horizontal, x_vertical]^T          (2D, planar position)
    Input  : u = [u_horizontal, u_vertical]^T          (2D, velocity command)
    Dynamics  (linear, stochastic, discrete-time):
        x(k+1) = A x(k) + B u(k) + Bw w(k),    w(k) ~ N(0, I_2)
    Identical to RA4Dkin's per-agent dynamics, with damped rotational
    coupling between the two position axes (eigenvalues 0.95 ± 0.10i — a
    slow-turn kinematic vehicle abstraction).

Joint state across 2 agents
---------------------------
Joint state dimension is 2 × 2 = 4.

DFA structure
-------------
Three live states plus an implicit sink. Numbering follows the
dfa_manipulation convention: accepting state is state 0, initial state
is state 1.

    state 1 = q0 (initial — before sync at top)
    state 2 = q1 (after sync at top, waiting for sync at bot)
    state 0 = qf (accepting — both agents at their bot corners)
    state 3 = sink (any unlisted joint label)

    q0 → q0   on  !p1 & !p4                 (neither agent at top)
    q0 → q1   on   p1 & p4                  (synchronise at top)
    q1 → q1   on  !p2 & !p3 & !p5 & !p6     (neither at centre or bot corners)
    q1 → q0   on   p3 & p6                  (joint centre → RESET)
    q1 → qf   on   p2 & p5                  (joint bot corners → ACCEPT)

Status
------
RUNNABLE with EPS=0, DELTA=0.

The 2D system abstraction (``compute_P='2d'``) is implemented and verified,
so abstraction, tree iteration, and visualisation all run end-to-end. The
robust (ε, δ)-correction has not yet been incorporated into value iteration
for the 2D case, so this tutorial is restricted to the non-robust regime:
``EPS = 0.0`` and ``DELTA = 0.0``. Setting either to a non-zero value will
execute without error but the (ε, δ) correction terms are silently ignored
downstream — the resulting numbers are not guaranteed satisfaction
probabilities.

Run from the repo root:
    python -m Tutorials.PD4Dkin

Prerequisite
------------
prepare_pipeline must accept list-valued `nx` and `nu`.
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
from src.specifications.utils.dfa_tool import (
    dfa_manipulation,
    check_letter_disjointness,
)


# ===========================================================================
# 1.  Algorithm knobs
# ===========================================================================
NX        = [50, 50]
NU        = [5, 5]
EPS       = 0.0
T         = 21
# PRUNE_TOL = 5e-3
PRUNE_TOL = 0.05


MODE      = "rt"
DELTA     = 0.0

SAMP_T    = 0.5
SIGMA_W   = 0.5

assert EPS == 0.0 and DELTA == 0.0, (
    "Robust (eps, delta) value iteration is not yet implemented for 2D "
    "agents; this tutorial requires EPS = 0 and DELTA = 0."
)


# ===========================================================================
# 2.  Per-agent dynamics (damped-rotation kinematics, identical to RA4Dkin)
# ===========================================================================
A_mat  = np.array([[ 0.95,  0.10],
                   [-0.10,  0.95]])
B_mat  = SAMP_T  * np.eye(2)
C_mat  = np.eye(2)
D_mat  = np.zeros((2, 2))
Bw_mat = SIGMA_W * np.eye(2)
mu_vec = np.zeros((2, 1))
sigma  = np.eye(2)

HORIZONTAL_LO, HORIZONTAL_HI = -10.0, 10.0
VERTICAL_LO,   VERTICAL_HI   = -10.0, 10.0
UH_LO,         UH_HI         =  -2.0,  2.0
UV_LO,         UV_HI         =  -2.0,  2.0


# ===========================================================================
# 3.  Four physical regions, six AP names (disjoint per-agent)
# ===========================================================================
S_2D = np.array([[ 1.0,  0.0],
                 [-1.0,  0.0],
                 [ 0.0,  1.0],
                 [ 0.0, -1.0]])


def box_to_b(lo_h: float, hi_h: float, lo_v: float, hi_v: float) -> np.ndarray:
    return np.array([hi_h, -lo_h, hi_v, -lo_v], dtype=float)


region_boxes = {
    "p1": ( 5.0,10.0,   -10.0,-5.0),    # PICKUP     (agent 0)
    "p4": ( 5.0,10.0,   -10.0,-5.0),    # PICKUP     (agent 1, same box)
    "p2": (-10.0, -7.0,  7.0, 10.0),     # DELIVER   (agent 0)
    "p5": ( -10.0, -7.0,  -10.0, -7.0),  # DELIVER    (agent 1)
    "p3": ( -2.0,  2.0,   -2.0,  2.0),  # LOSS       (agent 0)
    "p6": ( -2.0,  2.0,   -2.0,  2.0),  # LOSS       (agent 1, same box)
}
regions_b = {name: box_to_b(*box) for name, box in region_boxes.items()}
regions   = build_regions_from_cfg(S_2D, regions_b)

ap_by_agent = {
    0: ["p1", "p2", "p3"],
    1: ["p4", "p5", "p6"],
}


# ===========================================================================
# 4.  Build sysLTI for 2 agents
# ===========================================================================
AGENTS = list(range(2))
sysLTI = {}
for k in AGENTS:
    sysLTI[k] = LinModel(A_mat, B_mat, C_mat, D_mat, Bw_mat,
                         mu=mu_vec, sigma=sigma)
    sysLTI[k].X = pc.box2poly([[HORIZONTAL_LO, HORIZONTAL_HI],
                               [VERTICAL_LO,   VERTICAL_HI  ]])
    sysLTI[k].U = pc.box2poly([[UH_LO, UH_HI],
                               [UV_LO, UV_HI]])
    sysLTI[k].regions = [regions[ap] for ap in ap_by_agent[k]]
    sysLTI[k].AP      = list(ap_by_agent[k])


# ===========================================================================
# 5.  DFA — async pickup, sync delivery, per-agent loss reset
#
# Specification (informally, per agent):
#     reach pickup -> may visit centre (resets pickup status) ->
#     reach delivery; both agents must be at delivery simultaneously.
#
# Atomic propositions (per-agent, disjoint regions):
#     a0:   p1 = pickup,  p2 = delivery,  p3 = loss
#     a1:   p4 = pickup,  p5 = delivery,  p6 = loss
#
# Per-agent progress s_i  in  {init, picked}; "delivered" exists only
# jointly via the sync delivery transition into qf. The joint DFA state
# encodes (s_0, s_1):
#
#     state 0 = qf                     accepting (both at delivery this step)
#     state 1 = (init,   init)         initial
#     state 2 = (picked, init)         a0 picked, a1 not
#     state 3 = (init,   picked)       a1 picked, a0 not
#     state 4 = (picked, picked)       both picked, neither delivered
#
# Transition semantics:
#   - Pickup is async: p1 advances a0 from init to picked independently
#     of p4 (and vice versa). The (1,4) "p1 & p4" letter is a same-step
#     joint pickup, not a sync requirement.
#   - Delivery is sync: only "p2 & p5" from state 4 reaches qf. Single-
#     agent deliveries (p2 alone or p5 alone) are absorbed as state-4
#     self-loops since the agent's individual delivered status is not
#     tracked separately.
#   - Loss reset is per-agent: any picked agent visiting its centre
#     region (p3 for a0, p6 for a1) drops back to init on that agent
#     only. Joint centre visits (p3 & p6) reset both.
#   - Region disjointness within an agent (p1/p2/p3 are mutually
#     exclusive on a0's grid, p4/p5/p6 on a1's) means letters like
#     "p2 & p3" never fire; they're omitted from the transition list.
#
# Letter design: each source state's outgoing letters form a partition
# of the relevant AP subset's truth table:
#     state 1 -> over (p1, p4)
#     state 2 -> over (p3, p4)
#     state 3 -> over (p1, p6)
#     state 4 -> over (p2, p3, p5, p6), excluding intra-agent impossibles
#
# DFA size: 5 states, 21 live transitions + 1 qf self-loop (removed).
# Cost relative to the 3-state sync baseline is ~4x in tree branching;
# the async pickup is what introduces states 2 and 3 and accounts for
# most of the growth.
# ===========================================================================

AP = list(region_boxes.keys())

spec = "..."                       # overridden by transitions below
DFA  = translate(spec)

DFA, letters = dfa_manipulation(
    DFA,
    AP          = AP,
    states      = [0, 1, 2, 3, 4],
    initial     = 1,
    accepting   = 0,
    transitions = [
        (0, 0, "1"),                                # qf self-loop (removed)

        # state 1 = (init, init): only pickups matter
        (1, 1, "!p1 & !p4"),
        (1, 2, "p1 & !p4"),
        (1, 3, "!p1 & p4"),
        (1, 4, "p1 & p4"),

        # state 2 = (picked, init): a0 resets on p3
        (2, 2, "!p3 & !p4"),
        (2, 1, "p3 & !p4"),                         # a0 alone resets
        (2, 4, "!p3 & p4"),                         # a1 picks
        (2, 3, "p3 & p4"),                          # a0 resets, a1 picks

        # state 3 = (init, picked): a1 resets on p6
        (3, 3, "!p1 & !p6"),
        (3, 4, "p1 & !p6"),                         # a0 picks
        (3, 1, "!p1 & p6"),                         # a1 alone resets
        (3, 2, "p1 & p6"),                          # a0 picks, a1 resets

        # state 4 = (picked, picked): everything can happen
        (4, 4, "!p2 & !p3 & !p5 & !p6"),            # nothing
        (4, 4, "p2 & !p3 & !p5 & !p6"),             # a0 at deliver, waits
        (4, 3, "!p2 & p3 & !p5 & !p6"),             # a0 resets
        (4, 4, "!p2 & !p3 & p5 & !p6"),             # a1 at deliver, waits
        (4, 2, "!p2 & !p3 & !p5 & p6"),             # a1 resets
        (4, 0, "p2 & !p3 & p5 & !p6"),              # BOTH deliver — ACCEPT
        (4, 2, "p2 & !p3 & !p5 & p6"),              # a0 at deliver, a1 resets
        (4, 3, "!p2 & p3 & p5 & !p6"),              # a0 resets, a1 at deliver
        (4, 1, "!p2 & p3 & !p5 & p6"),              # both reset
    ],
    index_base          = 0,
    remove_qf_self_loop = True,
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
    compute_P    = "2d",
)

sysAbs, L, L_list, pol, rho, nx_list = prepare_pipeline(
    sysLTI  = sysLTI,
    DFA     = DFA,
    letters = letters,
    abs_cfg = abs_cfg,
    eps_val = EPS,
)

result = check_letter_disjointness(letters, L_list, DFA=DFA)
if not result["clean"]:
    raise RuntimeError("Labelling overlap detected on the abstraction grid")


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
print(f"  Agents     : {G.dim}")
print(f"  Tree nodes : {G.tree.number_of_nodes()}")
print(f"  Leaves     : {len(G.leafs)}")
print(f"  Max depth  : {max_depth}")


# ===========================================================================
# 9.  Visualisation: joint satisfaction probability when both agents
#      start at the same position.
# ===========================================================================
from src.dynprog.utils.tv_aggregate import tv_joint_same_state
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


REGION_LABELS = {
    "p1": "p1 (PICKUP, a0)",       "p4": "p4 (PICKUP, a1)",
    "p2": "p2 (DELIVER, a0)",      "p5": "p5 (DELIVER, a1)",
    "p3": "p3 (LOSS, a0)",         "p6": "p6 (LOSS, a1)",
}


def overlay_regions(ax, names):
    """Solid black box outlines with per-agent labels; shared boxes drawn once."""
    drawn_boxes = set()
    for name in names:
        box = region_boxes[name]
        if box not in drawn_boxes:
            lo_h, hi_h, lo_v, hi_v = box
            ax.add_patch(Rectangle(
                (lo_h, lo_v), hi_h - lo_h, hi_v - lo_v,
                fill=False, edgecolor="black", linewidth=1.5,
            ))
            drawn_boxes.add(box)
        lo_h, hi_h, lo_v, hi_v = box
        y_offset = -0.4 if name in ("p6",) else 0.4 if name in ("p3",) else 0.0
        ax.text(
            (lo_h + hi_h) / 2, (lo_v + hi_v) / 2 + y_offset, REGION_LABELS[name],
            ha="center", va="center", fontsize=8, color="black",
        )


tv_diag = tv_joint_same_state(G, agent_shape=(NX[0], NX[1]))

fig, ax = plt.subplots()
im = ax.imshow(
    tv_diag.T,
    origin="lower",
    extent=[HORIZONTAL_LO, HORIZONTAL_HI, VERTICAL_LO, VERTICAL_HI],
    aspect="equal",
    cmap="Oranges",
    vmin=0.0, vmax=1.0,
)
plt.colorbar(im, ax=ax, label=r"$P(\mathrm{spec} \;|\; x_0 = x_1)$")
overlay_regions(ax, ["p1", "p2", "p3", "p4", "p5", "p6"])
ax.set_xlabel("shared horizontal position")
ax.set_ylabel("shared vertical position")
ax.set_title("Joint satisfaction probability, both agents at the same start")
plt.show()