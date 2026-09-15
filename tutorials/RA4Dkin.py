# Tutorials/RA4Dkin.py
"""
2-agent reach-avoid on the plane — two holonomic vehicles, each with coupled
horizontal/vertical position dynamics.

Per agent
---------
    State  : x = [x_horizontal, x_vertical]^T          (2D, planar position)
    Input  : u = [u_horizontal, u_vertical]^T          (2D, velocity command)
    Dynamics  (linear, stochastic, discrete-time):
        x(k+1) = A x(k) + B u(k) + Bw w(k),    w(k) ~ N(0, I_2)
    where `A` carries a slight damped rotational coupling between the two
    position axes (eigenvalues 0.95 ± 0.10i — a slow-turn kinematic vehicle
    abstraction with a mild pull toward the origin).

Relation to RA4D
----------------
Both tutorials produce a 4D joint state across two agents, but the
per-agent model differs:
  - **RA4D**     : each agent moves on a line.
                   State = (position, velocity), input = force (1D).
                   Second-order dynamics along a single axis.
  - **RA4Dkin**  : each agent moves in the plane.
                   State = (x_position, y_position), input =
                   (x_velocity, y_velocity) (2D).
                   First-order kinematic dynamics with coupling between
                   the two position axes via the rotation block in A.
RA4Dkin therefore has one extra input dimension per agent compared to
RA4D, and trades second-order longitudinal dynamics for first-order
planar dynamics with cross-axis coupling.

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

Stages run:
    1.1  Continuous LTI system + atomic propositions  (2 agents, each 2D
         planar position)
    1.2  LTL specification compiled to a DFA — with label re-writes and an
         ordered letter set to remove the AND-vs-OR overlap on the
         q0 → q0 waiting letters.
    2    Abstraction + labeling                       (2D abstraction via
         compute_P='2d', nx and nu are per-dimension lists; non-robust only)
    3    Tree-based value iteration                   (non-robust)
    4    Joint satisfaction probability with both agents at the same start
         position (diagonal slice of the joint tensor).

For a successful run without visualisation:
  - Comment out Section 9;
  - the tree-based value iteration in Stage 3 still runs to completion.

Run from the repo root:
    python -m Tutorials.RA4Dkin

Prerequisite
------------
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
NX        = [100, 100]    # state grid: bins per state axis  (horizontal, vertical)
NU        = [10, 10]      # input grid: bins per input axis  (horizontal, vertical)
EPS       = 0.0           # abstraction-labeling correction  (0 = non-robust)
T         = 11           # max tree-iteration depth
PRUNE_TOL = 5e-3          # prune nodes whose value < PRUNE_TOL

MODE      = "rt"          # "rt" or "apos"
DELTA     = 0.0           # no abstraction-error correction in this tutorial

SAMP_T    = 0.5           # sampling time for the discrete-time dynamics
SIGMA_W   = 0.5           # per-axis process-noise standard deviation


# ===========================================================================
# 2.  System dynamics: 2 identical planar agents (2D state, 2D input,
#     coupled position axes)
# ===========================================================================
# Per-agent state-space model:
#     x(k+1) = A x(k) + B u(k) + Bw w(k),    w ~ N(mu, sigma)
# with  x = [x_horizontal, x_vertical]^T,    u = [u_horizontal, u_vertical]^T.
#
# A introduces damped rotational coupling between the two position axes
# (eigenvalues 0.95 ± 0.10i — stable spiral toward the origin without input).
A_mat  = np.array([[ 0.95,  0.10],
                   [-0.10,  0.95]])
B_mat  = SAMP_T  * np.eye(2)
C_mat  = np.eye(2)
D_mat  = np.zeros((2, 2))
Bw_mat = SIGMA_W * np.eye(2)         # per-axis noise std = SIGMA_W
mu_vec = np.zeros((2, 1))
sigma  = np.eye(2)

# State and input bounds (shared by both agents)
HORIZONTAL_LO, HORIZONTAL_HI = -10.0, 10.0
VERTICAL_LO,   VERTICAL_HI   = -10.0, 10.0
UH_LO,         UH_HI         =  -2.0,  2.0      # horizontal input bound
UV_LO,         UV_HI         =  -2.0,  2.0      # vertical input bound


# ===========================================================================
# 3.  Regions and atomic propositions  (2D — matched to the planar state)
#
# Disjoint AP partition: agent 0 carries (p1, p2), agent 1 carries (p3, p4).
# Each agent has its own goal and its own obstacle, so no AP is shared
# between agents — this keeps the per-agent labeling decomposition exact
# (no AND-vs-OR mismatch on negated shared literals).
# ===========================================================================
# 2D box constraint matrix:  S_2D x <= b  encodes  lo_h <= x_h <= hi_h
#                                                  lo_v <= x_v <= hi_v.
S_2D = np.array([[ 1.0,  0.0],   # x_h <= hi_h
                 [-1.0,  0.0],   # -x_h <= -lo_h
                 [ 0.0,  1.0],   # x_v <= hi_v
                 [ 0.0, -1.0]])  # -x_v <= -lo_v


def box_to_b(lo_h: float, hi_h: float, lo_v: float, hi_v: float) -> np.ndarray:
    return np.array([hi_h, -lo_h, hi_v, -lo_v], dtype=float)


region_boxes = {
    "p1": ( 3.0,  5.0,    3.0,  5.0),     # agent 0 goal      (upper-right)
    "p2": (-2.0,  2.0,   -2.0,  2.0),     # agent 0 obstacle  (centre)
    "p3": (-5.0, -3.0,   -5.0, -3.0),     # agent 1 goal      (lower-left)
    "p4": ( 0.0,  2.0,    5.0,  7.0),     # agent 1 obstacle  (upper-mid)
}
regions_b = {name: box_to_b(*box) for name, box in region_boxes.items()}
regions   = build_regions_from_cfg(S_2D, regions_b)

ap_by_agent = {
    0: ["p1", "p2"],         # agent 0: goal, obstacle
    1: ["p3", "p4"],         # agent 1: goal, obstacle — disjoint from agent 0
}

# ===========================================================================
# 4.  Build sysLTI  (Stage 1.1)
#
# Identical dynamics for both agents; only the assigned APs/regions differ.
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
# 5.  Specification → DFA  (Stage 1.2)
#
# Spec: (!p1 & !p2 & !p3 & !p4) U (p1 & p3)
#
# Strict semantics: until both agents are simultaneously at their goals
# (p1 ∧ p3), every agent must be inside NO labeled region whatsoever —
# not its own goal, not its own obstacle, not the other agent's. Any
# partial goal (p1 alone, or p3 alone) and any obstacle contact (p2 or
# p4) immediately violates.
#
# Target DFA structure:
#   q0 → qF  on  "p1 & p3"
#   q0 → q0  on  "!p1 & !p2 & !p3 & !p4"
#   (any other joint label) → no matching letter → trace rejected
#
# Spot may emit additional letters for the implicit-sink transitions;
# `desired_order` keeps only the two we want, dropping the rest.
# Run once without `desired_order` first if Spot's letter strings
# differ in spacing/parenthesisation, then adjust the strings to match.
# ===========================================================================
spec = "((!p2 & !p4) U (p1 & p3))"
DFA = translate(spec)

DFA, letters = dfa_manipulation(
    DFA,
    index_base          = 0,
    remove_qf_self_loop = True,
    replacements = {
        "!p2 & !p3 & !p4": "p1 & !p3 & !p2 & !p4",
    },
    desired_order = [
        "p1 & p3",
        "!p1 & !p2 & !p4",
        "p1 & !p3 & !p2 & !p4",
    ],
)
print(f"[DFA letters] {letters}")


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

from src.specifications.utils.dfa_tool import check_letter_disjointness
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

# ===========================================================================
# 9.  Visualisation: joint satisfaction probability assuming both agents
#      are initialised at the same position.
#
# Extracts the diagonal slice tv(k, k) of the joint satisfaction tensor.
# Every cell on the plot answers: "if I place agent 0 AND agent 1 at this
# position simultaneously, what's the probability they jointly satisfy
# the reach-avoid spec?"
# ===========================================================================
from src.dynprog.utils.tv_aggregate import tv_joint_same_state
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def overlay_regions(ax):
    """Solid black box outlines, descriptive text centered inside each."""
    for name, (lo_h, hi_h, lo_v, hi_v) in region_boxes.items():
        ax.add_patch(Rectangle(
            (lo_h, lo_v), hi_h - lo_h, hi_v - lo_v,
            fill=False, edgecolor="black", linewidth=1.5,
        ))
        ax.text(
            (lo_h + hi_h) / 2, (lo_v + hi_v) / 2, REGION_LABELS[name],
            ha="center", va="center",
            fontsize=7, color="black",
        )


tv_diag = tv_joint_same_state(G, agent_shape=(NX[0], NX[1]))

REGION_LABELS = {
    "p1": "goal of\nagent 0",
    "p2": "obstacle of\nagent 0",
    "p3": "goal of\nagent 1",
    "p4": "obstacle of\nagent 1",
}

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
overlay_regions(ax)
ax.set_xlabel("shared horizontal position")
ax.set_ylabel("shared vertical position")
ax.set_title("Joint satisfaction probability with both agents starting at the same position")
plt.show()

# ===========================================================================
# 10.  Export tv_diag to .mat for MATLAB
# ===========================================================================
from scipy.io import savemat

# 2D abstraction centers per axis (sysAbs[0].hx is a 2-tuple for 2D agents)
hx = sysAbs[0].hx
x1 = np.asarray(hx[0]).ravel()   # horizontal centers, length NX[0]
x2 = np.asarray(hx[1]).ravel()   # vertical centers, length NX[1]

savemat("tv_ra4dkin_diag.mat",
        {"P": tv_diag, "x1": x1, "x2": x2, "N": float(tv_diag.size)})
print("saved tv_ra4dkin_diag.mat")




# ===========================================================================
# 11.  Visualisation of G.tree — hierarchical DFA tree after VI
# ===========================================================================
import networkx as nx
from collections import defaultdict
from matplotlib.patches import Patch
from matplotlib.lines  import Line2D

SHOW_EDGE_LETTERS = False  # set True to overlay letter strings on edges

# node -> DFA state (G.Q maps q -> [nodes], invert it)
node_to_q = {n: int(q) for q, nodes in G.Q.items() for n in nodes}

# group nodes by depth for a row-per-depth layout
depth_to_nodes = defaultdict(list)
for n, d in G.depths.items():
    depth_to_nodes[int(d)].append(int(n))

pos = {}
max_depth = max(depth_to_nodes) if depth_to_nodes else 0
for d, nodes in depth_to_nodes.items():
    nodes_sorted = sorted(nodes)
    for i, n in enumerate(nodes_sorted):
        pos[n] = ((i + 0.5) / max(len(nodes_sorted), 1), -d)

# DFA-state display labels + node styling.
# dfa_manipulation with index_base=0 and remove_qf_self_loop=True puts the
# accepting state at index 0 and the initial state at index 1.
q_label = {0: r"$q_f$", 1: r"$q_0$"}
q_color = {0: "none",   1: "grey"}      # node face colour
q_edge  = {0: "black",  1: "grey"}      # node border colour

node_colors = [q_color.get(node_to_q.get(n, -1), "lightgray")
               for n in G.tree.nodes()]
edgecolors  = [q_edge.get(node_to_q.get(n, -1), "white")
               for n in G.tree.nodes()]
linewidths  = 0.6

# --- edges coloured by letter (canonical-form match, robust to whitespace
# /parens/literal order in whatever string dfa_manipulation emits) ---------
def _canon(s):
    s = s.replace("(", "").replace(")", "")
    return frozenset(p.strip() for p in s.split("&"))

letter_color_by_canon = {
    _canon("p1 & p3"):              "black",
    _canon("p1 & !p3 & !p2 & !p4"): "tab:green",
    _canon("!p1 & !p2 & !p4"):      "tab:red",
}

edge_letter_idx = {(u, v): int(data["l"])
                   for u, v, data in G.tree.edges(data=True) if "l" in data}
used_letters    = sorted(set(edge_letter_idx.values()))

letter_color = {l_idx: letter_color_by_canon.get(_canon(letters[l_idx]),
                                                 "tab:gray")
                for l_idx in used_letters}

edges_by_letter = defaultdict(list)
for (u, v), l_idx in edge_letter_idx.items():
    edges_by_letter[l_idx].append((u, v))

# scale figure with tree shape
max_breadth = max((len(v) for v in depth_to_nodes.values()), default=1)
fig_w = max(8, min(24, max_breadth * 0.25))
fig_h = max(5, min(18, (max_depth + 1) * 0.7))

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# Edges first so node markers render on top of arrowheads
for l_idx, edge_list in edges_by_letter.items():
    nx.draw_networkx_edges(
        G.tree, pos, edgelist=edge_list, ax=ax,
        edge_color=letter_color[l_idx], width=1.2, alpha=0.9,
        arrows=True, arrowstyle="-|>", arrowsize=10,
    )

nx.draw_networkx_nodes(G.tree, pos, ax=ax,
                       node_color=node_colors, node_size=180,
                       edgecolors=edgecolors, linewidths=linewidths)
nx.draw_networkx_labels(G.tree, pos, ax=ax,
                        font_size=6, font_color="black")

if SHOW_EDGE_LETTERS:
    edge_labels = {(u, v): letters[l_idx]
                   for (u, v), l_idx in edge_letter_idx.items()}
    nx.draw_networkx_edge_labels(
        G.tree, pos, ax=ax, edge_labels=edge_labels,
        font_size=5, label_pos=0.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5),
    )

# Legend: DFA states + per-letter arrow colour
present_qs = {node_to_q.get(n, -1) for n in G.tree.nodes()}
legend_handles  = [Patch(facecolor=q_color[q], edgecolor=q_edge[q],
                         label=q_label[q])
                   for q in sorted(q_label) if q in present_qs]
legend_handles += [Line2D([0], [0], color=letter_color[l_idx], lw=2.2,
                          label=letters[l_idx])
                   for l_idx in used_letters]

ax.legend(handles=legend_handles, loc="upper right",
          bbox_to_anchor=(1.0, 1.0), fontsize=8, framealpha=0.95)

ax.set_axis_off()
plt.tight_layout()
plt.savefig("G_tree.png", transparent=True, dpi=200, bbox_inches="tight")
plt.show()