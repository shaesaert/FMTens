# Tutorials/PD8D.py
"""
4-robot-8-dimensional package delivery — satisfaction probability + visualisation.

Status
------
RUNNABLE.

Stages run:
    1.1  Continuous LTI system + atomic propositions  (4 robots × 2D = 8 1D agents)
    1.2  LTL specification compiled to a DFA          (with manipulation over chosen letter set)
    2    Abstraction + labeling under eps
    3    Tree-based value iteration
    4    Per-point satisfaction-probability evaluation at chosen 8D initial states
    5    2D scatter visualisation (regions + per-robot markers annotated with sat-prob)

For a successful run without visualisation:
  - Comment out Section 9 (and the preceding "Section 9 helpers" block),
  the tree-based value iteration in Stage 3 still runs to completion.

Run from the repo root:
    python -m Tutorials.PD8D
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
NX        = 1200      # abstract states per agent
NU        = 10        # abstract inputs per agent
EPS       = 0.1       # abstraction-labeling correction  (set 0 for discrete-state regime)
T         = 20        # max tree-iteration depth
PRUNE_TOL = 5e-3      # prune nodes whose value < PRUNE_TOL

MODE      = "apos"    # "rt" or "apos"   (will collapse to a single `mode` parameter later)
DELTA     = 0.0004    # abstraction error in tree iteration


# ===========================================================================
# 2.  System: 8 identical 1D agents (4 robots × {x, y})
# ===========================================================================
S = np.array([[1.0], [-1.0]])

def interval_to_b(lo: float, hi: float) -> np.ndarray:
    return np.array([hi, -lo], dtype=float)

bx = interval_to_b(-20.0, 5.0)
bu = interval_to_b( -5.0, 5.0)

model = dict(
    A     = np.array([[0.9]]),
    B     = np.array([[0.5]]),
    C     = np.array([[1.0]]),
    D     = np.array([[0.0]]),
    Bw    = np.array([[0.5]]),
    mu    = np.array([0.0]),
    sigma = np.eye(1),
)

AGENTS = list(range(8))
models_by_agent = {k: model for k in AGENTS}
bx_by_agent     = {k: bx    for k in AGENTS}
bu_by_agent     = {k: bu    for k in AGENTS}
s_by_agent      = {k: S     for k in AGENTS}


# ===========================================================================
# 3.  Regions and atomic propositions
# ===========================================================================
region_intervals = {
    "p1":  (-20.0, -15.0),    # delivery-A x-range
    "p2":  (  0.0,   5.0),    # pickup x-range
    "p3":  (-20.0, -14.0),    # delivery-A x-range (looser)
    "p4":  (-14.0,  -8.0),    # delivery-B x-range
    "p5":  ( -8.0,  -2.0),    # delivery-C x-range
    "p6":  ( -2.0,   5.0),    # delivery-D x-range
    "p7":  (-10.0,  -5.0),    # robot A package-loss x-marker
    "p8":  (-10.0,  -5.0),    # robot B
    "p9":  (-10.0,  -5.0),    # robot C
    "p10": (-10.0,  -5.0),    # robot D
    "p11": (-10.0,  -5.0),    # robot A package-loss y-marker
    "p12": (-10.0,  -5.0),    # robot B
    "p13": (-10.0,  -5.0),    # robot C
    "p14": (-10.0,  -5.0),    # robot D
    "p15": ( -5.0,   5.0),    # safe zone 1 (y)
    "p16": (-15.0, -10.0),    # safe zone 3 (y)
}
regions_b = {name: interval_to_b(lo, hi) for name, (lo, hi) in region_intervals.items()}
regions   = build_regions_from_cfg(S, regions_b)

ap_by_agent = {
    0: ["p3",  "p7"],
    1: ["p1",  "p2",  "p11", "p15", "p16"],
    2: ["p4",  "p8"],
    3: ["p1",  "p2",  "p12", "p15", "p16"],
    4: ["p5",  "p9"],
    5: ["p1",  "p2",  "p13", "p15", "p16"],
    6: ["p6",  "p10"],
    7: ["p1",  "p2",  "p14", "p15", "p16"],
}
regions_by_agent = {k: [regions[ap] for ap in ap_by_agent[k]] for k in AGENTS}


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
# ===========================================================================
AP   = list(region_intervals.keys())

spec = (
    "F(p2 & F(p1 & p3 & p4 & p5 & p6))"
    "  & G(p15 | p16 | (p11 & p12 & p13 & p14 & !p7 & !p8 & !p9 & !p10))"
    "  & G((p7  & p11) -> F(p2))"
    "  & G((p8  & p12) -> F(p2))"
    "  & G((p9  & p13) -> F(p2))"
    "  & G((p10 & p14) -> F(p2))"
)

# Option A (Linux/macOS with Spot):
DFA = translate(spec)

# Option B (Windows or any Spot-less environment):
# comment the line above and
# use the lines down:
#
# from src.specifications.utils.dfa_tool import SimpleDFA, EXAMPLE_DFA_DATA
# DFA = SimpleDFA(EXAMPLE_DFA_DATA)

DFA, letters = dfa_manipulation(
    DFA,
    AP          = AP,
    states      = [0, 1, 2],
    initial     = 1,
    accepting   = 0,
    transitions = [
        (0, 0, "1"),
        (1, 1, "!p2"),
        (1, 2, "p2"),
        (2, 2, "p15"),
        (2, 2, "p16"),
        (2, 2, "p11 & p12 & p13 & p14 & !p7 & !p8 & !p9 & !p10"),
        (2, 1, "p7  & p11"),
        (2, 1, "p8  & p12"),
        (2, 1, "p9  & p13"),
        (2, 1, "p10 & p14"),
        (2, 0, "p1 & p3 & p4 & p5 & p6"),
    ],
    index_base          = 0,
    remove_qf_self_loop = True,
)


# ===========================================================================
# 6.  Abstraction + labeling + uniform initial policy/rho  (Stage 2)
# ===========================================================================
abs_cfg = SimpleNamespace(
    nx          = NX,
    nu          = NU,
    placement   = "centers",
    u_placement = "endpoints",
    tol         = 1e-19,
    compute_P   = "1d",
)

sysAbs, L, L_list, pol, rho, nx_list = prepare_pipeline(
    sysLTI  = sysLTI,
    DFA     = DFA,
    letters = letters,
    abs_cfg = abs_cfg,
    eps_val = EPS,
)


# ===========================================================================
# 7.  Tree-based value iteration  (Stage 3)
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
    compute_tv = False,        # the tree alone is enough; tv_value_at_point queries it directly
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
# Section 9 helpers — inlined here for self-containment.
#
# These four functions are generic library code (no PD-specific logic) and
# will be promoted out of this tutorial during the Phase-1 migration:
#
#   centers_1d, coord_to_index           ->  src/abstraction/utils/   (P1-1)
#   tv_value_at_point                    ->  src/dynprog/analysis.py  (P1-2)
#   scatter_point_sets_with_agent_colors ->  src/vis/                 (P1-3)
#
# After that migration, this block is replaced by three import lines.
# ===========================================================================
from typing import Dict, Sequence, Optional, Tuple
from matplotlib.lines import Line2D


def centers_1d(sysAbs_d, N_expected: int) -> np.ndarray:
    """
    Return 1D grid centers from sysAbs_d.hx.
    Assumes sysAbs_d.hx is either an array-like, or a list/tuple whose first
    element is the 1D centers.
    """
    hx = sysAbs_d.hx
    x = np.asarray(hx[0] if isinstance(hx, (list, tuple)) else hx).ravel()
    if x.size != int(N_expected):
        raise ValueError(f"hx length {x.size} != N {N_expected}")
    return x


def coord_to_index(
    centers: np.ndarray, val: float, *, mode: str = "nearest", tol: float = 1e-9,
):
    """
    Map a coordinate (in continuous center coordinates) to an abstract index.

    mode:
      - "nearest": choose nearest center (always succeeds)
      - "exact"  : require |center - val| <= tol
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        raise ValueError("centers is empty")

    idx = int(np.argmin(np.abs(centers - val)))
    err = float(abs(centers[idx] - val))
    if mode == "exact" and err > tol:
        raise ValueError(
            f"value {val} not found (nearest {centers[idx]} @ idx {idx}, err {err})"
        )
    return idx, err, float(centers[idx])


def tv_value_at_point(
    G, DFA, L, sysAbs,
    point: Sequence[float],
    *,
    index_mode: str = "nearest",
    tol: float = 1e-9,
) -> float:
    """
    Compute scalar tv at a single D-dimensional point in center coordinates,
    WITHOUT constructing the full tensor.

    point length must equal the number of dimensions/agents (len(sysAbs)).

    Algorithm:
      tv = sum_l [ prod_d L[d][l, idx_d] ] * [ sum_{n in Q[qdst]} prod_d V[d][n, idx_d] ]
    where qdst = trans[q0, l].
    """
    D = len(point)
    q0 = int(DFA.S0[0])
    trans = np.asarray(DFA.trans, dtype=int)
    n_letters = trans.shape[1]

    # indices per dimension
    idxs = []
    for d in range(D):
        c = centers_1d(sysAbs[d], sysAbs[d].N)
        idx, _, _ = coord_to_index(c, float(point[d]), mode=index_mode, tol=tol)
        idxs.append(idx)

    # reachable destination states from q0
    qdsts = np.unique(trans[q0, :]).astype(int)
    qdsts = [int(q) for q in qdsts if len(G.Q.get(int(q), [])) > 0]

    # sum over nodes at each qdst (at this fixed idx vector)
    node_sum: Dict[int, float] = {}
    for q in qdsts:
        acc = 0.0
        for n in G.Q.get(q, []):
            prod = 1.0
            for d in range(D):
                prod *= float(G.V[d][n, idxs[d]])
                if prod == 0.0:
                    break
            acc += prod
        node_sum[q] = float(acc)

    # accumulate over letters
    tv = 0.0
    for l in range(n_letters):
        qdst = int(trans[q0, l])
        acc = node_sum.get(qdst, None)
        if acc is None:
            continue

        mask = 1.0
        for d in range(D):
            mask *= float(L[d][l, idxs[d]])
            if mask == 0.0:
                break

        tv += mask * acc

    return float(tv)


def scatter_point_sets_with_agent_colors(
    ax,
    point_sets: Sequence[Sequence[float]],
    *,
    dims_pairs: Sequence[Tuple[str, int, int]],
    agent_face: Dict[str, str],
    markers: Sequence[str],
    annotate_agent: str = "A",
    dx: float = 0.2,
    dy: float = 0.2,
    tv_values: Optional[Sequence[float]] = None,
):
    """
    Scatter multiple D-dimensional point sets by projecting each robot/agent
    pair (x_i, y_i):
      - marker shape encodes the point-set index
      - fill colour encodes the agent label

    If tv_values is supplied, annotate the chosen 'annotate_agent' marker of
    each point set with the corresponding tv value.
    """
    for p_idx, p in enumerate(point_sets):
        mk = markers[p_idx % len(markers)]
        tv_val = None if tv_values is None else float(tv_values[p_idx])

        for agent, xi, yi in dims_pairs:
            px, py = float(p[xi]), float(p[yi])
            ax.scatter(
                px, py,
                marker=mk,
                s=90,
                facecolors=agent_face[agent],
                edgecolors="k",
                linewidths=0.8,
                zorder=5,
            )

            if agent == annotate_agent and tv_val is not None:
                ax.text(
                    px + dx, py + dy,
                    f"{tv_val:.3g}",
                    fontsize=25,
                    fontweight="bold",
                    zorder=6,
                )

    # legend for agent colours (independent of marker shape)
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o", linestyle="None",
            markerfacecolor=agent_face[a], markeredgecolor="k",
            markersize=9, label=f"Robot {a}",
        )
        for a in agent_face.keys()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.9,
    )


# ===========================================================================
# 9.  Visualisation — sat-prob at specific 8D initial states  (Stages 4–5)
#
# The joint state is 8D (4 robots × 2 axes), too high-dimensional for a
# heatmap. Instead, we evaluate the satisfaction probability at a handful
# of representative 8D initial-state vectors and plot them on the 2D
# (p_x, p_y) plane: one marker per robot per point set, with the chosen
# annotation agent's marker labelled by the sat-prob value.
# ===========================================================================
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Initial-state point sets to evaluate; each is an 8D vector
#   (px_A, py_A, px_B, py_B, px_C, py_C, px_D, py_D)
point_sets = [
    (-17.8375,  3.2875, -16.8625,  3.2875,   1.4125,  3.2875,   3.3625,  3.2875),
    ( -9.4625, -3.8625,  -6.3375, -3.8625,  -9.4625, -1.5125,  -6.3375, -1.5125),
    (-18.0875, -7.2875, -16.8625, -7.2875, -13.5375, -7.2875, -11.8625, -7.2875),
    (  1.4125,  1.2875,   3.3625,  1.2875, -17.8375,  1.2875, -16.8625,  1.2875),
]

# Regions to overlay on the figure: (label, (x, y) of lower-left, width, height, text_xy)
overlay_regions = [
    ("Pickup",       (-20,   0), 25, 5, (-19.5,   4.5)),
    ("Deliver-A",    (-20, -20),  6, 5, (-19.8, -15.5)),
    ("Deliver-B",    (-14, -20),  6, 5, (-13.8, -15.5)),
    ("Deliver-C",    ( -8, -20),  6, 5, ( -7.8, -15.5)),
    ("Deliver-D",    ( -2, -20),  7, 5, ( -1.8, -15.5)),
    ("Package-loss", (-10, -10),  5, 5, ( -9.8,  -5.2)),
]

# Robot ↔ joint-state dimension pairing
dims_pairs   = [("A", 0, 1), ("B", 2, 3), ("C", 4, 5), ("D", 6, 7)]
agent_colors = {"A": "red", "B": "blue", "C": "yellow", "D": "green"}
markers      = ["o", "s", "^", "D"]      # one marker per point set

# --- evaluate sat-prob at each point set ---
tv_vals = [
    tv_value_at_point(G, DFA, L, sysAbs, p8, index_mode="nearest")
    for p8 in point_sets
]
print("\nPer-point satisfaction probabilities:")
for i, val in enumerate(tv_vals):
    print(f"  point set {i}: sat-prob = {val:.4f}")

# --- figure setup ---
fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
ax.set_xlim(-20, 5)
ax.set_ylim(-20, 5)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$p_x^i$")
ax.set_ylabel(r"$p_y^i$")

# --- overlay regions ---
for label, xy, w, h, text_xy in overlay_regions:
    ax.add_patch(Rectangle(xy, w, h, fill=False,
                           edgecolor="black", linestyle="--", linewidth=1.2))
    ax.text(text_xy[0], text_xy[1], label,
            fontsize=13, fontweight="bold", ha="left", va="center")

# --- scatter point sets (one marker per robot per point set, sat-prob annotated on agent A) ---
scatter_point_sets_with_agent_colors(
    ax,
    point_sets,
    dims_pairs=dims_pairs,
    agent_face=agent_colors,
    markers=markers,
    annotate_agent="A",
    dx=0.55,
    dy=0.55,
    tv_values=tv_vals,
)

plt.tight_layout()
plt.show()