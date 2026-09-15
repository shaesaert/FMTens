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
PRUNE_TOL = 5e-3    # prune nodes whose value < PRUNE_TOL

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
        # (2, 1, "p7  & p11"),
        # (2, 1, "p8  & p12"),
        # (2, 1, "p9  & p13"),
        # (2, 1, "p10 & p14"),
        (2, 1, "p7 & p8 & p9 & p10 & p11 & p12 & p13 & p14"),
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

from src.specifications.utils.dfa_tool import check_letter_disjointness
result = check_letter_disjointness(letters, L_list, DFA=DFA)
if not result["clean"]:
    raise RuntimeError("Labelling overlap detected on the abstraction grid")

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
    # (-7.2875,-7.2875,-7.2875,-7.2875,-7.2875,-7.2875,-7.2875,-7.2875,)
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

# ===========================================================================
# 10.  2D tv over (agent 0, agent 1) with agents 2-7 pinned, exported to .mat
#
# Agents 2,4,6 are held at  1.4125  and agents 3,5,7 at  1.2875; agents 0 and 1
# sweep their full abstract grids. Because only two dimensions vary, the 8D
# point query factorises into matrix products, so the whole (N0 x N1) grid is
# built from a handful of matmuls instead of N0*N1 scalar tv_value_at_point
# calls. A runtime self-check compares random grid cells against the scalar
# tv_value_at_point, so equivalence is verified on the real tree.
# ===========================================================================
from scipy.io import savemat


def tv_grid_2d(G, DFA, L, sysAbs, fixed, *, free=(0, 1),
               index_mode="nearest", tol=1e-9, selfcheck=24, seed=0):
    """
    Satisfaction-probability grid over two free agents with all others pinned.

    Parameters
    ----------
    fixed : dict[int, float]
        Pinned agents -> coordinate value (center coordinates).
    free : (d0, d1)
        The two agents that vary; result axis 0 is d0, axis 1 is d1.

    `free` and `fixed` together must cover every agent exactly once.

    Returns
    -------
    (centers_d0, centers_d1, tv)  with tv of shape (N_d0, N_d1).

    Equivalent to evaluating tv_value_at_point on the full (d0, d1) grid, via
        tv[i0,i1] = sum_q  M_q[i0,i1] * LM_q[i0,i1]
    where, for each DFA destination state q reachable from q0,
        M_q[i0,i1]  = sum_{n in Q[q]} W(n) V[d0][n,i0] V[d1][n,i1]      (node sum)
        LM_q[i0,i1] = sum_{l: q0->q} Cmask[l] L[d0][l,i0] L[d1][l,i1]   (labels)
    and W(n), Cmask[l] collect the fixed-dimension contributions.
    """
    d0, d1 = free
    D = len(sysAbs)
    if set(free) | set(fixed) != set(range(D)):
        raise ValueError("free dims + fixed dims must cover all agents exactly")

    q0    = int(DFA.S0[0])
    trans = np.asarray(DFA.trans, dtype=int)

    centers0 = centers_1d(sysAbs[d0], sysAbs[d0].N)
    centers1 = centers_1d(sysAbs[d1], sysAbs[d1].N)
    N0, N1   = centers0.size, centers1.size

    fixed_idx = {}
    for d, val in fixed.items():
        cd = centers_1d(sysAbs[d], sysAbs[d].N)
        fixed_idx[d] = coord_to_index(cd, float(val), mode=index_mode, tol=tol)[0]
    fixed_dims = sorted(fixed_idx)

    L0 = np.asarray(L[d0], dtype=float)            # (n_letters, N0)
    L1 = np.asarray(L[d1], dtype=float)            # (n_letters, N1)
    Cmask = np.ones(trans.shape[1], dtype=float)   # fixed-dim label contribution
    for d in fixed_dims:
        Cmask *= np.asarray(L[d], dtype=float)[:, fixed_idx[d]]

    V0 = np.asarray(G.V[d0], dtype=float)          # (n_nodes, N0)
    V1 = np.asarray(G.V[d1], dtype=float)          # (n_nodes, N1)

    qdsts = [int(q) for q in np.unique(trans[q0, :]) if len(G.Q.get(int(q), [])) > 0]

    tv = np.zeros((N0, N1), dtype=float)
    for q in qdsts:
        nodes = np.asarray(G.Q[q], dtype=int)
        w = np.ones(nodes.size, dtype=float)       # fixed-dim value contribution / node
        for d in fixed_dims:
            w *= np.asarray(G.V[d], dtype=float)[nodes, fixed_idx[d]]
        M_q = (w[:, None] * V0[nodes, :]).T @ V1[nodes, :]          # (N0, N1)

        lsel = np.where(trans[q0, :] == q)[0]
        if lsel.size == 0:
            continue
        LM = (Cmask[lsel][:, None] * L0[lsel, :]).T @ L1[lsel, :]   # (N0, N1)
        tv += M_q * LM

    # ---- runtime equivalence check against the scalar reference ----
    if selfcheck:
        rng = np.random.default_rng(seed)
        for _ in range(int(selfcheck)):
            i0, i1 = int(rng.integers(N0)), int(rng.integers(N1))
            point = [0.0] * D
            point[d0], point[d1] = float(centers0[i0]), float(centers1[i1])
            for d in fixed_dims:
                point[d] = float(centers_1d(sysAbs[d], sysAbs[d].N)[fixed_idx[d]])
            ref = tv_value_at_point(G, DFA, L, sysAbs, point,
                                    index_mode=index_mode, tol=tol)
            if not np.isclose(ref, tv[i0, i1], rtol=1e-6, atol=1e-9):
                raise AssertionError(
                    f"self-check failed at ({i0},{i1}): "
                    f"vectorised {tv[i0, i1]:.6g} vs reference {ref:.6g}")

    return centers0, centers1, tv


# agents 2,4,6 -> 1.4125 ; agents 3,5,7 -> 1.2875 ; agents 0,1 swept
fixed_states = {2: 1.4125, 4: 1.4125, 6: 1.4125,
                3: 1.2875, 5: 1.2875, 7: 1.2875}

x1, x2, tv2d = tv_grid_2d(G, DFA, L, sysAbs, fixed_states, free=(0, 1))

print(f"\n2D tv grid: {tv2d.shape}  (agent 0 x agent 1)")
print(f"  range: [{tv2d.min():.4f}, {tv2d.max():.4f}]")

# P is (agent0, agent1); x1/x2 are the agent-0/agent-1 axes (centre coords).
savemat("tv_pd8d_a01.mat",
        {"P": tv2d, "x1": x1, "x2": x2, "N": float(tv2d.size)})
print("saved tv_pd8d_a01.mat")

# ===========================================================================
# 11.  2D tv with all 4 robots co-located
#      agents (0,2,4,6) share x ; agents (1,3,5,7) share y
#      exported to .mat
#
# Each (i_x, i_y) cell evaluates sat-prob for the 8D state where every robot
# starts at (x_{i_x}, y_{i_y}). The per-letter L and per-node V products
# factor across the even/odd agent groups:
#     L_even[l, i_x] = prod_{d in {0,2,4,6}} L[d][l, i_x]
#     L_odd [l, i_y] = prod_{d in {1,3,5,7}} L[d][l, i_y]
#     V_even[n, i_x] = prod_{d in {0,2,4,6}} V[d][n, i_x]
#     V_odd [n, i_y] = prod_{d in {1,3,5,7}} V[d][n, i_y]
# so the (N_x x N_y) grid is built from a handful of matmuls, with a
# self-check that random cells match the scalar tv_value_at_point reference.
# ===========================================================================


def tv_grid_shared_xy(G, DFA, L, sysAbs, *,
                      even_dims=(0, 2, 4, 6), odd_dims=(1, 3, 5, 7),
                      index_mode="nearest", tol=1e-9, selfcheck=24, seed=0):
    """
    Satisfaction-probability grid over (x, y) with every robot at (x, y):
        agents in even_dims share the x-coordinate;
        agents in odd_dims  share the y-coordinate.

    Returns (centers_x, centers_y, tv) with tv of shape (N_x, N_y).
    """
    even_dims = tuple(even_dims)
    odd_dims  = tuple(odd_dims)
    D = len(sysAbs)
    if set(even_dims) | set(odd_dims) != set(range(D)):
        raise ValueError("even_dims | odd_dims must cover all agents exactly")
    if set(even_dims) & set(odd_dims):
        raise ValueError("even_dims and odd_dims must be disjoint")

    q0    = int(DFA.S0[0])
    trans = np.asarray(DFA.trans, dtype=int)

    centers_x = centers_1d(sysAbs[even_dims[0]], sysAbs[even_dims[0]].N)
    centers_y = centers_1d(sysAbs[odd_dims[0]],  sysAbs[odd_dims[0]].N)
    N_x, N_y  = centers_x.size, centers_y.size

    # collapsed per-letter labels across each agent group
    L_even = np.ones_like(np.asarray(L[even_dims[0]], dtype=float))
    for d in even_dims:
        L_even *= np.asarray(L[d], dtype=float)
    L_odd  = np.ones_like(np.asarray(L[odd_dims[0]],  dtype=float))
    for d in odd_dims:
        L_odd  *= np.asarray(L[d], dtype=float)

    # collapsed per-node values across each agent group
    V_even = np.ones_like(np.asarray(G.V[even_dims[0]], dtype=float))
    for d in even_dims:
        V_even *= np.asarray(G.V[d], dtype=float)
    V_odd  = np.ones_like(np.asarray(G.V[odd_dims[0]],  dtype=float))
    for d in odd_dims:
        V_odd  *= np.asarray(G.V[d], dtype=float)

    qdsts = [int(q) for q in np.unique(trans[q0, :]) if len(G.Q.get(int(q), [])) > 0]

    tv = np.zeros((N_x, N_y), dtype=float)
    for q in qdsts:
        nodes = np.asarray(G.Q[q], dtype=int)
        M_q = V_even[nodes, :].T @ V_odd[nodes, :]                  # (N_x, N_y)

        lsel = np.where(trans[q0, :] == q)[0]
        if lsel.size == 0:
            continue
        LM = L_even[lsel, :].T @ L_odd[lsel, :]                     # (N_x, N_y)
        tv += M_q * LM

    # ---- runtime equivalence check against the scalar reference ----
    if selfcheck:
        rng = np.random.default_rng(seed)
        for _ in range(int(selfcheck)):
            ix, iy = int(rng.integers(N_x)), int(rng.integers(N_y))
            point = [0.0] * D
            for d in even_dims:
                point[d] = float(centers_x[ix])
            for d in odd_dims:
                point[d] = float(centers_y[iy])
            ref = tv_value_at_point(G, DFA, L, sysAbs, point,
                                    index_mode=index_mode, tol=tol)
            if not np.isclose(ref, tv[ix, iy], rtol=1e-6, atol=1e-9):
                raise AssertionError(
                    f"self-check failed at (x={ix}, y={iy}): "
                    f"vectorised {tv[ix, iy]:.6g} vs reference {ref:.6g}")

    return centers_x, centers_y, tv


x_shared, y_shared, tv_shared = tv_grid_shared_xy(G, DFA, L, sysAbs)

print(f"\n2D tv grid (all 4 robots co-located): {tv_shared.shape}")
print(f"  range: [{tv_shared.min():.4f}, {tv_shared.max():.4f}]")

savemat("tv_pd8d_shared.mat",
        {"P": tv_shared, "x1": x_shared, "x2": y_shared, "N": float(tv_shared.size)})
print("saved tv_pd8d_shared.mat")