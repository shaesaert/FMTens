# Tutorials/RA4D.py
"""
2-agent reach-avoid, each agent 2D (position + velocity) — satisfaction
probability through end of tree iteration.

Status
------
RUNNABLE with EPS=0, DELTA=0.

The 2D system abstraction (``compute_P='2d'``) is implemented and verified,
so the abstraction, tree iteration, and visualisation stages all run
end-to-end. The robust (ε, δ)-correction has not yet been incorporated into
value iteration for the 2D case, so this tutorial is restricted to the
non-robust regime: ``EPS = 0.0`` and ``DELTA = 0.0``. Under those settings
the pipeline runs and produces correct satisfaction probabilities. Setting
either to a non-zero value will execute without error but the (ε, δ)
correction terms are silently ignored downstream — the resulting numbers
are not guaranteed satisfaction probabilities.

Stages run:
    1.1  Continuous LTI system + atomic propositions  (2 agents, each 2D)
    1.2  LTL specification compiled to a DFA — no structural override needed:
         the translation is fine, only label re-writes are applied.
    2    Abstraction + labeling                       (2D abstraction via
         compute_P='2d', nx is a per-dimension list; non-robust only)
    3    Tree-based value iteration                   (non-robust)
    5    Velocity-averaged heatmap of the satisfaction probability

The satisfaction-probability tensor (Stage 4 in RA2D) is intentionally skipped:
for a 4D joint state the rank-1 tensor product over all dimensions is not the
natural visualisation. Instead, the per-agent value functions are aggregated
along the velocity axis in Section 9.

For a successful run without visualisation:
  - Comment out Sections 9 and 10;
  - the tree-based value iteration in Stage 3 still runs to completion.

Run from the repo root:
    python -m Tutorials.RA4D

Prerequisite
------------
prepare_pipeline must accept list-valued `nx` and `nu` (see the
`_maybe_int` patch in src/pipeline.py).
"""

from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import polytope as pc

from src.models.linmodel               import LinModel
from src.pipeline                      import (
    build_regions_from_cfg,
    prepare_pipeline, run_tree,
)
from src.specifications.translate      import translate
from src.specifications.utils.dfa_tool import dfa_manipulation


# ===========================================================================
# 1.  Algorithm knobs — tune these
# ===========================================================================
NX        = [100, 100]    # abstract states per dimension per agent  (2D agents)
NU        = 20            # abstract inputs per agent
EPS       = 0.0           # abstraction-labeling correction  (0 = non-robust)
T         = 20            # max tree-iteration depth
PRUNE_TOL = 5e-3          # prune nodes whose value < PRUNE_TOL

MODE      = "rt"          # "rt" or "apos"
DELTA     = 0.0           # no abstraction-error correction in this tutorial

SAMP_T    = 0.5           # sampling time for the discrete-time dynamics


# ===========================================================================
# 2.  System: 2 identical 2D agents (position + velocity)
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

AGENTS = list(range(2))


# ===========================================================================
# 3.  Regions and atomic propositions  (1D — defined on position only)
# ===========================================================================
# AP polytopes constrain the position coordinate only; velocity is
# unconstrained in the labeling. S_1D is the polytope facet matrix for 1D
# intervals, distinct from any 2D system matrix.
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

ap_by_agent = {
    0: ["p1", "p2"],
    1: ["p3"],
}


# ===========================================================================
# 4.  Build sysLTI  (Stage 1.1)
# ===========================================================================
# Built manually here because X (2D box) and U (1D box) have different
# dimensions, so the generic build_system_with_ap (which assumes one shared
# facet matrix S for both) does not apply.
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
# ===========================================================================
# Specification:  ((!p2 | !p3) U p1)
#   Until the goal p1 is reached, at least one of {!p2, !p3} holds.
# Translation is structurally adequate; only label re-writes and an ordered
# letter set are needed. No structural override.
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

from src.specifications.utils.dfa_tool import check_letter_disjointness
result = check_letter_disjointness(letters, L_list, DFA=DFA)
if not result["clean"]:
    raise RuntimeError("Labelling overlap detected on the abstraction grid")
# ===========================================================================
# 7.  Tree-based value iteration  (Stage 3)
# ===========================================================================
# `compute_tv=False` stops at the end of the tree iteration; the rank-1
# satisfaction tensor (Stage 4 in RA2D) is not constructed for 4D joint
# states. The per-agent value functions and DFA tree are kept on `G` and
# aggregated along velocity in Section 9.
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


# ===========================================================================
# 9.  Visualisation — velocity-averaged satisfaction heatmap  (Stage 5)
#
# The figure shows the satisfaction probability of the 2-agent system over
# its joint position plane: agent 0 position along the horizontal axis,
# agent 1 position along the vertical axis. For each (p_0, p_1) cell the
# velocity dimensions are averaged out, so the colour encodes the mean
# probability of satisfying the specification across all initial velocities.
# ===========================================================================
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from src.dynprog.utils.tv_aggregate import aggregate_tv_position


def _remove_titles(fig=None):
    fig = fig or plt.gcf()
    try:
        fig._suptitle = None
    except Exception:
        pass
    for ax in fig.axes:
        try:
            ax.set_title("")
        except Exception:
            pass


def _force_bold_ticklabels_tex(ax, fmt="{x:g}"):
    def _one(v, pos):
        return rf"$\mathbf{{{fmt.format(x=v)}}}$"
    ax.xaxis.set_major_formatter(FuncFormatter(_one))
    ax.yaxis.set_major_formatter(FuncFormatter(_one))


def _style_axes(fig=None):
    fig = fig or plt.gcf()
    for ax in fig.axes:
        ax.set_xlabel(r"$\boldsymbol{x_{1,0}}$")
        ax.set_ylabel(r"$\boldsymbol{x_{2,0}}$")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.tick_params(axis="both", which="both", width=1.6, length=6)
        _force_bold_ticklabels_tex(ax)


tv_avg = aggregate_tv_position(G, Np=NX[0], Nv=NX[1])

fig, ax = plt.subplots()
im = ax.imshow(
    tv_avg.T,
    origin="lower",
    extent=[POSITION_LO, POSITION_HI, POSITION_LO, POSITION_HI],
    aspect="equal",
    cmap="plasma",
    vmin=0.0, vmax=1.0,
)
fig.colorbar(im, ax=ax, label=r"$\overline{P}(\mathrm{spec})$")
_remove_titles(fig)
_style_axes(fig)
plt.show()

