# Tutorials/RA2D.py
"""
2-agent reach-avoid — satisfaction probability + visualisation.

Status
------
RUNNABLE.

Stages run:
    1.1  Continuous LTI system + atomic propositions (2 agents, each 1D)
    1.2  LTL specification compiled to a DFA — no structural override needed:
         the translation is fine, only label re-writes are applied.
    2    Abstraction + labeling under eps
    3    Tree-based value iteration
    4    Satisfaction-probability tensor
    5    Heatmap visualisation of the satisfaction probability

For a successful run without visualization:
  - Comment out Section 9;
  - `compute_tv=False` in `run_tree(...)` (Section 7) skips Stages 4 and 5
    entirely, since the heatmap needs the satisfaction-probability tensor;
    the tree-based value iteration in Stage 3 still runs to
    completion.

Run from the repo root:
    python -m Tutorials.RA2D
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
NX        = 1000      # abstract states per agent
NU        = 10        # abstract inputs per agent
EPS       = 0.1       # abstraction-labeling correction  (set 0 for discrete-state regime)
T         = 7         # max tree-iteration depth
PRUNE_TOL = 5e-3      # prune nodes whose value < PRUNE_TOL

MODE      = "rt"      # "rt" or "apos"   (will collapse to a single `mode` parameter later)
DELTA     = 0.002     # abstraction error in tree iteration


# ===========================================================================
# 2.  System: 2 identical 1D agents
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

AGENTS = list(range(2))
models_by_agent = {k: model for k in AGENTS}
bx_by_agent     = {k: bx    for k in AGENTS}
bu_by_agent     = {k: bu    for k in AGENTS}
s_by_agent      = {k: S     for k in AGENTS}


# ===========================================================================
# 3.  Regions and atomic propositions
# ===========================================================================
region_intervals = {
    "p1": (  0.0,   5.0),
    "p2": ( -5.0,   0.0),
    "p3": (-20.0, -15.0),
}
regions_b = {name: interval_to_b(lo, hi) for name, (lo, hi) in region_intervals.items()}
regions   = build_regions_from_cfg(S, regions_b)

ap_by_agent = {
    0: ["p1", "p2"],
    1: ["p3"],
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
# 7.  Tree-based value iteration + satisfaction probability  (Stages 3–4)
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

G, tv = run_tree(
    DFA        = DFA,
    sysAbs     = sysAbs,
    sysLTI     = sysLTI,
    L          = L,
    L_list     = L_list,
    pol        = pol,
    rho        = rho,
    nx_list    = nx_list,
    run_cfg    = run_cfg,
    compute_tv = True,          # Stage 4: compute the satisfaction-probability tensor
)


# ===========================================================================
# 8.  Summary
# ===========================================================================
max_depth = max(G.depths.values()) if G.depths else 0
print("Tree iteration complete.")
print(f"  Tree nodes : {G.tree.number_of_nodes()}")
print(f"  Leaves     : {len(G.leafs)}")
print(f"  Max depth  : {max_depth}")
print(f"  tv shape   : {getattr(tv, 'shape', None)}")


# ===========================================================================
# 9.  Visualisation — 2D heatmap of the satisfaction probability  (Stage 5)
#
# The figure shows a heatmap of the satisfaction probabilities of the
# 2-agent system over its 2D joint initial-state space: agent 0 along the
# horizontal axis, agent 1 along the vertical axis. Each cell's colour
# encodes the probability that the specification is satisfied when the
# system starts at that joint initial state.
# ===========================================================================
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from src.vis.plot_tv import plotV_rank1


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


plotV_rank1(sysAbs, tv)
_remove_titles()
_style_axes()
plt.show()