

# -----------------------
# Load modified modules if needed
# -----------------------
from importlib import reload
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
import gc

# -----------------------
# Matplotlib config: match isolate_Gd_drives_G0_copyPxx.py
# (must be BEFORE importing pyplot)
# -----------------------
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{bm}'
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "text.usetex": True,
    "font.size": 14,       # base text size
    "axes.labelsize": 16,  # axis label size
    "legend.fontsize": 14, # legend size
})

mpl.use("QtAgg")  # interactive window

import matplotlib.pyplot as plt

# from pathlib import Path; import h5py
from src.models.linmodel import LinModel
from src.models.mdpmodel import MDPModel
from src.specifications.translate import translate
from src.specifications.utils.dfa_tool import dfa_manipulation
from src.abstraction.utils.labeling import dim_label
from src.abstraction.utils.labeling import dim_label_eps
from src.performance.memory_check import print_workspace_memory
from src.dynprog.dfa_tree_r1 import DFATree
from src.vis.dfa_tree_viz import plot_tree_layered
from src.dynprog.utils.treebasedV import compute_tv_from_tree
from src.vis.plot_tv import plotV_rank1
from src.config.delta_ui import parse_delta_list  # parse single/CSV -> per-dim scalars

# ---------- tiny I/O ----------
def ask(prompt: str, default: str = "") -> str:
    try:
        s = input(prompt + " ").strip()
    except EOFError:
        s = ""
    return s if s else default

def ask_mode(prompt: str, default: str = "rt") -> str:
    while True:
        s = ask(f"{prompt} [rt/apos] (default {default}):", default).lower()
        if s in {"rt", "apos"}:
            return s
        print("Please type 'rt' or 'apos'.")

def make_delta_vectors_from_scalars(scalars, sysAbs) -> list:
    """Create per-dimension vectors from scalars."""
    out = []
    for i, k in enumerate(sorted(sysAbs.keys())):
        N = sysAbs[k].N
        out.append(np.full(N, float(scalars[i]), dtype=float))
    return out

def region_mean_exact(tv: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> float:
    """
    Exact region mean (for debugging/diagnostics at the very end).
    """
    if tv.ndim != 2:
        raise ValueError(f"Expected 2D tv; got shape {tv.shape}")
    nrows, ncols = tv.shape
    rr0 = max(0, min(nrows-1, r0))
    rr1 = max(0, min(nrows-1, r1))
    cc0 = max(0, min(ncols-1, c0))
    cc1 = max(0, min(ncols-1, c1))
    if rr1 < rr0 or cc1 < cc0:
        return 0.0
    sub = tv[rr0:rr1+1, cc0:cc1+1]
    return float(np.mean(sub))


# -----------------------
# Continuous system (2 dimensions, each 1D)
# -----------------------
A = {0: np.array([[0.9]]), 1: np.array([[0.9]])}
B = {0: np.array([[0.5]]), 1: np.array([[0.5]])}
C = {0: np.array([[1.0]]), 1: np.array([[1.0]])}
D = {0: np.array([[0.0]]), 1: np.array([[0.0]])}
Bw = {0: np.array([[0.5]]), 1: np.array([[0.5]])}
mu = {0: np.array([0.0]), 1: np.array([0.0])}
sigma = {0: np.eye(1), 1: np.eye(1)}

sysLTI = {
    0: LinModel(A[0], B[0], C[0], D[0], Bw[0], mu=mu[0], sigma=sigma[0]),
    1: LinModel(A[1], B[1], C[1], D[1], Bw[1], mu=mu[1], sigma=sigma[1]),
}

# -----------------------
# Specification
# -----------------------
DFA = translate('( (!p2 | !p3 ) U p1)')
DFA, letters = dfa_manipulation(
    DFA,
    index_base=0,
    ensure_transitions=True,
    remove_qf_self_loop=True,
    replacements={
        "!p1 & !p2": "p3&!p1&!p2",
        "!p1 & !p3": "!p3&!p1",
    },
    desired_order=['p1', 'p3&!p1&!p2', '!p3&!p1'],
    verbose=True
)

# -----------------------
# APs and regions
# -----------------------
s  = np.array([[1], [-1]])
bx = np.array([5, 20])
bu = np.array([5, 5])

bp1 = np.array([5, 0])      # p1: [0, 5]
bp2 = np.array([0, 5])      # p2: [-5, 0]
bp3 = np.array([-15, 20])   # p3: [-20, -15]

P1 = pc.Polytope(s, bp1)
P2 = pc.Polytope(s, bp2)
P3 = pc.Polytope(s, bp3)

for i in [0, 1]:
    sysLTI[i].X = pc.Polytope(s, bx)
    sysLTI[i].U = pc.Polytope(s, bu)

sysLTI[0].regions = [P1, P2]
sysLTI[1].regions = [P3]
sysLTI[0].AP = ['p1', 'p2']
sysLTI[1].AP = ['p3']

# -----------------------
# Abstraction
# -----------------------
nx_val = int(ask("Enter nx [default 1000]:", "1000"))
nu_val = int(ask("Enter nu [default 10]:", "10"))

sysAbs = {
    0: MDPModel.from_system(sysLTI[0], nx=nx_val, nu=nu_val, placement='centers', u_placement='endpoints',
                            tol=1e-19, contract_sum=None, compute_P='1d'),
    1: MDPModel.from_system(sysLTI[1], nx=nx_val, nu=nu_val, placement='centers', u_placement='endpoints',
                            tol=1e-19, contract_sum=None, compute_P='1d'),
}

# -----------------------
# Labeling
# -----------------------
eps_val = 0.1  # robustness margin
L = dim_label_eps(
    sysAbs,
    sysLTI,
    letters,
    eps=eps_val,
    visualize=False,
    outdir=None,
    prefix="L_eps"
)

# -----------------------
# Policy (uniform) & rho
# -----------------------
dims = sorted(sysAbs.keys())
nx_list = [sysAbs[d].N for d in dims]
nu_list = [sysAbs[d].M for d in dims]
nQ   = len(DFA.S)

pol = [
    [np.full((nx_list[d], nu_list[d]), 1.0/nu_list[d], dtype=float) for d in range(len(dims))]
    for _ in range(nQ)
]
rho = [np.full(nx_list[d], 1.0/nx_list[d], dtype=float) for d in range(len(dims))]
L_list  = [L[k] for k in dims]

# -----------------------
# Ask for modes and deltas
# -----------------------
VI_mode = ask_mode("Choose VI_mode", default="rt")   # 'rt' or 'apos'
deltaVI_in = ask("Enter Δ_VI (single or CSV) [default 0.0084] (If apos VI_mode, input 0):", "0.0084")
deltaVI_scalars = parse_delta_list(deltaVI_in, len(dims), default=0.0084)
delta_VI_vecs = make_delta_vectors_from_scalars(deltaVI_scalars, sysAbs)

pol_mode = ask_mode("Choose pol_mode", default="rt")  # 'rt' or 'apos'
deltaPol_in = ask("Enter Δ_pol (single or CSV) [default 0.0084]:", "0.0084")
deltaPol_scalars = parse_delta_list(deltaPol_in, len(dims), default=0.0084)
delta_pol_vecs = make_delta_vectors_from_scalars(deltaPol_scalars, sysAbs)

# --- NEW: user-defined apos correction delta ---
apos_corr_in = ask("Enter Δ for a-posteriori correction (apos_correction_delta) [default same as Δ_pol]:", "")
if apos_corr_in.strip() == "":
    apos_corr_scalars = deltaPol_scalars
else:
    apos_corr_scalars = parse_delta_list(apos_corr_in, len(dims), default=0.0084)

print(f"\nConfig:")
print(f"  VI_mode={VI_mode}, Δ_VI scalars={deltaVI_scalars}")
print(f"  pol_mode={pol_mode}, Δ_pol scalars={deltaPol_scalars}")
print(f"  apos_correction_delta scalars={apos_corr_scalars}")

# -----------------------
# Tree
# -----------------------
G = DFATree(
    DFA, sysAbs, pol, nx_list, L_list,
    delta_VI=delta_VI_vecs,
    delta_pol=delta_pol_vecs,
    pol_mode=pol_mode,
    VI_mode=VI_mode,
)
G.initiate()

fixed_seed   = 114514      # keep constant across iters for stable stopping
prev_score   = None
final_iter   = 0
score_history = []
idx_bounds = [(0, 798), (599, 699)]

prune_tol = float(ask("Enter prune tol [default 5e-3]:", "0.005"))
K_samp = 2000
T = 6
tol_growth = 1e-8
for it in range(1, T + 1):
    print(f"\n=== Iteration {it} (VI_mode={VI_mode}, pol_mode={pol_mode}) ===")
    G.set_iter(it)
    G.maxpolicy(rho)              # uses pol_mode & delta_pol internally
    G.update_tree()               # uses VI_mode & delta_VI internally
    G.prune(prune_tol, 'leafs')
    G.grow()

    score, gain, stop_now = G.progress_check(
        idx_bounds=idx_bounds,
        prev_score=prev_score,
        K=K_samp,
        tol_growth=tol_growth,
        seed=fixed_seed,
        it=it,
    )

    score_history.append(score)
    final_iter = it
    prev_score = score

    if stop_now:
        break

    # Optional visualize:
    plot_tree_layered(G)

# -----------------------
# Compute tv and optional APoS correction
# -----------------------
tv = compute_tv_from_tree(G, DFA, L, max_elements=50_000_000)

if pol_mode == 'apos':
    from src.dynprog.utils.v_apos import apply_delta_correction_apos
    tv = apply_delta_correction_apos(tv, G, DFA, L, sysAbs,
                                     delta_sys=apos_corr_scalars, T=T)

gc.collect()

print_workspace_memory(globals(), label="end-of-run")

# -----------------------
# visualize satProb (match styling of the comparison script)
# -----------------------
def remove_titles(fig=None):
    """Remove suptitle and all axes titles from the current (or given) figure."""
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
from matplotlib.ticker import FuncFormatter

def force_bold_ticklabels_tex(ax, fmt="{x:g}", bold_minus=False):
    """
    Make major tick labels bold using TeX.
    - fmt: python format string for numbers (e.g. "{x:.2f}")
    - bold_minus: if True, also bolds the minus sign via \\boldsymbol
    """
    def _one(v, pos):
        s = fmt.format(x=v)
        if bold_minus and s.startswith("-"):
            # Bold minus and digits (requires \\usepackage{bm} in your LaTeX preamble)
            return rf"$\boldsymbol{{-{s[1:]}}}$"
        return rf"$\mathbf{{{s}}}$"   # bold digits; minus stays normal weight

    ax.xaxis.set_major_formatter(FuncFormatter(_one))
    ax.yaxis.set_major_formatter(FuncFormatter(_one))


def style_axes_bold_labels_ticks(fig=None):
    """Bold axis labels x_1(0), x_2(0) and tick labels for all axes in the figure."""
    fig = fig or plt.gcf()
    for ax in fig.axes:
        ax.set_xlabel(r'$\boldsymbol{x_{1,0}}$')
        ax.set_ylabel(r'$\boldsymbol{x_{2,0}}$')
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
        ax.tick_params(axis="both", which="both", width=1.6, length=6)
        force_bold_ticklabels_tex(ax)


plotV_rank1(sysAbs, tv)
remove_titles()
style_axes_bold_labels_ticks()

plt.show()

exit = 0

