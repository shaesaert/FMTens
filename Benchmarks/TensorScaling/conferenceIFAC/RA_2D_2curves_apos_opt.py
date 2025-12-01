# plot_two_mean_apos_curves.py

import numpy as np
import matplotlib as mpl

# --- Matplotlib config (must be before pyplot import) ---
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{bm}'
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "text.usetex": True,
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})
try:
    mpl.use("QtAgg")   # interactive if Qt is available
except Exception:
    mpl.use("Agg")     # fallback (no window, but still saves figures)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- Load data from the two .py files ----
from mean_apos_corr_curve_data import mean_apos_corr_curve as curve_base
from mean_apos_corr_curve_data_mtl import mean_apos_corr_curve_mtl as curve_mtl

curve_base = np.asarray(curve_base, dtype=float)
curve_mtl  = np.asarray(curve_mtl, dtype=float)

T_base = np.arange(len(curve_base))
T_mtl  = np.arange(len(curve_mtl))

# Optional: assert same length; comment out if you expect different lengths.
if len(curve_base) != len(curve_mtl):
    print(f"Warning: curve lengths differ "
          f"({len(curve_base)} vs {len(curve_mtl)}). "
          "They will be plotted against their own horizons.")

# ---- Plot ----
fig, ax = plt.subplots(figsize=(12, 3))

lbl_base = r'$\textbf{Optimal A-posteriori corrected Value functions (discounted pol)}$'
lbl_mtl  = r'$\textbf{Optimal A-posteriori corrected Value functions (M.T.L. pol)}$'

ax.plot(T_base, curve_base, label=lbl_base, marker='s')
ax.plot(T_mtl,  curve_mtl,  label=lbl_mtl,  marker='o')

# Legend: bottom-right, one column (same style as your main script)
ax.legend(
    prop={'size': 18, 'weight': 'bold'},
    loc='lower right',
    bbox_to_anchor=(0.98, 0.02),
    bbox_transform=ax.transAxes,
    ncol=1,
    frameon=True, framealpha=0.9,
    borderpad=0.3, labelspacing=0.25,
    handlelength=2.0, handletextpad=0.5, markerscale=0.9
)

ax.xaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
ax.tick_params(axis="both", labelsize=20, length=8, width=2, pad=8)
ax.minorticks_off()
ax.tick_params(axis="both", which="minor", length=5, width=1.6)
ax.margins(x=0.08)

ax.set_xlabel(r'\textbf{Specification Horizon}')
# ax.set_ylabel(r'\textbf{Region mean value}')
for s in ax.spines.values():
    s.set_linewidth(2)
ax.set_title('')
ax.grid(False)
fig.tight_layout()

# Save + show
fig.savefig("mean_apos_two_curves.png", dpi=300, bbox_inches="tight")
fig.savefig("mean_apos_two_curves.eps", format="eps", bbox_inches="tight")

plt.show()
