# README:
# run this script to re-produce Fig. 4 in paper https://arxiv.org/pdf/2511.06873


import numpy as np
import matplotlib.ticker as mticker

from config.IFAC26_RA_setup import get_ifac26_ra_configs, apply_mpl_style


def apply_curve_plot_style():
    """
    Plot-only style tweaks (sizes/weights) layered on top of IFAC global style.
    Call AFTER apply_mpl_style() and AFTER importing pyplot.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
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


def main():
    # ---- load configs just to reuse backend + TeX style ----
    _sys_cfg, _abs_cfg, run_cfg, _spec_cfg = get_ifac26_ra_configs()

    # must be BEFORE pyplot import
    apply_mpl_style(getattr(run_cfg, "mpl_backend", "QtAgg"))

    import matplotlib.pyplot as plt
    apply_curve_plot_style()

    # ---- load data from the two modules ----
    from Data.IFAC26_2D_RA_AposDC import mean_apos_corr_curve as curve_base
    from Data.IFAC26_2D_RA_AposMTL import mean_apos_corr_curve_mtl as curve_mtl
    from Data.IFAC26_2D_RA_RT import mean_rt_curve as curve_rt

    curve_base = np.asarray(curve_base, dtype=float)
    curve_mtl  = np.asarray(curve_mtl, dtype=float)
    curve_rt = np.asarray(curve_rt, dtype=float)

    T_base = np.arange(len(curve_base))
    T_mtl  = np.arange(len(curve_mtl))
    T_rt = np.arange(len(curve_rt))

    if len(curve_base) != len(curve_mtl):
        print(f"Warning: curve lengths differ ({len(curve_base)} vs {len(curve_mtl)}). "
              "They will be plotted against their own horizons.")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(12, 3))

    lbl_base = r'$\textbf{Optimal A-posteriori corrected Value functions (discounted pol)}$'
    lbl_mtl  = r'$\textbf{Optimal A-posteriori corrected Value functions (M.T.L. pol)}$'
    lbl_rt = r'$\textbf{Optimal Robust-tree Value functions}$'
    ax.plot(T_rt, curve_rt, label=lbl_rt, marker='^')  # <-- add this

    ax.plot(T_base, curve_base, label=lbl_base, marker='s')
    ax.plot(T_mtl,  curve_mtl,  label=lbl_mtl,  marker='o')

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
    ax.margins(x=0.08)

    ax.set_xlabel(r'\textbf{Specification Horizon}')
    for sp in ax.spines.values():
        sp.set_linewidth(2)

    ax.set_title('')
    ax.grid(False)
    fig.tight_layout()

    # ---- save + show ----
    fig.savefig("mean_apos_two_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig("mean_apos_two_curves.eps", format="eps", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()