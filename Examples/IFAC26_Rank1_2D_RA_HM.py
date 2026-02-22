# README:
# run this script to re-produce Fig. 3 in paper https://arxiv.org/pdf/2511.06873

from __future__ import annotations

import gc
from matplotlib.ticker import FuncFormatter

from config.IFAC26_RA_setup import get_ifac26_ra_configs, apply_mpl_style


def remove_titles(fig=None):
    import matplotlib.pyplot as plt
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


def force_bold_ticklabels_tex(ax, fmt="{x:g}", bold_minus=False):
    def _one(v, pos):
        s = fmt.format(x=v)
        if bold_minus and s.startswith("-"):
            return rf"$\boldsymbol{{-{s[1:]}}}$"
        return rf"$\mathbf{{{s}}}$"
    ax.xaxis.set_major_formatter(FuncFormatter(_one))
    ax.yaxis.set_major_formatter(FuncFormatter(_one))


def style_axes_bold_labels_ticks(fig=None):
    import matplotlib.pyplot as plt
    fig = fig or plt.gcf()
    for ax in fig.axes:
        ax.set_xlabel(r"$\boldsymbol{x_{1,0}}$")
        ax.set_ylabel(r"$\boldsymbol{x_{2,0}}$")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.tick_params(axis="both", which="both", width=1.6, length=6)
        force_bold_ticklabels_tex(ax)


def main():
    # ---- load configs ----
    sys_cfg, abs_cfg, run_cfg, spec_cfg = get_ifac26_ra_configs()

    # ---- apply matplotlib style BEFORE importing pyplot ----
    apply_mpl_style(run_cfg.mpl_backend)
    import matplotlib.pyplot as plt
    import numpy as np

    from src.specifications.translate import translate
    from src.specifications.utils.dfa_tool import dfa_manipulation
    from src.performance.memory_check import print_workspace_memory
    from src.vis.plot_tv import plotV_rank1
    from src.pipeline import build_regions_from_cfg, build_system_with_ap, prepare_pipeline, run_tree

    # ---- build regions & per-agent regions/AP lists ----
    regions_by_agent = {}
    ap_by_agent = {}

    for k, s_k in sys_cfg.s_by_agent.items():
        region_polys = build_regions_from_cfg(s_k, sys_cfg.region_b_by_agent[k])  # dict name->Polytope
        region_names = sorted(region_polys.keys())
        regions_by_agent[k] = [region_polys[name] for name in region_names]
        ap_by_agent[k] = [sys_cfg.ap_label_by_region_by_agent[k][name] for name in region_names]

    # ---- build systems ----
    sysLTI = build_system_with_ap(
        sys_cfg=sys_cfg,
        bx_by_agent=sys_cfg.bx_by_agent,
        bu_by_agent=sys_cfg.bu_by_agent,
        s_by_agent=sys_cfg.s_by_agent,
        regions_by_agent=regions_by_agent,
        ap_by_agent=ap_by_agent,
    )

    # ---- DFA ----
    DFA = translate(spec_cfg.formula)
    DFA, letters = dfa_manipulation(
        DFA,
        index_base=0,
        ensure_transitions=True,
        remove_qf_self_loop=True,
        replacements=spec_cfg.replacements,
        desired_order=spec_cfg.desired_order,
        verbose=spec_cfg.verbose,
    )

    # ---- abstraction + labeling + policy/rho ----
    sysAbs, L, L_list, pol, rho, nx_list = prepare_pipeline(
        sysLTI=sysLTI,
        DFA=DFA,
        letters=letters,
        abs_cfg=abs_cfg,
        eps_val=run_cfg.eps_val,
    )

    # ---- tree + tv ----
    G, tv = run_tree(
        DFA=DFA,
        sysAbs=sysAbs,
        sysLTI=sysLTI,
        L=L,
        L_list=L_list,
        pol=pol,
        rho=rho,
        nx_list=nx_list,
        run_cfg=run_cfg,
    )

    # ---- memory ----
    gc.collect()
    if run_cfg.print_memory:
        print_workspace_memory(globals(), label="end-of-run")

    # ---- plot ----
    if run_cfg.plot_tv:
        plotV_rank1(sysAbs, tv)
        remove_titles()
        style_axes_bold_labels_ticks()
        plt.show()


if __name__ == "__main__":
    main()