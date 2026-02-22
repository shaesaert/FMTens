# README:
# run this script to re-produce Fig. 5 in paper https://arxiv.org/pdf/2511.06873

# run_ifac26_iv.py
from __future__ import annotations

import numpy as np
import matplotlib.ticker as mticker

from config.IFAC26_IV_setup import (
    get_ifac26_iv_config,
    apply_mpl_style,
    make_iv_sysLTI_and_safeset,
    make_iv_formula,
)
from config.IFAC26_IV_setup import build_dfa_from_formula
from src.pipeline import prepare_pipeline
from config.sweep_eval import evaluate_dim_sweep, SweepResult


def plot_results(cfg, results: list[SweepResult]):
    import matplotlib.pyplot as plt

    # Plot 1: RT/APoS curves for all dims in results
    fig1, ax1 = plt.subplots(figsize=(12, 3))
    markers = [("s", "o"), ("^", "v"), ("D", "x"), ("P", "*")]

    for i, res in enumerate(results):
        m_rt, m_ap = markers[i % len(markers)]
        N = int(res.DIM)
        ax1.plot(res.t_specs, res.V_rt_avg, marker=m_rt, linewidth=2.0,
                 label=rf'$\textbf{{Optimal RT Value functions}} \ (N={N})$')
        ax1.plot(res.t_specs, res.V_apos_avg, marker=m_ap, linewidth=2.0, linestyle="--",
                 label=rf'$\textbf{{Optimal APoS Value functions}} \ (N={N})$')

    ax1.legend(
        prop={'size': 16, 'weight': 'bold'},
        loc='upper right', bbox_to_anchor=(0.98, 0.98),
        bbox_transform=ax1.transAxes,
        ncol=1, frameon=True, framealpha=0.9,
        borderpad=0.3, labelspacing=0.25,
        handlelength=2.0, handletextpad=0.5, markerscale=0.9
    )
    ax1.xaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.tick_params(axis="both", labelsize=18, length=8, width=2, pad=8)
    ax1.minorticks_off()
    ax1.margins(x=0.08)
    ax1.set_xlabel(r'\textbf{Specification Horizon}')
    for sp in ax1.spines.values():
        sp.set_linewidth(2)
    ax1.grid(False)
    fig1.tight_layout()
    fig1.savefig(cfg.plot1_png, dpi=300, bbox_inches="tight")

    # Plot 2: FHT_avg vs T_build
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    for i, res in enumerate(results):
        N = int(res.DIM)
        ax2.plot(res.T_builds, res.FHT_avg, marker=markers[i % len(markers)][1], linewidth=2.2,
                 label=rf'$\textbf{{FHT term}} \ (N={N})$')

    ax2.legend(
        prop={'size': 16, 'weight': 'bold'},
        loc='upper left', bbox_to_anchor=(0.02, 0.98),
        bbox_transform=ax2.transAxes,
        ncol=1, frameon=True, framealpha=0.9,
        borderpad=0.3, labelspacing=0.25,
        handlelength=2.0, handletextpad=0.5, markerscale=0.9
    )
    ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter(r'\textbf{{{x:g}}}'))
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.tick_params(axis="both", labelsize=18, length=8, width=2, pad=8)
    ax2.minorticks_off()
    ax2.margins(x=0.08)
    ax2.set_xlabel(r'\textbf{$T_{\text{build}}$}')
    ax2.set_ylabel(r'\textbf{FHT term (avg over anchors)}')
    for sp in ax2.spines.values():
        sp.set_linewidth(2)
    ax2.grid(False)
    fig2.tight_layout()
    fig2.savefig(cfg.plot2_png, dpi=300, bbox_inches="tight")

    plt.show(block=True)


def main():
    cfg = get_ifac26_iv_config()

    # apply style BEFORE pyplot import
    apply_mpl_style(cfg.prefer_backend)
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    print("Matplotlib backend:", mpl.get_backend())

    results: list[SweepResult] = []

    for DIM in cfg.dims:
        out_npz = f"{cfg.save_prefix}_D{DIM}.npz"
        res = evaluate_dim_sweep(
            DIM=DIM,
            t_specs=cfg.t_specs,
            anchor_centers=cfg.anchor_centers,
            eps_val=cfg.eps_val,
            delta_i=cfg.delta_i,
            make_system_and_safeset=make_iv_sysLTI_and_safeset,
            make_formula=make_iv_formula,
            cfg_obj=cfg,
            abs_cfg=cfg.abs_cfg,
            build_dfa_from_formula=build_dfa_from_formula,
            spec_manip_cfg=cfg.spec_manip,
            prepare_pipeline_func=prepare_pipeline,
            save_npz_path=out_npz,
        )
        print(f"[D={DIM}] saved: {out_npz}")
        results.append(res)

    # combined save (generic for any number of dims)
    combined = {
        "t_specs": results[0].t_specs,
        "anchor_centers": np.asarray(cfg.anchor_centers, dtype=int),
    }
    for res in results:
        combined[f"V_rt_avg_D{res.DIM}"] = res.V_rt_avg
        combined[f"V_apos_avg_D{res.DIM}"] = res.V_apos_avg
        combined[f"FHT_avg_D{res.DIM}"] = res.FHT_avg
        combined[f"T_builds_D{res.DIM}"] = res.T_builds
    np.savez(cfg.save_combined_npz, **combined)
    print(f"Saved combined data to {cfg.save_combined_npz}")

    plot_results(cfg, results)


if __name__ == "__main__":
    main()
