# run_PD_Rank1_8D.py
from __future__ import annotations

from config.PD_setup import (
    get_pd8d_config,
    apply_mpl_style,
    build_sysLTI_pd8d,
    build_manual_dfa_pd8d,
)

from src.pipeline import prepare_pipeline, run_tree

from config.PD_8D_4robots_eval import (
    tv_value_at_point,
    OverlayBox,
    add_overlay_boxes,
    scatter_point_sets_with_agent_colors,
)


def main():
    cfg = get_pd8d_config()
    run_cfg = cfg.run_cfg
    abs_cfg = cfg.abs_cfg

    # ---- matplotlib style BEFORE pyplot ----
    apply_mpl_style(run_cfg.prefer_backend)
    import matplotlib.pyplot as plt

    # ---- build system + DFA (from config) ----
    sysLTI = build_sysLTI_pd8d(cfg)
    DFA, letters = build_manual_dfa_pd8d()

    # ---- abstraction + labeling + uniform policy/rho ----
    sysAbs, L, L_list, pol, rho, nx_list = prepare_pipeline(
        sysLTI=sysLTI,
        DFA=DFA,
        letters=letters,
        abs_cfg=abs_cfg,
        eps_val=run_cfg.eps_val,
    )

    # ---- build tree only (skip tv tensor) ----
    G, _ = run_tree(
        DFA=DFA,
        sysAbs=sysAbs,
        sysLTI=sysLTI,
        L=L,
        L_list=L_list,
        pol=pol,
        rho=rho,
        nx_list=nx_list,
        run_cfg=run_cfg,
        compute_tv=False,   # requires your updated pipeline.run_tree
    )

    # ---- figure (overlays + point sets only) ----
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    ax.set_xlim(*run_cfg.x_region)
    ax.set_ylim(*run_cfg.y_region)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "Satisfaction probabilities of different set of robots' initial positions\n"
        f"specification horizon={run_cfg.T + 1}"
    )

    # overlay boxes from config
    boxes = [OverlayBox(**b.__dict__) for b in cfg.overlay_boxes]
    add_overlay_boxes(ax, boxes, fontsize=10)

    # tv scalar at each 8D point-set + scatter/annotate
    point_sets = list(cfg.point_sets)
    tv_vals = [tv_value_at_point(G, DFA, L, sysAbs, p8, index_mode="nearest") for p8 in point_sets]

    dims_pairs = [("A", 0, 1), ("B", 2, 3), ("C", 4, 5), ("D", 6, 7)]
    agent_face = {"A": "red", "B": "blue", "C": "yellow", "D": "green"}
    markers = ["o", "s", "^", "D"]

    scatter_point_sets_with_agent_colors(
        ax,
        point_sets,
        dims_pairs=dims_pairs,
        agent_face=agent_face,
        markers=markers,
        annotate_agent="A",  # only annotate A
        dx=0.20,
        dy=0.20,
        tv_values=tv_vals,
    )

    # reserve room for legend on the right
    fig.subplots_adjust(right=0.78)
    plt.tight_layout()

    # optional save
    if getattr(run_cfg, "save_png", None):
        fig.savefig(run_cfg.save_png, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()