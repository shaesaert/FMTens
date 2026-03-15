# # run_PD_Rank1_8D.py
# from __future__ import annotations
#
# from config.PD_setup import (
#     get_pd8d_config,
#     apply_mpl_style,
#     build_sysLTI_pd8d,
#     build_manual_dfa_pd8d,
# )
#
# from src.pipeline import prepare_pipeline, run_tree
#
# from config.PD_8D_4robots_eval import (
#     tv_value_at_point,
#     OverlayBox,
#     add_overlay_boxes,
#     scatter_point_sets_with_agent_colors,
# )
#
#
# def main():
#     cfg = get_pd8d_config()
#     run_cfg = cfg.run_cfg
#     abs_cfg = cfg.abs_cfg
#
#     # ---- matplotlib style BEFORE pyplot ----
#     apply_mpl_style(run_cfg.prefer_backend)
#     import matplotlib.pyplot as plt
#
#     # ---- build system + DFA (from config) ----
#     sysLTI = build_sysLTI_pd8d(cfg)
#     DFA, letters = build_manual_dfa_pd8d()
#
#     # ---- abstraction + labeling + uniform policy/rho ----
#     sysAbs, L, L_list, pol, rho, nx_list = prepare_pipeline(
#         sysLTI=sysLTI,
#         DFA=DFA,
#         letters=letters,
#         abs_cfg=abs_cfg,
#         eps_val=run_cfg.eps_val,
#     )
#
#     # ---- build tree only (skip tv tensor) ----
#     G, _ = run_tree(
#         DFA=DFA,
#         sysAbs=sysAbs,
#         sysLTI=sysLTI,
#         L=L,
#         L_list=L_list,
#         pol=pol,
#         rho=rho,
#         nx_list=nx_list,
#         run_cfg=run_cfg,
#         compute_tv=False,   # requires your updated pipeline.run_tree
#     )
#
#     # ---- figure (overlays + point sets only) ----
#     fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
#     ax.set_xlim(*run_cfg.x_region)
#     ax.set_ylim(*run_cfg.y_region)
#     ax.set_aspect("equal", adjustable="box")
#     # ax.set_xlabel("x")
#     # ax.set_ylabel("y")
#     # ax.set_title(
#     #     "Satisfaction probabilities of different set of robots' initial positions\n"
#     #     f"specification horizon={run_cfg.T + 1}"
#     # )
#     ax.set_xlabel(r"$p_x^i$")
#     ax.set_ylabel(r"$p_y^i$")
#
#     # overlay boxes from config
#     boxes = [OverlayBox(**b.__dict__) for b in cfg.overlay_boxes]
#     add_overlay_boxes(ax, boxes, fontsize=10)
#
#     # tv scalar at each 8D point-set + scatter/annotate
#     point_sets = list(cfg.point_sets)
#     tv_vals = [tv_value_at_point(G, DFA, L, sysAbs, p8, index_mode="nearest") for p8 in point_sets]
#
#     dims_pairs = [("A", 0, 1), ("B", 2, 3), ("C", 4, 5), ("D", 6, 7)]
#     agent_face = {"A": "red", "B": "blue", "C": "yellow", "D": "green"}
#     markers = ["o", "s", "^", "D"]
#
#     scatter_point_sets_with_agent_colors(
#         ax,
#         point_sets,
#         dims_pairs=dims_pairs,
#         agent_face=agent_face,
#         markers=markers,
#         annotate_agent="A",  # only annotate A
#         dx=0.20,
#         dy=0.20,
#         tv_values=tv_vals,
#     )
#
#     # reserve room for legend on the right
#     fig.subplots_adjust(right=0.78)
#     plt.tight_layout()
#
#     # optional save
#     if getattr(run_cfg, "save_png", None):
#         fig.savefig(run_cfg.save_png, dpi=300, bbox_inches="tight")
#
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()


from __future__ import annotations

from config.IFAC26_8D_4robots_setup import (
    get_pd8d_config,
    apply_mpl_style,
    build_sysLTI_pd8d,
    build_manual_dfa_pd8d,
)

from src.pipeline import prepare_pipeline, run_tree

from config.IFAC26_8D_4robots_eval import (
    tv_value_at_point,
    scatter_point_sets_with_agent_colors,
)

from matplotlib.patches import Rectangle


def _box_label(box):
    d = box.__dict__
    for k in ("label", "name", "text"):
        if k in d:
            return str(d[k])
    return ""


def _box_xywh(box):
    d = box.__dict__

    # case 1: xy + width + height
    if all(k in d for k in ("xy", "width", "height")):
        return d["xy"], d["width"], d["height"]

    # case 2: xy + w + h
    if all(k in d for k in ("xy", "w", "h")):
        return d["xy"], d["w"], d["h"]

    # case 3: x + y + w + h
    if all(k in d for k in ("x", "y", "w", "h")):
        return (d["x"], d["y"]), d["w"], d["h"]

    # case 4: xmin + xmax + ymin + ymax
    if all(k in d for k in ("xmin", "xmax", "ymin", "ymax")):
        return (d["xmin"], d["ymin"]), d["xmax"] - d["xmin"], d["ymax"] - d["ymin"]

    raise ValueError(f"Unsupported OverlayBox fields: {list(d.keys())}")


def _keep_region(label: str) -> bool:
    s = label.lower()
    return (
        "pick" in s
        or "deliver" in s
        or ("package" in s and "loss" in s)
        or "loss" in s
    )


def _display_region_text(label: str) -> str | None:
    s = label.lower().replace("_", " ").replace("-", " ")

    if "pick" in s:
        return "Pickup"

    if "package" in s and "loss" in s:
        return "Package-lose"

    if "deliver" in s:
        tokens = set(s.split())
        if "a" in tokens:
            return "Deliver-A"
        if "b" in tokens:
            return "Deliver-B"
        if "c" in tokens:
            return "Deliver-C"
        if "d" in tokens:
            return "Delievr-D"   # exact spelling you asked for
        return "Deliver"

    return None


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
        compute_tv=False,
    )

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    ax.set_xlim(*run_cfg.x_region)
    ax.set_ylim(*run_cfg.y_region)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$p_x^i$")
    ax.set_ylabel(r"$p_y^i$")

    # ---- draw only pickup / deliver / package-loss regions ----
    selected_boxes = []
    for b in cfg.overlay_boxes:
        label = _box_label(b)
        if _keep_region(label):
            selected_boxes.append(b)

    for b in selected_boxes:
        (x, y), w, h = _box_xywh(b)
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                edgecolor="black",
                linestyle="--",
                linewidth=1.2,
            )
        )

    # ---- region labels: large and bold ----
    for b in selected_boxes:
        label = _box_label(b)
        show_text = _display_region_text(label)
        if show_text is None:
            continue

        d = b.__dict__
        if "text_xy" in d:
            tx, ty = d["text_xy"]
            ax.text(
                tx,
                ty,
                show_text,
                fontsize=22,
                fontweight="bold",
                color="black",
                ha="left",
                va="center",
            )
        else:
            (x, y), w, h = _box_xywh(b)
            ax.text(
                x + 0.04 * w,
                y + 0.96 * h,
                show_text,
                ha="left",
                va="top",
                fontsize=22,
                fontweight="bold",
                color="black",
            )

    # ---- tv scalar at each 8D point-set + scatter/annotate ----
    point_sets = list(cfg.point_sets)
    tv_vals = [
        tv_value_at_point(G, DFA, L, sysAbs, p8, index_mode="nearest")
        for p8 in point_sets
    ]

    dims_pairs = [("A", 0, 1), ("B", 2, 3), ("C", 4, 5), ("D", 6, 7)]
    agent_face = {"A": "red", "B": "blue", "C": "yellow", "D": "green"}
    markers = ["o", "s", "^", "D"]

    scatter_point_sets_with_agent_colors(
        ax,
        point_sets,
        dims_pairs=dims_pairs,
        agent_face=agent_face,
        markers=markers,
        annotate_agent="A",
        dx=0.55,   # move probability text farther away
        dy=0.55,
        tv_values=tv_vals,
    )

    # ---- make probability text larger + bold ----
    # This assumes the probability annotations are added as text objects
    # and contain numeric values. If needed, adjust this filter.
    for txt in ax.texts:
        s = txt.get_text().strip()
        try:
            float(s)
            txt.set_fontsize(12)
            txt.set_fontweight("bold")
        except ValueError:
            pass

    # ---- move legend inside axes ----
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l and not l.startswith("_") and l not in uniq:
            uniq[l] = h

    if uniq:
        ax.legend(
            uniq.values(),
            uniq.keys(),
            loc="upper right",
            frameon=True,
            fontsize=9,
        )

    plt.tight_layout()

    # optional save
    if getattr(run_cfg, "save_png", None):
        fig.savefig(run_cfg.save_png, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()