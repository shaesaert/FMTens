# paper_artifacts/arxiv_2511_06873/fig6_subformula.py
"""
Reproduces Fig. 6 of arxiv:2511.06873.

Illustrative figure showing the output-space partition for two subformulas
of the 4-robot-package delivery (PD8D) specification. Two side-by-side panels:

    Left panel  — subformula 1's output regions (alpha_1, alpha_2)
    Right panel — subformula 2's output regions (alpha_3 ... alpha_10),
                  with a small inline legend showing that the mixed colour
                  in the center block is the average of the four corner
                  colours.

This script is purely illustrative.
Drawing geometry is hard-coded.

Run from the repository root:
    python -m paper_artifacts.arxiv_2511_06873.fig6_subformula
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.patches import Rectangle


# ===========================================================================
# Paper parameters — published in arxiv:2511.06873
# ===========================================================================
# Morandi palette
MORANDI_BLUE        = (123 / 255, 143 / 255, 161 / 255)
MORANDI_ORANGE      = (214 / 255, 163 / 255, 116 / 255)  # not used here; kept for completeness
MORANDI_GREEN       = (182 / 255, 210 / 255, 196 / 255)
MORANDI_BLUE_PURPLE = (132 / 255, 136 / 255, 205 / 255)
MORANDI_YELLOW      = (246 / 255, 238 / 255, 138 / 255)
MORANDI_PINK        = (206 / 255, 162 / 255, 170 / 255)

# average of blue + yellow + pink + purple
MORANDI_MIXED = tuple(
    (MORANDI_BLUE[i] + MORANDI_YELLOW[i] + MORANDI_PINK[i] + MORANDI_BLUE_PURPLE[i]) / 4.0
    for i in range(3)
)

# Drawing extents — shared by both panels
X_MIN, X_MAX = -20, 5
Y_MIN, Y_MAX = -20, 5

HERE = Path(__file__).resolve().parent


# ===========================================================================
# Drawing helpers
# ===========================================================================
def _with_alpha(rgb, a):
    if a is None:
        return rgb
    return (rgb[0], rgb[1], rgb[2], a)


def _add_block(ax, x0, x1, y0, y1, face_rgb, *,
               label=None, fontsize=22, weight="bold",
               z=1, face_alpha=None):
    ax.add_patch(
        Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=_with_alpha(face_rgb, face_alpha),
            edgecolor="black", linewidth=1.8, zorder=z,
        )
    )
    if label:
        ax.text(
            0.5 * (x0 + x1), 0.5 * (y0 + y1), label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="black",
            zorder=z + 1,
        )


def _setup_axes(ax):
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # outer frame
    ax.add_patch(
        Rectangle(
            (X_MIN, Y_MIN), X_MAX - X_MIN, Y_MAX - Y_MIN,
            fill=False, edgecolor="black", linewidth=2.2, zorder=50,
        )
    )


# ===========================================================================
# Panel content
# ===========================================================================
def _draw_left_panel(ax, *, z_base=1, face_alpha=None):
    """Subformula 1 output regions: alpha_1 (upper band), alpha_2 (lower band)."""
    _add_block(ax, -20, 5,    0,  5, MORANDI_GREEN,
               label=r"$\alpha_1$", fontsize=30, z=z_base, face_alpha=face_alpha)
    _add_block(ax, -20, 5,  -20,  0, MORANDI_GREEN,
               label=r"$\alpha_2$", fontsize=30, z=z_base, face_alpha=face_alpha)
    ax.axhline(0, color="black", linewidth=2.2, zorder=z_base + 20)


def _draw_right_panel(ax, *, z_base=10, face_alpha=None):
    """Subformula 2 output regions: alpha_3 ... alpha_10."""
    _add_block(ax, -20,  5,   -5,   5, MORANDI_GREEN,
               label=r"$\alpha_3$", fontsize=28, z=z_base, face_alpha=face_alpha)

    _add_block(ax, -20, -10, -10,  -5, MORANDI_GREEN,
               label=r"$\alpha_4$", fontsize=24, z=z_base, face_alpha=face_alpha)
    _add_block(ax,  -5,   5, -10,  -5, MORANDI_GREEN,
               label=r"$\alpha_4$", fontsize=24, z=z_base, face_alpha=face_alpha)

    _add_block(ax, -20,   5, -15, -10, MORANDI_GREEN,
               label=r"$\alpha_5$", fontsize=24, z=z_base, face_alpha=face_alpha)

    _add_block(ax, -20, -14, -20, -15, MORANDI_BLUE,
               label=r"$\alpha_{10}$", fontsize=18, z=z_base, face_alpha=face_alpha)
    _add_block(ax, -14,  -8, -20, -15, MORANDI_YELLOW,
               label=r"$\alpha_{10}$", fontsize=18, z=z_base, face_alpha=face_alpha)
    _add_block(ax,  -8,  -2, -20, -15, MORANDI_PINK,
               label=r"$\alpha_{10}$", fontsize=18, z=z_base, face_alpha=face_alpha)
    _add_block(ax,  -2,   5, -20, -15, MORANDI_BLUE_PURPLE,
               label=r"$\alpha_{10}$", fontsize=18, z=z_base, face_alpha=face_alpha)

    _add_block(ax, -10, -5, -10, -5, MORANDI_MIXED,
               label=r"$\alpha_6,\alpha_7$" "\n" r"$\alpha_8,\alpha_9$",
               fontsize=16, z=z_base + 5, face_alpha=face_alpha)


def _draw_mix_explanation(ax):
    """Inline color-mix legend on the right panel: mixed = blue + yellow + pink + purple."""
    y = 0.90
    box_w = 0.055
    box_h = 0.055

    items = [
        ("box",  MORANDI_MIXED),
        ("text", "="),
        ("box",  MORANDI_BLUE),
        ("text", "+"),
        ("box",  MORANDI_YELLOW),
        ("text", "+"),
        ("box",  MORANDI_PINK),
        ("text", "+"),
        ("box",  MORANDI_BLUE_PURPLE),
    ]

    dx_box      = 0.085
    dx_text     = 0.035
    total_width = 5 * box_w + 4 * (dx_box - box_w) + 4 * dx_text
    x = 0.98 - total_width

    for kind, value in items:
        if kind == "box":
            ax.add_patch(
                Rectangle(
                    (x, y - box_h / 2), box_w, box_h,
                    facecolor=value, edgecolor="black", linewidth=1.2,
                    transform=ax.transAxes, clip_on=False, zorder=100,
                )
            )
            x += dx_box
        else:
            ax.text(
                x, y, value,
                ha="center", va="center",
                fontsize=13, transform=ax.transAxes, zorder=101,
            )
            x += dx_text


# ===========================================================================
# Plotting
# ===========================================================================
def _apply_paper_style():
    """Mild rcParam overrides; no LaTeX (labels are mathtext, render fine natively)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size":        16,
        "axes.titlesize":   18,
        "axes.labelsize":   18,
        "xtick.labelsize":  14,
        "ytick.labelsize":  14,
        "legend.fontsize":  14,
        "font.weight":      "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "savefig.bbox":     None,    # preserve manual axes positions on save
    })


def _combined_figure():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 6.2), dpi=140)

    ax_left  = fig.add_axes([0.08, 0.20, 0.38, 0.72])
    ax_right = fig.add_axes([0.54, 0.20, 0.38, 0.72])

    _setup_axes(ax_left)
    _setup_axes(ax_right)

    _draw_left_panel(ax_left)
    _draw_right_panel(ax_right)
    _draw_mix_explanation(ax_right)

    return fig


def main() -> int:
    _apply_paper_style()
    import matplotlib.pyplot as plt

    fig = _combined_figure()
    fig.canvas.draw()

    fig.savefig(HERE / "fig6.png", dpi=300)
    fig.savefig(HERE / "fig6.eps", format="eps")
    print("[fig6] wrote fig6.png, fig6.eps")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())