from __future__ import annotations

from matplotlib.patches import Rectangle

try:
    from config.PD_setup import apply_mpl_style
except Exception:
    apply_mpl_style = None


# ---- Morandi palette ----
morandiBlue = (123 / 255, 143 / 255, 161 / 255)
morandiOrange = (214 / 255, 163 / 255, 116 / 255)  # not used here
morandiGreen = (182 / 255, 210 / 255, 196 / 255)
morandiBluePurple = (132 / 255, 136 / 255, 205 / 255)
morandiYellow = (246 / 255, 238 / 255, 138 / 255)
morandiPink = (206 / 255, 162 / 255, 170 / 255)

# mixed color = average of blue + yellow + pink + purple
morandiMixed = tuple(
    (morandiBlue[i] + morandiYellow[i] + morandiPink[i] + morandiBluePurple[i]) / 4.0
    for i in range(3)
)


def _with_alpha(rgb, a: float | None):
    if a is None:
        return rgb
    return (rgb[0], rgb[1], rgb[2], a)


def _add_block(
    ax,
    x0,
    x1,
    y0,
    y1,
    face_rgb,
    label=None,
    fontsize=22,
    weight="bold",
    z=1,
    face_alpha: float | None = None,
):
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=_with_alpha(face_rgb, face_alpha),
            edgecolor="black",
            linewidth=1.8,
            zorder=z,
        )
    )
    if label:
        ax.text(
            0.5 * (x0 + x1),
            0.5 * (y0 + y1),
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color="black",
            zorder=z + 1,
        )


def _setup_axes(ax):
    x_min, x_max = -20, 5
    y_min, y_max = -20, 5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    # no title / no x-y labels, but keep ticks
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # outer frame
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="black",
            linewidth=2.2,
            zorder=50,
        )
    )


def _draw_fig1_content(ax, with_text=True, z_base=1, face_alpha: float | None = None):
    _add_block(
        ax,
        -20,
        5,
        0,
        5,
        morandiGreen,
        label=(r"$\alpha_1$" if with_text else None),
        fontsize=30,
        z=z_base,
        face_alpha=face_alpha,
    )
    _add_block(
        ax,
        -20,
        5,
        -20,
        0,
        morandiGreen,
        label=(r"$\alpha_2$" if with_text else None),
        fontsize=30,
        z=z_base,
        face_alpha=face_alpha,
    )
    ax.axhline(0, color="black", linewidth=2.2, zorder=z_base + 20)


def _draw_fig2_content(ax, with_text=True, z_base=10, face_alpha: float | None = None):
    _add_block(
        ax,
        -20,
        5,
        -5,
        5,
        morandiGreen,
        label=(r"$\alpha_3$" if with_text else None),
        fontsize=28,
        z=z_base,
        face_alpha=face_alpha,
    )

    _add_block(
        ax,
        -20,
        -10,
        -10,
        -5,
        morandiGreen,
        label=(r"$\alpha_4$" if with_text else None),
        fontsize=24,
        z=z_base,
        face_alpha=face_alpha,
    )
    _add_block(
        ax,
        -5,
        5,
        -10,
        -5,
        morandiGreen,
        label=(r"$\alpha_4$" if with_text else None),
        fontsize=24,
        z=z_base,
        face_alpha=face_alpha,
    )

    _add_block(
        ax,
        -20,
        5,
        -15,
        -10,
        morandiGreen,
        label=(r"$\alpha_5$" if with_text else None),
        fontsize=24,
        z=z_base,
        face_alpha=face_alpha,
    )

    _add_block(
        ax,
        -20,
        -14,
        -20,
        -15,
        morandiBlue,
        label=(r"$\alpha_{10}$" if with_text else None),
        fontsize=18,
        z=z_base,
        face_alpha=face_alpha,
    )
    _add_block(
        ax,
        -14,
        -8,
        -20,
        -15,
        morandiYellow,
        label=(r"$\alpha_{10}$" if with_text else None),
        fontsize=18,
        z=z_base,
        face_alpha=face_alpha,
    )
    _add_block(
        ax,
        -8,
        -2,
        -20,
        -15,
        morandiPink,
        label=(r"$\alpha_{10}$" if with_text else None),
        fontsize=18,
        z=z_base,
        face_alpha=face_alpha,
    )
    _add_block(
        ax,
        -2,
        5,
        -20,
        -15,
        morandiBluePurple,
        label=(r"$\alpha_{10}$" if with_text else None),
        fontsize=18,
        z=z_base,
        face_alpha=face_alpha,
    )

    _add_block(
        ax,
        -10,
        -5,
        -10,
        -5,
        morandiMixed,
        label=(r"$\alpha_6,\alpha_7$" "\n" r"$\alpha_8,\alpha_9$" if with_text else None),
        fontsize=16,
        z=z_base + 5,
        face_alpha=face_alpha,
    )


def _draw_mix_explanation_on_ax2(ax):
    # place inside the second subplot, near the top-right of alpha_3
    y = 0.90
    box_w = 0.055
    box_h = 0.055

    items = [
        ("box", morandiMixed),
        ("text", "="),
        ("box", morandiBlue),
        ("text", "+"),
        ("box", morandiYellow),
        ("text", "+"),
        ("box", morandiPink),
        ("text", "+"),
        ("box", morandiBluePurple),
    ]

    dx_box = 0.085
    dx_text = 0.035
    total_width = 5 * box_w + 4 * (dx_box - box_w) + 4 * dx_text

    # right-aligned inside ax2
    x = 0.98 - total_width

    for kind, value in items:
        if kind == "box":
            ax.add_patch(
                Rectangle(
                    (x, y - box_h / 2),
                    box_w,
                    box_h,
                    facecolor=value,
                    edgecolor="black",
                    linewidth=1.2,
                    transform=ax.transAxes,
                    clip_on=False,
                    zorder=100,
                )
            )
            x += dx_box
        else:
            ax.text(
                x,
                y,
                value,
                ha="center",
                va="center",
                fontsize=13,
                transform=ax.transAxes,
                zorder=101,
            )
            x += dx_text


def combined_figure():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 6.2), dpi=140)

    ax1 = fig.add_axes([0.08, 0.20, 0.38, 0.72])
    ax2 = fig.add_axes([0.54, 0.20, 0.38, 0.72])

    _setup_axes(ax1)
    _setup_axes(ax2)

    _draw_fig1_content(ax1, with_text=True, z_base=1, face_alpha=None)
    _draw_fig2_content(ax2, with_text=True, z_base=10, face_alpha=None)

    # draw the color-mix explanation inside the second plot
    _draw_mix_explanation_on_ax2(ax2)

    return fig


def main():
    if apply_mpl_style is not None:
        apply_mpl_style(prefer_backend="default")

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["savefig.bbox"] = None

    fig = combined_figure()
    fig.canvas.draw()
    fig.savefig("subformula_outputspace.eps", format="eps")
    plt.show()


if __name__ == "__main__":
    main()