# src/multirobot_eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Optional

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# -------------------------
# Grid helpers
# -------------------------
def centers_1d(sysAbs_d, N_expected: int) -> np.ndarray:
    """
    Return 1D grid centers from sysAbs_d.hx.
    Assumes sysAbs_d.hx is either an array-like, or a list/tuple whose first element is the 1D centers.
    """
    hx = sysAbs_d.hx
    x = np.asarray(hx[0] if isinstance(hx, (list, tuple)) else hx).ravel()
    if x.size != int(N_expected):
        raise ValueError(f"hx length {x.size} != N {N_expected}")
    return x


def centers_to_edges(c: np.ndarray) -> np.ndarray:
    """
    Convert centers to edges for pcolormesh.
    """
    c = np.asarray(c, dtype=float)
    if c.size == 0:
        return np.asarray([], dtype=float)
    if c.size == 1:
        dc = 1.0
        return np.array([c[0] - dc / 2, c[0] + dc / 2], dtype=float)

    mid = 0.5 * (c[1:] + c[:-1])
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]]).astype(float)


def coord_to_index(centers: np.ndarray, val: float, *, mode: str = "nearest", tol: float = 1e-9):
    """
    Map a coordinate (in continuous center coordinates) to an abstract index.

    mode:
      - "nearest": choose nearest center (always succeeds)
      - "exact": require |center-val| <= tol
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        raise ValueError("centers is empty")

    idx = int(np.argmin(np.abs(centers - val)))
    err = float(abs(centers[idx] - val))
    if mode == "exact" and err > tol:
        raise ValueError(f"value {val} not found (nearest {centers[idx]} @ idx {idx}, err {err})")
    return idx, err, float(centers[idx])


# -------------------------
# Scalar tv at one D-dimensional point
# -------------------------
def tv_value_at_point(
    G, DFA, L, sysAbs,
    point: Sequence[float],
    *,
    index_mode: str = "nearest",
    tol: float = 1e-9,
) -> float:
    """
    Compute scalar tv at a single point in center coordinates, WITHOUT constructing the full tensor.

    point length must equal the number of dimensions/agents (len(sysAbs)).

    Algorithm:
      tv = sum_{letters l} [ Π_d L[d][l, idx_d] ] * [ Σ_{nodes n in Q[qdst]} Π_d V[d][n, idx_d] ]
    where qdst = trans[q0, l].
    """
    D = len(point)
    q0 = int(DFA.S0[0])
    trans = np.asarray(DFA.trans, dtype=int)
    n_letters = trans.shape[1]

    # indices per dimension
    idxs = []
    for d in range(D):
        c = centers_1d(sysAbs[d], sysAbs[d].N)
        idx, _, _ = coord_to_index(c, float(point[d]), mode=index_mode, tol=tol)
        idxs.append(idx)

    # reachable destination states from q0
    qdsts = np.unique(trans[q0, :]).astype(int)
    qdsts = [int(q) for q in qdsts if len(G.Q.get(int(q), [])) > 0]

    # sum over nodes at each qdst (at this fixed idx vector)
    node_sum: Dict[int, float] = {}
    for q in qdsts:
        acc = 0.0
        for n in G.Q.get(q, []):
            prod = 1.0
            for d in range(D):
                prod *= float(G.V[d][n, idxs[d]])
                if prod == 0.0:
                    break
            acc += prod
        node_sum[q] = float(acc)

    # accumulate over letters
    tv = 0.0
    for l in range(n_letters):
        qdst = int(trans[q0, l])
        acc = node_sum.get(qdst, None)
        if acc is None:
            continue

        mask = 1.0
        for d in range(D):
            mask *= float(L[d][l, idxs[d]])
            if mask == 0.0:
                break

        tv += mask * acc

    return float(tv)


# -------------------------
# Diagonal coupled 2D tv slice for multi-robot (x/y pairing)
# -------------------------
def tv_diagonal_2d_from_tree(
    G, DFA, L, sysAbs,
    *,
    x_region: Tuple[float, float],
    y_region: Tuple[float, float],
    dims_x: Sequence[int],
    dims_y: Sequence[int],
    inclusive: bool = True,
    clip01: bool = True,
    verbose: bool = False,
):
    """
    Computes a 2D heatmap tv2 over (x,y) for a diagonal coupling:

      For each x-index i and y-index j:
        idx_vec[d] = i for d in dims_x
        idx_vec[d] = j for d in dims_y

      tv2[j,i] = tv(idx_vec) but computed efficiently without forming the full D-D tensor.

    Returns:
      tv2: (ny, nx)
      idx_x: indices used from x centers
      idx_y: indices used from y centers
    """
    q0 = int(DFA.S0[0])
    trans = np.asarray(DFA.trans, dtype=int)
    n_letters = trans.shape[1]

    # assume all dims_x share same x centers, dims_y share same y centers
    x_full = centers_1d(sysAbs[dims_x[0]], sysAbs[dims_x[0]].N)
    y_full = centers_1d(sysAbs[dims_y[0]], sysAbs[dims_y[0]].N)

    x_lo, x_hi = map(float, x_region)
    y_lo, y_hi = map(float, y_region)

    if inclusive:
        idx_x = np.where((x_full >= x_lo) & (x_full <= x_hi))[0]
        idx_y = np.where((y_full >= y_lo) & (y_full <= y_hi))[0]
    else:
        idx_x = np.where((x_full > x_lo) & (x_full < x_hi))[0]
        idx_y = np.where((y_full > y_lo) & (y_full < y_hi))[0]

    nx_r = int(len(idx_x))
    ny_r = int(len(idx_y))
    if nx_r == 0 or ny_r == 0:
        return np.zeros((ny_r, nx_r), dtype=float), idx_x, idx_y

    # destination states reachable from q0
    qdsts = np.unique(trans[q0, :]).astype(int)
    qdsts = [int(q) for q in qdsts if len(G.Q.get(int(q), [])) > 0]

    # Precompute for each q: sum over nodes of (yprod[:,None] * xprod[None,:])
    node_sum2D: Dict[int, np.ndarray] = {}
    for q in qdsts:
        acc = np.zeros((ny_r, nx_r), dtype=float)
        for n in G.Q.get(q, []):
            xprod = np.ones(nx_r, dtype=float)
            for d in dims_x:
                xprod *= np.asarray(G.V[d][n, idx_x], dtype=float)

            yprod = np.ones(ny_r, dtype=float)
            for d in dims_y:
                yprod *= np.asarray(G.V[d][n, idx_y], dtype=float)

            acc += yprod[:, None] * xprod[None, :]
        node_sum2D[q] = acc

    tv2 = np.zeros((ny_r, nx_r), dtype=float)

    # Accumulate over letters l:
    # tv2 += (ymask[:,None] * xmask[None,:]) * node_sum2D[qdst]
    for l in range(n_letters):
        qdst = int(trans[q0, l])
        acc = node_sum2D.get(qdst, None)
        if acc is None:
            continue

        xmask = np.ones(nx_r, dtype=float)
        for d in dims_x:
            xmask *= np.asarray(L[d][l, idx_x], dtype=float)

        ymask = np.ones(ny_r, dtype=float)
        for d in dims_y:
            ymask *= np.asarray(L[d][l, idx_y], dtype=float)

        tv2 += (ymask[:, None] * xmask[None, :]) * acc

    if clip01:
        tv2 = np.clip(tv2, 0.0, 1.0)

    if verbose:
        print(f"tv2 shape={tv2.shape}, nx={nx_r}, ny={ny_r}, cached qdsts={sorted(node_sum2D.keys())}")

    return tv2, idx_x, idx_y


# -------------------------
# Visualization overlays + point-set markers
# -------------------------
@dataclass
class OverlayBox:
    xy: Tuple[float, float]
    w: float
    h: float
    label: str
    color: str
    ls: str = "--"
    lw: float = 2.0
    text_xy: Optional[Tuple[float, float]] = None


def add_overlay_boxes(ax, boxes: Sequence[OverlayBox], *, fontsize: int = 10):
    """
    Add a list of OverlayBox rectangles + labels to an Axes.
    """
    for b in boxes:
        ax.add_patch(Rectangle(b.xy, b.w, b.h, fill=False, linewidth=b.lw, linestyle=b.ls, edgecolor=b.color))
        tx, ty = b.text_xy if b.text_xy is not None else (b.xy[0] + 0.5, b.xy[1] + b.h - 0.5)
        ax.text(tx, ty, b.label, fontsize=fontsize, va="top", color=b.color)


def scatter_point_sets_with_agent_colors(
    ax,
    point_sets: Sequence[Sequence[float]],
    *,
    dims_pairs: Sequence[Tuple[str, int, int]],
    agent_face: Dict[str, str],
    markers: Sequence[str],
    annotate_agent: str = "A",
    dx: float = 0.2,
    dy: float = 0.2,
    tv_values: Optional[Sequence[float]] = None,
):
    """
    Scatter multiple 8D (or D-dimensional) point sets by projecting each robot/agent pair (x_i,y_i)
    and using:
      - marker shape for point-set index
      - fill color for agent label

    If tv_values provided, annotate the selected 'annotate_agent' points with tv_values[p_idx].
    """
    for p_idx, p in enumerate(point_sets):
        mk = markers[p_idx % len(markers)]
        tv_val = None if tv_values is None else float(tv_values[p_idx])

        for agent, xi, yi in dims_pairs:
            px, py = float(p[xi]), float(p[yi])
            ax.scatter(
                px, py,
                marker=mk,
                s=90,
                facecolors=agent_face[agent],
                edgecolors="k",
                linewidths=0.8,
                zorder=5
            )

            if agent == annotate_agent and tv_val is not None:
                ax.text(
                    px + dx, py + dy,
                    f"{tv_val:.3g}",
                    fontsize=25,
                    fontweight="bold",
                    zorder=6
                )

    # legend for agent colors (independent of marker shape)
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=agent_face[a], markeredgecolor='k',
               markersize=9, label=f"Robot {a}")
        for a in agent_face.keys()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.9
    )