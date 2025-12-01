# src/vis/dfa_tree_viz.py
from __future__ import annotations
from collections import defaultdict, deque
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def _as_tree_and_dfa(
    obj_or_tree: Union[nx.DiGraph, object], dfa: Optional[object]
) -> tuple[nx.DiGraph, Optional[object]]:
    """Accept a DFATree instance (with .tree/.DFA) or a bare DiGraph + DFA."""
    if isinstance(obj_or_tree, nx.DiGraph):
        if dfa is None:
            raise ValueError("When passing a DiGraph directly, you must also pass DFA=")
        return obj_or_tree, dfa
    if hasattr(obj_or_tree, "tree"):
        return obj_or_tree.tree, (dfa or getattr(obj_or_tree, "DFA", None))
    raise TypeError("First arg must be a networkx.DiGraph or a DFATree-like object.")


def _layered_positions(G: nx.DiGraph, root: int = 0, xpad: float = 2.0, ystep: float = 2.6):
    """Simple top-down layered layout (BFS levels)."""
    depth = {root: 0}
    layers = defaultdict(list)
    q = deque([root])
    while q:
        u = q.popleft()
        d = depth[u]
        layers[d].append(u)
        for v in G.successors(u):
            if v not in depth:
                depth[v] = d + 1
                q.append(v)

    pos = {}
    for d in sorted(layers):
        nodes = layers[d]
        k = len(nodes)
        xs = np.linspace(-(k - 1) * xpad / 2.0, (k - 1) * xpad / 2.0, k) if k > 1 else np.array([0.0])
        for x, n in zip(xs, nodes):
            pos[n] = (float(x), -d * ystep)
    # catch any disconnected nodes
    for n in G.nodes:
        if n not in pos:
            pos[n] = (0.0, -(max(layers) + 1) * ystep)
    return pos


def _as_iter(v) -> list[int]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [int(x) for x in v]
    return [int(v)]

def _dfa_state_name(DFA, q: int) -> str:
    """Human name for a DFA state id q: prefer F and S0/ SO when applicable."""
    name_parts: list[str] = []
    # accepting
    if hasattr(DFA, "F"):
        try:
            if int(DFA.F) == int(q):
                name_parts.append("F")
        except Exception:
            pass
    # initial(s) (Spot uses SO; some code uses S0)
    for attr in ("SO", "S0", "initial", "start"):
        if hasattr(DFA, attr):
            try:
                if int(q) in _as_iter(getattr(DFA, attr)):
                    name_parts.append("S0")
                    break
            except Exception:
                pass
    if not name_parts:
        return f"q{q}"
    return ",".join(name_parts)


def plot_tree_layered(
    obj_or_tree: Union[nx.DiGraph, object],
    DFA: Optional[object] = None,
    *,
    root: int = 0,
    figsize: Tuple[float, float] = (12, 8),
    node_size: int = 2200,
    node_color: str = "#DCE6FF",  # base color for "other" states
    edge_color: str = "#334",
    font_size: int = 11,
    letter_font_size: Optional[int] = None,
    use_letters: bool = True,
    show_ids: bool = True,
    savepath: Optional[str] = None,
    xpad: float = 2.0,
    ystep: float = 2.6,
    seed: int = 0,  # kept for future randomness if needed
):
    """
    One-liner plot for a DFATree or a raw DiGraph in layered (top-down) style.

    • Node label shows "<id> | l=k" where l is the number of predecessors from the root,
      taken from .depths[node].
    • Node *color* encodes DFA state:
        - green filled node  -> accepting state F
        - blue filled node   -> initial state S0
        - base node_color    -> all other states
      A legend explains these colors.
    • Edge label shows DFA.act[l] when available, else the numeric l.
    """

    tree, DFA = _as_tree_and_dfa(obj_or_tree, DFA)
    pos = _layered_positions(tree, root=root, xpad=xpad, ystep=ystep)

    # --- depths: MUST come from .depths, no fallback ---
    if hasattr(obj_or_tree, "depths"):
        depths = getattr(obj_or_tree, "depths")
    elif hasattr(tree, "depths"):
        depths = getattr(tree, "depths")
    else:
        raise AttributeError(
            "plot_tree_layered expects a DFATree-like object or DiGraph "
            "with a 'depths' attribute mapping node -> depth."
        )

    def _get_depth(n: int) -> int:
        try:
            return int(depths[n])
        except Exception as e:
            raise KeyError(f"Depth not found for node {n}") from e

    # --- DFA state classification for coloring ---
    def _is_accepting(DFA, q: int) -> bool:
        if DFA is None or not hasattr(DFA, "F"):
            return False
        try:
            return int(DFA.F) == int(q)
        except Exception:
            return False

    def _is_initial(DFA, q: int) -> bool:
        if DFA is None:
            return False
        for attr in ("SO", "S0", "initial", "start"):
            if hasattr(DFA, attr):
                try:
                    if int(q) in _as_iter(getattr(DFA, attr)):
                        return True
                except Exception:
                    pass
        return False

    # colors for DFA states
    accept_color = "#8BC34A"  # green
    init_color   = "#42A5F5"  # blue

    node_labels = {}
    node_colors = []

    has_accepting = False
    has_initial = False

    for n in tree.nodes:
        q = tree.nodes[n].get("q", "?")
        depth_val = _get_depth(n)

        # label: "<id> | l=depth" or just "l=depth"
        if show_ids:
            node_labels[n] = f"{n} | l={depth_val}"
        else:
            node_labels[n] = f"l={depth_val}"

        # color by DFA state
        if _is_accepting(DFA, q):
            node_colors.append(accept_color)
            has_accepting = True
        elif _is_initial(DFA, q):
            node_colors.append(init_color)
            has_initial = True
        else:
            node_colors.append(node_color)

    # edge labels: letters if possible
    edge_labels = {}
    for u, v, data in tree.edges(data=True):
        l = data.get("l", None)
        if use_letters and l is not None and DFA is not None and hasattr(DFA, "act"):
            try:
                edge_labels[(u, v)] = str(DFA.act[int(l)])
            except Exception:
                edge_labels[(u, v)] = str(l)
        else:
            edge_labels[(u, v)] = "" if l is None else str(l)

    if letter_font_size is None:
        letter_font_size = max(9, font_size - 1)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(
        tree,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_colors,
        edgecolors=edge_color,
        linewidths=1.2,
    )
    nx.draw_networkx_edges(
        tree,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        width=1.5,
        edge_color=edge_color,
    )
    nx.draw_networkx_labels(tree, pos, labels=node_labels, ax=ax, font_size=font_size)

    nx.draw_networkx_edge_labels(
        tree,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=letter_font_size,
        label_pos=0.5,
    )

    # legend for colors
    legend_handles = []
    if has_accepting:
        legend_handles.append(Patch(facecolor=accept_color, edgecolor=edge_color, label="F (accepting)"))
    if has_initial:
        legend_handles.append(Patch(facecolor=init_color, edgecolor=edge_color, label="S0 (initial)"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")

    ax.set_axis_off()
    ax.set_title("DFA Tree (layered)", fontsize=font_size + 1)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=220)
    plt.show()
    return fig, ax


def plot_tree_layered_heatnode(
    obj_or_tree: Union[nx.DiGraph, object],
    DFA: Optional[object] = None,
    node_outer_max: dict[int, float] = None,
    *,
    root: int = 0,
    figsize: Tuple[float, float] = (12, 8),
    node_size: int = 2200,
    node_color: str = "#DCE6FF",  # unused base color, kept for signature compat
    edge_color: str = "#334",
    font_size: int = 11,
    letter_font_size: Optional[int] = None,
    use_letters: bool = True,
    show_ids: bool = True,
    savepath: Optional[str] = None,
    xpad: float = 2.0,
    ystep: float = 2.6,
    seed: int = 0,
):
    """
    Layered DFA tree plot where node *color* is a heatmap based on per-node
    max(node_outer) values from compute_tv_from_tree.

    • Node color encodes heat value h_n in [0,1] = node_outer_max.get(n, 0.0).
    • Nodes with h_n == 0 are much smaller, almost invisible, and have NO label.
    • Nodes with h_n > 0 are normal-sized, vivid, and have labels.
      - Label text is white if the node color is dark, black otherwise.
    • Edge label shows DFA.act[l] when available, else the numeric l.
    """
    if node_outer_max is None:
        raise ValueError("plot_tree_layered_heatnode requires node_outer_max dict (node -> max(node_outer)).")

    tree, DFA = _as_tree_and_dfa(obj_or_tree, DFA)
    pos = _layered_positions(tree, root=root, xpad=xpad, ystep=ystep)

    # --- depths: MUST come from .depths, no fallback ---
    if hasattr(obj_or_tree, "depths"):
        depths = getattr(obj_or_tree, "depths")
    elif hasattr(tree, "depths"):
        depths = getattr(tree, "depths")
    else:
        raise AttributeError(
            "plot_tree_layered_heatnode expects a DFATree-like object or DiGraph "
            "with a 'depths' attribute mapping node -> depth."
        )

    def _get_depth(n: int) -> int:
        try:
            return int(depths[n])
        except Exception as e:
            raise KeyError(f"Depth not found for node {n}") from e

    node_list = list(tree.nodes)

    # heat values in [0,1]
    heat_by_node = {}
    for n in node_list:
        _ = _get_depth(n)  # sanity check
        h = float(node_outer_max.get(n, 0.0))
        heat_by_node[n] = float(np.clip(h, 0.0, 1.0))

    heat_vals_arr = np.array([heat_by_node[n] for n in node_list], dtype=float)
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.cm.viridis

    # ---- IMPORTANT CHANGE: zero vs positive classification ----
    zero_nodes = [n for n in node_list if heat_by_node[n] == 0.0]
    pos_nodes  = [n for n in node_list if heat_by_node[n] >  0.0]
    zero_vals  = [heat_by_node[n] for n in zero_nodes]  # all 0.0
    pos_vals   = [heat_by_node[n] for n in pos_nodes]

    # colors for zero-heat nodes: faint & tiny
    zero_colors = []
    for v in zero_vals:
        r, g, b, a = cmap(norm(v))
        a = 0.04  # very faint
        zero_colors.append((r, g, b, a))

    # colors for positive-heat nodes: vivid
    pos_colors = []
    for v in pos_vals:
        r, g, b, a = cmap(norm(v))
        pos_colors.append((r, g, b, a))

    # labels ONLY for positive nodes
    node_labels = {}
    for n in pos_nodes:
        depth_val = _get_depth(n)
        if show_ids:
            node_labels[n] = f"{n} | l={depth_val}"
        else:
            node_labels[n] = f"l={depth_val}"

    # edge labels: letters if possible
    edge_labels = {}
    for u, v, data in tree.edges(data=True):
        l = data.get("l", None)
        if use_letters and l is not None and DFA is not None and hasattr(DFA, "act"):
            try:
                edge_labels[(u, v)] = str(DFA.act[int(l)])
            except Exception:
                edge_labels[(u, v)] = str(l)
        else:
            edge_labels[(u, v)] = "" if l is None else str(l)

    if letter_font_size is None:
        letter_font_size = max(9, font_size - 1)

    fig, ax = plt.subplots(figsize=figsize)

    # very small, faint nodes for maxvalue == 0 (no labels)
    if zero_nodes:
        nx.draw_networkx_nodes(
            tree,
            pos,
            nodelist=zero_nodes,
            ax=ax,
            node_size=node_size * 0.25,
            node_color=zero_colors,
            edgecolors="none",
            linewidths=0.0,
        )

    # normal nodes for maxvalue > 0 (with labels)
    if pos_nodes:
        nx.draw_networkx_nodes(
            tree,
            pos,
            nodelist=pos_nodes,
            ax=ax,
            node_size=node_size,
            node_color=pos_colors,
            edgecolors=edge_color,
            linewidths=1.2,
        )

    nx.draw_networkx_edges(
        tree,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        width=1.5,
        edge_color=edge_color,
    )

    # draw labels only for pos_nodes, with color depending on node brightness
    if node_labels:
        for n, label in node_labels.items():
            x, y = pos[n]
            v = heat_by_node[n]
            r, g, b, a = cmap(norm(v))
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if brightness < 0.5 else "black"
            ax.text(
                x,
                y,
                label,
                fontsize=font_size,
                ha="center",
                va="center",
                color=text_color,
            )

    nx.draw_networkx_edge_labels(
        tree,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=letter_font_size,
        label_pos=0.5,
    )

    # colorbar for heat (0–1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("max node_outer per node (0–1)")

    ax.set_axis_off()
    ax.set_title("DFA Tree (layered, node heatmap)", fontsize=font_size + 1)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=220)
    plt.show()
    return fig, ax







