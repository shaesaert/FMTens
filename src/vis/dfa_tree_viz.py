"""
DFA-tree visualisation helpers.

Two public functions:

    - :func:`plot_tree_layered` — top-down layered tree plot with node
      labels (``q_f`` at the root, ``q_0`` elsewhere). Closes the figure
      before returning; intended for save-to-file use.
    - :func:`plot_tree_layered_heatnode` — same layout, but node colour
      encodes a per-node heat value (max ``node_outer`` from
      :func:`compute_tv_from_tree`). Calls ``plt.show()`` before
      returning; intended for interactive display.

Both accept either a ``DFATree`` instance or a bare
``networkx.DiGraph`` plus the ``DFA`` separately.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# === input normalisation ===================================================

def _as_tree_and_dfa(
    obj_or_tree: Union[nx.DiGraph, object],
    dfa: Optional[object],
) -> tuple[nx.DiGraph, Optional[object]]:
    """
    Accept either a DFATree-like object (with ``.tree`` and ``.DFA``)
    or a bare ``nx.DiGraph`` plus a separate ``DFA``. Returns the
    underlying ``DiGraph`` and the ``DFA``.
    """
    if isinstance(obj_or_tree, nx.DiGraph):
        if dfa is None:
            raise ValueError(
                "When passing a DiGraph directly, you must also pass DFA="
            )
        return obj_or_tree, dfa
    if hasattr(obj_or_tree, "tree"):
        return obj_or_tree.tree, (dfa or getattr(obj_or_tree, "DFA", None))
    raise TypeError(
        "First arg must be a networkx.DiGraph or a DFATree-like object."
    )


def _as_iter(v) -> list[int]:
    """Coerce a scalar / list / tuple / set to ``list[int]``; ``None`` -> ``[]``."""
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [int(x) for x in v]
    return [int(v)]


def _dfa_state_name(DFA, q: int) -> str:
    """
    Human-readable name for a DFA state ``q``: ``"F"`` for accepting,
    ``"S0"`` for initial, otherwise ``f"q{q}"``.
    """
    parts: list[str] = []
    if hasattr(DFA, "F"):
        try:
            if int(DFA.F) == int(q):
                parts.append("F")
        except Exception:
            pass
    for attr in ("SO", "S0", "initial", "start"):
        if hasattr(DFA, attr):
            try:
                if int(q) in _as_iter(getattr(DFA, attr)):
                    parts.append("S0")
                    break
            except Exception:
                pass
    return ",".join(parts) if parts else f"q{q}"


# === layout ================================================================

def _layered_positions(
    G: nx.DiGraph,
    root: int = 0,
    xpad: float = 2.0,
    ystep: float = 2.6,
) -> dict:
    """
    Top-down layered layout with fixed horizontal spacing ``xpad``
    within each layer and vertical spacing ``ystep`` between layers.

    Layer 0 contains ``root``; subsequent layers are determined by BFS.
    Nodes unreachable from ``root`` are placed one layer below the
    deepest reachable layer.
    """
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

    pos: dict = {}
    for d in sorted(layers):
        nodes = layers[d]
        k = len(nodes)
        # Indices 0..k-1 shifted to be centred around 0.
        indices = np.arange(k)
        xs = (indices - (k - 1) / 2.0) * xpad
        for x, n in zip(xs, nodes):
            pos[n] = (float(x), -d * ystep)

    max_depth = max(layers) if layers else 0
    for n in G.nodes:
        if n not in pos:
            pos[n] = (0.0, -(max_depth + 1) * ystep)

    return pos


# === plotting ==============================================================

def plot_tree_layered(
    obj_or_tree: Union[nx.DiGraph, object],
    DFA: Optional[object] = None,
    *,
    root: int = 0,
    figsize: Tuple[float, float] = (12, 8),
    node_size: int = 2200,
    node_color: str = "#D8DEE8",
    edge_color: str = "black",
    font_size: int = 11,
    letter_font_size: Optional[int] = None,
    use_letters: bool = True,
    show_ids: bool = True,
    savepath: Optional[str] = None,
    xpad: float = 2.0,
    ystep: float = 2.6,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    seed: int = 0,
):
    """
    Layered top-down plot of a DFA tree.

    Node labels:
        - root node  -> ``q_f`` (LaTeX)
        - other nodes -> ``q_0`` (LaTeX)

    Node colours: classified as accepting (``F``), initial (``S0``), or
    other. The three colour constants in the body of this function are
    currently all set to the same light gray; change ``accept_color``
    and ``init_color`` inline to produce a colour-coded plot.

    Edge labels: ``DFA.act[l]`` when available, else the numeric ``l``.

    Parameters
    ----------
    obj_or_tree : DFATree-like object or networkx.DiGraph
        Source tree.
    DFA : object, optional
        Required when ``obj_or_tree`` is a bare DiGraph.
    root : int, optional
        Root node id. Default ``0``.
    figsize, node_size, node_color, edge_color, font_size,
    letter_font_size, use_letters, show_ids, savepath, xpad, ystep,
    xlim, ylim, seed :
        Cosmetic / layout knobs. ``show_ids`` and ``seed`` are accepted
        but currently unused.

    Returns
    -------
    (fig, ax) : matplotlib figure and axis
        The figure is closed before returning; the file saved via
        ``savepath`` is the persistent artefact.
    """
    tree, DFA = _as_tree_and_dfa(obj_or_tree, DFA)
    pos = _layered_positions(tree, root=root, xpad=xpad, ystep=ystep)

    # Depths must come from a .depths attribute on the input.
    if hasattr(obj_or_tree, "depths"):
        depths = obj_or_tree.depths
    elif hasattr(tree, "depths"):
        depths = tree.depths
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

    def _is_accepting(q) -> bool:
        if DFA is None or not hasattr(DFA, "F"):
            return False
        try:
            return int(DFA.F) == int(q)
        except Exception:
            return False

    def _is_initial(q) -> bool:
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

    # All three colours intentionally identical for a uniform look.
    # Change accept_color / init_color inline for a colour-coded plot.
    accept_color = "#D8DEE8"
    init_color   = "#D8DEE8"

    node_labels = {}
    node_colors = []
    for n in tree.nodes:
        q = tree.nodes[n].get("q", "?")
        _get_depth(n)  # validate that depth exists

        node_labels[n] = r"$q_f$" if n == root else r"$q_0$"

        if _is_accepting(q):
            node_colors.append(accept_color)
        elif _is_initial(q):
            node_colors.append(init_color)
        else:
            node_colors.append(node_color)

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
        tree, pos, ax=ax,
        node_size=node_size, node_color=node_colors,
        edgecolors=edge_color, linewidths=1.2,
    )
    nx.draw_networkx_edges(
        tree, pos, ax=ax,
        arrows=True, arrowstyle="-|>",
        width=1.5, edge_color=edge_color,
    )
    nx.draw_networkx_labels(tree, pos, labels=node_labels, ax=ax, font_size=font_size)
    nx.draw_networkx_edge_labels(
        tree, pos, edge_labels=edge_labels, ax=ax,
        font_size=letter_font_size, label_pos=0.5,
    )

    xs, ys = zip(*pos.values())
    if xlim is None:
        xlim = (min(xs) - xpad, max(xs) + xpad)
    if ylim is None:
        ylim = (min(ys) - ystep, max(ys) + ystep)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=220, transparent=True)

    plt.close(fig)
    return fig, ax


def plot_tree_layered_heatnode(
    obj_or_tree: Union[nx.DiGraph, object],
    DFA: Optional[object] = None,
    node_outer_max: Optional[dict] = None,
    *,
    root: int = 0,
    figsize: Tuple[float, float] = (12, 8),
    node_size: int = 2200,
    node_color: str = "#DCE6FF",
    edge_color: str = "black",
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
    Layered DFA-tree plot with per-node heatmap colouring.

    Node colour encodes a heat value ``h_n in [0, 1] = node_outer_max[n]``
    via the ``viridis`` colormap. Nodes with ``h_n == 0`` are rendered
    very small and faint, with no label. Nodes with ``h_n > 0`` are
    normal-sized, vivid, and labelled ``"<id> | l=<depth>"`` (or just
    ``"l=<depth>"`` if ``show_ids=False``). Label text is white on dark
    backgrounds, black on light.

    Edge labels: ``DFA.act[l]`` when available, else the numeric ``l``.

    Parameters
    ----------
    obj_or_tree : DFATree-like object or networkx.DiGraph
        Source tree.
    DFA : object, optional
        Required when ``obj_or_tree`` is a bare DiGraph.
    node_outer_max : dict[int, float]
        Mapping node id -> max-of-node-outer-product in ``[0, 1]``.
        Required; produced by :func:`compute_tv_from_tree`.
    root, figsize, node_size, node_color, edge_color, font_size,
    letter_font_size, use_letters, show_ids, savepath, xpad, ystep,
    seed :
        Cosmetic / layout knobs. ``node_color`` is accepted for
        signature compatibility with :func:`plot_tree_layered` but
        unused here; ``seed`` is accepted but unused.

    Returns
    -------
    (fig, ax) : matplotlib figure and axis
        ``plt.show()`` is called before returning.
    """
    if node_outer_max is None:
        raise ValueError(
            "plot_tree_layered_heatnode requires node_outer_max "
            "(dict: node -> max(node_outer))."
        )

    tree, DFA = _as_tree_and_dfa(obj_or_tree, DFA)
    pos = _layered_positions(tree, root=root, xpad=xpad, ystep=ystep)

    if hasattr(obj_or_tree, "depths"):
        depths = obj_or_tree.depths
    elif hasattr(tree, "depths"):
        depths = tree.depths
    else:
        raise AttributeError(
            "plot_tree_layered_heatnode expects a DFATree-like object "
            "or DiGraph with a 'depths' attribute mapping node -> depth."
        )

    def _get_depth(n: int) -> int:
        try:
            return int(depths[n])
        except Exception as e:
            raise KeyError(f"Depth not found for node {n}") from e

    node_list = list(tree.nodes)
    heat_by_node = {}
    for n in node_list:
        _get_depth(n)
        h = float(node_outer_max.get(n, 0.0))
        heat_by_node[n] = float(np.clip(h, 0.0, 1.0))

    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.cm.viridis

    # Split nodes by heat: zero-heat are rendered small and faint with no
    # labels; positive-heat are full-size, vivid, and labelled.
    zero_nodes = [n for n in node_list if heat_by_node[n] == 0.0]
    pos_nodes  = [n for n in node_list if heat_by_node[n] >  0.0]
    zero_colors = [(*cmap(norm(0.0))[:3], 0.04) for _ in zero_nodes]
    pos_colors  = [cmap(norm(heat_by_node[n])) for n in pos_nodes]

    node_labels = {}
    for n in pos_nodes:
        depth_val = _get_depth(n)
        node_labels[n] = f"{n} | l={depth_val}" if show_ids else f"l={depth_val}"

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

    if zero_nodes:
        nx.draw_networkx_nodes(
            tree, pos, nodelist=zero_nodes, ax=ax,
            node_size=node_size * 0.25, node_color=zero_colors,
            edgecolors="none", linewidths=0.0,
        )

    if pos_nodes:
        nx.draw_networkx_nodes(
            tree, pos, nodelist=pos_nodes, ax=ax,
            node_size=node_size, node_color=pos_colors,
            edgecolors=edge_color, linewidths=1.2,
        )

    nx.draw_networkx_edges(
        tree, pos, ax=ax,
        arrows=True, arrowstyle="-|>",
        width=1.5, edge_color=edge_color,
    )

    # Draw labels only on positive-heat nodes; pick text colour by
    # background brightness.
    for n, label in node_labels.items():
        x, y = pos[n]
        r, g, b, _a = cmap(norm(heat_by_node[n]))
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "white" if brightness < 0.5 else "black"
        ax.text(x, y, label, fontsize=font_size,
                ha="center", va="center", color=text_color)

    nx.draw_networkx_edge_labels(
        tree, pos, edge_labels=edge_labels, ax=ax,
        font_size=letter_font_size, label_pos=0.5,
    )

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