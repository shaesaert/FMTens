# src/vis/dfa_tree_viz.py
from __future__ import annotations
from collections import defaultdict, deque
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


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
    node_color: str = "#DCE6FF",
    edge_color: str = "#334",
    font_size: int = 11,
    letter_font_size: Optional[int] = None,
    use_letters: bool = True,
    show_ids: bool = True,
    savepath: Optional[str] = None,
    xpad: float = 2.0,
    ystep: float = 2.6,
    seed: int = 0,   # kept for future randomness if needed
):
    """
    One-liner plot for a DFATree or a raw DiGraph in layered (top-down) style.

    • Node label shows "<id> | F" or "<id> | S0" etc.  (id part can be hidden via show_ids=False)
    • Edge label shows DFA.act[l] when available, else the numeric l.
    • Circles are larger by default (node_size).
    """
    tree, DFA = _as_tree_and_dfa(obj_or_tree, DFA)
    pos = _layered_positions(tree, root=root, xpad=xpad, ystep=ystep)

    # node labels: "<id> | <state-name>"
    node_labels = {}
    for n in tree.nodes:
        q = tree.nodes[n].get("q", "?")
        name = _dfa_state_name(DFA, int(q)) if DFA is not None else f"q{q}"
        node_labels[n] = f"{n} | {name}" if show_ids else name

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
        tree, pos, ax=ax, node_size=node_size, node_color=node_color, edgecolors=edge_color, linewidths=1.2
    )
    nx.draw_networkx_edges(tree, pos, ax=ax, arrows=True, arrowstyle="-|>", width=1.5, edge_color=edge_color)
    nx.draw_networkx_labels(tree, pos, labels=node_labels, ax=ax, font_size=font_size)

    nx.draw_networkx_edge_labels(
        tree, pos, edge_labels=edge_labels, ax=ax, font_size=letter_font_size, label_pos=0.5
    )

    ax.set_axis_off()
    ax.set_title("DFA Tree (layered)", fontsize=font_size + 1)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=220)
    plt.show()
    return fig, ax
