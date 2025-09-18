# ===== place near imports =====
import networkx as nx
import matplotlib.pyplot as plt

# ===== place in main (define two helper functions) =====
def summarize_tree(tree, letters=None):
    """Print a summary of the tree: total nodes/leaves, each node's DFA mode q,
    and all edges with their labels."""
    G = tree.tree
    print(f"[tree] #nodes={G.number_of_nodes()}  #leafs={len(tree.leafs)}")
    print("nodes (id : q):")
    for n in sorted(G.nodes):
        print(f"  {n} : {tree.Lq(n)}")
    print("edges (u -> v [label]):")
    for u, v, data in G.edges(data=True):
        l1 = int(data.get("l", 0))  # 1-based label index
        if letters and 1 <= l1 <= len(letters):
            print(f"  {u} -> {v}  [{l1}:{letters[l1-1]}]")
        else:
            print(f"  {u} -> {v}  [l={l1}]")


def _layered_pos(G, root=1, xgap=1.6, ygap=1.6):
    from collections import deque, defaultdict
    depth = {root: 0}
    levels = defaultdict(list)
    q = deque([root])
    while q:
        u = q.popleft()
        d = depth[u]
        levels[d].append(u)
        for v in G.successors(u):
            if v not in depth:
                depth[v] = d + 1
                q.append(v)

    pos = {}
    for d in sorted(levels):
        nodes = levels[d]
        for i, n in enumerate(nodes):
            x = (i - (len(nodes) - 1)/2.0) * xgap
            y = -d * ygap
            pos[n] = (x, y)
    return pos

def _node_role(tree, n: int) -> str:
    q = tree.Lq(n)
    if q == getattr(tree, "F1", None):
        return "qf"
    if getattr(tree, "sink1", None) is not None and q == tree.sink1:
        return "qs"
    return "q0"

def draw_tree(tree, letters=None, label='idrole', show=True, save_path=None):
    """
    label:
      - 'id'      -> show node id only
      - 'q'       -> show DFA state q only
      - 'both'    -> 'n:q'
      - 'idrole'  -> 'n:qf/q0/qs'  ⬅️ the effect you want (default)
    """
    G = tree.tree

    # Prefer graphviz layout; fall back to a simple layered layout
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog='dot')
    except Exception:
        pos = _layered_pos(G, root=1)

    # Node labels
    if label == 'id':
        node_labels = {n: str(n) for n in G.nodes}
    elif label == 'q':
        node_labels = {n: str(tree.Lq(n)) for n in G.nodes}
    elif label == 'both':
        node_labels = {n: f"{n}:{tree.Lq(n)}" for n in G.nodes}
    else:  # 'idrole'
        node_labels = {n: f"{n}:{_node_role(tree, n)}" for n in G.nodes}

    # Optional: color nodes by role (make qf brighter)
    colors = []
    for n in G.nodes:
        role = _node_role(tree, n)
        if role == "qf":
            colors.append("#3cb371")  # green
        elif role == "qs":
            colors.append("#ff7f50")  # orange
        else:
            colors.append("#1f77b4")  # blue

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=700)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", width=1.2)

    # Edge labels (letters)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        l1 = int(data.get("l", 0))
        if letters and 1 <= l1 <= len(letters):
            edge_labels[(u, v)] = letters[l1-1]
        else:
            edge_labels[(u, v)] = str(l1)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[tree] saved figure -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
