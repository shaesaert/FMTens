import numpy as np


def dfa_remove_qf_self_loops_inplace(dfa):
    """Remove qf→qf self-loop edges in place;
    only modify transitions/graph, leave dfa.trans unchanged
    (to avoid SWIG deep-copy issues)."""
    F = int(np.asarray(dfa.F).ravel()[0])
    if hasattr(dfa, "transitions") and dfa.transitions is not None:
        dfa.transitions = [(u, v, d) for (u, v, d) in list(dfa.transitions)
                           if not (int(u) == F and int(v) == F)]
    if hasattr(dfa, "graph"):
        to_remove = []
        for u, v in dfa.graph.edges():
            if int(u) == F and int(v) == F:
                to_remove.append((u, v))
        for u, v in to_remove:
            dfa.graph.remove_edge(u, v)
