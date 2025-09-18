import numpy as np

def _to_int_list(x):
    """x -> list[int] (supports scalar, list/tuple, np.ndarray). If empty/None, return an empty list."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x).ravel()
        return [int(v) for v in arr.tolist()]
    return [int(x)]

def _to_single_int(x, name, pick_first=True):
    """Convert x to a single int. If there are multiple values and pick_first=True, use the first and print a notice."""
    vals = _to_int_list(x)
    if len(vals) == 0:
        raise ValueError(f"{name} is empty")
    if len(vals) > 1:
        if not pick_first:
            raise ValueError(f"{name} has multiple values: {vals}")
        print(f"[normalize] {name} has multiple values {vals}; using the first one ({vals[0]}).")
    return int(vals[0])

def normalize_dfa_to_1based_inplace(DFA, pick_first_accepting=True):
    """
    Normalize DFA in-place to 1-based indexing:
      - S: if 0-based, add +1; if already 1-based, keep as-is
      - F/sink: convert to a single 1-based integer (if multiple, pick the first by default)
      - trans: if present, add +1 to the entire matrix
      - transitions: if present, add +1 to (u, v)
    """
    # S -> list[int]
    S = _to_int_list(DFA.S)
    if not S:
        raise ValueError("DFA.S is empty")

    # Decide whether S is consecutive 0-based or 1-based
    is_zero_based = (min(S) == 0 and max(S) == len(S) - 1 and len(set(S)) == len(S))
    is_one_based  = (min(S) == 1 and max(S) == len(S)     and len(set(S)) == len(S))

    if not (is_zero_based or is_one_based):
        raise ValueError(f"DFA.S must be consecutive ints (0- or 1-based). Got: {S}")

    shift = 1 if is_zero_based else 0

    # S
    DFA.S = [s + shift for s in S]

    # F / sink
    if hasattr(DFA, "F"):
        F = _to_single_int(DFA.F, "DFA.F", pick_first=pick_first_accepting)
        DFA.F = F + shift
    if hasattr(DFA, "sink") and getattr(DFA, "sink") is not None:
        sink = _to_single_int(DFA.sink, "DFA.sink", pick_first=True)
        DFA.sink = sink + shift

    # trans: "row = source state, col = letter, value = target state (integer)"
    if hasattr(DFA, "trans") and isinstance(DFA.trans, (list, tuple, np.ndarray)):
        T = np.asarray(DFA.trans)
        # Only add +shift for integer-like target states
        if np.issubdtype(T.dtype, np.integer) or np.issubdtype(T.dtype, np.floating):
            DFA.trans = (T + shift).astype(int)
        else:
            # If it's some other type (rare), try element-wise conversion and shifting
            T2 = np.array([[int(v) + shift for v in row] for row in T], dtype=int)
            DFA.trans = T2

    # transitions: edge list (u, v, data)
    if hasattr(DFA, "transitions") and DFA.transitions is not None:
        new_tr = []
        for (u, v, data) in DFA.transitions:
            new_tr.append((int(u) + shift, int(v) + shift, dict(data)))
        DFA.transitions = new_tr

    # graph: if you store edges in a networkx graph, shift them as well (usually not needed)
    if hasattr(DFA, "graph") and getattr(DFA, "graph") is not None and shift != 0:
        G = DFA.graph
        edges = list(G.edges(data=True))
        G.clear()
        for (u, v, data) in edges:
            G.add_edge(int(u) + shift, int(v) + shift, **data)

    # act (alphabet) is kept as-is
    return DFA
