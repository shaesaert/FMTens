import numpy as np

def ensure_transitions_matrix_inplace(DFA):
    """
    If the DFA has no `trans` matrix, build it from `DFA.transitions` or `DFA.graph`:
      - `DFA.S` must already be 1-based and consecutive.
      - If `DFA.act` exists, use its order for the columns; otherwise, collect labels
        in order of appearance and write them to `DFA.act`.
      - `trans` has shape (|S|, |act|). Entries are 1-based target states;
        0 means there is no transition for that label.
    """
    # --- Read state set; must be 1-based consecutive ---
    S = list(np.asarray(DFA.S).ravel())
    nS = len(S)
    if min(S) != 1 or max(S) != nS or len(set(S)) != nS:
        raise ValueError(f"DFA.S must be 1..{nS} (1-based consecutive). Got: {S}")

    # --- Collect edges (u, v, label) ---
    edges = []
    if hasattr(DFA, "transitions") and DFA.transitions is not None:
        for (u, v, data) in DFA.transitions:
            lab = data.get("label", data.get("condition", "1"))
            edges.append((int(u), int(v), str(lab)))
    elif hasattr(DFA, "graph") and getattr(DFA, "graph") is not None:
        for (u, v, data) in DFA.graph.edges(data=True):
            lab = data.get("label", data.get("condition", "1"))
            edges.append((int(u), int(v), str(lab)))
    elif hasattr(DFA, "trans"):
        # Already has `trans`: nothing to do
        return DFA
    else:
        raise ValueError("DFA has neither 'trans' nor 'transitions'/'graph'.")

    # If any edge looks 0-based, shift everything by +1 to make it 1-based
    if any(u == 0 or v == 0 for (u, v, _) in edges):
        edges = [(u + 1, v + 1, lab) for (u, v, lab) in edges]

    # --- Alphabet order (`act`) ---
    if hasattr(DFA, "act") and getattr(DFA, "act") not in (None, [], ()):
        act = list(DFA.act)
    else:
        seen = []
        for _, _, lab in edges:
            if lab not in seen:
                seen.append(lab)
        act = seen
        DFA.act = act

    m = len(act)
    lab2idx = {lab: i for i, lab in enumerate(act)}

    # --- Build the `trans` matrix; 0 means "no transition for this label" ---
    trans = np.zeros((nS, m), dtype=int)
    for (u, v, lab) in edges:
        ui = u - 1  # row index
        li = lab2idx.get(lab)
        if li is None:
            # Label not in `act` (should be rare unless `act` was modified externally)
            act.append(lab)
            lab2idx[lab] = li = len(act) - 1
            # Extend the columns of `trans`
            trans = np.pad(trans, ((0, 0), (0, 1)), mode="constant", constant_values=0)
        trans[ui, li] = int(v)

    DFA.trans = trans
    return DFA
