# dfa_tools.py
# Base-agnostic DFA utilities + one-call orchestrator.
# - Choose 0- or 1-based state indexing (index_base=0 or 1).
# - For 0-based, the "no transition" sentinel is -1; for 1-based it's 0.
# - Optionally remove q_f -> q_f self-loops from edges.
# - Optionally rewrite labels (robust to whitespace/parentheses, AND/NOT).
# - Optionally reorder DFA.act and reorder columns of DFA.trans to match.

from __future__ import annotations
from collections import OrderedDict
from typing import List, Tuple, Optional
import numpy as np

# ---------------------------- helpers ----------------------------

def _to_int_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in np.asarray(x).ravel().tolist()]
    return [int(x)]

def _to_single_int(x, name, pick_first=True):
    vals = _to_int_list(x)
    if not vals:
        raise ValueError(f"{name} is empty")
    if len(vals) > 1:
        if not pick_first:
            raise ValueError(f"{name} has multiple values: {vals}")
        print(f"[normalize] {name} has multiple values {vals}; using {vals[0]}.")
    return int(vals[0])

# -------------------- state-base normalization -------------------

def normalize_dfa_to_base_inplace(
    DFA,
    base: int,
    pick_first_accepting: bool = True,
    missing_value: Optional[int] = None,
):
    """
    Shift DFA state numbering to 0- or 1-based *in place*.
    - base = 0 -> states 0..N-1, default missing_value = -1
    - base = 1 -> states 1..N,   default missing_value = 0

    Shifts: S, F, sink, transitions (edge list), graph edges, and adjusts
    DFA.trans entries (including converting old missing sentinel to new one).
    """
    if base not in (0, 1):
        raise ValueError("base must be 0 or 1")
    if missing_value is None:
        missing_value = -1 if base == 0 else 0

    # states
    S = _to_int_list(DFA.S)
    if not S:
        raise ValueError("DFA.S is empty")

    is_zero = (min(S) == 0 and max(S) == len(S) - 1 and len(set(S)) == len(S))
    is_one  = (min(S) == 1 and max(S) == len(S)     and len(set(S)) == len(S))
    if not (is_zero or is_one):
        raise ValueError(f"DFA.S must be consecutive ints. Got: {S}")
    cur_base = 0 if is_zero else 1
    shift = base - cur_base  # +1 if 0->1, -1 if 1->0, 0 if already correct

    DFA.S = [s + shift for s in S]

    # F / sink
    if hasattr(DFA, "F"):
        F = _to_single_int(DFA.F, "DFA.F", pick_first=pick_first_accepting)
        DFA.F = F + shift
    if hasattr(DFA, "sink") and getattr(DFA, "sink", None) is not None:
        DFA.sink = int(DFA.sink) + shift

    # trans matrix (convert missing sentinel, then shift)
    if hasattr(DFA, "trans") and isinstance(DFA.trans, (list, tuple, np.ndarray)):
        T = np.asarray(DFA.trans)
        old_missing = 0 if cur_base == 1 else -1
        T = np.where(T == old_missing, np.nan, T)  # mark missing
        T = (T + shift).astype(float)              # shift valid targets
        T = np.where(np.isnan(T), missing_value, T).astype(int)
        DFA.trans = T

    # transitions edge list
    if hasattr(DFA, "transitions") and getattr(DFA, "transitions", None) is not None:
        DFA.transitions = [(int(u) + shift, int(v) + shift, dict(data))
                           for (u, v, data) in list(DFA.transitions)]

    # graph (networkx)
    if hasattr(DFA, "graph") and getattr(DFA, "graph", None) is not None and shift != 0:
        G = DFA.graph
        edges = list(G.edges(data=True))
        G.clear()
        for (u, v, data) in edges:
            G.add_edge(int(u) + shift, int(v) + shift, **data)

    DFA._missing_value = missing_value
    return DFA

def dfa_to_base_copy(DFA, base: int, missing_value: Optional[int] = None):
    """Shallow copy + normalize to base (0 or 1) without touching the original."""
    import copy
    C = copy.copy(DFA)
    if hasattr(DFA, "S"):      C.S = list(DFA.S)
    if hasattr(DFA, "act"):    C.act = list(getattr(DFA, "act", []))
    if hasattr(DFA, "AP"):     C.AP = list(getattr(DFA, "AP", []))
    if hasattr(DFA, "trans"):  C.trans = np.array(DFA.trans, copy=True)
    if hasattr(DFA, "transitions") and DFA.transitions is not None:
        C.transitions = [(u, v, dict(d)) for (u, v, d) in DFA.transitions]
    if hasattr(DFA, "graph") and getattr(DFA, "graph", None) is not None:
        C.graph = DFA.graph.copy()
    return normalize_dfa_to_base_inplace(C, base=base, missing_value=missing_value)

# --------------- build/ensure transitions matrix ----------------

def ensure_transitions_matrix_inplace(DFA, missing_value: Optional[int] = None):
    """
    If DFA.trans is absent, build it from DFA.transitions or DFA.graph,
    using the *current* base of DFA.S.

    For 0-based states: missing_value defaults to -1.
    For 1-based states: missing_value defaults to 0.

    Columns of trans follow DFA.act; if DFA.act is missing, it is inferred
    from edges in their encounter order.
    """
    S = list(np.asarray(DFA.S).ravel())
    nS = len(S)
    if not nS:
        raise ValueError("DFA.S is empty")
    base = 0 if min(S) == 0 else 1
    if missing_value is None:
        missing_value = -1 if base == 0 else 0

    # already present
    if hasattr(DFA, "trans") and getattr(DFA, "trans", None) is not None:
        return DFA

    # collect edges (assume they already match the chosen base)
    edges = []
    if hasattr(DFA, "transitions") and getattr(DFA, "transitions", None) is not None:
        for (u, v, data) in DFA.transitions:
            lab = data.get("label", data.get("condition", "1"))
            edges.append((int(u), int(v), str(lab)))
    elif hasattr(DFA, "graph") and getattr(DFA, "graph", None) is not None:
        for (u, v, data) in DFA.graph.edges(data=True):
            lab = data.get("label", data.get("condition", "1"))
            edges.append((int(u), int(v), str(lab)))
    else:
        raise ValueError("DFA has neither 'trans' nor 'transitions'/'graph'.")

    # act order
    if hasattr(DFA, "act") and getattr(DFA, "act", None):
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
    T = np.full((nS, m), missing_value, dtype=int)

    for (u, v, lab) in edges:
        ui = int(u) if base == 0 else (int(u) - 1)
        vi = int(v) if base == 0 else (int(v))      # store as 1..N if base==1
        li = lab2idx.get(lab)
        if li is None:
            act.append(lab)
            lab2idx[lab] = li = len(act) - 1
            T = np.pad(T, ((0, 0), (0, 1)), mode="constant",
                       constant_values=missing_value)
        T[ui, li] = vi if base == 0 else vi  # already correct for chosen base

    DFA.trans = T
    DFA.act = act
    DFA._missing_value = missing_value
    return DFA

# ----------------- remove accepting self-loop -------------------

def dfa_remove_qf_self_loops_inplace(DFA):
    """Remove edges q_f -> q_f from transitions/graph. Leaves trans to caller."""
    F = _to_single_int(DFA.F, "DFA.F")
    if hasattr(DFA, "transitions") and getattr(DFA, "transitions", None) is not None:
        DFA.transitions = [(u, v, d) for (u, v, d) in list(DFA.transitions)
                           if not (int(u) == F and int(v) == F)]
    if hasattr(DFA, "graph") and getattr(DFA, "graph", None) is not None:
        to_remove = [(u, v) for (u, v) in DFA.graph.edges()
                     if int(u) == F and int(v) == F]
        DFA.graph.remove_edges_from(to_remove)
    # If you want DFA.trans to reflect this, call ensure_transitions_matrix_inplace() afterwards.

# ------------------------ label utilities -----------------------

import re

def _canon_label(s: str) -> str:
    # normalize: remove spaces/parentheses, AND->&, NOT->!, keep only !p\d parts
    s = (str(s)
         .replace("AND","&").replace("and","&")
         .replace("NOT","!").replace("not","!")
         .replace(" ", "").replace("(", "").replace(")", ""))
    if s in ("", "1"):
        return "1"
    toks = []
    for t in s.split("&"):
        if not t:
            continue
        m = re.fullmatch(r'!?p\d+', t.lower())
        if m:
            toks.append(m.group(0))
    if not toks:
        return "1"
    def key(tok):
        neg = tok.startswith("!")
        idx = int(tok[2:] if neg else tok[1:])
        return (idx, 1 if neg else 0)
    toks = sorted(set(toks), key=key)
    return "&".join(toks)

def rewrite_labels_inplace(DFA, replacements_pretty: dict):
    """
    replacements_pretty: dict[str -> str]
      keys: patterns to match (spacing/parentheses flexible)
      values: the **pretty** strings to write back exactly
    """
    if not replacements_pretty:
        return 0

    repl_canon = {_canon_label(k): v for k, v in replacements_pretty.items()}
    def _maybe(lbl: str) -> str:
        c = _canon_label(lbl)
        if c in repl_canon:
            # keep the pretty value as provided
            for k, v in replacements_pretty.items():
                if _canon_label(k) == c:
                    return v
        return lbl

    changed = 0
    if hasattr(DFA, "transitions") and getattr(DFA, "transitions", None) is not None:
        new_tr = []
        for (u, v, data) in list(DFA.transitions):
            lab = data.get("label", data.get("condition", "1"))
            new_lab = _maybe(lab)
            if new_lab != lab:
                changed += 1
            d2 = dict(data)
            d2["label"] = new_lab
            d2["condition"] = new_lab
            new_tr.append((u, v, d2))
        DFA.transitions = new_tr

    if hasattr(DFA, "graph") and getattr(DFA, "graph", None) is not None:
        for (_, _, data) in list(DFA.graph.edges(data=True)):
            lab = data.get("label", data.get("condition", "1"))
            new_lab = _maybe(lab)
            if new_lab != lab:
                changed += 1
            data["label"] = new_lab
            data["condition"] = new_lab

    # refresh DFA.act from edges (preserve encounter order)
    seen = OrderedDict()
    if hasattr(DFA, "transitions") and getattr(DFA, "transitions", None) is not None:
        for (_, _, data) in DFA.transitions:
            lab = data.get("label", data.get("condition", "1"))
            seen[_canon_label(lab)] = lab
    elif hasattr(DFA, "graph") and getattr(DFA, "graph", None) is not None:
        for (_, _, data) in DFA.graph.edges(data=True):
            lab = data.get("label", data.get("condition", "1"))
            seen[_canon_label(lab)] = lab
    if seen:
        DFA.act = list(seen.values())
    return changed

def _collect_letters(DFA) -> List[str]:
    if hasattr(DFA, "act") and getattr(DFA, "act", None):
        return list(DFA.act)
    if hasattr(DFA, "transitions") and getattr(DFA, "transitions", None) is not None:
        seen = []
        for (_, _, data) in DFA.transitions:
            lab = data.get("label", data.get("condition", "1"))
            if lab not in seen:
                seen.append(lab)
        return seen
    if hasattr(DFA, "graph") and getattr(DFA, "graph", None) is not None:
        seen = []
        for (_, _, data) in DFA.graph.edges(data=True):
            lab = data.get("label", data.get("condition", "1"))
            if lab not in seen:
                seen.append(lab)
        return seen
    if hasattr(DFA, "trans") and getattr(DFA, "trans", None) is not None:
        m = int(np.asarray(DFA.trans).shape[1])
        return [f"l{i+1}" for i in range(m)]
    return []

def _reorder_letters_and_trans_inplace(DFA, desired_order: List[str]) -> List[str]:
    """Reorder DFA.act to desired_order (present-first) and reorder DFA.trans columns to match."""
    letters = _collect_letters(DFA)
    if not letters:
        return []
    new_letters = [s for s in desired_order if s in letters] + [s for s in letters if s not in desired_order]
    if hasattr(DFA, "trans") and getattr(DFA, "trans", None) is not None:
        old_idx = {lab: i for i, lab in enumerate(letters)}
        cols = [old_idx[lab] for lab in new_letters]
        DFA.trans = np.asarray(DFA.trans)[:, cols]
    DFA.act = new_letters
    return new_letters

# ------------------- one-call orchestrator ----------------------

def dfa_manipulation(
    DFA,
    *,
    index_base: int = 1,              # 0 or 1
    ensure_transitions: bool = True,
    remove_qf_self_loop: bool = False,
    replacements: dict | None = None, # None or {}
    desired_order: List[str] | None = None,
    verbose: bool = True,
):
    """
    One-line DFA manipulation. Returns (DFA, letters).

    Steps:
      1) normalize states to chosen base (0 or 1). Missing sentinel becomes
         -1 for base=0, 0 for base=1.
      2) [optional] remove q_f -> q_f self-loops from edges.
      3) [optional] label rewrites via `replacements` dict.
      4) [optional] ensure transitions matrix exists (built from edges).
      5) collect letters; [optional] reorder to `desired_order` (and trans columns).

    Usage (no replacements):
        DFA0, letters = dfa_manipulation(DFA, index_base=0,
                                         replacements=None, desired_order=None)

    """
    # 1) base normalization
    normalize_dfa_to_base_inplace(DFA, base=index_base)

    # 2) remove qf self-loop (on edges)
    if remove_qf_self_loop:
        dfa_remove_qf_self_loops_inplace(DFA)

    # 3) label rewrites (on edges)
    if replacements:
        changed = rewrite_labels_inplace(DFA, replacements)
        if verbose and changed:
            print(f"[dfa] label replacements applied: {changed}")

    # 4) ensure transitions matrix from edges (respect chosen base)
    if ensure_transitions:
        ensure_transitions_matrix_inplace(DFA)

    # 5) letters & optional reorder
    letters = _collect_letters(DFA)
    if desired_order:
        letters = _reorder_letters_and_trans_inplace(DFA, desired_order)

    if verbose:
        miss = getattr(DFA, "_missing_value", (-1 if index_base == 0 else 0))
        print(f"[dfa_manipulation] base={index_base}  |S|={len(DFA.S)}  |act|={len(letters)}  "
              f"missing={miss}  letters={letters}")
    return DFA, letters
