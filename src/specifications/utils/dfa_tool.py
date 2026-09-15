"""
DFA representation and manipulation utilities for the FMTens pipeline.

This module covers both ways the pipeline obtains a DFA, plus the
post-construction utilities used to normalise and transform it.

DFA construction
----------------
Two paths are supported:

1. Spot-based (default, Linux/macOS):

       from src.specifications.translate import translate
       DFA = translate(spec)

   This requires the Spot library (https://spot.lre.epita.fr/), which does
   not install cleanly on Windows.

2. Hand-written fallback (Windows or any Spot-less environment):

       from src.specifications.utils.dfa_tool import SimpleDFA
       DFA = SimpleDFA({...DFA dict...})

   See :data:`EXAMPLE_DFA_DATA` for the schema and an example.

DFA manipulation
----------------
The single public entry point for downstream transformations is
:func:`dfa_manipulation`, a one-call orchestrator that:

    0. Optionally applies structural overrides (AP / states / initial /
       accepting / transitions / sink) when the translated DFA is not
       tight enough for the chosen letter set.
    1. Normalises state numbering to the requested 0- or 1-based indexing.
    2. Optionally removes q_f -> q_f self-loops.
    3. Optionally rewrites edge labels via a replacement dictionary.
    4. Builds the transitions matrix from edges (if absent).
    5. Optionally removes sink-only letters (letters whose every non-
       accepting transition goes to the sink state).
    6. Collects letters; optionally reorders them to a desired order.

Conventions
-----------
- State numbering can be 0- or 1-based; the "no transition" sentinel is
  -1 for 0-based and 0 for 1-based numbering.
- Edge labels are canonicalised to a compact form (e.g. "p1 & !p2") for
  matching against caller-provided rewrites.
- All helper functions other than :func:`dfa_manipulation`, :class:`SimpleDFA`,
  and :func:`load_dfa` are private.
"""

from __future__ import annotations
from collections import OrderedDict
from typing import Any, Dict, List, Optional
import re

import numpy as np
import networkx as nx


# === DFA representation (portable fallback) ================================

#: Example DFA encoding the "eventually p1" specification (◇p1):
#: from the initial state 1, observing p1 transitions to the accepting
#: state 0 (which then self-loops on every input). Provided as a starting
#: template; users construct an analogous dict for their own specification.
#:
#: Required fields:
#:     AP              list of atomic proposition names, e.g. ["p1", "p2"]
#:     S               list of state IDs (consecutive ints, 0- or 1-based)
#:     S0              list of initial states (typically a singleton)
#:     F               accepting set in Buchi form: list of lists of states
#:     nr_states       number of states (== len(S))
#:     automaton_state initial state (typically S0[0])
#:     transitions     list of (q_from, q_to, {"label": str, "condition": str})
#:
#: Optional fields:
#:     delta           Spot-internal transition table; safe to leave as [{}]
#:     graph_data      NetworkX node-link form of the DFA graph; reconstructed
#:                     by SimpleDFA into a MultiDiGraph if present
EXAMPLE_DFA_DATA: Dict[str, Any] = {
    "AP": ["p1"],
    "S": [0, 1],
    "S0": [1],
    "F": [[0]],
    "nr_states": 2,
    "automaton_state": 1,
    "delta": [{}],
    "transitions": [
        (0, 0, {"label": "1",  "condition": "1"}),
        (1, 0, {"label": "p1", "condition": "p1"}),
    ],
    "graph_data": {
        "directed": True,
        "multigraph": True,
        "graph": {},
        "nodes": [{"id": 0}, {"id": 1}],
        "links": [
            {"label": "1",  "condition": "1",  "source": 0, "target": 0, "key": 0},
            {"label": "p1", "condition": "p1", "source": 1, "target": 0, "key": 0},
        ],
    },
}


class SimpleDFA:
    """
    Lightweight DFA wrapper exposing dict fields as object attributes.

    Provides the interface that the rest of the FMTens pipeline expects
    from a Spot-built ``automaton``, without the Spot dependency. Use this
    when running on Windows or any environment where Spot is unavailable.

    Parameters
    ----------
    data : dict
        DFA description; see :data:`EXAMPLE_DFA_DATA` for the schema.
    """

    def __init__(self, data: Dict[str, Any]):
        self.AP = data.get("AP", [])
        self.S = data.get("S", [])
        self.S0 = data.get("S0", [])
        self.F = data.get("F", [])
        self.nr_states = data.get("nr_states", 0)
        self.automaton_state = data.get("automaton_state", 0)
        self.delta = data.get("delta", None)
        self.transitions = data.get("transitions", [])

        # Reconstruct the NetworkX graph from node-link form if provided.
        graph_data = data.get("graph_data")
        if graph_data is not None:
            try:
                self.graph = nx.node_link_graph(graph_data)
            except Exception:
                self.graph = None
        else:
            self.graph = None


def load_dfa(dfa_data: Optional[Dict[str, Any]] = None) -> SimpleDFA:
    """
    Build a :class:`SimpleDFA` from a dict.

    Parameters
    ----------
    dfa_data : dict, optional
        DFA description following the schema in :data:`EXAMPLE_DFA_DATA`.
        Defaults to :data:`EXAMPLE_DFA_DATA` (the "eventually p1" automaton).

    Returns
    -------
    SimpleDFA
        Wrapper compatible with the downstream FMTens pipeline.
    """
    if dfa_data is None:
        dfa_data = EXAMPLE_DFA_DATA
    return SimpleDFA(dfa_data)


# === helpers ===============================================================

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


# === state-base normalisation ==============================================

def _normalize_dfa_to_base_inplace(
    DFA,
    base: int,
    pick_first_accepting: bool = True,
    missing_value: Optional[int] = None,
):
    """
    Shift DFA state numbering to 0- or 1-based in place.

    Parameters
    ----------
    DFA : object
        DFA with attributes S, F, optionally sink, transitions, graph, trans.
    base : int
        Target indexing base: 0 -> states 0..N-1, 1 -> states 1..N.
    pick_first_accepting : bool, optional
        When DFA.F is a list with multiple accepting states, pick the first
        if True; raise otherwise. Defaults to True.
    missing_value : int, optional
        Sentinel value for "no transition" in DFA.trans. Defaults to -1 for
        base=0, 0 for base=1.

    Side effects: shifts S, F, sink, transitions (edge list), graph edges,
    and adjusts DFA.trans entries (converting the previous missing sentinel
    to the new one).
    """
    if base not in (0, 1):
        raise ValueError("base must be 0 or 1")
    if missing_value is None:
        missing_value = -1 if base == 0 else 0

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

    if hasattr(DFA, "F"):
        F = _to_single_int(DFA.F, "DFA.F", pick_first=pick_first_accepting)
        DFA.F = F + shift
    if hasattr(DFA, "sink") and getattr(DFA, "sink", None) is not None:
        DFA.sink = int(DFA.sink) + shift

    # trans matrix: convert previous missing sentinel, then shift valid targets
    if hasattr(DFA, "trans") and isinstance(DFA.trans, (list, tuple, np.ndarray)):
        T = np.asarray(DFA.trans)
        old_missing = 0 if cur_base == 1 else -1
        T = np.where(T == old_missing, np.nan, T)
        T = (T + shift).astype(float)
        T = np.where(np.isnan(T), missing_value, T).astype(int)
        DFA.trans = T

    if hasattr(DFA, "transitions") and getattr(DFA, "transitions", None) is not None:
        DFA.transitions = [
            (int(u) + shift, int(v) + shift, dict(data))
            for (u, v, data) in list(DFA.transitions)
        ]

    if hasattr(DFA, "graph") and getattr(DFA, "graph", None) is not None and shift != 0:
        G = DFA.graph
        edges = list(G.edges(data=True))
        G.clear()
        for (u, v, data) in edges:
            G.add_edge(int(u) + shift, int(v) + shift, **data)

    DFA._missing_value = missing_value
    return DFA


# === transitions matrix construction =======================================

def _ensure_transitions_matrix_inplace(DFA, missing_value: Optional[int] = None):
    """
    Build DFA.trans from DFA.transitions or DFA.graph (if absent), respecting
    the current base of DFA.S.

    Defaults: missing_value = -1 for 0-based states, 0 for 1-based states.
    Columns of trans follow DFA.act; DFA.act is inferred from edges (in
    encounter order) if absent.
    """
    S = list(np.asarray(DFA.S).ravel())
    nS = len(S)
    if not nS:
        raise ValueError("DFA.S is empty")
    base = 0 if min(S) == 0 else 1
    if missing_value is None:
        missing_value = -1 if base == 0 else 0

    if hasattr(DFA, "trans") and getattr(DFA, "trans", None) is not None:
        return DFA

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
        vi = int(v)  # stored as-is in the chosen base (0..N-1 or 1..N)
        li = lab2idx.get(lab)
        if li is None:
            act.append(lab)
            lab2idx[lab] = li = len(act) - 1
            T = np.pad(T, ((0, 0), (0, 1)), mode="constant",
                       constant_values=missing_value)
        T[ui, li] = vi

    DFA.trans = T
    DFA.act = act
    DFA._missing_value = missing_value
    return DFA


# === self-loop removal =====================================================

def _dfa_remove_qf_self_loops_inplace(DFA):
    """
    Remove q_f -> q_f edges from DFA.transitions and DFA.graph. DFA.trans is
    not touched; call _ensure_transitions_matrix_inplace afterwards if a
    refreshed trans matrix is needed.
    """
    F = _to_single_int(DFA.F, "DFA.F")
    if hasattr(DFA, "transitions") and getattr(DFA, "transitions", None) is not None:
        DFA.transitions = [
            (u, v, d) for (u, v, d) in list(DFA.transitions)
            if not (int(u) == F and int(v) == F)
        ]
    if hasattr(DFA, "graph") and getattr(DFA, "graph", None) is not None:
        to_remove = [
            (u, v) for (u, v) in DFA.graph.edges()
            if int(u) == F and int(v) == F
        ]
        DFA.graph.remove_edges_from(to_remove)


# === label utilities =======================================================

def _canon_label(s: str) -> str:
    """
    Canonicalise an edge label to compact form (e.g. "p1 & !p2" -> "p1&!p2").

    Strips whitespace and parentheses, normalises 'AND'/'and' -> '&' and
    'NOT'/'not' -> '!', then deduplicates and sorts the literal tokens. The
    string "1" (tautology) is preserved.
    """
    s = (str(s)
         .replace("AND", "&").replace("and", "&")
         .replace("NOT", "!").replace("not", "!")
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


def _rewrite_labels_inplace(DFA, replacements_pretty: dict) -> int:
    """
    Rewrite edge labels in DFA.transitions and DFA.graph using a replacement
    dictionary.

    Keys of `replacements_pretty` may use any spacing or parenthesisation;
    they are canonicalised before matching. Values are written back verbatim
    as the "pretty" label. Returns the number of edges changed.
    """
    if not replacements_pretty:
        return 0

    repl_canon = {_canon_label(k): v for k, v in replacements_pretty.items()}

    def _maybe(lbl: str) -> str:
        c = _canon_label(lbl)
        if c in repl_canon:
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

    # refresh DFA.act from edges, preserving encounter order
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
    """
    Collect the DFA's letters in encounter order from DFA.act, DFA.transitions,
    DFA.graph, or DFA.trans (in that priority).
    """
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
    """
    Reorder DFA.act so that letters in `desired_order` come first (in the
    given order), followed by any letters not in `desired_order` (in their
    current order). DFA.trans columns are reordered to match.
    """
    letters = _collect_letters(DFA)
    if not letters:
        return []
    new_letters = (
        [s for s in desired_order if s in letters]
        + [s for s in letters if s not in desired_order]
    )
    if hasattr(DFA, "trans") and getattr(DFA, "trans", None) is not None:
        old_idx = {lab: i for i, lab in enumerate(letters)}
        cols = [old_idx[lab] for lab in new_letters]
        DFA.trans = np.asarray(DFA.trans)[:, cols]
    DFA.act = new_letters
    return new_letters


# === structural overrides ==================================================

def _apply_structural_overrides_inplace(
    DFA,
    *,
    AP=None,
    states=None,
    initial=None,
    accepting=None,
    transitions=None,
    sink=None,
):
    """
    Replace selected parts of the DFA's structure in place.

    All parameters are optional; only the ones that are not None are applied.
    Intended to run BEFORE state-base normalisation -- any 0/1-base mismatch
    introduced here is shifted by the normaliser.

    `transitions` is a list of (q_from, q_to, formula_str) triples; each is
    expanded into the dict-of-data form used by DFA.transitions, DFA.graph
    is rebuilt accordingly, and DFA.trans is invalidated so it will be
    rebuilt by _ensure_transitions_matrix_inplace.
    """
    if AP is not None:
        DFA.AP = list(AP)

    if states is not None:
        DFA.S = list(states)
        DFA.nr_states = len(DFA.S)

    if initial is not None:
        DFA.S0 = list(initial) if isinstance(initial, (list, tuple)) else [int(initial)]
        DFA.automaton_state = DFA.S0[0]

    if accepting is not None:
        DFA.F = int(accepting)

    if sink is not None:
        DFA.sink = int(sink)

    if transitions is not None:
        DFA.transitions = [
            (int(q), int(qp), {"condition": str(f), "label": str(f)})
            for (q, qp, f) in transitions
        ]
        DFA.graph = nx.MultiDiGraph()
        if hasattr(DFA, "S") and DFA.S:
            DFA.graph.add_nodes_from(DFA.S)
        else:
            inferred = set()
            for q, qp, _ in DFA.transitions:
                inferred.add(int(q))
                inferred.add(int(qp))
            DFA.graph.add_nodes_from(sorted(inferred))
        for u, v, data in DFA.transitions:
            DFA.graph.add_edge(u, v, **data)
        DFA.delta = [{}]
        # Invalidate any previously-built trans matrix so it gets rebuilt later
        DFA.trans = None


def _remove_sink_only_letters_inplace(DFA):
    """
    Remove letters (columns of DFA.trans) for which every non-accepting state
    transitions to sink. Such letters can only trap the system in sink and
    never lead to acceptance, so they don't contribute to the satisfaction
    probability.

    Requires DFA.trans to be built (call _ensure_transitions_matrix_inplace
    first) and DFA.sink to be set.

    Mirrors the MATLAB post-processing:
        tr = DFA.trans
        tr(DFA.F, :) = []
        sinktr = all(tr == DFA.sink, 1)
        DFA.trans(:, sinktr') = []
        DFA.act(sinktr') = []
    """
    if getattr(DFA, "sink", None) is None:
        return
    if getattr(DFA, "trans", None) is None:
        return

    T    = np.asarray(DFA.trans)
    F    = _to_single_int(DFA.F, "DFA.F")
    sink = int(DFA.sink)

    nS = T.shape[0]
    non_F_rows = [i for i in range(nS) if i != F]
    if not non_F_rows:
        return

    sink_cols = np.all(T[non_F_rows, :] == sink, axis=0)
    if not sink_cols.any():
        return

    keep = ~sink_cols
    DFA.trans = T[:, keep]
    if getattr(DFA, "act", None) is not None:
        DFA.act = [a for a, k in zip(DFA.act, keep) if k]


# === orchestrator (public API) =============================================

def dfa_manipulation(
    DFA,
    *,
    # Structural overrides (optional). Applied first; whatever is None is left
    # untouched. Use these when the translated DFA isn't tight enough for the
    # chosen letter set.
    AP=None,
    states=None,
    initial=None,
    accepting=None,
    transitions=None,
    sink=None,
    # Manipulation flags
    index_base: int = 1,
    ensure_transitions: bool = True,
    remove_qf_self_loop: bool = False,
    remove_sink_only_letters: bool = False,
    replacements: Optional[dict] = None,
    desired_order: Optional[List[str]] = None,
    verbose: bool = True,
):
    """
    One-call DFA manipulation orchestrator. Returns (DFA, letters).

    Steps applied in order:
        0. Structural overrides (AP / states / initial / accepting /
           transitions / sink).
        1. Normalise state numbering to `index_base` (0 or 1). Missing-
           transition sentinel becomes -1 for base=0, 0 for base=1.
        2. Remove q_f -> q_f self-loops if `remove_qf_self_loop`.
        3. Apply label `replacements` (canonical match, pretty write-back).
        4. Build the transitions matrix from edges if `ensure_transitions`.
        5. Remove sink-only letters if `remove_sink_only_letters`.
        6. Collect letters; reorder to `desired_order` if given.

    Example
    -------
        DFA0, letters = dfa_manipulation(
            DFA, index_base=0, replacements=None, desired_order=None,
        )
    """
    _apply_structural_overrides_inplace(
        DFA,
        AP=AP, states=states, initial=initial,
        accepting=accepting, transitions=transitions, sink=sink,
    )

    _normalize_dfa_to_base_inplace(DFA, base=index_base)

    if remove_qf_self_loop:
        _dfa_remove_qf_self_loops_inplace(DFA)

    if replacements:
        changed = _rewrite_labels_inplace(DFA, replacements)
        if verbose and changed:
            print(f"[dfa] label replacements applied: {changed}")

    if ensure_transitions:
        _ensure_transitions_matrix_inplace(DFA)

    if remove_sink_only_letters:
        _remove_sink_only_letters_inplace(DFA)

    letters = _collect_letters(DFA)
    if desired_order:
        letters = _reorder_letters_and_trans_inplace(DFA, desired_order)

    if verbose:
        miss = getattr(DFA, "_missing_value", (-1 if index_base == 0 else 0))
        print(
            f"[dfa_manipulation] base={index_base}  |S|={len(DFA.S)}  "
            f"|act|={len(letters)}  missing={miss}  letters={letters}"
        )
    return DFA, letters

from typing import List, Optional, Dict, Any

def check_letter_disjointness(
    letters: List[str],
    L_list: Optional[List[np.ndarray]] = None,
    DFA = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verify DFA edge letters are pairwise non-overlapping per source state.

    Two letters can only conflict if they are both active transitions out of
    the same DFA state. With a DFA argument the check is scoped per source
    state; without it, falls back to a global check (over-strict).

    Two checks are run:

    1. **Boolean check** (always): pair is *boolean-disjoint* iff some shared
       atomic proposition has opposite polarity. A boolean overlap means a
       literal assignment exists that satisfies both letters — abstraction
       fragile (could trigger non-determinism on a different grid) even if
       the current grid happens to dodge it.

    2. **L-mask check** (when ``L_list`` is provided): pair has a joint
       overlap iff every dimension has at least one cell where both per-dim
       masks are simultaneously non-zero. This is the runtime-safety check
       on the actual abstraction.

    The ``clean`` verdict is based on the L-mask check when ``L_list`` is
    given (runtime safety), otherwise on the boolean check (necessary but
    strict). ``clean_boolean`` and ``clean_mask`` are exposed separately so
    callers can opt into stricter behaviour.

    Parameters
    ----------
    letters : list of str
        DFA edge labels (output of ``dfa_manipulation``).
    L_list : list of np.ndarray, optional
        Per-dim labelling tensors of shape ``(num_letters, N_d)``.
    DFA : object, optional
        DFA exposing ``S`` (state list), ``trans`` (|S| × n_letters
        destination array), and optionally ``sink``. When provided, only
        pairs of letters that are simultaneously active from a common
        source state are checked.
    verbose : bool
        Print a per-pair summary.

    Returns
    -------
    dict with keys
        'boolean_overlaps' : {q: [(i, j, witness), ...]}
        'mask_overlaps'    : {q: [(i, j, joint_cell), ...]}    [if L_list given]
        'clean'            : bool — True iff runtime-safe on current grid.
        'clean_boolean'    : bool — True iff no structural (boolean) overlap.
        'clean_mask'       : bool — True iff no L-mask overlap (or no L_list).
    """
    import warnings
    from src.abstraction.utils.labeling import parse_literals

    n = len(letters)
    parsed = [dict(parse_literals(s)) for s in letters]

    # --- group letters by source state ---
    if DFA is None:
        groups = {None: list(range(n))}
    else:
        trans = np.asarray(DFA.trans, dtype=int)
        sink = getattr(DFA, "sink", None)
        sink = int(sink) if sink is not None else None

        groups = {}
        for q in DFA.S:
            q_int = int(q)
            active = []
            for l in range(n):
                dst = int(trans[q_int, l])
                if dst < 0:
                    continue
                if sink is not None and dst == sink:
                    continue
                active.append(l)
            if len(active) >= 2:
                groups[q_int] = active

    # --- boolean check, per group ---
    boolean_overlaps: Dict[Any, list] = {}
    for q, group in groups.items():
        local = []
        for ii, i in enumerate(group):
            for j in group[ii+1:]:
                common = set(parsed[i]) & set(parsed[j])
                has_contradiction = any(parsed[i][ap] != parsed[j][ap] for ap in common)
                if not has_contradiction:
                    witness = {**parsed[i], **parsed[j]}
                    local.append((i, j, witness))
        if local:
            boolean_overlaps[q] = local

    # --- L-mask check, per group ---
    mask_overlaps: Dict[Any, list] = {}
    if L_list is not None:
        if L_list[0].shape[0] != n:
            warnings.warn(
                f"L_list has {L_list[0].shape[0]} rows but {n} letters; "
                "skipping L-mask check."
            )
            L_list_for_check = None
        else:
            L_list_for_check = L_list
            dim = len(L_list)
            for q, group in groups.items():
                local = []
                for ii, i in enumerate(group):
                    for j in group[ii+1:]:
                        per_dim_witness = []
                        fires_everywhere = True
                        for d in range(dim):
                            prod = L_list[d][i, :] * L_list[d][j, :]
                            if not np.any(prod > 0):
                                fires_everywhere = False
                                break
                            per_dim_witness.append(int(np.argmax(prod)))
                        if fires_everywhere:
                            local.append((i, j, tuple(per_dim_witness)))
                if local:
                    mask_overlaps[q] = local
    else:
        L_list_for_check = None

    # --- verdicts ---
    clean_boolean = (not boolean_overlaps)
    clean_mask    = (L_list_for_check is None) or (not mask_overlaps)
    # Runtime-safety verdict: L-mask when available (the algorithm-relevant
    # check), otherwise fall back to boolean (the structural check).
    clean = clean_mask if L_list_for_check is not None else clean_boolean

    # --- reporting ---
    if verbose:
        n_pairs = sum(len(g) * (len(g) - 1) // 2 for g in groups.values())
        scope = (f"across {len(groups)} source state(s)" if DFA is not None
                 else "global (no DFA structure)")
        print(f"[letter-disjointness] {n} letters, {n_pairs} pairs {scope}")

        def _q_sort_key(kv):
            return (-1 if kv[0] is None else kv[0])

        # boolean
        if not clean_boolean:
            total = sum(len(v) for v in boolean_overlaps.values())
            tag = "WARN" if clean_mask else "FAIL"
            print(f"  boolean check: {tag} — {total} structural overlap(s)")
            for q, lst in sorted(boolean_overlaps.items(), key=_q_sort_key):
                if q is not None:
                    print(f"  ── from DFA state q={q} ──")
                for i, j, w in lst:
                    w_str = ", ".join(f"{ap}={'F' if neg else 'T'}"
                                      for ap, neg in sorted(w.items()))
                    print(f"    [{i}]'{letters[i]}'  ∩  [{j}]'{letters[j]}'")
                    print(f"        witness: {{{w_str}}}")
        else:
            print(f"  boolean check: OK")

        # L-mask
        if L_list_for_check is not None:
            if not clean_mask:
                total = sum(len(v) for v in mask_overlaps.values())
                print(f"  L-mask check: FAIL — {total} grid overlap(s)")
                for q, lst in sorted(mask_overlaps.items(), key=_q_sort_key):
                    if q is not None:
                        print(f"  ── from DFA state q={q} ──")
                    for i, j, st in lst:
                        print(f"    [{i}]'{letters[i]}'  ∩  [{j}]'{letters[j]}'")
                        print(f"        joint cell: {st}")
            else:
                print(f"  L-mask check: OK")

        # verdict
        if clean:
            if clean_boolean:
                print(f"  → CLEAN ✓")
            else:
                print(f"  → CLEAN on current abstraction ✓  "
                      f"(boolean overlaps remain — fragile to grid changes)")
        else:
            print(f"  → OVERLAPS DETECTED ✗")

    return {
        "boolean_overlaps": boolean_overlaps,
        "mask_overlaps":    mask_overlaps,
        "clean":            clean,
        "clean_boolean":    clean_boolean,
        "clean_mask":       clean_mask,
    }