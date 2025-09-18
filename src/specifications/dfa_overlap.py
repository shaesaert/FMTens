# dfa_overlap.py
import re
from collections import defaultdict, OrderedDict

_LIT = re.compile(r'(?P<neg>!\s*|not\s+)?(?P<ap>p\d+)\b', re.IGNORECASE)

def _parse_conj_literals(label_str):
    s = str(label_str)
    lits = OrderedDict()
    for m in _LIT.finditer(s):
        ap = m.group('ap')
        neg = bool(m.group('neg'))
        val = (not neg)
        if ap in lits and lits[ap] != val:
            raise ValueError(f"Contradictory literals for {ap} in label: {label_str!r}")
        lits[ap] = val
    return lits  # e.g. {'p1': True, 'p2': False}

def _aps_in_labels(labels):
    S = OrderedDict()
    for lab in labels:
        for m in _LIT.finditer(str(lab)):
            S[m.group('ap')] = True
    return list(S.keys())

def _eval_label_on_assign(label_str, assign_dict):
    lits = _parse_conj_literals(label_str)
    for ap, required in lits.items():
        if assign_dict.get(ap) is None:
            return False
        if assign_dict[ap] != required:
            return False
    return True

def _assign_to_atom(assign_dict, ap_order):
    parts = []
    for ap in ap_order:
        val = assign_dict[ap]
        parts.append(ap if val else f'!{ap}')
    return ' & '.join(parts)

def refine_labels_to_disjoint_atoms(labels):
    ap_list = _aps_in_labels(labels)
    k = len(ap_list)
    if k == 0:
        return ['1'], {lab: ['1'] for lab in labels}

    atoms = []
    cover = {lab: [] for lab in labels}
    for mask in range(1 << k):
        assign = {ap_list[i]: bool((mask >> i) & 1) for i in range(k)}
        satisfied = [lab for lab in labels if _eval_label_on_assign(lab, assign)]
        if not satisfied:
            continue
        atom = _assign_to_atom(assign, ap_list)
        atoms.append(atom)
        for lab in satisfied:
            cover[lab].append(atom)

    # dedup
    seen = OrderedDict()
    atoms = [a for a in atoms if not (a in seen or seen.setdefault(a, False))]
    for lab in cover:
        seenL = OrderedDict()
        cover[lab] = [a for a in cover[lab] if not (a in seenL or seenL.setdefault(a, False))]
    return atoms, cover

def _iter_edges_from_dfa(DFA):
    if hasattr(DFA, "transitions") and DFA.transitions is not None:
        for (u, v, data) in DFA.transitions:
            lab = data.get("label", data.get("condition", "1"))
            yield int(u), int(v), str(lab), data
        return
    if hasattr(DFA, "graph"):
        for (u, v, data) in DFA.graph.edges(data=True):
            lab = data.get("label", data.get("condition", "1"))
            yield int(u), int(v), str(lab), data
        return
    raise AttributeError("DFA must have 'transitions' or 'graph' with edge 'label'/'condition'.")

def remove_overlap_inplace(DFA, verbose=True):
    out_by_src = defaultdict(list)
    edges_all = list(_iter_edges_from_dfa(DFA))
    for u, v, lab, data in edges_all:
        out_by_src[u].append((u, v, lab, data))

    new_edges = []
    for u, bundle in out_by_src.items():
        labels = [lab for (_, _, lab, _) in bundle]
        try:
            atoms, cover = refine_labels_to_disjoint_atoms(labels)
        except ValueError as e:
            if verbose:
                print(f"[overlap] skip src {u}: {e}")
            new_edges.extend(bundle)
            continue

        before, after = len(bundle), 0
        for (uu, vv, lab, data) in bundle:
            alist = cover.get(lab, [])
            for a in alist:
                nd = dict(data)
                nd['label'] = a
                nd['condition'] = a
                new_edges.append((uu, vv, a, nd))
                after += 1
        if verbose:
            print(f"[overlap] src {u}: {before} labels -> {after} atom edges (|AP|={len(_aps_in_labels(labels))})")

    # dedup (u,v,label)
    dedup = OrderedDict()
    for u, v, lab, data in new_edges:
        key = (u, v, lab)
        if key not in dedup:
            dedup[key] = data
    new_edges = [(u, v, lab, dedup[(u, v, lab)]) for (u, v, lab) in dedup.keys()]

    # write back
    if hasattr(DFA, "transitions") and DFA.transitions is not None:
        DFA.transitions = [(u, v, data) for (u, v, _, data) in new_edges]

    if hasattr(DFA, "graph"):
        G = DFA.graph
        G.remove_edges_from(list(G.edges()))
        for (u, v, lab, data) in new_edges:
            dd = dict(data)
            dd['label'] = lab
            dd['condition'] = lab
            G.add_edge(u, v, **dd)

    # refresh DFA.act if present
    if hasattr(DFA, "act"):
        letters_seen = OrderedDict()
        for _, _, data in new_edges:
            lab = data.get("label", data.get("condition"))
            letters_seen[str(lab)] = True
        DFA.act = list(letters_seen.keys())

    return DFA
