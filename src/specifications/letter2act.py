import re
from collections import OrderedDict
from typing import List, Tuple, Dict

def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith('(') and s.endswith(')'):
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
        if ok:
            s = s[1:-1].strip()
        else:
            break
    return s

def _normalize_cond(s: str) -> str:
    s = str(s).strip()
    s = s.replace('&&', 'and').replace('||', '|')
    s = re.sub(r'\s+', ' ', s)           # compress spaces
    s = _strip_outer_parens(s)            # strip outermost matching parentheses
    return s

def merge_groups(edges_with_AP: List[Tuple[int, int, str]]):
    """
    Input:  [(src, dst, condition_str), ...]
    Output:
      groups:          list of merged expressions (ordered by first occurrence of (src, dst))
      groups_by_edge:  dict mapping (src, dst) -> 'expr1 | expr2 | ...'
    Notes: constant conditions ('1','0','True','False') are ignored; duplicates are removed.
    """
    grouped: Dict[Tuple[int, int], List[str]] = OrderedDict()
    for u, v, cond in edges_with_AP:
        c = _normalize_cond(cond)
        if c in {'1', '0', 'True', 'False', ''}:
            continue
        key = (u, v)
        if key not in grouped:
            grouped[key] = []
        if c not in grouped[key]:
            grouped[key].append(c)

    groups_by_edge = {key: ' | '.join(conds) if len(conds) > 1 else conds[0]
                      for key, conds in grouped.items()}
    groups = list(groups_by_edge.values())
    return groups, groups_by_edge
