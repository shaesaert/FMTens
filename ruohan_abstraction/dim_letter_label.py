# dim_letter_label.py
import re
from typing import List, Dict, Tuple, Iterable, Callable
import numpy as np

__all__ = [
    "parse_literals",
    "make_ap_predicates",
    "letter_mask_on_axis",
    "label_axis_by_letters",
]

# ---------- Parsing utilities: extract literals (supports "!p1" or "not p1") ----------
_LIT = re.compile(r'(?P<neg>!\s*|not\s+)?(?P<ap>p\d+)\b', re.IGNORECASE)

def parse_literals(expr: str) -> List[Tuple[str, bool]]:
    """
    Parse a letter expression and return a list of (ap_name, is_negated) literals.
    Example: 'not p1 and not p2' -> [('p1', True), ('p2', True)]
    NOTE: We do not parse AND/OR structure here; per-letter is assumed to be a conjunction.
    """
    lits: List[Tuple[str, bool]] = []
    for m in _LIT.finditer(str(expr)):
        ap = m.group('ap')
        neg = bool(m.group('neg'))
        lits.append((ap, neg))
    return lits


# ---------- Interval predicate construction: each AP is a union of closed intervals (robust scalarization) ----------
def make_ap_predicates(
    intervals_by_ap: Dict[str, Iterable[Tuple[float, float]]]
) -> Dict[str, Callable[[float], bool]]:
    """
    Build predicates ap -> f(x)->bool, where x can be a Python/numpy scalar,
    or a length-1 numpy array. Closed intervals are used.
    """
    pred: Dict[str, Callable[[float], bool]] = {}

    for ap, intervals in intervals_by_ap.items():
        # Normalize intervals and ensure lo <= hi
        ivals = []
        for lo, hi in intervals:
            lo_f, hi_f = float(lo), float(hi)
            if lo_f <= hi_f:
                ivals.append((lo_f, hi_f))
            else:
                ivals.append((hi_f, lo_f))

        def f(x, ivals=ivals):
            # robust scalarization: accept scalar, numpy scalar, or length-1 arrays
            xa = np.asarray(x)
            if xa.ndim > 0:
                if xa.size != 1:
                    raise ValueError(f"x must be scalar or length-1, got shape {xa.shape}")
                x0 = float(xa.ravel()[0])
            else:
                x0 = float(xa)
            return any(lo <= x0 <= hi for (lo, hi) in ivals)

        pred[ap] = f

    return pred


# ---------- Core: for a single letter, produce a 0/1 mask along one dimension's axis ----------
def letter_mask_on_axis(
    xax: Iterable[float],
    letter: str,
    aps_in_dim: set,
    ap_regions: Dict[str, Iterable[Tuple[float, float]]],
) -> List[int]:
    """
    Compute a 0/1 mask along xax for a single letter on a given dimension.

    Semantics:
      - Only literals whose AP ∈ aps_in_dim are enforced on this dimension.
      - If a letter contains both ap and !ap (contradiction) on this dimension,
        the mask is all zeros.
      - If after projection there are no local APs in the letter, it's unconstrained
        on this dimension -> mask is all ones (MATLAB dim_label behavior).
    """
    # parse and keep only literals whose AP lies in this dimension
    lits_all = parse_literals(letter)
    lits = [(ap, neg) for (ap, neg) in lits_all if ap in aps_in_dim]

    # contradiction check: same AP appears both positive and negative
    signs = {}
    for ap, neg in lits:
        if ap in signs and signs[ap] != neg:
            return [0] * int(np.asarray(xax).size)
        signs[ap] = neg

    ap_pred = make_ap_predicates(ap_regions)

    # no local constraints -> unconstrained on this dimension => all ones
    if not lits:
        return [1] * int(np.asarray(xax).size)

    mask: List[int] = []
    for x in np.asarray(xax).ravel():  # flatten in case xax is an array
        ok = True
        for ap, neg in lits:
            v = ap_pred[ap](x)            # bool
            ok = ok and ((not v) if neg else v)
            if not ok:
                break
        mask.append(1 if ok else 0)
    return mask


# ---------- Batch: generate a 0/1 matrix for a set of letters (rows=letters, cols=x points) ----------
def label_axis_by_letters(
    xax: Iterable[float],
    letters: List[str],
    aps_in_dim: set,
    ap_regions: Dict[str, Iterable[Tuple[float, float]]],
) -> List[List[int]]:
    """
    Return a len(letters) x len(xax) 0/1 matrix as a list of lists.
    Row i corresponds to letters[i].
    """
    return [letter_mask_on_axis(xax, L, aps_in_dim, ap_regions) for L in letters]
