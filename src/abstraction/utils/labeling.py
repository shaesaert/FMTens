# labeling.py
# Build per-dimension letter masks (L[d]) and optionally visualize them.

from __future__ import annotations
from typing import Dict, List, Iterable, Tuple, Optional, Sequence, Union
import numpy as np
import polytope as pc
import re
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

Array = np.ndarray

# =========================
# ---- parsing helpers ----
# =========================

_LIT = re.compile(r'(?P<neg>!\s*|not\s+)?(?P<ap>p\d+)\b', re.IGNORECASE)

def parse_literals(expr: str) -> List[Tuple[str, bool]]:
    """Return [(ap_name, is_negated)] found in a letter string."""
    out: List[Tuple[str, bool]] = []
    for m in _LIT.finditer(str(expr)):
        out.append((m.group('ap'), bool(m.group('neg'))))
    return out


# ======================================
# ---- axis & interval utilities ----
# ======================================

def axis_from_hx(hx: Sequence[Array]) -> Array:
    """For a 1D MDPModel, return the 1D state axis (hx[0])."""
    if not isinstance(hx, (list, tuple)) or len(hx) == 0:
        raise ValueError("hx must be a non-empty list/tuple of 1D axes.")
    return np.asarray(hx[0], dtype=float).ravel()

def intervals_from_polytope_1d(P) -> List[Tuple[float, float]]:
    """
    Convert a 1D convex polytope to a list with one closed interval [(lo, hi)].
    (Good enough for your current usage.)
    """
    V = np.asarray([np.asarray(v).ravel() for v in pc.extreme(P)], dtype=float).reshape(-1)
    lo, hi = float(np.min(V)), float(np.max(V))
    if lo > hi:
        lo, hi = hi, lo
    return [(lo, hi)]


# ===========================================
# ---- build 0/1 masks for one dimension ----
# ===========================================

def make_ap_predicates(intervals_by_ap: Dict[str, Iterable[Tuple[float, float]]]):
    """ap -> f(x) where f(x) checks membership in union of closed intervals."""
    pred = {}
    for ap, intervals in intervals_by_ap.items():
        ivals = []
        for lo, hi in intervals:
            a, b = float(lo), float(hi)
            ivals.append((a if a <= b else b, b if b >= a else a))
        def f(x, ivals=ivals):
            x0 = float(np.asarray(x).reshape(-1)[0])
            return any(lo <= x0 <= hi for lo, hi in ivals)
        pred[ap] = f
    return pred

def letter_mask_on_axis(
    xax: Iterable[float],
    letter: str,
    aps_in_dim: set,
    ap_regions: Dict[str, Iterable[Tuple[float, float]]],
) -> List[int]:
    """
    Compute a 0/1 mask along xax for a single letter on one dimension.

    - Only literals whose AP ∈ aps_in_dim are enforced on this dimension.
    - If a letter contains both ap and !ap for some AP *in this dim*, mask is all zeros.
    - If no local APs appear in the letter, mask is all ones (unconstrained).
    """
    lits_all = parse_literals(letter)
    lits = [(ap, neg) for ap, neg in lits_all if ap in aps_in_dim]

    # contradiction?
    seen = {}
    for ap, neg in lits:
        if ap in seen and seen[ap] != neg:
            return [0] * int(np.asarray(xax).size)
        seen[ap] = neg

    if not lits:
        return [1] * int(np.asarray(xax).size)

    pred = make_ap_predicates(ap_regions)
    out = []
    for x in np.asarray(xax).ravel():
        ok = True
        for ap, neg in lits:
            v = pred[ap](x)
            ok = ok and ((not v) if neg else v)
            if not ok:
                break
        out.append(1 if ok else 0)
    return out

def label_axis_by_letters(
    xax: Iterable[float],
    letters: List[str],
    aps_in_dim: set,
    ap_regions: Dict[str, Iterable[Tuple[float, float]]],
) -> Array:
    """Return a (len(letters) × len(xax)) 0/1 numpy array."""
    return np.asarray([letter_mask_on_axis(xax, L, aps_in_dim, ap_regions) for L in letters], dtype=int)


# ==========================
# ---- visualization ----
# ==========================

def plot_binary_matrix(mat, title="", row_labels=None, outfile=None,
                       x_tick_step=None, show=True):
    A = np.asarray(mat, dtype=int)
    n_rows, n_cols = A.shape
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["red", "green"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    w = min(max(n_cols / 80.0, 8.0), 24.0)
    h = min(max(n_rows * 0.6, 2.8), 12.0)
    fig, ax = plt.subplots(figsize=(w, h), dpi=120)

    ax.imshow(A, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    ax.set_title(title, fontsize=12)

    if row_labels is not None and len(row_labels) == n_rows:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(row_labels, fontsize=9)
    else:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([str(i) for i in range(n_rows)], fontsize=8)

    if x_tick_step is None:
        x_tick_step = 100 if n_cols > 200 else 20 if n_cols > 80 else 10
    xticks = np.arange(0, n_cols, x_tick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks], fontsize=8, rotation=45)

    ax.set_xlabel("state index")
    ax.set_ylabel("letter index")
    ax.grid(False)
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ==============================================
# ---- one-call API: build L for all dims ----
# ==============================================

from typing import Union, Sequence, Dict, List, Optional, Tuple

def dim_label(
    sysAbs: Union[Dict[Union[int, str], object], Sequence[object]],
    sysLTI: Union[Dict[Union[int, str], object], Sequence[object]],
    letters: List[str],
    *,
    visualize: bool = False,
    outfiles: Optional[Union[bool, str, Sequence[str]]] = None,
    x_tick_step: Optional[int] = None,
    show: bool = True,
) -> List[np.ndarray]:
    """
    Build L[d] per dimension. Optionally visualize and/or save plots.

    outfiles:
      - None         -> don't save
      - True         -> save as 'L{i}.png'
      - 'pattern'    -> if contains '{i}' or '{idx}', format with the dim index
                        e.g. 'plots/L{idx}.pdf'
                        if no placeholder and multiple dims, auto-suffix:
                        'pattern_0.png', 'pattern_1.png', ...
      - [names...]   -> per-dim filenames (length ≥ #dims)
    """
    # normalize to ordered keys
    if isinstance(sysAbs, dict):
        keys = list(sysAbs.keys())
    else:
        keys = list(range(len(sysAbs)))
    nd = len(keys)

    # helper: resolve filename for dim i
    def _resolve_name(i: int) -> Optional[str]:
        if outfiles is None:
            return None
        if outfiles is True:
            return f"L{i}.png"
        if isinstance(outfiles, (list, tuple)):
            if i < len(outfiles):
                return outfiles[i]
            # if shorter, fall back to default
            return f"L{i}.png"
        if isinstance(outfiles, str):
            if ("{i}" in outfiles) or ("{idx}" in outfiles):
                return outfiles.format(i=i, idx=i)
            # single string with multiple dims -> auto suffix
            if nd == 1:
                return outfiles
            # try to insert before extension
            import os
            root, ext = os.path.splitext(outfiles)
            ext = ext or ".png"
            return f"{root}_{i}{ext}"
        return None

    L_list: List[np.ndarray] = []

    for pos, k in enumerate(keys):
        mdp = sysAbs[k]
        cont = sysLTI[k]

        xax = axis_from_hx(mdp.hx)
        aps_in_dim = set(cont.AP)
        ap_regions = {ap: intervals_from_polytope_1d(reg)
                      for ap, reg in zip(cont.AP, cont.regions)}

        L = label_axis_by_letters(xax, letters, aps_in_dim, ap_regions).astype(float)
        L_list.append(L)

        if visualize or outfiles is not None:
            fname = _resolve_name(pos)
            title = f"L[{pos}] (letters × x-grid)"
            plot_binary_matrix(L.astype(int), title=title, row_labels=letters,
                               outfile=fname, x_tick_step=x_tick_step, show=show)

    return L_list