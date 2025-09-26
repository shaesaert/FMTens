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

    # if outfile:
        # plt.savefig(outfile, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ==============================================
# ---- one-call API: build L for all dims ----
# ==============================================

from typing import Union, Sequence, Dict, List, Optional, Tuple

# src/abstraction/utils/labeling.py
import os

# assume these helpers live in this same module/file as per your merge:
# - _axis_from_hx
# - _intervals_from_polytope_1d
# - label_axis_by_letters
# - plot_binary_matrix

import numpy as np
import polytope as pc

def _axis_from_hx(hx):
    """Return a 1D numpy array representing the (only) axis in hx."""
    if isinstance(hx, (list, tuple)):
        if len(hx) != 1:
            raise ValueError(f"_axis_from_hx expected a single axis, got {len(hx)}")
        return np.asarray(hx[0], dtype=float).ravel()
    return np.asarray(hx, dtype=float).ravel()

def intervals_from_polytope_1d(P, atol: float = 1e-12):
    """
    Convert a 1D H-polytope {x | A x <= b} into a list of closed intervals [(lo, hi)].
    - Assumes a single connected interval (typical for your AP regions).
    - If P is a list/tuple of polytopes, flattens and returns the union of intervals.
    """
    def _one(poly):
        A = np.asarray(poly.A, dtype=float).reshape(-1, 1)
        b = np.asarray(poly.b, dtype=float).ravel()
        lo, hi = -np.inf, np.inf
        for a, bb in zip(A.ravel(), b):
            if abs(a) <= atol:
                continue
            bound = bb / a
            if a > 0:   #  a*x <= b  ->  x <= b/a
                hi = min(hi, bound)
            else:       #  a*x <= b  ->  -|a|*x <= b -> x >= b/a
                lo = max(lo, bound)
        return [] if lo > hi else [(lo, hi)]

    # support a union passed as list/tuple
    if isinstance(P, (list, tuple)):
        out = []
        for poly in P:
            out.extend(_one(poly))
        return out
    return _one(P)

# keep backward compatibility with code that still calls the underscored name
def _intervals_from_polytope_1d(P, atol: float = 1e-12):
    return intervals_from_polytope_1d(P, atol=atol)


def dim_label(sysAbs, sysLTI, letters, visualize=False, outdir=None, prefix="L"):
    """
    Build per-dimension label matrices.

    Returns:
      - dict keyed by dimension if sysAbs is a dict
      - list in order if sysAbs is a list/tuple

    L[d] has shape (len(letters), N_d).
    """
    # normalize keys and return type
    if isinstance(sysAbs, dict):
        keys = sorted(sysAbs.keys())
        get_abs = lambda k: sysAbs[k]
        get_lti = lambda k: sysLTI[k]
        return_as_dict = True
    else:
        keys = list(range(len(sysAbs)))
        get_abs = lambda k: sysAbs[k]
        get_lti = lambda k: sysLTI[k]
        return_as_dict = False

    L_map = {}

    # ensure output dir
    if visualize and outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    for k in keys:
        mdp_k = get_abs(k)
        lti_k = get_lti(k)

        # 1) axis from hx
        ## label states
        #xax_k = _axis_from_hx(mdp_k.hx)
        ## label outputs
        xax_k = _axis_from_hx(mdp_k.outputs)


        # 2) AP sets and 1D region intervals
        aps_set = set(getattr(lti_k, "AP", []))
        regions = {ap: _intervals_from_polytope_1d(reg)
                   for ap, reg in zip(getattr(lti_k, "AP", []),
                                      getattr(lti_k, "regions", []))}

        # 3) label matrix for this dimension
        Lk = np.asarray(
            label_axis_by_letters(xax_k, letters, aps_set, regions),
            dtype=float
        )
        L_map[k] = Lk

        # 4) optional visualization
        if visualize:
            fname = f"{prefix}{k}.png"
            if outdir:
                fname = os.path.join(outdir, fname)
            # pick a light tick step
            step = max(1, len(xax_k) // 10)
            plot_binary_matrix(
                Lk.astype(int),
                title=f"L[{k}] (letters × x{k}-grid)",
                row_labels=letters,
                outfile=fname,
                x_tick_step=step,
            )

    if return_as_dict:
        return L_map
    else:
        return [L_map[k] for k in keys]
