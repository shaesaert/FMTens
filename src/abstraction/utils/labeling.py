# # labeling.py
# # Build per-dimension letter masks (L[d]) and optionally visualize them.
#
from __future__ import annotations
from typing import Dict, List, Iterable, Tuple, Optional, Sequence, Union
import numpy as np
import polytope as pc
import re
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
#
# Array = np.ndarray
#
# # =========================
# # ---- parsing helpers ----
# # =========================
#
# _LIT = re.compile(r'(?P<neg>!\s*|not\s+)?(?P<ap>p\d+)\b', re.IGNORECASE)
#
# def parse_literals(expr: str) -> List[Tuple[str, bool]]:
#     """Return [(ap_name, is_negated)] found in a letter string."""
#     out: List[Tuple[str, bool]] = []
#     for m in _LIT.finditer(str(expr)):
#         out.append((m.group('ap'), bool(m.group('neg'))))
#     return out
#
#
# # ======================================
# # ---- axis & interval utilities ----
# # ======================================
#
# def axis_from_hx(hx: Sequence[Array]) -> Array:
#     """For a 1D MDPModel, return the 1D state axis (hx[0])."""
#     if not isinstance(hx, (list, tuple)) or len(hx) == 0:
#         raise ValueError("hx must be a non-empty list/tuple of 1D axes.")
#     return np.asarray(hx[0], dtype=float).ravel()
#
# def intervals_from_polytope_1d(P) -> List[Tuple[float, float]]:
#     """
#     Convert a 1D convex polytope to a list with one closed interval [(lo, hi)].
#     (Good enough for your current usage.)
#     """
#     V = np.asarray([np.asarray(v).ravel() for v in pc.extreme(P)], dtype=float).reshape(-1)
#     lo, hi = float(np.min(V)), float(np.max(V))
#     if lo > hi:
#         lo, hi = hi, lo
#     return [(lo, hi)]
#
#
# # ===========================================
# # ---- build 0/1 masks for one dimension ----
# # ===========================================
#
# def make_ap_predicates(intervals_by_ap: Dict[str, Iterable[Tuple[float, float]]]):
#     """ap -> f(x) where f(x) checks membership in union of closed intervals."""
#     pred = {}
#     for ap, intervals in intervals_by_ap.items():
#         ivals = []
#         for lo, hi in intervals:
#             a, b = float(lo), float(hi)
#             ivals.append((a if a <= b else b, b if b >= a else a))
#         def f(x, ivals=ivals):
#             x0 = float(np.asarray(x).reshape(-1)[0])
#             return any(lo <= x0 <= hi for lo, hi in ivals)
#         pred[ap] = f
#     return pred
#
# def letter_mask_on_axis(
#     xax: Iterable[float],
#     letter: str,
#     aps_in_dim: set,
#     ap_regions: Dict[str, Iterable[Tuple[float, float]]],
# ) -> List[int]:
#     """
#     Compute a 0/1 mask along xax for a single letter on one dimension.
#
#     - Only literals whose AP ∈ aps_in_dim are enforced on this dimension.
#     - If a letter contains both ap and !ap for some AP *in this dim*, mask is all zeros.
#     - If no local APs appear in the letter, mask is all ones (unconstrained).
#     """
#     lits_all = parse_literals(letter)
#     lits = [(ap, neg) for ap, neg in lits_all if ap in aps_in_dim]
#
#     # contradiction?
#     seen = {}
#     for ap, neg in lits:
#         if ap in seen and seen[ap] != neg:
#             return [0] * int(np.asarray(xax).size)
#         seen[ap] = neg
#
#     if not lits:
#         return [1] * int(np.asarray(xax).size)
#
#     pred = make_ap_predicates(ap_regions)
#     out = []
#     for x in np.asarray(xax).ravel():
#         ok = True
#         for ap, neg in lits:
#             v = pred[ap](x)
#             ok = ok and ((not v) if neg else v)
#             if not ok:
#                 break
#         out.append(1 if ok else 0)
#     return out
#
# def label_axis_by_letters(
#     xax: Iterable[float],
#     letters: List[str],
#     aps_in_dim: set,
#     ap_regions: Dict[str, Iterable[Tuple[float, float]]],
# ) -> Array:
#     """Return a (len(letters) × len(xax)) 0/1 numpy array."""
#     return np.asarray([letter_mask_on_axis(xax, L, aps_in_dim, ap_regions) for L in letters], dtype=int)
#
#
# # ==========================
# # ---- visualization ----
# # ==========================
#
# def plot_binary_matrix(mat, title="", row_labels=None, outfile=None,
#                        x_tick_step=None, show=True):
#     A = np.asarray(mat, dtype=int)
#     n_rows, n_cols = A.shape
#     from matplotlib.colors import ListedColormap, BoundaryNorm
#     cmap = ListedColormap(["red", "green"])
#     norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
#
#     w = min(max(n_cols / 80.0, 8.0), 24.0)
#     h = min(max(n_rows * 0.6, 2.8), 12.0)
#     fig, ax = plt.subplots(figsize=(w, h), dpi=120)
#
#     ax.imshow(A, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
#     ax.set_title(title, fontsize=12)
#
#     if row_labels is not None and len(row_labels) == n_rows:
#         ax.set_yticks(np.arange(n_rows))
#         ax.set_yticklabels(row_labels, fontsize=9)
#     else:
#         ax.set_yticks(np.arange(n_rows))
#         ax.set_yticklabels([str(i) for i in range(n_rows)], fontsize=8)
#
#     if x_tick_step is None:
#         x_tick_step = 100 if n_cols > 200 else 20 if n_cols > 80 else 10
#     xticks = np.arange(0, n_cols, x_tick_step)
#     ax.set_xticks(xticks)
#     ax.set_xticklabels([str(int(x)) for x in xticks], fontsize=8, rotation=45)
#
#     ax.set_xlabel("state index")
#     ax.set_ylabel("letter index")
#     ax.grid(False)
#     plt.tight_layout()
#
#     # if outfile:
#         # plt.savefig(outfile, bbox_inches="tight")
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)
#
#
# # ==============================================
# # ---- one-call API: build L for all dims ----
# # ==============================================
#
# from typing import Union, Sequence, Dict, List, Optional, Tuple
#
# # src/abstraction/utils/labeling.py
# import os
#
# # assume these helpers live in this same module/file as per your merge:
# # - _axis_from_hx
# # - _intervals_from_polytope_1d
# # - label_axis_by_letters
# # - plot_binary_matrix
#
# import numpy as np
# import polytope as pc
#
# def _axis_from_hx(hx):
#     """Return a 1D numpy array representing the (only) axis in hx."""
#     if isinstance(hx, (list, tuple)):
#         if len(hx) != 1:
#             raise ValueError(f"_axis_from_hx expected a single axis, got {len(hx)}")
#         return np.asarray(hx[0], dtype=float).ravel()
#     return np.asarray(hx, dtype=float).ravel()
#
# def intervals_from_polytope_1d(P, atol: float = 1e-12):
#     """
#     Convert a 1D H-polytope {x | A x <= b} into a list of closed intervals [(lo, hi)].
#     - Assumes a single connected interval (typical for your AP regions).
#     - If P is a list/tuple of polytopes, flattens and returns the union of intervals.
#     """
#     def _one(poly):
#         A = np.asarray(poly.A, dtype=float).reshape(-1, 1)
#         b = np.asarray(poly.b, dtype=float).ravel()
#         lo, hi = -np.inf, np.inf
#         for a, bb in zip(A.ravel(), b):
#             if abs(a) <= atol:
#                 continue
#             bound = bb / a
#             if a > 0:   #  a*x <= b  ->  x <= b/a
#                 hi = min(hi, bound)
#             else:       #  a*x <= b  ->  -|a|*x <= b -> x >= b/a
#                 lo = max(lo, bound)
#         return [] if lo > hi else [(lo, hi)]
#
#     # support a union passed as list/tuple
#     if isinstance(P, (list, tuple)):
#         out = []
#         for poly in P:
#             out.extend(_one(poly))
#         return out
#     return _one(P)
#
# # keep backward compatibility with code that still calls the underscored name
# def _intervals_from_polytope_1d(P, atol: float = 1e-12):
#     return intervals_from_polytope_1d(P, atol=atol)
#
#
# def dim_label(sysAbs, sysLTI, letters, visualize=False, outdir=None, prefix="L"):
#     """
#     Build per-dimension label matrices.
#
#     Returns:
#       - dict keyed by dimension if sysAbs is a dict
#       - list in order if sysAbs is a list/tuple
#
#     L[d] has shape (len(letters), N_d).
#     """
#     # normalize keys and return type
#     if isinstance(sysAbs, dict):
#         keys = sorted(sysAbs.keys())
#         get_abs = lambda k: sysAbs[k]
#         get_lti = lambda k: sysLTI[k]
#         return_as_dict = True
#     else:
#         keys = list(range(len(sysAbs)))
#         get_abs = lambda k: sysAbs[k]
#         get_lti = lambda k: sysLTI[k]
#         return_as_dict = False
#
#     L_map = {}
#
#     # ensure output dir
#     if visualize and outdir is not None:
#         os.makedirs(outdir, exist_ok=True)
#
#     for k in keys:
#         mdp_k = get_abs(k)
#         lti_k = get_lti(k)
#
#         # 1) axis from hx
#         ## label states
#         #xax_k = _axis_from_hx(mdp_k.hx)
#         ## label outputs
#         xax_k = _axis_from_hx(mdp_k.outputs)
#
#
#         # 2) AP sets and 1D region intervals
#         aps_set = set(getattr(lti_k, "AP", []))
#         regions = {ap: _intervals_from_polytope_1d(reg)
#                    for ap, reg in zip(getattr(lti_k, "AP", []),
#                                       getattr(lti_k, "regions", []))}
#
#         # 3) label matrix for this dimension
#         Lk = np.asarray(
#             label_axis_by_letters(xax_k, letters, aps_set, regions),
#             dtype=float
#         )
#         L_map[k] = Lk
#
#         # 4) optional visualization
#         if visualize:
#             fname = f"{prefix}{k}.png"
#             if outdir:
#                 fname = os.path.join(outdir, fname)
#             # pick a light tick step
#             step = max(1, len(xax_k) // 10)
#             plot_binary_matrix(
#                 Lk.astype(int),
#                 title=f"L[{k}] (letters × x{k}-grid)",
#                 row_labels=letters,
#                 outfile=fname,
#                 x_tick_step=step,
#             )
#
#     if return_as_dict:
#         return L_map
#     else:
#         return [L_map[k] for k in keys]
# labeling.py
# Build per-dimension letter masks (L[d]) and optionally visualize them.
# Includes robust labeling L_eps where positive literals are SHRUNK by eps
# and negated literals are the COMPLEMENT of EXPANDED regions.

# from __future__ import annotations

from typing import (
    Dict, List, Iterable, Tuple, Optional, Sequence, Union, Mapping, Callable
)

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

# NOTE: Kept for completeness, but not used by the main APIs below.
def intervals_from_polytope_1d_extreme(P) -> List[Tuple[float, float]]:
    """
    Convert a 1D convex polytope to a list with one closed interval [(lo, hi)]
    using extreme points (requires vertex enumeration). Generally slower;
    prefer the H-polytope version further below.
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
    pred: Dict[str, Callable[[float], bool]] = {}
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
    seen: Dict[str, bool] = {}
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

def plot_binary_matrix(mat, title: str = "", row_labels=None, outfile: Optional[str] = None,
                       x_tick_step: Optional[int] = None, show: bool = True):
    A = np.asarray(mat, dtype=int)
    n_rows, n_cols = A.shape
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
    #     plt.savefig(outfile, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ==============================================
# ---- one-call API: build L for all dims ----
# ==============================================

import os

def _axis_from_hx(hx):
    """Return a 1D numpy array representing the (only) axis in hx or an array-like."""
    if isinstance(hx, (list, tuple)):
        if len(hx) != 1:
            raise ValueError(f"_axis_from_hx expected a single axis, got {len(hx)}")
        return np.asarray(hx[0], dtype=float).ravel()
    return np.asarray(hx, dtype=float).ravel()

def intervals_from_polytope_1d(P, atol: float = 1e-12):
    """
    Convert a 1D H-polytope {x | A x <= b} into a list of closed intervals [(lo, hi)].
    - Assumes a single connected interval per polytope (typical for 1D AP regions).
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


def dim_label(sysAbs, sysLTI, letters, visualize: bool = False,
              outdir: Optional[str] = None, prefix: str = "L"):
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

    L_map: Dict[int, Array] = {}

    # ensure output dir
    if visualize and outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    for k in keys:
        mdp_k = get_abs(k)
        lti_k = get_lti(k)

        # choose outputs axis (your current choice)
        xax_k = _axis_from_hx(getattr(mdp_k, "outputs", getattr(mdp_k, "hx", None)))
        if xax_k is None:
            raise ValueError("MDPModel has neither 'outputs' nor 'hx' axis available.")

        # AP sets and 1D region intervals
        aps = list(getattr(lti_k, "AP", []))
        regions = list(getattr(lti_k, "regions", []))
        aps_set = set(aps)
        regions_map = {ap: _intervals_from_polytope_1d(reg) for ap, reg in zip(aps, regions)}

        # label matrix for this dimension
        Lk = np.asarray(
            label_axis_by_letters(xax_k, letters, aps_set, regions_map),
            dtype=float
        )
        L_map[k] = Lk

        # optional visualization
        if visualize:
            fname = f"{prefix}{k}.png"
            if outdir:
                fname = os.path.join(outdir, fname)
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


# ============================
# ---- robust interval ops ----
# ============================

def _normalize_intervals(iv: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sort & merge overlapping/adjacent closed intervals."""
    iv = [(float(a), float(b)) if a <= b else (float(b), float(a)) for a, b in iv]
    iv.sort()
    merged: List[List[float]] = []
    for a, b in iv:
        if not merged:
            merged.append([a, b])
        else:
            A, B = merged[-1]
            if a <= B:      # closed intervals: touching/overlap merge
                merged[-1][1] = max(B, b)
            else:
                merged.append([a, b])
    return [(a, b) for a, b in merged]

def _clip_intervals(iv: Iterable[Tuple[float, float]], lo: float, hi: float) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in iv:
        aa, bb = max(lo, a), min(hi, b)
        if aa <= bb:
            out.append((aa, bb))
    return _normalize_intervals(out)

def _expand_intervals(iv: Iterable[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
    if eps <= 0:
        return _normalize_intervals(iv)
    return _normalize_intervals([(a - eps, b + eps) for a, b in iv])

def _shrink_intervals(iv: Iterable[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
    if eps <= 0:
        return _normalize_intervals(iv)
    out: List[Tuple[float, float]] = []
    for a, b in iv:
        aa, bb = a + eps, b - eps
        if aa <= bb:
            out.append((aa, bb))
    return _normalize_intervals(out)

def _complement_intervals(iv: Iterable[Tuple[float, float]], lo: float, hi: float) -> List[Tuple[float, float]]:
    """Complement of union iv within [lo, hi] (all closed intervals)."""
    iv = _clip_intervals(_normalize_intervals(iv), lo, hi)
    if not iv:
        return [(lo, hi)]
    out: List[Tuple[float, float]] = []
    cur = lo
    for a, b in iv:
        if cur < a:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        out.append((cur, hi))
    return _normalize_intervals(out)

def _ensure_eps_map(eps: Union[float, Mapping[str, float]], aps: Iterable[str]) -> Dict[str, float]:
    if isinstance(eps, (int, float)):
        return {ap: float(eps) for ap in aps}
    # mapping
    return {ap: float(eps.get(ap, 0.0)) for ap in aps}

def make_pos_neg_predicates_robust(
    ap_regions: Dict[str, Iterable[Tuple[float, float]]],
    domain: Tuple[float, float],
    eps_map: Dict[str, float],
) -> Tuple[Dict[str, Callable[[float], bool]], Dict[str, Callable[[float], bool]]]:
    """
    Return two dicts:
      pos_pred[ap](x): membership in SHRUNK(ap, eps_ap)
      neg_pred[ap](x): membership in COMPLEMENT(EXPANDED(ap, eps_ap), domain)
    """
    lo_dom, hi_dom = map(float, domain)

    pos_pred: Dict[str, Callable[[float], bool]] = {}
    neg_pred: Dict[str, Callable[[float], bool]] = {}
    for ap, iv in ap_regions.items():
        ivn = _normalize_intervals(iv)
        iv_pos = _clip_intervals(_shrink_intervals(ivn, eps_map.get(ap, 0.0)), lo_dom, hi_dom)
        iv_neg_base = _clip_intervals(_expand_intervals(ivn, eps_map.get(ap, 0.0)), lo_dom, hi_dom)
        iv_neg = _complement_intervals(iv_neg_base, lo_dom, hi_dom)

        def _pred_factory(intervals: List[Tuple[float, float]]):
            intervals = list(intervals)  # bind
            def f(x) -> bool:
                x0 = float(np.asarray(x).reshape(-1)[0])
                return any(a <= x0 <= b for a, b in intervals)
            return f

        pos_pred[ap] = _pred_factory(iv_pos)
        neg_pred[ap] = _pred_factory(iv_neg)

    return pos_pred, neg_pred

def letter_mask_on_axis_robust(
    xax: Iterable[float],
    letter: str,
    aps_in_dim: set,
    pos_pred: Dict[str, Callable[[float], bool]],
    neg_pred: Dict[str, Callable[[float], bool]],
) -> List[int]:
    """
    Robust 0/1 mask for a single letter on one dimension:
      - use pos_pred[ap] for ap
      - use neg_pred[ap] for !ap
    Only applies to APs in this dimension; contradictions -> all zeros.
    """
    lits_all = parse_literals(letter)
    lits = [(ap, neg) for ap, neg in lits_all if ap in aps_in_dim]

    # contradiction check within *this* dimension
    seen: Dict[str, bool] = {}
    for ap, neg in lits:
        if ap in seen and seen[ap] != neg:
            return [0] * int(np.asarray(xax).size)
        seen[ap] = neg

    # unconstrained on this dim
    if not lits:
        return [1] * int(np.asarray(xax).size)

    out: List[int] = []
    for x in np.asarray(xax).ravel():
        ok = True
        for ap, neg in lits:
            v = (neg_pred[ap](x) if neg else pos_pred[ap](x))
            ok = ok and v
            if not ok: break
        out.append(1 if ok else 0)
    return out

def label_axis_by_letters_robust(
    xax: Iterable[float],
    letters: List[str],
    aps_in_dim: set,
    ap_regions: Dict[str, Iterable[Tuple[float, float]]],
    eps: Union[float, Dict[str, float]],
) -> Array:
    """Return (len(letters) × len(xax)) robust 0/1 array."""
    xax_arr = np.asarray(xax).ravel()
    lo_dom, hi_dom = float(xax_arr.min()), float(xax_arr.max())
    eps_map = _ensure_eps_map(eps, aps_in_dim)
    pos_pred, neg_pred = make_pos_neg_predicates_robust(ap_regions, (lo_dom, hi_dom), eps_map)
    rows = [
        letter_mask_on_axis_robust(xax_arr, L, aps_in_dim, pos_pred, neg_pred)
        for L in letters
    ]
    return np.asarray(rows, dtype=int)

# ===============================================
# ---- robust one-call API: build L_eps all dims
# ===============================================

def dim_label_eps(sysAbs, sysLTI, letters, eps: Union[float, Dict[str, float]],
                  visualize: bool=False, outdir: Optional[str]=None, prefix: str = "L_eps"):
    """
    Build per-dimension robust label matrices L_eps:
      - positive AP -> shrink by eps_ap
      - negated AP  -> complement(expand by eps_ap)
    Only APs present in that dimension are enforced.

    eps: float or dict[str, float] (per-AP).
    Returns dict (if sysAbs is dict) or list (if sysAbs is list/tuple).
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

    L_map: Dict[int, Array] = {}
    if visualize and outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    for k in keys:
        mdp_k = get_abs(k)
        lti_k = get_lti(k)

        # work on outputs axis (your current choice); fallback to hx if needed
        xax_k = _axis_from_hx(getattr(mdp_k, "outputs", getattr(mdp_k, "hx", None)))
        if xax_k is None:
            raise ValueError("MDPModel has neither 'outputs' nor 'hx' axis available.")

        aps = list(getattr(lti_k, "AP", []))
        regions = list(getattr(lti_k, "regions", []))
        aps_set = set(aps)
        base_regions = {
            ap: _intervals_from_polytope_1d(reg)
            for ap, reg in zip(aps, regions)
        }

        Lk = label_axis_by_letters_robust(
            xax_k, letters, aps_set, base_regions, eps=eps
        ).astype(float)

        L_map[k] = Lk

        if visualize:
            fname = f"{prefix}{k}.png"
            if outdir:
                fname = os.path.join(outdir, fname)
            step = max(1, len(xax_k) // 10)
            plot_binary_matrix(
                Lk.astype(int),
                title=f"{prefix}[{k}] (letters × x{k}-grid, eps={eps})",
                row_labels=letters,
                outfile=fname,
                x_tick_step=step,
            )

    return L_map if return_as_dict else [L_map[k] for k in keys]
