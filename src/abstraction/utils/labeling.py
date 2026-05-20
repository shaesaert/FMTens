# src/abstraction/utils/labeling.py
"""
Per-dimension letter-mask construction (robust labeling under abstraction tolerance).

Builds the 0/1 matrices `L[d]` indexed by (letter, abstract-state) that say
which letters of the DFA alphabet are admissible at each abstract state of
dimension d. Construction supports a robustness parameter `eps`: positive
literals are evaluated against shrunken AP regions and negated literals
against the complement of expanded regions, providing a correct-by-design
tolerance for abstraction error.

Public API
----------
- `dim_label(sysAbs, sysLTI, letters, eps=0.0, visualize=False, ...)`
    Per-dimension label matrices. With `eps=0.0`, exact membership; with
    `eps>0`, the robust shrunk/expanded variant. The two old entry points
    `dim_label` and `dim_label_eps` are unified here.
- `intervals_from_polytope_1d(P)`
    1D H-polytope (or union) to a list of closed intervals.
- `plot_binary_matrix(...)`
    Visualise a binary matrix; used by `dim_label(visualize=True)`.
- `parse_literals(expr)`
    Parse a DFA letter string into [(ap_name, is_negated)] pairs.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


Array = np.ndarray


# ===========================================================================
# Letter parsing
# ===========================================================================
_LITERAL_RE = re.compile(r"(?P<neg>!\s*|not\s+)?(?P<ap>p\d+)\b", re.IGNORECASE)


def parse_literals(expr: str) -> List[Tuple[str, bool]]:
    """Return [(ap_name, is_negated)] found in a DFA letter string."""
    return [
        (m.group("ap"), bool(m.group("neg")))
        for m in _LITERAL_RE.finditer(str(expr))
    ]


# ===========================================================================
# Interval operations (private)
# ===========================================================================
def _normalize_intervals(iv: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sort and merge overlapping/adjacent closed intervals."""
    iv = [(float(a), float(b)) if a <= b else (float(b), float(a)) for a, b in iv]
    iv.sort()
    merged: List[List[float]] = []
    for a, b in iv:
        if not merged:
            merged.append([a, b])
        else:
            _, B = merged[-1]
            if a <= B:                       # closed intervals: touching/overlap -> merge
                merged[-1][1] = max(B, b)
            else:
                merged.append([a, b])
    return [(a, b) for a, b in merged]


def _clip_intervals(iv, lo: float, hi: float) -> List[Tuple[float, float]]:
    """Clip each interval to [lo, hi]; drop any that become empty."""
    out = [(max(lo, a), min(hi, b)) for a, b in iv]
    return _normalize_intervals([(a, b) for a, b in out if a <= b])


def _expand_intervals(iv, eps: float) -> List[Tuple[float, float]]:
    """Expand each interval by `eps` on both sides."""
    if eps <= 0:
        return _normalize_intervals(iv)
    return _normalize_intervals([(a - eps, b + eps) for a, b in iv])


def _shrink_intervals(iv, eps: float) -> List[Tuple[float, float]]:
    """Shrink each interval by `eps` on both sides; drop any that vanish."""
    if eps <= 0:
        return _normalize_intervals(iv)
    out = [(a + eps, b - eps) for a, b in iv]
    return _normalize_intervals([(a, b) for a, b in out if a <= b])


def _complement_intervals(iv, lo: float, hi: float) -> List[Tuple[float, float]]:
    """Complement of `iv` within [lo, hi] (closed intervals)."""
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


# ===========================================================================
# Polytope -> intervals
# ===========================================================================
def intervals_from_polytope_1d(P, atol: float = 1e-12) -> List[Tuple[float, float]]:
    """
    Convert a 1D H-polytope {x | A x <= b} to a list of closed intervals.

    Accepts either a single polytope or a list/tuple of polytopes (in which
    case the returned intervals are the union, normalised). Assumes one
    connected interval per polytope, which holds for typical 1D AP regions
    defined by upper/lower bounds.
    """
    def _one(poly) -> List[Tuple[float, float]]:
        A = np.asarray(poly.A, dtype=float).reshape(-1, 1)
        b = np.asarray(poly.b, dtype=float).ravel()
        lo, hi = -np.inf, np.inf
        for a_val, b_val in zip(A.ravel(), b):
            if abs(a_val) <= atol:
                continue
            bound = b_val / a_val
            if a_val > 0:        # a*x <= b  =>  x <= b/a
                hi = min(hi, bound)
            else:                 # a*x <= b  =>  x >= b/a
                lo = max(lo, bound)
        return [] if lo > hi else [(lo, hi)]

    if isinstance(P, (list, tuple)):
        out: List[Tuple[float, float]] = []
        for poly in P:
            out.extend(_one(poly))
        return _normalize_intervals(out)
    return _one(P)


# ===========================================================================
# Predicate builders (private)
# ===========================================================================
def _ensure_eps_map(
    eps: Union[float, Mapping[str, float]], aps: Iterable[str]
) -> Dict[str, float]:
    """Normalize `eps` (scalar or per-AP mapping) to a {ap: float} dict."""
    if isinstance(eps, (int, float)):
        return {ap: float(eps) for ap in aps}
    return {ap: float(eps.get(ap, 0.0)) for ap in aps}


def _make_pos_neg_predicates(
    ap_regions: Dict[str, Iterable[Tuple[float, float]]],
    domain: Tuple[float, float],
    eps_map: Dict[str, float],
) -> Tuple[
    Dict[str, Callable[[float], bool]],
    Dict[str, Callable[[float], bool]],
]:
    """
    Return (`pos_pred`, `neg_pred`):
        pos_pred[ap](x) -> membership in SHRUNK(ap, eps_ap)
        neg_pred[ap](x):
            eps_ap > 0  -> membership in COMPLEMENT(EXPANDED(ap, eps_ap), domain)
            eps_ap = 0  -> strict logical negation of pos_pred[ap](x)

    The two negation definitions agree in the interior of either set; the
    eps=0 special case avoids a closed-interval boundary double-count that
    would otherwise let a region's endpoint satisfy both `ap` and `!ap`.
    """
    lo_dom, hi_dom = map(float, domain)
    pos_pred: Dict[str, Callable[[float], bool]] = {}
    neg_pred: Dict[str, Callable[[float], bool]] = {}

    def _make_pred(intervals: List[Tuple[float, float]]):
        ivs = list(intervals)
        def f(x) -> bool:
            x0 = float(np.asarray(x).reshape(-1)[0])
            return any(a <= x0 <= b for a, b in ivs)
        return f

    for ap, iv in ap_regions.items():
        ivn = _normalize_intervals(iv)
        eps_ap = eps_map.get(ap, 0.0)

        iv_pos = _clip_intervals(_shrink_intervals(ivn, eps_ap), lo_dom, hi_dom)
        pos_pred[ap] = _make_pred(iv_pos)

        if eps_ap == 0.0:
            # eps=0 special case: closed-interval complement creates a boundary
            # double-count (x = endpoint of R is in BOTH R and its complement
            # in [lo, hi]). Use strict logical negation to recover the exact-
            # membership semantics of the original non-robust path.
            p = pos_pred[ap]
            neg_pred[ap] = lambda x, _p=p: not _p(x)
        else:
            iv_neg_base = _clip_intervals(_expand_intervals(ivn, eps_ap), lo_dom, hi_dom)
            iv_neg = _complement_intervals(iv_neg_base, lo_dom, hi_dom)
            neg_pred[ap] = _make_pred(iv_neg)

    return pos_pred, neg_pred


# ===========================================================================
# Per-axis mask builders (private)
# ===========================================================================
def _letter_mask_on_axis(
    xax: Iterable[float],
    letter: str,
    aps_in_dim: set,
    pos_pred: Dict[str, Callable[[float], bool]],
    neg_pred: Dict[str, Callable[[float], bool]],
) -> List[int]:
    """
    Compute the 0/1 mask along `xax` for one letter on one dimension.

    Only literals whose AP is in `aps_in_dim` are enforced; others are
    passed through unconstrained. A letter containing both `ap` and `!ap`
    for an in-dim AP yields the all-zero mask (contradiction within this
    dimension).
    """
    lits = [(ap, neg) for ap, neg in parse_literals(letter) if ap in aps_in_dim]

    # contradiction within this dimension
    seen: Dict[str, bool] = {}
    for ap, neg in lits:
        if ap in seen and seen[ap] != neg:
            return [0] * int(np.asarray(xax).size)
        seen[ap] = neg

    if not lits:
        return [1] * int(np.asarray(xax).size)

    out: List[int] = []
    for x in np.asarray(xax).ravel():
        ok = True
        for ap, neg in lits:
            v = neg_pred[ap](x) if neg else pos_pred[ap](x)
            ok = ok and v
            if not ok:
                break
        out.append(1 if ok else 0)
    return out


def _label_axis_by_letters(
    xax: Iterable[float],
    letters: Sequence[str],
    aps_in_dim: set,
    ap_regions: Dict[str, Iterable[Tuple[float, float]]],
    eps: Union[float, Mapping[str, float]],
) -> Array:
    """Return a (n_letters × n_axis) 0/1 array."""
    xax_arr = np.asarray(xax).ravel()
    lo_dom, hi_dom = float(xax_arr.min()), float(xax_arr.max())
    eps_map = _ensure_eps_map(eps, aps_in_dim)
    pos_pred, neg_pred = _make_pos_neg_predicates(ap_regions, (lo_dom, hi_dom), eps_map)
    rows = [
        _letter_mask_on_axis(xax_arr, L, aps_in_dim, pos_pred, neg_pred)
        for L in letters
    ]
    return np.asarray(rows, dtype=int)


def _axis_from_outputs_or_hx(mdp) -> Array:
    """Return the 1D axis from `mdp.outputs` (preferred) or `mdp.hx` (fallback)."""
    axis = getattr(mdp, "outputs", None)
    if axis is None:
        axis = getattr(mdp, "hx", None)
    if axis is None:
        raise ValueError("MDPModel has neither 'outputs' nor 'hx' axis available.")
    if isinstance(axis, (list, tuple)):
        if len(axis) != 1:
            raise ValueError(f"expected a single axis, got {len(axis)}")
        return np.asarray(axis[0], dtype=float).ravel()
    return np.asarray(axis, dtype=float).ravel()


# ===========================================================================
# Visualization
# ===========================================================================
def plot_binary_matrix(
    mat: Array,
    title: str = "",
    row_labels: Optional[Sequence[str]] = None,
    outfile: Optional[str] = None,
    x_tick_step: Optional[int] = None,
    show: bool = True,
) -> None:
    """Render a 0/1 matrix as a heatmap (red=0, green=1) for inspection."""
    A = np.asarray(mat, dtype=int)
    n_rows, n_cols = A.shape
    cmap = ListedColormap(["red", "green"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    w = min(max(n_cols / 80.0, 8.0), 24.0)
    h = min(max(n_rows * 0.6, 2.8), 12.0)
    fig, ax = plt.subplots(figsize=(w, h), dpi=120)

    ax.imshow(A, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=12)

    ax.set_yticks(np.arange(n_rows))
    if row_labels is not None and len(row_labels) == n_rows:
        ax.set_yticklabels(row_labels, fontsize=9)
    else:
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


# ===========================================================================
# Public API
# ===========================================================================
def dim_label(
    sysAbs,
    sysLTI,
    letters: Sequence[str],
    *,
    eps: Union[float, Mapping[str, float]] = 0.0,
    visualize: bool = False,
    outdir: Optional[str] = None,
    prefix: str = "L",
) -> Union[Dict[int, Array], List[Array]]:
    """
    Build per-dimension letter-mask matrices `L[d]` for the given DFA letters.

    Each `L[d]` has shape (len(letters), N_d), where N_d is the number of
    abstract states in dimension d. Entry `L[d][l, n] = 1` iff letter `l`
    is admissible at abstract state `n` in dimension `d`.

    Parameters
    ----------
    sysAbs : dict[int, MDPModel] or list[MDPModel]
        Per-dimension abstract MDPs.
    sysLTI : dict[int, LinModel] or list[LinModel]
        Per-dimension continuous LTI systems (provides AP names and regions).
    letters : sequence of str
        DFA letters as Boolean expressions over atomic propositions.
    eps : float or dict[str, float], default 0.0
        Robustness tolerance. With `eps=0`, exact membership is used. With
        `eps>0`, positive literals are evaluated against shrunken AP regions
        and negated literals against the complement of expanded regions.
        A dict can specify a different eps per AP.
    visualize : bool, default False
        If True, save a heatmap per dimension to `outdir`.
    outdir : str, optional
        Directory for the visualisation PNGs (created if missing).
    prefix : str, default "L"
        Filename prefix; output is `{outdir}/{prefix}{d}.png`.

    Returns
    -------
    dict or list
        Dict keyed by dimension if `sysAbs` is a dict; otherwise a list in
        dimension order.
    """
    if isinstance(sysAbs, dict):
        keys = sorted(sysAbs.keys())
        return_as_dict = True
    else:
        keys = list(range(len(sysAbs)))
        return_as_dict = False

    if visualize and outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    L_map: Dict[int, Array] = {}

    for k in keys:
        mdp_k = sysAbs[k]
        lti_k = sysLTI[k]

        xax_k = _axis_from_outputs_or_hx(mdp_k)
        aps = list(getattr(lti_k, "AP", []))
        regions = list(getattr(lti_k, "regions", []))
        aps_set = set(aps)
        regions_map = {ap: intervals_from_polytope_1d(reg) for ap, reg in zip(aps, regions)}

        Lk = _label_axis_by_letters(xax_k, letters, aps_set, regions_map, eps).astype(float)
        L_map[k] = Lk

        if visualize:
            fname = f"{prefix}{k}.png"
            if outdir:
                fname = os.path.join(outdir, fname)
            step = max(1, len(xax_k) // 10)
            plot_binary_matrix(
                Lk.astype(int),
                title=f"{prefix}[{k}] (letters × x{k}-grid, eps={eps})",
                row_labels=list(letters),
                outfile=fname,
                x_tick_step=step,
            )

    return L_map if return_as_dict else [L_map[k] for k in keys]