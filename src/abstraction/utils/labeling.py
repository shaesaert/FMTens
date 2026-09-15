# src/abstraction/utils/labeling.py
"""
Per-dimension letter-mask construction (robust labeling under abstraction tolerance).

Builds the 0/1 matrices `L[d]` indexed by (letter, abstract-state) that say
which letters of the DFA alphabet are admissible at each abstract state of
dimension d.

Two construction paths, dispatched on the per-agent output dimensionality:

    - 1D output  : evaluate per-axis interval membership with optional
      robustness eps (shrink positives / expand negatives). This is the
      historical path used by RA2D / RA4D where `C` projects to a single
      position coordinate.
    - n-D output : evaluate H-polytope membership `A x <= b` directly on
      each abstract state. Used when `C = I_d` and labeling regions live
      in the full d-dimensional output space (e.g. planar reach-avoid).
      Robust labeling with eps > 0 is not currently implemented for the
      multi-D path; eps > 0 emits a warning and falls back to exact
      membership for those APs.

Public API
----------
- `dim_label(sysAbs, sysLTI, letters, eps=0.0, visualize=False, ...)`
    Per-dimension label matrices. With `eps=0.0`, exact membership; with
    `eps>0` on a 1D output, the robust shrunk/expanded variant.
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
import warnings
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
# Predicate builders (private) — 1D path
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
# Per-axis mask builders (private) — 1D path
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
    """Return a (n_letters × n_axis) 0/1 array, 1D path."""
    xax_arr = np.asarray(xax).ravel()
    lo_dom, hi_dom = float(xax_arr.min()), float(xax_arr.max())
    eps_map = _ensure_eps_map(eps, aps_in_dim)
    pos_pred, neg_pred = _make_pos_neg_predicates(ap_regions, (lo_dom, hi_dom), eps_map)
    rows = [
        _letter_mask_on_axis(xax_arr, L, aps_in_dim, pos_pred, neg_pred)
        for L in letters
    ]
    return np.asarray(rows, dtype=int)


# ===========================================================================
# Multi-D path: direct H-polytope membership
# ===========================================================================
def _label_states_by_letters_nd(
    states: Array,
    letters: Sequence[str],
    aps_in_dim: set,
    ap_polys: Dict[str, "pc.Polytope"],   # noqa: F821 — forward type, never imported here
    eps: Union[float, Mapping[str, float]],
    atol: float = 1e-9,
) -> Array:
    """
    Multi-dim labeling via direct H-polytope membership.

    Parameters
    ----------
    states : (n_dim, N) ndarray
        Each column is one abstract state in the agent's output space.
    letters : sequence of str
        DFA letters as Boolean expressions over atomic propositions.
    aps_in_dim : set of str
        APs this agent's labeling matrix actually enforces.
    ap_polys : dict {ap -> polytope}
        Raw polytope objects; membership tested via `poly.A @ x <= poly.b`.
    eps : float or dict
        Robustness margin. For eps > 0 on this multi-D path, exact
        membership is used and a warning is emitted per AP — robust
        labeling for general polytopes requires Minkowski sum/diff with
        an eps-ball, which is not implemented here.
    atol : float
        Numerical tolerance on the per-constraint check.

    Returns
    -------
    (n_letters, N) int array of 0/1 labels.
    """
    n_dim, N = states.shape
    eps_map = _ensure_eps_map(eps, aps_in_dim)

    # Precompute per-AP boolean membership across all N states (vectorised).
    pos_mask: Dict[str, np.ndarray] = {}
    for ap, poly in ap_polys.items():
        if ap not in aps_in_dim:
            continue
        eps_ap = eps_map.get(ap, 0.0)
        if eps_ap != 0.0:
            warnings.warn(
                f"dim_label: eps > 0 not implemented for multi-D labeling "
                f"of AP {ap!r}; using exact membership.",
                UserWarning,
            )
        A_poly = np.asarray(poly.A, dtype=float)        # (n_constraints, n_dim_poly)
        b_poly = np.asarray(poly.b, dtype=float).ravel()  # (n_constraints,)
        if A_poly.shape[1] != n_dim:
            raise ValueError(
                f"AP {ap!r}: polytope is {A_poly.shape[1]}-D but states are "
                f"{n_dim}-D. Region and state dimensionality must match."
            )
        violation = A_poly @ states - b_poly.reshape(-1, 1)  # (n_constraints, N)
        pos_mask[ap] = np.all(violation <= atol, axis=0)

    rows: List[np.ndarray] = []
    for letter in letters:
        lits = [(ap, neg) for ap, neg in parse_literals(letter) if ap in aps_in_dim]

        # in-dim contradiction
        seen: Dict[str, bool] = {}
        contradiction = False
        for ap, neg in lits:
            if ap in seen and seen[ap] != neg:
                contradiction = True
                break
            seen[ap] = neg
        if contradiction:
            rows.append(np.zeros(N, dtype=int))
            continue

        if not lits:
            rows.append(np.ones(N, dtype=int))
            continue

        ok = np.ones(N, dtype=bool)
        for ap, neg in lits:
            ok = ok & (~pos_mask[ap] if neg else pos_mask[ap])
        rows.append(ok.astype(int))

    return np.asarray(rows, dtype=int)


# ===========================================================================
# Output-axis extraction (shape-preserving)
# ===========================================================================
def _states_from_outputs(mdp) -> Array:
    """
    Return the per-agent abstract states as a `(n_dim, N)` array,
    preserving the original output dimensionality. Used by both the 1D
    and multi-D labeling paths.

    Falls back to `mdp.hx` only when `mdp.outputs` is absent AND `hx` is
    a single 1D axis; multi-D abstractions are required to populate
    `outputs` (the abstraction factory does so automatically).
    """
    out = getattr(mdp, "outputs", None)
    if out is not None:
        arr = np.asarray(out, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        raise ValueError(f"unexpected outputs shape {arr.shape}")

    hx = getattr(mdp, "hx", None)
    if hx is None:
        raise ValueError("MDPModel has neither 'outputs' nor 'hx' available.")
    if isinstance(hx, (list, tuple)):
        if len(hx) != 1:
            raise ValueError(
                f"`hx` has {len(hx)} axes; multi-D labeling requires `outputs` "
                f"to be populated by the abstraction factory."
            )
        return np.asarray(hx[0], dtype=float).reshape(1, -1)
    return np.asarray(hx, dtype=float).reshape(1, -1)


# Kept for backward compatibility with any external caller importing it.
def _axis_from_outputs_or_hx(mdp) -> Array:
    """
    Legacy 1D axis extractor. Prefer :func:`_states_from_outputs`, which
    is shape-preserving. This wrapper still ravels and returns a 1D array
    but raises if the output is genuinely multi-D, so the silent-doubling
    bug for 2D outputs cannot happen.
    """
    states = _states_from_outputs(mdp)
    if states.shape[0] != 1:
        raise ValueError(
            f"_axis_from_outputs_or_hx called on a {states.shape[0]}-D output; "
            f"use _states_from_outputs instead."
        )
    return states[0, :]


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

    Each `L[d]` has shape `(len(letters), N_d)`, where `N_d` is the number of
    abstract states in agent `d`. Entry `L[d][l, n] = 1` iff letter `l` is
    admissible at abstract state `n` of agent `d`.

    Dispatch:
        - if agent `d`'s output is 1D: use the historical interval-based
          path (supports robustness `eps > 0`).
        - if agent `d`'s output is multi-D: use direct H-polytope
          membership on each abstract state. `eps > 0` falls back to exact
          membership with a warning.

    Parameters
    ----------
    sysAbs : dict[int, MDPModel] or list[MDPModel]
        Per-agent abstract MDPs.
    sysLTI : dict[int, LinModel] or list[LinModel]
        Per-agent continuous LTI systems (provides AP names and regions).
    letters : sequence of str
        DFA letters as Boolean expressions over atomic propositions.
    eps : float or dict[str, float], default 0.0
        Robustness tolerance. See dispatch above for the multi-D caveat.
    visualize : bool, default False
        If True, save a heatmap per agent to `outdir`.
    outdir : str, optional
        Directory for the visualisation PNGs (created if missing).
    prefix : str, default "L"
        Filename prefix; output is `{outdir}/{prefix}{d}.png`.

    Returns
    -------
    dict or list
        Dict keyed by agent if `sysAbs` is a dict; otherwise a list in
        agent order.
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

        states_k = _states_from_outputs(mdp_k)   # (n_dim, N)
        n_dim, N = states_k.shape

        aps = list(getattr(lti_k, "AP", []))
        regions = list(getattr(lti_k, "regions", []))
        aps_set = set(aps)

        if n_dim == 1:
            xax_k = states_k[0, :]
            regions_map = {
                ap: intervals_from_polytope_1d(reg)
                for ap, reg in zip(aps, regions)
            }
            Lk = _label_axis_by_letters(
                xax_k, letters, aps_set, regions_map, eps,
            ).astype(float)
        else:
            ap_polys = {ap: reg for ap, reg in zip(aps, regions)}
            Lk = _label_states_by_letters_nd(
                states_k, letters, aps_set, ap_polys, eps,
            ).astype(float)

        L_map[k] = Lk

        if visualize:
            fname = f"{prefix}{k}.png"
            if outdir:
                fname = os.path.join(outdir, fname)
            step = max(1, N // 10)
            plot_binary_matrix(
                Lk.astype(int),
                title=f"{prefix}[{k}] (letters x state-grid, eps={eps}, dim={n_dim})",
                row_labels=list(letters),
                outfile=fname,
                x_tick_step=step,
            )

    return L_map if return_as_dict else [L_map[k] for k in keys]