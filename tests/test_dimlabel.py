# tests/test_dimlabel.py
"""
Tests for `src.abstraction.utils.labeling.dim_label` — covers both the 1D
interval-based path and the multi-D H-polytope-membership path.
"""

import warnings
import pytest
import numpy as np
import polytope as pc
from types import SimpleNamespace

from src.abstraction.utils.labeling import (
    dim_label,
    intervals_from_polytope_1d,
    parse_literals,
)


# ============================================================================
# Helpers
# ============================================================================
def _interval_poly(a, b):
    """Return a 1D [a, b] polytope in A x <= b form."""
    A  = np.array([[1.0], [-1.0]])
    bb = np.array([b, -a], dtype=float)
    return pc.Polytope(A, bb)


def _rect_poly(x_lo, x_hi, y_lo, y_hi):
    """Return a 2D rectangle [x_lo, x_hi] × [y_lo, y_hi] as an H-polytope."""
    A = np.array([
        [-1.0,  0.0],
        [ 1.0,  0.0],
        [ 0.0, -1.0],
        [ 0.0,  1.0],
    ])
    b = np.array([-x_lo, x_hi, -y_lo, y_hi], dtype=float)
    return pc.Polytope(A, b)


class DummyAbs:
    """Minimal 1D abstraction stub exposing `hx` like MDPModel."""
    def __init__(self, axis):
        self.hx = [np.asarray(axis, dtype=float).ravel()]


class DummyAbs2D:
    """Multi-D abstraction stub exposing `outputs` of shape (n_dim, N)."""
    def __init__(self, states):
        self.outputs = np.asarray(states, dtype=float)


# ============================================================================
# 1.  parse_literals — direct unit test
# ============================================================================
def test_parse_literals_basic():
    assert parse_literals("p1")             == [("p1", False)]
    assert parse_literals("!p1")            == [("p1", True)]
    assert parse_literals("! p1")           == [("p1", True)]
    assert parse_literals("not p1")         == [("p1", True)]
    assert parse_literals("p1 & !p2")       == [("p1", False), ("p2", True)]
    assert parse_literals("p1 & p2 & !p3")  == [("p1", False), ("p2", False), ("p3", True)]
    assert parse_literals("")               == []
    assert parse_literals("1")              == []   # tautology, no APs
    assert parse_literals("true")           == []   # not a p\d+ match


# ============================================================================
# 2.  intervals_from_polytope_1d — single, union, merge
# ============================================================================
def test_intervals_from_polytope_1d_handles_union_and_merge():
    P1 = _interval_poly(0.0, 1.0)
    P2 = _interval_poly(2.0, 3.0)
    P3 = _interval_poly(1.0, 2.5)        # overlaps both

    assert intervals_from_polytope_1d(P1)            == [(0.0, 1.0)]
    assert intervals_from_polytope_1d([P1, P2])      == [(0.0, 1.0), (2.0, 3.0)]
    assert intervals_from_polytope_1d([P1, P3])      == [(0.0, 2.5)]
    assert intervals_from_polytope_1d([P1, P3, P2])  == [(0.0, 3.0)]


# ============================================================================
# 3.  Shapes & per-dimension coverage (existing test)
# ============================================================================
def test_dim_label_shapes_and_coverage_per_dim():
    ax0 = np.linspace(-5, 5, 10)         # N0 = 10
    ax1 = np.linspace(-2, 3, 7)          # N1 = 7
    sysAbs = {0: DummyAbs(ax0), 1: DummyAbs(ax1)}

    # dim 0 has AP p1: true on [0, 5]
    # dim 1 has AP p2: true on [-2, 0]
    P1_dim0 = _interval_poly(0.0, 5.0)
    P2_dim1 = _interval_poly(-2.0, 0.0)
    sysLTI = {
        0: SimpleNamespace(AP=['p1'], regions=[P1_dim0]),
        1: SimpleNamespace(AP=['p2'], regions=[P2_dim1]),
    }

    letters = ['p1', '!p1', 'p2', '!p2']
    L = dim_label(sysAbs, sysLTI, letters, visualize=False)
    assert isinstance(L, dict) and set(L.keys()) == {0, 1}

    # Shapes
    assert L[0].shape == (len(letters), ax0.size)
    assert L[1].shape == (len(letters), ax1.size)

    # Coverage ONLY for APs defined in the dimension:
    np.testing.assert_allclose(L[0][0, :] + L[0][1, :], 1.0)   # p1 + !p1 in dim 0
    np.testing.assert_allclose(L[1][2, :] + L[1][3, :], 1.0)   # p2 + !p2 in dim 1

    # APs not in the dim are unconstrained → all ones
    assert np.all(L[0][2, :] == 1.0) and np.all(L[0][3, :] == 1.0)
    assert np.all(L[1][0, :] == 1.0) and np.all(L[1][1, :] == 1.0)


# ============================================================================
# 4.  Letter consistency for a single dimension (existing test)
# ============================================================================
def test_letter_consistency_in_dim():
    ax = np.linspace(-1.0, 1.0, 9)
    sysAbs = {0: DummyAbs(ax)}

    P1     = _interval_poly(0.0, 1.0)
    sysLTI = {0: SimpleNamespace(AP=['p1'], regions=[P1])}

    letters = ['p1', '!p1']
    L = dim_label(sysAbs, sysLTI, letters, visualize=False)
    L0 = L[0]
    assert L0.shape == (2, ax.size)

    in_p1 = ((ax >= 0.0) & (ax <= 1.0)).astype(float)
    np.testing.assert_allclose(L0[0, :], in_p1)
    np.testing.assert_allclose(L0[1, :], 1.0 - in_p1)


# ============================================================================
# 5.  1D path — edge cases
# ============================================================================
def test_dim_label_empty_letter_gives_all_ones():
    """A letter with no AP literals (e.g. '1') is unconstrained."""
    ax     = np.linspace(-1, 1, 5)
    sysAbs = {0: DummyAbs(ax)}
    sysLTI = {0: SimpleNamespace(AP=[], regions=[])}

    L = dim_label(sysAbs, sysLTI, ['1', 'true'], visualize=False)
    np.testing.assert_array_equal(L[0][0, :], np.ones(5))
    np.testing.assert_array_equal(L[0][1, :], np.ones(5))


def test_dim_label_contradiction_within_dim_is_zero():
    """`p1 & !p1` is unsatisfiable for any in-dim AP → all-zero mask."""
    ax     = np.linspace(-1, 1, 5)
    sysAbs = {0: DummyAbs(ax)}
    P1     = _interval_poly(0.0, 1.0)
    sysLTI = {0: SimpleNamespace(AP=['p1'], regions=[P1])}

    L = dim_label(sysAbs, sysLTI, ['p1 & !p1'], visualize=False)
    np.testing.assert_array_equal(L[0][0, :], np.zeros(5))


def test_dim_label_conjunction_within_dim():
    """`p1 & p2` is the intersection of both regions along this axis."""
    ax     = np.linspace(-2.0, 4.0, 13)              # step 0.5
    sysAbs = {0: DummyAbs(ax)}
    P1     = _interval_poly(0.0, 3.0)                 # [0, 3]
    P2     = _interval_poly(1.0, 4.0)                 # [1, 4]
    sysLTI = {0: SimpleNamespace(AP=['p1', 'p2'], regions=[P1, P2])}

    L = dim_label(sysAbs, sysLTI, ['p1 & p2'], visualize=False)
    expected = ((ax >= 1.0) & (ax <= 3.0)).astype(int)
    np.testing.assert_array_equal(L[0][0, :], expected)


def test_dim_label_1d_eps_shrinks_positive_and_forbids_boundary():
    """
    Robust labeling (eps > 0) on the 1D path:
      - p1 region [0, 2] shrinks to [0+eps, 2-eps]
      - !p1 = COMPLEMENT(EXPAND(p1, eps), domain)
      - At exact region boundaries the point belongs to NEITHER p1 nor !p1
        — the "forbidden zone" that makes the abstraction robust.
    """
    ax     = np.array([-0.5, 0.0, 1.0, 2.0, 2.5])
    sysAbs = {0: DummyAbs(ax)}
    P1     = _interval_poly(0.0, 2.0)
    sysLTI = {0: SimpleNamespace(AP=['p1'], regions=[P1])}

    L = dim_label(sysAbs, sysLTI, ['p1', '!p1'], eps=0.1, visualize=False)

    # p1 shrunk to [0.1, 1.9]
    np.testing.assert_array_equal(L[0][0, :], [0, 0, 1, 0, 0])
    # !p1 outside the expanded [-0.1, 2.1] within domain [-0.5, 2.5]
    np.testing.assert_array_equal(L[0][1, :], [1, 0, 0, 0, 1])

    # Forbidden zone: at exact boundary points x=0 and x=2, BOTH masks are 0
    assert L[0][0, 1] == 0 and L[0][1, 1] == 0, "x=0.0 should be neither p nor !p"
    assert L[0][0, 3] == 0 and L[0][1, 3] == 0, "x=2.0 should be neither p nor !p"


def test_dim_label_list_input_returns_list():
    """sysAbs/sysLTI as a list (not a dict) → returned as a list, same order."""
    ax     = np.linspace(-1, 1, 5)
    sysAbs = [DummyAbs(ax)]
    P1     = _interval_poly(0.0, 1.0)
    sysLTI = [SimpleNamespace(AP=['p1'], regions=[P1])]

    L = dim_label(sysAbs, sysLTI, ['p1'], visualize=False)
    assert isinstance(L, list) and len(L) == 1
    assert L[0].shape == (1, 5)


# ============================================================================
# 6.  Multi-D path (2D state) — the entirely new code path
# ============================================================================
def test_dim_label_2d_rectangle_membership():
    """Direct H-polytope membership on a 2D state grid."""
    xs = np.array([-2.0, -1.0, 0.5, 1.5, 2.5])
    ys = np.array([-1.0,  0.5, 1.5, 2.5])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    states = np.vstack([X.ravel(), Y.ravel()])           # (2, 20)

    sysAbs = {0: DummyAbs2D(states)}
    P_rect = _rect_poly(0.0, 2.0, 0.0, 2.0)
    sysLTI = {0: SimpleNamespace(AP=['p1'], regions=[P_rect])}

    L = dim_label(sysAbs, sysLTI, ['p1', '!p1'], visualize=False)

    expected_p1 = np.array([
        (0.0 <= float(x) <= 2.0) and (0.0 <= float(y) <= 2.0)
        for x, y in zip(states[0], states[1])
    ]).astype(int)

    np.testing.assert_array_equal(L[0][0, :], expected_p1)
    np.testing.assert_array_equal(L[0][1, :], 1 - expected_p1)


def test_dim_label_2d_contradiction_gives_zeros():
    """`p1 & !p1` on the multi-D path → all-zero mask."""
    states = np.array([[0.5, 1.5, 2.5], [0.5, 1.5, 2.5]])    # (2, 3)
    sysAbs = {0: DummyAbs2D(states)}
    P_rect = _rect_poly(0.0, 2.0, 0.0, 2.0)
    sysLTI = {0: SimpleNamespace(AP=['p1'], regions=[P_rect])}

    L = dim_label(sysAbs, sysLTI, ['p1 & !p1'], visualize=False)
    np.testing.assert_array_equal(L[0][0, :], np.zeros(3, dtype=int))


def test_dim_label_2d_eps_warns_and_uses_exact_membership():
    """eps > 0 on the multi-D path: warns and falls back to exact membership."""
    states = np.array([[0.5, 1.5, 2.5], [0.5, 1.5, 2.5]])    # (2, 3)
    sysAbs = {0: DummyAbs2D(states)}
    P_rect = _rect_poly(0.0, 2.0, 0.0, 2.0)
    sysLTI = {0: SimpleNamespace(AP=['p1'], regions=[P_rect])}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        L = dim_label(sysAbs, sysLTI, ['p1'], eps=0.1, visualize=False)

    assert any("not implemented" in str(rec.message) for rec in w), (
        f"expected warning about multi-D eps>0, got "
        f"{[str(rec.message) for rec in w]}"
    )

    expected_p1 = np.array([
        (0.0 <= float(x) <= 2.0) and (0.0 <= float(y) <= 2.0)
        for x, y in zip(states[0], states[1])
    ]).astype(int)
    np.testing.assert_array_equal(L[0][0, :], expected_p1)


def test_dim_label_2d_dimensionality_mismatch_raises():
    """1D polytope on a 2D state grid → ValueError, not silent corruption."""
    states = np.array([[0.0, 1.0], [0.0, 1.0]])              # (2, 2)
    sysAbs = {0: DummyAbs2D(states)}
    P_1d   = _interval_poly(0.0, 1.0)                         # 1D polytope
    sysLTI = {0: SimpleNamespace(AP=['p1'], regions=[P_1d])}

    with pytest.raises(ValueError, match="polytope is.*states are"):
        dim_label(sysAbs, sysLTI, ['p1'], visualize=False)