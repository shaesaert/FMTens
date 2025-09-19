# tests/test_dim_label_core.py
import numpy as np
from types import SimpleNamespace
import polytope as pc

from src.abstraction.utils.labeling import dim_label

# -------------------------
# Helpers
# -------------------------
def _interval_poly(a, b):
    """Return a 1D [a, b] polytope in A x <= b form."""
    A = np.array([[1.0], [-1.0]])
    bb = np.array([b, -a], dtype=float)
    return pc.Polytope(A, bb)


class DummyAbs:
    """Minimal abstraction stub exposing hx like MDPModel."""
    def __init__(self, axis):
        self.hx = [np.asarray(axis, dtype=float).ravel()]


# =====================================================================
# 1) Shapes & coverage (per-dimension, only for APs that exist in that
#    dimension, p + !p == 1 holds; APs not in the dim are unconstrained)
# =====================================================================
def test_dim_label_shapes_and_coverage_per_dim():
    # two 1D axes with different sizes
    ax0 = np.linspace(-5, 5, 10)  # N0=10
    ax1 = np.linspace(-2, 3, 7)   # N1=7
    sysAbs = {0: DummyAbs(ax0), 1: DummyAbs(ax1)}

    # dim 0 has AP p1: true on [0, 5]
    # dim 1 has AP p2: true on [-2, 0]
    P1_dim0 = _interval_poly(0.0, 5.0)
    P2_dim1 = _interval_poly(-2.0, 0.0)
    sysLTI = {
        0: SimpleNamespace(AP=['p1'], regions=[P1_dim0]),
        1: SimpleNamespace(AP=['p2'], regions=[P2_dim1]),
    }

    # global letters set (contains APs not present in each dim as well)
    letters = ['p1', '!p1', 'p2', '!p2']

    L = dim_label(sysAbs, sysLTI, letters, visualize=False)
    assert isinstance(L, dict) and set(L.keys()) == {0, 1}

    # Shapes
    assert L[0].shape == (len(letters), ax0.size)
    assert L[1].shape == (len(letters), ax1.size)

    # Coverage ONLY for APs defined in the dimension:
    # - in dim 0, p1 is defined: p1 + !p1 == 1 for all columns
    np.testing.assert_allclose(L[0][0, :] + L[0][1, :], 1.0, atol=0, rtol=0)

    # - in dim 1, p2 is defined: p2 + !p2 == 1 for all columns
    np.testing.assert_allclose(L[1][2, :] + L[1][3, :], 1.0, atol=0, rtol=0)

    # APs not defined in a dimension are unconstrained → mask of ones
    #   (by design in letter_mask_on_axis)
    # In dim 0, p2 and !p2 should be all ones
    assert np.all(L[0][2, :] == 1.0) and np.all(L[0][3, :] == 1.0)
    # In dim 1, p1 and !p1 should be all ones
    assert np.all(L[1][0, :] == 1.0) and np.all(L[1][1, :] == 1.0)


# ===================================================
# 2) Letter consistency for a single dimension (p/!p)
# ===================================================
def test_letter_consistency_in_dim():
    # one 1D axis
    ax = np.linspace(-1.0, 1.0, 9)  # N=9
    sysAbs = {0: DummyAbs(ax)}

    # AP p1 is true on [0, 1]
    P1 = _interval_poly(0.0, 1.0)
    sysLTI = {0: SimpleNamespace(AP=['p1'], regions=[P1])}

    letters = ['p1', '!p1']  # only local AP and its negation
    L = dim_label(sysAbs, sysLTI, letters, visualize=False)
    L0 = L[0]
    assert L0.shape == (2, ax.size)

    # Build ground-truth mask for membership in [0,1]
    in_p1 = ((ax >= 0.0) & (ax <= 1.0)).astype(float)

    # Row 0 is 'p1' ⇒ 1 on [0,1]
    np.testing.assert_allclose(L0[0, :], in_p1, atol=0, rtol=0)
    # Row 1 is '!p1' ⇒ complement
    np.testing.assert_allclose(L0[1, :], 1.0 - in_p1, atol=0, rtol=0)
