# tests/test_dfatree.py
"""
DFATree unit tests — covers both the 1D and 2D abstraction dispatch paths.

The 1D path uses dense (N, N·nu) flat transition matrices; the 2D path uses
the separable `TransitionProbability2D` operator via `V @ P.stoch`. Tests are
parametrised over both where the underlying DFATree logic is dimension-aware
(maxpolicy, update_tree, Q_n) so the same assertion runs against both stubs.
"""

import numpy as np
import pytest
from types import SimpleNamespace
from scipy.sparse import csr_matrix

from src.dynprog.dfa_tree_r1                                import DFATree
from src.abstraction.utils.tensor_transition_probability_2d import TransitionProbability2D


# ============================================================================
# Stubs
# ============================================================================
class DummyAbs:
    """1D abstraction stub: dense flat P, used to exercise the 1D dispatch."""

    def __init__(self, P_flat):
        self.P      = np.asarray(P_flat, dtype=float)
        self.P_flat = self.P                # _transition_arg uses .P_flat for 1D
        self.N      = P_flat.shape[0]
        self.M      = P_flat.shape[1] // self.N
        self.dim    = 1


class DummyAbs2D:
    """
    2D abstraction stub built on the real TransitionProbability2D operator.
    Identity per-axis kernels (Pi0 = I_Np, Pi1 = I_Nv) make the noise smearing
    a no-op, so V @ P.stoch reduces to V @ P_det. Keeps the DFATree dispatch
    on .dim == 2 / .stoch routing real while isolating the test from any
    numerical subtleties of the noise kernel.
    """

    def __init__(self, Np, Nv, nu, P_det):
        N        = Np * Nv
        self.N   = N
        self.M   = nu
        self.dim = 2
        self.P   = TransitionProbability2D(
            (Np, Nv), P_det,
            Pi0=np.eye(Np), Pi1=np.eye(Nv),
        )
        self.P_flat = None                  # unused in the 2D path


def _build_indicator_P_det(Np, Nv, nu, shift_fn=None):
    """
    Sparse (N, N·nu) indicator: P_det[shift(src, u), src + u·N] = 1.
    Default shift maps (src, u) -> (src + u) mod N — arbitrary but deterministic.
    """
    N = Np * Nv
    if shift_fn is None:
        shift_fn = lambda src, u: (src + u) % N
    rows = [shift_fn(s, u) for s in range(N) for u in range(nu)]
    cols = [s + u * N     for s in range(N) for u in range(nu)]
    data = np.ones(len(rows), dtype=float)
    return csr_matrix((data, (rows, cols)), shape=(N, N * nu))


# ============================================================================
# Shared fixtures
# ============================================================================
@pytest.fixture
def tiny_dfa():
    """Minimal DFA: S={0(F),1}, one letter l0, both states transition to F."""
    return SimpleNamespace(
        S=[0, 1],
        F=0,
        sink=None,
        act=['l0'],
        trans=np.array([[0],
                        [0]], dtype=int),
    )


# ----- 1D fixtures -----
@pytest.fixture
def tiny_abs():
    """1D abstract MDP, N=3, nu=2, row-stochastic blocks."""
    rng = np.random.default_rng(0)
    N, nu = 3, 2
    blocks = []
    for _ in range(nu):
        B = rng.random((N, N))
        B /= B.sum(axis=1, keepdims=True)
        blocks.append(B)
    P_flat = np.hstack(blocks)
    return DummyAbs(P_flat)


@pytest.fixture
def L_mask(tiny_abs):
    """One letter, all-ones mask (no labeling constraint)."""
    return np.ones((1, tiny_abs.N), dtype=float)


@pytest.fixture
def uniform_pol(tiny_abs, tiny_dfa):
    """pol[q][d] shape (N, nu), uniform across actions."""
    N  = tiny_abs.N
    nu = tiny_abs.M
    return [[np.full((N, nu), 1.0 / nu, dtype=float)] for _ in tiny_dfa.S]


# ----- 2D fixtures -----
@pytest.fixture
def tiny_abs_2d():
    """2D abstract MDP, Np=3, Nv=2 -> N=6, nu=2."""
    Np, Nv, nu = 3, 2, 2
    return DummyAbs2D(Np, Nv, nu, _build_indicator_P_det(Np, Nv, nu))


@pytest.fixture
def L_mask_2d(tiny_abs_2d):
    return np.ones((1, tiny_abs_2d.N), dtype=float)


@pytest.fixture
def uniform_pol_2d(tiny_abs_2d, tiny_dfa):
    N, nu = tiny_abs_2d.N, tiny_abs_2d.M
    return [[np.full((N, nu), 1.0 / nu, dtype=float)] for _ in tiny_dfa.S]


# ============================================================================
# Existing structural tests (1D)
# ============================================================================
def test_initiate_builds_root_and_children(tiny_dfa, tiny_abs, L_mask, uniform_pol):
    G = DFATree(tiny_dfa, {0: tiny_abs}, uniform_pol,
                [tiny_abs.N], [L_mask]).initiate()

    # root exists and carries F
    assert 0 in G.tree
    assert G.Lq(0) == tiny_dfa.F

    # two children of the root: q=0 and q=1
    assert G.tree.number_of_nodes() == 3
    succ = list(G.tree.successors(0))
    assert len(succ) == 2
    child_labels = {G.Lq(n) for n in succ}
    assert child_labels == {0, 1}

    # value-table sanity: root row ones, children zeros
    for d in range(G.dim):
        np.testing.assert_allclose(G.V[d][0, :], 1.0)
        for n in succ:
            np.testing.assert_allclose(G.V[d][n, :], 0.0)


def test_grow_adds_predecessors(tiny_dfa, tiny_abs, L_mask, uniform_pol):
    G = DFATree(tiny_dfa, {0: tiny_abs}, uniform_pol,
                [tiny_abs.N], [L_mask]).initiate()
    before = G.tree.number_of_nodes()
    G.grow()
    after = G.tree.number_of_nodes()
    assert after > before
    assert 1 not in G.leafs


def test_growleaf_by_id_from_current_leafs(tiny_dfa, tiny_abs, L_mask, uniform_pol):
    G = DFATree(tiny_dfa, {0: tiny_abs}, uniform_pol,
                [tiny_abs.N], [L_mask]).initiate()
    n = G.leafs[0]
    G.growleaf(n)
    assert n in G.tree
    assert len(list(G.tree.successors(n))) > 0


def test_removeBranch_relabels_consistently(tiny_dfa, tiny_abs, L_mask, uniform_pol):
    G = DFATree(tiny_dfa, {0: tiny_abs}, uniform_pol,
                [tiny_abs.N], [L_mask]).initiate()
    G.grow()

    root_children = list(G.tree.successors(0))
    assert root_children
    G.removeBranch([root_children[0]])

    # relabeled nodes are contiguous 0..N-1
    new_nodes = sorted(G.tree.nodes)
    assert new_nodes == list(range(len(new_nodes)))
    # V rows must match new number of nodes
    for d in range(G.dim):
        assert G.V[d].shape[0] == len(new_nodes)


def test_plot_runs_without_error(tiny_dfa, tiny_abs, L_mask, uniform_pol):
    # Smoke test only — uses matplotlib; skip in CI if needed.
    G = DFATree(tiny_dfa, {0: tiny_abs}, uniform_pol,
                [tiny_abs.N], [L_mask]).initiate()
    G.plot(use_letters=False)


# ============================================================================
# 2D dispatch coverage — new tests
# ============================================================================
def test_nu_of_dim_1d(tiny_dfa, tiny_abs, L_mask, uniform_pol):
    G = DFATree(tiny_dfa, {0: tiny_abs}, uniform_pol,
                [tiny_abs.N], [L_mask]).initiate()
    assert G._nu_of_dim(0) == tiny_abs.M


def test_nu_of_dim_2d(tiny_dfa, tiny_abs_2d, L_mask_2d, uniform_pol_2d):
    G = DFATree(tiny_dfa, {0: tiny_abs_2d}, uniform_pol_2d,
                [tiny_abs_2d.N], [L_mask_2d]).initiate()
    assert G._nu_of_dim(0) == tiny_abs_2d.M


def test_initiate_works_with_2d_abs(tiny_dfa, tiny_abs_2d,
                                    L_mask_2d, uniform_pol_2d):
    G = DFATree(tiny_dfa, {0: tiny_abs_2d}, uniform_pol_2d,
                [tiny_abs_2d.N], [L_mask_2d]).initiate()

    assert 0 in G.tree and G.Lq(0) == tiny_dfa.F
    assert G.tree.number_of_nodes() == 3
    np.testing.assert_allclose(G.V[0][0, :], 1.0)
    for n in G.leafs:
        np.testing.assert_allclose(G.V[0][n, :], 0.0)


@pytest.mark.parametrize("abs_fixture,L_fixture,pol_fixture", [
    ("tiny_abs",    "L_mask",    "uniform_pol"),
    ("tiny_abs_2d", "L_mask_2d", "uniform_pol_2d"),
])
def test_maxpolicy_sets_Pxx_for_non_final_states(
        tiny_dfa, request, abs_fixture, L_fixture, pol_fixture
):
    a   = request.getfixturevalue(abs_fixture)
    L_  = request.getfixturevalue(L_fixture)
    pol = request.getfixturevalue(pol_fixture)
    G   = DFATree(tiny_dfa, {0: a}, pol, [a.N], [L_]).initiate()

    rho = [np.ones(a.N) / a.N]
    G.maxpolicy(rho)

    for q in tiny_dfa.S:
        if int(q) == tiny_dfa.F:
            continue
        assert G.Pxx[int(q)][0] is not None, (
            f"Pxx not set for q={q} on {abs_fixture}"
        )


@pytest.mark.parametrize("abs_fixture,L_fixture,pol_fixture", [
    ("tiny_abs",    "L_mask",    "uniform_pol"),
    ("tiny_abs_2d", "L_mask_2d", "uniform_pol_2d"),
])
def test_update_tree_propagates_finite_values(
        tiny_dfa, request, abs_fixture, L_fixture, pol_fixture
):
    a   = request.getfixturevalue(abs_fixture)
    L_  = request.getfixturevalue(L_fixture)
    pol = request.getfixturevalue(pol_fixture)
    G   = DFATree(tiny_dfa, {0: a}, pol, [a.N], [L_]).initiate()

    rho = [np.ones(a.N) / a.N]
    G.maxpolicy(rho)        # populates Pxx via the .dim-dispatched path
    G.update_tree()         # exercises update_node_value via the same dispatch

    for n in G.tree.nodes:
        if n == 0:
            continue
        v = G.V[0][n, :]
        assert v.shape == (a.N,)
        assert np.isfinite(v).all(), (
            f"non-finite V at node {n} on {abs_fixture}"
        )


@pytest.mark.parametrize("abs_fixture,L_fixture,pol_fixture", [
    ("tiny_abs",    "L_mask",    "uniform_pol"),
    ("tiny_abs_2d", "L_mask_2d", "uniform_pol_2d"),
])
def test_Q_n_shape(tiny_dfa, request, abs_fixture, L_fixture, pol_fixture):
    a   = request.getfixturevalue(abs_fixture)
    L_  = request.getfixturevalue(L_fixture)
    pol = request.getfixturevalue(pol_fixture)
    G   = DFATree(tiny_dfa, {0: a}, pol, [a.N], [L_]).initiate()

    child = next(iter(G.tree.successors(0)))
    Qv    = G.Q_n(child)

    assert isinstance(Qv, list) and len(Qv) == G.dim
    for d in range(G.dim):
        assert Qv[d].shape == (G.nx[d], G._nu_of_dim(d))


def test_VI_mode_apos_runs_on_2d(tiny_dfa, tiny_abs_2d,
                                 L_mask_2d, uniform_pol_2d):
    """VI_mode='apos' skips the delta_VI subtraction in update_node_value;
    cover this branch on the 2D dispatch path."""
    G = DFATree(tiny_dfa, {0: tiny_abs_2d}, uniform_pol_2d,
                [tiny_abs_2d.N], [L_mask_2d],
                VI_mode="apos").initiate()

    rho = [np.ones(tiny_abs_2d.N) / tiny_abs_2d.N]
    G.maxpolicy(rho)
    G.update_tree()

    for n in G.tree.nodes:
        if n != 0:
            assert np.isfinite(G.V[0][n, :]).all()