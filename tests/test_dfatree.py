import numpy as np
from types import SimpleNamespace
import pytest
from src.dynprog.dfa_tree_r1 import DFATree

# --- minimal DFA: S={0(F),1}, one letter l0, transitions to F on l0
@pytest.fixture
def tiny_dfa():
    return SimpleNamespace(
        S=[0, 1],
        F=0,
        sink=None,
        act=['l0'],
        trans=np.array([[0],
                        [0]], dtype=int)  # from 0->0, 1->0 under l0
    )

# --- 1D “abs” stub with flat P
class DummyAbs:
    def __init__(self, P_flat):
        self.P = np.asarray(P_flat, dtype=float)
        self.N = P_flat.shape[0]
        self.M = P_flat.shape[1] // self.N

@pytest.fixture
def tiny_abs():
    # N=3, nu=2, row-stochastic blocks
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
    # one letter, all-ones mask (no constraint)
    N = tiny_abs.N
    return np.ones((1, N), dtype=float)

@pytest.fixture
def uniform_pol(tiny_abs, tiny_dfa):
    N = tiny_abs.N
    nu = tiny_abs.M
    # pol[q][d] shape (N,nu), 1 dim only
    return [[np.full((N, nu), 1.0/nu, dtype=float)] for _ in tiny_dfa.S]

def test_initiate_builds_root_and_children(tiny_dfa, tiny_abs, L_mask, uniform_pol):
    G = DFATree(tiny_dfa, {0: tiny_abs}, uniform_pol, [tiny_abs.N], [L_mask]).initiate()

    # root exists and carries F
    assert 0 in G.tree
    assert G.Lq(0) == tiny_dfa.F

    # there should be two children of the root: q=0 and q=1
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
    G = DFATree(tiny_dfa, {0: tiny_abs}, uniform_pol, [tiny_abs.N], [L_mask]).initiate()
    # Grow from the only leaf; it should create its predecessors under l0 (which are both states)
    before = G.tree.number_of_nodes()
    G.grow()
    after = G.tree.number_of_nodes()
    assert after > before
    # The old leaf should no longer be a leaf
    assert 1 not in G.leafs

def test_growleaf_by_id_from_current_leafs(tiny_dfa, tiny_abs, L_mask, uniform_pol):
    G = DFATree(tiny_dfa, {0: tiny_abs}, uniform_pol, [tiny_abs.N], [L_mask]).initiate()
    # pick an actual current leaf instead of a magic number
    n = G.leafs[0]
    G.growleaf(n)
    assert n in G.tree  # still present
    # it must have at least one child
    assert len(list(G.tree.successors(n))) > 0

def test_removeBranch_relabels_consistently(tiny_dfa, tiny_abs, L_mask, uniform_pol):
    G = DFATree(tiny_dfa, {0: tiny_abs}, uniform_pol, [tiny_abs.N], [L_mask]).initiate()
    # grow once to create deeper nodes
    G.grow()
    old_nodes = sorted(G.tree.nodes)
    # remove subtree rooted at the first non-root node
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
    # Non-assertive smoke test (you may skip in CI): just ensure no exceptions
    G = DFATree(tiny_dfa, {0: tiny_abs}, uniform_pol, [tiny_abs.N], [L_mask]).initiate()
    G.plot(use_letters=False)
