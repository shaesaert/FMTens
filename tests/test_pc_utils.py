import numpy as np
import pytest
from scipy.sparse import csr_matrix

# Adjust the import to match your project layout:
from src.abstraction.utils.pc_utils import Pc



def build_P_flat(blocks):
    """
    Given a list of nu blocks [P0, P1, ..., P_{nu-1}], each N×N,
    horizontally concatenate into shape (N, N*nu).
    """
    return np.hstack(blocks)


def expected_Pcomp_columnwise(P_blocks, policy_rows):
    """
    What your Pc computes: for each row index r, take the r-th column
    from the block chosen by policy_rows[r] and place it as column r.

    P_blocks: list of nu arrays of shape (N, N)
    policy_rows: array of shape (N,), policy_rows[r] = action for row r
    """
    N = P_blocks[0].shape[0]
    E = np.zeros((N, N), dtype=float)
    for r in range(N):
        a_r = int(policy_rows[r])
        E[:, r] = P_blocks[a_r][:, r]
    return E


def test_tiny_handcrafted():
    # N=3, nu=2
    P0 = np.array([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6],
                   [0.5, 0.3, 0.1]], dtype=float)
    P1 = np.array([[0.9, 0.8, 0.7],
                   [0.6, 0.5, 0.4],
                   [0.1, 0.3, 0.5]], dtype=float)
    P_flat = build_P_flat([P0, P1])  # (3, 6)

    # one-hot policy per ROW: a = [1,0,1]
    a = np.array([1, 0, 1], dtype=int)
    pol = csr_matrix((np.ones_like(a, dtype=float),
                     (np.arange(3), a)),
                     shape=(3, 2))

    Pcomp = Pc(P_flat, pol)

    E = expected_Pcomp_columnwise([P0, P1], a)
    assert Pcomp.shape == (3, 3)
    assert np.allclose(Pcomp, E, atol=1e-12)


@pytest.mark.parametrize("N,nu", [(5, 3), (8, 4)])
def test_random_one_hot_matches_column_rule(N, nu):
    rng = np.random.default_rng(0)
    blocks = [rng.random((N, N)) for _ in range(nu)]
    P_flat = build_P_flat(blocks)

    # One-hot row-wise policy: choose a_r ~ Uniform{0..nu-1}
    a = rng.integers(0, nu, size=N)
    pol = csr_matrix((np.ones(N, dtype=float),
                     (np.arange(N), a)),
                     shape=(N, nu))

    Pcomp = Pc(P_flat, pol)
    E = expected_Pcomp_columnwise(blocks, a)

    assert Pcomp.shape == (N, N)
    assert np.allclose(Pcomp, E, atol=1e-12)


def test_multiple_ones_in_a_row_sums_columns():
    """
    If a row has 1s in multiple actions, Pc keeps multiple columns
    (same column index across different blocks) and sums them.
    """
    N, nu = 4, 3
    rng = np.random.default_rng(1)
    blocks = [rng.random((N, N)) for _ in range(nu)]
    P_flat = build_P_flat(blocks)

    # Build a CSR with some rows having 2 ones:
    rows = []
    cols = []
    data = []
    for r in range(N):
        # first choose one action deterministically
        cols.append(r % nu)
        rows.append(r)
        data.append(1.0)
        # add a second action on even rows
        if r % 2 == 0:
            cols.append((r + 1) % nu)
            rows.append(r)
            data.append(1.0)

    pol = csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(N, nu))

    Pcomp = Pc(P_flat, pol)

    # Expected: for each column r, sum that column r across all blocks
    # for which policy row r has a 1.
    E = np.zeros((N, N), dtype=float)
    for r in range(N):
        actions = np.flatnonzero(pol[r].toarray())
        for a in actions:
            E[:, r] += blocks[a][:, r]

    assert np.allclose(Pcomp, E, atol=1e-12)


def test_nu_equals_one_returns_block():
    N, nu = 5, 1
    rng = np.random.default_rng(42)
    P0 = rng.random((N, N))
    P_flat = build_P_flat([P0])

    # Policy is trivially one-hot with a_r = 0
    pol = csr_matrix((np.ones(N, dtype=float), (np.arange(N), np.zeros(N, int))), shape=(N, nu))

    Pcomp = Pc(P_flat, pol)
    # With nu=1, column r always comes from P0[:, r]
    assert np.allclose(Pcomp, P0, atol=1e-12)
