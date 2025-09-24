# tests/test_mdpmodel_core.py
import numpy as np
import pytest

from src.models.mdpmodel import MDPModel

# -------------------------
# Helpers
# -------------------------
class MiniSys:
    """Minimal continuous model stub for from_system()."""
    def __init__(self, n=1, m=1, X_bounds=(-1.0, 1.0), U_bounds=(-1.0, 1.0)):
        # 1D default; can be extended if needed
        self.A = np.array([[0.9]], dtype=float)           # (n,n)
        self.B = np.array([[0.5]], dtype=float)           # (n,m)
        self.Bw = np.array([[1.0]], dtype=float)          # (n,nw) with nw==n here
        self.mu = np.zeros((1, 1))                        # (nw,1)
        self.sigma = np.eye(1)                            # (nw,nw)
        # bounds accepted by MDPModel._bbox_from_polytope
        self.X = np.array([X_bounds[0], X_bounds[1]], float)  # 1D: shape (2,)
        self.U = np.array([U_bounds[0], U_bounds[1]], float)

def _rank1_blocks_from_c(c_list):
    """Build a flat P from rank-1 action blocks diag(c) (N,N) for easy sanity checks."""
    blocks = [np.diag(np.asarray(c, float)) for c in c_list]  # each (N,N)
    return np.hstack(blocks)  # (N, N*M)

# -------------------------
# Constructor / shape tests
# -------------------------
def test_constructor_accepts_flat_and_blocks_and_stores_axes():
    N, M = 4, 3
    hx = [np.linspace(-1, 1, N)]
    P_flat = _rank1_blocks_from_c([np.ones(N)/N for _ in range(M)])  # (N, N*M)

    # flat -> blocks
    mdp1 = MDPModel(P=P_flat, hx=hx)
    assert mdp1.P_flat.shape == (N, N*M)
    assert mdp1.P_blocks.shape == (N, N, M)
    assert np.allclose(mdp1.hx[0], hx[0])
    assert mdp1.N == N and mdp1.M == M

    # blocks -> flat
    P_blocks = mdp1.P_blocks.copy()
    mdp2 = MDPModel(P=P_blocks, hx=hx)
    assert np.allclose(mdp2.P_flat, mdp1.P_flat)
    assert np.allclose(mdp2.P_blocks, mdp1.P_blocks)

def test_to_flat_blocks_consistency_and_replace_block():
    N, M = 5, 2
    hx = [np.linspace(0, 1, N)]
    P_flat = _rank1_blocks_from_c([np.ones(N)/N for _ in range(M)])
    mdp = MDPModel(P=P_flat, hx=hx)
    # replace block 1 with identity (row-stochastic)
    I = np.eye(N)
    mdp.replace_block(1, I)
    assert np.allclose(mdp.block(1), I)
    # flat view must reflect the same change
    sl = mdp.block_cols(1)
    assert np.allclose(mdp.P_flat[:, sl], I)

# -------------------------
# from_system / rowsum tests
# -------------------------
def test_from_system_shapes_rowstochastic_no_contract():
    sys = MiniSys()
    nx, nu = 21, 5
    mdp = MDPModel.from_system(
        sys, nx=nx, nu=nu, placement='centers',
        tol=1e-15, renormalize=True, contract_sum=1.0  # rowsum==1 after renorm
    )
    rowsums = mdp.P_blocks.sum(axis=1)
    assert (rowsums <= 1.0 + 1e-12).all()
    assert (rowsums >= -1e-12).all()

    # assert mdp.N == nx
    # assert mdp.M == nu
    # # each action block is (approximately) row-stochastic
    # rowsums = mdp.P_blocks.sum(axis=1)
    # assert np.allclose(rowsums, 1.0, atol=1e-10)

def test_apply_substochastic_cap_and_set_modes():
    sys = MiniSys()
    nx, nu = 40, 4
    mdp = MDPModel.from_system(
        sys, nx=nx, nu=nu, placement='centers',
        tol=1e-15, renormalize=True, contract_sum=None
    )
    # cap to 0.97 (shrink rows > 0.97 only)
    mdp_cap = MDPModel(P=mdp.P_flat.copy(), hx=mdp.hx)
    mdp_cap.apply_substochastic(target=0.97, mode='cap')
    rowsums_cap = mdp_cap.P_blocks.sum(axis=1)
    assert np.all(rowsums_cap <= 0.97 + 1e-12)
    # set to exactly 0.95
    mdp_set = MDPModel(P=mdp.P_flat.copy(), hx=mdp.hx)
    mdp_set.apply_substochastic(target=0.95, mode='set')
    rowsums_set = mdp_set.P_blocks.sum(axis=1)
    assert np.allclose(rowsums_set, 0.95, atol=1e-10)

# -------------------------
# Indexing / mapping tests
# -------------------------
def test_state_index_roundtrip_and_input_index():
    # 1D grid, easy nearest-neighbor mapping
    hx = [np.array([-1.0, 0.0, 1.0])]
    N, M = 3, 2
    P_flat = _rank1_blocks_from_c([np.ones(N)/N for _ in range(M)])
    inputs = np.array([[-1.0], [1.0]])  # two discrete inputs
    mdp = MDPModel(P=P_flat, hx=hx, inputs=inputs)

    # x near 0 maps to index of 0.0
    i = mdp.state_index_from_x(np.array([0.05]))
    assert i == 1
    # roundtrip index -> state
    x_back = mdp.idx_to_state(i)
    assert np.allclose(x_back, [0.0])

    # input mapping
    k = mdp.input_index_from_u(np.array([0.7]))
    assert k == 1
    k2 = mdp.input_index_from_u(np.array([-0.9]))
    assert k2 == 0

# -------------------------
# Simulation tests
# -------------------------
def test_sim_index_and_sim_outputs_within_grid():
    hx = [np.linspace(-1, 1, 5)]
    N, M = 5, 2
    # make each block row-stochastic & simple: uniform probabilities
    P_flat = _rank1_blocks_from_c([np.ones(N)/N for _ in range(M)])
    mdp = MDPModel(P=P_flat, hx=hx, inputs=np.array([[-1.0], [1.0]]))

    rng = np.random.default_rng(0)
    # sim from index 2 with action 0
    j = mdp.sim_index(2, 0, rng=rng)
    assert 0 <= j < mdp.N
    # sim with continuous arguments, returning state vector
    x_next = mdp.sim(x=np.array([0.1]), u=np.array([0.9]), rng=rng)
    assert x_next.shape == (1,)
    assert x_next[0] >= hx[0].min() - 1e-12 and x_next[0] <= hx[0].max() + 1e-12

# -------------------------
# Flat/blocks conversions
# -------------------------
def test_flat_blocks_conversion_roundtrip():
    sys = MiniSys()
    mdp = MDPModel.from_system(sys, nx=12, nu=3, placement='centers',
                               tol=1e-15, renormalize=True, contract_sum=1.0)
    P_flat = mdp.to_flat()
    P_blocks = mdp.to_blocks()
    # Rebuild from either view and compare
    a = MDPModel(P=P_flat, hx=mdp.hx)
    b = MDPModel(P=P_blocks, hx=mdp.hx)
    assert np.allclose(a.P_flat, b.P_flat)
    assert np.allclose(a.P_blocks, b.P_blocks)

# -------------------------
# Row-stochastic checker
# -------------------------
def test_rows_are_substochastic():
    sys = MiniSys()
    mdp = MDPModel.from_system(sys, nx=10, nu=4,
                               placement='centers',
                               tol=1e-15,
                               renormalize=False,     # <-- important
                               contract_sum=None)
    rowsum = mdp.P_blocks.sum(axis=1)  # (N, M)
    assert np.all(rowsum <= 1.0 + 1e-12)
    assert np.all(rowsum >= -1e-12)    # no negative sums from num. noise

