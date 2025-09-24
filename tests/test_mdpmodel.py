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
# Row-substochastic checker
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

