# tests/test_mdpmodel.py
import numpy as np
import pytest
from types import SimpleNamespace

from src.models.mdpmodel import MDPModel


# ============================================================================
# Helpers
# ============================================================================
class MiniSys:
    """Minimal 1D continuous model stub for from_system()."""
    def __init__(self, n=1, m=1, X_bounds=(-1.0, 1.0), U_bounds=(-1.0, 1.0)):
        self.A     = np.array([[0.9]], dtype=float)
        self.B     = np.array([[0.5]], dtype=float)
        self.Bw    = np.array([[1.0]], dtype=float)
        self.mu    = np.zeros((1, 1))
        self.sigma = np.eye(1)
        self.X = np.array([X_bounds[0], X_bounds[1]], float)
        self.U = np.array([U_bounds[0], U_bounds[1]], float)


def _rank1_blocks_from_c(c_list):
    """Build a flat P from rank-1 action blocks diag(c) (N,N) for sanity checks."""
    blocks = [np.diag(np.asarray(c, float)) for c in c_list]
    return np.hstack(blocks)


def _mock_tp2d(Np, Nv, NU):
    """
    Minimal TransitionProbability2D-like duck-typed object.
    Has the four attributes the MDPModel constructor checks for.
    """
    N = Np * Nv
    # Sparse-ish indicator P_det; only .shape is touched by MDPModel.
    P_det = np.zeros((N, N * NU))
    np.fill_diagonal(P_det[:, :N], 1.0)
    return SimpleNamespace(
        P_det=P_det,
        Pi=[np.eye(Np), np.eye(Nv)],
        dim=2,
        l=(Np, Nv),
    )


# ============================================================================
# 1. Original tests (preserved)
# ============================================================================
def test_constructor_accepts_flat_and_blocks_and_stores_axes():
    N, M = 4, 3
    hx = [np.linspace(-1, 1, N)]
    P_flat = _rank1_blocks_from_c([np.ones(N) / N for _ in range(M)])

    mdp1 = MDPModel(P=P_flat, hx=hx)
    assert mdp1.P_flat.shape == (N, N * M)
    assert mdp1.P_blocks.shape == (N, N, M)
    assert np.allclose(mdp1.hx[0], hx[0])
    assert mdp1.N == N and mdp1.M == M

    P_blocks = mdp1.P_blocks.copy()
    mdp2 = MDPModel(P=P_blocks, hx=hx)
    assert np.allclose(mdp2.P_flat, mdp1.P_flat)
    assert np.allclose(mdp2.P_blocks, mdp1.P_blocks)


def test_from_system_shapes_rowstochastic_no_contract():
    sys = MiniSys()
    nx, nu = 21, 5
    mdp = MDPModel.from_system(
        sys, nx=nx, nu=nu, placement='centers',
        tol=1e-15, renormalize=True, contract_sum=1.0,
    )
    rowsums = mdp.P_blocks.sum(axis=1)
    assert (rowsums <= 1.0 + 1e-12).all()
    assert (rowsums >= -1e-12).all()


def test_rows_are_substochastic():
    sys = MiniSys()
    mdp = MDPModel.from_system(
        sys, nx=10, nu=4, placement='centers',
        tol=1e-15, renormalize=False, contract_sum=None,
    )
    rowsum = mdp.P_blocks.sum(axis=1)
    assert np.all(rowsum <= 1.0 + 1e-12)
    assert np.all(rowsum >= -1e-12)


# ============================================================================
# 2. Operator path (TransitionProbability2D-like P)
# ============================================================================
def test_constructor_operator_path_stores_and_mirrors_correctly():
    Np, Nv, NU = 4, 3, 2
    hx   = [np.linspace(-1, 1, Np), np.linspace(-1, 1, Nv)]
    tp2d = _mock_tp2d(Np, Nv, NU)
    mdp  = MDPModel(P=tp2d, hx=hx)

    assert mdp.P is tp2d                       # operator stored as-is
    assert mdp.P_det is tp2d.P_det             # mirrored
    assert mdp.Pi[0] is tp2d.Pi[0]             # mirrored
    assert mdp.Pi[1] is tp2d.Pi[1]
    assert mdp.P_flat is None                  # dense formats not populated
    assert mdp.P_blocks is None
    assert mdp.dim == 2
    assert mdp.l == (Np, Nv)
    assert mdp.N == Np * Nv
    assert mdp.M == NU                         # exercises operator-path M


# ============================================================================
# 3. Constructor error paths
# ============================================================================
def test_constructor_typeerror_when_P_neither_ndarray_nor_operator():
    hx = [np.linspace(-1, 1, 5)]
    with pytest.raises(TypeError):
        MDPModel(P="not an array", hx=hx)
    with pytest.raises(TypeError):
        # list lacks the four required operator attributes
        MDPModel(P=[[1, 2], [3, 4]], hx=hx)


def test_constructor_valueerror_when_dense_P_has_wrong_ndim():
    hx = [np.linspace(-1, 1, 5)]
    with pytest.raises(ValueError):
        MDPModel(P=np.ones((5,)), hx=hx)              # 1D
    with pytest.raises(ValueError):
        MDPModel(P=np.ones((5, 5, 5, 5)), hx=hx)      # 4D


# ============================================================================
# 4. Construction defaults
# ============================================================================
def test_states_default_to_cartesian_product_of_hx():
    """When `states` is omitted, MDPModel builds the Cartesian product of hx."""
    hx = [np.array([0.0, 1.0, 2.0]), np.array([-1.0, 1.0])]
    Np, Nv = 3, 2
    N = Np * Nv
    P = np.zeros((N, N))                              # M=1 dense P
    P[:, :N] = np.eye(N)
    mdp = MDPModel(P=P, hx=hx)
    assert mdp.states.shape == (N, 2)
    np.testing.assert_allclose(np.sort(np.unique(mdp.states[:, 0])), hx[0])
    np.testing.assert_allclose(np.sort(np.unique(mdp.states[:, 1])), hx[1])


def test_inputs_1d_reshaped_to_column():
    hx = [np.linspace(0, 1, 3)]
    P  = np.eye(3)                                    # (N, N*M) with M=1
    inputs_1d = np.array([0.0, 0.5, 1.0])
    mdp = MDPModel(P=P, hx=hx, inputs=inputs_1d)
    assert mdp.inputs.shape == (3, 1)
    np.testing.assert_allclose(mdp.inputs.ravel(), inputs_1d)


# ============================================================================
# 5. outputs = (C @ states.T).T
# ============================================================================
def test_outputs_computed_when_orig_has_C():
    hx = [np.array([0.0, 1.0, 2.0])]
    P  = np.eye(3)
    class MiniLin:
        C = np.array([[2.0]])                         # output = 2 * state
    mdp = MDPModel(P=P, hx=hx, orig=MiniLin())
    assert mdp.outputs is not None
    np.testing.assert_allclose(mdp.outputs.ravel(), 2.0 * hx[0])


def test_outputs_none_when_orig_absent_or_has_no_C():
    hx = [np.array([0.0, 1.0, 2.0])]
    P  = np.eye(3)
    mdp1 = MDPModel(P=P, hx=hx)
    assert mdp1.outputs is None
    class NoC: pass
    mdp2 = MDPModel(P=P, hx=hx, orig=NoC())
    assert mdp2.outputs is None


# ============================================================================
# 6. Namespace delegators
# ============================================================================
def test_static_method_delegators_callable():
    """`MDPModel.make_uniform_grid` and `transition_matrix_nd_separable` are
    accessible as static aliases on the class (full behaviour tested elsewhere)."""
    assert callable(MDPModel.make_uniform_grid)
    assert callable(MDPModel.transition_matrix_nd_separable)

# ============================================================================
# 7. from_system with compute_P='2d' — factory smoke tests
# ============================================================================
import polytope as pc
from src.models.linmodel import LinModel


def _make_2d_sys_1u():
    """2-state-1-input-2-noise LinModel for the 2D factory path."""
    A  = np.array([[1.0, 0.5],
                   [0.0, 1.0]])
    B  = np.array([[0.0], [1.0]])
    C  = np.array([[1.0, 0.0]])             # 1D output (position)
    D  = np.array([[0.0]])
    Bw = np.eye(2)
    sys = LinModel(A, B, C, D, Bw,
                   mu=np.zeros(2), sigma=0.25 * np.eye(2))
    sys.X = pc.box2poly([[-2, 2], [-1, 1]])
    sys.U = pc.box2poly([[-1, 1]])
    return sys


def _make_2d_sys_2u():
    """2-state-2-input-2-noise LinModel for the asymmetric-input variant."""
    A  = np.array([[1.0, 0.5],
                   [0.0, 1.0]])
    B  = np.array([[0.5, 0.0],
                   [0.0, 0.5]])              # 2D input
    C  = np.eye(2)
    D  = np.zeros((2, 2))
    Bw = np.eye(2)
    sys = LinModel(A, B, C, D, Bw,
                   mu=np.zeros(2), sigma=0.25 * np.eye(2))
    sys.X = pc.box2poly([[-2, 2], [-1, 1]])
    sys.U = pc.box2poly([[-1, 1], [-1, 1]])
    return sys


def test_from_system_2d_path_returns_operator_mdp():
    """End-to-end: real LinModel → MDPModel.from_system(..., compute_P='2d')."""
    sys = _make_2d_sys_1u()
    Np, Nv, NU = 10, 6, 5
    mdp = MDPModel.from_system(sys, nx=[Np, Nv], nu=NU, compute_P='2d')

    # Operator-path: dense formats not populated
    assert mdp.P_flat is None
    assert mdp.P_blocks is None

    # Operator components mirrored from tP_2d
    assert mdp.P is not None
    assert mdp.P_det is not None
    assert len(mdp.Pi) == 2

    # Structural shapes (l overridden by tP_2d.l; sizes are rotated-frame grid)
    assert mdp.dim == 2
    assert len(mdp.l) == 2
    Np_eff, Nv_eff = mdp.l
    N = Np_eff * Nv_eff
    assert mdp.N == N
    assert mdp.M == NU
    assert mdp.P_det.shape == (N, N * NU)
    assert mdp.Pi[0].shape == (Np_eff, Np_eff)
    assert mdp.Pi[1].shape == (Nv_eff, Nv_eff)

    # Grid axes
    assert len(mdp.hx) == 2
    assert mdp.hx[0].size == Np_eff
    assert mdp.hx[1].size == Nv_eff

    # 2D-path-only auxiliary fields populated
    assert mdp.zstates is not None
    assert mdp.Uz is not None
    assert mdp.Uz.shape == (2, 2)
    assert mdp.outputs is not None
    assert mdp.outputmap is not None


def test_from_system_2d_with_2d_input_M_is_product():
    """2D U polytope + list-valued nu: M = nu[0] * nu[1], symmetric and asymmetric."""
    sys = _make_2d_sys_2u()
    Np, Nv = 6, 4

    mdp_sym = MDPModel.from_system(sys, nx=[Np, Nv], nu=[3, 3], compute_P='2d')
    assert mdp_sym.M == 9
    assert mdp_sym.inputs.shape[0] == 2     # 2 input dimensions
    assert mdp_sym.inputs.shape[1] == 9     # 9 input letters

    mdp_asym = MDPModel.from_system(sys, nx=[Np, Nv], nu=[3, 4], compute_P='2d')
    assert mdp_asym.M == 12
    assert mdp_asym.inputs.shape == (2, 12)


# ============================================================================
# 8. compute_P validation
# ============================================================================
def test_from_system_compute_P_nd_raises_not_implemented():
    sys = MiniSys()
    with pytest.raises(NotImplementedError, match="nd"):
        MDPModel.from_system(sys, nx=5, nu=3, compute_P='nd')


def test_from_system_compute_P_invalid_raises_value_error():
    sys = MiniSys()
    with pytest.raises(ValueError, match="compute_P must be"):
        MDPModel.from_system(sys, nx=5, nu=3, compute_P='bogus')