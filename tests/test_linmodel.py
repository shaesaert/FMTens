# tests/test_linmodel.py
import numpy as np
import pytest
from src.models.linmodel import LinModel


# ============================================================================
# Fixtures
# ============================================================================
def make_model_2x2_1w():
    """2-state, 1-input, 1-noise model with simple numbers."""
    A = np.array([[1.0, 0.1],
                  [0.0, 1.0]])
    B = np.array([[1.0],
                  [0.0]])
    C = np.array([[1.0, 1.0]])
    D = np.array([[0.0]])
    Bw = np.array([[0.0],
                   [1.0]])
    mu = np.array([0.0])
    sigma = np.array([[1.0]])
    return LinModel(A, B, C, D, Bw, mu=mu, sigma=sigma)


def make_model_2x2_2u_2w():
    """2-state, 2-input, 2-noise model — RA4D-style."""
    A  = np.array([[1.0, 0.5],
                   [0.0, 1.0]])
    B  = np.array([[0.5, 0.0],
                   [0.0, 0.5]])
    C  = np.eye(2)
    D  = np.zeros((2, 2))
    Bw = np.eye(2)
    mu    = np.zeros(2)
    sigma = 0.25 * np.eye(2)
    return LinModel(A, B, C, D, Bw, mu=mu, sigma=sigma)


# ============================================================================
# 1. Original tests (preserved)
# ============================================================================
def test_f_det_matches_affine_update():
    model = make_model_2x2_1w()
    x = np.array([[1.0], [2.0]])
    u = np.array([[3.0]])
    x_next = model.f_det(x, u)
    expected = model.A @ x + model.B @ u
    np.testing.assert_allclose(x_next, expected, rtol=1e-12, atol=1e-12)


def test_f_stoch_reproducible_with_seed():
    model = make_model_2x2_1w()
    x = np.array([[0.0], [0.0]])
    u = np.array([[0.0]])

    np.random.seed(42)
    x1, w1 = model.f_stoch(x, u)
    np.random.seed(42)
    x2, w2 = model.f_stoch(x, u)
    np.testing.assert_allclose(x1, x2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(w1, w2, rtol=1e-12, atol=1e-12)


def test_f_stoch_uses_provided_w_exactly():
    model = make_model_2x2_1w()
    x = np.array([[0.0], [0.0]])
    u = np.array([[0.0]])
    w = np.array([[2.0]])
    x_next, w_used = model.f_stoch(x, u, w=w)
    expected = model.A @ x + model.B @ u + model.Bw @ w
    np.testing.assert_allclose(x_next, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(w_used, w, rtol=1e-12, atol=1e-12)


def test_output_matches_Cx_plus_Du():
    model = make_model_2x2_1w()
    x = np.array([[1.5], [2.5]])
    u = np.array([[4.0]])
    y = model.output(x, u)
    expected = model.C @ x + model.D @ u
    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)


def test_no_inplace_mutation_inputs():
    model = make_model_2x2_1w()
    x = np.array([[1.0], [2.0]])
    u = np.array([[3.0]])
    x_in, u_in = x.copy(), u.copy()
    _ = model.f_det(x, u)
    assert np.array_equal(x, x_in) and np.array_equal(u, u_in)
    _ = model.f_stoch(x, u, w=np.array([[0.0]]))
    assert np.array_equal(x, x_in) and np.array_equal(u, u_in)


def test_dim_matches_state_dimension():
    model = make_model_2x2_1w()
    assert model.dim == model.A.shape[0] == 2


def test_wsupport_normalization_repeats_rows_when_needed():
    A = np.eye(2); B = np.array([[1.0], [0.0]])
    C = np.array([[1.0, 1.0]]); D = np.array([[0.0]])
    Bw = np.eye(2)
    wsupport = np.array([[-1.0, 1.0]])  # 1x2 → should repeat to (2,2)
    m = LinModel(A, B, C, D, Bw, mu=None, sigma=None, wsupport=wsupport)
    assert m.wsupport.shape == (2, 2)
    np.testing.assert_allclose(m.wsupport, np.array([[-1.0, 1.0],
                                                     [-1.0, 1.0]]))


@pytest.mark.parametrize("shape_x, shape_u", [
    ((2, 1), (1, 1)),
    ((2,),   (1,)),
])
def test_accepts_various_input_shapes(shape_x, shape_u):
    model = make_model_2x2_1w()
    x = np.ones(shape_x)
    u = np.full(shape_u, 2.0)
    x_det = model.f_det(x, u)
    assert x_det.ndim == 2 and x_det.shape == (2, 1)
    x_st, w_st = model.f_stoch(x, u, w=np.array([[0.0]]))
    assert x_st.shape == x_det.shape
    assert w_st.shape == (1, 1)


# ============================================================================
# 2. Multi-input / multi-noise
# ============================================================================
def test_f_det_multi_input_uses_full_B():
    """`B @ u` with m=2 — both columns must contribute."""
    model = make_model_2x2_2u_2w()
    x = np.array([[1.0], [2.0]])
    u = np.array([[0.3], [-0.4]])
    expected = model.A @ x + model.B @ u
    np.testing.assert_allclose(model.f_det(x, u), expected, rtol=1e-12, atol=1e-12)


def test_f_stoch_explicit_w_multi_noise():
    """`Bw @ w` with nw=2 — both noise channels must contribute."""
    model = make_model_2x2_2u_2w()
    x = np.zeros((2, 1)); u = np.zeros((2, 1))
    w = np.array([[0.7], [-0.3]])
    x_next, w_used = model.f_stoch(x, u, w=w)
    np.testing.assert_allclose(x_next, model.Bw @ w, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(w_used, w, rtol=1e-12, atol=1e-12)


def test_f_stoch_sampled_w_multi_noise_shape_and_reproducibility():
    """Sampled w with nw=2 has shape (2,1) and respects the global seed."""
    model = make_model_2x2_2u_2w()
    x = np.zeros((2, 1)); u = np.zeros((2, 1))
    np.random.seed(7); _, w1 = model.f_stoch(x, u)
    np.random.seed(7); _, w2 = model.f_stoch(x, u)
    assert w1.shape == (2, 1)
    np.testing.assert_allclose(w1, w2, rtol=1e-12, atol=1e-12)


# ============================================================================
# 3. Noise initialisation paths
# ============================================================================
def test_sigma_defaults_to_identity_when_mu_given():
    A = np.eye(2); B = np.zeros((2, 1)); C = np.eye(2); D = np.zeros((2, 1))
    Bw = np.eye(2)
    m = LinModel(A, B, C, D, Bw, mu=np.zeros(2))      # sigma omitted
    np.testing.assert_allclose(m.sigma, np.eye(2))


def test_default_wsupport_is_pm_inf_when_mu_given():
    A = np.eye(2); B = np.zeros((2, 1)); C = np.eye(2); D = np.zeros((2, 1))
    Bw = np.eye(2)
    m = LinModel(A, B, C, D, Bw, mu=np.zeros(2))
    assert m.wsupport.shape == (2, 2)
    assert np.all(np.isneginf(m.wsupport[:, 0]))
    assert np.all(np.isposinf(m.wsupport[:, 1]))


def test_deterministic_init_when_mu_and_wsupport_omitted():
    """Third init branch: no mu, no wsupport → mu=0, sigma=0, f_stoch(w=None) gives 0."""
    A = np.eye(2); B = np.zeros((2, 1)); C = np.eye(2); D = np.zeros((2, 1))
    Bw = np.eye(2)
    m = LinModel(A, B, C, D, Bw)                      # both omitted
    np.testing.assert_allclose(m.mu, np.zeros((2, 1)))
    np.testing.assert_allclose(m.sigma, np.zeros((2, 2)))
    x_next, w = m.f_stoch(np.zeros((2, 1)), np.zeros((1, 1)))
    np.testing.assert_allclose(w, np.zeros((2, 1)))
    np.testing.assert_allclose(x_next, np.zeros((2, 1)))


# ============================================================================
# 4. Assertion / error paths
# ============================================================================
def test_raises_on_dim_mismatch_with_A():
    A = np.eye(2); B = np.zeros((2, 1)); C = np.eye(2); D = np.zeros((2, 1))
    Bw = np.eye(2)
    with pytest.raises(AssertionError):
        LinModel(A, B, C, D, Bw, dim=3)


def test_raises_on_bad_sigma_shape():
    A = np.eye(2); B = np.zeros((2, 1)); C = np.eye(2); D = np.zeros((2, 1))
    Bw = np.eye(2)
    with pytest.raises(AssertionError):
        LinModel(A, B, C, D, Bw, mu=np.zeros(2), sigma=np.eye(3))


# ============================================================================
# 5. Optional attributes (X, U, regions, AP)
# ============================================================================
def test_optional_attrs_default_to_none_and_empty_lists():
    """X, U default to None; regions, AP default to fresh empty lists per instance."""
    m1 = make_model_2x2_1w()
    m2 = make_model_2x2_1w()
    assert m1.X is None and m1.U is None
    assert m1.regions == [] and m1.AP == []
    # Mutate one → the other instance is untouched (default_factory, not shared list).
    m1.regions.append("region1")
    m1.AP.append("p1")
    assert m2.regions == [] and m2.AP == []


def test_optional_attrs_stored_verbatim():
    m = make_model_2x2_1w()
    m.X = "X_box"; m.U = "U_box"
    m.regions = [1, 2, 3]; m.AP = ['p1', 'p2']
    assert m.X == "X_box" and m.U == "U_box"
    assert m.regions == [1, 2, 3] and m.AP == ['p1', 'p2']