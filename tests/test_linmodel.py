# tests/test_linmodel.py
import numpy as np
import pytest
from src.models.linmodel import LinModel


def make_model_2x2_1w():
    """2-state, 1-input, 1-noise model with simple numbers."""
    A = np.array([[1.0, 0.1],
                  [0.0, 1.0]])
    B = np.array([[1.0],
                  [0.0]])
    C = np.array([[1.0, 1.0]])
    D = np.array([[0.0]])
    Bw = np.array([[0.0],
                   [1.0]])     # 1-d noise enters second state
    mu = np.array([0.0])
    sigma = np.array([[1.0]])
    return LinModel(A, B, C, D, Bw, mu=mu, sigma=sigma)


def test_f_det_matches_affine_update():
    model = make_model_2x2_1w()
    x = np.array([[1.0],
                  [2.0]])
    u = np.array([[3.0]])
    x_next = model.f_det(x, u)
    expected = model.A @ x + model.B @ u
    np.testing.assert_allclose(x_next, expected, rtol=1e-12, atol=1e-12)


def test_f_stoch_reproducible_with_seed():
    model = make_model_2x2_1w()
    x = np.array([[0.0],
                  [0.0]])
    u = np.array([[0.0]])

    np.random.seed(42)
    x1, w1 = model.f_stoch(x, u)

    np.random.seed(42)
    x2, w2 = model.f_stoch(x, u)

    np.testing.assert_allclose(x1, x2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(w1, w2, rtol=1e-12, atol=1e-12)


def test_f_stoch_uses_provided_w_exactly():
    model = make_model_2x2_1w()
    x = np.array([[0.0],
                  [0.0]])
    u = np.array([[0.0]])
    w = np.array([[2.0]])  # 1-d noise sample
    x_next, w_used = model.f_stoch(x, u, w=w)
    expected = model.A @ x + model.B @ u + model.Bw @ w
    np.testing.assert_allclose(x_next, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(w_used, w, rtol=1e-12, atol=1e-12)


def test_output_matches_Cx_plus_Du():
    model = make_model_2x2_1w()
    x = np.array([[1.5],
                  [2.5]])
    u = np.array([[4.0]])
    y = model.output(x, u)
    expected = model.C @ x + model.D @ u
    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)


def test_no_inplace_mutation_inputs():
    model = make_model_2x2_1w()
    x = np.array([[1.0],
                  [2.0]])
    u = np.array([[3.0]])
    x_det_in = x.copy()
    u_det_in = u.copy()
    _ = model.f_det(x, u)
    assert np.array_equal(x, x_det_in)
    assert np.array_equal(u, u_det_in)

    x_st_in = x.copy()
    u_st_in = u.copy()
    _ = model.f_stoch(x, u, w=np.array([[0.0]]))
    assert np.array_equal(x, x_st_in)
    assert np.array_equal(u, u_st_in)


def test_dim_matches_state_dimension():
    model = make_model_2x2_1w()
    assert model.dim == model.A.shape[0] == 2


def test_wsupport_normalization_repeats_rows_when_needed():
    """
    If wsupport is provided as a single interval [a,b] but noise dim > 1,
    __post_init__ should broadcast it to n_w rows.
    """
    # make a model with 2 noise dimensions (Bw: 2x2), no mu/sigma, but give wsupport
    A = np.eye(2)
    B = np.array([[1.0], [0.0]])
    C = np.array([[1.0, 1.0]])
    D = np.array([[0.0]])
    Bw = np.eye(2)  # 2 noise channels
    wsupport = np.array([[-1.0, 1.0]])  # 1x2 -> should repeat to (2,2)
    m = LinModel(A, B, C, D, Bw, mu=None, sigma=None, wsupport=wsupport)
    assert m.wsupport.shape == (2, 2)
    np.testing.assert_allclose(m.wsupport, np.array([[-1.0, 1.0],
                                                     [-1.0, 1.0]]))


@pytest.mark.parametrize("shape_x, shape_u", [
    ((2, 1), (1, 1)),     # column vectors (preferred)
    ((2,),   (1,)),       # 1D arrays should be coerced to 2D internally
])
def test_accepts_various_input_shapes(shape_x, shape_u):
    model = make_model_2x2_1w()
    x = np.ones(shape_x)
    u = np.full(shape_u, 2.0)
    # Should not raise and should return 2D arrays
    x_det = model.f_det(x, u)
    assert x_det.ndim == 2 and x_det.shape[0] == 2 and x_det.shape[1] == 1

    x_st, w_st = model.f_stoch(x, u, w=np.array([[0.0]]))
    assert x_st.ndim == 2 and x_st.shape == x_det.shape
    assert w_st.ndim == 2 and w_st.shape == (1, 1)
