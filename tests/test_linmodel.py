import unittest
import numpy as np
from src.models.linmodel import LinModel


def make_simple_model():
    A = np.eye(2)
    B = np.array([[1], [0]])
    C = np.array([[1, 1]])
    D = np.array([[0]])
    Bw = np.array([[0], [1]])
    mu = np.zeros(1)
    sigma = np.eye(1)
    return LinModel(A, B, C, D, Bw, mu, sigma)


class TestLinModel(unittest.TestCase):

    def test_f_det(self):
        model = make_simple_model()
        x = np.array([1, 2])
        u = np.array([3])
        x_next = model.f_det(x, u)
        expected = model.A @ x + model.B @ u
        self.assertTrue(np.allclose(x_next, expected))

    def test_f_stoch_reproducible(self):
        model = make_simple_model()
        x = np.array([0, 0])
        u = np.array([0])

        np.random.seed(42)
        x1, w1 = model.f_stoch(x, u)

        np.random.seed(42)
        x2, w2 = model.f_stoch(x, u)

        self.assertTrue(np.allclose(x1, x2))
        self.assertTrue(np.allclose(w1, w2))

    def test_f_stoch_with_given_w(self):
        model = make_simple_model()
        x = np.array([0, 0])
        u = np.array([0])
        w = np.array([2.0])
        x_next, w_used = model.f_stoch(x, u, w=w)
        expected = model.A @ x + model.B @ u + model.Bw @ w
        self.assertTrue(np.allclose(x_next, expected))
        self.assertTrue(np.allclose(w_used, w))

    def test_simulation_deterministic(self):
        model = make_simple_model()
        x0 = np.array([0, 0])
        U = [np.array([1])] * 5
        X, W = model.simulate(x0, U, stochastic=False)
        # Second state remains zero
        self.assertTrue(np.allclose(X[:, 1], 0.0))
        # First state should be cumulative sum of inputs
        self.assertTrue(np.allclose(X[:, 0], np.arange(0, 6)))

    def test_simulation_stochastic_reproducibility(self):
        model = make_simple_model()
        x0 = np.array([0, 0])
        U = [np.array([0])] * 3
        X1, W1 = model.simulate(x0, U, stochastic=True, random_seed=123)
        X2, W2 = model.simulate(x0, U, stochastic=True, random_seed=123)
        self.assertTrue(np.allclose(X1, X2))
        self.assertTrue(np.allclose(W1, W2))


if __name__ == "__main__":
    unittest.main()
