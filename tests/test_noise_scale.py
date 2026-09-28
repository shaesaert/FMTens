"""Regression tests: the abstraction must use the noise STANDARD DEVIATION
sqrt(eig(Bw Sigma Bw^T)), not the variance. Includes variances below and
above 1, because with variance exactly 1 the two coincide and a bug is invisible."""
import os, sys
sys.path.insert(0, os.getcwd())   # run from the repository root
import numpy as np
import polytope as pc
from scipy.stats import norm
from src.models.linmodel import LinModel
from src.models.mdpmodel import MDPModel


def _model_1d(bw):
    s = LinModel(np.array([[0.9]]), np.array([[0.5]]), np.array([[1.0]]), np.array([[0.0]]),
                 np.array([[bw]]), mu=np.array([0.0]), sigma=np.eye(1))
    s.X = pc.Polytope(np.array([[1.0], [-1.0]]), np.array([5.0, 20.0]))
    s.U = pc.Polytope(np.array([[1.0], [-1.0]]), np.array([5.0, 5.0]))
    return s


def test_1d_rows_equal_exact_gaussian():
    for bw in (0.5, 2.0):
        m = MDPModel.from_system(_model_1d(bw), nx=200, nu=5, placement="centers",
                                 u_placement="endpoints", tol=1e-19, compute_P="1d")
        x = np.asarray(m.states).ravel(); u = np.asarray(m.inputs).ravel()
        h = x[1] - x[0]
        edges = np.concatenate(([x[0] - h / 2], (x[:-1] + x[1:]) / 2, [x[-1] + h / 2]))
        Pb = m.P_blocks                                    # (N_next, N_from, M)
        err = 0.0
        for k in range(u.size):
            for i in (0, x.size // 2, x.size - 1):
                exact = np.diff(norm.cdf(edges, 0.9 * x[i] + 0.5 * u[k], bw))
                exact[exact < 1e-19] = 0.0
                err = max(err, np.abs(Pb[:, i, k] - exact).max())
        assert err < 1e-12, f"Bw={bw}: max row error {err:.2e}"
        print(f"  1D  Bw={bw}: max |row - exact| = {err:.1e}  OK")


def test_2d_kernel_std_equals_sqrt_eigenvalue():
    Bw = np.diag([0.3, 2.0])
    s = LinModel(0.9 * np.eye(2), np.array([[0.5], [0.5]]), np.eye(2), np.zeros((2, 1)),
                 Bw, mu=np.zeros(2), sigma=np.eye(2))
    s.X = pc.Polytope(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float), np.full(4, 10.0))
    s.U = pc.Polytope(np.array([[1.0], [-1.0]]), np.array([1.0, 1.0]))
    L = 200
    m = MDPModel.from_system(s, nx=[L, L], nu=3, placement="centers",
                             u_placement="endpoints", tol=1e-19, compute_P="2d")
    Pi = getattr(m, "Pi", None) or list(m.P.Pi)
    eig = np.linalg.svd(Bw @ Bw.T, compute_uv=False)
    h = 20.0 / L
    for d in range(2):
        K = Pi[d].toarray() if hasattr(Pi[d], "toarray") else np.asarray(Pi[d])
        c = K[:, 0]; k = np.arange(c.size)
        mass = c[0] + 2 * c[1:].sum()
        std = np.sqrt(2 * np.sum(c[1:] * (k[1:] * h) ** 2) / mass - h ** 2 / 12)
        assert abs(std - np.sqrt(eig[d])) < 1e-3, f"axis {d}: kernel std {std:.4f} vs {np.sqrt(eig[d]):.4f}"
        print(f"  2D  axis {d}: kernel std {std:.4f} = sqrt(eig) {np.sqrt(eig[d]):.4f}  OK")


if __name__ == "__main__":
    test_1d_rows_equal_exact_gaussian()
    test_2d_kernel_std_equals_sqrt_eigenvalue()
    print("all noise-scale tests passed")
