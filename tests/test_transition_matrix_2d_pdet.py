# test_transition_matrix_2d_pdet.py
import numpy as np
import polytope as pc
import pytest

from src.models.mdpmodel import MDPModel  # <-- adjust module path

class _Sys2D:
    def __init__(self, A, B, Bw, mu, sigma, X, U=None, C=None):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.Bw = np.asarray(Bw, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.X = X
        self.U = U
        self.C = np.eye(2) if C is None else np.asarray(C, dtype=float)

from src.abstraction.utils.abstraction_grid_helper import make_uniform_grid
def _build_axes(bounds, counts):
    axes, _ = make_uniform_grid(bounds, grid_counts=counts,
                                         filter_inside=True, placement="centers")
    return [axes[0], axes[1]]

def _lin_index_2d(i0, i1, l, order='F'):
    return np.ravel_multi_index((i0, i1), dims=tuple(l), order=order)

@pytest.mark.parametrize("l", [(3, 3), (4, 2)])
def test_pdet_columns_point_to_expected_cells(l):
    A = np.eye(2)
    B = np.eye(2)
    Bw = np.eye(2)
    mu = np.zeros((2, 1))
    sigma = np.diag([1e-6, 1e-6])

    # State space: convex hull of unit square
    V = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [1.0, 1.0],
                  [0.0, 1.0]])
    X_poly = pc.qhull(V)

    sys = _Sys2D(A=A, B=B, Bw=Bw, mu=mu, sigma=sigma, X=X_poly)

    X_axes = _build_axes(X_poly, counts=l)

    U_points = np.array([[0.0, 0.2],
                         [0.0, -0.1]], dtype=float)

    from src.abstraction.utils.abstraction_transition_builder import transition_matrix_nd_separable_impl
    tP_2d, hz, XhatSpace, beta, sys_ret, ZhatSpace, Uz = transition_matrix_nd_separable_impl(
        sys=sys,
        X_axes=X_axes,
        U_points=U_points,
        X_poly=X_poly,
        tol=1e-12,
        renormalize=True,
        return_flat=False,
        mode='2d'
    )

    N = XhatSpace.shape[1]
    M = U_points.shape[1]
    assert tP_2d.P_det.shape == (N, N * M)

    V_z = np.asarray(pc.extreme(sys.X))
    V_z2 = (Uz.T @ V_z.T).T
    Zl = V_z2.min(axis=0)
    Zu = V_z2.max(axis=0)
    l_vec = np.array(l, dtype=int)
    gridSize = (Zu - Zl) / l_vec

    for k in range(M):
        u_k = U_points[:, k].reshape(-1, 1)
        xn = sys.A @ XhatSpace + sys.B @ u_k
        z_n = Uz.T @ xn

        gridSize_inv = 1.0 / gridSize
        z_n_ind = np.floor(np.diag(gridSize_inv) @ (z_n - Zl[:, None])).astype(int)

        valid = (z_n_ind[0] >= 0) & (z_n_ind[0] <= l_vec[0] - 1) & \
                (z_n_ind[1] >= 0) & (z_n_ind[1] <= l_vec[1] - 1)

        i0 = z_n_ind[0, valid].astype(np.int64)
        i1 = z_n_ind[1, valid].astype(np.int64)
        expected_rows = _lin_index_2d(i0, i1, l_vec, order='F').ravel()

        cols = np.arange(k * N, (k + 1) * N, dtype=int)
        Pk = tP_2d.P_det[:, cols]

        col_sums = np.array(Pk.sum(axis=0)).ravel()
        assert np.all((col_sums == 0.0) | (col_sums == 1.0))

        one_rows, one_cols = tP_2d.P_det[:, cols].nonzero()
        order = np.argsort(one_cols)
        one_cols = one_cols[order]
        one_rows = one_rows[order]

        local_cols = one_cols - k * N
        local_valid_positions = np.where(valid)[0]
        assert len(local_cols) == len(local_valid_positions)

        assert np.array_equal(one_rows, expected_rows)

        invalid_positions = np.where(~valid)[0]
        if invalid_positions.size > 0:
            assert np.all(col_sums[invalid_positions] == 0.0)

    data = tP_2d.P_det.data
    assert np.all((data == 0.0) | (data == 1.0))
