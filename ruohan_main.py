

import numpy as np
from control import ss
import polytope as pc
from model.linmodel import LinModel
from ruohan_abstraction.ugrid_util import make_uniform_grid 
from ruohan_abstraction.sastransition import transition_matrix_nd_separable
from model.mdp_model import MDPModel

# ====== System dynamics defined for each agent ======
A = {}
A[1] = np.array([[0.9]])
A[2] = np.array([[0.9]])  

B = {}
B[1] = np.array([[0.5]])   
B[2] = np.array([[0.5]])  

C = {}
C[1] = np.array([[1.0]])   
C[2] = np.array([[1.0]])  

D = {} 
D[1] = np.array([[0.0]])   
D[2] = np.array([[0.0]])   

Bw = {}
Bw[1] = np.array([[1.0]])
Bw[2] = np.array([[1.0]])

mu = {}
mu[1] = np.array([1])     # noise mean
mu[2] = np.array([1])     # noise mean

sigma = {}
sigma[1] = np.eye(1)      # noise covariance
sigma[2] = np.eye(1)      # noise covariance

sysLTI = {}
sysLTI[1] = LinModel(A[1], B[1], C[1], D[1], Bw[1], mu=mu[1], sigma=sigma[1])
sysLTI[2] = LinModel(A[2], B[2], C[2], D[2], Bw[2], mu=mu[2], sigma=sigma[2])

# Constraints: -20 <= x <= 5  (H-representation: s x <= b)
s = np.array([[ 1],      #  x <= 5
              [-1]])     # -x <= 20  →  x >= -20
bx  = np.array([5, 20])
bu  = np.array([5, 5])
bp1 = np.array([5, 0])
bp2 = np.array([0, 5])
bp3 = np.array([15, 20])

P1 = pc.Polytope(s, bp1)
P2 = pc.Polytope(s, bp2)
P3 = pc.Polytope(s, bp3)

# Define state/input polytopes
sysLTI[1].X = pc.Polytope(s, bx)
sysLTI[2].X = pc.Polytope(s, bx)

sysLTI[1].U = pc.Polytope(s, bu)
sysLTI[2].U = pc.Polytope(s, bu)

sysLTI[1].regions = [P1, P2]
sysLTI[2].regions = [P3]

# Atomic propositions
sysLTI[1].AP = ['p1', 'p2']
sysLTI[2].AP = ['p3']

# Actions (list of strings)
act = {}
act[1] = [['', 'p1', 'p2', 'p1p2']]
act[2] = [['', 'p3']]

# Grid inputs (U) for each agent
uhat = {}
uax  = {}
uax[1], uhat[1] = make_uniform_grid(sysLTI[1].U, grid_counts=5, filter_inside=True)
uax[2], uhat[2] = make_uniform_grid(sysLTI[2].U, grid_counts=5, filter_inside=True)

# Grid states (X) for each agent
xax  = {}
xhat = {}
xax[1], xhat[1] = make_uniform_grid(sysLTI[1].X, grid_counts=1000, filter_inside=True)
xax[2], xhat[2] = make_uniform_grid(sysLTI[2].X, grid_counts=1000, filter_inside=True)

# Compute transition matrix P for each agent (flat shape: N(X) x (N(X)*N(U)))
P = {}
P[1] = transition_matrix_nd_separable(
    sys=sysLTI[1],
    X_axes=xax[1],              # list of axes; 1D → [axis]
    U_points=uhat[1].ravel(),   # discrete inputs
    X_poly=sysLTI[1].X,
    tol=1e-15,
    renormalize=True,
    return_flat=True
)

P[2] = transition_matrix_nd_separable(
    sys=sysLTI[2],
    X_axes=xax[2],              # list of axes; 1D → [axis]
    U_points=uhat[2].ravel(),   # discrete inputs
    X_poly=sysLTI[2].X,
    tol=1e-15,
    renormalize=True,
    return_flat=True
)

# ==== Build abstract MDP for each agent ====
sysAbs = {}
sysAbs[1] = MDPModel(
    P=P[1],            # flat transition matrix (N x N*M)
    hx=xax[1],         # list of state grid axes (1D → [axis])
    orig=sysLTI[1],    # original continuous model (optional)
    inputs=uhat[1],    # discrete input set; class reshapes if needed
    labels=sysLTI[1].AP
)

sysAbs[2] = MDPModel(
    P=P[2],            # flat transition matrix (N x N*M)
    hx=xax[2],         # list of state grid axes (1D → [axis])
    orig=sysLTI[2],    # original continuous model (optional)
    inputs=uhat[2],    # discrete input set; class reshapes if needed
    labels=sysLTI[2].AP
)

# Quick checks
ok, max_err = sysAbs[1].check_rowsum(1e-10)
print(ok, max_err)

# Access the k-th action block in both views
k = 0
Bk  = sysAbs[1].block(k)                 # (N, N)
cols = sysAbs[1].block_cols(k)           # slice for flat view
Bk2 = sysAbs[1].P_flat[:, cols]          # (N, N), same as Bk

# ====== Deterministic labeling ======
