import numpy as np
from scipy.linalg import svd
from scipy.stats import norm
from scipy.sparse import coo_matrix
from scipy.linalg import toeplitz
from itertools import product

class MDPModel:
    def __init__(self, P, hz, states, beta, sys):
        self.P = P
        self.hz = hz
        self.states = states
        self.beta = beta
        self.sys = sys
        self.zstates = None
        self.outputs = None
        self.inputs = None
        self.l = None
        self.outputmap = None

def GridSpace_nonlin_tensor(sys, uhat, l, tol, tensortool="tensortoolbox"):
    if not hasattr(sys, 'Bw'):
        raise ValueError('sys.Bw does not exist')
    if not hasattr(sys, 'sigma'):
        raise ValueError('sys.sigma does not exist')
    if not hasattr(sys, 'dim'):
        raise ValueError('sys.dim does not exist')
    
    dim = sys.dim
    mu = getattr(sys, 'mu', np.zeros(dim))
    
    if np.isscalar(sys.sigma):
        sigma = np.diag([sys.sigma] * dim)
    elif len(sys.sigma.shape) == 1: # check whether sigma is a vector
        sigma = np.diag(sys.sigma)
    else:
        raise ValueError('sys.sigma is not a vector')
    
    if np.isscalar(mu):
        mu = np.full(dim, mu)
    
    # Transform state space
    if np.array_equal(sys.Bw, np.eye(dim)):
        Uz = np.eye(dim)
        sigma_z = np.diag(sigma)
        mu_z = np.dot(np.eye(dim), sys.mu)
    else:
        Sigma = sys.Bw @ sigma @ sys.Bw.T
        Uz, Sz, _ = svd(Sigma)
        sigma_z = np.diag(Sz)
        mu_z = Uz.T @ sys.Bw @ sys.mu
    
    if np.sum(np.abs(mu_z)) > 0:
        print('Warning: Implementation does not hold for mu not equal to zero')
    
    # Compute bounding box in transformed space
    if not hasattr(sys, 'X'):
        raise ValueError('sys.X does not exist or is not a Polyhedron')
    
    Z = Uz.T @ sys.X
    Zl = np.min(Z.V, axis=0)
    Zu = np.max(Z.V, axis=0)
    
    if np.isscalar(l):
       raise ValueError('l must be a vector')
    
    # Compute uniform grid
    gridSize = (Zu - Zl) / l
    hz = [np.arange(zl + gs / 2, zu, gs) for zl, zu, gs in zip(Zl, Zu, gridSize)]
    ZhatSpace = np.array(list(product(*hz))).T
    XhatSpace = Uz @ ZhatSpace
    
    # Compute deterministic probability matrix
    nXhatSpace = XhatSpace.shape[1]
    nUhat = uhat.shape[1]
    index_total = np.arange(len(ZhatSpace))
    i_indices, j_indices = [], []
    
    for k in range(nUhat):
        z_n = Uz.T @ sys.f_det(XhatSpace, np.tile(uhat[:, k], (1, nXhatSpace)))
        z_n_ind = np.floor(np.linalg.inv(np.diag(gridSize)) @ (z_n - Zl[:, None])) + 1
        valid_indices = np.all((1 <= z_n_ind) & (z_n_ind <= np.array(l)[:, None]), axis=0)
        
        zi_indices = [z_n_ind[i, valid_indices] for i in range(dim)]
        i_indices.extend(np.ravel_multi_index(zi_indices, l))
        j_indices.extend(index_total[valid_indices] + k * nXhatSpace)
    
    P_det = coo_matrix((np.ones(len(i_indices)), (i_indices, j_indices)), 
                        shape=(nXhatSpace, nXhatSpace * nUhat))
    
    # Compute stochastic probabilities
    mx2 = [np.arange(-gs * 0.5, gs * (li - 0.5), gs) for gs, li in zip(gridSize, l)]
    Ps = []
    

    for d in range(dim):
        cp = np.diff(norm.cdf(mx2[d], mu_z[d], sigma_z[d]))
        cp[cp < tol] = 0
        Ps.append(toeplitz(cp))

    P = [P_det,Ps]
    
    beta = Uz @ np.array([np.diag(2 * gridSize) @ (np.array(ff2n(dim)) - 0.5).T])
    states = XhatSpace
    sysAbs = MDPModel(P, hz, states, beta, sys)
    sysAbs.zstates = ZhatSpace
    sysAbs.outputs = sys.C @ XhatSpace
    sysAbs.inputs = uhat
    sysAbs.l = l
    sysAbs.outputmap = sys.C @ Uz
    
    return sysAbs
