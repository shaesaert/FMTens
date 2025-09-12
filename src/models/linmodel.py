import numpy as np
from numpy.random import multivariate_normal

class LinModel:
    """
    LINMODEL Class of LTI systems with noise on the transitions.
    Defines a model with dynamics:
        x(t+1) = A x(t) + B u(t) + Bw w(t)
        y(t)   = C x(t)
    with w(t) ~ N(mu, sigma)
    """

    def __init__(self, A, B, C, D, Bw, *args):
        self.type = "LTI"  # Type of model
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)
        self.Bw = np.array(Bw)
        self.dim = self.Bw.shape[0]

        # Initialize optional attributes
        self.mu = None
        self.sigma = None
        self.wsupport = None
        self.X = None
        self.U = None
        self.regions = None
        self.AP = None
        self.MOR = False
        self.P = None
        self.Q = None
        self.original = None
        self.KKfilter = False
        self.InitState = None
        self.Xdare = None
        self.K = None
        self.Cobs = None

        # Parse arguments
        if len(args) == 0:
            raise ValueError("LinModel requires at least 5 arguments.")
        elif len(args) == 1 and len(args[0]) >= 2:
            # Bounded uniform distribution
            self.set_wsupport(np.array(args[0]))
        elif len(args) == 2:
            # Unbounded Gaussian distribution
            self.mu = np.array(args[0])
            self.sigma = np.array(args[1])
            self.set_wsupport(np.array([[-np.inf, np.inf]]))
        elif len(args) == 3:
            # Bounded Gaussian distribution
            self.mu = np.array(args[0])
            self.sigma = np.array(args[1])
            self.set_wsupport(np.array(args[2]))
        else:
            raise ValueError("Unsupported arguments for calling LinModel.")

    def f_det(self, x, u):
        """
        Computes the deterministic next state:
            x(t+1) = A x(t) + B u(t)
        """
        x = np.array(x)
        u = np.array(u)
        return self.A @ x + self.B @ u

    def f_stoch(self, x, u, w=None):
        """
        Computes the stochastic next state:
            x(t+1) = A x(t) + B u(t) + Bw w(t)
        If w is not provided, samples from N(mu, sigma).
        Returns (x_next, w).
        """
        x = np.array(x)
        u = np.array(u)

        if w is None:
            w = multivariate_normal(self.mu, self.sigma)
        else:
            w = np.array(w)

        x_next = self.A @ x + self.B @ u + self.Bw @ w
        return x_next, w

    def simulate(self, x0, U, steps=None, stochastic=True, random_seed=None):
        """
        Simulates the system trajectory.
        Args:
            x0 : initial state (vector)
            U : sequence of inputs (list/array of vectors)
            steps : number of simulation steps (defaults to len(U))
            stochastic : if True, include noise; otherwise deterministic
            random_seed : seed for reproducibility
        Returns:
            X : array of states (steps+1, dim)
            W : array of noises (steps, noise_dim) if stochastic, else zeros
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        x0 = np.array(x0).flatten()
        steps = steps or len(U)
        noise_dim = self.Bw.shape[1]

        X = np.zeros((steps + 1, self.dim))
        W = np.zeros((steps, noise_dim))
        X[0, :] = x0

        for t in range(steps):
            u = np.array(U[t]).flatten()
            if stochastic:
                x_next, w = self.f_stoch(X[t], u)
                W[t, :] = w
            else:
                x_next = self.f_det(X[t], u)
            X[t + 1, :] = x_next

        return X, W

    def set_wsupport(self, wsupport):
        """
        Sets noise support.
        """
        wsupport = np.array(wsupport)
        if wsupport.shape[1] != 2:
            raise ValueError("Invalid noise support.")

        if wsupport.shape[0] == 1 and wsupport.shape[0] != self.dim:
            # Copy support for all dimensions
            self.wsupport = np.tile(wsupport, (self.dim, 1))
        elif wsupport.shape[0] == self.dim:
            self.wsupport = wsupport
        else:
            raise ValueError("Invalid noise support.")
