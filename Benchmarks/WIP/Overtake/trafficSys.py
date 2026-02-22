import numpy as np


def traffic_sys(delta_t):
    """
    Function to create the LTI system for the scenario
    
    Parameters
    ----------
    delta_t : float
        The time between the time steps
        
    Returns
    -------
    sys : LinModel
        The LTI system modeling the traffic
    """
    
    # Define state space matrices
    A = np.array([
        [1, 0, delta_t, 0],
        [0, 1, 0, delta_t],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    B = np.array([
        [0, 0],
        [0, 0],
        [delta_t, 0],
        [0, delta_t]
    ])
    
    C = np.eye(4)
    D = np.zeros((4, 2))
    Bw = np.eye(4)
    
    dim = len(A)
    
    # Specify mean and variance of disturbance w(t)
    mu = np.zeros(dim)  # mean of disturbance
    sigma = np.eye(dim)  # variance of disturbance
    
    # Set up an LTI model
    sys = LinModel(A, B, C, D, Bw, mu, sigma)
    
    return sys

