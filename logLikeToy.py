import numpy as np

def logLike(theta):
    """
    Toy log-likelihood for testing.
    For now, assume a standard normal target: log p(theta) ∝ -0.5 * theta^2
    """
    return -0.5 * np.sum(theta**2)
