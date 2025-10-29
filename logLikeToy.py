import numpy as np

def logLike(theta):
    """
    Toy log-likelihood for testing.
    For now, assume a standard normal target: log p(theta) ∝ -0.5 * theta^2
    """
    return -0.5 * np.sum(theta**2)

def logLikeMultimodal1D(theta):
    """1D mixture of two Gaussians at -3 and +3."""
    return np.log(0.5 * np.exp(-0.5*(theta + 3)**2) + 0.5 * np.exp(-0.5*(theta - 3)**2))

def logLike2DGaussian(theta):
    """2D single Gaussian centered at (0,0)."""
    return -0.5 * np.sum(theta**2)

def logLikeMultimodal2D(theta):
    """2D mixture of two Gaussians centered at (-3, -3) and (3, 3)."""
    return np.log(
        0.5 * np.exp(-0.5 * np.sum((theta - np.array([-3, -3]))**2)) +
        0.5 * np.exp(-0.5 * np.sum((theta - np.array([3, 3]))**2))
    )