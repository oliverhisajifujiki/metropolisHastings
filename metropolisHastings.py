# Metropolis–Hastings algorithm (likelihood-only version)
import numpy as np

def metropolis_hastings(log_like, start, proposal, n_samples):
    samples = [start]
    theta = start
    for _ in range(n_samples):
        theta_star = proposal(theta)
        r = np.exp(log_like(theta_star) - log_like(theta))
        if np.random.rand() < min(1, r):
            theta = theta_star
        samples.append(theta)
    return np.array(samples)
