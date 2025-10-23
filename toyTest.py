from metropolisHastings import metropolisHastings, proposal, stopWhenStable
from logLikeToy import logLike
import numpy as np
import matplotlib.pyplot as plt


# === RUN METROPOLIS–HASTINGS SAMPLER ========================================

# Run the sampler using the toy log-likelihood and a stability-based stop condition
samples, acceptRate = metropolisHastings(
    logLikeFunc=logLike,       # our toy target distribution
    proposalFunc=proposal,     # symmetric Gaussian proposal
    start=[0.0],               # starting point of chain
    stepSize=0.5,              # proposal width
    stopFunc=stopWhenStable,   # dynamic stopping condition
    nSamples=10000             # maximum allowed iterations
)

print(f"Acceptance rate: {acceptRate:.2f}")
print(f"Total samples drawn: {len(samples)}")

# --- verify empirical mean and variance ---
mean_est = np.mean(samples)
var_est = np.var(samples)

print(f"Empirical mean: {mean_est:.4f} (true = 0)")
print(f"Empirical variance: {var_est:.4f} (true = 1)")


# === VISUALIZATION SECTION ===================================================

# Flatten in case of single dimension
samples = samples.flatten()

# --- Trace plot --------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(samples, lw=0.5)
plt.title("Trace Plot")
plt.xlabel("Iteration")
plt.ylabel("θ")

# --- Histogram vs. true distribution -----------------------------------------
plt.subplot(1,2,2)
plt.hist(samples, bins=50, density=True, alpha=0.7, label="Sampled")
x = np.linspace(-4, 4, 200)
plt.plot(x, (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**2), 'r--', label="True N(0,1)")
plt.title("Sampled vs True Distribution")
plt.xlabel("θ")
plt.legend()

plt.tight_layout()
plt.show()


