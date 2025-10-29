

# mhMultimodal1DPlot.py (minimal change)
import numpy as np
import matplotlib.pyplot as plt
from metropolisHastings import metropolisHastings
from logLikeToy import logLikeMultimodal1D

start = np.array([0.0])
samples, acceptRate = metropolisHastings(
    logLikeMultimodal1D,
    lambda theta, step: np.random.normal(theta, step),
    start=start,
    nSamples=80000,
    stepSize=1.0
)

burnIn = len(samples) // 5
samples = samples[burnIn:]
# (…same plotting code…)

nSamples = 80000
# === post-process ===
burnIn = nSamples // 5
samples = samples[burnIn:]  # discard early chain
xVals = np.linspace(-8, 8, 400)
truePdf = 0.5 * np.exp(-0.5*(xVals + 3)**2) + 0.5 * np.exp(-0.5*(xVals - 3)**2)
truePdf /= np.trapz(truePdf, xVals)

# === plotting ===
plt.style.use("default")
fig, ax = plt.subplots(figsize=(6, 4))

# histogram of samples
ax.hist(samples, bins=60, density=True, color="#E76219", alpha=0.6, label="Samples")

# true distribution
ax.plot(xVals, truePdf, color="#562717", lw=2, label="True Distribution")

ax.set_xlabel("θ")
ax.set_ylabel("Density")
ax.set_title("Metropolis–Hastings on a 1D Multimodal Distribution")
ax.legend(frameon=False)

# display acceptance rate
textStr = f"accept = {acceptRate:.2%}"
ax.text(0.98, 0.95, textStr, ha='right', va='top',
        transform=ax.transAxes, color="#562717",
        fontsize=10,
        bbox=dict(facecolor=(1, 1, 1, 0.6), edgecolor="#E76219", boxstyle='round,pad=0.4'))

plt.tight_layout()
plt.savefig("mhMultimodal1D.png", dpi=300, transparent=True)
plt.show()

