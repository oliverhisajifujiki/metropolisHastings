# mhUnimodal2DPlot.py
import numpy as np
import matplotlib.pyplot as plt
from metropolisHastings import metropolisHastings
from logLikeToy import logLike2DGaussian
from scipy.stats import gaussian_kde

# starting point and symmetric Gaussian proposal
start = np.array([0.0, 0.0])
proposal = lambda theta, step: theta + np.random.normal(0, step, size=2)

# run sampler
samples, acceptRate = metropolisHastings(
    logLike2DGaussian,
    proposal,
    start=start,
    nSamples=80000,
    stepSize=1.0
)


# convert to array and remove burn-in
samples = np.array(samples)
burnIn = len(samples) // 5
samples = samples[burnIn:]

x, y = samples[:, 0], samples[:, 1]

# estimate KDE of sample density
xy = np.vstack([x, y])
kde = gaussian_kde(xy)

# make a grid to evaluate KDE and the true density
xgrid = np.linspace(-4, 4, 200)
ygrid = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(xgrid, ygrid)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# true Gaussian for contour reference
Ztrue = np.exp(-0.5 * (X**2 + Y**2))
Ztrue /= np.sum(Ztrue)

plt.style.use("default")
fig, ax = plt.subplots(figsize=(6, 5))

# filled contour for KDE-estimated sample density
cf = ax.contourf(X, Y, Z, levels=30, cmap="Oranges", alpha=0.8)

# true Gaussian contour outlines
ax.contour(X, Y, Ztrue, colors="#562717", linewidths=1.2, levels=8)

ax.set_xlabel("θ₁")
ax.set_ylabel("θ₂")
ax.set_title("Metropolis–Hastings on 2D Gaussian (KDE density)")

plt.colorbar(cf, ax=ax, label="Sample density (KDE estimate)")

# info box
ax.text(
    0.98, 0.95,
    f"accept = {acceptRate:.2%}",
    ha='right', va='top',
    transform=ax.transAxes,
    color="#562717",
    fontsize=10,
    bbox=dict(facecolor=(1,1,1,0.6), edgecolor="#E76219", boxstyle='round,pad=0.4')
)

plt.tight_layout()
plt.savefig("mhUnimodal2D_KDE.png", dpi=300, transparent=True)