import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- add your metropolisHastings folder to the import path ---
sys.path.append(r"C:\Users\olive\2025projectsOctober")

# --- import your sampler and log-likelihood ---
from metropolisHastings import metropolisHastings, proposal
from logLikeToy import logLike  # assuming you have this in the same folder

# --- run sampler ---
samples, acceptRate = metropolisHastings(
    logLikeFunc=logLike,
    proposalFunc=proposal,
    start=[0.0],
    nSamples=5000,
    stepSize=0.5
)
samples = samples.flatten()

# --- compute summary stats ---
empiricalMean = np.mean(samples)
empiricalVar = np.var(samples)
print(f"Empirical mean: {empiricalMean:.4f}, variance: {empiricalVar:.4f}")
print(f"Acceptance rate: {acceptRate:.2%}")

# --- create performance plot ---
plt.style.use('default')
fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
cream = "#fff5e6"
orange = "#e76219"
brown = "#562717"

# match your site’s palette
fig.patch.set_facecolor(cream)
for ax in axes:
    ax.set_facecolor(cream)
    ax.tick_params(colors=brown)
    for spine in ax.spines.values():
        spine.set_color(brown)

# trace plot
axes[0].plot(samples, color=orange, alpha=0.8)
axes[0].set_title("Metropolis–Hastings Trace", color=brown, pad=10)
axes[0].set_ylabel("θ", color=brown)

# histogram
axes[1].hist(samples, bins=40, color=orange, alpha=0.7, edgecolor=brown, density=True)
axes[1].set_title("Posterior Distribution", color=brown, pad=10)
axes[1].set_xlabel("θ", color=brown)
axes[1].set_ylabel("Density", color=brown)

# add summary stats
textStr = f"mean = {empiricalMean:.2f}\nvar = {empiricalVar:.2f}\naccept = {acceptRate:.2%}"
axes[1].text(0.98, 0.95, textStr, ha='right', va='top',
             transform=axes[1].transAxes, color=brown,
             fontsize=10, bbox=dict(facecolor='none' , edgecolor=orange, boxstyle='round,pad=0.4'))

# --- save to your website images folder ---
outputPath = r"C:\Users\olive\ollyfujiki.github.io\images\mh_performance.png"
os.makedirs(os.path.dirname(outputPath), exist_ok=True)
plt.savefig(outputPath, dpi=300, transparent=True)
plt.close(fig)

print(f"Saved performance plot to: {outputPath}")
