import numpy as np

# === PROPOSAL FUNCTION =======================================================

def proposal(theta, stepSize=0.5):
    """
    Symmetric proposal distribution.
    Given the current state `theta`, propose a new value `thetaStar`
    by adding Gaussian noise with standard deviation = stepSize.
    """
    return np.random.normal(theta, stepSize)



# === METROPOLIS–HASTINGS SAMPLER ============================================
from stoppingConditions import checkStop, meanStability, essThreshold, combineConditions, STOP_SETTINGS

def metropolisHastings(
    logLikeFunc, proposalFunc, start,
    nSamples=200000, stepSize=0.5
):
    samples = []
    theta = np.array(start, dtype=float)
    accepted = 0

    # optional: tweak global settings at runtime
    STOP_SETTINGS["printEvery"] = 10000
    STOP_SETTINGS["enableIterationLimit"] = True
    STOP_SETTINGS["iterationLimit"] = 100000
    STOP_SETTINGS["enableTimeLimit"] = False  # or True, and set timeLimitSec

    # pick/compose your conditions
    stopCond = combineConditions(
        lambda s, i: meanStability(s, i, window=3000, tol=5e-4),
        lambda s, i: essThreshold(s, i, threshold=4000),
        mode="and"
    )

    for i in range(nSamples):
        thetaStar = proposalFunc(theta, stepSize)
        logR = logLikeFunc(thetaStar) - logLikeFunc(theta)
        if np.log(np.random.rand()) < logR:
            theta = thetaStar
            accepted += 1

        samples.append(np.copy(theta))

        if checkStop(np.array(samples), i, conditions=[stopCond], printLabel="MH"):
            break

    samples = np.array(samples)
    acceptRate = accepted / len(samples)
    print(f"\nFinal accept rate: {acceptRate:.2%}")
    return samples, acceptRate

