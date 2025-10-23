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

def metropolisHastings(logLikeFunc, proposalFunc, start, stepSize=0.5, stopFunc=None, nSamples=5000):
    """
    Generic Metropolis–Hastings sampler with optional stopping condition.

    Parameters
    ----------
    logLikeFunc : function
        Function returning the log-likelihood (or unnormalized log posterior)
        of the target distribution at a given parameter value.

    proposalFunc : function
        Function that generates a proposed sample given the current one.
        Must take (theta, stepSize) as arguments.

    start : float or array-like
        Starting value for the chain (initial parameter guess).

    stepSize : float
        Standard deviation of the proposal distribution (controls jump size).

    stopFunc : function or None
        Optional stopping condition.
        Must accept arguments (samples, i, acceptRate) and return True to stop.

    nSamples : int
        Maximum number of iterations (acts as a safety cap).

    Returns
    -------
    samples : np.ndarray
        Array of sampled parameter values, shape (nSamples, dim) truncated to stop iteration.

    acceptRate : float
        Fraction of proposed samples that were accepted.
    """

    # Create an array to store samples; shape depends on parameter dimension
    samples = np.zeros((nSamples, np.size(start)))

    # Convert the starting point to an array (handles both scalars and vectors)
    theta = np.array(start, dtype=float)

    # Counter for accepted proposals
    accepted = 0

    # --- MAIN SAMPLING LOOP ---------------------------------------------------
    for i in range(nSamples):

        # (1) Draw a candidate sample from the proposal distribution
        thetaStar = proposalFunc(theta, stepSize)

        # (2) Compute log of acceptance ratio:
        #     logR = log(p(thetaStar)) - log(p(theta))
        logR = logLikeFunc(thetaStar) - logLikeFunc(theta)

        # (3) Accept or reject based on random uniform threshold
        if np.log(np.random.rand()) < logR:
            theta = thetaStar
            accepted += 1

        # (4) Save current state (accepted or not)
        samples[i] = theta

        # (5) Update running acceptance rate
        acceptRate = accepted / (i + 1)

        # (6) Check for stopping condition (if provided)
        if stopFunc is not None and stopFunc(samples[:i+1], i, acceptRate):
            print(f"Stopping condition reached at iteration {i}.")
            samples = samples[:i+1]
            break

    else:
        # If loop completes naturally, compute final acceptance rate
        acceptRate = accepted / nSamples

    return samples, acceptRate



# === STOPPING CONDITION EXAMPLE ==============================================

def stopWhenStable(samples, i, acceptRate, window=200, tol=0.01):
    """
    Example stopping condition:
    Stop when the mean of the last `window` samples changes by less than `tol`
    compared to the previous window.

    Parameters
    ----------
    samples : np.ndarray
        Chain of samples up to current iteration.
    i : int
        Current iteration index.
    acceptRate : float
        Current acceptance rate (not used here, but included for flexibility).
    window : int
        Number of samples per window for mean comparison.
    tol : float
        Threshold for mean difference.

    Returns
    -------
    bool
        True if mean change < tol, False otherwise.
    """

    # Require enough samples to form two full windows
    if i < max(2 * window, 1000):
        return False

    # Compute mean of most recent window
    recentMean = np.mean(samples[i-window:i])

    # Compute mean of previous window
    prevMean = np.mean(samples[i-2*window:i-window])

    # Stop if absolute mean difference < tol
    return abs(recentMean - prevMean) < tol
