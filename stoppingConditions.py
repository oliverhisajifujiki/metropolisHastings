# stoppingConditions.py
import numpy as np
import time

# -----------------------------
# Global Settings
# -----------------------------
STOP_SETTINGS = {
    "printEvery": 1000,         # heartbeat
    "enableIterationLimit": True,
    "iterationLimit": 100_000,

    "enableTimeLimit": False,
    "timeLimit": 300,           # seconds

    "enableESS": True,
    "essThreshold": 4000,

    "enableStability": False,
    "stabilityWindow": 1000,
    "stabilityTolerance": 1e-3,
}

_START_TIME = time.time()

# -----------------------------
# Helpers: 1D vs ND samples
# -----------------------------
def as2D(samples):
    """Return samples as shape (n, d)."""
    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

# -----------------------------
# ESS utilities
# -----------------------------
def autocorr1D(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.array([1.0])
    x = x - np.mean(x)
    acf = np.correlate(x, x, mode="full")
    acf = acf[acf.size // 2 :]
    if acf[0] == 0:
        return np.array([1.0])
    return acf / acf[0]

def effectiveSampleSize1D(x):
    n = len(x)
    if n <= 1:
        return 1.0
    acf = autocorr1D(x)
    # Geyer positive-sequence truncation: keep positive until first non-positive
    pos = []
    for k in range(1, len(acf)):
        if acf[k] <= 0:
            break
        pos.append(acf[k])
    tau = 1.0 + 2.0 * np.sum(pos) if pos else 1.0
    ess = n / max(tau, 1e-12)
    return max(1.0, float(ess))

def effectiveSampleSizeND(samples):
    arr = as2D(samples)
    ess_vals = [effectiveSampleSize1D(arr[:, j]) for j in range(arr.shape[1])]
    return float(np.min(ess_vals))

def essThreshold(samples, iteration, threshold=None):
    if threshold is None:
        threshold = STOP_SETTINGS["essThreshold"]
    ess = effectiveSampleSizeND(samples)
    if ess > threshold:
        print(f"\n[STOP] ESS threshold reached → ESS = {ess:.1f} > {threshold}")
        return True
    return False

# -----------------------------
# Stability (running-mean)
# -----------------------------
def stabilityWindow(samples, iteration, window=None, tol=None):
    if window is None:
        window = STOP_SETTINGS["stabilityWindow"]
    if tol is None:
        tol = STOP_SETTINGS["stabilityTolerance"]
    arr = as2D(samples)
    n = arr.shape[0]
    if window <= 0 or n < 2 * window:
        return False
    recent = arr[-window:]
    prev   = arr[-2*window:-window]
    diff = np.linalg.norm(np.mean(recent, axis=0) - np.mean(prev, axis=0))
    if diff < tol:
        print(f"\n[STOP] Stability window converged → Δmean = {diff:.2e} < {tol}")
        return True
    return False

# ---- Backward compatible alias (old name) ----
def meanStability(samples, iteration, window=None, tol=None):
    return stabilityWindow(samples, iteration, window=window, tol=tol)

# -----------------------------
# Iteration / Time limits
# -----------------------------
def iterationLimit(samples, iteration):
    if STOP_SETTINGS["enableIterationLimit"] and iteration >= STOP_SETTINGS["iterationLimit"]:
        print(f"\n[STOP] Iteration limit reached → {iteration} ≥ {STOP_SETTINGS['iterationLimit']}")
        return True
    return False

def timeLimit(samples, iteration):
    if not STOP_SETTINGS["enableTimeLimit"]:
        return False
    elapsed = time.time() - _START_TIME
    if elapsed > STOP_SETTINGS["timeLimit"]:
        print(f"\n[STOP] Time limit reached → {elapsed:.1f}s > {STOP_SETTINGS['timeLimit']}s")
        return True
    return False

# -----------------------------
# Composition utilities
# -----------------------------
def combined(*conds):
    """OR-combine conditions (no kwargs)."""
    def inner(samples, iteration):
        for c in conds:
            try:
                if c(samples, iteration):
                    return True
            except TypeError:
                # allow legacy conds that only accept samples
                if c(samples):
                    return True
        return False
    inner.__name__ = "combined"
    return inner

# ---- Backward compatible: combineConditions with mode= ----
def combineConditions(*conds, mode="or"):
    """
    Backward-compatible combiner.
    mode: 'or' (default) or 'and'
    Each cond is called as cond(samples, iteration) if possible, otherwise cond(samples).
    """
    mode = str(mode).lower()
    if mode not in ("or", "and"):
        raise ValueError("mode must be 'or' or 'and'")

    def inner(samples, iteration):
        results = []
        for c in conds:
            try:
                results.append(bool(c(samples, iteration)))
            except TypeError:
                results.append(bool(c(samples)))
        return any(results) if mode == "or" else all(results)

    inner.__name__ = f"combineConditions_{mode}"
    return inner

# -----------------------------
# Main check
# -----------------------------
def checkStop(samples, iteration, conditions=None, printLabel=""):
    if conditions is None:
        conditions = []

    active = []
    if STOP_SETTINGS["enableIterationLimit"]:
        active.append(iterationLimit)
    if STOP_SETTINGS["enableTimeLimit"]:
        active.append(timeLimit)
    if STOP_SETTINGS["enableESS"]:
        active.append(lambda s, i: essThreshold(s, i))
    if STOP_SETTINGS["enableStability"]:
        active.append(lambda s, i: stabilityWindow(s, i))

    active.extend(conditions)

    for cond in active:
        if cond(samples, iteration):
            name = getattr(cond, "__name__", repr(cond))
            print(f"[INFO] {printLabel} stopped due to: {name}")
            return True

    if STOP_SETTINGS["printEvery"] and iteration > 0 and iteration % STOP_SETTINGS["printEvery"] == 0:
        print(f"[INFO] {printLabel} iteration {iteration}")

    return False
