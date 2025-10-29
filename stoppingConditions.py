# stoppingConditions.py
import numpy as np
import time
from typing import Callable, List, Optional

# ============================================================
# Global settings (tweak from your driver code if you want)
# ============================================================
STOP_SETTINGS = {
    "printEvery": 1000,        # print progress every N iters
    "enableIterationLimit": True,
    "iterationLimit": 100_000, # hard cap on iterations
    "enableTimeLimit": False,
    "timeLimitSec": 300        # stop after N seconds (if enabled)
}

_startTime = time.time()  # used for elapsed time prints

# ============================================================
# Utilities
# ============================================================

def _printProgress(samples: np.ndarray, iteration: int, label: Optional[str] = None) -> None:
    """Internal: periodic progress output."""
    if iteration == 0:
        return
    if iteration % STOP_SETTINGS["printEvery"] != 0:
        return
    mean = float(np.mean(samples))
    sd   = float(np.std(samples))
    elapsed = time.time() - _startTime
    tag = f"[{label}] " if label else ""
    print(f"{tag}[iter {iteration:>6}] mean={mean: .4f}, sd={sd: .4f}, elapsed={elapsed: .1f}s")

def _announceTriggers(iteration: int, triggers: List[str]) -> None:
    print(f"\n✅ stopping condition triggered at iteration {iteration}")
    for name in triggers:
        print(f"  ↳ {name} = True")

# ============================================================
# Single-chain conditions (take: samples, iteration) -> bool
# ============================================================

def meanStability(samples: np.ndarray, iteration: int, *, window: int = 2000, tol: float = 1e-3) -> bool:
    """
    Convergence via running-mean stability:
    compare the mean of two consecutive windows of equal size.
    """
    n = len(samples)
    if n < 2 * window:
        return False
    recent = samples[-window:]
    past   = samples[-2*window:-window]
    return abs(float(np.mean(recent) - np.mean(past))) < tol

def effectiveSampleSize1D(samples: np.ndarray) -> float:
    """
    Very lightweight ESS estimate based on the autocorrelation function.
    Assumes a 1D chain.
    """
    x = samples - np.mean(samples)
    n = len(x)
    if n < 2:
        return 0.0
    acf = np.correlate(x, x, mode="full")
    acf = acf[acf.size // 2:]  # keep non-negative lags
    if acf[0] == 0:
        return 0.0
    acf = acf / acf[0]

    # positive lags until the first non-positive point (simple truncation rule)
    pos = []
    for k in range(1, len(acf)):
        if acf[k] <= 0:
            break
        pos.append(acf[k])
    tau = 1.0 + 2.0 * float(np.sum(pos))
    if tau <= 0:
        return 0.0
    return n / tau

def essThreshold(samples: np.ndarray, iteration: int, *, threshold: int = 5000) -> bool:
    """Stop when 1D ESS exceeds a threshold."""
    ess = effectiveSampleSize1D(samples)
    return ess >= threshold

def maxIterations(samples: np.ndarray, iteration: int) -> bool:
    """Stop if iteration cap enabled and reached."""
    return STOP_SETTINGS["enableIterationLimit"] and (iteration >= STOP_SETTINGS["iterationLimit"])

def maxWallTime(samples: np.ndarray, iteration: int) -> bool:
    """Stop if wall-time cap enabled and exceeded."""
    if not STOP_SETTINGS["enableTimeLimit"]:
        return False
    elapsed = time.time() - _startTime
    return elapsed >= STOP_SETTINGS["timeLimitSec"]

# ============================================================
# Multi-chain (R-hat) condition
# ============================================================

def gelmanRubinRHat(chains: List[np.ndarray]) -> float:
    """
    Classic potential scale reduction (R-hat) for multiple chains, 1D parameter.
    Expects each chain to be a 1D array of same length.
    """
    m = len(chains)
    if m < 2:
        return np.inf
    n = min(len(c) for c in chains)
    if n < 2:
        return np.inf
    # align lengths
    aligned = np.stack([np.asarray(c[:n], dtype=float) for c in chains], axis=0)  # (m, n)
    chainMeans = np.mean(aligned, axis=1)              # (m,)
    overallMean = float(np.mean(chainMeans))
    B = n * float(np.var(chainMeans, ddof=1))
    W = float(np.mean(np.var(aligned, axis=1, ddof=1)))
    if W == 0:
        return np.inf
    Vhat = (1 - 1/n) * W + B / n
    Rhat = np.sqrt(max(Vhat / W, 0.0))
    return float(Rhat)

def makeRHatCondition(chainsRef: List[np.ndarray], *, tol: float = 1.05, minLength: int = 2000) -> Callable[[np.ndarray, int], bool]:
    """
    Factory: returns a condition(samples, iteration) that checks R-hat across the
    provided chainsRef (a list you keep updating in your driver). We use samples
    (the main chain) length to gate the check until minLength.
    """
    def cond(_samples: np.ndarray, _iteration: int) -> bool:
        # Only evaluate once all chains are long enough.
        if any(len(c) < minLength for c in chainsRef):
            return False
        rhat = gelmanRubinRHat(chainsRef)
        return rhat < tol
    cond.__name__ = f"rHatBelow{tol}"
    return cond

# ============================================================
# Condition combinators
# ============================================================

def combineConditions(*conds: Callable[[np.ndarray, int], bool], mode: str = "and") -> Callable[[np.ndarray, int], bool]:
    """
    Combine several (samples, iteration)->bool conditions into one.
    mode: "and" (all must be True) or "or" (any True).
    """
    m = mode.lower()
    assert m in ("and", "or"), "mode must be 'and' or 'or'"

    def combined(samples: np.ndarray, iteration: int) -> bool:
        results = [c(samples, iteration) for c in conds]
        return all(results) if m == "and" else any(results)

    combined.__name__ = f"combine({','.join(getattr(c, '__name__', 'cond') for c in conds)};{m})"
    return combined

# ============================================================
# Main entry point from your sampler
# ============================================================

def checkStop(
    samples: np.ndarray,
    iteration: int,
    conditions: Optional[List[Callable[[np.ndarray, int], bool]]] = None,
    printLabel: Optional[str] = None
) -> bool:
    """
    Centralized stop checker:
      • prints periodic progress
      • evaluates all conditions (plus optional global guards)
      • prints which ones fired when stopping
    """
    _printProgress(samples, iteration, printLabel)

    # default: only maxIterations if nothing provided
    activeConds: List[Callable[[np.ndarray, int], bool]] = list(conditions or [])
    # global guards at the end so they’re always enforced:
    activeConds.append(maxIterations)
    activeConds.append(maxWallTime)

    # evaluate
    namesTriggered = []
    stop = False
    for cond in activeConds:
        try:
            ok = cond(samples, iteration)
        except TypeError:
            # allow legacy conditions that accept only samples
            ok = cond(samples)  # type: ignore
        if ok:
            stop = True
            namesTriggered.append(getattr(cond, "__name__", "condition"))

    if stop:
        _announceTriggers(iteration, namesTriggered)
        return True

    return False
