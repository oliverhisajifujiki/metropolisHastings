import unittest
import numpy as np
from metropolisHastings import metropolisHastings, proposal, stopWhenStable
from logLikeToy import logLike


class TestMetropolisHastings(unittest.TestCase):
    def testBasicRun(self):
        """Test that the sampler runs and returns expected shapes."""
        samples, acceptRate = metropolisHastings(
            logLikeFunc=logLike,
            proposalFunc=proposal,
            start=[0.0],
            stepSize=0.5,
            stopFunc=stopWhenStable,
            nSamples=5000
        )

        # check shapes
        self.assertTrue(isinstance(samples, np.ndarray))
        self.assertTrue(len(samples.shape) == 2)
        self.assertTrue(samples.shape[1] == 1)

        # check acceptance rate is within a valid range
        self.assertTrue(0 <= acceptRate <= 1)

    def testStabilityStop(self):
        """Ensure the stopping function triggers within reasonable bounds and the chain has stabilized."""
        samples, _ = metropolisHastings(
            logLikeFunc=logLike,
            proposalFunc=proposal,
            start=[0.0],
            stepSize=0.5,
            stopFunc=stopWhenStable,
            nSamples=20000
        )

        # confirm the sampler actually stopped early
        self.assertTrue(len(samples) < 20000, "Stopping condition failed to trigger early")

        # use the same window definition as in stopWhenStable
        window = 200
        if len(samples) >= 2 * window:
            meanRecent = np.mean(samples[-window:])
            meanPrev = np.mean(samples[-2*window:-window])
            diff = abs(meanRecent - meanPrev)

            # use a slightly looser stability check
            self.assertTrue(diff < 0.02, f"Chain did not stabilize: mean diff = {diff}")


if __name__ == "__main__":
    unittest.main()
