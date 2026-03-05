"""Tests for Benford's Law analysis of lottery draws."""

import random
import unittest

from shared.analysis_result import AnalysisResult
from shared.benford import run_benford_analysis


class TestBenfordAnalysis(unittest.TestCase):

    def test_uniform_draws_pass(self):
        """500 uniform random draws from pool 1-45 should pass.

        'Passed' means the data is consistent with uniform (not Benford-like).
        """
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(500)]

        results = run_benford_analysis(draws, pool_size=45)

        conclusive = [r for r in results if r.passed is not None]
        self.assertGreater(len(conclusive), 0)
        for r in conclusive:
            self.assertTrue(
                r.passed,
                f"{r.test_name} unexpectedly failed: "
                f"p={r.p_value}, summary={r.summary}",
            )

    def test_returns_benford_vs_uniform_comparison(self):
        """Details dict includes both benford_chi2 and uniform_chi2."""
        rng = random.Random(77)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(500)]

        results = run_benford_analysis(draws, pool_size=45)

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsInstance(r, AnalysisResult)
            self.assertIn("benford_chi2", r.details)
            self.assertIn("uniform_chi2", r.details)
            self.assertIn("benford_p", r.details)
            self.assertIn("uniform_p", r.details)
            self.assertIn("observed_probs", r.details)
            self.assertIn("fits_benford", r.details)
            self.assertIn("fits_uniform", r.details)

    def test_small_sample_inconclusive(self):
        """5 draws (fewer than 50 total numbers) should be inconclusive."""
        rng = random.Random(99)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(5)]

        results = run_benford_analysis(draws, pool_size=45)

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsNone(
                r.passed,
                f"Expected inconclusive for small sample, got passed={r.passed}",
            )


if __name__ == "__main__":
    unittest.main()
