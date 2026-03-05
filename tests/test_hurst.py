"""Tests for the Hurst exponent analysis module."""

from __future__ import annotations

import random
import unittest

from shared.analysis_result import AnalysisResult
from shared.hurst import rescaled_range_hurst, run_hurst_analysis


class TestHurst(unittest.TestCase):
    """Tests for Hurst exponent computation and analysis."""

    def test_uniform_draws_hurst_near_half(self) -> None:
        """500 uniform draws should yield mean H between 0.3 and 0.7."""
        rng = random.Random(42)
        pool_size = 20
        numbers_per_draw = 5
        draws: list[list[int]] = []
        for _ in range(500):
            draw = sorted(rng.sample(range(1, pool_size + 1), numbers_per_draw))
            draws.append(draw)

        results = run_hurst_analysis(draws, pool_size)
        self.assertTrue(len(results) > 0)

        aggregate = [r for r in results if r.test_name == "hurst_aggregate"]
        self.assertEqual(len(aggregate), 1)

        mean_h = aggregate[0].details["mean_hurst"]
        self.assertGreaterEqual(mean_h, 0.3)
        self.assertLessEqual(mean_h, 0.7)

    def test_rescaled_range_basic(self) -> None:
        """A random walk series should yield H between 0.2 and 0.8."""
        rng = random.Random(99)
        series = [rng.gauss(0, 1) for _ in range(200)]

        h = rescaled_range_hurst(series, min_window=8)
        self.assertIsNotNone(h)
        self.assertGreaterEqual(h, 0.2)
        self.assertLessEqual(h, 0.8)

    def test_persistent_series_high_hurst(self) -> None:
        """A trending series [0,1,2,...,199] should have H > 0.7."""
        series = list(range(200))

        h = rescaled_range_hurst(series, min_window=8)
        self.assertIsNotNone(h)
        self.assertGreater(h, 0.7)

    def test_too_few_draws_inconclusive(self) -> None:
        """30 draws should still produce a result (possibly inconclusive)."""
        rng = random.Random(7)
        pool_size = 20
        draws: list[list[int]] = []
        for _ in range(30):
            draw = sorted(rng.sample(range(1, pool_size + 1), 5))
            draws.append(draw)

        results = run_hurst_analysis(draws, pool_size)
        self.assertTrue(len(results) > 0)

    def test_returns_analysis_results(self) -> None:
        """All results should be AnalysisResult with 'hurst' in test_name."""
        rng = random.Random(123)
        pool_size = 20
        draws: list[list[int]] = []
        for _ in range(500):
            draw = sorted(rng.sample(range(1, pool_size + 1), 5))
            draws.append(draw)

        results = run_hurst_analysis(draws, pool_size)
        for result in results:
            self.assertIsInstance(result, AnalysisResult)
            self.assertIn("hurst", result.test_name)


if __name__ == "__main__":
    unittest.main()
