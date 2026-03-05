import random
import unittest

from shared.randomness_tests import run_randomness_tests
from shared.analysis_result import AnalysisResult


class TestRandomnessTests(unittest.TestCase):
    def test_uniform_data_passes_all(self):
        """500 uniform random draws from pool 1-45, pick 5 -> all tests pass."""
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(500)]

        results = run_randomness_tests(draws, pool_size=45)

        conclusive = [r for r in results if r.passed is not None]
        for r in conclusive:
            self.assertTrue(
                r.passed,
                f"{r.test_name} unexpectedly failed: p={r.p_value}, summary={r.summary}",
            )

    def test_biased_data_fails_frequency(self):
        """Draws heavily biased toward low numbers -> frequency monobit fails."""
        rng = random.Random(99)
        biased_pool = list(range(1, 11))  # Only numbers 1-10
        draws = [sorted(rng.sample(biased_pool, 5)) for _ in range(500)]

        results = run_randomness_tests(draws, pool_size=45)

        frequency_results = [r for r in results if r.test_name == "frequency_monobit"]
        self.assertTrue(len(frequency_results) == 1)
        self.assertFalse(frequency_results[0].passed)

    def test_returns_analysis_results(self):
        """All returned items have proper AnalysisResult fields."""
        rng = random.Random(7)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(100)]

        results = run_randomness_tests(draws, pool_size=45)

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsInstance(r, AnalysisResult)
            self.assertIsInstance(r.test_name, str)
            self.assertEqual(r.game, "all")
            self.assertIn(r.passed, (True, False, None))
            self.assertIsInstance(r.statistic, float)
            self.assertIsInstance(r.threshold, float)
            self.assertIsInstance(r.sample_size, int)
            self.assertIsInstance(r.details, dict)
            self.assertIsInstance(r.summary, str)

    def test_too_few_draws_returns_inconclusive(self):
        """With only 10 draws, serial and approximate_entropy should be inconclusive."""
        rng = random.Random(123)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(10)]

        results = run_randomness_tests(draws, pool_size=45)

        serial = [r for r in results if r.test_name == "serial"]
        approx_entropy = [r for r in results if r.test_name == "approximate_entropy"]

        self.assertTrue(len(serial) == 1)
        self.assertIsNone(serial[0].passed)

        self.assertTrue(len(approx_entropy) == 1)
        self.assertIsNone(approx_entropy[0].passed)

    def test_different_pool_sizes(self):
        """Works for (45,5), (49,6), (40,6)."""
        rng = random.Random(55)

        configs = [(45, 5), (49, 6), (40, 6)]
        for pool_size, pick_count in configs:
            draws = [
                sorted(rng.sample(range(1, pool_size + 1), pick_count))
                for _ in range(200)
            ]
            results = run_randomness_tests(draws, pool_size=pool_size)
            self.assertGreater(
                len(results),
                0,
                f"No results for pool_size={pool_size}, pick={pick_count}",
            )
            for r in results:
                self.assertIsInstance(r, AnalysisResult)
                self.assertEqual(r.game, "all")

    def test_six_tests_returned(self):
        """Exactly 6 test types are returned regardless of data size."""
        rng = random.Random(1)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(500)]

        results = run_randomness_tests(draws, pool_size=45)
        test_names = {r.test_name for r in results}

        expected = {
            "frequency_monobit",
            "runs",
            "longest_run",
            "cumulative_sums",
            "serial",
            "approximate_entropy",
        }
        self.assertEqual(test_names, expected)

    def test_custom_significance(self):
        """Custom significance level is respected."""
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(500)]

        results_strict = run_randomness_tests(draws, pool_size=45, significance=0.001)
        results_loose = run_randomness_tests(draws, pool_size=45, significance=0.1)

        # With looser significance, at least as many tests should pass
        strict_passes = sum(1 for r in results_strict if r.passed is True)
        loose_passes = sum(1 for r in results_loose if r.passed is True)
        self.assertGreaterEqual(loose_passes, strict_passes)

    def test_inconclusive_for_runs_with_few_draws(self):
        """Runs test needs >= 100 draws; with 50 it should be inconclusive."""
        rng = random.Random(10)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(50)]

        results = run_randomness_tests(draws, pool_size=45)
        runs = [r for r in results if r.test_name == "runs"]
        self.assertTrue(len(runs) == 1)
        self.assertIsNone(runs[0].passed)

    def test_inconclusive_for_longest_run_with_few_draws(self):
        """Longest run needs >= 200 draws; with 100 it should be inconclusive."""
        rng = random.Random(10)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(100)]

        results = run_randomness_tests(draws, pool_size=45)
        longest = [r for r in results if r.test_name == "longest_run"]
        self.assertTrue(len(longest) == 1)
        self.assertIsNone(longest[0].passed)

    def test_p_values_in_valid_range(self):
        """All non-None p-values should be in [0, 1]."""
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(500)]

        results = run_randomness_tests(draws, pool_size=45)
        for r in results:
            if r.p_value is not None:
                self.assertGreaterEqual(r.p_value, 0.0, f"{r.test_name} p<0")
                self.assertLessEqual(r.p_value, 1.0, f"{r.test_name} p>1")


if __name__ == "__main__":
    unittest.main()
