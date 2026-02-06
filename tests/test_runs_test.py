import unittest
import random

from shared.runs_test import (
    wald_wolfowitz_runs_test,
    per_number_runs_analysis,
    RunsReport,
)


class TestRunsReport(unittest.TestCase):
    def test_fields(self):
        report = RunsReport(
            number=5,
            observed_runs=10,
            expected_runs=12.0,
            z_score=-0.5,
            p_value=0.6,
            significant=False,
        )
        self.assertEqual(report.number, 5)
        self.assertFalse(report.significant)


class TestWaldWolfowitzRunsTest(unittest.TestCase):
    def test_alternating_sequence(self):
        sequence = [True, False] * 50
        report = wald_wolfowitz_runs_test(sequence, number=1)
        self.assertEqual(report.observed_runs, 100)
        # A perfectly alternating pattern has far too many runs
        # compared to random expectation (~51), so it is significant.
        self.assertTrue(report.significant)

    def test_clustered_sequence(self):
        sequence = [True] * 50 + [False] * 50
        report = wald_wolfowitz_runs_test(sequence, number=1)
        self.assertTrue(report.significant)
        self.assertEqual(report.observed_runs, 2)

    def test_random_sequence_not_significant(self):
        rng = random.Random(42)
        sequence = [rng.random() < 0.5 for _ in range(200)]
        report = wald_wolfowitz_runs_test(sequence, number=1)
        self.assertFalse(report.significant)

    def test_empty_sequence(self):
        report = wald_wolfowitz_runs_test([], number=1)
        self.assertFalse(report.significant)

    def test_all_same(self):
        report = wald_wolfowitz_runs_test([True] * 100, number=1)
        self.assertFalse(report.significant)


class TestPerNumberRunsAnalysis(unittest.TestCase):
    def test_returns_reports_for_all_numbers(self):
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 11), 3)) for _ in range(100)]
        reports = per_number_runs_analysis(draws, pool_size=10)
        self.assertEqual(len(reports), 10)

    def test_report_numbers_match_pool(self):
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 11), 3)) for _ in range(100)]
        reports = per_number_runs_analysis(draws, pool_size=10)
        numbers = {r.number for r in reports}
        self.assertEqual(numbers, set(range(1, 11)))

    def test_random_draws_mostly_not_significant(self):
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(500)]
        reports = per_number_runs_analysis(draws, pool_size=45)
        significant_count = sum(1 for r in reports if r.significant)
        self.assertLessEqual(significant_count, 10)

    def test_biased_number_detected(self):
        draws = []
        rng = random.Random(42)
        for i in range(50):
            pick = sorted(rng.sample(range(2, 11), 2) + [1])
            draws.append(pick)
        for i in range(50):
            pick = sorted(rng.sample(range(2, 11), 3))
            draws.append(pick)
        reports = per_number_runs_analysis(draws, pool_size=10)
        report_1 = next(r for r in reports if r.number == 1)
        self.assertTrue(report_1.significant)

    def test_custom_significance(self):
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 11), 3)) for _ in range(100)]
        reports = per_number_runs_analysis(draws, pool_size=10, alpha=0.001)
        self.assertEqual(len(reports), 10)


if __name__ == "__main__":
    unittest.main()
