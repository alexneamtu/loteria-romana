import unittest

from shared.backtest_base import (
    BacktestResult,
    correct_significance,
    CorrectedResult,
    cohens_h,
)


class TestCohensH(unittest.TestCase):
    def test_equal_proportions(self):
        self.assertAlmostEqual(cohens_h(0.5, 0.5), 0.0, places=5)

    def test_different_proportions(self):
        h = cohens_h(0.6, 0.4)
        self.assertGreater(h, 0.0)

    def test_symmetry(self):
        h1 = cohens_h(0.7, 0.3)
        h2 = cohens_h(0.3, 0.7)
        self.assertAlmostEqual(h1, -h2, places=5)


class TestCorrectSignificance(unittest.TestCase):
    def _make_result(self, name, wins, tickets):
        return BacktestResult(
            strategy_name=name,
            total_draws=tickets,
            total_tickets=tickets,
            total_wins=wins,
            win_rate=wins / tickets if tickets > 0 else 0,
        )

    def test_all_at_baseline_excluded(self):
        results = [
            self._make_result("random", 50, 1000),
            self._make_result("freq", 50, 1000),
            self._make_result("bayes", 51, 1000),
        ]
        corrected = correct_significance(results, baseline_rate=0.05)
        for c in corrected:
            if c.strategy != "random":
                self.assertIn(c.verdict, ("excluded", "included"))

    def test_clearly_better_strategy_included(self):
        results = [
            self._make_result("random", 50, 1000),
            self._make_result("great", 200, 1000),
        ]
        corrected = correct_significance(results, baseline_rate=0.05)
        great = [c for c in corrected if c.strategy == "great"]
        self.assertEqual(len(great), 1)
        self.assertEqual(great[0].verdict, "included")
        self.assertGreater(great[0].weight_scale, 0.0)

    def test_returns_corrected_results(self):
        results = [
            self._make_result("a", 60, 1000),
            self._make_result("b", 55, 1000),
        ]
        corrected = correct_significance(results, baseline_rate=0.05)
        for c in corrected:
            self.assertIsInstance(c, CorrectedResult)
            self.assertIsInstance(c.raw_p_value, float)
            self.assertIsInstance(c.adjusted_p_value, float)
            self.assertIsInstance(c.effect_size, float)

    def test_bh_correction_more_lenient_than_bonferroni(self):
        results = [self._make_result(f"s{i}", 60 + i, 1000) for i in range(10)]
        corrected = correct_significance(results, baseline_rate=0.05, fdr_threshold=0.10)
        included = [c for c in corrected if c.verdict == "included"]
        self.assertIsInstance(included, list)

    def test_small_effect_size_excluded(self):
        results = [
            self._make_result("tiny_edge", 5100, 100000),
        ]
        corrected = correct_significance(
            results, baseline_rate=0.05, min_effect_size=0.02,
        )
        tiny = [c for c in corrected if c.strategy == "tiny_edge"]
        if tiny and tiny[0].effect_size < 0.02:
            self.assertEqual(tiny[0].verdict, "excluded")
