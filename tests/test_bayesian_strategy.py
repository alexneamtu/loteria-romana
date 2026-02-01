import math
import random
import unittest

from shared.bayesian import BayesianScorer


class TestBayesianScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = BayesianScorer(
            pool_size=45,
            numbers_to_pick=5,
        )
        self.draws = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [1, 2, 11, 12, 13],
            [3, 4, 14, 15, 16],
            [5, 6, 17, 18, 19],
        ]

    def test_fit_returns_bayesian_result(self):
        result = self.scorer.fit(self.draws)
        self.assertEqual(len(result.posterior), 45)
        self.assertEqual(len(result.alpha), 45)

    def test_posterior_sums_to_one(self):
        result = self.scorer.fit(self.draws)
        self.assertAlmostEqual(sum(result.posterior), 1.0, places=6)

    def test_frequent_numbers_have_higher_posterior(self):
        result = self.scorer.fit(self.draws)
        # Number 1 appears in draws 0 and 2, number 45 appears nowhere
        self.assertGreater(result.posterior[0], result.posterior[44])

    def test_get_probabilities_matches_fit(self):
        probs = self.scorer.get_probabilities(self.draws)
        result = self.scorer.fit(self.draws)
        for a, b in zip(probs, result.posterior):
            self.assertAlmostEqual(a, b)

    def test_sample_dirichlet_sums_to_one(self):
        rng = random.Random(42)
        alpha = [1.0] * 45
        sample = self.scorer._sample_dirichlet(alpha, rng)
        self.assertAlmostEqual(sum(sample), 1.0, places=6)

    def test_sample_dirichlet_all_positive(self):
        rng = random.Random(42)
        alpha = [1.0] * 45
        sample = self.scorer._sample_dirichlet(alpha, rng)
        self.assertTrue(all(s > 0 for s in sample))

    def test_sample_dirichlet_different_each_time(self):
        rng = random.Random(42)
        alpha = [2.0] * 45
        s1 = self.scorer._sample_dirichlet(alpha, rng)
        s2 = self.scorer._sample_dirichlet(alpha, rng)
        self.assertFalse(all(
            abs(a - b) < 1e-10 for a, b in zip(s1, s2)
        ))

    def test_generate_returns_correct_count(self):
        rng = random.Random(42)
        lines = self.scorer.generate(self.draws, 3, rng)
        self.assertEqual(len(lines), 3)

    def test_generate_lines_are_valid(self):
        rng = random.Random(42)
        lines = self.scorer.generate(self.draws, 5, rng)
        for line in lines:
            self.assertEqual(len(line), 5)
            self.assertEqual(sorted(line), line)
            self.assertTrue(all(1 <= n <= 45 for n in line))
            self.assertEqual(len(set(line)), 5)

    def test_generate_lines_are_unique(self):
        rng = random.Random(42)
        lines = self.scorer.generate(self.draws, 10, rng)
        keys = [tuple(l) for l in lines]
        self.assertEqual(len(keys), len(set(keys)))

    def test_name(self):
        self.assertEqual(self.scorer.name, "bayesian")

    def test_empty_draws(self):
        rng = random.Random(42)
        lines = self.scorer.generate([], 2, rng)
        self.assertEqual(len(lines), 2)

    def test_recency_weighting_affects_posterior(self):
        scorer_short = BayesianScorer(45, 5, half_life=5.0)
        scorer_long = BayesianScorer(45, 5, half_life=500.0)
        result_short = scorer_short.fit(self.draws)
        result_long = scorer_long.fit(self.draws)
        # With short half-life, recent draws matter more
        # so posteriors should differ
        diffs = [abs(a - b) for a, b in zip(result_short.posterior, result_long.posterior)]
        self.assertGreater(max(diffs), 0.0001)


if __name__ == "__main__":
    unittest.main()
