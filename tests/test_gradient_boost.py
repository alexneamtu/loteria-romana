import random
import unittest

from shared.gradient_boost import GradientBoostStrategy, SKLEARN_AVAILABLE


class TestGradientBoostStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = GradientBoostStrategy(
            pool_size=45,
            numbers_to_pick=5,
            numbers_drawn=5,
        )
        rng = random.Random(42)
        self.draws = [
            sorted(rng.sample(range(1, 46), 5))
            for _ in range(80)
        ]

    def test_name(self):
        self.assertEqual(self.strategy.name, "gradient_boost")

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed")
    def test_generate_returns_correct_count(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 5, rng)
        self.assertEqual(len(lines), 5)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed")
    def test_generate_returns_valid_lines(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 3, rng)
        for line in lines:
            self.assertEqual(len(line), 5)
            self.assertEqual(sorted(line), line)
            self.assertTrue(all(1 <= n <= 45 for n in line))
            self.assertEqual(len(set(line)), 5)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed")
    def test_generate_returns_unique_lines(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 10, rng)
        keys = [tuple(l) for l in lines]
        self.assertEqual(len(keys), len(set(keys)))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed")
    def test_get_probabilities_returns_distribution(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertEqual(len(probs), 45)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertTrue(all(p >= 0 for p in probs))

    def test_fallback_without_sklearn(self):
        strategy = GradientBoostStrategy(45, 5, 5)
        strategy._sklearn_available = False
        rng = random.Random(42)
        lines = strategy.generate(self.draws, 3, rng)
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertEqual(len(line), 5)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed")
    def test_deterministic_with_seed(self):
        lines1 = self.strategy.generate(self.draws, 3, random.Random(99))
        lines2 = self.strategy.generate(self.draws, 3, random.Random(99))
        self.assertEqual(lines1, lines2)

    def test_generate_with_few_draws(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws[:5], 3, rng)
        self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
