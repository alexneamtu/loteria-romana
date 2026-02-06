import unittest
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestNormalizingFlows(unittest.TestCase):
    def setUp(self):
        from shared.normalizing_flows import NormalizingFlowStrategy
        self.rng = random.Random(42)
        self.strategy = NormalizingFlowStrategy(pool_size=10, numbers_to_pick=3)
        self.draws = [
            sorted(random.Random(i).sample(range(1, 11), 3))
            for i in range(50)
        ]

    def test_name(self):
        self.assertEqual(self.strategy.name, "normalizing_flow")

    def test_generate_returns_correct_count(self):
        picks = self.strategy.generate(self.draws, 3, self.rng)
        self.assertEqual(len(picks), 3)

    def test_generate_returns_sorted_numbers(self):
        picks = self.strategy.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(pick, sorted(pick))

    def test_generate_numbers_in_range(self):
        picks = self.strategy.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(len(pick), 3)
            for n in pick:
                self.assertGreaterEqual(n, 1)
                self.assertLessEqual(n, 10)

    def test_generate_with_few_draws(self):
        picks = self.strategy.generate(self.draws[:3], 2, self.rng)
        self.assertEqual(len(picks), 2)

    def test_get_probabilities_length(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertEqual(len(probs), 10)

    def test_get_probabilities_sum_to_one(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_get_probabilities_all_positive(self):
        probs = self.strategy.get_probabilities(self.draws)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)

    def test_generate_unique_picks(self):
        picks = self.strategy.generate(self.draws, 5, self.rng)
        keys = [tuple(p) for p in picks]
        self.assertEqual(len(keys), len(set(keys)))


class TestNormalizingFlowsImport(unittest.TestCase):
    def test_module_importable(self):
        import shared.normalizing_flows


if __name__ == "__main__":
    unittest.main()
