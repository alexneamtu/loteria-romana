import unittest
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestRLAgent(unittest.TestCase):
    def setUp(self):
        from shared.rl_agent import RLAgent
        self.rng = random.Random(42)
        self.agent = RLAgent(pool_size=10, numbers_to_pick=3)
        self.draws = [
            sorted(random.Random(i).sample(range(1, 11), 3))
            for i in range(50)
        ]

    def test_name(self):
        self.assertEqual(self.agent.name, "rl")

    def test_generate_returns_correct_count(self):
        picks = self.agent.generate(self.draws, 3, self.rng)
        self.assertEqual(len(picks), 3)

    def test_generate_returns_sorted_numbers(self):
        picks = self.agent.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(pick, sorted(pick))

    def test_generate_numbers_in_range(self):
        picks = self.agent.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(len(pick), 3)
            for n in pick:
                self.assertGreaterEqual(n, 1)
                self.assertLessEqual(n, 10)

    def test_generate_unique_picks(self):
        picks = self.agent.generate(self.draws, 5, self.rng)
        keys = [tuple(p) for p in picks]
        self.assertEqual(len(keys), len(set(keys)))

    def test_generate_with_few_draws(self):
        picks = self.agent.generate(self.draws[:3], 2, self.rng)
        self.assertEqual(len(picks), 2)

    def test_get_probabilities_length(self):
        probs = self.agent.get_probabilities(self.draws)
        self.assertEqual(len(probs), 10)

    def test_get_probabilities_sum_to_one(self):
        probs = self.agent.get_probabilities(self.draws)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_get_probabilities_all_positive(self):
        probs = self.agent.get_probabilities(self.draws)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)

    def test_training_episodes_configurable(self):
        from shared.rl_agent import RLAgent
        agent = RLAgent(pool_size=10, numbers_to_pick=3, episodes=5)
        picks = agent.generate(self.draws, 2, self.rng)
        self.assertEqual(len(picks), 2)

    def test_reward_based_on_matches(self):
        from shared.rl_agent import _compute_reward
        self.assertEqual(_compute_reward(0, 3), 0.0)
        r1 = _compute_reward(1, 3)
        r2 = _compute_reward(2, 3)
        r3 = _compute_reward(3, 3)
        self.assertGreater(r2, r1)
        self.assertGreater(r3, r2)


class TestRLAgentFallback(unittest.TestCase):
    def test_fallback_without_torch(self):
        from shared.rl_agent import RLAgent
        agent = RLAgent(pool_size=10, numbers_to_pick=3)
        rng = random.Random(42)
        draws = [sorted(random.Random(i).sample(range(1, 11), 3)) for i in range(5)]
        picks = agent.generate(draws, 2, rng)
        self.assertEqual(len(picks), 2)


class TestRLAgentImport(unittest.TestCase):
    def test_module_importable(self):
        import shared.rl_agent


if __name__ == "__main__":
    unittest.main()
