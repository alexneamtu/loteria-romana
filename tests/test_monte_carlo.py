import random
import unittest

from shared.monte_carlo import generate_synthetic_draws, monte_carlo_validate
from shared.game_config import JOKER_CONFIG


class TestGenerateSyntheticDraws(unittest.TestCase):
    def test_correct_count(self):
        draws = generate_synthetic_draws(JOKER_CONFIG, 100, random.Random(42))
        self.assertEqual(len(draws), 100)

    def test_draws_are_valid(self):
        draws = generate_synthetic_draws(JOKER_CONFIG, 50, random.Random(42))
        for draw in draws:
            self.assertEqual(len(draw), JOKER_CONFIG.numbers_drawn)
            self.assertEqual(sorted(draw), draw)
            self.assertTrue(all(n in JOKER_CONFIG.pool_range for n in draw))
            self.assertEqual(len(set(draw)), len(draw))

    def test_deterministic_with_seed(self):
        d1 = generate_synthetic_draws(JOKER_CONFIG, 20, random.Random(99))
        d2 = generate_synthetic_draws(JOKER_CONFIG, 20, random.Random(99))
        self.assertEqual(d1, d2)


class TestMonteCarloValidate(unittest.TestCase):
    def test_returns_validation_result(self):
        result = monte_carlo_validate(
            config=JOKER_CONFIG,
            strategy_name="frequency",
            strategy_win_rate=0.12,
            num_simulations=10,
            draws_per_simulation=50,
            rng=random.Random(42),
        )
        self.assertIn("synthetic_mean_win_rate", result)
        self.assertIn("strategy_is_plausible", result)
        self.assertIn("num_simulations", result)
        self.assertEqual(result["num_simulations"], 10)

    def test_random_strategy_is_plausible(self):
        result = monte_carlo_validate(
            config=JOKER_CONFIG,
            strategy_name="random",
            strategy_win_rate=0.10,
            num_simulations=20,
            draws_per_simulation=100,
            rng=random.Random(42),
        )
        self.assertTrue(result["strategy_is_plausible"])


if __name__ == "__main__":
    unittest.main()
