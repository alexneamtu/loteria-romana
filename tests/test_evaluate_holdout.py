import random
import unittest

from shared.holdout_evaluator import (
    evaluate_strategy_on_holdout,
    evaluate_all_strategies,
    HoldoutResult,
)
from shared.game_config import JOKER_CONFIG, LOTO_649_CONFIG, LOTO_540_CONFIG


class TestHoldoutResult(unittest.TestCase):
    def test_result_fields(self):
        result = HoldoutResult(
            strategy_name="test",
            game_name="joker",
            holdout_size=50,
            wins=5,
            win_rate=0.1,
            matches_distribution={2: 3, 3: 2},
        )
        self.assertEqual(result.strategy_name, "test")
        self.assertEqual(result.game_name, "joker")
        self.assertEqual(result.holdout_size, 50)
        self.assertEqual(result.win_rate, 0.1)


class TestEvaluateStrategyOnHoldout(unittest.TestCase):
    def _make_draws(self, config, count=100):
        rng = random.Random(0)
        pool = list(config.pool_range)
        return [sorted(rng.sample(pool, config.numbers_drawn)) for _ in range(count)]

    def test_evaluate_random_on_holdout(self):
        draws = self._make_draws(JOKER_CONFIG, count=100)
        result = evaluate_strategy_on_holdout(
            config=JOKER_CONFIG,
            strategy_name="random",
            train_draws=draws[:80],
            holdout_draws=draws[80:],
            rng=random.Random(42),
        )
        self.assertEqual(result.strategy_name, "random")
        self.assertEqual(result.game_name, "joker")
        self.assertEqual(result.holdout_size, 20)
        self.assertIsInstance(result.win_rate, float)

    def test_evaluate_frequency_on_holdout(self):
        draws = self._make_draws(JOKER_CONFIG, count=100)
        result = evaluate_strategy_on_holdout(
            config=JOKER_CONFIG,
            strategy_name="frequency",
            train_draws=draws[:80],
            holdout_draws=draws[80:],
            rng=random.Random(42),
        )
        self.assertEqual(result.strategy_name, "frequency")
        self.assertGreaterEqual(result.win_rate, 0.0)


class TestEvaluateAllStrategies(unittest.TestCase):
    def _make_draws(self, config, count=80):
        rng = random.Random(0)
        pool = list(config.pool_range)
        return [sorted(rng.sample(pool, config.numbers_drawn)) for _ in range(count)]

    def test_evaluate_all_returns_results_for_each_strategy(self):
        draws = self._make_draws(JOKER_CONFIG, count=80)
        results = evaluate_all_strategies(
            config=JOKER_CONFIG,
            train_draws=draws[:60],
            holdout_draws=draws[60:],
            rng=random.Random(42),
        )
        self.assertGreater(len(results), 0)
        names = [r.strategy_name for r in results]
        self.assertIn("random", names)
        self.assertIn("frequency", names)

    def test_evaluate_all_for_multiple_games(self):
        for config in [JOKER_CONFIG, LOTO_649_CONFIG, LOTO_540_CONFIG]:
            draws = self._make_draws(config, count=80)
            results = evaluate_all_strategies(
                config=config,
                train_draws=draws[:60],
                holdout_draws=draws[60:],
                rng=random.Random(42),
            )
            self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
