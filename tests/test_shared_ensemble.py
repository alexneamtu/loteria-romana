import unittest
import random

from shared.ensemble import EnsembleVoter, StrategySelector
from shared.stats import DeltaStrategy, HotColdStrategy, PairStrategy


class TestEnsembleVoter(unittest.TestCase):
    def setUp(self):
        self.strategies = [
            DeltaStrategy(45, 5),
            HotColdStrategy(45, 5),
            PairStrategy(45, 5),
        ]
        # Generate valid draws within number pool (1-45)
        rng = random.Random(42)
        self.draws = [
            sorted(rng.sample(range(1, 46), 5))
            for _ in range(50)
        ]

    def test_init_equal_weights(self):
        ensemble = EnsembleVoter(self.strategies, number_pool=45, numbers_to_pick=5)

        # All strategies should have equal weight
        expected_weight = 1.0 / len(self.strategies)
        for strategy in self.strategies:
            self.assertAlmostEqual(
                ensemble.strategy_weights[strategy.name],
                expected_weight,
                places=5,
            )

    def test_init_custom_weights(self):
        weights = {"delta": 0.5, "hotcold": 0.3, "pair": 0.2}
        ensemble = EnsembleVoter(
            self.strategies,
            weights=weights,
            number_pool=45,
            numbers_to_pick=5,
        )

        self.assertAlmostEqual(ensemble.strategy_weights["delta"], 0.5, places=5)
        self.assertAlmostEqual(ensemble.strategy_weights["hotcold"], 0.3, places=5)
        self.assertAlmostEqual(ensemble.strategy_weights["pair"], 0.2, places=5)

    def test_update_weights(self):
        ensemble = EnsembleVoter(self.strategies, number_pool=45, numbers_to_pick=5)

        # Update with backtest results
        backtest_results = {"delta": 0.1, "hotcold": 0.3, "pair": 0.2}
        ensemble.update_weights(backtest_results)

        # Weights should be normalized
        total = sum(ensemble.strategy_weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)

        # Higher performance should get higher weight
        self.assertGreater(
            ensemble.strategy_weights["hotcold"],
            ensemble.strategy_weights["delta"],
        )

    def test_combine_probabilities(self):
        ensemble = EnsembleVoter(self.strategies, number_pool=45, numbers_to_pick=5)

        probs = ensemble.combine_probabilities(self.draws)

        self.assertEqual(len(probs), 45)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)

    def test_combine_probabilities_empty_draws(self):
        ensemble = EnsembleVoter(self.strategies, number_pool=45, numbers_to_pick=5)

        probs = ensemble.combine_probabilities([])

        self.assertEqual(len(probs), 45)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)

    def test_generate(self):
        ensemble = EnsembleVoter(self.strategies, number_pool=45, numbers_to_pick=5)
        rng = random.Random(42)

        lines = ensemble.generate(self.draws, count=5, rng=rng)

        self.assertEqual(len(lines), 5)
        for main in lines:
            self.assertEqual(len(main), 5)
            self.assertEqual(len(set(main)), 5)  # All unique
            self.assertTrue(all(1 <= n <= 45 for n in main))

    def test_generate_unique_lines(self):
        ensemble = EnsembleVoter(self.strategies, number_pool=45, numbers_to_pick=5)
        rng = random.Random(42)

        lines = ensemble.generate(self.draws, count=10, rng=rng)

        seen = set()
        for main in lines:
            key = tuple(main)
            self.assertNotIn(key, seen)
            seen.add(key)

    def test_generate_diverse(self):
        ensemble = EnsembleVoter(self.strategies, number_pool=45, numbers_to_pick=5)
        rng = random.Random(42)

        lines = ensemble.generate_diverse(self.draws, count=5, rng=rng)

        self.assertEqual(len(lines), 5)
        for main in lines:
            self.assertEqual(len(main), 5)
            self.assertTrue(all(1 <= n <= 45 for n in main))

    def test_get_probabilities_protocol(self):
        ensemble = EnsembleVoter(self.strategies, number_pool=45, numbers_to_pick=5)

        probs = ensemble.get_probabilities(self.draws)

        self.assertEqual(len(probs), 45)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)

    def test_ensemble_deterministic_with_seed(self):
        ensemble1 = EnsembleVoter(self.strategies, number_pool=45, numbers_to_pick=5)
        ensemble2 = EnsembleVoter(self.strategies, number_pool=45, numbers_to_pick=5)

        lines1 = ensemble1.generate(self.draws, count=3, rng=random.Random(123))
        lines2 = ensemble2.generate(self.draws, count=3, rng=random.Random(123))

        self.assertEqual(lines1, lines2)


class TestStrategySelector(unittest.TestCase):
    def setUp(self):
        self.strategies = [
            DeltaStrategy(45, 5),
            HotColdStrategy(45, 5),
            PairStrategy(45, 5),
        ]
        # Generate valid draws within number pool (1-45)
        rng = random.Random(42)
        self.draws = [
            sorted(rng.sample(range(1, 46), 5))
            for _ in range(100)
        ]

    def test_init(self):
        selector = StrategySelector(
            self.strategies,
            lookback=50,
            number_pool=45,
            numbers_to_pick=5,
        )

        self.assertEqual(selector.lookback, 50)
        self.assertEqual(len(selector.strategies), 3)

    def test_select_best(self):
        selector = StrategySelector(
            self.strategies,
            lookback=50,
            number_pool=45,
            numbers_to_pick=5,
        )
        rng = random.Random(42)

        best = selector.select_best(self.draws, rng=rng)

        # Should return one of the strategies
        self.assertIn(best, self.strategies)

    def test_generate(self):
        selector = StrategySelector(
            self.strategies,
            lookback=50,
            number_pool=45,
            numbers_to_pick=5,
        )
        rng = random.Random(42)

        lines = selector.generate(self.draws, count=5, rng=rng)

        self.assertEqual(len(lines), 5)
        for main in lines:
            self.assertEqual(len(main), 5)
            self.assertTrue(all(1 <= n <= 45 for n in main))

    def test_get_probabilities(self):
        selector = StrategySelector(
            self.strategies,
            lookback=50,
            number_pool=45,
            numbers_to_pick=5,
        )

        probs = selector.get_probabilities(self.draws)

        self.assertEqual(len(probs), 45)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)

    def test_evaluate_strategy(self):
        selector = StrategySelector(
            self.strategies,
            lookback=50,
            number_pool=45,
            numbers_to_pick=5,
        )
        rng = random.Random(42)

        score = selector.evaluate_strategy(
            self.strategies[0],
            self.draws,
            n_eval=10,
            rng=rng,
        )

        # Score should be average matches per ticket
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 5.0)  # Can't match more than 5 numbers


class TestEnsembleIntegration(unittest.TestCase):
    def test_ensemble_with_backtest_weights(self):
        """Test ensemble can update weights from backtest-like results."""
        strategies = [
            DeltaStrategy(45, 5),
            HotColdStrategy(45, 5),
            PairStrategy(45, 5),
        ]
        # Generate valid draws within number pool (1-45)
        rng_setup = random.Random(42)
        draws = [
            sorted(rng_setup.sample(range(1, 46), 5))
            for _ in range(50)
        ]

        ensemble = EnsembleVoter(strategies, number_pool=45, numbers_to_pick=5)

        # Simulate backtest results
        backtest_results = {
            "delta": 0.05,
            "hotcold": 0.08,
            "pairs": 0.03,
        }
        ensemble.update_weights(backtest_results)

        # hotcold should have highest weight
        self.assertGreater(
            ensemble.strategy_weights["hotcold"],
            ensemble.strategy_weights["delta"],
        )
        self.assertGreater(
            ensemble.strategy_weights["hotcold"],
            ensemble.strategy_weights["pairs"],
        )

        # Should still generate valid lines
        rng = random.Random(42)
        lines = ensemble.generate(draws, count=5, rng=rng)
        self.assertEqual(len(lines), 5)


if __name__ == "__main__":
    unittest.main()
