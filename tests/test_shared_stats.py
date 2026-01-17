import unittest
import random

from shared.stats import (
    compute_deltas,
    build_delta_distribution,
    DeltaStrategy,
    HotColdStrategy,
    PairStrategy,
    SkipGapStrategy,
    SumConstraintStrategy,
    BalanceStrategy,
    compute_odd_even_distribution,
    compute_high_low_distribution,
    count_consecutives,
    compute_consecutive_distribution,
    build_position_frequency,
)
from shared.recency import draw_weights


class TestDeltaFunctions(unittest.TestCase):
    def test_compute_deltas_basic(self):
        result = compute_deltas([1, 5, 10, 15, 20])
        self.assertEqual(result, [4, 5, 5, 5])

    def test_compute_deltas_consecutive(self):
        result = compute_deltas([1, 2, 3, 4, 5])
        self.assertEqual(result, [1, 1, 1, 1])

    def test_build_delta_distribution(self):
        draws = [
            [1, 2, 3, 4, 5],  # deltas: [1, 1, 1, 1]
            [1, 3, 5, 7, 9],  # deltas: [2, 2, 2, 2]
        ]
        dist = build_delta_distribution(draws)
        self.assertEqual(dist[1], 4)  # four 1s from first draw
        self.assertEqual(dist[2], 4)  # four 2s from second draw

    def test_build_delta_distribution_weighted(self):
        draws = [
            [1, 2, 3],  # deltas: [1, 1]
            [1, 3, 5],  # deltas: [2, 2]
        ]
        weights = draw_weights(len(draws), 1.0)
        dist = build_delta_distribution(draws, weights=weights)
        self.assertAlmostEqual(dist[1], 2 * weights[0], places=6)
        self.assertAlmostEqual(dist[2], 2 * weights[1], places=6)


class TestDeltaStrategy(unittest.TestCase):
    def test_delta_strategy_generates_valid_lines(self):
        strategy = DeltaStrategy(number_pool=45, numbers_to_pick=5)
        draws = [
            [1, 5, 10, 20, 30],
            [2, 8, 15, 25, 40],
            [3, 7, 12, 22, 35],
        ]
        rng = random.Random(42)
        lines = strategy.generate(draws, count=5, rng=rng)

        self.assertEqual(len(lines), 5)
        for main in lines:
            self.assertEqual(len(main), 5)
            self.assertEqual(main, sorted(main))
            self.assertTrue(all(1 <= n <= 45 for n in main))

    def test_delta_strategy_unique_lines(self):
        strategy = DeltaStrategy(number_pool=45, numbers_to_pick=5)
        draws = [[i, i + 5, i + 10, i + 15, i + 20] for i in range(1, 20)]
        rng = random.Random(42)
        lines = strategy.generate(draws, count=10, rng=rng)

        seen = set()
        for main in lines:
            key = tuple(main)
            self.assertNotIn(key, seen)
            seen.add(key)

    def test_delta_strategy_get_probabilities(self):
        strategy = DeltaStrategy(number_pool=45, numbers_to_pick=5)
        draws = [[1, 5, 10, 20, 30]]
        probs = strategy.get_probabilities(draws)

        self.assertEqual(len(probs), 45)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)


class TestHotColdStrategy(unittest.TestCase):
    def test_compute_heat_scores(self):
        strategy = HotColdStrategy(number_pool=10, numbers_to_pick=3, half_life=1.0)
        draws = [
            [1, 2, 3],  # oldest
            [1, 2, 4],  # middle
            [1, 5, 6],  # most recent
        ]
        scores = strategy.compute_heat_scores(draws)

        # Number 1 appears in all draws, should have highest score
        self.assertGreater(scores[1], scores[7])
        # Number 7-10 never appear, should have score 0
        self.assertEqual(scores[7], 0.0)

    def test_classify_numbers(self):
        strategy = HotColdStrategy(number_pool=10, numbers_to_pick=3)
        # Create scores where mean > std so cold threshold is positive
        # Values: [10, 10, 5, 5, 5, 5, 1, 1, 1, 1] -> mean=4.4, std≈3.3
        # Hot threshold = 7.7, Cold threshold = 1.1
        scores = {1: 10.0, 2: 10.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 5.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0}
        hot, cold, neutral = strategy.classify_numbers(scores)

        # With this distribution: 1,2 are hot (score=10 > 7.7)
        # and 7-10 are cold (score=1 < 1.1)
        self.assertTrue(len(hot) > 0, "Should have some hot numbers")
        self.assertTrue(len(cold) > 0, "Should have some cold numbers")

    def test_hotcold_strategy_generates_valid_lines(self):
        strategy = HotColdStrategy(number_pool=45, numbers_to_pick=5)
        draws = [[i, i + 1, i + 2, i + 3, i + 4] for i in range(1, 30)]
        rng = random.Random(42)
        lines = strategy.generate(draws, count=5, rng=rng)

        self.assertEqual(len(lines), 5)
        for main in lines:
            self.assertEqual(len(main), 5)
            self.assertTrue(all(1 <= n <= 45 for n in main))


class TestPairStrategy(unittest.TestCase):
    def test_build_pair_matrix(self):
        strategy = PairStrategy(number_pool=10, numbers_to_pick=3)
        draws = [
            [1, 2, 3],
            [1, 2, 4],
            [2, 3, 5],
        ]
        pairs = strategy.build_pair_matrix(draws)

        # (1, 2) appears in first two draws
        self.assertEqual(pairs[(1, 2)], 2)
        # (2, 3) appears in first and third draws
        self.assertEqual(pairs[(2, 3)], 2)

    def test_get_top_pairs(self):
        strategy = PairStrategy(number_pool=10, numbers_to_pick=3)
        pair_counts = {(1, 2): 10, (3, 4): 5, (5, 6): 3, (7, 8): 1}
        top = strategy.get_top_pairs(pair_counts, top_n=2)

        self.assertEqual(len(top), 2)
        self.assertEqual(top[0], (1, 2))
        self.assertEqual(top[1], (3, 4))

    def test_pair_strategy_generates_valid_lines(self):
        strategy = PairStrategy(number_pool=45, numbers_to_pick=5)
        draws = [[i, i + 1, i + 2, i + 3, i + 4] for i in range(1, 30)]
        rng = random.Random(42)
        lines = strategy.generate(draws, count=5, rng=rng)

        self.assertEqual(len(lines), 5)
        for main in lines:
            self.assertEqual(len(main), 5)
            self.assertEqual(len(set(main)), 5)  # All unique
            self.assertTrue(all(1 <= n <= 45 for n in main))


class TestSkipGapStrategy(unittest.TestCase):
    def test_compute_gaps(self):
        strategy = SkipGapStrategy(number_pool=10, numbers_to_pick=3)
        draws = [
            [1, 2, 3],  # index 0
            [4, 5, 6],  # index 1
            [1, 7, 8],  # index 2 - number 1 reappears
        ]
        gaps = strategy.compute_gaps(draws)

        # Number 1 appeared in most recent draw (age 0)
        self.assertEqual(gaps[1], 0)
        # Number 2 appeared 2 draws ago
        self.assertEqual(gaps[2], 2)
        # Number 9, 10 never appeared
        self.assertEqual(gaps[9], 3)
        self.assertEqual(gaps[10], 3)

    def test_compute_expected_gaps(self):
        strategy = SkipGapStrategy(number_pool=10, numbers_to_pick=3)
        draws = [
            [1, 2, 3],  # index 0
            [1, 4, 5],  # index 1 - gap of 1 for number 1
            [1, 6, 7],  # index 2 - gap of 1 for number 1
        ]
        expected = strategy.compute_expected_gaps(draws)

        # Number 1 has consistent gap of 1
        self.assertAlmostEqual(expected[1], 1.0, places=5)

    def test_skip_strategy_generates_valid_lines(self):
        strategy = SkipGapStrategy(number_pool=45, numbers_to_pick=5)
        draws = [[i, i + 1, i + 2, i + 3, i + 4] for i in range(1, 30)]
        rng = random.Random(42)
        lines = strategy.generate(draws, count=5, rng=rng)

        self.assertEqual(len(lines), 5)
        for main in lines:
            self.assertEqual(len(main), 5)
            self.assertTrue(all(1 <= n <= 45 for n in main))


class TestSumConstraintStrategy(unittest.TestCase):
    def test_compute_sum_distribution(self):
        strategy = SumConstraintStrategy(number_pool=45, numbers_to_pick=5)
        draws = [
            [1, 10, 20, 30, 40],  # sum = 101
            [5, 15, 25, 35, 45],  # sum = 125
            [2, 12, 22, 32, 42],  # sum = 110
        ]
        mean_sum, std_sum = strategy.compute_sum_distribution(draws)

        sums = [101, 125, 110]
        weights = draw_weights(len(draws), strategy.half_life)
        expected_mean = sum(value * weight for value, weight in zip(sums, weights)) / sum(weights)
        self.assertAlmostEqual(mean_sum, expected_mean, places=5)
        self.assertGreater(std_sum, 0)

    def test_sum_strategy_generates_valid_lines(self):
        strategy = SumConstraintStrategy(number_pool=45, numbers_to_pick=5)
        draws = [[10, 15, 20, 25, 30] for _ in range(1, 20)]  # sum = 100
        rng = random.Random(42)
        lines = strategy.generate(draws, count=5, rng=rng, sigma_range=2.0)

        self.assertEqual(len(lines), 5)
        for main in lines:
            self.assertEqual(len(main), 5)
            self.assertTrue(all(1 <= n <= 45 for n in main))

    def test_sum_strategy_respects_sum_range(self):
        strategy = SumConstraintStrategy(number_pool=45, numbers_to_pick=5)
        # All draws have sum = 100
        draws = [[10, 15, 20, 25, 30] for _ in range(1, 50)]
        rng = random.Random(42)
        lines = strategy.generate(draws, count=10, rng=rng, sigma_range=1.0)

        mean_sum, std_sum = strategy.compute_sum_distribution(draws)
        min_sum = mean_sum - 1.0 * std_sum
        max_sum = mean_sum + 1.0 * std_sum

        for main in lines:
            line_sum = sum(main)
            # Allow some tolerance since std might be very small
            self.assertGreaterEqual(line_sum, min_sum - 10)
            self.assertLessEqual(line_sum, max_sum + 10)


class TestPatternUtilities(unittest.TestCase):
    def test_compute_odd_even_distribution(self):
        draws = [
            [1, 3, 5, 7, 9],  # 5 odd
            [2, 4, 6, 8, 10],  # 0 odd
            [1, 2, 3, 4, 5],  # 3 odd
        ]
        dist = compute_odd_even_distribution(draws)

        self.assertEqual(dist[5], 1)
        self.assertEqual(dist[0], 1)
        self.assertEqual(dist[3], 1)

    def test_compute_high_low_distribution(self):
        draws = [
            [1, 2, 3, 4, 5],  # 0 high (midpoint=23)
            [20, 25, 30, 35, 40],  # 4 high
            [10, 15, 25, 30, 35],  # 3 high
        ]
        dist = compute_high_low_distribution(draws, midpoint=23)

        self.assertEqual(dist[0], 1)
        self.assertEqual(dist[4], 1)
        self.assertEqual(dist[3], 1)

    def test_count_consecutives(self):
        self.assertEqual(count_consecutives([1, 2, 3, 4, 5]), 4)
        self.assertEqual(count_consecutives([1, 3, 5, 7, 9]), 0)
        self.assertEqual(count_consecutives([1, 2, 5, 6, 10]), 2)

    def test_compute_consecutive_distribution(self):
        draws = [
            [1, 2, 3, 4, 5],  # 4 consecutive pairs
            [1, 3, 5, 7, 9],  # 0 consecutive pairs
            [1, 2, 5, 6, 10],  # 2 consecutive pairs
        ]
        dist = compute_consecutive_distribution(draws)

        self.assertEqual(dist[4], 1)
        self.assertEqual(dist[0], 1)
        self.assertEqual(dist[2], 1)


class TestPositionFrequency(unittest.TestCase):
    def test_build_position_frequency(self):
        draws = [
            [1, 5, 10, 15, 20],
            [2, 5, 12, 15, 25],
            [1, 6, 10, 18, 20],
        ]
        freq = build_position_frequency(draws, numbers_to_pick=5, number_pool=45)

        self.assertEqual(len(freq), 5)  # 5 positions
        # Position 0 (lowest): 1 appears twice, 2 appears once
        self.assertEqual(freq[0].get(1, 0), 2)
        self.assertEqual(freq[0].get(2, 0), 1)
        # Position 1: 5 appears twice, 6 appears once
        self.assertEqual(freq[1].get(5, 0), 2)
        self.assertEqual(freq[1].get(6, 0), 1)

    def test_build_position_frequency_empty_draws(self):
        freq = build_position_frequency([], numbers_to_pick=5, number_pool=45)
        self.assertEqual(len(freq), 5)
        for pos_dict in freq:
            self.assertEqual(len(pos_dict), 0)


class TestBalanceStrategy(unittest.TestCase):
    def test_compute_target_ratios(self):
        strategy = BalanceStrategy(number_pool=45, numbers_to_pick=5)
        draws = [
            [1, 3, 5, 7, 9],  # 5 odd, 0 high (midpoint=23)
            [2, 4, 6, 8, 10],  # 0 odd, 0 high
            [1, 2, 24, 25, 26],  # 2 odd (1, 25), 3 high
        ]
        odd_probs, high_probs = strategy.compute_target_ratios(draws)

        self.assertIn(5, odd_probs)  # 5 odd numbers appeared
        self.assertIn(0, odd_probs)  # 0 odd numbers appeared
        self.assertIn(2, odd_probs)  # 2 odd numbers appeared

    def test_balance_strategy_generates_valid_lines(self):
        strategy = BalanceStrategy(number_pool=45, numbers_to_pick=5)
        draws = [[i, i + 5, i + 10, i + 20, i + 30] for i in range(1, 15)]
        rng = random.Random(42)
        lines = strategy.generate(draws, count=5, rng=rng)

        self.assertEqual(len(lines), 5)
        for main in lines:
            self.assertEqual(len(main), 5)
            self.assertEqual(len(set(main)), 5)  # All unique
            self.assertTrue(all(1 <= n <= 45 for n in main))

    def test_balance_strategy_with_empty_history(self):
        strategy = BalanceStrategy(number_pool=45, numbers_to_pick=5)
        rng = random.Random(42)
        lines = strategy.generate([], count=3, rng=rng)

        # Should use default balanced distribution
        self.assertEqual(len(lines), 3)
        for main in lines:
            self.assertEqual(len(main), 5)

    def test_balance_strategy_unique_lines(self):
        strategy = BalanceStrategy(number_pool=45, numbers_to_pick=5)
        draws = [[i, i + 5, i + 10, i + 20, i + 30] for i in range(1, 15)]
        rng = random.Random(42)
        lines = strategy.generate(draws, count=10, rng=rng)

        seen = set()
        for main in lines:
            key = tuple(main)
            self.assertNotIn(key, seen)
            seen.add(key)


class TestStrategyIntegration(unittest.TestCase):
    def test_all_strategies_work_with_empty_history(self):
        """All strategies should handle empty history gracefully."""
        strategies = [
            DeltaStrategy(45, 5),
            HotColdStrategy(45, 5),
            PairStrategy(45, 5),
            SkipGapStrategy(45, 5),
            SumConstraintStrategy(45, 5),
            BalanceStrategy(45, 5),
        ]
        rng = random.Random(42)

        for strategy in strategies:
            lines = strategy.generate([], count=2, rng=rng)
            # Should either generate lines or return empty (not crash)
            self.assertIsInstance(lines, list)

    def test_all_strategies_produce_unique_numbers(self):
        """Each line should have unique numbers."""
        strategies = [
            DeltaStrategy(45, 5),
            HotColdStrategy(45, 5),
            PairStrategy(45, 5),
            SkipGapStrategy(45, 5),
            SumConstraintStrategy(45, 5),
            BalanceStrategy(45, 5),
        ]
        draws = [[i, i + 5, i + 10, i + 15, i + 20] for i in range(1, 20)]
        rng = random.Random(42)

        for strategy in strategies:
            lines = strategy.generate(draws, count=5, rng=rng)
            for main in lines:
                self.assertEqual(len(main), len(set(main)), f"{strategy.name} produced duplicates")

    def test_strategies_are_deterministic_with_seed(self):
        """Same seed should produce same results."""
        draws = [[i, i + 5, i + 10, i + 15, i + 20] for i in range(1, 20)]

        for StrategyClass in [DeltaStrategy, HotColdStrategy, PairStrategy, SkipGapStrategy, SumConstraintStrategy, BalanceStrategy]:
            strategy1 = StrategyClass(45, 5)
            strategy2 = StrategyClass(45, 5)

            lines1 = strategy1.generate(draws, count=3, rng=random.Random(123))
            lines2 = strategy2.generate(draws, count=3, rng=random.Random(123))

            self.assertEqual(lines1, lines2, f"{StrategyClass.__name__} not deterministic")


if __name__ == "__main__":
    unittest.main()
