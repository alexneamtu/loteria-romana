import random
import unittest

from shared.ensemble_blend import (
    generate_blended_picks,
    _allocate_counts,
    _score_random,
)
from shared.game_config import JOKER_CONFIG, LOTO_649_CONFIG, LOTO_540_CONFIG


class TestAllocateCounts(unittest.TestCase):
    def test_proportional_allocation(self):
        weights = {"a": 0.5, "b": 0.3, "c": 0.2}
        counts = _allocate_counts(weights, 10)
        self.assertEqual(sum(counts.values()), 10)

    def test_zero_weights(self):
        weights = {"a": 0.0, "b": 0.0}
        counts = _allocate_counts(weights, 4)
        self.assertEqual(sum(counts.values()), 4)

    def test_single_strategy(self):
        weights = {"only": 1.0}
        counts = _allocate_counts(weights, 5)
        self.assertEqual(counts["only"], 5)

    def test_rounding(self):
        weights = {"a": 1.0, "b": 1.0, "c": 1.0}
        counts = _allocate_counts(weights, 10)
        self.assertEqual(sum(counts.values()), 10)


class TestScoreRandom(unittest.TestCase):
    def test_returns_integer(self):
        draws = [[1, 2, 3, 4, 5]] * 10
        rng = random.Random(42)
        score = _score_random(JOKER_CONFIG, draws, rng)
        self.assertIsInstance(score, int)
        self.assertGreaterEqual(score, 0)


class TestGenerateBlendedPicks(unittest.TestCase):
    def _make_draws(self, config, count=20):
        rng = random.Random(0)
        draws = []
        pool = list(config.pool_range)
        for _ in range(count):
            draws.append(sorted(rng.sample(pool, config.numbers_drawn)))
        return draws

    def test_joker_returns_correct_count(self):
        draws = self._make_draws(JOKER_CONFIG)
        rng = random.Random(42)
        lines = generate_blended_picks(JOKER_CONFIG, draws, 5, rng)
        self.assertEqual(len(lines), 5)

    def test_loto_649_returns_correct_count(self):
        draws = self._make_draws(LOTO_649_CONFIG)
        rng = random.Random(42)
        lines = generate_blended_picks(LOTO_649_CONFIG, draws, 4, rng)
        self.assertEqual(len(lines), 4)

    def test_loto_540_returns_correct_count(self):
        draws = self._make_draws(LOTO_540_CONFIG)
        rng = random.Random(42)
        lines = generate_blended_picks(LOTO_540_CONFIG, draws, 3, rng)
        self.assertEqual(len(lines), 3)

    def test_picks_are_valid(self):
        draws = self._make_draws(JOKER_CONFIG)
        rng = random.Random(42)
        lines = generate_blended_picks(JOKER_CONFIG, draws, 5, rng)
        for line in lines:
            self.assertEqual(len(line), JOKER_CONFIG.numbers_to_pick)
            self.assertEqual(sorted(line), line)
            self.assertTrue(all(n in JOKER_CONFIG.pool_range for n in line))
            self.assertEqual(len(set(line)), JOKER_CONFIG.numbers_to_pick)

    def test_picks_are_unique(self):
        draws = self._make_draws(JOKER_CONFIG)
        rng = random.Random(42)
        lines = generate_blended_picks(JOKER_CONFIG, draws, 10, rng)
        keys = [tuple(l) for l in lines]
        self.assertEqual(len(keys), len(set(keys)))

    def test_few_draws_falls_back_to_random(self):
        draws = [[1, 2, 3, 4, 5]]
        rng = random.Random(42)
        lines = generate_blended_picks(JOKER_CONFIG, draws, 2, rng)
        self.assertEqual(len(lines), 2)

    def test_deterministic_with_seed(self):
        draws = self._make_draws(JOKER_CONFIG)
        lines1 = generate_blended_picks(JOKER_CONFIG, draws, 3, random.Random(99))
        lines2 = generate_blended_picks(JOKER_CONFIG, draws, 3, random.Random(99))
        self.assertEqual(lines1, lines2)


if __name__ == "__main__":
    unittest.main()
