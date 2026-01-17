import unittest
import random

from loto_649_model.metrics import is_loto_649_prize
from loto_649_model.strategies import build_frequency, generate_random_lines, generate_frequency_lines
from shared.recency import draw_weights, DEFAULT_HALF_LIFE


class TestLoto649Strategies(unittest.TestCase):
    def test_prize_rules(self):
        self.assertTrue(is_loto_649_prize(main_matches=3))
        self.assertTrue(is_loto_649_prize(main_matches=4))
        self.assertFalse(is_loto_649_prize(main_matches=2))
        self.assertFalse(is_loto_649_prize(main_matches=0))

    def test_generate_random_lines_unique(self):
        rng = random.Random(1234)
        lines = generate_random_lines(3, rng=rng)
        self.assertEqual(len(lines), 3)
        self.assertEqual(len({tuple(l) for l in lines}), 3)

    def test_build_frequency_counts(self):
        draws = [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 10, 11, 12, 13],
        ]
        weights = draw_weights(len(draws), DEFAULT_HALF_LIFE)
        freq = build_frequency(draws)
        self.assertAlmostEqual(freq[1], weights[0] + weights[1], places=6)
        self.assertAlmostEqual(freq[2], weights[0] + weights[1], places=6)
        self.assertAlmostEqual(freq[3], weights[0], places=6)
        self.assertEqual(freq[49], 0.0)


if __name__ == "__main__":
    unittest.main()
