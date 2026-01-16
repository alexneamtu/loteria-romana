import unittest
import random

from loto_649_model.metrics import is_loto_649_prize
from loto_649_model.strategies import build_frequency, generate_random_lines, generate_frequency_lines


class TestLoto649Strategies(unittest.TestCase):
    def test_prize_rules(self):
        self.assertTrue(is_loto_649_prize(main_matches=3, noroc_match=False))
        self.assertTrue(is_loto_649_prize(main_matches=0, noroc_match=True))
        self.assertFalse(is_loto_649_prize(main_matches=2, noroc_match=False))

    def test_prize_rules_without_noroc(self):
        self.assertFalse(is_loto_649_prize(main_matches=0, noroc_match=True, include_noroc=False))

    def test_generate_random_lines_unique(self):
        rng = random.Random(1234)
        lines = generate_random_lines(3, rng=rng)
        self.assertEqual(len(lines), 3)
        self.assertEqual(len({tuple(l[0]) + (l[1],) for l in lines}), 3)

    def test_build_frequency_counts(self):
        draws = [
            ([1, 2, 3, 4, 5, 6], 1234567),
            ([1, 2, 10, 11, 12, 13], 7654321),
        ]
        freq = build_frequency(draws)
        self.assertEqual(freq[1], 2)
        self.assertEqual(freq[2], 2)
        self.assertEqual(freq[3], 1)
        self.assertEqual(freq[49], 0)


if __name__ == "__main__":
    unittest.main()
