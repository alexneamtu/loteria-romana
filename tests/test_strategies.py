import unittest
import random

from joker_model.metrics import is_joker_prize
from joker_model.strategies import build_frequency, generate_random_lines, generate_frequency_lines


class TestStrategies(unittest.TestCase):
    def test_joker_prize_rules(self):
        self.assertTrue(is_joker_prize(main_matches=5, joker_match=True))
        self.assertTrue(is_joker_prize(main_matches=3, joker_match=False))
        self.assertTrue(is_joker_prize(main_matches=1, joker_match=True))
        self.assertFalse(is_joker_prize(main_matches=2, joker_match=False))

    def test_generate_random_lines_unique(self):
        rng = random.Random(1234)
        lines = generate_random_lines(3, rng=rng)
        self.assertEqual(len(lines), 3)
        self.assertEqual(len({tuple(l[0]) + (l[1],) for l in lines}), 3)

    def test_build_frequency_counts(self):
        draws = [
            ([1, 2, 3, 4, 5], 1),
            ([1, 2, 10, 11, 12], 2),
        ]
        freq = build_frequency(draws)
        self.assertEqual(freq[1], 2)
        self.assertEqual(freq[2], 2)
        self.assertEqual(freq[3], 1)
        self.assertEqual(freq[45], 0)

    def test_generate_frequency_lines_unique(self):
        freq = {n: 1 for n in range(1, 46)}
        rng = random.Random(42)
        lines = generate_frequency_lines(2, freq, rng=rng)
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
