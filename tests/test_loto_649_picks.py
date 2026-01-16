import unittest
import random

from loto_649_model.picks import generate_picks


class TestLoto649Picks(unittest.TestCase):
    def test_generate_picks_count(self):
        draws = [
            ([1, 2, 3, 4, 5, 6], 1234567),
            ([7, 8, 9, 10, 11, 12], 7654321),
        ]
        rng = random.Random(0)
        lines = generate_picks(draws, count=2, rng=rng)
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
