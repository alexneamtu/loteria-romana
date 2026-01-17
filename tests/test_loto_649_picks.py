import unittest
import random

from loto_649_model.picks import generate_picks


class TestLoto649Picks(unittest.TestCase):
    def test_generate_picks_count(self):
        draws = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ]
        rng = random.Random(0)
        lines = generate_picks(draws, count=2, rng=rng)
        self.assertEqual(len(lines), 2)

    def test_generate_picks_valid_numbers(self):
        draws = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ]
        rng = random.Random(0)
        lines = generate_picks(draws, count=2, rng=rng)
        for line in lines:
            self.assertEqual(len(line), 6)
            for n in line:
                self.assertGreaterEqual(n, 1)
                self.assertLessEqual(n, 49)


if __name__ == "__main__":
    unittest.main()
