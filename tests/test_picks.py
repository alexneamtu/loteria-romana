import random
import unittest

from joker_model.picks import generate_picks


class TestPicks(unittest.TestCase):
    def test_generate_picks_default_count(self):
        draws = [
            ([1, 2, 3, 4, 5], 1),
            ([6, 7, 8, 9, 10], 2),
            ([11, 12, 13, 14, 15], 3),
        ]
        lines = generate_picks(draws, rng=random.Random(0))
        self.assertEqual(len(lines), 2)
        for main, joker in lines:
            self.assertEqual(len(main), 5)
            self.assertTrue(all(1 <= n <= 45 for n in main))
            self.assertTrue(1 <= joker <= 20)

    def test_generate_picks_custom_count(self):
        draws = [
            ([1, 2, 3, 4, 5], 1),
            ([6, 7, 8, 9, 10], 2),
            ([11, 12, 13, 14, 15], 3),
        ]
        lines = generate_picks(draws, count=3, rng=random.Random(1))
        self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
