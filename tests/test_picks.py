import random
import unittest
from unittest.mock import patch

from joker_model.picks import generate_picks
from shared.recency import DEFAULT_HALF_LIFE


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

    def test_generate_picks_passes_rng_to_strategy(self):
        draws = [
            ([1, 2, 3, 4, 5], 1),
            ([6, 7, 8, 9, 10], 2),
            ([11, 12, 13, 14, 15], 3),
        ]
        rng = random.Random(0)
        with patch("joker_model.picks.pick_best_strategy") as pick_best:
            pick_best.return_value = "random"
            generate_picks(draws, rng=rng)
            pick_best.assert_called_once_with(draws, rng=rng, half_life=DEFAULT_HALF_LIFE)


if __name__ == "__main__":
    unittest.main()
