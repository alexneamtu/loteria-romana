import unittest
import random

from joker_model.backtest import pick_best_strategy


class TestBacktest(unittest.TestCase):
    def test_pick_best_strategy(self):
        draws = [
            ([1, 2, 3, 4, 5], 1),
            ([6, 7, 8, 9, 10], 2),
            ([11, 12, 13, 14, 15], 3),
        ]
        best = pick_best_strategy(draws, rng=random.Random(0))
        self.assertIn(best, {"random", "frequency", "neural"})


if __name__ == "__main__":
    unittest.main()
