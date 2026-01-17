import unittest
import random
from unittest.mock import patch

from loto_649_model.backtest import pick_best_strategy


class TestLoto649Backtest(unittest.TestCase):
    def test_pick_best_strategy(self):
        draws = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
        ]
        best = pick_best_strategy(draws, rng=random.Random(0))
        self.assertIn(best, {"random", "frequency", "neural"})

    def test_frequency_uses_rolling_history(self):
        draws = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [1, 2, 13, 14, 15, 16],
        ]
        with patch("loto_649_model.backtest.build_frequency") as build_frequency:
            build_frequency.side_effect = lambda _: {n: 0 for n in range(1, 50)}
            pick_best_strategy(draws, rng=random.Random(0))
            self.assertGreaterEqual(build_frequency.call_count, 1)
            self.assertEqual(build_frequency.call_args_list[0][0][0], [])
            self.assertEqual(build_frequency.call_args_list[1][0][0], draws[:1])


if __name__ == "__main__":
    unittest.main()
