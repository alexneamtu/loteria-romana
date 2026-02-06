import unittest

from shared.holdout import split_holdout, HoldoutSplit


class TestSplitHoldout(unittest.TestCase):
    def test_split_reserves_correct_count(self):
        draws = [[i, i+1, i+2, i+3, i+4] for i in range(1, 201)]
        result = split_holdout(draws, holdout_size=100)
        self.assertEqual(len(result.train), 100)
        self.assertEqual(len(result.holdout), 100)

    def test_holdout_is_most_recent(self):
        draws = [[i] for i in range(10)]
        result = split_holdout(draws, holdout_size=3)
        self.assertEqual(result.holdout, [[7], [8], [9]])
        self.assertEqual(result.train, [[i] for i in range(7)])

    def test_holdout_fraction(self):
        draws = [[i] for i in range(100)]
        result = split_holdout(draws, holdout_fraction=0.2)
        self.assertEqual(len(result.holdout), 20)
        self.assertEqual(len(result.train), 80)

    def test_holdout_size_takes_precedence(self):
        draws = [[i] for i in range(100)]
        result = split_holdout(draws, holdout_size=10, holdout_fraction=0.5)
        self.assertEqual(len(result.holdout), 10)

    def test_insufficient_data_returns_all_train(self):
        draws = [[i] for i in range(5)]
        result = split_holdout(draws, holdout_size=100)
        self.assertEqual(len(result.train), 5)
        self.assertEqual(len(result.holdout), 0)

    def test_dates_split_with_draws(self):
        draws = [[i] for i in range(10)]
        dates = [f"2024-01-{i+1:02d}" for i in range(10)]
        result = split_holdout(draws, holdout_size=3, dates=dates)
        self.assertEqual(len(result.train_dates), 7)
        self.assertEqual(len(result.holdout_dates), 3)


if __name__ == "__main__":
    unittest.main()
