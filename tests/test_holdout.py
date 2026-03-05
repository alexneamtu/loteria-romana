import unittest

from shared.holdout import split_holdout, HoldoutSplit, TemporalSplit, temporal_holdout_split


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


def _make_draws(count):
    """Create anonymous draw objects with date and main_numbers attributes."""
    return [
        type("Draw", (), {
            "date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            "main_numbers": [i],
        })()
        for i in range(count)
    ]


class TestTemporalHoldoutSplit(unittest.TestCase):

    def test_basic_split(self):
        draws = _make_draws(200)
        result = temporal_holdout_split(draws, holdout_size=50)
        self.assertIsInstance(result, TemporalSplit)
        self.assertEqual(len(result.train), 150)
        self.assertEqual(len(result.holdout), 50)
        self.assertEqual(result.holdout_size, 50)

    def test_holdout_is_most_recent(self):
        draws = _make_draws(200)
        result = temporal_holdout_split(draws, holdout_size=50)
        self.assertEqual(result.holdout, draws[-50:])
        self.assertEqual(result.train, draws[:150])

    def test_default_holdout_100(self):
        draws = _make_draws(500)
        result = temporal_holdout_split(draws)
        self.assertEqual(len(result.train), 400)
        self.assertEqual(len(result.holdout), 100)
        self.assertEqual(result.holdout_size, 100)

    def test_holdout_larger_than_data(self):
        draws = _make_draws(50)
        result = temporal_holdout_split(draws, holdout_size=100)
        # Cap at 20% of 50 = 10
        self.assertEqual(len(result.holdout), 10)
        self.assertEqual(len(result.train), 40)
        self.assertEqual(result.holdout_size, 10)

    def test_split_date_boundary(self):
        draws = _make_draws(200)
        result = temporal_holdout_split(draws, holdout_size=50)
        expected_date = str(draws[150].date)
        self.assertEqual(result.split_date, expected_date)

    def test_no_overlap(self):
        draws = _make_draws(200)
        result = temporal_holdout_split(draws, holdout_size=50)
        train_ids = {id(obj) for obj in result.train}
        holdout_ids = {id(obj) for obj in result.holdout}
        self.assertEqual(len(train_ids & holdout_ids), 0)


if __name__ == "__main__":
    unittest.main()
