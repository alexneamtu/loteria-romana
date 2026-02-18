import unittest

from shared.crowding import anti_crowding_score, average_anti_crowding_score


class TestCrowdingScores(unittest.TestCase):
    def test_penalizes_birthday_heavy_lines(self):
        birthday_heavy = [1, 7, 14, 21, 28]
        higher_line = [33, 36, 39, 41, 44]
        self.assertLess(
            anti_crowding_score(birthday_heavy, pool_size=45),
            anti_crowding_score(higher_line, pool_size=45),
        )

    def test_penalizes_consecutive_patterns(self):
        patterned = [1, 2, 3, 4, 5]
        mixed = [4, 13, 26, 34, 41]
        self.assertLess(
            anti_crowding_score(patterned, pool_size=45),
            anti_crowding_score(mixed, pool_size=45),
        )

    def test_average_score_bounds(self):
        lines = [[1, 2, 3, 4, 5], [33, 36, 39, 41, 44]]
        score = average_anti_crowding_score(lines, pool_size=45)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
