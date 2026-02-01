import random
import unittest

from shared.cooccurrence import CooccurrenceStrategy


class TestCooccurrenceStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = CooccurrenceStrategy(
            number_pool=45,
            numbers_to_pick=5,
        )
        self.draws = [
            [1, 2, 3, 4, 5],
            [1, 2, 6, 7, 8],
            [3, 4, 9, 10, 11],
            [1, 3, 5, 7, 9],
            [2, 4, 6, 8, 10],
        ]

    def test_build_matrix_shape(self):
        matrix = self.strategy.build_matrix(self.draws)
        self.assertEqual(len(matrix), 45)
        self.assertEqual(len(matrix[0]), 45)

    def test_build_matrix_symmetry(self):
        matrix = self.strategy.build_matrix(self.draws)
        for i in range(45):
            for j in range(45):
                self.assertAlmostEqual(matrix[i][j], matrix[j][i])

    def test_build_matrix_diagonal_zero(self):
        matrix = self.strategy.build_matrix(self.draws)
        for i in range(45):
            self.assertAlmostEqual(matrix[i][i], 0.0)

    def test_cooccurring_pairs_have_positive_weight(self):
        matrix = self.strategy.build_matrix(self.draws)
        self.assertGreater(matrix[0][1], 0)  # 1 and 2 co-occur
        self.assertGreater(matrix[0][2], 0)  # 1 and 3 co-occur

    def test_non_cooccurring_pairs_are_zero(self):
        matrix = self.strategy.build_matrix(self.draws)
        self.assertAlmostEqual(matrix[43][44], 0.0)

    def test_empty_draws(self):
        matrix = self.strategy.build_matrix([])
        self.assertTrue(all(all(v == 0.0 for v in row) for row in matrix))

    def test_get_probabilities_sums_to_one(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

    def test_get_probabilities_length(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertEqual(len(probs), 45)

    def test_generate_returns_correct_count(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 3, rng)
        self.assertEqual(len(lines), 3)

    def test_generate_lines_are_valid(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 5, rng)
        for line in lines:
            self.assertEqual(len(line), 5)
            self.assertEqual(sorted(line), line)
            self.assertTrue(all(1 <= n <= 45 for n in line))
            self.assertEqual(len(set(line)), 5)

    def test_generate_lines_are_unique(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 10, rng)
        keys = [tuple(l) for l in lines]
        self.assertEqual(len(keys), len(set(keys)))

    def test_name(self):
        self.assertEqual(self.strategy.name, "cooccurrence")


if __name__ == "__main__":
    unittest.main()
