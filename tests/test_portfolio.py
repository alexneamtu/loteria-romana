import unittest
import random

from shared.portfolio import (
    compute_ticket_covariance,
    optimize_ticket_portfolio,
    diversity_score,
)
from shared.game_config import JOKER_CONFIG, LOTO_649_CONFIG


class TestComputeTicketCovariance(unittest.TestCase):
    def test_identical_tickets_zero_variance(self):
        tickets = [[1, 2, 3, 4, 5]] * 3
        cov = compute_ticket_covariance(tickets, pool_size=45)
        # Identical tickets have zero variance (each equals the mean)
        self.assertAlmostEqual(cov[0][0], 0.0)
        self.assertAlmostEqual(cov[0][1], cov[0][0])

    def test_overlapping_tickets_positive_covariance(self):
        tickets = [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [10, 20, 30, 40, 45]]
        cov = compute_ticket_covariance(tickets, pool_size=45)
        # Tickets sharing numbers have higher covariance than disjoint ones
        self.assertGreater(cov[0][1], cov[0][2])

    def test_disjoint_tickets_negative_covariance(self):
        tickets = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        cov = compute_ticket_covariance(tickets, pool_size=45)
        # Completely disjoint tickets have negative off-diagonal covariance
        self.assertLess(cov[0][1], 0.0)

    def test_matrix_dimensions(self):
        tickets = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        cov = compute_ticket_covariance(tickets, pool_size=10)
        self.assertEqual(len(cov), 3)
        self.assertEqual(len(cov[0]), 3)

    def test_symmetry(self):
        tickets = [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [8, 9, 10, 11, 12]]
        cov = compute_ticket_covariance(tickets, pool_size=45)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(cov[i][j], cov[j][i])


class TestOptimizeTicketPortfolio(unittest.TestCase):
    def test_selects_correct_count(self):
        rng = random.Random(42)
        candidates = [sorted(rng.sample(range(1, 46), 5)) for _ in range(20)]
        selected = optimize_ticket_portfolio(candidates, select_count=5, pool_size=45)
        self.assertEqual(len(selected), 5)

    def test_selected_from_candidates(self):
        rng = random.Random(42)
        candidates = [sorted(rng.sample(range(1, 46), 5)) for _ in range(20)]
        candidate_set = {tuple(c) for c in candidates}
        selected = optimize_ticket_portfolio(candidates, select_count=5, pool_size=45)
        for s in selected:
            self.assertIn(tuple(s), candidate_set)

    def test_diversity_better_than_random(self):
        rng = random.Random(42)
        diverse = [sorted(range(i * 5 + 1, i * 5 + 6)) for i in range(8)]
        redundant = [[1, 2, 3, 4, 5]] * 12
        candidates = diverse + redundant
        selected = optimize_ticket_portfolio(candidates, select_count=5, pool_size=45)
        score = diversity_score(selected, pool_size=45)
        self.assertGreater(score, 0.5)

    def test_fewer_candidates_than_requested(self):
        candidates = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        selected = optimize_ticket_portfolio(candidates, select_count=5, pool_size=45)
        self.assertEqual(len(selected), 2)

    def test_empty_candidates(self):
        selected = optimize_ticket_portfolio([], select_count=5, pool_size=45)
        self.assertEqual(len(selected), 0)


class TestDiversityScore(unittest.TestCase):
    def test_identical_tickets_low_score(self):
        tickets = [[1, 2, 3, 4, 5]] * 5
        score = diversity_score(tickets, pool_size=45)
        self.assertLess(score, 0.2)

    def test_diverse_tickets_high_score(self):
        tickets = [
            [1, 2, 3, 4, 5],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
            [40, 41, 42, 43, 44],
        ]
        score = diversity_score(tickets, pool_size=45)
        self.assertGreater(score, 0.8)

    def test_single_ticket(self):
        score = diversity_score([[1, 2, 3, 4, 5]], pool_size=45)
        self.assertEqual(score, 1.0)

    def test_empty_list(self):
        score = diversity_score([], pool_size=45)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
