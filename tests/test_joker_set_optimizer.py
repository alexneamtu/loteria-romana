import random
import unittest

from shared.portfolio import diversity_score
from shared.joker_set_optimizer import (
    assign_max_coverage_jokers,
    optimize_main_ticket_set,
)


class TestAssignMaxCoverageJokers(unittest.TestCase):
    def test_unique_within_first_pool(self):
        jokers = assign_max_coverage_jokers(20, random.Random(42), joker_pool=20)
        self.assertEqual(len(jokers), 20)
        self.assertEqual(len(set(jokers)), 20)
        self.assertEqual(sorted(jokers), list(range(1, 21)))

    def test_cycles_for_more_than_pool(self):
        jokers = assign_max_coverage_jokers(25, random.Random(7), joker_pool=20)
        self.assertEqual(len(jokers), 25)
        self.assertEqual(len(set(jokers[:20])), 20)
        self.assertEqual(sorted(set(jokers)), list(range(1, 21)))


class TestOptimizeMainTicketSet(unittest.TestCase):
    def test_deduplicates_and_fills_to_target_count(self):
        candidates = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ]
        selected = optimize_main_ticket_set(
            candidates,
            select_count=5,
            pool_size=45,
            numbers_to_pick=5,
            rng=random.Random(123),
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(len({tuple(line) for line in selected}), 5)

    def test_improves_diversity_over_duplicate_prefix(self):
        duplicate = [1, 2, 3, 4, 5]
        candidates = [duplicate] * 10 + [
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30],
        ]
        baseline = candidates[:5]
        baseline_score = diversity_score(baseline, pool_size=45)

        selected = optimize_main_ticket_set(
            candidates,
            select_count=5,
            pool_size=45,
            numbers_to_pick=5,
            rng=random.Random(999),
        )
        score = diversity_score(selected, pool_size=45)
        self.assertGreater(score, baseline_score)

    def test_anti_crowding_weight_steers_selection(self):
        candidates = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [33, 36, 39, 41, 44],
            [32, 35, 37, 40, 45],
            [34, 38, 42, 43, 44],
            [31, 33, 37, 41, 45],
        ]
        selected = optimize_main_ticket_set(
            candidates,
            select_count=4,
            pool_size=45,
            numbers_to_pick=5,
            rng=random.Random(77),
            anti_crowding_weight=0.6,
        )
        avg_low_numbers = sum(
            sum(1 for n in line if n <= 31)
            for line in selected
        ) / len(selected)
        self.assertLess(avg_low_numbers, 3.0)


if __name__ == "__main__":
    unittest.main()
