import random
import unittest
from types import SimpleNamespace
from unittest import mock

import scripts.generate_recommended_picks as recommended_script
from shared.game_recommender import BudgetAllocation


class TestGenerateJokerPicks(unittest.TestCase):
    def test_applies_set_optimization_and_joker_coverage(self):
        draws = [
            SimpleNamespace(date="2024-01-01", main_numbers=[1, 2, 3, 4, 5]),
            SimpleNamespace(date="2024-01-05", main_numbers=[2, 3, 4, 5, 6]),
        ]

        duplicate_main = [[1, 2, 3, 4, 5]] * 5
        with mock.patch.object(
            recommended_script,
            "generate_blended_picks",
            return_value=duplicate_main,
        ):
            lines = recommended_script.generate_joker_picks(
                draws=draws,
                count=5,
                rng=random.Random(123),
                half_life=50.0,
                half_life_mode="draws",
            )

        mains = [tuple(main) for main, _ in lines]
        jokers = [joker for _, joker in lines]

        self.assertEqual(len(lines), 5)
        self.assertEqual(len(set(mains)), 5)
        self.assertEqual(len(set(jokers)), 5)


class TestEVGate(unittest.TestCase):
    def test_gate_disabled_keeps_allocation(self):
        allocation = BudgetAllocation(
            tickets={"joker": 2, "loto_649": 1, "loto_540": 0},
            total_cost=22.0,
            p_any_win=0.10,
            budget=22.0,
        )
        gated, details = recommended_script.apply_ev_gate(
            allocation=allocation,
            enabled=False,
            min_ratio=0.8,
            jackpots={},
        )
        self.assertEqual(gated.tickets, allocation.tickets)
        self.assertEqual(gated.total_cost, allocation.total_cost)
        self.assertEqual(details, {})

    def test_gate_filters_below_threshold_games(self):
        allocation = BudgetAllocation(
            tickets={"joker": 2, "loto_649": 2, "loto_540": 1},
            total_cost=32.0,
            p_any_win=0.15,
            budget=32.0,
        )
        jackpots = {
            "joker": 1.0,
            "loto_649": 1_000_000_000.0,
            "loto_540": 1.0,
        }
        gated, details = recommended_script.apply_ev_gate(
            allocation=allocation,
            enabled=True,
            min_ratio=0.8,
            jackpots=jackpots,
        )
        self.assertEqual(gated.tickets["joker"], 0)
        self.assertGreaterEqual(gated.tickets["loto_649"], 1)
        self.assertEqual(gated.tickets["loto_540"], 0)
        self.assertIn("joker", details)
        self.assertIn("loto_649", details)


if __name__ == "__main__":
    unittest.main()
