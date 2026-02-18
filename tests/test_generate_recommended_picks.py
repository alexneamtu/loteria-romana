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


class TestGenerationDBPersistenceHook(unittest.TestCase):
    def test_main_persists_even_when_allocation_has_zero_win_probability(self):
        zero_allocation = BudgetAllocation(
            tickets={"joker": 0, "loto_649": 0, "loto_540": 0},
            total_cost=0.0,
            p_any_win=0.0,
            budget=40.0,
        )
        with mock.patch.object(recommended_script, "resolve_recency_settings", return_value=(50.0, "draws")), \
            mock.patch.object(recommended_script, "optimize_budget", return_value=zero_allocation), \
            mock.patch.object(recommended_script, "apply_ev_gate", return_value=(zero_allocation, {})), \
            mock.patch.object(recommended_script, "format_recommendation", return_value="ok"), \
            mock.patch.object(recommended_script, "persist_generation_run", return_value=True) as persist, \
            mock.patch("builtins.print"), \
            mock.patch("sys.argv", ["generate_recommended_picks.py", "--budget", "40"]):
            recommended_script.main()

        self.assertEqual(persist.call_count, 1)
        _, kwargs = persist.call_args
        self.assertEqual(kwargs["tickets"], [])
        self.assertEqual(kwargs["allocation"]["p_any_win"], 0.0)

    def test_main_persists_generated_ticket_rows(self):
        allocation = BudgetAllocation(
            tickets={"joker": 1, "loto_649": 0, "loto_540": 0},
            total_cost=8.0,
            p_any_win=0.1,
            budget=8.0,
        )
        draws = [SimpleNamespace(date="2024-01-01", main_numbers=[1, 2, 3, 4, 5], joker=7)]
        with mock.patch.object(recommended_script, "resolve_recency_settings", return_value=(50.0, "draws")), \
            mock.patch.object(recommended_script, "optimize_budget", return_value=allocation), \
            mock.patch.object(recommended_script, "apply_ev_gate", return_value=(allocation, {})), \
            mock.patch.object(recommended_script, "format_recommendation", return_value="ok"), \
            mock.patch.object(recommended_script, "load_joker_draws", return_value=draws), \
            mock.patch.object(
                recommended_script,
                "generate_joker_picks",
                return_value=[([1, 2, 3, 4, 5], 7)],
            ), \
            mock.patch.object(recommended_script, "persist_generation_run", return_value=True) as persist, \
            mock.patch("builtins.print"), \
            mock.patch("sys.argv", ["generate_recommended_picks.py", "--budget", "8"]):
            recommended_script.main()

        self.assertEqual(persist.call_count, 1)
        _, kwargs = persist.call_args
        self.assertEqual(len(kwargs["tickets"]), 1)
        self.assertEqual(kwargs["tickets"][0]["game"], "joker")
        self.assertEqual(kwargs["tickets"][0]["main_numbers"], [1, 2, 3, 4, 5])
        self.assertEqual(kwargs["tickets"][0]["joker_number"], 7)


if __name__ == "__main__":
    unittest.main()
