import unittest
from math import comb

from shared.game_recommender import (
    BudgetAllocation,
    GameProbability,
    calculate_win_probability,
    format_recommendation,
    optimize_budget,
)


class TestGameProbability(unittest.TestCase):
    def test_win_rate_per_ron(self):
        gp = GameProbability(name="test", win_rate=0.04, ticket_cost=8.0)
        self.assertAlmostEqual(gp.win_rate_per_ron, 0.005)


class TestCalculateWinProbability(unittest.TestCase):
    def test_joker_win_rate(self):
        gp = calculate_win_probability("joker")
        self.assertAlmostEqual(gp.win_rate, 0.02929, places=4)
        self.assertEqual(gp.ticket_cost, 14.5)
        self.assertEqual(gp.name, "joker")

    def test_loto_649_win_rate(self):
        gp = calculate_win_probability("loto_649")
        self.assertAlmostEqual(gp.win_rate, 0.01864, places=4)
        self.assertEqual(gp.ticket_cost, 24.5)

    def test_loto_540_win_rate(self):
        gp = calculate_win_probability("loto_540")
        self.assertAlmostEqual(gp.win_rate, 0.000784, places=5)
        self.assertEqual(gp.ticket_cost, 20.5)

    def test_joker_is_best_per_ron(self):
        joker = calculate_win_probability("joker")
        loto649 = calculate_win_probability("loto_649")
        loto540 = calculate_win_probability("loto_540")
        self.assertGreater(joker.win_rate_per_ron, loto649.win_rate_per_ron)
        self.assertGreater(loto649.win_rate_per_ron, loto540.win_rate_per_ron)

    def test_unknown_game_raises(self):
        with self.assertRaises(ValueError):
            calculate_win_probability("unknown_game")

    def test_joker_probability_exact(self):
        """Verify Joker probability against manual calculation."""
        total = comb(45, 5)
        p_joker = 1 / 20
        p3 = comb(5, 3) * comb(40, 2) / total
        p4 = comb(5, 4) * comb(40, 1) / total
        p5 = comb(5, 5) * comb(40, 0) / total
        p2j = comb(5, 2) * comb(40, 3) / total * p_joker
        p1j = comb(5, 1) * comb(40, 4) / total * p_joker
        expected = p3 + p4 + p5 + p2j + p1j

        gp = calculate_win_probability("joker")
        self.assertAlmostEqual(gp.win_rate, expected, places=10)

    def test_loto_649_probability_exact(self):
        total = comb(49, 6)
        expected = sum(comb(6, m) * comb(43, 6 - m) / total for m in range(3, 7))

        gp = calculate_win_probability("loto_649")
        self.assertAlmostEqual(gp.win_rate, expected, places=10)

    def test_loto_540_probability_exact(self):
        total = comb(40, 5)
        expected = sum(comb(6, m) * comb(34, 5 - m) / total for m in range(4, 6))

        gp = calculate_win_probability("loto_540")
        self.assertAlmostEqual(gp.win_rate, expected, places=10)


class TestOptimizeBudget(unittest.TestCase):
    def test_budget_14_5_is_one_joker(self):
        alloc = optimize_budget(14.5)
        self.assertEqual(alloc.tickets["joker"], 1)
        self.assertEqual(alloc.tickets.get("loto_649", 0), 0)
        self.assertEqual(alloc.total_cost, 14.5)

    def test_budget_39_is_two_jokers(self):
        # 2 jokers (29 RON) beats 1 joker + 1 loto_649 (39 RON) on P(any win)
        alloc = optimize_budget(39)
        self.assertEqual(alloc.tickets["joker"], 2)
        self.assertEqual(alloc.tickets.get("loto_649", 0), 0)
        self.assertAlmostEqual(alloc.total_cost, 29.0)

    def test_budget_43_5_is_three_jokers(self):
        alloc = optimize_budget(43.5)
        self.assertEqual(alloc.tickets["joker"], 3)
        self.assertAlmostEqual(alloc.total_cost, 43.5)

    def test_budget_100_primarily_joker(self):
        alloc = optimize_budget(100)
        self.assertGreater(alloc.tickets["joker"], 0)
        self.assertLessEqual(alloc.total_cost, 100)
        self.assertGreater(alloc.p_any_win, 0.15)

    def test_budget_too_small(self):
        alloc = optimize_budget(3)
        self.assertEqual(alloc.p_any_win, 0.0)
        self.assertEqual(alloc.tickets, {})

    def test_budget_zero(self):
        alloc = optimize_budget(0)
        self.assertEqual(alloc.p_any_win, 0.0)

    def test_budget_preserved(self):
        alloc = optimize_budget(50)
        self.assertEqual(alloc.budget, 50)

    def test_does_not_exceed_budget(self):
        for budget in [10, 20, 30, 50, 75, 100, 200]:
            alloc = optimize_budget(budget)
            self.assertLessEqual(alloc.total_cost, budget)

    def test_p_any_win_increases_with_budget(self):
        """P(any win) should be monotonically non-decreasing with budget."""
        prev_p = 0
        for budget in range(4, 120, 4):
            alloc = optimize_budget(budget)
            self.assertGreaterEqual(
                alloc.p_any_win, prev_p,
                f"P(win) decreased from {prev_p} to {alloc.p_any_win} at budget {budget}",
            )
            prev_p = alloc.p_any_win

    def test_joker_dominates_over_540(self):
        """At any budget, Joker tickets should be preferred over 5/40."""
        for budget in [16, 24, 40, 80]:
            alloc = optimize_budget(budget)
            self.assertEqual(
                alloc.tickets.get("loto_540", 0), 0,
                f"5/40 tickets allocated at budget {budget}",
            )

    def test_single_ticket_probabilities(self):
        """Single-ticket allocation should match per-game probabilities."""
        alloc = optimize_budget(14.5)
        joker_p = calculate_win_probability("joker").win_rate
        self.assertAlmostEqual(alloc.p_any_win, joker_p, places=6)


class TestFormatRecommendation(unittest.TestCase):
    def test_format_includes_budget(self):
        alloc = optimize_budget(24)
        text = format_recommendation(alloc)
        self.assertIn("24 RON", text)

    def test_format_includes_p_win(self):
        alloc = optimize_budget(24)
        text = format_recommendation(alloc)
        self.assertIn("P(any win)", text)

    def test_format_too_small_budget(self):
        alloc = optimize_budget(2)
        text = format_recommendation(alloc)
        self.assertIn("too small", text)

    def test_format_shows_game_names(self):
        # Joker always dominates at correct pricing; verify it appears in output
        alloc = optimize_budget(39)
        text = format_recommendation(alloc)
        self.assertIn("Joker", text)

    def test_format_shows_unspent(self):
        # 1 Joker costs 14.5 RON, leaving 2.0 RON unspent from 16.5
        alloc = optimize_budget(16.5)
        text = format_recommendation(alloc)
        self.assertIn("Unspent: 2 RON", text)


if __name__ == "__main__":
    unittest.main()
