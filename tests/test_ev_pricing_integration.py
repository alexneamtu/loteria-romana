import unittest

from shared.ev_calculator import EVCalculator


class TestEVCalculatorUsesSharedPricing(unittest.TestCase):
    def test_joker_default_cost_matches_pricing_module(self):
        game = EVCalculator.create_joker()
        # 2 variants × 7.0 + 0.5 fee (no side game in EV main tiers) = 14.5
        self.assertEqual(game.ticket_cost, 14.5)

    def test_loto_649_default_cost_matches_pricing_module(self):
        game = EVCalculator.create_loto_649()
        # 3 × 8.0 + 0.5 = 24.5
        self.assertEqual(game.ticket_cost, 24.5)

    def test_loto_540_default_cost_matches_pricing_module(self):
        game = EVCalculator.create_loto_540()
        # 4 × 5.0 + 0.5 = 20.5
        self.assertEqual(game.ticket_cost, 20.5)

    def test_explicit_override_still_works(self):
        game = EVCalculator.create_joker(ticket_cost=99.0)
        self.assertEqual(game.ticket_cost, 99.0)


class TestBreakevenSpreadsCostAcrossLines(unittest.TestCase):
    """A ticket buys N independent lines but the prize probabilities are
    per line, so breakeven must charge cost/N, not the whole-ticket cost.
    Charging the full multi-variant cost against a single line's jackpot
    probability overstated breakeven by ~N (the variant count)."""

    def test_breakeven_uses_per_line_cost(self):
        calc = EVCalculator()
        for game in (calc.create_joker(), calc.create_loto_649(), calc.create_loto_540()):
            breakeven = calc._calculate_positive_ev_jackpot(game, 1.0, 0.0)  # noqa: SLF001
            jackpot_tier = next(
                t for t in game.prize_tiers
                if t.matches_required == game.numbers_drawn
                or (t.matches_required == game.numbers_picked
                    and t.bonus_required and game.has_bonus)
            )
            fixed_ev = sum(
                t.probability * t.fixed_prize
                for t in game.prize_tiers
                if t is not jackpot_tier and t.fixed_prize is not None
            )
            per_line = (game.ticket_cost / game.lines_per_ticket - fixed_ev) / jackpot_tier.probability
            whole_ticket = (game.ticket_cost - fixed_ev) / jackpot_tier.probability
            self.assertAlmostEqual(breakeven, per_line, delta=1.0)
            # The bug charged the whole ticket against one line; guard it.
            self.assertLess(breakeven, whole_ticket * 0.99)

    def test_loto_540_breakeven_is_reachable_by_real_rollovers(self):
        # Real 5/40 rollovers reach 400K-1M RON (see README). The per-line
        # breakeven (~548K) sits inside that band; the old whole-ticket
        # figure (~2.23M) wrongly put 5/40 out of reach.
        calc = EVCalculator()
        breakeven = calc._calculate_positive_ev_jackpot(calc.create_loto_540(), 1.0, 0.0)  # noqa: SLF001
        self.assertGreater(breakeven, 400_000)
        self.assertLess(breakeven, 1_000_000)


if __name__ == "__main__":
    unittest.main()
