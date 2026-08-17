import math
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
        # Stated as an invariant rather than a re-derivation: a game whose
        # whole-ticket cost is already one line's worth must give the same
        # breakeven as the N-line ticket. Recomputing the formula here would
        # just re-encode whatever the model does, bug included.
        calc = EVCalculator()
        for factory in (calc.create_joker, calc.create_loto_649, calc.create_loto_540):
            multi = factory()
            single = factory(ticket_cost=multi.ticket_cost / multi.lines_per_ticket)
            single.lines_per_ticket = 1
            self.assertAlmostEqual(
                calc._calculate_positive_ev_jackpot(multi, 1.0, None),  # noqa: SLF001
                calc._calculate_positive_ev_jackpot(single, 1.0, None),  # noqa: SLF001
                delta=1.0,
                msg=multi.name,
            )

    def test_loto_540_category_i_is_the_first_five_drawn(self):
        # loto.ro Category I is "5 numere din primele 5 extrase": exactly one
        # of the C(40,5) sets wins it. Category II is the other five 5-subsets
        # of the six drawn. Modelling both as C(6,5)/C(40,5) counted one event
        # twice and made the jackpot 6x too likely.
        game = EVCalculator().create_loto_540()
        total = math.comb(40, 5)
        by_matches = {t.matches_required: t.probability for t in game.prize_tiers}
        self.assertAlmostEqual(by_matches[6], 1 / total)
        self.assertAlmostEqual(by_matches[5], 5 / total)
        self.assertAlmostEqual(by_matches[4], 510 / total)
        # 3 matches pays nothing in 5/40 — there is no fourth category.
        self.assertNotIn(3, by_matches)

    def test_loto_540_breakeven_is_far_above_routine_rollovers(self):
        # With Category I correct the breakeven is ~3.36M, not ~548K. Routine
        # 5/40 rollovers (400K-1M) are nowhere near it, so the EV gate must
        # not treat them as close to breakeven.
        calc = EVCalculator()
        breakeven = calc._calculate_positive_ev_jackpot(calc.create_loto_540(), 1.0, None)  # noqa: SLF001
        self.assertGreater(breakeven, 3_000_000)
        self.assertLess(1_000_000 / breakeven, 0.35)  # below the boost trigger


class TestTaxAndParimutuel(unittest.TestCase):
    def test_progressive_tax_raises_breakeven_over_flat_zero(self):
        calc = EVCalculator()
        game = calc.create_loto_540()
        untaxed = calc._calculate_positive_ev_jackpot(game, 1.0, 0.0)  # noqa: SLF001
        taxed = calc._calculate_positive_ev_jackpot(game, 1.0, None)  # noqa: SLF001
        # Top bracket keeps 60% of the excess, so the jackpot must be larger.
        self.assertGreater(taxed, untaxed * 1.5)

    def test_joker_category_ii_is_not_paid_the_jackpot(self):
        # "5 main, no Joker" has matches_required == numbers_drawn, which the
        # old jackpot-tier test also matched — it was paid the full jackpot.
        calc = EVCalculator()
        game = calc.create_joker()
        result = calc.calculate_ev(game, jackpot=100_000_000.0, tax_rate=0.0)
        by_tier = {t["tier"]: t["prize"] for t in result.tier_breakdown}
        self.assertEqual(by_tier["Category I (5+Joker)"], 100_000_000.0)
        self.assertLess(by_tier["Category II (5 matches)"], 1_000_000.0)

    def test_parimutuel_prize_scales_with_rarity(self):
        # The old `1000 * pct` ignored probability, so a 1-in-54k tier and a
        # 1-in-1000 tier priced the same. Rarer must pay more.
        calc = EVCalculator()
        game = calc.create_loto_649()
        cat2 = next(t for t in game.prize_tiers if t.matches_required == 5)
        self.assertGreater(calc._parimutuel_prize(game, cat2), 5_000)  # noqa: SLF001

    def test_ticket_ev_covers_every_line_it_buys(self):
        # Returns are per line; the cost is for lines_per_ticket of them.
        calc = EVCalculator()
        game = calc.create_loto_540()
        result = calc.calculate_ev(game, jackpot=445_294.0, tax_rate=0.0)
        per_line_return = sum(t["ev_contribution"] for t in result.tier_breakdown)
        self.assertAlmostEqual(
            result.expected_value,
            per_line_return * game.lines_per_ticket - game.ticket_cost,
            delta=0.01,
        )


if __name__ == "__main__":
    unittest.main()
