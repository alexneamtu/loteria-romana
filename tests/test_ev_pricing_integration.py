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


if __name__ == "__main__":
    unittest.main()
