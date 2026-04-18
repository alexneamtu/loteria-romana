import unittest

from shared.pricing import (
    PRICE_PER_VARIANT,
    VARIANTS_PER_TICKET,
    SIDE_GAME_PRICE,
    PROCESSING_FEE_RON,
    TICKET_PRICE_NEEDS_VERIFICATION,
    compute_ticket_cost,
)


class TestPricing(unittest.TestCase):
    def test_variants_per_ticket_matches_loto_ro_format(self):
        self.assertEqual(VARIANTS_PER_TICKET["joker"], 2)
        self.assertEqual(VARIANTS_PER_TICKET["loto_649"], 3)
        self.assertEqual(VARIANTS_PER_TICKET["loto_540"], 4)

    def test_per_variant_prices_match_confirmed_loto_ro_values(self):
        self.assertEqual(PRICE_PER_VARIANT["joker"], 7.0)
        self.assertEqual(PRICE_PER_VARIANT["loto_649"], 8.0)
        self.assertEqual(PRICE_PER_VARIANT["loto_540"], 5.0)

    def test_processing_fee_is_half_ron(self):
        self.assertEqual(PROCESSING_FEE_RON, 0.5)

    def test_side_game_prices_match_confirmed_values(self):
        self.assertEqual(SIDE_GAME_PRICE["joker"], 3.0)       # Noroc Plus
        self.assertEqual(SIDE_GAME_PRICE["loto_649"], 4.0)    # Noroc
        self.assertEqual(SIDE_GAME_PRICE["loto_540"], 2.0)    # Super Noroc

    def test_verification_flag_is_false_now_that_all_prices_are_confirmed(self):
        self.assertFalse(TICKET_PRICE_NEEDS_VERIFICATION)

    def test_full_ticket_cost_includes_variants_fee_and_side_game(self):
        # Joker: 2 * 7.0 + 0.5 + 3.0 = 17.5
        self.assertEqual(compute_ticket_cost("joker"), 17.5)
        # Loto 6/49: 3 * 8.0 + 0.5 + 4.0 = 28.5
        self.assertEqual(compute_ticket_cost("loto_649"), 28.5)
        # Loto 5/40: 4 * 5.0 + 0.5 + 2.0 = 22.5
        self.assertEqual(compute_ticket_cost("loto_540"), 22.5)

    def test_variant_count_override_still_pays_fee_once(self):
        # 1 joker variant + fee, no side game: 7.0 + 0.5 = 7.5
        self.assertEqual(
            compute_ticket_cost("joker", variants=1, include_side_game=False),
            7.5,
        )
        # 2 loto_649 variants + fee: 16.0 + 0.5 = 16.5
        self.assertEqual(
            compute_ticket_cost("loto_649", variants=2, include_side_game=False),
            16.5,
        )

    def test_exclude_processing_fee_for_theoretical_variant_only_cost(self):
        self.assertEqual(
            compute_ticket_cost("joker", include_side_game=False, include_fee=False),
            14.0,
        )

    def test_exclude_side_game(self):
        # Full ticket without side game still pays variants + fee: 14.0 + 0.5
        self.assertEqual(compute_ticket_cost("joker", include_side_game=False), 14.5)

    def test_unknown_game_raises(self):
        with self.assertRaises(KeyError):
            compute_ticket_cost("powerball")


if __name__ == "__main__":
    unittest.main()
