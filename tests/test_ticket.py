import unittest

from shared.ticket import Variant


class TestVariant(unittest.TestCase):
    def test_joker_variant_accepts_5_mains_and_1_bonus(self):
        v = Variant(main_numbers=(3, 7, 12, 19, 28), bonus_number=11, game="joker")
        self.assertEqual(v.main_numbers, (3, 7, 12, 19, 28))
        self.assertEqual(v.bonus_number, 11)

    def test_loto_649_variant_accepts_6_mains_and_no_bonus(self):
        v = Variant(main_numbers=(1, 5, 17, 23, 34, 49), bonus_number=None, game="loto_649")
        self.assertIsNone(v.bonus_number)

    def test_loto_540_variant_accepts_5_mains_and_no_bonus(self):
        v = Variant(main_numbers=(2, 9, 15, 27, 40), bonus_number=None, game="loto_540")
        self.assertIsNone(v.bonus_number)

    def test_rejects_wrong_main_count(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(1, 2, 3, 4), bonus_number=5, game="joker")

    def test_rejects_out_of_range_main(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(1, 2, 3, 4, 46), bonus_number=5, game="joker")

    def test_rejects_duplicate_mains(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(1, 2, 3, 4, 4), bonus_number=5, game="joker")

    def test_rejects_unsorted_mains(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(5, 1, 2, 3, 4), bonus_number=5, game="joker")

    def test_rejects_missing_bonus_for_joker(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(1, 2, 3, 4, 5), bonus_number=None, game="joker")

    def test_rejects_bonus_for_non_joker(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(1, 5, 17, 23, 34, 49), bonus_number=3, game="loto_649")

    def test_rejects_out_of_range_bonus(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(1, 2, 3, 4, 5), bonus_number=21, game="joker")

    def test_is_frozen_hashable(self):
        v = Variant(main_numbers=(3, 7, 12, 19, 28), bonus_number=11, game="joker")
        self.assertEqual(hash(v), hash(v))

    def test_count_main_matches(self):
        v = Variant(main_numbers=(3, 7, 12, 19, 28), bonus_number=11, game="joker")
        self.assertEqual(v.count_main_matches((3, 7, 99, 28, 100)), 3)


from shared.ticket import Ticket


class TestTicket(unittest.TestCase):
    def _joker_variants(self) -> tuple[Variant, Variant]:
        v1 = Variant(main_numbers=(3, 7, 12, 19, 28), bonus_number=11, game="joker")
        v2 = Variant(main_numbers=(3, 7, 12, 19, 33), bonus_number=11, game="joker")
        return (v1, v2)

    def test_joker_ticket_requires_2_variants(self):
        variants = self._joker_variants()
        t = Ticket(
            game="joker",
            variants=variants,
            side_game_number="NP14",
            strategy="core_share",
            cost_ron=17.5,
        )
        self.assertEqual(len(t.variants), 2)
        self.assertEqual(t.side_game_number, "NP14")

    def test_joker_ticket_rejects_wrong_variant_count(self):
        v1, _ = self._joker_variants()
        with self.assertRaises(ValueError):
            Ticket(
                game="joker",
                variants=(v1,),
                side_game_number="NP14",
                strategy="core_share",
                cost_ron=10.5,
            )

    def test_ticket_rejects_variants_of_wrong_game(self):
        joker_v = Variant(main_numbers=(3, 7, 12, 19, 28), bonus_number=11, game="joker")
        loto_v = Variant(main_numbers=(1, 5, 17, 23, 34, 49), bonus_number=None, game="loto_649")
        with self.assertRaises(ValueError):
            Ticket(
                game="joker",
                variants=(joker_v, loto_v),
                side_game_number="NP14",
                strategy="core_share",
                cost_ron=17.5,
            )

    def test_best_main_match_across_variants(self):
        variants = self._joker_variants()
        t = Ticket(
            game="joker",
            variants=variants,
            side_game_number="NP14",
            strategy="core_share",
            cost_ron=17.5,
        )
        # winning = 3,7,12,44,45 + J11 → v1 matches 3, v2 matches 3 on mains
        self.assertEqual(t.best_main_match((3, 7, 12, 44, 45)), 3)
        # winning = 3,7,12,19,28 → v1 matches 5, v2 matches 4
        self.assertEqual(t.best_main_match((3, 7, 12, 19, 28)), 5)

    def test_loto_649_ticket_requires_3_variants(self):
        mk = lambda n: Variant(
            main_numbers=tuple(sorted((1, 5, 17, 23, 34, n))),
            bonus_number=None,
            game="loto_649",
        )
        variants = (mk(45), mk(46), mk(47))
        t = Ticket(
            game="loto_649",
            variants=variants,
            side_game_number="1234567",
            strategy="wheel_3if9",
            cost_ron=28.5,
        )
        self.assertEqual(len(t.variants), 3)

    def test_loto_540_ticket_requires_4_variants(self):
        mk = lambda n: Variant(
            main_numbers=tuple(sorted((2, 9, 15, 27, n))),
            bonus_number=None,
            game="loto_540",
        )
        variants = (mk(35), mk(36), mk(37), mk(38))
        t = Ticket(
            game="loto_540",
            variants=variants,
            side_game_number="123456",
            strategy="independent",
            cost_ron=22.5,
        )
        self.assertEqual(len(t.variants), 4)

    def test_side_game_number_is_string_preserving_leading_zeros(self):
        variants = self._joker_variants()
        t = Ticket(
            game="joker",
            variants=variants,
            side_game_number="NP07",
            strategy="core_share",
            cost_ron=17.5,
        )
        self.assertEqual(t.side_game_number, "NP07")

    def test_ticket_is_frozen(self):
        variants = self._joker_variants()
        t = Ticket(
            game="joker",
            variants=variants,
            side_game_number="NP14",
            strategy="core_share",
            cost_ron=17.5,
        )
        with self.assertRaises(Exception):
            t.strategy = "other"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
