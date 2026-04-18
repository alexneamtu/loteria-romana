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


if __name__ == "__main__":
    unittest.main()
