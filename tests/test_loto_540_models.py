import unittest

from loto_540_model.models import Loto540Draw


class TestLoto540DrawModel(unittest.TestCase):
    def test_super_noroc_defaults_to_none(self):
        d = Loto540Draw(date="2026-01-15", main_numbers=[2, 9, 15, 27, 34, 38])
        self.assertIsNone(d.super_noroc)

    def test_super_noroc_accepts_6_digit_string(self):
        d = Loto540Draw(
            date="2026-01-15",
            main_numbers=[2, 9, 15, 27, 34, 38],
            super_noroc="012345",
        )
        self.assertEqual(d.super_noroc, "012345")


if __name__ == "__main__":
    unittest.main()
