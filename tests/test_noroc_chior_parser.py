import unittest
from pathlib import Path

from shared.noroc_chior import extract_years, parse_archive_draws

FIXTURES = Path("tests/fixtures")


class TestNorocChiorParser(unittest.TestCase):
    def test_extract_years(self):
        html = (FIXTURES / "noroc_chior_years.html").read_text(encoding="utf-8")
        self.assertEqual(extract_years(html), [1993, 2024])

    def test_parse_loto_649_archive(self):
        html = (FIXTURES / "noroc_chior_649_2024.html").read_text(encoding="utf-8")
        draws = parse_archive_draws(html, numbers_count=6)
        self.assertEqual(draws, [("2024-12-31", [27, 5, 49, 19, 11, 44])])

    def test_parse_loto_540_archive(self):
        html = (FIXTURES / "noroc_chior_540_2024.html").read_text(encoding="utf-8")
        draws = parse_archive_draws(html, numbers_count=6)
        self.assertEqual(draws, [("2024-12-31", [21, 16, 2, 3, 14, 17])])

    def test_parse_joker_archive(self):
        html = (FIXTURES / "noroc_chior_joker_2024.html").read_text(encoding="utf-8")
        draws = parse_archive_draws(html, numbers_count=6)
        self.assertEqual(draws, [("2024-12-31", [23, 42, 24, 37, 16, 1])])


if __name__ == "__main__":
    unittest.main()
