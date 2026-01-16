import unittest
from pathlib import Path

from joker_model.parser import parse_joker_results


class TestParser(unittest.TestCase):
    def test_parse_snippet(self):
        html = Path("tests/fixtures/joker_results_snippet.html").read_text(encoding="utf-8")
        draws = parse_joker_results(html)
        self.assertEqual(len(draws), 2)
        self.assertEqual(draws[0].date, "2026-01-15")
        self.assertEqual(draws[0].main_numbers, [7, 11, 44, 45, 46])
        self.assertEqual(draws[0].joker, 13)


if __name__ == "__main__":
    unittest.main()
