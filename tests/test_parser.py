import unittest
from pathlib import Path

from joker_model.parser import parse_joker_results
from joker_model.models import JokerDraw


class TestParser(unittest.TestCase):
    def test_parse_snippet(self):
        html = Path("tests/fixtures/joker_results_snippet.html").read_text(encoding="utf-8")
        draws = parse_joker_results(html)
        self.assertEqual(len(draws), 2)
        self.assertEqual(draws[0].date, "2026-01-15")
        self.assertEqual(draws[0].main_numbers, [7, 11, 44, 45, 46])
        self.assertEqual(draws[0].joker, 13)


class TestJokerDrawModel(unittest.TestCase):
    def test_noroc_plus_defaults_to_none(self):
        d = JokerDraw(date="2026-01-15", main_numbers=[7, 11, 44, 45, 46], joker=13)
        self.assertIsNone(d.noroc_plus)

    def test_noroc_plus_accepts_string(self):
        d = JokerDraw(
            date="2026-01-15",
            main_numbers=[7, 11, 44, 45, 46],
            joker=13,
            noroc_plus="NP07",
        )
        self.assertEqual(d.noroc_plus, "NP07")

    def test_joker_draw_still_hashable(self):
        d = JokerDraw(date="2026-01-15", main_numbers=[7, 11, 44, 45, 46], joker=13)
        self.assertEqual(hash(d), hash(d))


class TestParserWithNorocPlus(unittest.TestCase):
    def test_parse_extracts_noroc_plus_when_present(self):
        html = Path("tests/fixtures/joker_with_noroc_plus.html").read_text(encoding="utf-8")
        draws = parse_joker_results(html)
        self.assertEqual(len(draws), 2)
        self.assertEqual(draws[0].date, "2026-01-15")
        self.assertEqual(draws[0].noroc_plus, "NP07")

    def test_parse_returns_none_when_noroc_plus_absent(self):
        # Old fixture has no noroc-plus img in its 2000-char window.
        html = Path("tests/fixtures/joker_results_snippet.html").read_text(encoding="utf-8")
        draws = parse_joker_results(html)
        self.assertEqual(len(draws), 2)
        self.assertIsNone(draws[0].noroc_plus)
        self.assertIsNone(draws[1].noroc_plus)


if __name__ == "__main__":
    unittest.main()
