import unittest
from pathlib import Path

from loto_649_model.models import Loto649Draw
from loto_649_model.parser import parse_loto_649_results


class TestLoto649Parser(unittest.TestCase):
    def test_parse_snippet(self):
        html = Path("tests/fixtures/loto_649_results_snippet.html").read_text(encoding="utf-8")
        draws = parse_loto_649_results(html)
        self.assertEqual(len(draws), 2)
        draw_by_date = {draw.date: draw for draw in draws}
        self.assertIn("2026-01-15", draw_by_date)
        self.assertIn("2026-01-11", draw_by_date)
        self.assertEqual(draw_by_date["2026-01-15"].main_numbers, [11, 19, 33, 44, 45, 46])


class TestLoto649DrawModel(unittest.TestCase):
    def test_noroc_defaults_to_none(self):
        d = Loto649Draw(date="2026-01-15", main_numbers=[1, 5, 17, 23, 34, 49])
        self.assertIsNone(d.noroc)

    def test_noroc_accepts_7_digit_string(self):
        d = Loto649Draw(
            date="2026-01-15",
            main_numbers=[1, 5, 17, 23, 34, 49],
            noroc="0123456",
        )
        self.assertEqual(d.noroc, "0123456")


class TestLoto649ParserNoroc(unittest.TestCase):
    def test_parse_extracts_noroc_when_present(self):
        html = Path("tests/fixtures/loto_649_with_noroc.html").read_text(encoding="utf-8")
        draws = parse_loto_649_results(html)
        by_date = {d.date: d for d in draws}
        self.assertEqual(by_date["2026-01-15"].noroc, "0123456")

    def test_parse_returns_none_when_noroc_absent(self):
        html = Path("tests/fixtures/loto_649_with_noroc.html").read_text(encoding="utf-8")
        draws = parse_loto_649_results(html)
        by_date = {d.date: d for d in draws}
        self.assertIsNone(by_date["2026-01-11"].noroc)


if __name__ == "__main__":
    unittest.main()
