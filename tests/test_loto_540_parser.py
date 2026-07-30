import unittest
from pathlib import Path

from loto_540_model.parser import parse_loto_540_results


class TestLoto540Parser(unittest.TestCase):
    def test_parse_extracts_main_numbers(self):
        html = Path("tests/fixtures/loto_540_with_super_noroc.html").read_text(encoding="utf-8")
        draws = parse_loto_540_results(html)
        self.assertEqual([d.date for d in draws], ["2026-01-11", "2026-01-15"])
        self.assertEqual(draws[1].main_numbers, [2, 9, 15, 27, 34, 38])

    def test_parse_extracts_super_noroc_when_present(self):
        html = Path("tests/fixtures/loto_540_with_super_noroc.html").read_text(encoding="utf-8")
        draws = parse_loto_540_results(html)
        by_date = {d.date: d for d in draws}
        self.assertEqual(by_date["2026-01-15"].super_noroc, "012345")

    def test_parse_returns_none_when_super_noroc_absent(self):
        html = Path("tests/fixtures/loto_540_with_super_noroc.html").read_text(encoding="utf-8")
        draws = parse_loto_540_results(html)
        by_date = {d.date: d for d in draws}
        self.assertIsNone(by_date["2026-01-11"].super_noroc)


if __name__ == "__main__":
    unittest.main()
