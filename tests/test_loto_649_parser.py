import unittest
from pathlib import Path

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
        self.assertEqual(draw_by_date["2026-01-15"].noroc, 6026250)


if __name__ == "__main__":
    unittest.main()
