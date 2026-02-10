import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_results import append_history, parse_picks, check_matches, HISTORY_COLUMNS


class TestParseAndMatch(unittest.TestCase):
    def test_parse_picks(self):
        text = "1. 3, 12, 25, 33, 41\n2. 7, 14, 22, 30, 45\n"
        picks = parse_picks(text)
        self.assertEqual(len(picks), 2)
        self.assertEqual(picks[0], [3, 12, 25, 33, 41])

    def test_check_matches(self):
        picks = [[1, 2, 3, 4, 5]]
        winning = [3, 4, 5, 6, 7]
        results = check_matches(picks, winning)
        self.assertEqual(results[0]["count"], 3)
        self.assertEqual(results[0]["matched"], [3, 4, 5])


class TestAppendHistory(unittest.TestCase):
    def test_creates_file_with_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.csv"
            rows = [("2025-01-15", "joker", "1, 2, 3 + J5", {
                "recommended": {
                    "results": [{"pick": [1, 2, 3], "matched": [3], "count": 1}],
                    "score": 1,
                    "best_match": 1,
                },
            }, 5)]
            append_history(path, rows)

            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["game"], "joker")
            self.assertEqual(data[0]["strategy"], "recommended")
            self.assertEqual(data[0]["total_matches"], "1")
            self.assertEqual(set(reader.fieldnames), set(HISTORY_COLUMNS))

    def test_appends_without_duplicate_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.csv"
            row = [("2025-01-15", "joker", "1, 2, 3", {
                "rec": {"results": [], "score": 0, "best_match": 0},
            }, 5)]
            append_history(path, row)
            append_history(path, row)

            lines = path.read_text().strip().split("\n")
            header_count = sum(1 for l in lines if l.startswith("date,"))
            self.assertEqual(header_count, 1)
            self.assertEqual(len(lines), 3)  # header + 2 data rows

    def test_skips_games_without_strategies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.csv"
            rows = [("2025-01-15", "loto540", "1, 2, 3", {}, 5)]
            append_history(path, rows)

            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)
            self.assertEqual(len(data), 0)
