import csv
import tempfile
import unittest
from pathlib import Path

from scripts.migrate_history_schema import migrate

NEW_COLUMNS = [
    "ticket_id",
    "variants_count",
    "side_game_match",
    "side_game_digits",
    "winning_side_game",
    "builder_name",
]


class TestMigrateHistorySchema(unittest.TestCase):
    def _write_legacy(self, path: Path) -> None:
        path.write_text(
            "date,game,strategy,picks_count,total_matches,best_match,match_total,winning_numbers\n"
            "2026-02-12,joker,recommended,5,4,2,5,\"6, 9, 18, 22, 27 + J11\"\n"
            "2026-03-05,joker,mix1,5,4,2,5,\"6, 15, 22, 33, 43 + J16\"\n",
            encoding="utf-8",
        )

    def test_adds_new_columns_with_empty_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            self._write_legacy(path)
            migrate(path)
            with path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            for col in NEW_COLUMNS:
                self.assertIn(col, reader.fieldnames)
                for row in rows:
                    self.assertEqual(row[col], "")

    def test_preserves_existing_column_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            self._write_legacy(path)
            migrate(path)
            with path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["date"], "2026-02-12")
            self.assertEqual(rows[0]["game"], "joker")
            self.assertEqual(rows[0]["total_matches"], "4")

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            self._write_legacy(path)
            migrate(path)
            migrate(path)
            with path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            # Each new column should appear exactly once
            self.assertEqual(len([c for c in reader.fieldnames if c == "ticket_id"]), 1)
            self.assertEqual(len(rows), 2)

    def test_empty_file_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            path.touch()
            migrate(path)
            self.assertEqual(path.read_text(), "")


if __name__ == "__main__":
    unittest.main()
