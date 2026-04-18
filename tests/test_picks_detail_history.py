import json
import tempfile
import unittest
from pathlib import Path

from shared.picks_detail_history import append_detail_rows


class TestPicksDetailHistory(unittest.TestCase):
    def test_appends_one_json_line_per_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "picks_detail.jsonl"
            rows = [
                {
                    "draw_date": "2026-04-21",
                    "game": "joker",
                    "ticket_id": "t-01",
                    "best_main_match": 3,
                    "side_game_match": 0,
                    "side_game_digits_matched": 0,
                    "cost_ron": 17.5,
                },
                {
                    "draw_date": "2026-04-21",
                    "game": "loto_540",
                    "ticket_id": "t-02",
                    "best_main_match": 2,
                    "side_game_match": 0,
                    "side_game_digits_matched": 1,
                    "cost_ron": 22.5,
                },
            ]
            append_detail_rows(path, rows)
            with path.open() as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            parsed = [json.loads(ln) for ln in lines]
            self.assertEqual(parsed[0]["ticket_id"], "t-01")
            self.assertEqual(parsed[1]["best_main_match"], 2)

    def test_appends_preserve_existing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "picks_detail.jsonl"
            append_detail_rows(path, [{"draw_date": "2026-04-21", "ticket_id": "a"}])
            append_detail_rows(path, [{"draw_date": "2026-04-24", "ticket_id": "b"}])
            with path.open() as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["ticket_id"], "a")
            self.assertEqual(json.loads(lines[1])["ticket_id"], "b")

    def test_empty_rows_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "picks_detail.jsonl"
            append_detail_rows(path, [])
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
