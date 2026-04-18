import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_results
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


class TestMainDBPersistenceHook(unittest.TestCase):
    def test_main_calls_db_persistence_with_history_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            picks_dir = Path(tmpdir) / "picks"
            results_dir = Path(tmpdir) / "results"
            history_csv = Path(tmpdir) / "history.csv"
            picks_dir.mkdir(parents=True, exist_ok=True)

            game_payload = (
                "2026-02-18",
                "1, 2, 3, 4, 5",
                "msg",
                {
                    "recommended": {
                        "results": [{"pick": [1, 2, 3, 4, 5], "matched": [1], "count": 1}],
                        "score": 1,
                        "best_match": 1,
                    }
                },
            )

            with mock.patch.object(
                check_results,
                "fetch_results_with_retry",
                return_value=([], [], []),
            ), \
                mock.patch.object(
                    check_results,
                    "process_game",
                    return_value=game_payload,
                ), \
                mock.patch.object(check_results, "append_history") as append_history_mock, \
                mock.patch.object(check_results, "persist_check_results", return_value=True) as persist_mock, \
                mock.patch.dict(
                    os.environ,
                    {
                        "PICKS_DIR": str(picks_dir),
                        "RESULTS_DIR": str(results_dir),
                        "HISTORY_CSV": str(history_csv),
                        "MAX_RETRIES": "3",
                        "RETRY_DELAY_MINUTES": "5",
                    },
                    clear=False,
                ):
                check_results.main()

            self.assertEqual(append_history_mock.call_count, 1)
            self.assertEqual(persist_mock.call_count, 1)
            _, kwargs = persist_mock.call_args
            self.assertEqual(kwargs["max_retries"], 3)
            self.assertEqual(kwargs["retry_delay_minutes"], 5)
            self.assertEqual(len(kwargs["history_rows"]), 3)


class TestCheckResultsTicketsJSON(unittest.TestCase):
    def test_parse_tickets_json(self):
        import json
        import tempfile
        from scripts.check_results import parse_tickets_json
        doc = {
            "generated_at": "x",
            "budget_ron": 70.0,
            "total_cost_ron": 17.5,
            "allocation": {"joker": 1, "loto_649": 0, "loto_540": 0},
            "tickets": [{
                "game": "joker",
                "variants": [
                    {"main_numbers": [3, 7, 12, 19, 28], "bonus_number": 11},
                    {"main_numbers": [3, 7, 12, 19, 33], "bonus_number": 11},
                ],
                "side_game_number": "NP07",
                "strategy": "core_share",
                "cost_ron": 17.5,
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tickets.json"
            path.write_text(json.dumps(doc))
            tickets = parse_tickets_json(path)
            self.assertEqual(len(tickets), 1)
            self.assertEqual(tickets[0].game, "joker")
            self.assertEqual(tickets[0].side_game_number, "NP07")

    def test_score_ticket_full_side_game_match_joker(self):
        from scripts.check_results import score_side_game_match
        self.assertEqual(score_side_game_match("joker", "NP07", "NP07"), (1, 1))
        self.assertEqual(score_side_game_match("joker", "NP07", "NP14"), (0, 0))
        self.assertEqual(score_side_game_match("joker", "NP07", None), (0, 0))

    def test_score_ticket_noroc_partial_digit_match(self):
        from scripts.check_results import score_side_game_match
        # last 4 digits match
        self.assertEqual(score_side_game_match("loto_649", "1234567", "9994567"), (0, 4))

    def test_score_ticket_noroc_exact_match(self):
        from scripts.check_results import score_side_game_match
        self.assertEqual(score_side_game_match("loto_649", "1234567", "1234567"), (1, 7))

    def test_score_ticket_noroc_no_match(self):
        from scripts.check_results import score_side_game_match
        self.assertEqual(score_side_game_match("loto_649", "1234567", "9999999"), (0, 0))

    def test_score_ticket_super_noroc_partial(self):
        from scripts.check_results import score_side_game_match
        self.assertEqual(score_side_game_match("loto_540", "123456", "999456"), (0, 3))


class TestCheckResultsWritesJSONL(unittest.TestCase):
    def test_write_picks_detail_appends_rows(self):
        import tempfile
        from pathlib import Path
        from scripts.check_results import write_picks_detail

        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "picks_detail.jsonl"
            write_picks_detail(
                jsonl_path=jsonl,
                rows=[{
                    "draw_date": "2026-04-21",
                    "game": "joker",
                    "ticket_id": "tkt-1",
                    "strategy": "core_share",
                    "best_main_match": 3,
                    "variants": [{"main_numbers": [3, 7, 12, 19, 28], "bonus_number": 11}],
                    "side_game_number": "NP07",
                    "winning_side_game": "NP14",
                    "side_game_match": 0,
                    "side_game_digits_matched": 0,
                    "cost_ron": 17.5,
                }],
            )
            self.assertTrue(jsonl.exists())
            text = jsonl.read_text().strip()
            self.assertIn("tkt-1", text)
            self.assertIn("core_share", text)
