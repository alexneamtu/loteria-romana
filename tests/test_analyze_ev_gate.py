import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_ev_gate import (
    LedgerSummary,
    _calibration_flags,
    load_ledger,
    load_picks_detail,
    render_report,
    summarize_play_outcomes,
)


def _write_ledger(path: Path, entries: list[dict], balance: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"balance": balance, "entries": entries}))


def _write_picks_detail(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class TestLoadLedger(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        entries, summary = load_ledger(Path("/nonexistent"))
        self.assertEqual(entries, [])
        self.assertEqual(summary.total_entries, 0)

    def test_summarizes_credits_and_debits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.json"
            _write_ledger(
                path,
                [
                    {"kind": "credit", "amount": 70.0, "draw_date": "2026-04-20", "reason": "low"},
                    {"kind": "credit", "amount": 70.0, "draw_date": "2026-04-24", "reason": "low"},
                    {"kind": "debit", "amount": 50.0, "draw_date": "2026-04-27", "reason": "boost"},
                ],
                balance=90.0,
            )
            entries, summary = load_ledger(path)
            self.assertEqual(summary.total_entries, 3)
            self.assertEqual(summary.credits, 2)
            self.assertEqual(summary.debits, 1)
            self.assertEqual(summary.total_credited, 140.0)
            self.assertEqual(summary.total_debited, 50.0)
            self.assertEqual(summary.balance, 90.0)


class TestLoadPicksDetail(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        self.assertEqual(load_picks_detail(Path("/nonexistent")), [])

    def test_loads_jsonl_and_skips_corrupt_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "picks.jsonl"
            path.write_text(
                json.dumps({"game": "joker", "best_main_match": 3}) + "\n"
                + "not-json\n"
                + json.dumps({"game": "loto_649", "best_main_match": 2}) + "\n"
            )
            rows = load_picks_detail(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["game"], "joker")


class TestSummarizePlayOutcomes(unittest.TestCase):
    def test_aggregates_per_game(self):
        rows = [
            {"game": "joker", "cost_ron": 17.5, "payout": 0, "best_main_match": 1, "side_game_match": 0},
            {"game": "joker", "cost_ron": 17.5, "payout": 14, "best_main_match": 3, "side_game_match": 0},
            {"game": "loto_649", "cost_ron": 28.5, "payout": 80, "best_main_match": 4, "side_game_match": 1},
        ]
        by_game = summarize_play_outcomes(rows)
        self.assertEqual(by_game["joker"]["tickets"], 2)
        self.assertEqual(by_game["joker"]["total_cost"], 35.0)
        self.assertEqual(by_game["joker"]["total_payout"], 14)
        self.assertEqual(by_game["loto_649"]["side_game_hits"], 1)
        self.assertEqual(by_game["joker"]["best_match_distribution"], {1: 1, 3: 1})


class TestCalibrationFlags(unittest.TestCase):
    def test_empty_ledger_produces_one_flag(self):
        flags = _calibration_flags(LedgerSummary(0, 0, 0, 0.0, 0.0, 0.0), {})
        self.assertTrue(any("No ledger entries" in f for f in flags))

    def test_all_credits_triggers_strict_flag(self):
        summary = LedgerSummary(
            total_entries=12, credits=12, debits=0,
            total_credited=840.0, total_debited=0.0, balance=840.0,
        )
        flags = _calibration_flags(summary, {})
        self.assertTrue(any("too strict" in f for f in flags))

    def test_play_without_boost_is_noted(self):
        summary = LedgerSummary(
            total_entries=5, credits=5, debits=0,
            total_credited=350.0, total_debited=0.0, balance=350.0,
        )
        plays = {"joker": {"tickets": 3, "total_cost": 52.5, "total_payout": 0, "best_match_distribution": {}, "side_game_hits": 0}}
        flags = _calibration_flags(summary, plays)
        self.assertTrue(any("no boosts have fired" in f for f in flags))


class TestRenderReport(unittest.TestCase):
    def test_renders_key_sections(self):
        entries = [{"kind": "credit", "amount": 70.0, "draw_date": "2026-04-20", "reason": "low ratios"}]
        summary = LedgerSummary(1, 1, 0, 70.0, 0.0, 70.0)
        plays = {}
        report = render_report(entries, summary, plays)
        self.assertIn("EV-gate calibration report", report)
        self.assertIn("Ledger summary", report)
        self.assertIn("70.00 RON", report)
        self.assertIn("Calibration flags", report)

    def test_includes_play_table_when_plays_present(self):
        summary = LedgerSummary(1, 1, 0, 70.0, 0.0, 70.0)
        plays = {"joker": {"tickets": 1, "total_cost": 17.5, "total_payout": 14.0, "best_match_distribution": {3: 1}, "side_game_hits": 0}}
        report = render_report([], summary, plays)
        self.assertIn("| Game |", report)
        self.assertIn("joker", report)
        self.assertIn("17.50", report)


if __name__ == "__main__":
    unittest.main()
