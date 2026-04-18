import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shared.results_db import persist_check_results, persist_generation_run


class TestResultsDBPersistence(unittest.TestCase):
    def test_persist_generation_run_creates_workflow_generation_and_ticket_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "results.db"
            with mock.patch.dict(
                os.environ,
                {
                    "RESULTS_DB_PATH": str(db_path),
                    "GITHUB_RUN_ID": "12345",
                    "GITHUB_SHA": "abc123",
                },
                clear=False,
            ):
                saved = persist_generation_run(
                    budget=40.0,
                    seed=7,
                    ev_gate=True,
                    ev_min_ratio=0.8,
                    jackpots={"joker": 10_000_000.0, "loto_649": None, "loto_540": None},
                    allocation={
                        "tickets": {"joker": 5, "loto_649": 0, "loto_540": 0},
                        "total_cost": 40.0,
                        "p_any_win": 0.03,
                    },
                    gate_details={"joker": {"passes": True, "ratio": 1.2}},
                    tickets=[
                        {
                            "game": "joker",
                            "strategy": "recommended",
                            "line_no": 1,
                            "main_numbers": [1, 2, 3, 4, 5],
                            "joker_number": 7,
                        },
                        {
                            "game": "joker",
                            "strategy": "recommended",
                            "line_no": 2,
                            "main_numbers": [6, 7, 8, 9, 10],
                            "joker_number": 11,
                        },
                    ],
                )

            self.assertTrue(saved)
            with sqlite3.connect(db_path) as conn:
                workflow_count = conn.execute("SELECT COUNT(*) FROM workflow_runs").fetchone()[0]
                generation_count = conn.execute("SELECT COUNT(*) FROM generation_runs").fetchone()[0]
                ticket_count = conn.execute("SELECT COUNT(*) FROM generated_tickets").fetchone()[0]

            self.assertEqual(workflow_count, 1)
            self.assertEqual(generation_count, 1)
            self.assertEqual(ticket_count, 2)

    def test_persist_check_results_creates_rows_per_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "results.db"
            with mock.patch.dict(
                os.environ,
                {
                    "RESULTS_DB_PATH": str(db_path),
                    "GITHUB_RUN_ID": "67890",
                    "GITHUB_SHA": "def456",
                },
                clear=False,
            ):
                saved = persist_check_results(
                    max_retries=3,
                    retry_delay_minutes=5,
                    history_rows=[
                        (
                            "2026-02-18",
                            "joker",
                            "6, 9, 18, 22, 27 + J11",
                            {
                                "recommended": {
                                    "results": [
                                        {"pick": [1, 2, 3, 4, 5], "matched": [3], "count": 1}
                                    ],
                                    "score": 1,
                                    "best_match": 1,
                                },
                                "blend": {
                                    "results": [
                                        {"pick": [6, 7, 8, 9, 10], "matched": [6, 9], "count": 2}
                                    ],
                                    "score": 2,
                                    "best_match": 2,
                                },
                            },
                            5,
                        )
                    ],
                )

            self.assertTrue(saved)
            with sqlite3.connect(db_path) as conn:
                check_runs = conn.execute("SELECT COUNT(*) FROM check_runs").fetchone()[0]
                check_results = conn.execute("SELECT COUNT(*) FROM check_results").fetchone()[0]

            self.assertEqual(check_runs, 1)
            self.assertEqual(check_results, 2)

    def test_returns_false_when_no_database_configuration_is_present(self):
        with mock.patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            self.assertFalse(
                persist_generation_run(
                    budget=10.0,
                    seed=None,
                    ev_gate=False,
                    ev_min_ratio=0.8,
                    jackpots={},
                    allocation={"tickets": {}, "total_cost": 0.0, "p_any_win": 0.0},
                    gate_details={},
                    tickets=[],
                )
            )


class TestResultsDBTicketSchema(unittest.TestCase):
    def test_ensure_schema_adds_ticket_columns(self):
        from shared.results_db import _ResultsDB
        with tempfile.TemporaryDirectory() as tmp:
            dsn = f"sqlite:///{tmp}/test.db"
            db = _ResultsDB(dsn)
            try:
                db.ensure_schema()
                cursor = db.connection.cursor()

                cursor.execute("PRAGMA table_info(generated_tickets)")
                cols = {row[1] for row in cursor.fetchall()}
                self.assertIn("ticket_id", cols)
                self.assertIn("variant_no", cols)
                self.assertIn("side_game_number", cols)
                self.assertIn("cost_ron", cols)
                self.assertIn("builder_name", cols)

                cursor.execute("PRAGMA table_info(check_results)")
                cols = {row[1] for row in cursor.fetchall()}
                self.assertIn("ticket_id", cols)
                self.assertIn("side_game_match", cols)
                self.assertIn("side_game_digits_matched", cols)
                self.assertIn("winning_side_game", cols)

                cursor.execute("PRAGMA table_info(budget_ledger_entries)")
                cols = {row[1] for row in cursor.fetchall()}
                self.assertIn("draw_date", cols)
                self.assertIn("kind", cols)
                self.assertIn("amount", cols)
                self.assertIn("balance_after", cols)
                cursor.close()
            finally:
                db.close()

    def test_ensure_schema_is_idempotent(self):
        from shared.results_db import _ResultsDB
        with tempfile.TemporaryDirectory() as tmp:
            dsn = f"sqlite:///{tmp}/idempotent.db"
            db = _ResultsDB(dsn)
            try:
                db.ensure_schema()
                db.ensure_schema()  # must not error on re-run
                cursor = db.connection.cursor()
                cursor.execute("PRAGMA table_info(generated_tickets)")
                cols = [row[1] for row in cursor.fetchall()]
                # columns appear exactly once
                self.assertEqual(len([c for c in cols if c == "ticket_id"]), 1)
                cursor.close()
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
