"""Database persistence for generation and result-check workflows.

DB writes are optional and controlled by environment:
- DATABASE_URL: PostgreSQL or sqlite URL
- RESULTS_DB_PATH: sqlite file path fallback
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _to_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _resolve_database_dsn() -> str | None:
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return dsn

    sqlite_path = os.environ.get("RESULTS_DB_PATH")
    if sqlite_path:
        return f"sqlite:///{sqlite_path}"
    return None


class _ResultsDB:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.kind, self.connection = self._connect(dsn)

    @staticmethod
    def _connect(dsn: str):
        if dsn.startswith("postgresql://") or dsn.startswith("postgres://"):
            try:
                import psycopg
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL DATABASE_URL configured but psycopg is not installed. "
                    "Install with: pip install psycopg[binary]"
                ) from exc

            conn = psycopg.connect(dsn)
            return "postgres", conn

        if dsn.startswith("sqlite:///"):
            sqlite_path = dsn[len("sqlite:///"):]
        else:
            sqlite_path = dsn

        path = Path(sqlite_path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys = ON")
        return "sqlite", conn

    @property
    def placeholder(self) -> str:
        return "%s" if self.kind == "postgres" else "?"

    def close(self) -> None:
        self.connection.close()

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> None:
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql, params)
        finally:
            cursor.close()

    def insert(self, table: str, row: dict[str, Any]) -> None:
        columns = list(row.keys())
        placeholders = ", ".join(self.placeholder for _ in columns)
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        values = [row[column] for column in columns]
        self.execute(sql, values)

    def ensure_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS workflow_runs (
                id TEXT PRIMARY KEY,
                workflow_name TEXT NOT NULL,
                github_run_id TEXT,
                git_sha TEXT,
                status TEXT NOT NULL,
                conclusion TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                inputs_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS generation_runs (
                id TEXT PRIMARY KEY,
                workflow_run_id TEXT NOT NULL,
                budget REAL NOT NULL,
                seed BIGINT,
                ev_gate INTEGER NOT NULL,
                ev_min_ratio REAL NOT NULL,
                jackpots_json TEXT NOT NULL,
                allocation_json TEXT NOT NULL,
                gate_details_json TEXT NOT NULL,
                p_any_win REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(workflow_run_id) REFERENCES workflow_runs(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS generated_tickets (
                id TEXT PRIMARY KEY,
                generation_run_id TEXT NOT NULL,
                game TEXT NOT NULL,
                strategy TEXT NOT NULL,
                line_no INTEGER NOT NULL,
                main_numbers_json TEXT NOT NULL,
                joker_number INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(generation_run_id) REFERENCES generation_runs(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS check_runs (
                id TEXT PRIMARY KEY,
                workflow_run_id TEXT NOT NULL,
                max_retries INTEGER NOT NULL,
                retry_delay_minutes INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(workflow_run_id) REFERENCES workflow_runs(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS check_results (
                id TEXT PRIMARY KEY,
                check_run_id TEXT NOT NULL,
                game TEXT NOT NULL,
                draw_date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                picks_count INTEGER NOT NULL,
                total_matches INTEGER NOT NULL,
                best_match INTEGER NOT NULL,
                match_total INTEGER NOT NULL,
                winning_numbers TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(check_run_id) REFERENCES check_runs(id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_generation_runs_workflow ON generation_runs(workflow_run_id)",
            "CREATE INDEX IF NOT EXISTS idx_generated_tickets_generation ON generated_tickets(generation_run_id)",
            "CREATE INDEX IF NOT EXISTS idx_check_runs_workflow ON check_runs(workflow_run_id)",
            "CREATE INDEX IF NOT EXISTS idx_check_results_check_run ON check_results(check_run_id)",
        ]
        for statement in statements:
            self.execute(statement)

        # Backward-compatible fix for existing Postgres schemas where seed was INT4.
        if self.kind == "postgres":
            self.execute(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'generation_runs'
                          AND column_name = 'seed'
                          AND data_type = 'integer'
                    ) THEN
                        ALTER TABLE generation_runs ALTER COLUMN seed TYPE BIGINT;
                    END IF;
                END $$;
                """
            )


def _new_id() -> str:
    return uuid.uuid4().hex


def _persist_guard():
    dsn = _resolve_database_dsn()
    if not dsn:
        return None
    db = _ResultsDB(dsn)
    db.ensure_schema()
    return db


def persist_generation_run(
    *,
    budget: float,
    seed: int | None,
    ev_gate: bool,
    ev_min_ratio: float,
    jackpots: dict[str, float | None],
    allocation: dict[str, Any],
    gate_details: dict[str, Any],
    tickets: list[dict[str, Any]],
) -> bool:
    """Persist one generation execution and emitted tickets."""
    db = None
    try:
        db = _persist_guard()
        if db is None:
            return False

        now = _utc_now_iso()
        workflow_run_id = _new_id()
        generation_run_id = _new_id()

        db.insert(
            "workflow_runs",
            {
                "id": workflow_run_id,
                "workflow_name": "generate",
                "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                "git_sha": os.environ.get("GITHUB_SHA"),
                "status": "completed",
                "conclusion": "success",
                "started_at": now,
                "finished_at": now,
                "inputs_json": _to_json(
                    {
                        "budget": budget,
                        "seed": seed,
                        "ev_gate": ev_gate,
                        "ev_min_ratio": ev_min_ratio,
                        "jackpots": jackpots,
                    }
                ),
                "created_at": now,
            },
        )

        db.insert(
            "generation_runs",
            {
                "id": generation_run_id,
                "workflow_run_id": workflow_run_id,
                "budget": budget,
                "seed": seed,
                "ev_gate": 1 if ev_gate else 0,
                "ev_min_ratio": ev_min_ratio,
                "jackpots_json": _to_json(jackpots),
                "allocation_json": _to_json(allocation),
                "gate_details_json": _to_json(gate_details),
                "p_any_win": float(allocation.get("p_any_win", 0.0) or 0.0),
                "created_at": now,
            },
        )

        for ticket in tickets:
            db.insert(
                "generated_tickets",
                {
                    "id": _new_id(),
                    "generation_run_id": generation_run_id,
                    "game": str(ticket.get("game", "")),
                    "strategy": str(ticket.get("strategy", "recommended")),
                    "line_no": int(ticket.get("line_no", 0)),
                    "main_numbers_json": _to_json(ticket.get("main_numbers", [])),
                    "joker_number": ticket.get("joker_number"),
                    "created_at": now,
                },
            )

        db.commit()
        return True
    except Exception as exc:
        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass
        print(f"WARNING: Could not persist generation run to DB: {exc}", file=sys.stderr)
        return False
    finally:
        if db is not None:
            db.close()


def persist_check_results(
    *,
    max_retries: int,
    retry_delay_minutes: int,
    history_rows: list[tuple[str, str, str, dict[str, dict[str, Any]], int]],
) -> bool:
    """Persist one check-results execution and per-strategy rows."""
    db = None
    try:
        db = _persist_guard()
        if db is None:
            return False

        now = _utc_now_iso()
        workflow_run_id = _new_id()
        check_run_id = _new_id()

        db.insert(
            "workflow_runs",
            {
                "id": workflow_run_id,
                "workflow_name": "check",
                "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                "git_sha": os.environ.get("GITHUB_SHA"),
                "status": "completed",
                "conclusion": "success",
                "started_at": now,
                "finished_at": now,
                "inputs_json": _to_json(
                    {
                        "max_retries": max_retries,
                        "retry_delay_minutes": retry_delay_minutes,
                    }
                ),
                "created_at": now,
            },
        )

        db.insert(
            "check_runs",
            {
                "id": check_run_id,
                "workflow_run_id": workflow_run_id,
                "max_retries": int(max_retries),
                "retry_delay_minutes": int(retry_delay_minutes),
                "created_at": now,
            },
        )

        for draw_date, game_key, winning_str, strategy_results, match_total in history_rows:
            if not strategy_results:
                continue
            for strategy, data in strategy_results.items():
                db.insert(
                    "check_results",
                    {
                        "id": _new_id(),
                        "check_run_id": check_run_id,
                        "game": game_key,
                        "draw_date": draw_date,
                        "strategy": strategy,
                        "picks_count": len(data.get("results", [])),
                        "total_matches": int(data.get("score", 0)),
                        "best_match": int(data.get("best_match", 0)),
                        "match_total": int(match_total),
                        "winning_numbers": winning_str,
                        "created_at": now,
                    },
                )

        db.commit()
        return True
    except Exception as exc:
        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass
        print(f"WARNING: Could not persist check results to DB: {exc}", file=sys.stderr)
        return False
    finally:
        if db is not None:
            db.close()
