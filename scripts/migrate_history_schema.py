"""One-shot migration: add ticket-aware columns to data/results/history.csv.

The new columns default to empty strings; this is purely additive and
does not affect any existing consumer of the file. Plans B–D populate
the new columns going forward.

Usage:
    PYTHONPATH=src python scripts/migrate_history_schema.py

Safe to re-run: idempotent.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

NEW_COLUMNS = [
    "ticket_id",
    "variants_count",
    "side_game_match",
    "side_game_digits",
    "winning_side_game",
    "builder_name",
]


def migrate(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_columns = list(reader.fieldnames or [])
        rows = list(reader)

    missing = [c for c in NEW_COLUMNS if c not in existing_columns]
    if not missing:
        return

    all_columns = existing_columns + missing
    for row in rows:
        for col in missing:
            row[col] = ""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    path = Path("data/results/history.csv")
    migrate(path)
    print(f"migrated {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
