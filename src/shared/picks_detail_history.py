"""Append-only picks_detail.jsonl — full per-ticket check results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_detail_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
