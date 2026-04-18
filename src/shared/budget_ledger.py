"""File-backed budget ledger for EV-skip/boost accounting.

On a low-EV draw the orchestrator skips play and credits the ledger
with the draw's normal budget. On a rollover / high-EV draw the
orchestrator debits the ledger to increase the effective budget.

Balance is clipped to non-negative; a debit greater than balance
succeeds with a partial amount equal to the current balance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BudgetLedger:
    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"balance": 0.0, "entries": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"balance": 0.0, "entries": []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def balance(self) -> float:
        return float(self._data.get("balance", 0.0))

    def entries(self) -> list[dict[str, Any]]:
        return list(self._data.get("entries", []))

    def credit_skip(self, draw_date: str, amount: float, reason: str) -> None:
        if amount <= 0:
            return
        self._data["balance"] = self.balance() + amount
        self._data["entries"].append(
            {
                "kind": "credit",
                "draw_date": draw_date,
                "amount": amount,
                "reason": reason,
            }
        )
        self._save()

    def debit_boost(self, draw_date: str, amount: float, reason: str) -> float:
        if amount <= 0:
            return 0.0
        actual = min(amount, self.balance())
        if actual <= 0:
            return 0.0
        self._data["balance"] = self.balance() - actual
        self._data["entries"].append(
            {
                "kind": "debit",
                "draw_date": draw_date,
                "amount": actual,
                "requested": amount,
                "reason": reason,
            }
        )
        self._save()
        return actual
