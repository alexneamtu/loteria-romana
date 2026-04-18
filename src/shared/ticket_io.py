"""Serialize Ticket objects to and from picks/tickets.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ticket import Ticket, Variant


def _variant_to_dict(v: Variant) -> dict[str, Any]:
    return {
        "main_numbers": list(v.main_numbers),
        "bonus_number": v.bonus_number,
    }


def _variant_from_dict(d: dict[str, Any], game: str) -> Variant:
    return Variant(
        main_numbers=tuple(d["main_numbers"]),
        bonus_number=d["bonus_number"],
        game=game,
    )


def _ticket_to_dict(t: Ticket) -> dict[str, Any]:
    return {
        "game": t.game,
        "variants": [_variant_to_dict(v) for v in t.variants],
        "side_game_number": t.side_game_number,
        "strategy": t.strategy,
        "cost_ron": t.cost_ron,
    }


def _ticket_from_dict(d: dict[str, Any]) -> Ticket:
    game = d["game"]
    return Ticket(
        game=game,
        variants=tuple(_variant_from_dict(v, game) for v in d["variants"]),
        side_game_number=d["side_game_number"],
        strategy=d["strategy"],
        cost_ron=float(d["cost_ron"]),
    )


def dump_tickets(
    path: Path,
    tickets: list[Ticket],
    *,
    budget_ron: float,
    total_cost_ron: float,
    allocation: dict[str, int],
    generated_at: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_at": generated_at,
        "budget_ron": budget_ron,
        "total_cost_ron": total_cost_ron,
        "allocation": allocation,
        "tickets": [_ticket_to_dict(t) for t in tickets],
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def load_tickets(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt tickets.json at {path}: {exc}") from exc
    return {
        **doc,
        "tickets": [_ticket_from_dict(d) for d in doc.get("tickets", [])],
    }
