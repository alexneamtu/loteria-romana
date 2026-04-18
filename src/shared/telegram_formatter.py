"""Telegram message builders for ticket-aware workflow output.

Consumers (GitHub Actions) call emit_messages_for_workflow with the
path to tickets.json and the draw date; they receive an ordered list
of Telegram-Markdown messages to send. Each message should fit under
Telegram's 4096-char limit (tickets max out around ~300 chars).
"""
from __future__ import annotations

import json
from pathlib import Path

_GAME_LABEL = {"joker": "JOKER", "loto_649": "LOTO 6/49", "loto_540": "LOTO 5/40"}
_GAME_EMOJI = {"joker": "🃏", "loto_649": "🎱", "loto_540": "🎯"}
_SIDE_NAME = {"joker": "Noroc Plus", "loto_649": "Noroc", "loto_540": "Super Noroc"}
_DISPLAY_GAME = {"joker": "Joker", "loto_649": "Loto 6/49", "loto_540": "Loto 5/40"}


def _format_variant(game: str, v: dict) -> str:
    nums = ", ".join(str(n) for n in v["main_numbers"])
    if game == "joker":
        return f"{nums} + J{v['bonus_number']}"
    return nums


def format_tickets(tickets: list[dict], draw_date: str) -> list[str]:
    messages: list[str] = []
    for t in tickets:
        game = t["game"]
        header = f"{_GAME_EMOJI[game]} *{_GAME_LABEL[game]} Ticket ({t['strategy']}) - {draw_date}*"
        cost_line = f"Cost: {t['cost_ron']} RON"
        body_lines = [
            f"V{i}: {_format_variant(game, v)}"
            for i, v in enumerate(t["variants"], start=1)
        ]
        side_line = f"{_SIDE_NAME[game]}: {t['side_game_number']}"
        text = "\n".join([header, cost_line, "```", *body_lines, side_line, "```"])
        messages.append(text)
    return messages


def format_summary(
    tickets: list[dict],
    *,
    budget_ron: float,
    total_cost_ron: float,
    draw_date: str,
) -> str:
    games_count: dict[str, int] = {}
    for t in tickets:
        games_count[t["game"]] = games_count.get(t["game"], 0) + 1
    if not games_count:
        body = "_no tickets (EV gate skipped this draw)_"
    else:
        body = "\n".join(
            f"- {_DISPLAY_GAME[g]}: {n} ticket(s)"
            for g, n in games_count.items()
        )
    header = f"🎰 *Lottery Picks - {draw_date}*"
    totals = f"Budget: {budget_ron} RON\nSpent:  {total_cost_ron} RON"
    return "\n".join([header, "", totals, "", body])


def emit_messages_for_workflow(tickets_json_path: Path, draw_date: str) -> list[str]:
    if not tickets_json_path.exists():
        return [
            f"🎰 *Lottery Picks - {draw_date}*\n\n_No tickets emitted (skip or error)._"
        ]
    doc = json.loads(tickets_json_path.read_text(encoding="utf-8"))
    summary = format_summary(
        doc["tickets"],
        budget_ron=doc["budget_ron"],
        total_cost_ron=doc["total_cost_ron"],
        draw_date=draw_date,
    )
    ticket_msgs = format_tickets(doc["tickets"], draw_date)
    return [summary, *ticket_msgs]


def format_check_results(
    *,
    game: str,
    draw_date: str,
    winning_main: list[int] | tuple[int, ...],
    winning_side: str | None,
    ticket_outcomes: list[dict],
) -> str:
    header = f"{_GAME_EMOJI[game]} *{_GAME_LABEL[game]} Results - {draw_date}*"
    win_main = ", ".join(str(n) for n in winning_main)
    winning_line = f"Winning: `{win_main}`"
    side_line = f"{_SIDE_NAME[game]}: `{winning_side or '—'}`"
    if not ticket_outcomes:
        return "\n".join([header, winning_line, side_line, "\n_No picks to compare_"])

    parts = [header, winning_line, side_line, ""]
    for out in ticket_outcomes:
        parts.append(f"*{out['strategy']}* (best {out['best_main_match']} main)")
        parts.append("```")
        parts.extend(out.get("variants_rendered", []))
        side_note = (
            f"side: {out['ticket_side']} → {out['side_game_digits_matched']} digits"
            + (" ✅" if out["side_game_match"] else "")
        )
        parts.append(side_note)
        parts.append("```")
    return "\n".join(parts)
