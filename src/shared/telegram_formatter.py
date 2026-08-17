"""Telegram message builders for ticket-aware workflow output.

Consumers (GitHub Actions) call emit_messages_for_workflow with the
path to tickets.json and the draw date; they receive an ordered list
of Telegram-Markdown messages to send. Each message should fit under
Telegram's 4096-char limit (tickets max out around ~300 chars).
"""
from __future__ import annotations

import json
from pathlib import Path

from .pricing import VARIANTS_PER_TICKET
from .ticket_allocator import _p_any_win, _p_ticket_any_win

# Telegram caps messages at 4096 chars; leave room for the fence and header.
_MAX_MSG_CHARS = 3800


def _odds(p: float) -> str:
    return f"1 in {round(1 / p):,}" if p > 0 else "—"

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
    """One message per game: the bulletins to fill in, in buy order.

    Each block is one physical ticket — the variant lines and the side-game
    number go on the same bulletin. Split across messages when a game's
    tickets exceed Telegram's size limit.
    """
    by_game: dict[str, list[dict]] = {}
    for t in tickets:
        by_game.setdefault(t["game"], []).append(t)

    messages: list[str] = []
    for game, group in _by_best_odds(by_game):
        per_ticket = VARIANTS_PER_TICKET[game]
        strategies = sorted({t["strategy"] for t in group})
        header = [
            f"{_GAME_EMOJI[game]} *{_GAME_LABEL[game]} — buy {len(group)} ticket(s)* - {draw_date}",
            f"Each bulletin: {per_ticket} variants + {_SIDE_NAME[game]}"
            f" = {group[0]['cost_ron']} RON",
            f"Total: {sum(t['cost_ron'] for t in group):.2f} RON"
            f" · built by {', '.join(strategies)}",
            f"Any prize: {_odds(_p_ticket_any_win(game))} per ticket",
        ]
        # A ticket that does not fill the bulletin cannot be bought as printed.
        wrong = [i for i, t in enumerate(group, 1) if len(t["variants"]) != per_ticket]
        if wrong:
            header.append(f"⚠️ ticket(s) {wrong} do not have {per_ticket} variants")

        blocks = []
        for i, t in enumerate(group, start=1):
            blocks.append("\n".join([
                f"#{i}",
                *(f" V{j} {_format_variant(game, v)}"
                  for j, v in enumerate(t["variants"], start=1)),
                f" {_SIDE_NAME[game]}: {t['side_game_number']}",
            ]))
        messages.extend(_chunk("\n".join(header), blocks))
    return messages


def _chunk(header: str, blocks: list[str]) -> list[str]:
    """Pack blocks into fenced messages under Telegram's size limit."""
    out, cur = [], []
    for b in blocks:
        candidate = cur + [b]
        if cur and len(header) + sum(len(x) + 1 for x in candidate) + 8 > _MAX_MSG_CHARS:
            out.append("\n".join([header, "```", *cur, "```"]))
            cur = [b]
        else:
            cur = candidate
    if cur:
        out.append("\n".join([header, "```", *cur, "```"]))
    return out


def format_summary(
    tickets: list[dict],
    *,
    budget_ron: float,
    total_cost_ron: float,
    draw_date: str,
) -> str:
    games: dict[str, list[dict]] = {}
    for t in tickets:
        games.setdefault(t["game"], []).append(t)
    header = f"🎰 *Lottery Picks - {draw_date}*"
    totals = f"Budget: {budget_ron} RON\nSpent:  {total_cost_ron} RON"
    if not games:
        return "\n".join([header, "", totals, "", "_no tickets (EV gate skipped this draw)_"])

    body = "\n".join(
        f"- {_DISPLAY_GAME[g]}: {len(ts)} × ({VARIANTS_PER_TICKET[g]} variants"
        f" + {_SIDE_NAME[g]}) = {sum(t['cost_ron'] for t in ts):.2f} RON"
        f" · any prize {_odds(_p_ticket_any_win(g))}/ticket"
        for g, ts in _by_best_odds(games)
    )
    p_total = _p_any_win({g: len(ts) for g, ts in games.items()})
    chance = f"P(win anything this draw): {p_total:.1%}"
    return "\n".join([header, "", totals, "", "Buy (best odds first):", body, "", chance])


def _by_best_odds(games: dict[str, list[dict]]) -> list[tuple[str, list[dict]]]:
    return sorted(games.items(), key=lambda kv: -_p_ticket_any_win(kv[0]))


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
