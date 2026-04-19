"""EV-gate calibration analyzer.

Reads `data/budget_bank.json` (ledger) and `data/results/picks_detail.jsonl`
(ticket outcomes) to report on whether the current EV-gate thresholds are
behaving reasonably over accumulated production runs.

Produces a markdown report at the chosen output path. Intentionally
limited: we do NOT have historical jackpot-by-draw data, so we can't
retro-simulate "what would have happened if we had played the skipped
draw?". Instead we report observable facts and flag obvious failure modes
(ledger growing unboundedly = skip_ratio too strict; ledger always empty
on draws that look +EV = boost_ratio too loose).

Usage:
    PYTHONPATH=src python scripts/analyze_ev_gate.py
    PYTHONPATH=src python scripts/analyze_ev_gate.py --output docs/2026-plans/ev-gate-calibration.md

Minimum meaningful sample:
    - >= 10 ledger entries to report on skip frequency
    - >= 30 ledger entries to call miscalibration signals significant
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@dataclass
class LedgerSummary:
    total_entries: int
    credits: int
    debits: int
    total_credited: float
    total_debited: float
    balance: float


def load_ledger(path: Path) -> tuple[list[dict], LedgerSummary]:
    if not path.exists():
        return [], LedgerSummary(0, 0, 0, 0.0, 0.0, 0.0)
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    credits = [e for e in entries if e.get("kind") == "credit"]
    debits = [e for e in entries if e.get("kind") == "debit"]
    return entries, LedgerSummary(
        total_entries=len(entries),
        credits=len(credits),
        debits=len(debits),
        total_credited=sum(float(e.get("amount", 0)) for e in credits),
        total_debited=sum(float(e.get("amount", 0)) for e in debits),
        balance=float(data.get("balance", 0.0)),
    )


def load_picks_detail(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # skip corrupt lines, don't fail the whole report
    return rows


def summarize_play_outcomes(rows: list[dict]) -> dict[str, dict]:
    """Per-game: ticket count, prize hit rate, total payout, mean best_match."""
    out: dict[str, dict] = {}
    for row in rows:
        game = row.get("game")
        if not game:
            continue
        g = out.setdefault(
            game,
            {
                "tickets": 0,
                "total_cost": 0.0,
                "total_payout": 0.0,
                "best_match_distribution": {},
                "side_game_hits": 0,
            },
        )
        g["tickets"] += 1
        g["total_cost"] += float(row.get("cost_ron", 0.0))
        g["total_payout"] += float(row.get("payout", 0.0))  # may be absent
        bm = int(row.get("best_main_match", 0))
        g["best_match_distribution"][bm] = g["best_match_distribution"].get(bm, 0) + 1
        if row.get("side_game_match"):
            g["side_game_hits"] += 1
    return out


def _calibration_flags(ledger: LedgerSummary, plays_by_game: dict[str, dict]) -> list[str]:
    flags: list[str] = []
    total_plays = sum(g["tickets"] for g in plays_by_game.values())

    if ledger.total_entries == 0:
        flags.append("No ledger entries yet — analyzer is a no-op until the gate fires at least once.")
        return flags

    if ledger.credits == ledger.total_entries and ledger.total_entries >= 10:
        flags.append(
            "All ledger events are credits (no plays, no boosts). "
            "Either `ev_skip_ratio` is too strict or jackpots in this period never cleared the threshold. "
            "Consider lowering `ev_skip_ratio` or accepting the gate saves 100% of budget in this period."
        )

    if ledger.debits > 0 and ledger.balance < 50 and ledger.total_credited < ledger.total_debited * 2:
        flags.append(
            "Ledger is churning — debits approach credits. "
            "`ev_boost_ratio` may be too permissive; boosts may be firing on draws that aren't clearly +EV."
        )

    if total_plays > 0 and ledger.debits == 0:
        flags.append(
            "Some draws have been played but no boosts have fired. "
            "This is expected when jackpots sit between skip and boost thresholds."
        )

    return flags


def render_report(
    ledger_entries: list[dict],
    ledger: LedgerSummary,
    plays_by_game: dict[str, dict],
) -> str:
    lines: list[str] = ["# EV-gate calibration report", ""]

    lines.append("## Ledger summary")
    lines.append("")
    lines.append(f"- Total entries: **{ledger.total_entries}**")
    lines.append(f"- Skip credits: {ledger.credits}  (total {ledger.total_credited:,.2f} RON)")
    lines.append(f"- Boost debits: {ledger.debits}  (total {ledger.total_debited:,.2f} RON)")
    lines.append(f"- Current balance: **{ledger.balance:,.2f} RON**")
    lines.append("")

    if ledger_entries:
        lines.append("### Last 5 ledger events")
        lines.append("")
        lines.append("| Date | Kind | Amount | Reason |")
        lines.append("|---|---|---|---|")
        for entry in ledger_entries[-5:]:
            lines.append(
                f"| {entry.get('draw_date', '?')} | {entry.get('kind', '?')} | "
                f"{float(entry.get('amount', 0)):,.2f} | {entry.get('reason', '')} |"
            )
        lines.append("")

    lines.append("## Play outcomes (from picks_detail.jsonl)")
    lines.append("")
    if not plays_by_game:
        lines.append("_No played tickets recorded yet._")
    else:
        lines.append("| Game | Tickets | Total cost | Total payout | Net | Side hits |")
        lines.append("|---|---|---|---|---|---|")
        for game in sorted(plays_by_game):
            g = plays_by_game[game]
            net = g["total_payout"] - g["total_cost"]
            lines.append(
                f"| {game} | {g['tickets']} | {g['total_cost']:,.2f} | "
                f"{g['total_payout']:,.2f} | {net:+,.2f} | {g['side_game_hits']} |"
            )
        lines.append("")
        lines.append("### Best-main-match distribution")
        lines.append("")
        for game in sorted(plays_by_game):
            g = plays_by_game[game]
            dist = g["best_match_distribution"]
            dist_str = "  ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
            lines.append(f"- **{game}**: {dist_str}")
        lines.append("")

    flags = _calibration_flags(ledger, plays_by_game)
    lines.append("## Calibration flags")
    lines.append("")
    if not flags:
        lines.append("_No flags — current thresholds look reasonable against accumulated data._")
    else:
        for flag in flags:
            lines.append(f"- ⚠️  {flag}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_Minimum meaningful sample: 10 ledger entries for a sanity read, 30+ "
        "before drawing any calibration conclusion. Below that, treat this as a smoke test._"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="EV-gate calibration analyzer")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/budget_bank.json"),
        help="Ledger JSON path (default: data/budget_bank.json)",
    )
    parser.add_argument(
        "--picks-detail",
        type=Path,
        default=Path("data/results/picks_detail.jsonl"),
        help="Picks-detail JSONL path (default: data/results/picks_detail.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path. If omitted, report is printed to stdout.",
    )
    args = parser.parse_args()

    ledger_entries, ledger = load_ledger(args.ledger)
    plays = load_picks_detail(args.picks_detail)
    plays_by_game = summarize_play_outcomes(plays)
    report = render_report(ledger_entries, ledger, plays_by_game)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
