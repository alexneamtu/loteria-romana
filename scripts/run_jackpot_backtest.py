"""Run the ticket backtester over historical draws and emit a report.

Usage:
    PYTHONPATH=src python scripts/run_jackpot_backtest.py
    PYTHONPATH=src python scripts/run_jackpot_backtest.py --output docs/2026-plans/jackpot-backtest.md --seed 42
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shared.game_config import JOKER_CONFIG, LOTO_540_CONFIG, LOTO_649_CONFIG
from shared.ticket_backtester import TicketBacktester
from shared.ticket_builders import (
    CoreShareBuilder,
    IndependentBuilder,
    WheelBuilder,
)


GAMES = {
    "Joker": {
        "key": "joker",
        "config": JOKER_CONFIG,
        "csv": "data/clean/joker_draws.csv",
    },
    "Loto 6/49": {
        "key": "loto_649",
        "config": LOTO_649_CONFIG,
        "csv": "data/clean/loto_649_draws.csv",
    },
    "Loto 5/40": {
        "key": "loto_540",
        "config": LOTO_540_CONFIG,
        "csv": "data/clean/loto_540_draws.csv",
    },
}


def _load_draws(csv_path: str, game_key: str):
    p = Path(csv_path)
    if not p.exists():
        return [], [], None
    if game_key == "joker":
        from joker_model.storage import load_draws
        draws = load_draws(p)
        return (
            [d.main_numbers for d in draws],
            [d.date for d in draws],
            [d.joker for d in draws],
        )
    if game_key == "loto_649":
        from loto_649_model.storage import load_draws
        draws = load_draws(p)
        return [d.main_numbers for d in draws], [d.date for d in draws], None
    from loto_540_model.storage import load_draws
    draws = load_draws(p)
    return [d.main_numbers for d in draws], [d.date for d in draws], None


def run(output_path: Path, seed: int, warmup: int, jackpot: float) -> None:
    lines: list[str] = ["# Jackpot Backtest Report", ""]
    lines.append(f"seed={seed} warmup={warmup} jackpot={jackpot:.0f} RON")
    lines.append("")

    builders = {
        "IndependentBuilder": IndependentBuilder(n_tickets=1),
        "CoreShareBuilder": CoreShareBuilder(),
        "WheelBuilder": WheelBuilder(pool_size=8),
    }

    for game_label, meta in GAMES.items():
        lines.append(f"## {game_label}")
        lines.append("")
        main, dates, bonuses = _load_draws(meta["csv"], meta["key"])
        if not main:
            lines.append("_no historical data_")
            lines.append("")
            continue
        if len(main) <= warmup:
            local_warmup = max(5, len(main) // 2)
            lines.append(f"_limited history: using warmup={local_warmup}_")
            lines.append("")
        else:
            local_warmup = warmup

        lines.append(
            "| builder | n | median_roi | mean_roi | P(>=3) | P(>=4) | P(>=5) | skewness |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for name, builder in builders.items():
            bt = TicketBacktester(
                game=meta["key"],
                config=meta["config"],
                draws=main,
                draw_dates=dates,
                warmup=local_warmup,
                rng=random.Random(seed),
                joker_bonuses=bonuses,
                jackpot=jackpot,
            )
            try:
                outcomes = bt.run(builder)
                s = bt.summarize(outcomes)
                lines.append(
                    f"| {name} | {int(s['n'])} | {s['median_roi']:+.2f} | "
                    f"{s['mean_roi']:+.2f} | {s['p_best_match_ge_3']:.3%} | "
                    f"{s['p_best_match_ge_4']:.3%} | {s['p_best_match_ge_5']:.3%} | "
                    f"{s['skewness']:+.3f} |"
                )
            except Exception as exc:  # pragma: no cover
                lines.append(f"| {name} | error | {exc} | | | | | |")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/2026-plans/jackpot-backtest.md"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--jackpot", type=float, default=1_000_000.0)
    args = parser.parse_args()
    run(args.output, args.seed, args.warmup, args.jackpot)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
