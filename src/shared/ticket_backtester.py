"""Walk-forward ticket backtester.

For each historical draw past the warmup window, trains each builder
on the prefix and scores a single ticket's best match and estimated
payout against the draw.
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .game_config import GameConfig
from .ticket_builders import BuilderContext, TicketBuilder
from .ticket_metrics import (
    TicketOutcome,
    estimate_ticket_payout,
    summarize as _summarize,
)


@dataclass
class TicketBacktester:
    game: str
    config: GameConfig
    draws: list[list[int]]
    draw_dates: list[str] | None
    warmup: int
    rng: Random
    joker_bonuses: list[int] | None = None
    winning_side_games: list[str | None] | None = None
    jackpot: float = 1_000_000.0

    def run(self, builder: TicketBuilder) -> list[TicketOutcome]:
        outcomes: list[TicketOutcome] = []
        for i in range(self.warmup, len(self.draws)):
            prefix = self.draws[:i]
            prefix_dates = self.draw_dates[:i] if self.draw_dates else None
            ctx = BuilderContext(
                game=self.game,
                config=self.config,
                draws=prefix,
                draw_dates=prefix_dates,
                rng=Random(self.rng.random()),
            )
            tickets = builder.build(ctx)
            drawn = self.draws[i]
            bonus = self.joker_bonuses[i] if self.joker_bonuses else None
            side_win = self.winning_side_games[i] if self.winning_side_games else None
            for t in tickets:
                best = t.best_main_match(drawn)
                payout = estimate_ticket_payout(
                    t,
                    winning_main=drawn,
                    winning_joker=bonus,
                    winning_side=side_win,
                    jackpot=self.jackpot,
                )
                outcomes.append(
                    TicketOutcome(payout=payout, cost=t.cost_ron, best_match=best)
                )
        return outcomes

    @staticmethod
    def summarize(outcomes: list[TicketOutcome]) -> dict[str, float]:
        return _summarize(outcomes)
