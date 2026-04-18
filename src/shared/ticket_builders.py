"""Ticket builders — produce Ticket objects from game context.

A ticket carries N variants (per VARIANTS_PER_TICKET) + a side-game
number. Builders differ in how they correlate the variants:

- IndependentBuilder: each variant is an independent blended-picks draw.
  Status-quo baseline; no jackpot tilt.
- CoreShareBuilder: all variants share a top-K core from the softmax
  signal; remaining slots permute over a larger pool. High variance.
- WheelBuilder: abbreviated covering wheel over a pool of K numbers
  chosen from the top of the softmax; guarantees N-match coverage if
  enough pool numbers are drawn. Atomic — must consume a whole budget
  slot for the target game.

Each builder consumes a BuilderContext (game config, draws, rng) and
emits a list of complete Ticket objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from .ensemble_blend import generate_blended_picks
from .game_config import GameConfig
from .joker_set_optimizer import assign_max_coverage_jokers
from .pricing import VARIANTS_PER_TICKET, compute_ticket_cost
from .side_games import generate_noroc, generate_noroc_plus, generate_super_noroc
from .ticket import Ticket, Variant


_SIDE_GAME_GENERATORS = {
    "joker": generate_noroc_plus,
    "loto_649": generate_noroc,
    "loto_540": generate_super_noroc,
}

_JOKER_BONUS_POOL = 20


@dataclass(frozen=True)
class BuilderContext:
    game: str
    config: GameConfig
    draws: list[list[int]]
    draw_dates: list[str] | None
    rng: Random


class TicketBuilder(Protocol):
    def build(self, ctx: BuilderContext) -> list[Ticket]: ...


def _make_variants(
    ctx: BuilderContext,
    main_lines: list[list[int]],
) -> list[Variant]:
    variants: list[Variant] = []
    if ctx.game == "joker":
        jokers = assign_max_coverage_jokers(
            count=len(main_lines),
            rng=ctx.rng,
            joker_pool=_JOKER_BONUS_POOL,
        )
        for main, j in zip(main_lines, jokers):
            variants.append(Variant(tuple(sorted(main)), j, ctx.game))
    else:
        for main in main_lines:
            variants.append(Variant(tuple(sorted(main)), None, ctx.game))
    return variants


def _make_ticket(
    ctx: BuilderContext,
    variants: list[Variant],
    strategy: str,
) -> Ticket:
    side_gen = _SIDE_GAME_GENERATORS[ctx.game]
    return Ticket(
        game=ctx.game,
        variants=tuple(variants),
        side_game_number=side_gen(ctx.rng),
        strategy=strategy,
        cost_ron=compute_ticket_cost(ctx.game),
    )


class IndependentBuilder:
    """Baseline: each variant is an independent blended pick."""

    strategy = "independent"

    def __init__(self, n_tickets: int = 1):
        if n_tickets < 1:
            raise ValueError("n_tickets must be >= 1")
        self.n_tickets = n_tickets

    def build(self, ctx: BuilderContext) -> list[Ticket]:
        variants_per = VARIANTS_PER_TICKET[ctx.game]
        total_lines = self.n_tickets * variants_per

        picks = generate_blended_picks(
            ctx.config,
            ctx.draws,
            total_lines,
            ctx.rng,
            draw_dates=ctx.draw_dates,
        )

        tickets: list[Ticket] = []
        for i in range(self.n_tickets):
            slice_ = picks[i * variants_per : (i + 1) * variants_per]
            variants = _make_variants(ctx, slice_)
            tickets.append(_make_ticket(ctx, variants, self.strategy))
        return tickets
