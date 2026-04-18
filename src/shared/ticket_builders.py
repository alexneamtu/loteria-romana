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
from .math_utils import softmax
from .pricing import VARIANTS_PER_TICKET, compute_ticket_cost
from .side_games import generate_noroc, generate_noroc_plus, generate_super_noroc
from .ticket import Ticket, Variant
from .wheeling import OptimizedWheelGenerator


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


_CORE_K = {"joker": 3, "loto_649": 4, "loto_540": 3}
_PETAL_M = {"joker": 6, "loto_649": 5, "loto_540": 6}


def _top_k_from_signal(
    ctx: BuilderContext,
    k: int,
    exclude: set[int] | None = None,
) -> list[int]:
    """Return top-k numbers ranked by per-number frequency softmax.

    Uses a simple recent-frequency softmax over the last 200 draws.
    """
    exclude = exclude or set()
    recent = ctx.draws[-200:] if len(ctx.draws) > 200 else ctx.draws
    counts: dict[int, float] = {n: 0.0 for n in ctx.config.pool_range}
    for draw in recent:
        for n in draw:
            if n in counts:
                counts[n] += 1.0

    nums = [n for n in ctx.config.pool_range if n not in exclude]
    weights = [counts[n] for n in nums]
    probs = softmax(weights)

    ranked = sorted(zip(nums, probs), key=lambda p: p[1], reverse=True)
    return [n for n, _ in ranked[:k]]


class CoreShareBuilder:
    """All variants share a top-K core; petals rotate through pool of M."""

    strategy = "core_share"

    def build(self, ctx: BuilderContext) -> list[Ticket]:
        core_k = _CORE_K[ctx.game]
        petal_m = _PETAL_M[ctx.game]
        variants_per = VARIANTS_PER_TICKET[ctx.game]
        per_variant = ctx.config.numbers_to_pick
        if core_k >= per_variant:
            raise ValueError(f"core K={core_k} must be < numbers_to_pick={per_variant}")

        core = _top_k_from_signal(ctx, core_k)
        petal_pool = _top_k_from_signal(ctx, core_k + petal_m, exclude=set(core))[core_k:]
        if not petal_pool:
            petal_pool = _top_k_from_signal(ctx, petal_m, exclude=set(core))

        slots_per_variant = per_variant - core_k

        variants_picks: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        attempts = 0
        while len(variants_picks) < variants_per and attempts < variants_per * 20:
            attempts += 1
            petals = ctx.rng.sample(petal_pool, k=min(slots_per_variant, len(petal_pool)))
            if len(petals) < slots_per_variant:
                extra_pool = [
                    n for n in ctx.config.pool_range
                    if n not in core and n not in petals
                ]
                petals.extend(ctx.rng.sample(extra_pool, slots_per_variant - len(petals)))
            full = tuple(sorted(core + petals))
            if full in seen:
                continue
            seen.add(full)
            variants_picks.append(list(full))

        while len(variants_picks) < variants_per:
            extras = ctx.rng.sample(list(ctx.config.pool_range), per_variant)
            variants_picks.append(sorted(set(extras))[:per_variant])

        variants = _make_variants(ctx, variants_picks)
        return [_make_ticket(ctx, variants, self.strategy)]


class WheelBuilder:
    """Abbreviated covering wheel over a signal-ranked pool.

    pool_size must be > numbers_per_variant; builder picks top
    `pool_size` numbers from the softmax signal and wheels them into
    VARIANTS_PER_TICKET[game] variants using a greedy 3-guarantee
    covering algorithm.
    """

    strategy = "wheel"

    def __init__(self, pool_size: int, guarantee: int = 3):
        self.pool_size = pool_size
        self.guarantee = guarantee

    def build(self, ctx: BuilderContext) -> list[Ticket]:
        per_variant = ctx.config.numbers_to_pick
        if self.pool_size <= per_variant:
            raise ValueError(
                f"pool_size={self.pool_size} must exceed per_variant={per_variant}"
            )

        pool = _top_k_from_signal(ctx, self.pool_size)
        gen = OptimizedWheelGenerator(per_variant, self.guarantee)
        variants_per = VARIANTS_PER_TICKET[ctx.game]
        wheel_lines = gen.generate(pool, max_tickets=variants_per)

        if len(wheel_lines) < variants_per:
            # Pad with randomized pool subsets
            existing = {tuple(sorted(line)) for line in wheel_lines}
            while len(wheel_lines) < variants_per:
                extra = tuple(sorted(ctx.rng.sample(pool, per_variant)))
                if extra in existing:
                    continue
                existing.add(extra)
                wheel_lines.append(list(extra))

        variants = _make_variants(ctx, wheel_lines[:variants_per])
        return [_make_ticket(ctx, variants, self.strategy)]
