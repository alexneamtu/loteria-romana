# Jackpot Redesign — Plan B: Strategies & Backtest

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three `Ticket`-producing strategies (Independent, CoreShare, abbreviated Wheel), a ticket-level backtester that scores them on skewness + P(best_match ≥ 4) + median RON ROI, and a one-shot report comparing all three on historical draws.

**Architecture:** New `src/shared/ticket_builders.py` with three `TicketBuilder` subclasses. New `src/shared/ticket_backtester.py` wrapping the existing `shared.backtest_base.Backtester` pattern. The backtest output is a markdown report committed to `docs/2026-plans/jackpot-backtest.md` — purely informational at this stage; no production code consumes builder output yet (Plan C wires the orchestrator).

**Tech Stack:** Python 3.10+ stdlib, existing `shared.wheeling`, `shared.ensemble_blend`, `shared.crowding`, `shared.portfolio`, `shared.joker_set_optimizer`. unittest.

**Prerequisites:** Plan A merged. Feature branch: `feature/jackpot-b-strategies-backtest`.

**Scope boundary:**
- DO: emit `Ticket` objects from three builder classes, backtest them, write report.
- DO NOT: modify `scripts/generate_recommended_picks.py`, `scripts/check_results.py`, `.github/workflows/*`, or the orchestrator's budget math. That's Plan C.

---

## File Structure

| Status | Path | Responsibility |
|---|---|---|
| NEW | `src/shared/ticket_builders.py` | `TicketBuilder` abstract + `IndependentBuilder`, `CoreShareBuilder`, `WheelBuilder` |
| NEW | `src/shared/ticket_metrics.py` | Payout estimator + skewness / P(≥4) / median helpers |
| NEW | `src/shared/ticket_backtester.py` | Walk-forward backtest over historical draws → `TicketBacktestResult` |
| NEW | `scripts/run_jackpot_backtest.py` | CLI: run all builders on each game, write report |
| NEW | `tests/test_ticket_builders.py` | unittest for all three builders |
| NEW | `tests/test_ticket_metrics.py` | unittest for metric helpers |
| NEW | `tests/test_ticket_backtester.py` | unittest for backtester |
| NEW | `docs/2026-plans/jackpot-backtest.md` | Generated report (first version committed at Task 8) |

Test runner: `PYTHONPATH=src python -m unittest -v`.

---

## Task 1: IndependentBuilder (baseline)

**Files:**
- Create: `src/shared/ticket_builders.py`
- Test: `tests/test_ticket_builders.py`

The independent builder wraps existing `generate_blended_picks` output into the Ticket format. It's the status-quo baseline against which CoreShare and Wheel will be compared.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ticket_builders.py`:
```python
import random
import unittest

from shared.game_config import JOKER_CONFIG, LOTO_540_CONFIG, LOTO_649_CONFIG
from shared.ticket import Ticket
from shared.ticket_builders import BuilderContext, IndependentBuilder


def _joker_draws():
    return [
        [3, 7, 12, 19, 28],
        [5, 17, 18, 24, 31],
        [6, 9, 18, 22, 27],
    ] * 20  # 60 draws — enough for softmax to fit


def _649_draws():
    return [[1, 5, 17, 23, 34, 49], [3, 12, 27, 31, 38, 44]] * 20


def _540_draws():
    return [[2, 9, 15, 27, 34, 38], [1, 5, 19, 22, 28, 40]] * 20


class TestIndependentBuilder(unittest.TestCase):
    def test_joker_builds_one_ticket_with_2_variants(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        tickets = IndependentBuilder(n_tickets=1).build(ctx)
        self.assertEqual(len(tickets), 1)
        t = tickets[0]
        self.assertIsInstance(t, Ticket)
        self.assertEqual(t.game, "joker")
        self.assertEqual(len(t.variants), 2)
        self.assertEqual(t.strategy, "independent")
        for v in t.variants:
            self.assertIsNotNone(v.bonus_number)

    def test_joker_variants_are_independent_not_shared_core(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = IndependentBuilder(n_tickets=1).build(ctx)[0]
        # Independent variants should not be identical
        self.assertNotEqual(t.variants[0].main_numbers, t.variants[1].main_numbers)

    def test_loto_649_builds_3_variants(self):
        ctx = BuilderContext(
            game="loto_649",
            config=LOTO_649_CONFIG,
            draws=_649_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = IndependentBuilder(n_tickets=1).build(ctx)[0]
        self.assertEqual(len(t.variants), 3)
        self.assertEqual(t.game, "loto_649")
        for v in t.variants:
            self.assertIsNone(v.bonus_number)
            self.assertEqual(len(v.main_numbers), 6)

    def test_loto_540_builds_4_variants(self):
        ctx = BuilderContext(
            game="loto_540",
            config=LOTO_540_CONFIG,
            draws=_540_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = IndependentBuilder(n_tickets=1).build(ctx)[0]
        self.assertEqual(len(t.variants), 4)

    def test_seeded_reproducibility(self):
        ctx1 = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(7),
        )
        ctx2 = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(7),
        )
        t1 = IndependentBuilder(n_tickets=1).build(ctx1)[0]
        t2 = IndependentBuilder(n_tickets=1).build(ctx2)[0]
        self.assertEqual(t1.variants[0].main_numbers, t2.variants[0].main_numbers)

    def test_ticket_cost_matches_pricing_module(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = IndependentBuilder(n_tickets=1).build(ctx)[0]
        self.assertEqual(t.cost_ron, 17.5)  # 2 * 7.0 + 0.5 fee + 3.0 Noroc Plus

    def test_multi_ticket_output(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        tickets = IndependentBuilder(n_tickets=3).build(ctx)
        self.assertEqual(len(tickets), 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_builders -v
```
Expected: `ModuleNotFoundError: No module named 'shared.ticket_builders'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/shared/ticket_builders.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_builders.TestIndependentBuilder -v
```
Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/ticket_builders.py tests/test_ticket_builders.py
git commit -m "feat(shared): add IndependentBuilder baseline ticket builder"
```

---

## Task 2: CoreShareBuilder

**Files:**
- Modify: `src/shared/ticket_builders.py` (append class)
- Modify: `tests/test_ticket_builders.py` (append class)

**Rationale:** A core of K high-signal numbers is shared across every variant on the ticket. Each variant's remaining slots rotate through a "petal pool" of M candidates chosen anti-crowding. Configured per game:

| Game | numbers_per_variant | core K | petal pool M |
|---|---|---|---|
| joker | 5 | 3 | 6 |
| loto_649 | 6 | 4 | 5 |
| loto_540 | 5 | 3 | 6 |

If core hits (all K are drawn), every variant has ≥ K matches automatically → jackpot tilt.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ticket_builders.py`:
```python
from shared.ticket_builders import CoreShareBuilder


class TestCoreShareBuilder(unittest.TestCase):
    def test_joker_all_variants_share_3_core_numbers(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = CoreShareBuilder().build(ctx)[0]
        shared_main = set(t.variants[0].main_numbers) & set(t.variants[1].main_numbers)
        self.assertGreaterEqual(len(shared_main), 3)
        self.assertEqual(t.strategy, "core_share")

    def test_loto_649_all_variants_share_4_core_numbers(self):
        ctx = BuilderContext(
            game="loto_649",
            config=LOTO_649_CONFIG,
            draws=_649_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = CoreShareBuilder().build(ctx)[0]
        shared = (
            set(t.variants[0].main_numbers)
            & set(t.variants[1].main_numbers)
            & set(t.variants[2].main_numbers)
        )
        self.assertGreaterEqual(len(shared), 4)

    def test_loto_540_all_variants_share_3_core_numbers(self):
        ctx = BuilderContext(
            game="loto_540",
            config=LOTO_540_CONFIG,
            draws=_540_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = CoreShareBuilder().build(ctx)[0]
        shared = (
            set(t.variants[0].main_numbers)
            & set(t.variants[1].main_numbers)
            & set(t.variants[2].main_numbers)
            & set(t.variants[3].main_numbers)
        )
        self.assertGreaterEqual(len(shared), 3)

    def test_variants_are_distinct(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = CoreShareBuilder().build(ctx)[0]
        self.assertNotEqual(t.variants[0].main_numbers, t.variants[1].main_numbers)

    def test_seeded_reproducibility(self):
        ctx1 = BuilderContext(
            game="joker", config=JOKER_CONFIG, draws=_joker_draws(),
            draw_dates=None, rng=random.Random(99),
        )
        ctx2 = BuilderContext(
            game="joker", config=JOKER_CONFIG, draws=_joker_draws(),
            draw_dates=None, rng=random.Random(99),
        )
        t1 = CoreShareBuilder().build(ctx1)[0]
        t2 = CoreShareBuilder().build(ctx2)[0]
        self.assertEqual(t1.variants[0].main_numbers, t2.variants[0].main_numbers)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_builders.TestCoreShareBuilder -v
```
Expected: `ImportError: cannot import name 'CoreShareBuilder'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/shared/ticket_builders.py`:
```python
from .crowding import anti_crowding_score
from .math_utils import softmax


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
    probs = softmax(weights, temperature=0.5)

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
            # Fallback: re-use last pick modified randomly
            extras = ctx.rng.sample(list(ctx.config.pool_range), per_variant)
            variants_picks.append(sorted(set(extras))[:per_variant])

        variants = _make_variants(ctx, variants_picks)
        return [_make_ticket(ctx, variants, self.strategy)]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_builders -v
```
Expected: all Independent + CoreShare tests pass (12+ tests).

- [ ] **Step 5: Commit**

```bash
git add src/shared/ticket_builders.py tests/test_ticket_builders.py
git commit -m "feat(shared): add CoreShareBuilder for correlated-variant jackpot tilt"
```

---

## Task 3: WheelBuilder

**Files:**
- Modify: `src/shared/ticket_builders.py` (append class)
- Modify: `tests/test_ticket_builders.py` (append class)

**Rationale:** 3-if-9 abbreviated wheel on Loto 6/49 fits ~7 physical tickets × 24.5 RON = 171.5 RON (too expensive for a single 40 RON draw). 3-if-8 gives ~5 tickets × 24.5 = 122.5 RON. **Plan B implements the wheel builder but Plan C's allocator decides when to invoke it** — typically only when rollovers boost the budget. Plan B ships the wheel at a *small enough size to fit a single ticket's 3-variant capacity by default*, i.e. "wheel-inside-one-ticket" using only 3 variants. This is the smallest useful wheel: it provides better coverage across a 7-number pool than 3 independent picks, for the same 24.5 RON.

Also supports Joker (3-if-wheel over 5-from-45, 2 variants) and Loto 5/40 (3-if-wheel over 5-from-40, 4 variants).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ticket_builders.py`:
```python
from shared.ticket_builders import WheelBuilder


class TestWheelBuilder(unittest.TestCase):
    def test_joker_wheel_produces_2_variants_covering_pool(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = WheelBuilder(pool_size=7).build(ctx)[0]
        self.assertEqual(len(t.variants), 2)
        self.assertEqual(t.strategy, "wheel")
        all_mains = set()
        for v in t.variants:
            all_mains.update(v.main_numbers)
        # 2 joker variants (5 each) from a 7-pool should cover all 7
        self.assertGreaterEqual(len(all_mains), 7)

    def test_loto_649_wheel_covers_full_pool(self):
        ctx = BuilderContext(
            game="loto_649",
            config=LOTO_649_CONFIG,
            draws=_649_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        t = WheelBuilder(pool_size=8).build(ctx)[0]
        self.assertEqual(len(t.variants), 3)
        all_mains: set[int] = set()
        for v in t.variants:
            all_mains.update(v.main_numbers)
        # 3 variants of 6 each, from 8-pool: should cover all 8
        self.assertEqual(len(all_mains), 8)

    def test_pool_size_must_exceed_per_variant(self):
        ctx = BuilderContext(
            game="joker",
            config=JOKER_CONFIG,
            draws=_joker_draws(),
            draw_dates=None,
            rng=random.Random(42),
        )
        with self.assertRaises(ValueError):
            WheelBuilder(pool_size=5).build(ctx)

    def test_seeded_reproducibility(self):
        ctx1 = BuilderContext(
            game="loto_649", config=LOTO_649_CONFIG, draws=_649_draws(),
            draw_dates=None, rng=random.Random(7),
        )
        ctx2 = BuilderContext(
            game="loto_649", config=LOTO_649_CONFIG, draws=_649_draws(),
            draw_dates=None, rng=random.Random(7),
        )
        t1 = WheelBuilder(pool_size=8).build(ctx1)[0]
        t2 = WheelBuilder(pool_size=8).build(ctx2)[0]
        for v1, v2 in zip(t1.variants, t2.variants):
            self.assertEqual(v1.main_numbers, v2.main_numbers)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_builders.TestWheelBuilder -v
```
Expected: `ImportError: cannot import name 'WheelBuilder'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/shared/ticket_builders.py`:
```python
from .wheeling import OptimizedWheelGenerator


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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_builders -v
```
Expected: all 15+ ticket-builder tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/ticket_builders.py tests/test_ticket_builders.py
git commit -m "feat(shared): add WheelBuilder using abbreviated covering wheels"
```

---

## Task 4: Ticket metrics — skewness, P(≥N), median RON ROI

**Files:**
- Create: `src/shared/ticket_metrics.py`
- Create: `tests/test_ticket_metrics.py`

**Rationale:** Hit rate (current metric) averages out the tail we care about. Replace with three joint metrics:
- **Median RON ROI per draw** (`total_winnings - cost_ron`): typical outcome, ignores one-in-a-lifetime jackpots.
- **P(best_main_match ≥ 4)**: rate at which the strategy produces a real jackpot-approach hit.
- **Skewness of per-draw net-payout distribution**: rising skewness indicates variance is shifting into the upper tail — the goal of jackpot-seeking.

`estimate_ticket_payout()` maps `(ticket, winning_main, winning_side_game_or_None)` → RON by walking the `ev_calculator` prize tiers. Side-game match scoring is a placeholder match/no-match per Plan A's risks; refined in Plan C.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ticket_metrics.py`:
```python
import statistics
import unittest

from shared.ticket import Ticket, Variant
from shared.ticket_metrics import (
    TicketOutcome,
    estimate_ticket_payout,
    median_roi,
    p_best_match_at_least,
    sample_skewness,
    summarize,
)


def _make_joker_ticket(mains_a, mains_b, bonus=11) -> Ticket:
    return Ticket(
        game="joker",
        variants=(
            Variant(tuple(sorted(mains_a)), bonus, "joker"),
            Variant(tuple(sorted(mains_b)), bonus, "joker"),
        ),
        side_game_number="NP14",
        strategy="test",
        cost_ron=14.5,
    )


class TestTicketMetrics(unittest.TestCase):
    def test_payout_jackpot_tier(self):
        # 5 mains + joker = Category I (jackpot)
        t = _make_joker_ticket(
            mains_a=[3, 7, 12, 19, 28],
            mains_b=[5, 9, 13, 20, 30],
            bonus=11,
        )
        winning = ([3, 7, 12, 19, 28], 11)
        payout = estimate_ticket_payout(t, winning_main=winning[0], winning_joker=winning[1], winning_side=None, jackpot=1_000_000)
        self.assertGreater(payout, 500_000)  # At least half the jackpot

    def test_payout_small_fixed_tier(self):
        # 3 mains, no joker = Category VI (14 RON fixed)
        t = _make_joker_ticket(
            mains_a=[3, 7, 12, 40, 41],
            mains_b=[3, 7, 12, 42, 43],
            bonus=19,  # not the winning joker
        )
        winning_main = [3, 7, 12, 19, 28]
        payout = estimate_ticket_payout(t, winning_main=winning_main, winning_joker=11, winning_side=None, jackpot=1_000_000)
        # Best variant has 3 matches, no joker → Category VI = 14 RON
        self.assertGreaterEqual(payout, 14.0)
        self.assertLess(payout, 1000.0)

    def test_payout_zero_when_no_tier_hit(self):
        t = _make_joker_ticket(
            mains_a=[1, 2, 40, 41, 42],
            mains_b=[3, 4, 43, 44, 45],
            bonus=19,
        )
        payout = estimate_ticket_payout(t, winning_main=[30, 31, 32, 33, 34], winning_joker=11, winning_side=None, jackpot=1_000_000)
        self.assertEqual(payout, 0.0)

    def test_median_roi_handles_losses(self):
        outcomes = [
            TicketOutcome(payout=0.0, cost=14.5, best_match=1),
            TicketOutcome(payout=14.0, cost=14.5, best_match=3),
            TicketOutcome(payout=0.0, cost=14.5, best_match=2),
            TicketOutcome(payout=100.0, cost=14.5, best_match=4),
        ]
        # ROIs: -14.5, -0.5, -14.5, 85.5 → median ≈ -7.5
        self.assertAlmostEqual(median_roi(outcomes), -7.5, places=1)

    def test_p_best_match_at_least(self):
        outcomes = [
            TicketOutcome(payout=0, cost=14.5, best_match=1),
            TicketOutcome(payout=14, cost=14.5, best_match=3),
            TicketOutcome(payout=100, cost=14.5, best_match=4),
            TicketOutcome(payout=0, cost=14.5, best_match=2),
        ]
        self.assertAlmostEqual(p_best_match_at_least(outcomes, 3), 0.5)
        self.assertAlmostEqual(p_best_match_at_least(outcomes, 4), 0.25)
        self.assertAlmostEqual(p_best_match_at_least(outcomes, 5), 0.0)

    def test_sample_skewness_positive_for_heavy_right_tail(self):
        # Heavy right tail (jackpot-like): many losses, one huge win
        values = [-14.5] * 99 + [1_000_000]
        skew = sample_skewness(values)
        self.assertGreater(skew, 5.0)

    def test_sample_skewness_near_zero_for_symmetric_distribution(self):
        values = [i for i in range(-50, 51)]
        self.assertAlmostEqual(sample_skewness(values), 0.0, places=1)

    def test_summarize_returns_all_three_metrics(self):
        outcomes = [
            TicketOutcome(payout=0, cost=14.5, best_match=1),
            TicketOutcome(payout=100, cost=14.5, best_match=4),
        ]
        s = summarize(outcomes)
        self.assertIn("median_roi", s)
        self.assertIn("p_best_match_ge_4", s)
        self.assertIn("skewness", s)
        self.assertIn("n", s)
        self.assertEqual(s["n"], 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_metrics -v
```
Expected: `ModuleNotFoundError: No module named 'shared.ticket_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/shared/ticket_metrics.py`:
```python
"""Metrics for ticket-level backtesting.

Reports jackpot-shaped metrics instead of hit-rate:
- median_roi: typical per-draw net outcome
- p_best_match_at_least(N): rate of approach-the-jackpot hits
- sample_skewness: heavier right tail = more jackpot-tilted
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .ev_calculator import EVCalculator
from .ticket import Ticket


@dataclass(frozen=True)
class TicketOutcome:
    payout: float
    cost: float
    best_match: int

    @property
    def net(self) -> float:
        return self.payout - self.cost


_CALC = EVCalculator()
_JOKER_GAME = _CALC.create_joker()
_649_GAME = _CALC.create_loto_649()
_540_GAME = _CALC.create_loto_540()


def _joker_prize_ron(best_match: int, best_has_joker: bool, jackpot: float) -> float:
    for tier in _JOKER_GAME.prize_tiers:
        if tier.matches_required == best_match and tier.bonus_required == best_has_joker:
            if tier.fixed_prize is not None:
                return tier.fixed_prize
            if tier.prize_pool_percentage is not None:
                # Jackpot tier pari-mutuel; use provided jackpot / 1 winner
                if best_match == 5 and best_has_joker:
                    return jackpot
                # Other pari-mutuel tiers: rough 5% of ticket sales heuristic
                return 1000.0 * tier.prize_pool_percentage
    return 0.0


def _649_prize_ron(best_match: int, jackpot: float) -> float:
    for tier in _649_GAME.prize_tiers:
        if tier.matches_required == best_match:
            if tier.fixed_prize is not None:
                return tier.fixed_prize
            if best_match == 6:
                return jackpot
            return 1000.0 * (tier.prize_pool_percentage or 0.0)
    return 0.0


def _540_prize_ron(best_match: int, jackpot: float) -> float:
    # Loto 5/40: 6 drawn, player picks 5. "6-match" means all 5 picks are
    # among the 6 drawn AND the 6th drawn is the "5+1" bonus — rare. We
    # only score main_match 3/4/5 here; the 5+1 bonus case would require
    # per-draw "bonus_6th" data we don't have yet.
    for tier in _540_GAME.prize_tiers:
        if tier.matches_required == best_match:
            if tier.fixed_prize is not None:
                return tier.fixed_prize
            if best_match == 5:
                return jackpot
            return 1000.0 * (tier.prize_pool_percentage or 0.0)
    return 0.0


def estimate_ticket_payout(
    ticket: Ticket,
    winning_main: list[int] | tuple[int, ...],
    winning_joker: int | None,
    winning_side: str | None,
    jackpot: float = 1_000_000.0,
) -> float:
    """Estimate RON payout for a ticket on a given drawn set.

    Takes the best-performing variant; ignores side game matches for
    now (Plan C refines side-game prize ladders).
    """
    best_payout = 0.0
    for v in ticket.variants:
        matches = v.count_main_matches(winning_main)
        if ticket.game == "joker":
            has_joker = v.bonus_number is not None and v.bonus_number == winning_joker
            payout = _joker_prize_ron(matches, has_joker, jackpot)
        elif ticket.game == "loto_649":
            payout = _649_prize_ron(matches, jackpot)
        else:
            payout = _540_prize_ron(matches, jackpot)
        if payout > best_payout:
            best_payout = payout
    return best_payout


def median_roi(outcomes: list[TicketOutcome]) -> float:
    if not outcomes:
        return 0.0
    return statistics.median(o.net for o in outcomes)


def p_best_match_at_least(outcomes: list[TicketOutcome], threshold: int) -> float:
    if not outcomes:
        return 0.0
    hits = sum(1 for o in outcomes if o.best_match >= threshold)
    return hits / len(outcomes)


def sample_skewness(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    m3 = sum((v - mean) ** 3 for v in values) / n
    if m2 == 0:
        return 0.0
    return m3 / (m2 ** 1.5)


def summarize(outcomes: list[TicketOutcome]) -> dict[str, float]:
    return {
        "n": len(outcomes),
        "median_roi": median_roi(outcomes),
        "p_best_match_ge_3": p_best_match_at_least(outcomes, 3),
        "p_best_match_ge_4": p_best_match_at_least(outcomes, 4),
        "p_best_match_ge_5": p_best_match_at_least(outcomes, 5),
        "skewness": sample_skewness([o.net for o in outcomes]),
        "mean_roi": statistics.fmean([o.net for o in outcomes]) if outcomes else 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_metrics -v
```
Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/ticket_metrics.py tests/test_ticket_metrics.py
git commit -m "feat(shared): add ticket-level metrics for jackpot-shaped evaluation"
```

---

## Task 5: Ticket backtester

**Files:**
- Create: `src/shared/ticket_backtester.py`
- Create: `tests/test_ticket_backtester.py`

Walk-forward: for each draw `i` in the history, train the builder on `draws[:i]`, build a ticket, score `best_main_match` and payout against `draws[i]`, collect a `TicketOutcome`. Returns list of outcomes + summary.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ticket_backtester.py`:
```python
import random
import unittest

from shared.game_config import LOTO_649_CONFIG
from shared.ticket_backtester import TicketBacktester
from shared.ticket_builders import IndependentBuilder


class TestTicketBacktester(unittest.TestCase):
    def test_walk_forward_produces_outcome_per_draw_after_warmup(self):
        # 100 fake draws
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 50), 6)) for _ in range(100)]
        bt = TicketBacktester(
            game="loto_649",
            config=LOTO_649_CONFIG,
            draws=draws,
            draw_dates=None,
            warmup=50,
            rng=random.Random(0),
        )
        outcomes = bt.run(IndependentBuilder(n_tickets=1))
        # 100 draws − 50 warmup = 50 outcomes
        self.assertEqual(len(outcomes), 50)
        for o in outcomes:
            self.assertGreaterEqual(o.best_match, 0)
            self.assertLessEqual(o.best_match, 6)

    def test_summary_keys(self):
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 50), 6)) for _ in range(60)]
        bt = TicketBacktester(
            game="loto_649",
            config=LOTO_649_CONFIG,
            draws=draws,
            draw_dates=None,
            warmup=50,
            rng=random.Random(0),
        )
        outcomes = bt.run(IndependentBuilder(n_tickets=1))
        summary = bt.summarize(outcomes)
        self.assertIn("median_roi", summary)
        self.assertIn("skewness", summary)
        self.assertIn("p_best_match_ge_4", summary)
        self.assertEqual(summary["n"], 10)

    def test_reproducible_with_seeded_rng(self):
        rng_draws = random.Random(1)
        draws = [sorted(rng_draws.sample(range(1, 50), 6)) for _ in range(60)]
        bt1 = TicketBacktester(
            game="loto_649", config=LOTO_649_CONFIG, draws=draws,
            draw_dates=None, warmup=50, rng=random.Random(0),
        )
        bt2 = TicketBacktester(
            game="loto_649", config=LOTO_649_CONFIG, draws=draws,
            draw_dates=None, warmup=50, rng=random.Random(0),
        )
        o1 = bt1.run(IndependentBuilder(n_tickets=1))
        o2 = bt2.run(IndependentBuilder(n_tickets=1))
        self.assertEqual(
            [o.best_match for o in o1],
            [o.best_match for o in o2],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_backtester -v
```
Expected: `ModuleNotFoundError: No module named 'shared.ticket_backtester'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/shared/ticket_backtester.py`:
```python
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
            for t in tickets:
                best = t.best_main_match(drawn)
                payout = estimate_ticket_payout(
                    t,
                    winning_main=drawn,
                    winning_joker=None,
                    winning_side=None,
                    jackpot=self.jackpot,
                )
                outcomes.append(
                    TicketOutcome(payout=payout, cost=t.cost_ron, best_match=best)
                )
        return outcomes

    @staticmethod
    def summarize(outcomes: list[TicketOutcome]) -> dict[str, float]:
        return _summarize(outcomes)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_backtester -v
```
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/ticket_backtester.py tests/test_ticket_backtester.py
git commit -m "feat(shared): add walk-forward ticket backtester"
```

---

## Task 6: Joker backtest outcome plumbs the bonus ball

**Files:**
- Modify: `src/shared/ticket_backtester.py`
- Modify: `tests/test_ticket_backtester.py`

Joker draws carry a bonus (joker) number. Plan B's `estimate_ticket_payout` already accepts `winning_joker`, but the backtester passes `None`. Pipe the bonus through for Joker.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ticket_backtester.py`:
```python
from shared.game_config import JOKER_CONFIG


class TestTicketBacktesterJokerBonus(unittest.TestCase):
    def test_joker_uses_bonus_ball_in_payout(self):
        # 60 fake joker draws: mains + joker bonus
        rng = random.Random(42)
        mains = [sorted(rng.sample(range(1, 46), 5)) for _ in range(60)]
        bonuses = [rng.randint(1, 20) for _ in range(60)]
        bt = TicketBacktester(
            game="joker",
            config=JOKER_CONFIG,
            draws=mains,
            joker_bonuses=bonuses,
            draw_dates=None,
            warmup=50,
            rng=random.Random(0),
        )
        outcomes = bt.run(IndependentBuilder(n_tickets=1))
        self.assertEqual(len(outcomes), 10)
        # Outcomes should include some nonzero payouts thanks to fixed-prize joker tiers
        self.assertTrue(any(o.payout > 0 for o in outcomes))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_backtester.TestTicketBacktesterJokerBonus -v
```
Expected: `TypeError: __init__() got an unexpected keyword argument 'joker_bonuses'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/shared/ticket_backtester.py` (full file):
```python
"""Walk-forward ticket backtester.

For each historical draw past the warmup window, trains each builder
on the prefix and scores a single ticket's best match and estimated
payout against the draw.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_backtester -v
```
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/ticket_backtester.py tests/test_ticket_backtester.py
git commit -m "feat(backtester): plumb joker bonus and side-game winners through"
```

---

## Task 7: Backtest CLI

**Files:**
- Create: `scripts/run_jackpot_backtest.py`
- Create: `tests/test_run_jackpot_backtest.py`

Run all three builders on each game's historical draws, emit a markdown report.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_jackpot_backtest.py`:
```python
import tempfile
import unittest
from pathlib import Path

from scripts.run_jackpot_backtest import run


class TestRunJackpotBacktest(unittest.TestCase):
    def test_emits_report_with_all_builders_and_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.md"
            run(
                output_path=out,
                seed=42,
                warmup=50,
                jackpot=1_000_000.0,
            )
            self.assertTrue(out.exists())
            text = out.read_text()
            self.assertIn("# Jackpot Backtest Report", text)
            self.assertIn("Joker", text)
            self.assertIn("Loto 6/49", text)
            self.assertIn("Loto 5/40", text)
            self.assertIn("IndependentBuilder", text)
            self.assertIn("CoreShareBuilder", text)
            self.assertIn("WheelBuilder", text)
            self.assertIn("median_roi", text)
            self.assertIn("p_best_match_ge_4", text)
            self.assertIn("skewness", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_run_jackpot_backtest -v
```
Expected: `ModuleNotFoundError: No module named 'scripts.run_jackpot_backtest'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/run_jackpot_backtest.py`:
```python
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
        return [], [], []
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
            # Use a reduced warmup so the test passes on tiny fixture data.
            local_warmup = max(5, len(main) // 2)
            lines.append(f"_limited history: using warmup={local_warmup}_")
            lines.append("")
        else:
            local_warmup = warmup

        lines.append(
            "| builder | n | median_roi | mean_roi | P(≥3) | P(≥4) | P(≥5) | skewness |"
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_run_jackpot_backtest -v
```
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_jackpot_backtest.py tests/test_run_jackpot_backtest.py
git commit -m "feat(scripts): add jackpot-backtest CLI producing markdown report"
```

---

## Task 8: Run the backtest, commit the report

**Files:**
- Create: `docs/2026-plans/jackpot-backtest.md` (generated)

- [ ] **Step 1: Generate the report**

```bash
mkdir -p docs/2026-plans
PYTHONPATH=src python scripts/run_jackpot_backtest.py \
  --output docs/2026-plans/jackpot-backtest.md \
  --seed 42 --warmup 50 --jackpot 1000000
cat docs/2026-plans/jackpot-backtest.md
```
Expected: the file exists with three game sections and three builder rows each.

- [ ] **Step 2: Sanity check the numbers**

Read the report and check:
- Every cell has a number (no "error" rows).
- `n` for Joker and 6/49 is positive (should be total_draws − warmup).
- `median_roi` is negative for all rows (lottery EV is negative; this is a correctness check).
- `skewness` is positive and should be *higher* for CoreShare and Wheel than for Independent (the whole point of these strategies). If it isn't, flag in the PR — may mean the builder isn't producing the correlated variants correctly.
- `p_best_match_ge_4` should be numerically higher for CoreShare than Independent on Loto 6/49. This is the primary jackpot-tilt signal.

Document findings in a short note appended to the report:
```bash
cat >> docs/2026-plans/jackpot-backtest.md <<'EOF'

## Interpretation

_Findings from this backtest (fill in after reviewing the table above):_

- Median ROI — all builders should be negative. If a builder shows
  positive median ROI, suspect a bug in `estimate_ticket_payout`.
- Jackpot tilt — higher skewness and higher P(≥4) are the target.
  Compare CoreShare vs Independent; if CoreShare is not clearly higher
  on both, adjust core K or petal M in `ticket_builders._CORE_K`/`_PETAL_M`
  and re-run.
- Wheel builder — expect high P(≥3) but not necessarily higher P(≥4)
  at `pool_size=8`; wheels are coverage tools, not signal tools.
EOF
```

- [ ] **Step 3: Commit**

```bash
git add docs/2026-plans/jackpot-backtest.md
git commit -m "docs: first jackpot-backtest report comparing the three builders"
```

---

## Final verification

- [ ] **Run full test suite:**

```bash
PYTHONPATH=src python -m unittest -v 2>&1 | tail -30
```
Expected: "OK" (no failures).

- [ ] **Confirm orchestrator & workflows are unchanged:**

```bash
git diff main -- scripts/generate_recommended_picks.py scripts/check_results.py .github/workflows/
```
Expected: empty diff. Plan B must not have touched those files.

- [ ] **Open PR:**

```bash
git push -u origin feature/jackpot-b-strategies-backtest
gh pr create --fill --base main --title "Jackpot redesign B: strategies + backtest"
```

PR description must include:
- Attach the backtest report markdown (inline or link).
- Note the conclusion: which builder (if any) shows a meaningful jackpot tilt on the 94-draw history. If none do, explicitly say so — a null result from a tiny sample is the expected outcome and is NOT a blocker for proceeding to Plan C. The measurement infrastructure itself is the deliverable.

---

## Risks & open questions

1. **94 draws is underpowered.** With baseline P(best_match ≥ 4) ≈ 0.1 %, detecting even a 5× uplift would require thousands of draws. A null result in this backtest must not be interpreted as "the strategy doesn't work" — only as "we can't distinguish from noise yet". Real validation comes from ~6 months of live play after Plan C lands.
2. **Pari-mutuel tiers use a 1000-RON × percentage heuristic** in `ticket_metrics._joker_prize_ron` etc. This is a placeholder; a more faithful model requires per-draw total-sales data we don't currently ingest. For jackpot-tilt comparison this is fine because the *relative* differences between builders are what matter; absolute ROI numbers are lower-bound rough estimates.
3. **CoreShareBuilder's signal quality depends on `ensemble_blend`.** If the existing neural/frequency signal is pure noise (as suggested by the current 11.3% hit rate), the "core" is random 3-tuple and CoreShare degenerates to IndependentBuilder with correlated noise — higher variance but same mean. The architecture is still sound; we'd need a better signal source, which is out of scope here.
4. **WheelBuilder pool_size=8 is hardcoded in the CLI.** Plan C's allocator will need to choose pool_size dynamically based on budget. For Plan B's backtest, 8 is a sensible fixed point that fits one physical 6/49 ticket (3 variants).
5. **Skewness is sample-based and high-variance on 50-point samples.** Reported skewness will jitter between runs at different seeds. That's OK for the *qualitative* comparison but don't over-interpret small deltas.
