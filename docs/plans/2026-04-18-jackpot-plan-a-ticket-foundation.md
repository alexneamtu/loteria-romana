# Jackpot Redesign — Plan A: Ticket Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce the `Ticket`/`Variant` data model, a centralized pricing module, and side-game (Noroc / Super Noroc / Noroc Plus) parsing + storage for all three games — without changing generator behavior yet.

**Architecture:** New `src/shared/{pricing,ticket,side_games}.py` modules. Each game's existing `models.py`/`parser.py`/`storage.py` gets one additive side-game field with backward-compatible CSV read. A one-shot migration script extends `data/results/history.csv` with empty columns for future use. No orchestrator or output changes — those live in Plans C/D.

**Tech Stack:** Python 3.10+ stdlib only. unittest. Existing HTML-fixture-based parser tests.

**Prerequisites:** Main branch, clean working tree. Feature branch: `feature/jackpot-a-ticket-foundation`.

**Scope boundary (read before starting):**
- DO: add types, parse new fields, write/read new CSV columns, migrate history schema.
- DO NOT: change how picks are generated, change Telegram output, modify `scripts/generate_recommended_picks.py` beyond importing the new types, or touch `scripts/check_results.py` match logic.
- `src/shared/noroc_chior.py` and `scripts/backfill_noroc_chior.py` are for a different side game ("Noroc Chior" is a retroactive product, unrelated to Noroc / Super Noroc / Noroc Plus). Leave them alone.

---

## File Structure

| Status | Path | Responsibility |
|---|---|---|
| NEW | `src/shared/pricing.py` | Ticket price table + `compute_ticket_cost()` |
| NEW | `src/shared/ticket.py` | `Variant`, `Ticket`, `VARIANTS_PER_TICKET` |
| NEW | `src/shared/side_games.py` | Random side-game number generation + validation |
| MODIFY | `src/joker_model/models.py` | Add `noroc_plus: str \| None` |
| MODIFY | `src/joker_model/parser.py` | Extract Noroc Plus from HTML window |
| MODIFY | `src/joker_model/storage.py` | Read/write `noroc_plus` CSV column (backward-compatible) |
| MODIFY | `src/loto_649_model/models.py` | Add `noroc: str \| None` |
| MODIFY | `src/loto_649_model/parser.py` | Extract Noroc from HTML block |
| MODIFY | `src/loto_649_model/storage.py` | Read/write `noroc` CSV column |
| MODIFY | `src/loto_540_model/models.py` | Add `super_noroc: str \| None` |
| MODIFY | `src/loto_540_model/parser.py` | Extract Super Noroc from HTML window |
| MODIFY | `src/loto_540_model/storage.py` | Read/write `super_noroc` CSV column |
| NEW | `scripts/migrate_history_schema.py` | One-shot: add new columns to `data/results/history.csv` |
| NEW | `tests/test_pricing.py` | unittest for pricing module |
| NEW | `tests/test_ticket.py` | unittest for Variant/Ticket |
| NEW | `tests/test_side_games.py` | unittest for side-game helpers |
| NEW | `tests/fixtures/joker_with_noroc_plus.html` | HTML fixture with Noroc Plus number |
| NEW | `tests/fixtures/loto_649_with_noroc.html` | HTML fixture with Noroc number |
| NEW | `tests/fixtures/loto_540_with_super_noroc.html` | HTML fixture with Super Noroc number |
| MODIFY | `tests/test_parser.py` | Add test for Noroc Plus extraction, keep existing test green |
| MODIFY | `tests/test_loto_649_parser.py` | Add test for Noroc extraction |
| MODIFY | `tests/test_noroc_chior_parser.py` — **DO NOT TOUCH** | Different product. |

Test runner: `PYTHONPATH=src python -m unittest -v`.

---

## Task 1: Pricing module

**Files:**
- Create: `src/shared/pricing.py`
- Test: `tests/test_pricing.py`

**Confirmed prices (user-supplied 2026-04-18):**

| Game | Per variant | Per-ticket fee | Side game | Full ticket |
|---|---|---|---|---|
| Joker (2 variants + Noroc Plus) | 7.0 RON | 0.5 RON | 3.0 RON | **17.5 RON** |
| Loto 6/49 (3 variants + Noroc) | 8.0 RON | 0.5 RON | 4.0 RON | **28.5 RON** |
| Loto 5/40 (4 variants + Super Noroc) | 5.0 RON | 0.5 RON | 2.0 RON | **22.5 RON** |

The 0.5 RON per-ticket processing fee applies once per physical ticket regardless of variant count. Side games (Noroc / Noroc Plus / Super Noroc) are optional add-ons; they do not pay a separate processing fee. `TICKET_PRICE_NEEDS_VERIFICATION` is set to `False` — all prices are user-confirmed.

**Note for implementers:** these variant prices differ from the legacy `game_recommender.TICKET_COSTS` constants (which have 8/6/4). The legacy constants are used by `src/shared/ev_calculator.py` for break-even math and by the budget allocator. Updating them is out of scope for Plan A (it would break existing callers and requires the allocator changes in Plan C). Plan A introduces `PRICE_PER_VARIANT` as the canonical source of truth; Plan C migrates callers over.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pricing.py`:
```python
import unittest

from shared.pricing import (
    PRICE_PER_VARIANT,
    VARIANTS_PER_TICKET,
    SIDE_GAME_PRICE,
    PROCESSING_FEE_RON,
    TICKET_PRICE_NEEDS_VERIFICATION,
    compute_ticket_cost,
)


class TestPricing(unittest.TestCase):
    def test_variants_per_ticket_matches_loto_ro_format(self):
        self.assertEqual(VARIANTS_PER_TICKET["joker"], 2)
        self.assertEqual(VARIANTS_PER_TICKET["loto_649"], 3)
        self.assertEqual(VARIANTS_PER_TICKET["loto_540"], 4)

    def test_per_variant_prices_match_confirmed_loto_ro_values(self):
        self.assertEqual(PRICE_PER_VARIANT["joker"], 7.0)
        self.assertEqual(PRICE_PER_VARIANT["loto_649"], 8.0)
        self.assertEqual(PRICE_PER_VARIANT["loto_540"], 5.0)

    def test_processing_fee_is_half_ron(self):
        self.assertEqual(PROCESSING_FEE_RON, 0.5)

    def test_side_game_prices_match_confirmed_values(self):
        self.assertEqual(SIDE_GAME_PRICE["joker"], 3.0)       # Noroc Plus
        self.assertEqual(SIDE_GAME_PRICE["loto_649"], 4.0)    # Noroc
        self.assertEqual(SIDE_GAME_PRICE["loto_540"], 2.0)    # Super Noroc

    def test_verification_flag_is_false_now_that_all_prices_are_confirmed(self):
        self.assertFalse(TICKET_PRICE_NEEDS_VERIFICATION)

    def test_full_ticket_cost_includes_variants_fee_and_side_game(self):
        # Joker: 2 * 7.0 + 0.5 + 3.0 = 17.5
        self.assertEqual(compute_ticket_cost("joker"), 17.5)
        # Loto 6/49: 3 * 8.0 + 0.5 + 4.0 = 28.5
        self.assertEqual(compute_ticket_cost("loto_649"), 28.5)
        # Loto 5/40: 4 * 5.0 + 0.5 + 2.0 = 22.5
        self.assertEqual(compute_ticket_cost("loto_540"), 22.5)

    def test_variant_count_override_still_pays_fee_once(self):
        # 1 joker variant + fee, no side game: 7.0 + 0.5 = 7.5
        self.assertEqual(
            compute_ticket_cost("joker", variants=1, include_side_game=False),
            7.5,
        )
        # 2 loto_649 variants + fee: 16.0 + 0.5 = 16.5
        self.assertEqual(
            compute_ticket_cost("loto_649", variants=2, include_side_game=False),
            16.5,
        )

    def test_exclude_processing_fee_for_theoretical_variant_only_cost(self):
        self.assertEqual(
            compute_ticket_cost("joker", include_side_game=False, include_fee=False),
            14.0,
        )

    def test_exclude_side_game(self):
        # Full ticket without side game still pays variants + fee: 14.0 + 0.5
        self.assertEqual(compute_ticket_cost("joker", include_side_game=False), 14.5)

    def test_unknown_game_raises(self):
        with self.assertRaises(KeyError):
            compute_ticket_cost("powerball")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_pricing -v
```
Expected: `ModuleNotFoundError: No module named 'shared.pricing'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/shared/pricing.py`:
```python
"""Ticket pricing for loto.ro games.

Prices confirmed by user 2026-04-18:

    Joker variant:     7.0 RON   (2 variants per ticket)
    Loto 6/49 variant: 8.0 RON   (3 variants per ticket)
    Loto 5/40 variant: 5.0 RON   (4 variants per ticket)
    Processing fee:    0.5 RON   (flat, one per main-game ticket)
    Noroc Plus stake:  3.0 RON   (optional Joker side game)
    Noroc stake:       4.0 RON   (optional 6/49 side game)
    Super Noroc stake: 2.0 RON   (optional 5/40 side game)

Full ticket prices (all fields combined):
    Joker     = 2*7.0 + 0.5 + 3.0 = 17.5 RON
    Loto 6/49 = 3*8.0 + 0.5 + 4.0 = 28.5 RON
    Loto 5/40 = 4*5.0 + 0.5 + 2.0 = 22.5 RON

Legacy `game_recommender.TICKET_COSTS` holds incorrect per-variant
values (8/6/4) and is used by ev_calculator break-even math and the
current budget allocator. Plan C migrates callers to this module and
removes the legacy constants.
"""
from __future__ import annotations

PRICE_PER_VARIANT: dict[str, float] = {
    "joker": 7.0,
    "loto_649": 8.0,
    "loto_540": 5.0,
}

VARIANTS_PER_TICKET: dict[str, int] = {
    "joker": 2,
    "loto_649": 3,
    "loto_540": 4,
}

SIDE_GAME_PRICE: dict[str, float] = {
    "joker": 3.0,      # Noroc Plus
    "loto_649": 4.0,   # Noroc
    "loto_540": 2.0,   # Super Noroc
}

PROCESSING_FEE_RON: float = 0.5

TICKET_PRICE_NEEDS_VERIFICATION: bool = False


def compute_ticket_cost(
    game: str,
    variants: int | None = None,
    include_side_game: bool = True,
    include_fee: bool = True,
) -> float:
    """Return ticket cost in RON.

    Args:
        game: "joker" | "loto_649" | "loto_540"
        variants: number of variants on the ticket. Defaults to
            VARIANTS_PER_TICKET[game] (full ticket).
        include_side_game: add SIDE_GAME_PRICE[game] when True.
        include_fee: add PROCESSING_FEE_RON when True. The fee is paid
            once per physical ticket regardless of variant count.
    """
    if game not in PRICE_PER_VARIANT:
        raise KeyError(f"Unknown game: {game}")
    n_variants = VARIANTS_PER_TICKET[game] if variants is None else variants
    cost = PRICE_PER_VARIANT[game] * n_variants
    if include_fee:
        cost += PROCESSING_FEE_RON
    if include_side_game:
        cost += SIDE_GAME_PRICE[game]
    return cost
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_pricing -v
```
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/pricing.py tests/test_pricing.py
git commit -m "feat(shared): add pricing module for full-ticket costs"
```

---

## Task 2: Variant dataclass

**Files:**
- Create: `src/shared/ticket.py` (first half — Variant only)
- Test: `tests/test_ticket.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ticket.py`:
```python
import unittest

from shared.ticket import Variant


class TestVariant(unittest.TestCase):
    def test_joker_variant_accepts_5_mains_and_1_bonus(self):
        v = Variant(main_numbers=(3, 7, 12, 19, 28), bonus_number=11, game="joker")
        self.assertEqual(v.main_numbers, (3, 7, 12, 19, 28))
        self.assertEqual(v.bonus_number, 11)

    def test_loto_649_variant_accepts_6_mains_and_no_bonus(self):
        v = Variant(main_numbers=(1, 5, 17, 23, 34, 49), bonus_number=None, game="loto_649")
        self.assertIsNone(v.bonus_number)

    def test_loto_540_variant_accepts_5_mains_and_no_bonus(self):
        v = Variant(main_numbers=(2, 9, 15, 27, 40), bonus_number=None, game="loto_540")
        self.assertIsNone(v.bonus_number)

    def test_rejects_wrong_main_count(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(1, 2, 3, 4), bonus_number=5, game="joker")

    def test_rejects_out_of_range_main(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(1, 2, 3, 4, 46), bonus_number=5, game="joker")

    def test_rejects_duplicate_mains(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(1, 2, 3, 4, 4), bonus_number=5, game="joker")

    def test_rejects_unsorted_mains(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(5, 1, 2, 3, 4), bonus_number=5, game="joker")

    def test_rejects_missing_bonus_for_joker(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(1, 2, 3, 4, 5), bonus_number=None, game="joker")

    def test_rejects_bonus_for_non_joker(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(1, 5, 17, 23, 34, 49), bonus_number=3, game="loto_649")

    def test_rejects_out_of_range_bonus(self):
        with self.assertRaises(ValueError):
            Variant(main_numbers=(1, 2, 3, 4, 5), bonus_number=21, game="joker")

    def test_is_frozen_hashable(self):
        v = Variant(main_numbers=(3, 7, 12, 19, 28), bonus_number=11, game="joker")
        self.assertEqual(hash(v), hash(v))

    def test_count_main_matches(self):
        v = Variant(main_numbers=(3, 7, 12, 19, 28), bonus_number=11, game="joker")
        self.assertEqual(v.count_main_matches((3, 7, 99, 28, 100)), 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket -v
```
Expected: `ModuleNotFoundError: No module named 'shared.ticket'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/shared/ticket.py`:
```python
"""Ticket and Variant dataclasses matching loto.ro physical tickets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Game = Literal["joker", "loto_649", "loto_540"]

_MAIN_COUNT: dict[str, int] = {"joker": 5, "loto_649": 6, "loto_540": 5}
_MAIN_POOL_MAX: dict[str, int] = {"joker": 45, "loto_649": 49, "loto_540": 40}
_JOKER_POOL_MAX = 20


@dataclass(frozen=True)
class Variant:
    main_numbers: tuple[int, ...]
    bonus_number: int | None
    game: str

    def __post_init__(self) -> None:
        if self.game not in _MAIN_COUNT:
            raise ValueError(f"unknown game: {self.game!r}")

        expected = _MAIN_COUNT[self.game]
        if len(self.main_numbers) != expected:
            raise ValueError(
                f"{self.game} variant expects {expected} main numbers, "
                f"got {len(self.main_numbers)}"
            )

        if len(set(self.main_numbers)) != expected:
            raise ValueError(f"{self.game} variant has duplicate main numbers")

        if list(self.main_numbers) != sorted(self.main_numbers):
            raise ValueError(f"{self.game} variant main numbers must be sorted ascending")

        pool_max = _MAIN_POOL_MAX[self.game]
        for n in self.main_numbers:
            if not 1 <= n <= pool_max:
                raise ValueError(
                    f"{self.game} main number {n} out of range 1..{pool_max}"
                )

        if self.game == "joker":
            if self.bonus_number is None:
                raise ValueError("joker variant requires a bonus_number (1..20)")
            if not 1 <= self.bonus_number <= _JOKER_POOL_MAX:
                raise ValueError(
                    f"joker bonus_number {self.bonus_number} out of range 1..20"
                )
        else:
            if self.bonus_number is not None:
                raise ValueError(f"{self.game} variant must not carry a bonus_number")

    def count_main_matches(self, winning: tuple[int, ...] | list[int]) -> int:
        return len(set(self.main_numbers) & set(winning))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket -v
```
Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/ticket.py tests/test_ticket.py
git commit -m "feat(shared): add Variant dataclass for per-game picks"
```

---

## Task 3: Ticket dataclass

**Files:**
- Modify: `src/shared/ticket.py` (append Ticket)
- Modify: `tests/test_ticket.py` (append TestTicket class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ticket.py` (before the `if __name__` guard):
```python
from shared.ticket import Ticket


class TestTicket(unittest.TestCase):
    def _joker_variants(self) -> tuple[Variant, Variant]:
        v1 = Variant(main_numbers=(3, 7, 12, 19, 28), bonus_number=11, game="joker")
        v2 = Variant(main_numbers=(3, 7, 12, 19, 33), bonus_number=11, game="joker")
        return (v1, v2)

    def test_joker_ticket_requires_2_variants(self):
        variants = self._joker_variants()
        t = Ticket(
            game="joker",
            variants=variants,
            side_game_number="NP14",
            strategy="core_share",
            cost_ron=17.5,
        )
        self.assertEqual(len(t.variants), 2)
        self.assertEqual(t.side_game_number, "NP14")

    def test_joker_ticket_rejects_wrong_variant_count(self):
        v1, _ = self._joker_variants()
        with self.assertRaises(ValueError):
            Ticket(
                game="joker",
                variants=(v1,),
                side_game_number="NP14",
                strategy="core_share",
                cost_ron=10.5,
            )

    def test_ticket_rejects_variants_of_wrong_game(self):
        joker_v = Variant(main_numbers=(3, 7, 12, 19, 28), bonus_number=11, game="joker")
        loto_v = Variant(main_numbers=(1, 5, 17, 23, 34, 49), bonus_number=None, game="loto_649")
        with self.assertRaises(ValueError):
            Ticket(
                game="joker",
                variants=(joker_v, loto_v),
                side_game_number="NP14",
                strategy="core_share",
                cost_ron=17.5,
            )

    def test_best_main_match_across_variants(self):
        variants = self._joker_variants()
        t = Ticket(
            game="joker",
            variants=variants,
            side_game_number="NP14",
            strategy="core_share",
            cost_ron=17.5,
        )
        # winning = 3,7,12,44,45 + J11 → v1 matches 3, v2 matches 3 on mains
        self.assertEqual(t.best_main_match((3, 7, 12, 44, 45)), 3)
        # winning = 3,7,12,19,28 → v1 matches 5, v2 matches 4
        self.assertEqual(t.best_main_match((3, 7, 12, 19, 28)), 5)

    def test_loto_649_ticket_requires_3_variants(self):
        mk = lambda n: Variant(
            main_numbers=tuple(sorted((1, 5, 17, 23, 34, n))),
            bonus_number=None,
            game="loto_649",
        )
        variants = (mk(45), mk(46), mk(47))
        t = Ticket(
            game="loto_649",
            variants=variants,
            side_game_number="1234567",
            strategy="wheel_3if9",
            cost_ron=24.5,
        )
        self.assertEqual(len(t.variants), 3)

    def test_loto_540_ticket_requires_4_variants(self):
        mk = lambda n: Variant(
            main_numbers=tuple(sorted((2, 9, 15, 27, n))),
            bonus_number=None,
            game="loto_540",
        )
        variants = (mk(35), mk(36), mk(37), mk(38))
        t = Ticket(
            game="loto_540",
            variants=variants,
            side_game_number="123456",
            strategy="independent",
            cost_ron=20.5,
        )
        self.assertEqual(len(t.variants), 4)

    def test_side_game_number_is_string_preserving_leading_zeros(self):
        variants = self._joker_variants()
        t = Ticket(
            game="joker",
            variants=variants,
            side_game_number="NP07",
            strategy="core_share",
            cost_ron=17.5,
        )
        self.assertEqual(t.side_game_number, "NP07")

    def test_ticket_is_frozen(self):
        variants = self._joker_variants()
        t = Ticket(
            game="joker",
            variants=variants,
            side_game_number="NP14",
            strategy="core_share",
            cost_ron=17.5,
        )
        with self.assertRaises(Exception):
            t.strategy = "other"  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket.TestTicket -v
```
Expected: `ImportError: cannot import name 'Ticket' from 'shared.ticket'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/shared/ticket.py`:
```python
from .pricing import VARIANTS_PER_TICKET


@dataclass(frozen=True)
class Ticket:
    game: str
    variants: tuple[Variant, ...]
    side_game_number: str
    strategy: str
    cost_ron: float

    def __post_init__(self) -> None:
        if self.game not in VARIANTS_PER_TICKET:
            raise ValueError(f"unknown game: {self.game!r}")

        expected_variants = VARIANTS_PER_TICKET[self.game]
        if len(self.variants) != expected_variants:
            raise ValueError(
                f"{self.game} ticket requires {expected_variants} variants, "
                f"got {len(self.variants)}"
            )

        for v in self.variants:
            if v.game != self.game:
                raise ValueError(
                    f"variant game={v.game!r} does not match ticket game={self.game!r}"
                )

        if not isinstance(self.side_game_number, str):
            raise ValueError("side_game_number must be a string (leading zeros matter)")

        if self.cost_ron < 0:
            raise ValueError("cost_ron must be non-negative")

    def best_main_match(self, winning: tuple[int, ...] | list[int]) -> int:
        return max(v.count_main_matches(winning) for v in self.variants)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket -v
```
Expected: all TestVariant + TestTicket tests pass (20 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/shared/ticket.py tests/test_ticket.py
git commit -m "feat(shared): add Ticket dataclass grouping variants + side game"
```

---

## Task 4: Side-game helpers

**Files:**
- Create: `src/shared/side_games.py`
- Test: `tests/test_side_games.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_side_games.py`:
```python
import random
import unittest

from shared.side_games import (
    NOROC_DIGITS,
    SUPER_NOROC_DIGITS,
    NOROC_PLUS_MIN,
    NOROC_PLUS_MAX,
    generate_noroc,
    generate_super_noroc,
    generate_noroc_plus,
    validate_side_game_number,
)


class TestSideGames(unittest.TestCase):
    def test_noroc_constants(self):
        self.assertEqual(NOROC_DIGITS, 7)
        self.assertEqual(SUPER_NOROC_DIGITS, 6)
        self.assertEqual((NOROC_PLUS_MIN, NOROC_PLUS_MAX), (1, 20))

    def test_generate_noroc_is_7_digit_string(self):
        rng = random.Random(0)
        n = generate_noroc(rng)
        self.assertEqual(len(n), 7)
        self.assertTrue(n.isdigit())

    def test_generate_noroc_preserves_leading_zeros(self):
        rng = random.Random(1)  # seed picked so first draw has leading zero
        samples = {generate_noroc(rng) for _ in range(200)}
        self.assertTrue(any(s.startswith("0") for s in samples))

    def test_generate_super_noroc_is_6_digit_string(self):
        rng = random.Random(0)
        n = generate_super_noroc(rng)
        self.assertEqual(len(n), 6)
        self.assertTrue(n.isdigit())

    def test_generate_noroc_plus_is_np_prefix(self):
        rng = random.Random(0)
        n = generate_noroc_plus(rng)
        self.assertTrue(n.startswith("NP"))
        inner = int(n[2:])
        self.assertTrue(NOROC_PLUS_MIN <= inner <= NOROC_PLUS_MAX)

    def test_validate_noroc(self):
        self.assertTrue(validate_side_game_number("loto_649", "1234567"))
        self.assertFalse(validate_side_game_number("loto_649", "123456"))
        self.assertFalse(validate_side_game_number("loto_649", "12345678"))
        self.assertFalse(validate_side_game_number("loto_649", "12a4567"))

    def test_validate_super_noroc(self):
        self.assertTrue(validate_side_game_number("loto_540", "012345"))
        self.assertFalse(validate_side_game_number("loto_540", "12345"))

    def test_validate_noroc_plus(self):
        self.assertTrue(validate_side_game_number("joker", "NP14"))
        self.assertTrue(validate_side_game_number("joker", "NP01"))
        self.assertFalse(validate_side_game_number("joker", "14"))
        self.assertFalse(validate_side_game_number("joker", "NP21"))

    def test_seed_reproduces(self):
        r1 = random.Random(42)
        r2 = random.Random(42)
        self.assertEqual(generate_noroc(r1), generate_noroc(r2))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_side_games -v
```
Expected: `ModuleNotFoundError: No module named 'shared.side_games'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/shared/side_games.py`:
```python
"""Side-game number helpers for Noroc / Super Noroc / Noroc Plus.

Romanian loto.ro ticket format:
- Loto 6/49 ticket carries a Noroc number: 7 digits, leading zeros allowed.
- Loto 5/40 ticket carries a Super Noroc number: 6 digits, leading zeros allowed.
- Joker ticket carries a Noroc Plus number: prefix "NP" + integer 1..20.

Numbers are always strings to preserve leading zeros.
"""
from __future__ import annotations

import random
import re

NOROC_DIGITS = 7
SUPER_NOROC_DIGITS = 6
NOROC_PLUS_MIN = 1
NOROC_PLUS_MAX = 20

_NOROC_RE = re.compile(rf"^\d{{{NOROC_DIGITS}}}$")
_SUPER_NOROC_RE = re.compile(rf"^\d{{{SUPER_NOROC_DIGITS}}}$")
_NOROC_PLUS_RE = re.compile(r"^NP(\d{1,2})$")


def generate_noroc(rng: random.Random) -> str:
    return f"{rng.randint(0, 10**NOROC_DIGITS - 1):0{NOROC_DIGITS}d}"


def generate_super_noroc(rng: random.Random) -> str:
    return f"{rng.randint(0, 10**SUPER_NOROC_DIGITS - 1):0{SUPER_NOROC_DIGITS}d}"


def generate_noroc_plus(rng: random.Random) -> str:
    n = rng.randint(NOROC_PLUS_MIN, NOROC_PLUS_MAX)
    return f"NP{n:02d}"


def validate_side_game_number(game: str, value: str) -> bool:
    if game == "loto_649":
        return bool(_NOROC_RE.match(value))
    if game == "loto_540":
        return bool(_SUPER_NOROC_RE.match(value))
    if game == "joker":
        match = _NOROC_PLUS_RE.match(value)
        if not match:
            return False
        return NOROC_PLUS_MIN <= int(match.group(1)) <= NOROC_PLUS_MAX
    return False
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_side_games -v
```
Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/side_games.py tests/test_side_games.py
git commit -m "feat(shared): add side-game number helpers (Noroc, Super Noroc, Noroc Plus)"
```

---

## Task 5: Joker model — add `noroc_plus` field

**Files:**
- Modify: `src/joker_model/models.py`
- Test: `tests/test_parser.py` (check existing test stays green — no new test here, the default-None case)

- [ ] **Step 1: Write the failing test**

Append a test class to `tests/test_parser.py` (before the `if __name__` guard):
```python
from joker_model.models import JokerDraw


class TestJokerDrawModel(unittest.TestCase):
    def test_noroc_plus_defaults_to_none(self):
        d = JokerDraw(date="2026-01-15", main_numbers=[7, 11, 44, 45, 46], joker=13)
        self.assertIsNone(d.noroc_plus)

    def test_noroc_plus_accepts_string(self):
        d = JokerDraw(
            date="2026-01-15",
            main_numbers=[7, 11, 44, 45, 46],
            joker=13,
            noroc_plus="NP07",
        )
        self.assertEqual(d.noroc_plus, "NP07")

    def test_joker_draw_still_hashable(self):
        d = JokerDraw(date="2026-01-15", main_numbers=[7, 11, 44, 45, 46], joker=13)
        self.assertEqual(hash(d), hash(d))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_parser.TestJokerDrawModel -v
```
Expected: `TypeError: __init__() got an unexpected keyword argument 'noroc_plus'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/joker_model/models.py`:
```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class JokerDraw:
    date: str
    main_numbers: list[int]
    joker: int
    noroc_plus: str | None = None

    def __hash__(self) -> int:
        return hash((self.date, tuple(self.main_numbers), self.joker, self.noroc_plus))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_parser -v
```
Expected: existing TestParser passes and new TestJokerDrawModel (3 tests) passes. Also run the full suite to catch regressions:
```bash
PYTHONPATH=src python -m unittest -v 2>&1 | tail -20
```
Expected: "OK" (no failures).

- [ ] **Step 5: Commit**

```bash
git add src/joker_model/models.py tests/test_parser.py
git commit -m "feat(joker): add optional noroc_plus field to JokerDraw"
```

---

## Task 6: Joker parser — extract Noroc Plus number

**Files:**
- Create: `tests/fixtures/joker_with_noroc_plus.html`
- Modify: `src/joker_model/parser.py`
- Modify: `tests/test_parser.py`

**Why this fixture format:** loto.ro renders Noroc Plus as a separate `<img src="/bile/noroc-plus/NN.png" />` tag inside the same `content-rezultate` block as the main draw. The parser windows back 2000 chars from the date tag and scans for `/bile/*.png` — we need a distinct pattern for the Noroc Plus image.

- [ ] **Step 1: Write the failing test — create fixture**

Create `tests/fixtures/joker_with_noroc_plus.html`:
```html
<div class="content-rezultate">
  <img src="/bile/45.png" />
  <img src="/bile/11.png" />
  <img src="/bile/46.png" />
  <img src="/bile/7.png" />
  <img src="/bile/44.png" />
  <img src="/bile/joker/13.png" />
  <img src="/bile/noroc-plus/07.png" />
  <p>Detalii castiguri  la joker din <span>15.01.2026</span></p>
</div>
<div class="content-rezultate">
  <img src="/bile/1.png" />
  <img src="/bile/22.png" />
  <img src="/bile/13.png" />
  <img src="/bile/30.png" />
  <img src="/bile/33.png" />
  <img src="/bile/joker/6.png" />
  <p>Detalii castiguri  la joker din <span>11.01.2026</span></p>
</div>
```

Append to `tests/test_parser.py`:
```python
class TestParserWithNorocPlus(unittest.TestCase):
    def test_parse_extracts_noroc_plus_when_present(self):
        html = Path("tests/fixtures/joker_with_noroc_plus.html").read_text(encoding="utf-8")
        draws = parse_joker_results(html)
        self.assertEqual(len(draws), 2)
        self.assertEqual(draws[0].date, "2026-01-15")
        self.assertEqual(draws[0].noroc_plus, "NP07")

    def test_parse_returns_none_when_noroc_plus_absent(self):
        # Old fixture has no noroc-plus img in its 2000-char window.
        html = Path("tests/fixtures/joker_results_snippet.html").read_text(encoding="utf-8")
        draws = parse_joker_results(html)
        self.assertEqual(len(draws), 2)
        self.assertIsNone(draws[0].noroc_plus)
        self.assertIsNone(draws[1].noroc_plus)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_parser.TestParserWithNorocPlus -v
```
Expected: first test fails with `AssertionError: None != 'NP07'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/joker_model/parser.py`:
```python
import re
from datetime import datetime

from .models import JokerDraw


def _normalize_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")


def parse_joker_results(html: str) -> list[JokerDraw]:
    draws = []
    date_pattern = re.compile(r"Detalii castiguri\s+la joker\s+din\s+<span>(\d{2}\.\d{2}\.\d{4})</span>")

    for match in date_pattern.finditer(html):
        window_start = max(0, match.start() - 2000)
        window = html[window_start:match.start()]

        main_nums = [int(n) for n in re.findall(r"/bile/(\d{1,2})\.png", window)]
        joker_nums = [int(n) for n in re.findall(r"/bile/joker/(\d{1,2})\.png", window)]
        noroc_plus_nums = re.findall(r"/bile/noroc-plus/(\d{1,2})\.png", window)

        if len(main_nums) < 5 or not joker_nums:
            continue

        main = sorted(main_nums[-5:])
        joker = joker_nums[-1]
        noroc_plus = f"NP{int(noroc_plus_nums[-1]):02d}" if noroc_plus_nums else None
        draws.append(JokerDraw(_normalize_date(match.group(1)), main, joker, noroc_plus))

    return draws
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_parser -v
```
Expected: all parser tests pass (original + TestJokerDrawModel + TestParserWithNorocPlus).

- [ ] **Step 5: Commit**

```bash
git add src/joker_model/parser.py tests/fixtures/joker_with_noroc_plus.html tests/test_parser.py
git commit -m "feat(joker): parse Noroc Plus number from loto.ro HTML"
```

---

## Task 7: Joker storage — read/write `noroc_plus` CSV column

**Files:**
- Modify: `src/joker_model/storage.py`
- Modify: `tests/test_fetch_storage.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fetch_storage.py` (or create a new test class if that file doesn't cover storage directly — inspect first with `PYTHONPATH=src python -m unittest tests.test_fetch_storage -v` to see existing test names, then add):
```python
import csv
import tempfile
import unittest
from pathlib import Path

from joker_model.models import JokerDraw
from joker_model.storage import append_draws, load_draws


class TestJokerStorageNorocPlus(unittest.TestCase):
    def test_round_trip_with_noroc_plus(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draws.csv"
            draws = [
                JokerDraw(
                    date="2026-01-15",
                    main_numbers=[7, 11, 44, 45, 46],
                    joker=13,
                    noroc_plus="NP07",
                ),
            ]
            append_draws(path, draws)
            loaded = load_draws(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].noroc_plus, "NP07")

    def test_load_legacy_csv_without_noroc_plus_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            path.write_text(
                "date,main_1,main_2,main_3,main_4,main_5,joker\n"
                "2026-01-15,7,11,44,45,46,13\n",
                encoding="utf-8",
            )
            loaded = load_draws(path)
            self.assertEqual(len(loaded), 1)
            self.assertIsNone(loaded[0].noroc_plus)

    def test_round_trip_with_noroc_plus_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draws.csv"
            draws = [
                JokerDraw(date="2026-01-15", main_numbers=[7, 11, 44, 45, 46], joker=13),
            ]
            append_draws(path, draws)
            loaded = load_draws(path)
            self.assertIsNone(loaded[0].noroc_plus)
            with path.open() as f:
                header = next(csv.reader(f))
            self.assertIn("noroc_plus", header)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_fetch_storage.TestJokerStorageNorocPlus -v
```
Expected: first test fails (column not written or not loaded).

- [ ] **Step 3: Write minimal implementation**

Replace `src/joker_model/storage.py`:
```python
import csv
from pathlib import Path

from .models import JokerDraw

_FIELDNAMES = ["date"] + [f"main_{i}" for i in range(1, 6)] + ["joker", "noroc_plus"]


def load_draws(path: Path) -> list[JokerDraw]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            main = [int(row[f"main_{i}"]) for i in range(1, 6)]
            noroc_plus = row.get("noroc_plus") or None
            rows.append(
                JokerDraw(
                    row["date"],
                    sorted(main),
                    int(row["joker"]),
                    noroc_plus,
                )
            )
    return sorted(rows, key=lambda d: d.date)


def append_draws(path: Path, draws: list[JokerDraw]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        if not exists:
            writer.writeheader()
        for draw in draws:
            writer.writerow({
                "date": draw.date,
                "main_1": draw.main_numbers[0],
                "main_2": draw.main_numbers[1],
                "main_3": draw.main_numbers[2],
                "main_4": draw.main_numbers[3],
                "main_5": draw.main_numbers[4],
                "joker": draw.joker,
                "noroc_plus": draw.noroc_plus or "",
            })
    return len(draws)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_fetch_storage -v
```
Expected: all fetch/storage tests pass.

Run full suite to catch any regression from storage changes (e.g. the live `data/clean/joker_draws.csv` file is still readable):
```bash
PYTHONPATH=src python -c "from pathlib import Path; import sys; sys.path.insert(0, 'src'); from joker_model.storage import load_draws; draws = load_draws(Path('data/clean/joker_draws.csv')); print(f'loaded {len(draws)} draws; last noroc_plus={draws[-1].noroc_plus!r}')"
```
Expected: prints "loaded N draws; last noroc_plus=None".

- [ ] **Step 5: Commit**

```bash
git add src/joker_model/storage.py tests/test_fetch_storage.py
git commit -m "feat(joker): persist noroc_plus in CSV with backward-compat read"
```

---

## Task 8: Loto 6/49 model — add `noroc` field

**Files:**
- Modify: `src/loto_649_model/models.py`
- Modify: `tests/test_loto_649_parser.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_loto_649_parser.py`:
```python
from loto_649_model.models import Loto649Draw


class TestLoto649DrawModel(unittest.TestCase):
    def test_noroc_defaults_to_none(self):
        d = Loto649Draw(date="2026-01-15", main_numbers=[1, 5, 17, 23, 34, 49])
        self.assertIsNone(d.noroc)

    def test_noroc_accepts_7_digit_string(self):
        d = Loto649Draw(
            date="2026-01-15",
            main_numbers=[1, 5, 17, 23, 34, 49],
            noroc="0123456",
        )
        self.assertEqual(d.noroc, "0123456")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_loto_649_parser.TestLoto649DrawModel -v
```
Expected: `TypeError: __init__() got an unexpected keyword argument 'noroc'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/loto_649_model/models.py`:
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Loto649Draw:
    date: str
    main_numbers: list[int]
    noroc: str | None = None

    def __hash__(self) -> int:
        return hash((self.date, tuple(self.main_numbers), self.noroc))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_loto_649_parser -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/loto_649_model/models.py tests/test_loto_649_parser.py
git commit -m "feat(loto_649): add optional noroc field to Loto649Draw"
```

---

## Task 9: Loto 6/49 parser — extract Noroc number

**Files:**
- Create: `tests/fixtures/loto_649_with_noroc.html`
- Modify: `src/loto_649_model/parser.py`
- Modify: `tests/test_loto_649_parser.py`

**HTML investigation note:** Loto 6/49 Noroc renders as seven separate ball images `<img src="/bile/noroc/N.png" />` in the same `rezultate-extrageri-content` block. Each image is a single digit 0-9, and the 7 images concatenate left-to-right into the Noroc number. Confirm this assumption by inspecting a live page if the fixture parser fails; if the format differs (e.g. one image with a 7-digit number in the filename), update the regex accordingly.

- [ ] **Step 1: Write the failing test — create fixture**

Create `tests/fixtures/loto_649_with_noroc.html`:
```html
<div class="rezultate-extrageri-content loto649">
  <img src="/bile/3.png" />
  <img src="/bile/12.png" />
  <img src="/bile/27.png" />
  <img src="/bile/31.png" />
  <img src="/bile/38.png" />
  <img src="/bile/44.png" />
  <img src="/bile/noroc/0.png" />
  <img src="/bile/noroc/1.png" />
  <img src="/bile/noroc/2.png" />
  <img src="/bile/noroc/3.png" />
  <img src="/bile/noroc/4.png" />
  <img src="/bile/noroc/5.png" />
  <img src="/bile/noroc/6.png" />
  <p>Detalii castiguri la 6/49 din <span>15.01.2026</span></p>
</div>
<div class="rezultate-extrageri-content loto649">
  <img src="/bile/2.png" />
  <img src="/bile/9.png" />
  <img src="/bile/15.png" />
  <img src="/bile/22.png" />
  <img src="/bile/28.png" />
  <img src="/bile/41.png" />
  <p>Detalii castiguri la 6/49 din <span>11.01.2026</span></p>
</div>
```

Append to `tests/test_loto_649_parser.py`:
```python
class TestLoto649ParserNoroc(unittest.TestCase):
    def test_parse_extracts_noroc_when_present(self):
        html = Path("tests/fixtures/loto_649_with_noroc.html").read_text(encoding="utf-8")
        draws = parse_loto_649_results(html)
        by_date = {d.date: d for d in draws}
        self.assertEqual(by_date["2026-01-15"].noroc, "0123456")

    def test_parse_returns_none_when_noroc_absent(self):
        html = Path("tests/fixtures/loto_649_with_noroc.html").read_text(encoding="utf-8")
        draws = parse_loto_649_results(html)
        by_date = {d.date: d for d in draws}
        self.assertIsNone(by_date["2026-01-11"].noroc)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_loto_649_parser.TestLoto649ParserNoroc -v
```
Expected: `AssertionError: None != '0123456'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/loto_649_model/parser.py`:
```python
import re
from datetime import datetime

from .models import Loto649Draw


def _normalize_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")


def parse_loto_649_results(html: str) -> list[Loto649Draw]:
    blocks = re.split(r"<div class=\"rezultate-extrageri-content[^\"]*\">", html)
    date_pattern = re.compile(r"Detalii castiguri[^<]*<span>(\d{2}\.\d{2}\.\d{4})</span>", re.IGNORECASE)

    main_by_date: dict[str, list[int]] = {}
    noroc_by_date: dict[str, str | None] = {}

    for block in blocks:
        date_match = date_pattern.search(block)
        if not date_match:
            continue
        date = _normalize_date(date_match.group(1))

        numbers = [int(n) for n in re.findall(r"/bile/(\d{1,2})\.png", block)]
        if len(numbers) >= 6:
            main_by_date[date] = sorted(numbers[-6:])

        noroc_digits = re.findall(r"/bile/noroc/(\d)\.png", block)
        noroc_by_date[date] = "".join(noroc_digits) if len(noroc_digits) == 7 else None

    draws = []
    for date, main in sorted(main_by_date.items()):
        draws.append(Loto649Draw(date, main, noroc_by_date.get(date)))

    return draws
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_loto_649_parser -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/loto_649_model/parser.py tests/fixtures/loto_649_with_noroc.html tests/test_loto_649_parser.py
git commit -m "feat(loto_649): parse Noroc 7-digit number from loto.ro HTML"
```

---

## Task 10: Loto 6/49 storage — read/write `noroc` CSV column

**Files:**
- Modify: `src/loto_649_model/storage.py`
- Modify: `tests/test_loto_649_fetch_storage.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_loto_649_fetch_storage.py`:
```python
import csv
import tempfile
from pathlib import Path

from loto_649_model.models import Loto649Draw
from loto_649_model.storage import append_draws, load_draws


class TestLoto649StorageNoroc(unittest.TestCase):
    def test_round_trip_with_noroc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draws.csv"
            draws = [
                Loto649Draw(
                    date="2026-01-15",
                    main_numbers=[3, 12, 27, 31, 38, 44],
                    noroc="0123456",
                ),
            ]
            append_draws(path, draws)
            loaded = load_draws(path)
            self.assertEqual(loaded[0].noroc, "0123456")

    def test_load_legacy_csv_without_noroc_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            path.write_text(
                "date,main_1,main_2,main_3,main_4,main_5,main_6\n"
                "2026-01-15,3,12,27,31,38,44\n",
                encoding="utf-8",
            )
            loaded = load_draws(path)
            self.assertIsNone(loaded[0].noroc)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_loto_649_fetch_storage.TestLoto649StorageNoroc -v
```
Expected: first test fails (noroc column not written).

- [ ] **Step 3: Write minimal implementation**

Replace `src/loto_649_model/storage.py`:
```python
import csv
from pathlib import Path

from .models import Loto649Draw

_FIELDNAMES = ["date"] + [f"main_{i}" for i in range(1, 7)] + ["noroc"]


def load_draws(path: Path) -> list[Loto649Draw]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            main = [int(row[f"main_{i}"]) for i in range(1, 7)]
            noroc = row.get("noroc") or None
            rows.append(Loto649Draw(row["date"], sorted(main), noroc))
    return rows


def append_draws(path: Path, draws: list[Loto649Draw]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        if not exists:
            writer.writeheader()
        for draw in draws:
            writer.writerow({
                "date": draw.date,
                "main_1": draw.main_numbers[0],
                "main_2": draw.main_numbers[1],
                "main_3": draw.main_numbers[2],
                "main_4": draw.main_numbers[3],
                "main_5": draw.main_numbers[4],
                "main_6": draw.main_numbers[5],
                "noroc": draw.noroc or "",
            })
    return len(draws)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_loto_649_fetch_storage -v
```
Expected: all pass. Then verify live CSV still loads:
```bash
PYTHONPATH=src python -c "import sys; sys.path.insert(0, 'src'); from pathlib import Path; from loto_649_model.storage import load_draws; ds = load_draws(Path('data/clean/loto_649_draws.csv')); print(len(ds), ds[-1].noroc)"
```

- [ ] **Step 5: Commit**

```bash
git add src/loto_649_model/storage.py tests/test_loto_649_fetch_storage.py
git commit -m "feat(loto_649): persist noroc in CSV with backward-compat read"
```

---

## Task 11: Loto 5/40 model — add `super_noroc` field

**Files:**
- Modify: `src/loto_540_model/models.py`
- Modify: `tests/test_loto_649_parser.py` (no — create `tests/test_loto_540_models.py` if one doesn't exist, or append to an existing test file that imports from `loto_540_model`)

- [ ] **Step 1: Identify the right test file**

Check which existing test file covers Loto 5/40 model semantics. Run:
```bash
grep -l "Loto540Draw" tests/*.py
```
If a file exists (e.g. `tests/test_loto_540_parser.py`), append the test class there. If nothing covers the model directly, create `tests/test_loto_540_models.py`:
```python
import unittest

from loto_540_model.models import Loto540Draw


class TestLoto540DrawModel(unittest.TestCase):
    def test_super_noroc_defaults_to_none(self):
        d = Loto540Draw(date="2026-01-15", main_numbers=[2, 9, 15, 27, 34, 38])
        self.assertIsNone(d.super_noroc)

    def test_super_noroc_accepts_6_digit_string(self):
        d = Loto540Draw(
            date="2026-01-15",
            main_numbers=[2, 9, 15, 27, 34, 38],
            super_noroc="012345",
        )
        self.assertEqual(d.super_noroc, "012345")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_loto_540_models -v
```
Expected: `TypeError: __init__() got an unexpected keyword argument 'super_noroc'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/loto_540_model/models.py`:
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Loto540Draw:
    """A single Loto 5/40 draw result.

    The game draws 6 numbers from 1-40. Players pick 5 numbers and win
    by matching 4 or 5 of the 6 drawn numbers. super_noroc is the
    6-digit side game printed on the same ticket.
    """
    date: str
    main_numbers: list[int]
    super_noroc: str | None = None

    def __hash__(self) -> int:
        return hash((self.date, tuple(self.main_numbers), self.super_noroc))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_loto_540_models -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/loto_540_model/models.py tests/test_loto_540_models.py
git commit -m "feat(loto_540): add optional super_noroc field to Loto540Draw"
```

---

## Task 12: Loto 5/40 parser — extract Super Noroc

**Files:**
- Create: `tests/fixtures/loto_540_with_super_noroc.html`
- Modify: `src/loto_540_model/parser.py`
- Modify: `tests/test_loto_649_parser.py` (no — use whichever file contains `parse_loto_540_results` tests; check with `grep -l parse_loto_540_results tests/*.py`)

**HTML assumption:** Super Noroc renders as 6 single-digit image tags `<img src="/bile/super-noroc/N.png" />` in the same `content-rezultate` block. Same pattern as Loto 6/49 Noroc, just six digits and a different path.

- [ ] **Step 1: Write the failing test — create fixture**

Create `tests/fixtures/loto_540_with_super_noroc.html`:
```html
<div class="content-rezultate">
  <img src="/bile/2.png" />
  <img src="/bile/9.png" />
  <img src="/bile/15.png" />
  <img src="/bile/27.png" />
  <img src="/bile/34.png" />
  <img src="/bile/38.png" />
  <img src="/bile/super-noroc/0.png" />
  <img src="/bile/super-noroc/1.png" />
  <img src="/bile/super-noroc/2.png" />
  <img src="/bile/super-noroc/3.png" />
  <img src="/bile/super-noroc/4.png" />
  <img src="/bile/super-noroc/5.png" />
  <p>Detalii castiguri  la 5/40 din <span>15.01.2026</span></p>
</div>
<div class="content-rezultate">
  <img src="/bile/1.png" />
  <img src="/bile/5.png" />
  <img src="/bile/19.png" />
  <img src="/bile/22.png" />
  <img src="/bile/28.png" />
  <img src="/bile/40.png" />
  <p>Detalii castiguri  la 5/40 din <span>11.01.2026</span></p>
</div>
```

Identify the test file (look for `parse_loto_540_results` imports). If no test file exists for the 5/40 parser, create `tests/test_loto_540_parser.py`:
```python
import unittest
from pathlib import Path

from loto_540_model.parser import parse_loto_540_results


class TestLoto540Parser(unittest.TestCase):
    def test_parse_extracts_super_noroc_when_present(self):
        html = Path("tests/fixtures/loto_540_with_super_noroc.html").read_text(encoding="utf-8")
        draws = parse_loto_540_results(html)
        by_date = {d.date: d for d in draws}
        self.assertEqual(by_date["2026-01-15"].super_noroc, "012345")

    def test_parse_returns_none_when_super_noroc_absent(self):
        html = Path("tests/fixtures/loto_540_with_super_noroc.html").read_text(encoding="utf-8")
        draws = parse_loto_540_results(html)
        by_date = {d.date: d for d in draws}
        self.assertIsNone(by_date["2026-01-11"].super_noroc)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_loto_540_parser -v
```
Expected: `AssertionError: None != '012345'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/loto_540_model/parser.py`:
```python
import re
from datetime import datetime

from .models import Loto540Draw


def _normalize_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")


def parse_loto_540_results(html: str) -> list[Loto540Draw]:
    """Parse Loto 5/40 results from HTML.

    The game draws 6 numbers from 1-40. Super Noroc is 6 single-digit
    images in /bile/super-noroc/ when present.
    """
    draws = []
    date_pattern = re.compile(
        r"Detalii castiguri\s+la 5/40\s+din\s+<span>(\d{2}\.\d{2}\.\d{4})</span>"
    )

    for match in date_pattern.finditer(html):
        window_start = max(0, match.start() - 2000)
        window = html[window_start:match.start()]

        main_nums = [int(n) for n in re.findall(r"/bile/(\d{1,2})\.png", window)]
        super_noroc_digits = re.findall(r"/bile/super-noroc/(\d)\.png", window)

        if len(main_nums) < 6:
            continue

        date = _normalize_date(match.group(1))
        main = sorted(main_nums[-6:])
        super_noroc = (
            "".join(super_noroc_digits) if len(super_noroc_digits) == 6 else None
        )
        draws.append(Loto540Draw(date, main, super_noroc))

    return sorted(draws, key=lambda d: d.date)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_loto_540_parser -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/loto_540_model/parser.py tests/fixtures/loto_540_with_super_noroc.html tests/test_loto_540_parser.py
git commit -m "feat(loto_540): parse Super Noroc 6-digit number from loto.ro HTML"
```

---

## Task 13: Loto 5/40 storage — read/write `super_noroc` CSV column

**Files:**
- Modify: `src/loto_540_model/storage.py`
- Modify or create: a test file that tests 5/40 storage (check `grep -l loto_540_model.storage tests/*.py`; `tests/test_fetch_storage.py` is a likely host — if not, create `tests/test_loto_540_storage.py`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_loto_540_storage.py`:
```python
import csv
import tempfile
import unittest
from pathlib import Path

from loto_540_model.models import Loto540Draw
from loto_540_model.storage import append_draws, load_draws


class TestLoto540StorageSuperNoroc(unittest.TestCase):
    def test_round_trip_with_super_noroc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draws.csv"
            draws = [
                Loto540Draw(
                    date="2026-01-15",
                    main_numbers=[2, 9, 15, 27, 34, 38],
                    super_noroc="012345",
                ),
            ]
            append_draws(path, draws)
            loaded = load_draws(path)
            self.assertEqual(loaded[0].super_noroc, "012345")

    def test_load_legacy_csv_without_super_noroc_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            path.write_text(
                "date,main_1,main_2,main_3,main_4,main_5,main_6\n"
                "2026-01-15,2,9,15,27,34,38\n",
                encoding="utf-8",
            )
            loaded = load_draws(path)
            self.assertIsNone(loaded[0].super_noroc)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_loto_540_storage -v
```
Expected: first test fails (column missing).

- [ ] **Step 3: Write minimal implementation**

Replace `src/loto_540_model/storage.py`:
```python
import csv
from pathlib import Path

from .models import Loto540Draw

_FIELDNAMES = ["date"] + [f"main_{i}" for i in range(1, 7)] + ["super_noroc"]


def load_draws(path: Path) -> list[Loto540Draw]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            main = [int(row[f"main_{i}"]) for i in range(1, 7)]
            super_noroc = row.get("super_noroc") or None
            rows.append(Loto540Draw(row["date"], sorted(main), super_noroc))
    return sorted(rows, key=lambda d: d.date)


def append_draws(path: Path, draws: list[Loto540Draw]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        if not exists:
            writer.writeheader()
        for draw in draws:
            writer.writerow({
                "date": draw.date,
                "main_1": draw.main_numbers[0],
                "main_2": draw.main_numbers[1],
                "main_3": draw.main_numbers[2],
                "main_4": draw.main_numbers[3],
                "main_5": draw.main_numbers[4],
                "main_6": draw.main_numbers[5],
                "super_noroc": draw.super_noroc or "",
            })
    return len(draws)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_loto_540_storage -v
PYTHONPATH=src python -m unittest -v 2>&1 | tail -5
```
Expected: both specific and full suite "OK".

Also verify live CSV readability:
```bash
PYTHONPATH=src python -c "import sys; sys.path.insert(0, 'src'); from pathlib import Path; from loto_540_model.storage import load_draws; ds = load_draws(Path('data/clean/loto_540_draws.csv')); print(len(ds), ds[-1].super_noroc)"
```

- [ ] **Step 5: Commit**

```bash
git add src/loto_540_model/storage.py tests/test_loto_540_storage.py
git commit -m "feat(loto_540): persist super_noroc in CSV with backward-compat read"
```

---

## Task 14: Results-history schema migration

**Files:**
- Create: `scripts/migrate_history_schema.py`
- Create: `tests/test_migrate_history_schema.py`

**Context:** `data/results/history.csv` has columns `date, game, strategy, picks_count, total_matches, best_match, match_total, winning_numbers`. Plans B–D will add ticket-level fields. This migration is purely additive: new columns default to empty string. Nothing reads the new columns yet, so no downstream code breaks.

- [ ] **Step 1: Write the failing test**

Create `tests/test_migrate_history_schema.py`:
```python
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.migrate_history_schema import migrate

NEW_COLUMNS = [
    "ticket_id",
    "variants_count",
    "side_game_match",
    "side_game_digits",
    "winning_side_game",
    "builder_name",
]


class TestMigrateHistorySchema(unittest.TestCase):
    def _write_legacy(self, path: Path) -> None:
        path.write_text(
            "date,game,strategy,picks_count,total_matches,best_match,match_total,winning_numbers\n"
            "2026-02-12,joker,recommended,5,4,2,5,\"6, 9, 18, 22, 27 + J11\"\n"
            "2026-03-05,joker,mix1,5,4,2,5,\"6, 15, 22, 33, 43 + J16\"\n",
            encoding="utf-8",
        )

    def test_adds_new_columns_with_empty_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            self._write_legacy(path)
            migrate(path)
            with path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            for col in NEW_COLUMNS:
                self.assertIn(col, reader.fieldnames)
                for row in rows:
                    self.assertEqual(row[col], "")

    def test_preserves_existing_column_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            self._write_legacy(path)
            migrate(path)
            with path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["date"], "2026-02-12")
            self.assertEqual(rows[0]["game"], "joker")
            self.assertEqual(rows[0]["total_matches"], "4")

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            self._write_legacy(path)
            migrate(path)
            migrate(path)
            with path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            # Each new column should appear exactly once
            self.assertEqual(len([c for c in reader.fieldnames if c == "ticket_id"]), 1)
            self.assertEqual(len(rows), 2)

    def test_empty_file_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            path.touch()
            migrate(path)
            self.assertEqual(path.read_text(), "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_migrate_history_schema -v
```
Expected: `ModuleNotFoundError: No module named 'scripts.migrate_history_schema'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/migrate_history_schema.py`:
```python
"""One-shot migration: add ticket-aware columns to data/results/history.csv.

The new columns default to empty strings; this is purely additive and
does not affect any existing consumer of the file. Plans B–D populate
the new columns going forward.

Usage:
    PYTHONPATH=src python scripts/migrate_history_schema.py

Safe to re-run: idempotent.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

NEW_COLUMNS = [
    "ticket_id",
    "variants_count",
    "side_game_match",
    "side_game_digits",
    "winning_side_game",
    "builder_name",
]


def migrate(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_columns = list(reader.fieldnames or [])
        rows = list(reader)

    missing = [c for c in NEW_COLUMNS if c not in existing_columns]
    if not missing:
        return

    all_columns = existing_columns + missing
    for row in rows:
        for col in missing:
            row[col] = ""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    path = Path("data/results/history.csv")
    migrate(path)
    print(f"migrated {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_migrate_history_schema -v
```
Expected: 4 tests pass.

Now run the migration on the live file:
```bash
PYTHONPATH=src python scripts/migrate_history_schema.py
head -2 data/results/history.csv
```
Expected: `head -2` shows the header with the 6 new columns and the first row with trailing empty fields.

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_history_schema.py tests/test_migrate_history_schema.py data/results/history.csv
git commit -m "chore: migrate history.csv with ticket-aware columns (empty defaults)"
```

---

## Final verification

- [ ] **Run the full test suite:**

```bash
PYTHONPATH=src python -m unittest -v 2>&1 | tail -30
```
Expected: "OK" with no failures.

- [ ] **Confirm no orchestrator or workflow behavior changed:**

```bash
PYTHONPATH=src python scripts/generate_recommended_picks.py --budget 40 --seed 42 --mixes 3 --output-dir /tmp/picks-plan-a
ls /tmp/picks-plan-a/
```
Expected: same files emitted as before (`joker_mix1.txt`, `loto649_mix1.txt`, `loto540_mix1.txt`, etc.). This plan does not touch the orchestrator; if output differs, something regressed.

- [ ] **Push branch and open PR:**

```bash
git push -u origin feature/jackpot-a-ticket-foundation
gh pr create --fill --base main
```

PR description must include:
- Summary of the 3 new modules + 9 modified files
- Note that `TICKET_PRICE_NEEDS_VERIFICATION = True` is intentional and will be flipped in Plan C after human-in-the-loop ticket price confirmation
- Note that Plans B/C/D will consume these types but A alone is non-functional from a user-visible perspective

---

## Risks & open questions

1. **HTML fixture accuracy** (Tasks 6, 9, 12). The parsers assume specific image-path patterns for Noroc Plus, Noroc, and Super Noroc balls. If live loto.ro HTML uses different paths (e.g. `/bile/noroc_plus/` with underscore, or a single image with the full digit string in the filename), the parsers will return `None` instead of the number. Mitigation: after merging, a human should fetch a live HTML page and verify extraction before the first scheduled run that relies on side-game data (Plan D's territory).
2. **Noroc digit format**. The plan assumes 7 single-digit images for Noroc; if loto.ro groups them into one 7-digit image, the regex must be updated. The test only validates the *fixture* — live verification is needed.
3. **CSV migration is one-way**. Old CSV files remain readable, but new files cannot be read by pre-migration code. Downstream tools outside this repo (if any) that read `data/clean/*.csv` directly need updating.
4. **All prices are user-confirmed** as of 2026-04-18. `TICKET_PRICE_NEEDS_VERIFICATION = False`. If loto.ro changes prices, only `src/shared/pricing.py` needs updating.
5. **Legacy `game_recommender.TICKET_COSTS` has wrong per-variant values** (8/6/4 instead of the confirmed 7/8/5). These feed `ev_calculator` break-even math and the current budget allocator. Plan A intentionally does **not** touch them — updating them breaks callers. Plan C must migrate callers to `shared.pricing` and delete the legacy constants in a single commit with a full-suite green run.
6. **40 RON budget implications** (for Plan C). With confirmed prices, full tickets cost 17.5 Joker / 28.5 Loto 6/49 / 22.5 Loto 5/40. Feasible full-ticket combinations within 40 RON:
   - 1 Joker (17.5) + 1 Loto 5/40 (22.5) = **40.0** exact
   - 1 Loto 6/49 (28.5) alone = 28.5 (11.5 unspent; fits 1 extra Joker variant + fee = 7.5, or nothing useful)
   - 1 Joker (17.5) + 1 Loto 6/49 (28.5) = 46.0 — **over budget**
   - 2 Joker (35.0) + nothing else = 35.0 (5 RON unspent)
   All three games in one draw requires ≥ 68.5 RON. Plan C's allocator will need to surface this: either the user bumps the default budget or the "mix" concept collapses to at most 2 active games per draw.
