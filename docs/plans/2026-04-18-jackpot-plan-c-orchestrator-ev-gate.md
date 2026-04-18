# Jackpot Redesign — Plan C: Orchestrator & EV Gate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the orchestrator to allocate physical tickets (not lines), emit `picks/tickets.json` + the Ticket-aware `check_results.py`, add a budget-ledger-backed EV skip/boost gate, and remove the legacy per-variant `TICKET_COSTS` constants.

**Architecture:** New `src/shared/ticket_allocator.py` enumerates ticket bundles that fit a budget. `scripts/generate_recommended_picks.py` loses its per-line generators and calls `TicketBuilder` instances from Plan B, writing a structured JSON artifact. `scripts/check_results.py` parses that JSON, scores main + side-game matches, writes richer history rows. The EV gate grows skip/boost semantics tied to a new `budget_ledger` table in `shared.results_db`.

**Tech Stack:** Plan A + B modules. No new dependencies.

**Prerequisites:** Plans A and B merged. Feature branch: `feature/jackpot-c-orchestrator-ev-gate`.

**Scope boundary:**
- DO: allocator, orchestrator rewrite, `tickets.json`, check_results side-game scoring, ledger, legacy-cost removal.
- DO NOT: change Telegram formatter or GitHub Actions yaml (Plan D). Do not alter historical CSV fixtures beyond the columns Plan A migrated.

---

## File Structure

| Status | Path | Responsibility |
|---|---|---|
| NEW | `src/shared/ticket_allocator.py` | Enumerate ticket bundles fitting a budget; EV skip/boost decision |
| NEW | `src/shared/ticket_io.py` | `tickets.json` serialize / deserialize (dict ↔ Ticket) |
| NEW | `src/shared/budget_ledger.py` | Read/write skipped-draw credit ledger |
| MODIFY | `src/shared/results_db.py` | Add `budget_ledger` table + Ticket-aware `generated_tickets` columns |
| MODIFY | `src/shared/ev_calculator.py` | Read ticket costs from `shared.pricing`, not hardcoded 8/6/4 |
| MODIFY | `src/shared/game_recommender.py` | `TICKET_COSTS` becomes a deprecation-only re-export of `shared.pricing.PRICE_PER_VARIANT` |
| MODIFY | `scripts/generate_recommended_picks.py` | Rewrite `_generate_picks_for_allocation` to emit `Ticket` objects + `tickets.json` |
| MODIFY | `scripts/check_results.py` | Add `parse_tickets_json`, side-game match scoring, extended history CSV rows |
| NEW | `tests/test_ticket_allocator.py` | allocator + EV gate decisions |
| NEW | `tests/test_ticket_io.py` | JSON round-trip |
| NEW | `tests/test_budget_ledger.py` | ledger |
| MODIFY | `tests/test_results_db.py` | new schema coverage |
| MODIFY | `tests/test_generate_recommended_picks.py` | ticket-output fixture |
| MODIFY | `tests/test_check_results.py` | tickets.json + side-game scoring fixture |

---

## Task 1: Ticket allocator — enumerate feasible bundles

**Files:**
- Create: `src/shared/ticket_allocator.py`
- Create: `tests/test_ticket_allocator.py`

The allocator enumerates non-empty ticket combinations `(n_joker, n_649, n_540)` such that `total_cost ≤ budget`, and ranks them by the metric the caller chooses (max P(any win) for baseline, max E[jackpot-hit] for CoreShare mode).

- [ ] **Step 1: Write the failing test**

Create `tests/test_ticket_allocator.py`:
```python
import unittest

from shared.ticket_allocator import TicketAllocation, enumerate_allocations, best_allocation


class TestTicketAllocator(unittest.TestCase):
    def test_40_ron_fits_joker_plus_loto_540_full(self):
        allocs = enumerate_allocations(budget_ron=40.0)
        # Expect: 1 joker + 1 loto_540 = 17.5 + 22.5 = 40.0
        self.assertIn(
            TicketAllocation(tickets={"joker": 1, "loto_649": 0, "loto_540": 1}, total_cost=40.0),
            allocs,
        )

    def test_40_ron_excludes_joker_plus_loto_649(self):
        allocs = enumerate_allocations(budget_ron=40.0)
        joker_plus_649 = [
            a for a in allocs
            if a.tickets.get("joker", 0) == 1
            and a.tickets.get("loto_649", 0) == 1
            and a.tickets.get("loto_540", 0) == 0
        ]
        # 17.5 + 28.5 = 46 — over budget
        self.assertEqual(joker_plus_649, [])

    def test_zero_allocation_excluded(self):
        allocs = enumerate_allocations(budget_ron=40.0)
        for a in allocs:
            self.assertGreater(sum(a.tickets.values()), 0)

    def test_under_cheapest_ticket_returns_empty(self):
        allocs = enumerate_allocations(budget_ron=17.0)  # cheapest = 17.5 Joker
        self.assertEqual(allocs, [])

    def test_best_allocation_prefers_more_tickets_at_equal_probability(self):
        # At 40 RON, the higher-ticket-count allocation should win the
        # P(any win) tiebreaker against a single-ticket allocation.
        best = best_allocation(budget_ron=40.0)
        self.assertGreater(sum(best.tickets.values()), 1)

    def test_allowed_games_filter(self):
        allocs = enumerate_allocations(budget_ron=40.0, allowed_games={"joker"})
        for a in allocs:
            self.assertEqual(a.tickets.get("loto_649", 0), 0)
            self.assertEqual(a.tickets.get("loto_540", 0), 0)

    def test_total_cost_exact(self):
        alloc = TicketAllocation(tickets={"joker": 2, "loto_649": 0, "loto_540": 0}, total_cost=0.0)
        # recompute externally
        self.assertEqual(alloc.recompute_total_cost(), 35.0)  # 2 × 17.5


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_allocator -v
```
Expected: `ModuleNotFoundError: No module named 'shared.ticket_allocator'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/shared/ticket_allocator.py`:
```python
"""Ticket-level budget allocator.

Enumerates feasible (n_joker, n_649, n_540) ticket combinations under
a RON budget and ranks by P(any win) across the combined independent
tickets. Replaces game_recommender.optimize_budget which ranked by
per-variant tickets using stale pricing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

from .pricing import compute_ticket_cost


_GAMES = ("joker", "loto_649", "loto_540")


@dataclass(frozen=True)
class TicketAllocation:
    tickets: dict[str, int]
    total_cost: float
    p_any_win: float = 0.0

    def __post_init__(self) -> None:
        for g in _GAMES:
            if self.tickets.get(g, 0) < 0:
                raise ValueError(f"negative ticket count for {g}")

    def recompute_total_cost(self) -> float:
        return sum(
            compute_ticket_cost(game) * count
            for game, count in self.tickets.items()
            if count > 0
        )


# Per-variant single-line win probabilities (standard combinatorics).
# For ticket-level P(any win) we combine across variants inside the
# ticket using (1 - (1 - p_variant)**variants) per ticket.
def _variant_win_prob(game: str) -> float:
    if game == "joker":
        total = comb(45, 5)
        main_probs = {m: comb(5, m) * comb(40, 5 - m) / total for m in range(6)}
        p_joker = 1 / 20
        return (
            main_probs[3] + main_probs[4] + main_probs[5]
            + main_probs[2] * p_joker + main_probs[1] * p_joker
        )
    if game == "loto_649":
        total = comb(49, 6)
        return sum(comb(6, m) * comb(43, 6 - m) / total for m in range(3, 7))
    # loto_540: pick 5 from 40, 6 drawn, win at 4+ main matches
    total = comb(40, 5)
    return sum(comb(6, m) * comb(34, 5 - m) / total for m in range(4, 6))


_VARIANTS_PER_TICKET = {"joker": 2, "loto_649": 3, "loto_540": 4}


def _p_ticket_any_win(game: str) -> float:
    p = _variant_win_prob(game)
    v = _VARIANTS_PER_TICKET[game]
    return 1.0 - (1.0 - p) ** v


def _p_any_win(tickets: dict[str, int]) -> float:
    p_none = 1.0
    for game, count in tickets.items():
        if count <= 0:
            continue
        p_loss = (1.0 - _p_ticket_any_win(game)) ** count
        p_none *= p_loss
    return 1.0 - p_none


def enumerate_allocations(
    budget_ron: float,
    allowed_games: set[str] | None = None,
    max_per_game: int = 8,
) -> list[TicketAllocation]:
    allowed = set(_GAMES) if allowed_games is None else set(allowed_games) & set(_GAMES)
    costs = {g: compute_ticket_cost(g) for g in _GAMES}

    out: list[TicketAllocation] = []
    for nj in range(max_per_game + 1):
        if "joker" not in allowed and nj > 0:
            continue
        for n9 in range(max_per_game + 1):
            if "loto_649" not in allowed and n9 > 0:
                continue
            for n5 in range(max_per_game + 1):
                if "loto_540" not in allowed and n5 > 0:
                    continue
                total = nj + n9 + n5
                if total == 0:
                    continue
                cost = costs["joker"] * nj + costs["loto_649"] * n9 + costs["loto_540"] * n5
                if cost > budget_ron + 1e-9:
                    continue
                tickets = {"joker": nj, "loto_649": n9, "loto_540": n5}
                out.append(
                    TicketAllocation(
                        tickets=tickets,
                        total_cost=round(cost, 2),
                        p_any_win=_p_any_win(tickets),
                    )
                )
    return out


def best_allocation(
    budget_ron: float,
    allowed_games: set[str] | None = None,
) -> TicketAllocation:
    allocs = enumerate_allocations(budget_ron=budget_ron, allowed_games=allowed_games)
    if not allocs:
        return TicketAllocation(
            tickets={"joker": 0, "loto_649": 0, "loto_540": 0},
            total_cost=0.0,
            p_any_win=0.0,
        )
    return max(
        allocs,
        key=lambda a: (
            a.p_any_win,
            sum(a.tickets.values()),
            a.total_cost,
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_allocator -v
```
Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/ticket_allocator.py tests/test_ticket_allocator.py
git commit -m "feat(shared): add ticket-level budget allocator using confirmed prices"
```

---

## Task 2: Ticket JSON I/O

**Files:**
- Create: `src/shared/ticket_io.py`
- Create: `tests/test_ticket_io.py`

Structured artifact format:
```json
{
  "generated_at": "2026-04-19T10:00:00Z",
  "budget_ron": 40.0,
  "total_cost_ron": 40.0,
  "allocation": {"joker": 1, "loto_649": 0, "loto_540": 1},
  "tickets": [
    {
      "game": "joker",
      "variants": [
        {"main_numbers": [3, 7, 12, 19, 28], "bonus_number": 11},
        {"main_numbers": [3, 7, 12, 19, 33], "bonus_number": 11}
      ],
      "side_game_number": "NP14",
      "strategy": "core_share",
      "cost_ron": 17.5
    }
  ]
}
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_ticket_io.py`:
```python
import json
import tempfile
import unittest
from pathlib import Path

from shared.ticket import Ticket, Variant
from shared.ticket_io import dump_tickets, load_tickets


def _joker_ticket() -> Ticket:
    return Ticket(
        game="joker",
        variants=(
            Variant((3, 7, 12, 19, 28), 11, "joker"),
            Variant((3, 7, 12, 19, 33), 11, "joker"),
        ),
        side_game_number="NP14",
        strategy="core_share",
        cost_ron=17.5,
    )


class TestTicketIO(unittest.TestCase):
    def test_round_trip_single_ticket(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tickets.json"
            dump_tickets(
                path,
                tickets=[_joker_ticket()],
                budget_ron=40.0,
                total_cost_ron=17.5,
                allocation={"joker": 1, "loto_649": 0, "loto_540": 0},
                generated_at="2026-04-19T10:00:00Z",
            )
            loaded = load_tickets(path)
            self.assertEqual(len(loaded["tickets"]), 1)
            t = loaded["tickets"][0]
            self.assertEqual(t.game, "joker")
            self.assertEqual(t.side_game_number, "NP14")
            self.assertEqual(t.variants[0].main_numbers, (3, 7, 12, 19, 28))
            self.assertEqual(t.variants[0].bonus_number, 11)
            self.assertEqual(loaded["budget_ron"], 40.0)
            self.assertEqual(loaded["allocation"]["joker"], 1)

    def test_json_preserves_leading_zero_side_game(self):
        t = Ticket(
            game="joker",
            variants=(
                Variant((1, 2, 3, 4, 5), 1, "joker"),
                Variant((6, 7, 8, 9, 10), 2, "joker"),
            ),
            side_game_number="NP07",
            strategy="independent",
            cost_ron=17.5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tickets.json"
            dump_tickets(path, [t], budget_ron=40.0, total_cost_ron=17.5, allocation={"joker": 1}, generated_at="x")
            raw = json.loads(path.read_text())
            self.assertEqual(raw["tickets"][0]["side_game_number"], "NP07")

    def test_load_raises_on_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tickets.json"
            path.write_text("{ not valid json")
            with self.assertRaises(ValueError):
                load_tickets(path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_io -v
```
Expected: `ModuleNotFoundError: No module named 'shared.ticket_io'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/shared/ticket_io.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_ticket_io -v
```
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/ticket_io.py tests/test_ticket_io.py
git commit -m "feat(shared): add tickets.json serializer/loader for orchestrator handoff"
```

---

## Task 3: Budget ledger

**Files:**
- Create: `src/shared/budget_ledger.py`
- Create: `tests/test_budget_ledger.py`

Simple file-backed JSON ledger: each skipped draw credits the ledger; each boosted draw debits it. Kept on disk at `picks/budget_bank.json` so it rides workflow artifacts. Not in the database until Task 5 schema change, because the ledger needs to survive workflow runs without DB availability.

- [ ] **Step 1: Write the failing test**

Create `tests/test_budget_ledger.py`:
```python
import tempfile
import unittest
from pathlib import Path

from shared.budget_ledger import BudgetLedger


class TestBudgetLedger(unittest.TestCase):
    def test_new_ledger_has_zero_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = BudgetLedger(Path(tmp) / "ledger.json")
            self.assertEqual(led.balance(), 0.0)
            self.assertEqual(led.entries(), [])

    def test_credit_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = BudgetLedger(Path(tmp) / "ledger.json")
            led.credit_skip(draw_date="2026-04-20", amount=40.0, reason="low-ev")
            self.assertEqual(led.balance(), 40.0)
            es = led.entries()
            self.assertEqual(len(es), 1)
            self.assertEqual(es[0]["kind"], "credit")
            self.assertEqual(es[0]["reason"], "low-ev")

    def test_debit_boost_cannot_exceed_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = BudgetLedger(Path(tmp) / "ledger.json")
            led.credit_skip(draw_date="2026-04-20", amount=40.0, reason="low-ev")
            actual = led.debit_boost(draw_date="2026-05-04", amount=100.0, reason="jackpot")
            self.assertEqual(actual, 40.0)  # clipped to balance
            self.assertEqual(led.balance(), 0.0)

    def test_round_trip_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            led = BudgetLedger(path)
            led.credit_skip(draw_date="2026-04-20", amount=40.0, reason="low-ev")
            led2 = BudgetLedger(path)
            self.assertEqual(led2.balance(), 40.0)

    def test_debit_without_balance_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            led = BudgetLedger(Path(tmp) / "ledger.json")
            actual = led.debit_boost(draw_date="d", amount=10.0, reason="r")
            self.assertEqual(actual, 0.0)
            self.assertEqual(led.balance(), 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_budget_ledger -v
```
Expected: `ModuleNotFoundError: No module named 'shared.budget_ledger'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/shared/budget_ledger.py`:
```python
"""File-backed budget ledger for EV-skip/boost accounting.

On a low-EV draw the orchestrator skips play and credits the ledger
with the draw's normal budget. On a rollover / high-EV draw the
orchestrator debits the ledger to increase the effective budget.

Balance is clipped to non-negative; a debit greater than balance
succeeds with a partial amount equal to the current balance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BudgetLedger:
    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"balance": 0.0, "entries": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"balance": 0.0, "entries": []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def balance(self) -> float:
        return float(self._data.get("balance", 0.0))

    def entries(self) -> list[dict[str, Any]]:
        return list(self._data.get("entries", []))

    def credit_skip(self, draw_date: str, amount: float, reason: str) -> None:
        if amount <= 0:
            return
        self._data["balance"] = self.balance() + amount
        self._data["entries"].append(
            {
                "kind": "credit",
                "draw_date": draw_date,
                "amount": amount,
                "reason": reason,
            }
        )
        self._save()

    def debit_boost(self, draw_date: str, amount: float, reason: str) -> float:
        if amount <= 0:
            return 0.0
        actual = min(amount, self.balance())
        if actual <= 0:
            return 0.0
        self._data["balance"] = self.balance() - actual
        self._data["entries"].append(
            {
                "kind": "debit",
                "draw_date": draw_date,
                "amount": actual,
                "requested": amount,
                "reason": reason,
            }
        )
        self._save()
        return actual
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_budget_ledger -v
```
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/budget_ledger.py tests/test_budget_ledger.py
git commit -m "feat(shared): add file-backed budget ledger for EV skip/boost"
```

---

## Task 4: Migrate `ev_calculator` to `shared.pricing`

**Files:**
- Modify: `src/shared/ev_calculator.py`
- Modify: `tests/test_ev_calculator.py` (exists or create if absent; check with `grep -l ev_calculator tests/*.py`)

The EV calculator's factory methods take a hardcoded `ticket_cost`. Change the default to come from `shared.pricing.compute_ticket_cost(game, include_side_game=False, include_fee=True)` — the *main-ticket* cost, without side game (EV tiers are for the main game only; side games have separate prize ladders).

- [ ] **Step 1: Identify & adjust test**

```bash
grep -l "create_loto_649\|create_joker\|create_loto_540" tests/*.py
```

Append to the identified test file (or create `tests/test_ev_pricing_integration.py`):
```python
import unittest

from shared.ev_calculator import EVCalculator


class TestEVCalculatorUsesSharedPricing(unittest.TestCase):
    def test_joker_default_cost_matches_pricing_module(self):
        game = EVCalculator.create_joker()
        # 2 variants × 7.0 + 0.5 fee (no side game in EV main tiers) = 14.5
        self.assertEqual(game.ticket_cost, 14.5)

    def test_loto_649_default_cost_matches_pricing_module(self):
        game = EVCalculator.create_loto_649()
        # 3 × 8.0 + 0.5 = 24.5
        self.assertEqual(game.ticket_cost, 24.5)

    def test_loto_540_default_cost_matches_pricing_module(self):
        game = EVCalculator.create_loto_540()
        # 4 × 5.0 + 0.5 = 20.5
        self.assertEqual(game.ticket_cost, 20.5)

    def test_explicit_override_still_works(self):
        game = EVCalculator.create_joker(ticket_cost=99.0)
        self.assertEqual(game.ticket_cost, 99.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_ev_pricing_integration -v
```
Expected: default values match the *old* hardcoded 8/6/4; new test fails.

- [ ] **Step 3: Write minimal implementation**

Edit `src/shared/ev_calculator.py`. Change the three factory method signatures:

Replace:
```python
    @staticmethod
    def create_loto_649(ticket_cost: float = 6.0) -> LotteryGame:
```
With:
```python
    @staticmethod
    def create_loto_649(ticket_cost: float | None = None) -> LotteryGame:
        if ticket_cost is None:
            from .pricing import compute_ticket_cost
            ticket_cost = compute_ticket_cost("loto_649", include_side_game=False)
```

Apply the same pattern to `create_loto_540` and `create_joker`. Update the docstring to note that side-game cost is excluded because EV tiers only cover the main game.

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src python -m unittest tests.test_ev_pricing_integration -v
PYTHONPATH=src python -m unittest -v 2>&1 | tail -10
```
Expected: new tests pass, full suite remains OK. **Any existing test that asserted 8.0/6.0/4.0 will fail here** — update those tests to the new defaults (14.5/24.5/20.5) in the same commit. This is expected and correct.

- [ ] **Step 5: Commit**

```bash
git add src/shared/ev_calculator.py tests/test_ev_pricing_integration.py
git commit -m "feat(ev): derive ticket cost from shared.pricing (main ticket, no side game)"
```

---

## Task 5: Replace `game_recommender.TICKET_COSTS`

**Files:**
- Modify: `src/shared/game_recommender.py`
- Modify: `scripts/generate_recommended_picks.py` (imports)

`TICKET_COSTS` is imported by `generate_recommended_picks.py`. Remove it by making `game_recommender`'s public API stop exposing it, and redirect callers to `shared.pricing`. Keep the functions `calculate_win_probability`, `top_diverse_allocations`, etc. since Plan D still needs them for the non-ticket "mixes" fallback path — though Task 6 removes that path too. Delete in this order:

- [ ] **Step 1: Identify every use**

```bash
grep -rn "TICKET_COSTS" src/ scripts/ tests/
```
All hits must be either removed or rewritten to `compute_ticket_cost`.

- [ ] **Step 2: Edit `src/shared/game_recommender.py`**

Delete the `TICKET_COSTS` dict. Replace every internal `TICKET_COSTS[game]` with `compute_ticket_cost(game, include_side_game=False)` — using the main-ticket cost (variants + fee, no side game). Update `GameProbability.ticket_cost` to read from `compute_ticket_cost` in `calculate_win_probability`.

For the orchestrator's interim compatibility, preserve the module-level name as a deprecated alias:
```python
# Deprecated: kept for one release cycle. Use shared.pricing.compute_ticket_cost.
def _legacy_ticket_costs() -> dict[str, float]:
    from .pricing import compute_ticket_cost
    return {g: compute_ticket_cost(g, include_side_game=False) for g in ("joker", "loto_649", "loto_540")}


TICKET_COSTS = _legacy_ticket_costs()
```
Note: values now reflect real pricing (14.5 / 24.5 / 20.5) not 8/6/4.

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=src python -m unittest -v 2>&1 | tail -30
```
Expected: failures in `test_game_recommender.py` that asserted `TICKET_COSTS["joker"] == 8.0` etc. Update those assertions to the new values. Every callsite of `optimize_budget` / `top_diverse_allocations` that previously assumed 8/6/4 costs may produce different allocations; update fixtures accordingly.

This is the noisiest task in Plan C — budget ~45 minutes on test cleanup. Pattern: update expected `tickets={"joker": N, ...}` dicts and `total_cost=...` values to reflect real pricing.

- [ ] **Step 4: Confirm full suite green**

```bash
PYTHONPATH=src python -m unittest -v 2>&1 | tail -5
```
Expected: OK.

- [ ] **Step 5: Commit**

```bash
git add src/shared/game_recommender.py tests/test_game_recommender.py
git commit -m "refactor(recommender): source ticket costs from shared.pricing"
```

---

## Task 6: Orchestrator rewrite — emit Tickets, drop lines

**Files:**
- Modify: `scripts/generate_recommended_picks.py`
- Modify: `tests/test_generate_recommended_picks.py`

The orchestrator becomes: (1) EV gate, (2) allocator → `TicketAllocation`, (3) for each game's count, invoke a `TicketBuilder`, collect `Ticket[]`, (4) write `tickets.json` and the legacy-format `*.txt` files for backward-compatible Telegram formatting (deleted in Plan D).

Strategy selection per game is a new `--strategy` flag:
- `independent` (default, preserves current behavior modulo ticket grouping)
- `core_share`
- `wheel:<pool_size>`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generate_recommended_picks.py`:
```python
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGenerateTicketsJSON(unittest.TestCase):
    def test_emits_tickets_json_with_expected_allocation_at_40_ron(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "picks"
            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/generate_recommended_picks.py",
                    "--budget", "40",
                    "--seed", "42",
                    "--output-dir", str(out),
                    "--strategy", "independent",
                ],
                env={"PYTHONPATH": "src", "PATH": ""},
            )
            tickets_path = out / "tickets.json"
            self.assertTrue(tickets_path.exists())
            doc = json.loads(tickets_path.read_text())
            self.assertEqual(doc["budget_ron"], 40.0)
            self.assertGreater(len(doc["tickets"]), 0)
            # 40 RON best fits 1 joker + 1 loto_540
            games = [t["game"] for t in doc["tickets"]]
            self.assertIn("joker", games)
            self.assertIn("loto_540", games)

    def test_each_ticket_has_correct_variant_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "picks"
            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/generate_recommended_picks.py",
                    "--budget", "40",
                    "--seed", "42",
                    "--output-dir", str(out),
                    "--strategy", "core_share",
                ],
                env={"PYTHONPATH": "src", "PATH": ""},
            )
            doc = json.loads((out / "tickets.json").read_text())
            expected_variants = {"joker": 2, "loto_649": 3, "loto_540": 4}
            for t in doc["tickets"]:
                self.assertEqual(len(t["variants"]), expected_variants[t["game"]])

    def test_legacy_txt_files_still_emitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "picks"
            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/generate_recommended_picks.py",
                    "--budget", "40",
                    "--seed", "42",
                    "--output-dir", str(out),
                    "--strategy", "independent",
                ],
                env={"PYTHONPATH": "src", "PATH": ""},
            )
            # Plan C preserves legacy .txt for Plan D's transition.
            txts = list(out.glob("*.txt"))
            self.assertGreater(len(txts), 0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_generate_recommended_picks.TestGenerateTicketsJSON -v
```
Expected: fails — tickets.json does not exist.

- [ ] **Step 3: Rewrite the orchestrator**

Replace `scripts/generate_recommended_picks.py`. Key structural changes:

1. Replace all `_generate_picks_for_allocation` logic with a new `_build_tickets_for_allocation(allocation, strategy_spec, rng, half_life, half_life_mode)` that:
   - Loads game draws per existing `load_joker_draws` etc.
   - For each game with `count > 0`, instantiates a `TicketBuilder` matching `strategy_spec` and calls `builder.build(ctx)` `count` times (advancing the rng each ticket).
   - Collects all `Ticket` objects into one list.

2. After tickets are built:
   - Serialize to `picks/tickets.json` via `shared.ticket_io.dump_tickets`.
   - Write one legacy `.txt` file per game summarizing the main variants (numbered list preserving the current `"{idx}. {nums} + J{joker}"` format) so Plan D's Telegram renderer keeps working during transition.
   - Keep the `persist_generation_run` call, passing ticket dicts.

3. Add `--strategy` argument:
   ```python
   parser.add_argument(
       "--strategy",
       type=str,
       default="independent",
       help="independent | core_share | wheel:<pool_size> (default: independent)",
   )
   ```
   Parse the string into a `TicketBuilder` instance:
   ```python
   def _builder_for_spec(spec: str):
       from shared.ticket_builders import CoreShareBuilder, IndependentBuilder, WheelBuilder
       if spec == "independent":
           return IndependentBuilder(n_tickets=1)
       if spec == "core_share":
           return CoreShareBuilder()
       if spec.startswith("wheel:"):
           return WheelBuilder(pool_size=int(spec.split(":", 1)[1]))
       raise SystemExit(f"unknown strategy: {spec}")
   ```

4. Replace the call to `optimize_budget` (which uses stale per-line math) with `shared.ticket_allocator.best_allocation(budget_ron, allowed_games=allowed)`. The `--mixes N` option now enumerates top N allocations via `enumerate_allocations` sorted by `p_any_win`; for each it emits a separate `tickets_mixN.json` file.

5. Delete the obsolete helpers `generate_joker_picks`, `generate_649_picks`, `generate_540_picks` (their logic now lives inside the `TicketBuilder` subclasses). Keep `load_joker_draws` / `load_loto_649_draws` / `load_loto_540_draws`.

Concrete replacement for the inner "write output" section:
```python
from shared.ticket_io import dump_tickets
from datetime import datetime, UTC

generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
dump_tickets(
    output_dir / "tickets.json",
    tickets=tickets,
    budget_ron=args.budget,
    total_cost_ron=allocation.total_cost,
    allocation=allocation.tickets,
    generated_at=generated_at,
)

# Legacy .txt — one file per game, numbered variants. Plan D deletes this.
for game, label in (("joker", "joker"), ("loto_649", "loto649"), ("loto_540", "loto540")):
    game_tickets = [t for t in tickets if t.game == game]
    if not game_tickets:
        continue
    lines: list[str] = []
    idx = 1
    for t in game_tickets:
        for v in t.variants:
            main = ", ".join(str(n) for n in v.main_numbers)
            if game == "joker":
                lines.append(f"{idx}. {main} + J{v.bonus_number}")
            else:
                lines.append(f"{idx}. {main}")
            idx += 1
    (output_dir / f"{label}.txt").write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=src python -m unittest tests.test_generate_recommended_picks -v
PYTHONPATH=src python -m unittest -v 2>&1 | tail -10
```
Expected: new TestGenerateTicketsJSON passes. **Existing tests that parsed the exact text output may break** — update them to read from `tickets.json` where appropriate.

Integration sanity check:
```bash
mkdir -p /tmp/picks-plan-c
PYTHONPATH=src python scripts/generate_recommended_picks.py \
  --budget 40 --seed 42 --strategy independent --output-dir /tmp/picks-plan-c
cat /tmp/picks-plan-c/tickets.json | python -m json.tool | head -30
```
Expected: valid JSON with `tickets` array, each with `variants` + `side_game_number`.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_recommended_picks.py tests/test_generate_recommended_picks.py
git commit -m "refactor(orchestrator): emit Ticket JSON + legacy txt shim for Plan D"
```

---

## Task 7: EV gate skip/boost

**Files:**
- Modify: `scripts/generate_recommended_picks.py`
- Modify: `tests/test_generate_recommended_picks.py`

Three new flags:
- `--ev-skip-ratio` (float, default 0.5): if every game's jackpot/breakeven ratio is below this, skip the entire draw and credit the ledger.
- `--ev-boost-ratio` (float, default 1.2): if any game's ratio exceeds this, debit up to `args.budget` from the ledger and proportionally increase the effective budget.
- `--ledger-path` (str, default `picks/budget_bank.json`): location of the ledger.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generate_recommended_picks.py`:
```python
class TestEVSkipBoost(unittest.TestCase):
    def test_skip_when_all_ratios_below_skip_ratio(self):
        import subprocess, sys, tempfile, json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "picks"
            # All jackpots zero → ratio 0 → skip
            subprocess.check_call([
                sys.executable, "scripts/generate_recommended_picks.py",
                "--budget", "40", "--seed", "42", "--output-dir", str(out),
                "--ev-gate", "--ev-skip-ratio", "0.5",
                "--joker-jackpot", "0",
                "--loto649-jackpot", "0",
                "--loto540-jackpot", "0",
                "--ledger-path", str(out / "ledger.json"),
            ], env={"PYTHONPATH": "src", "PATH": ""})
            # Expect: no tickets.json emitted, ledger has +40 credit
            self.assertFalse((out / "tickets.json").exists())
            led = json.loads((out / "ledger.json").read_text())
            self.assertEqual(led["balance"], 40.0)

    def test_boost_when_ratio_above_boost_ratio(self):
        import subprocess, sys, tempfile, json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "picks"
            ledger_path = out / "ledger.json"
            # Pre-credit the ledger
            out.mkdir(parents=True)
            ledger_path.write_text(json.dumps({"balance": 40.0, "entries": []}))
            # Large jackpot that should push ratio > 1.2 (exact threshold depends on game)
            subprocess.check_call([
                sys.executable, "scripts/generate_recommended_picks.py",
                "--budget", "40", "--seed", "42", "--output-dir", str(out),
                "--ev-gate", "--ev-boost-ratio", "1.2",
                "--joker-jackpot", "50000000",
                "--ledger-path", str(ledger_path),
            ], env={"PYTHONPATH": "src", "PATH": ""})
            # Expect: tickets.json emitted with budget > 40
            doc = json.loads((out / "tickets.json").read_text())
            self.assertGreater(doc["budget_ron"], 40.0)
            # Ledger debited
            led = json.loads(ledger_path.read_text())
            self.assertLess(led["balance"], 40.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_generate_recommended_picks.TestEVSkipBoost -v
```
Expected: failures (skip path not wired, boost flag unknown).

- [ ] **Step 3: Implement**

In `scripts/generate_recommended_picks.py`, replace `apply_ev_gate` with `apply_ev_gate_v2` that returns either:
- `("skip", reason, credit_amount)` → caller credits ledger, does NOT emit tickets.json, persists a skip record.
- `("boost", extra_budget)` → caller debits ledger, runs allocator with `budget + extra_budget`.
- `("play", 0)` → normal allocator run at standard budget.

Pseudocode inside `main`:
```python
from shared.budget_ledger import BudgetLedger

ledger_path = Path(args.ledger_path) if getattr(args, "ledger_path", None) else output_dir / "budget_bank.json"
ledger = BudgetLedger(ledger_path) if args.ev_gate else None
effective_budget = args.budget

if args.ev_gate:
    decision = apply_ev_gate_v2(
        budget=args.budget,
        jackpots=jackpots,
        skip_ratio=args.ev_skip_ratio,
        boost_ratio=args.ev_boost_ratio,
    )
    if decision.action == "skip" and ledger is not None:
        ledger.credit_skip(draw_date=datetime.now(UTC).date().isoformat(), amount=args.budget, reason=decision.reason)
        persist_generation_run(...)  # with tickets=[]
        print(f"EV gate: skip ({decision.reason}). Ledger balance now {ledger.balance():.2f} RON.")
        return
    if decision.action == "boost" and ledger is not None:
        actual_boost = ledger.debit_boost(draw_date=..., amount=args.budget, reason=decision.reason)
        effective_budget = args.budget + actual_boost
        print(f"EV gate: boost +{actual_boost:.2f} RON. Effective budget {effective_budget:.2f}.")

allocation = best_allocation(effective_budget, allowed_games=decision.allowed_games)
# ... rest unchanged
```

Implement `apply_ev_gate_v2` (module-local) that walks the three games, computes each ratio via the existing `EVCalculator._calculate_positive_ev_jackpot`, and returns the structured decision.

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=src python -m unittest tests.test_generate_recommended_picks.TestEVSkipBoost -v
```
Expected: 2 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_recommended_picks.py tests/test_generate_recommended_picks.py
git commit -m "feat(orchestrator): EV gate skip/boost using budget ledger"
```

---

## Task 8: check_results — parse tickets.json + side-game match scoring

**Files:**
- Modify: `scripts/check_results.py`
- Modify: `tests/test_check_results.py`

`parse_tickets_json(picks_dir) -> list[Ticket]` replaces the regex-based `parse_picks`. For each draw's winning main + side game, the checker computes:
- best_main_match per ticket (via `Ticket.best_main_match`)
- side_game_match for each ticket: exact string equality for Noroc/Super Noroc/Noroc Plus. If the draw's side game wasn't parsed (None), score 0.
- side_game_digits_matched: for Noroc/Super Noroc, count matching prefix digits (rightmost) — loto.ro's Noroc prize ladder awards prizes for matching the last N digits. Noroc Plus is single-ball match/no-match.

History CSV gets: `ticket_id`, `variants_count`, `side_game_match` (0/1), `side_game_digits` (int 0-7), `winning_side_game` (str), `builder_name` (str). Columns added by Plan A Task 14.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_check_results.py`:
```python
class TestCheckResultsTicketsJSON(unittest.TestCase):
    def test_parse_tickets_json(self):
        import json, tempfile
        from pathlib import Path
        from scripts.check_results import parse_tickets_json
        doc = {
            "generated_at": "x",
            "budget_ron": 40.0,
            "total_cost_ron": 17.5,
            "allocation": {"joker": 1, "loto_649": 0, "loto_540": 0},
            "tickets": [{
                "game": "joker",
                "variants": [
                    {"main_numbers": [3, 7, 12, 19, 28], "bonus_number": 11},
                    {"main_numbers": [3, 7, 12, 19, 33], "bonus_number": 11},
                ],
                "side_game_number": "NP07",
                "strategy": "core_share",
                "cost_ron": 17.5,
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tickets.json"
            path.write_text(json.dumps(doc))
            tickets = parse_tickets_json(path)
            self.assertEqual(len(tickets), 1)
            self.assertEqual(tickets[0].game, "joker")
            self.assertEqual(tickets[0].side_game_number, "NP07")

    def test_score_ticket_full_side_game_match(self):
        from scripts.check_results import score_side_game_match
        # Full match
        self.assertEqual(score_side_game_match("joker", "NP07", "NP07"), (1, 1))
        # No match
        self.assertEqual(score_side_game_match("joker", "NP07", "NP14"), (0, 0))
        # None winning
        self.assertEqual(score_side_game_match("joker", "NP07", None), (0, 0))

    def test_score_ticket_noroc_partial_digit_match(self):
        from scripts.check_results import score_side_game_match
        # 7-digit noroc; match last 4 digits
        # ticket: 1234567  winning: 9994567
        self.assertEqual(score_side_game_match("loto_649", "1234567", "9994567"), (0, 4))

    def test_score_ticket_noroc_exact_match(self):
        from scripts.check_results import score_side_game_match
        self.assertEqual(score_side_game_match("loto_649", "1234567", "1234567"), (1, 7))

    def test_score_ticket_noroc_no_match(self):
        from scripts.check_results import score_side_game_match
        self.assertEqual(score_side_game_match("loto_649", "1234567", "9999999"), (0, 0))

    def test_score_ticket_super_noroc_partial(self):
        from scripts.check_results import score_side_game_match
        self.assertEqual(score_side_game_match("loto_540", "123456", "999456"), (0, 3))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_check_results.TestCheckResultsTicketsJSON -v
```
Expected: `ImportError: cannot import name 'parse_tickets_json'`.

- [ ] **Step 3: Implement**

Add to `scripts/check_results.py`:
```python
def parse_tickets_json(path: Path) -> list:
    from shared.ticket_io import load_tickets
    doc = load_tickets(path)
    return doc["tickets"]


def score_side_game_match(game: str, ticket_side: str, winning_side: str | None) -> tuple[int, int]:
    """Returns (exact_match, digits_matched).

    exact_match is 1/0. digits_matched counts rightmost aligned digits
    (Noroc/Super Noroc prize ladder). For Joker's Noroc Plus it's the
    same as exact_match (no partial credit).
    """
    if winning_side is None:
        return 0, 0
    if game == "joker":
        return int(ticket_side == winning_side), int(ticket_side == winning_side)
    # Right-aligned digit match
    matched = 0
    for ch1, ch2 in zip(reversed(ticket_side), reversed(winning_side)):
        if ch1 == ch2:
            matched += 1
        else:
            break
    exact = int(ticket_side == winning_side)
    return exact, matched
```

Then rewrite `check_game_strategies` and the history-writing path to:
1. Prefer `picks_dir/tickets.json` when present; fall back to the legacy `game_prefix_*.txt` glob.
2. For each ticket, compute best main match AND side-game match vs the draw's winning side game.
3. Extend the history CSV row with the new columns.

Draw-side-game extraction: after `fetch_results_with_retry` returns, look up each draw's `noroc_plus` / `noroc` / `super_noroc` attribute (Plan A populates these).

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=src python -m unittest tests.test_check_results -v
```
Expected: all pass.

Integration: run the checker against Plan C's orchestrator output on a known-past draw (or skip if no fresh data).
```bash
PYTHONPATH=src PICKS_DIR=/tmp/picks-plan-c RESULTS_DIR=/tmp/results-plan-c HISTORY_CSV=/tmp/history.csv python scripts/check_results.py
head /tmp/history.csv
```
Expected: rows include the new columns.

- [ ] **Step 5: Commit**

```bash
git add scripts/check_results.py tests/test_check_results.py
git commit -m "feat(checker): parse tickets.json + score side-game matches with partial credit"
```

---

## Task 9: Extend `results_db` schema for ticket columns and ledger

**Files:**
- Modify: `src/shared/results_db.py`
- Modify: `tests/test_results_db.py`

Add to the DB what we already write to files so Grafana/ad-hoc SQL stays useful.

`generated_tickets` new columns:
- `ticket_id TEXT NOT NULL` (groups variants into a physical ticket)
- `variant_no INTEGER NOT NULL`
- `bonus_number INTEGER` (Joker only; already exists as joker_number)
- `side_game_number TEXT`
- `cost_ron REAL`
- `builder_name TEXT` (independent | core_share | wheel:N)

`check_results` new columns:
- `ticket_id TEXT`
- `side_game_match INTEGER NOT NULL DEFAULT 0`
- `side_game_digits_matched INTEGER NOT NULL DEFAULT 0`
- `winning_side_game TEXT`

New `budget_ledger_entries` table (SQLite/PG).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_results_db.py`:
```python
class TestResultsDBTicketSchema(unittest.TestCase):
    def test_ensure_schema_adds_ticket_columns(self):
        import tempfile, os
        from shared.results_db import _ResultsDB
        with tempfile.TemporaryDirectory() as tmp:
            dsn = f"sqlite:///{tmp}/test.db"
            db = _ResultsDB(dsn)
            try:
                db.ensure_schema()
                # Probe column existence via pragma
                cursor = db.connection.cursor()
                cursor.execute("PRAGMA table_info(generated_tickets)")
                cols = {row[1] for row in cursor.fetchall()}
                self.assertIn("ticket_id", cols)
                self.assertIn("variant_no", cols)
                self.assertIn("side_game_number", cols)
                self.assertIn("cost_ron", cols)
                self.assertIn("builder_name", cols)

                cursor.execute("PRAGMA table_info(check_results)")
                cols = {row[1] for row in cursor.fetchall()}
                self.assertIn("side_game_match", cols)
                self.assertIn("side_game_digits_matched", cols)
                self.assertIn("winning_side_game", cols)

                cursor.execute("PRAGMA table_info(budget_ledger_entries)")
                cols = {row[1] for row in cursor.fetchall()}
                self.assertIn("balance_after", cols)
                self.assertIn("amount", cols)
                cursor.close()
            finally:
                db.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src python -m unittest tests.test_results_db.TestResultsDBTicketSchema -v
```
Expected: missing-column errors.

- [ ] **Step 3: Implement**

In `src/shared/results_db.py` `ensure_schema`, add to the `statements` list:
```python
"""
CREATE TABLE IF NOT EXISTS budget_ledger_entries (
    id TEXT PRIMARY KEY,
    draw_date TEXT NOT NULL,
    kind TEXT NOT NULL,
    amount REAL NOT NULL,
    balance_after REAL NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
)
""",
```

For the additive columns on existing tables, adopt the idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` pattern (SQLite supports `ADD COLUMN` plus a try/except wrapper for duplicate-column errors):

```python
def _add_column_if_missing(self, table: str, column_def: str) -> None:
    column_name = column_def.split()[0]
    if self.kind == "postgres":
        self.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column_def}")
        return
    # sqlite: catch "duplicate column name"
    try:
        self.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise

# In ensure_schema, after creating the base tables:
for column in [
    "ticket_id TEXT",
    "variant_no INTEGER NOT NULL DEFAULT 0",
    "side_game_number TEXT",
    "cost_ron REAL",
    "builder_name TEXT",
]:
    self._add_column_if_missing("generated_tickets", column)

for column in [
    "ticket_id TEXT",
    "side_game_match INTEGER NOT NULL DEFAULT 0",
    "side_game_digits_matched INTEGER NOT NULL DEFAULT 0",
    "winning_side_game TEXT",
]:
    self._add_column_if_missing("check_results", column)
```

Update `persist_generation_run` and `persist_check_results` to include the new columns when inserting rows.

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=src python -m unittest tests.test_results_db -v
```
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add src/shared/results_db.py tests/test_results_db.py
git commit -m "feat(db): extend generated_tickets/check_results + budget_ledger_entries"
```

---

## Final verification

- [ ] **Full suite green:**

```bash
PYTHONPATH=src python -m unittest -v 2>&1 | tail -15
```

- [ ] **End-to-end dry run:**

```bash
rm -rf /tmp/pc-e2e && mkdir -p /tmp/pc-e2e
PYTHONPATH=src python scripts/generate_recommended_picks.py \
  --budget 40 --seed 42 --strategy core_share --output-dir /tmp/pc-e2e
PYTHONPATH=src PICKS_DIR=/tmp/pc-e2e RESULTS_DIR=/tmp/pc-e2e HISTORY_CSV=/tmp/pc-e2e/history.csv \
  python scripts/check_results.py
cat /tmp/pc-e2e/tickets.json | python -m json.tool | head -40
cat /tmp/pc-e2e/history.csv
```
Expected: tickets.json is well-formed; history.csv has new columns populated with side-game match data.

- [ ] **Confirm workflows + Telegram unchanged:**

```bash
git diff main -- .github/workflows/
```
Expected: empty.

- [ ] **Open PR:**

```bash
git push -u origin feature/jackpot-c-orchestrator-ev-gate
gh pr create --fill --base main --title "Jackpot redesign C: orchestrator + EV gate"
```

PR description must highlight:
- Legacy `TICKET_COSTS` values removed; allocator now uses confirmed prices.
- At 40 RON budget the allocator now chooses "1 Joker + 1 Loto 5/40 = 40 RON exact" by default. This is a visible change from previous runs. Flag in PR.
- `tickets.json` is the new source of truth; `.txt` shim is kept for Plan D.
- Schema changes are additive and tested idempotent.

---

## Risks & open questions

1. **Allocator tiebreakers are arbitrary.** When two allocations have equal `p_any_win`, the ranker prefers more tickets, then higher cost. This may pick 2×Joker (35 RON) over 1×Joker+1×5/40 (40 RON) if their probabilities tie. Verify with a concrete test or tune the tiebreaker.
2. **EV gate boost math is simplistic.** The current proposal debits up to `args.budget` per boost, doubling at most. True Kelly sizing would scale with edge. The ledger-backed doubling is conservative enough to not over-bet; refine in a later iteration.
3. **Skipping means zero DB inserts for that draw.** Plan D's check_results on the *following* draw will find no picks to compare against — handle this gracefully (`picks/tickets.json` may be absent when the orchestrator skipped).
4. **Running the subprocess tests requires the full `PYTHONPATH=src` env.** CI already sets it; local contributors must too.
5. **`_add_column_if_missing` SQLite path catches a generic `OperationalError`.** If the message changes across SQLite versions the idempotency guard may leak errors. Pin SQLite in CI or tighten the check.
6. **The legacy `.txt` shim produced by the orchestrator is technically redundant once Plan D lands.** Plan D deletes it; make sure Plan D's telegram reader doesn't fall back to it accidentally.
