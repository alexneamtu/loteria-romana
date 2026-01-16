# Loto.ro Any-Prize Optimizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a small, test-backed toolkit that computes any-prize probability per RON, allocates a 50 RON budget across loto.ro games, and generates unique Loto 6/49 lines. The toolkit is optional; the workflow also supports a spreadsheet-only approach.

**Architecture:** Data for odds/prices lives in a simple JSON or CSV file updated from loto.ro. Core logic computes efficiency (P(any prize)/price), runs a greedy allocation with a local swap check, and (optionally) generates unique 6/49 lines. Output is a short purchase plan plus a spreadsheet template for no-code use.

**Tech Stack:** Python 3.x (stdlib only), JSON/CSV, `unittest`, Markdown.

## Current Data Snapshot (2026-01-16)
**Sources:**
- Prices: https://www.loto.ro/?page_id=1068
- Loto 6/49 odds: https://www.loto.ro/?p=3876
- Loto 5/40 odds: https://www.loto.ro/?p=3921
- Joker odds + Noroc Plus rules: https://www.loto.ro/?p=3904

**Prices (pret varianta simpla):**
- Loto 6/49: 8.00 RON
- Joker: 7.00 RON
- Loto 5/40: 5.00 RON
- Noroc: 4.00 RON (only with Loto 6/49)
- Noroc Plus: 3.00 RON (only with Joker)
- Super Noroc: 2.00 RON (only with Loto 5/40)
- Ticket overhead listed: 0.5 RON per ticket; assume lines are grouped to minimize overhead.

**Odds tables (as published):**
- Loto 6/49: 1 in 13.983.816 (6/6), 54.200,8 (5/6), 1.032,4 (4/6), 56,66 (3/6)
- Loto 5/40: 1 in 658.008 (5/5 in first 5), 131.602 (5/6), 1.290 (4/6)
- Joker: 1 in 24.435.180, 1.221.759, 122.759, 6.109, 3.140, 157, 240, 60

**Derived any-prize probabilities:**
- Loto 6/49: 0.01863627 (1.8636%, ~1 in 53.66)
- Loto 5/40: 0.00078431 (0.0784%, ~1 in 1275.00)
- Joker: 0.02769393 (2.7694%, ~1 in 36.11)
- Noroc: 0.0010002 (0.1000%, ~1 in 999.80)
- Noroc Plus / Super Noroc: 0.0199 (1.99%, ~1 in 50.25)

**Efficiency per RON (per line):**
- Loto 6/49: 0.00232953
- Joker: 0.00395628
- Loto 5/40: 0.00015686
- Loto 6/49 + Noroc: 0.00163482
- Joker + Noroc Plus: 0.00470428
- Loto 5/40 + Super Noroc: 0.00295267

**Recommended allocation (50 RON):**
- 5 x Joker + Noroc Plus (10 RON each)
- Per-line any-prize: ~4.704% (~1 in 21.26)
- Per-draw any-prize (5 lines): ~21.41%

### Task 1: Data model + efficiency calculation

**Files:**
- Create: `src/loto_optimizer/__init__.py`
- Create: `src/loto_optimizer/models.py`
- Create: `src/loto_optimizer/optimizer.py`
- Test: `tests/test_efficiency.py`

**Step 1: Write the failing test**

```python
# tests/test_efficiency.py
import unittest

from loto_optimizer.models import GameOption
from loto_optimizer.optimizer import compute_efficiency


class TestEfficiency(unittest.TestCase):
    def test_compute_efficiency(self):
        game = GameOption(
            game_id="loto_6_49",
            label="Loto 6/49",
            line_price_ron=5.0,
            any_prize_prob=0.05,
            number_pool=49,
            numbers_per_line=6,
        )
        self.assertAlmostEqual(compute_efficiency(game), 0.01)

    def test_invalid_price(self):
        game = GameOption(
            game_id="bad",
            label="Bad",
            line_price_ron=0.0,
            any_prize_prob=0.05,
            number_pool=49,
            numbers_per_line=6,
        )
        with self.assertRaises(ValueError):
            compute_efficiency(game)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_efficiency.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loto_optimizer'` or missing `compute_efficiency`.

**Step 3: Write minimal implementation**

```python
# src/loto_optimizer/models.py
from dataclasses import dataclass


@dataclass(frozen=True)
class GameOption:
    game_id: str
    label: str
    line_price_ron: float
    any_prize_prob: float
    number_pool: int
    numbers_per_line: int
```

```python
# src/loto_optimizer/optimizer.py
from .models import GameOption


def compute_efficiency(game: GameOption) -> float:
    if game.line_price_ron <= 0:
        raise ValueError("line_price_ron must be > 0")
    if not (0 < game.any_prize_prob < 1):
        raise ValueError("any_prize_prob must be between 0 and 1")
    return game.any_prize_prob / game.line_price_ron
```

```python
# src/loto_optimizer/__init__.py
from .models import GameOption
from .optimizer import compute_efficiency

__all__ = ["GameOption", "compute_efficiency"]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_efficiency.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/loto_optimizer/__init__.py src/loto_optimizer/models.py src/loto_optimizer/optimizer.py tests/test_efficiency.py
git commit -m "feat: add game model and efficiency calculation"
```

### Task 2: Budget allocation with local swap check

**Files:**
- Modify: `src/loto_optimizer/optimizer.py`
- Test: `tests/test_allocator.py`

**Step 1: Write the failing test**

```python
# tests/test_allocator.py
import unittest

from loto_optimizer.models import GameOption
from loto_optimizer.optimizer import allocate_budget


class TestAllocator(unittest.TestCase):
    def test_swap_improves_budget_use(self):
        games = [
            GameOption("A", "A", 7.0, 0.07, 49, 6),
            GameOption("B", "B", 5.0, 0.04, 49, 6),
        ]
        counts = allocate_budget(games, budget_ron=10.0)
        self.assertEqual(counts.get("A", 0), 0)
        self.assertEqual(counts.get("B", 0), 2)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_allocator.py -v`
Expected: FAIL with `ImportError` or missing `allocate_budget`.

**Step 3: Write minimal implementation**

```python
# src/loto_optimizer/optimizer.py
from .models import GameOption


def _price_cents(value: float) -> int:
    return int(round(value * 100))


def _total_prob(counts: dict, games: list[GameOption]) -> float:
    lookup = {g.game_id: g for g in games}
    return sum(counts.get(gid, 0) * lookup[gid].any_prize_prob for gid in counts)


def _total_spend_cents(counts: dict, games: list[GameOption]) -> int:
    lookup = {g.game_id: g for g in games}
    return sum(counts.get(gid, 0) * _price_cents(lookup[gid].line_price_ron) for gid in counts)


def allocate_budget(games: list[GameOption], budget_ron: float) -> dict[str, int]:
    ranked = sorted(games, key=compute_efficiency, reverse=True)
    budget_cents = _price_cents(budget_ron)
    counts = {g.game_id: 0 for g in ranked}
    remaining = budget_cents

    for g in ranked:
        price = _price_cents(g.line_price_ron)
        buy = remaining // price
        counts[g.game_id] = int(buy)
        remaining -= int(buy) * price

    best_counts = dict(counts)
    best_prob = _total_prob(best_counts, ranked)

    if ranked:
        top = ranked[0]
        for swap_out in range(1, min(2, best_counts[top.game_id]) + 1):
            trial = dict(best_counts)
            trial[top.game_id] -= swap_out
            trial_remaining = budget_cents - _total_spend_cents(trial, ranked)

            for g in ranked[1:]:
                price = _price_cents(g.line_price_ron)
                add = trial_remaining // price
                if add:
                    trial[g.game_id] += int(add)
                    trial_remaining -= int(add) * price

            trial_prob = _total_prob(trial, ranked)
            if trial_prob > best_prob:
                best_prob = trial_prob
                best_counts = trial

    return best_counts
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_allocator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/loto_optimizer/optimizer.py tests/test_allocator.py
git commit -m "feat: add budget allocation with swap check"
```

### Task 3: Load game data from JSON with validation

**Files:**
- Create: `src/loto_optimizer/data.py`
- Modify: `src/loto_optimizer/__init__.py`
- Test: `tests/test_data.py`

**Step 1: Write the failing test**

```python
# tests/test_data.py
import json
import tempfile
import unittest

from loto_optimizer.data import load_game_options


class TestData(unittest.TestCase):
    def test_load_game_options(self):
        payload = {
            "games": [
                {
                    "game_id": "loto_6_49",
                    "label": "Loto 6/49",
                    "line_price_ron": 5.0,
                    "any_prize_prob": 0.05,
                    "number_pool": 49,
                    "numbers_per_line": 6,
                },
                {
                    "game_id": "loto_5_40",
                    "label": "Loto 5/40",
                    "line_price_ron": 4.0,
                    "any_prize_prob": 0.08,
                    "number_pool": 40,
                    "numbers_per_line": 5,
                },
            ]
        }
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name

        games = load_game_options(tmp_path)
        self.assertEqual(len(games), 2)
        self.assertEqual(games[0].game_id, "loto_6_49")

    def test_invalid_probability(self):
        payload = {
            "games": [
                {
                    "game_id": "bad",
                    "label": "Bad",
                    "line_price_ron": 4.0,
                    "any_prize_prob": 1.2,
                    "number_pool": 40,
                    "numbers_per_line": 5,
                }
            ]
        }
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name

        with self.assertRaises(ValueError):
            load_game_options(tmp_path)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_data.py -v`
Expected: FAIL with `ImportError` or missing `load_game_options`.

**Step 3: Write minimal implementation**

```python
# src/loto_optimizer/data.py
import json

from .models import GameOption
from .optimizer import compute_efficiency


def load_game_options(path: str) -> list[GameOption]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    games = []
    for item in payload.get("games", []):
        game = GameOption(
            game_id=item["game_id"],
            label=item["label"],
            line_price_ron=float(item["line_price_ron"]),
            any_prize_prob=float(item["any_prize_prob"]),
            number_pool=int(item["number_pool"]),
            numbers_per_line=int(item["numbers_per_line"]),
        )
        compute_efficiency(game)
        games.append(game)

    if not games:
        raise ValueError("No games found in data file")

    return games
```

```python
# src/loto_optimizer/__init__.py
from .data import load_game_options

__all__ = ["GameOption", "compute_efficiency", "load_game_options"]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_data.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/loto_optimizer/data.py src/loto_optimizer/__init__.py tests/test_data.py
git commit -m "feat: add JSON data loader with validation"
```

### Task 4: Unique line generator for Loto 6/49

**Files:**
- Create: `src/loto_optimizer/generator.py`
- Test: `tests/test_generator.py`

**Step 1: Write the failing test**

```python
# tests/test_generator.py
import random
import unittest

from loto_optimizer.generator import generate_unique_lines


class TestGenerator(unittest.TestCase):
    def test_generate_unique_lines(self):
        rng = random.Random(1234)
        lines = generate_unique_lines(
            count=2,
            number_pool=10,
            numbers_per_line=3,
            rng=rng,
        )
        self.assertEqual(lines, [[1, 2, 8], [1, 2, 10]])
        self.assertEqual(len({tuple(line) for line in lines}), 2)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_generator.py -v`
Expected: FAIL with `ImportError` or missing `generate_unique_lines`.

**Step 3: Write minimal implementation**

```python
# src/loto_optimizer/generator.py
import random


def generate_unique_lines(count: int, number_pool: int, numbers_per_line: int, rng=None) -> list[list[int]]:
    if count <= 0:
        return []
    if numbers_per_line <= 0 or number_pool < numbers_per_line:
        raise ValueError("Invalid pool or line size")

    rng = rng or random.Random()
    lines = set()
    max_attempts = count * 50

    attempts = 0
    while len(lines) < count and attempts < max_attempts:
        line = tuple(sorted(rng.sample(range(1, number_pool + 1), numbers_per_line)))
        lines.add(line)
        attempts += 1

    if len(lines) < count:
        raise RuntimeError("Unable to generate enough unique lines")

    return [list(line) for line in sorted(lines)]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_generator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/loto_optimizer/generator.py tests/test_generator.py
git commit -m "feat: add unique line generator"
```

### Task 5: Spreadsheet template and README workflow

**Files:**
- Create: `docs/optimizer-template.csv`
- Create: `README.md`
- Test: `tests/test_template.py`

**Step 1: Write the failing test**

```python
# tests/test_template.py
import csv
import unittest


class TestTemplate(unittest.TestCase):
    def test_template_header(self):
        with open("docs/optimizer-template.csv", "r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
        self.assertEqual(
            header,
            [
                "game_id",
                "label",
                "line_price_ron",
                "any_prize_prob",
                "number_pool",
                "numbers_per_line",
            ],
        )
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_template.py -v`
Expected: FAIL with `FileNotFoundError: docs/optimizer-template.csv`.

**Step 3: Write minimal implementation**

```csv
# docs/optimizer-template.csv
game_id,label,line_price_ron,any_prize_prob,number_pool,numbers_per_line
loto_6_49,Loto 6/49,REPLACE_WITH_PRICE,REPLACE_WITH_ODDS,49,6
loto_5_40,Loto 5/40,REPLACE_WITH_PRICE,REPLACE_WITH_ODDS,40,5
joker,Joker,REPLACE_WITH_PRICE,REPLACE_WITH_ODDS,REPLACE_WITH_RULES,REPLACE_WITH_RULES
```

```markdown
# README.md

## Loto.ro Any-Prize Optimizer

This project helps allocate a 50 RON budget across loto.ro games to maximize the probability of any prize. It does not claim any advantage over randomness.

## Data updates
- Before each draw, update prices and odds from loto.ro.
- If data is missing, default to Loto 6/49 only.

## No-code workflow
1. Fill in `docs/optimizer-template.csv` with current loto.ro odds and prices.
2. Rank games by `any_prize_prob / line_price_ron`.
3. Allocate budget with greedy + local swap (replace 1-2 top lines with the next-best option if it improves total probability).
4. Generate unique Loto 6/49 lines with a trusted RNG. For multi-pool games like Joker, use quick-pick or the official rules.

## Current allocation example (2026-01-16)
- 5 x Joker + Noroc Plus (10 RON each)

## Disclaimer
Lotteries are random. This workflow only maximizes probability per RON and reduces shared-prize risk by avoiding common patterns.
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_template.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/optimizer-template.csv README.md tests/test_template.py
git commit -m "docs: add spreadsheet template and workflow"
```
