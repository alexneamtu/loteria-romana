# Loto 6/49 Noroc Optional Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Noroc optional for Loto 6/49 so callers can omit Noroc from outputs and scoring while preserving existing default behavior.

**Architecture:** Extend the Loto 6/49 pipeline with an `include_noroc` flag that flows through metrics, strategies, backtest, and picks. The CLI adds `--no-noroc` to disable Noroc output and scoring. Default remains unchanged.

**Tech Stack:** Python stdlib, unittest.

### Task 1: Tests for Noroc-optional behavior

**Files:**
- Modify: `tests/test_loto_649_strategies.py`
- Modify: `tests/test_loto_649_picks.py`

**Step 1: Add failing test for Noroc-disabled prize logic**

```python
    def test_prize_rules_without_noroc(self):
        self.assertFalse(is_loto_649_prize(main_matches=0, noroc_match=True, include_noroc=False))
```

**Step 2: Add failing test for Noroc-disabled picks**

```python
    def test_generate_picks_without_noroc(self):
        draws = [
            ([1, 2, 3, 4, 5, 6], 1234567),
            ([7, 8, 9, 10, 11, 12], 7654321),
        ]
        rng = random.Random(0)
        lines = generate_picks(draws, count=2, rng=rng, include_noroc=False)
        self.assertEqual(len(lines), 2)
        for _main, noroc in lines:
            self.assertIsNone(noroc)
```

**Step 3: Run tests to verify they fail**

Run:
- `PYTHONPATH=src python -m unittest tests.test_loto_649_strategies.TestLoto649Strategies.test_prize_rules_without_noroc`
- `PYTHONPATH=src python -m unittest tests.test_loto_649_picks.TestLoto649Picks.test_generate_picks_without_noroc`
Expected: FAIL (missing param / behavior).

**Step 4: Commit tests**

```bash
git add tests/test_loto_649_strategies.py tests/test_loto_649_picks.py
git commit -m "test: cover noroc optional behavior"
```

### Task 2: Core implementation updates

**Files:**
- Modify: `src/loto_649_model/metrics.py`
- Modify: `src/loto_649_model/strategies.py`
- Modify: `src/loto_649_model/backtest.py`
- Modify: `src/loto_649_model/picks.py`

**Step 1: Update prize logic to accept include_noroc**

```python
def is_loto_649_prize(main_matches: int, noroc_match: bool, include_noroc: bool = True) -> bool:
    if main_matches >= 3:
        return True
    if include_noroc and noroc_match:
        return True
    return False
```

**Step 2: Add include_noroc to strategy generators**

```python
def generate_random_lines(count: int, rng=None, include_noroc: bool = True):
    ...
    noroc = rng.randint(0, NOROC_MAX) if include_noroc else None
    key = tuple(main) + (() if noroc is None else (noroc,))
```

```python
def generate_frequency_lines(count: int, freq: dict[int, int], rng=None, include_noroc: bool = True):
    ...
    noroc = rng.randint(0, NOROC_MAX) if include_noroc else None
    key = tuple(main) + (() if noroc is None else (noroc,))
```

**Step 3: Thread include_noroc into backtest + picks**

```python
def pick_best_strategy(draws, rng=None, include_noroc: bool = True):
    ...
    noroc_match = (line_noroc == noroc) if include_noroc else False
    if is_loto_649_prize(main_matches, noroc_match, include_noroc=include_noroc):
        ...
```

```python
def generate_picks(draws, count=2, rng=None, include_noroc: bool = True):
    ...
    if best == "frequency":
        return generate_frequency_lines(count, freq, rng=rng, include_noroc=include_noroc)
```

**Step 4: Run tests to verify they pass**

Run:
- `PYTHONPATH=src python -m unittest tests.test_loto_649_strategies.TestLoto649Strategies.test_prize_rules_without_noroc`
- `PYTHONPATH=src python -m unittest tests.test_loto_649_picks.TestLoto649Picks.test_generate_picks_without_noroc`
Expected: PASS

**Step 5: Commit**

```bash
git add src/loto_649_model/metrics.py src/loto_649_model/strategies.py src/loto_649_model/backtest.py src/loto_649_model/picks.py
git commit -m "feat: make noroc optional for loto 6/49"
```

### Task 3: CLI + README

**Files:**
- Modify: `scripts/generate_loto_649_picks.py`
- Modify: `README.md`

**Step 1: Add `--no-noroc` flag to CLI**

```python
    parser.add_argument("--no-noroc", action="store_true", help="Omit Noroc from picks")
    include_noroc = not args.no_noroc
```

**Step 2: Conditional output**

```python
    if include_noroc:
        print(f"{idx}. {', '.join(...)} + N{noroc_str}")
    else:
        print(f"{idx}. {', '.join(...)}")
```

**Step 3: Update README usage**

Add an example for `--no-noroc` and mention that Noroc is optional.

**Step 4: Commit**

```bash
git add scripts/generate_loto_649_picks.py README.md
git commit -m "feat: add no-noroc option"
```
