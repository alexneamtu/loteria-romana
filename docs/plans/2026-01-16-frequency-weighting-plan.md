# Frequency Weighting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Use full-history frequency counts (with smoothing) for the Joker frequency strategy and avoid leakage in backtests.

**Architecture:** Add a frequency builder helper that computes counts from draw history, wire it into backtesting and weekly picks, and update tests + README to reflect the new behavior.

**Tech Stack:** Python 3.x (stdlib only), `unittest`.

### Task 1: Add frequency builder helper

**Files:**
- Modify: `src/joker_model/strategies.py`
- Test: `tests/test_strategies.py`

**Step 1: Write the failing test**

```python
# tests/test_strategies.py
    def test_build_frequency_counts(self):
        draws = [
            ([1, 2, 3, 4, 5], 1),
            ([1, 2, 10, 11, 12], 2),
        ]
        freq = build_frequency(draws)
        self.assertEqual(freq[1], 2)
        self.assertEqual(freq[2], 2)
        self.assertEqual(freq[3], 1)
        self.assertEqual(freq[45], 0)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_strategies.py -v`
Expected: FAIL with `NameError` or missing `build_frequency`.

**Step 3: Write minimal implementation**

```python
# src/joker_model/strategies.py
def build_frequency(draws):
    freq = {n: 0 for n in range(1, 46)}
    for main, _ in draws:
        for n in main:
            freq[n] += 1
    return freq
```

Update `generate_frequency_lines` weights to apply smoothing:

```python
weights = [freq.get(n, 0) + 1 for n in numbers]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_strategies.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/joker_model/strategies.py tests/test_strategies.py
git commit -m "feat: add frequency builder"
```

### Task 2: Avoid leakage in backtest

**Files:**
- Modify: `src/joker_model/backtest.py`
- Test: `tests/test_backtest.py`

**Step 1: Write the failing test**

```python
# tests/test_backtest.py
    def test_pick_best_strategy_with_frequency(self):
        draws = [
            ([1, 2, 3, 4, 5], 1),
            ([6, 7, 8, 9, 10], 2),
            ([1, 2, 11, 12, 13], 3),
        ]
        best = pick_best_strategy(draws, rng=random.Random(0))
        self.assertIn(best, {"random", "frequency", "neural"})
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_backtest.py -v`
Expected: FAIL if `build_frequency` is not wired or backtest throws.

**Step 3: Write minimal implementation**

```python
# src/joker_model/backtest.py
from .strategies import build_frequency, generate_random_lines, generate_frequency_lines


def _score_frequency(draws, rng):
    wins = 0
    for idx, (main, joker) in enumerate(draws):
        freq = build_frequency(draws[:idx])
        lines = generate_frequency_lines(1, freq, rng=rng)
        line_main, line_joker = lines[0]
        main_matches = len(set(line_main) & set(main))
        joker_match = line_joker == joker
        if is_joker_prize(main_matches, joker_match):
            wins += 1
    return wins
```

Replace the frequency score in `pick_best_strategy` to use `_score_frequency`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_backtest.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/joker_model/backtest.py tests/test_backtest.py
git commit -m "feat: use rolling frequency in backtest"
```

### Task 3: Use full-history frequency in weekly picks + update README

**Files:**
- Modify: `scripts/generate_joker_picks.py`
- Modify: `README.md`

**Step 1: Write the failing test**

No new test; change is wiring and documentation only.

**Step 2: Write minimal implementation**

```python
# scripts/generate_joker_picks.py
from joker_model.strategies import build_frequency, generate_random_lines, generate_frequency_lines

# ...
    draw_tuples = [(d.main_numbers, d.joker) for d in draws]
    best = pick_best_strategy(draw_tuples)

    if best == "frequency":
        freq = build_frequency(draw_tuples)
        lines = generate_frequency_lines(7, freq, rng=rng)
```

Update README to remove the roadmap caveat and state that frequency uses historical counts with smoothing.

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v`
Expected: PASS

**Step 4: Commit**

```bash
git add scripts/generate_joker_picks.py README.md
git commit -m "docs: document frequency weighting"
```
