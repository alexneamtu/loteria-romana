# Seeded Picks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add reproducible seeds (CLI + env) for Joker picks while keeping default random behavior unchanged.

**Architecture:** Introduce a small seed resolver helper, wire it into the CLI script, and ensure the picker uses a passed RNG for deterministic output. Update README to document `--seed` and `JOKER_SEED`.

**Tech Stack:** Python 3.x (stdlib only), `unittest`.

### Task 1: Seed resolution helper

**Files:**
- Create: `src/joker_model/seed.py`
- Test: `tests/test_seed.py`

**Step 1: Write the failing test**

```python
# tests/test_seed.py
import unittest

from joker_model.seed import resolve_seed


class TestSeed(unittest.TestCase):
    def test_cli_seed_overrides_env(self):
        self.assertEqual(resolve_seed(cli_seed=7, env_seed="11"), 7)

    def test_env_seed_used_when_cli_missing(self):
        self.assertEqual(resolve_seed(cli_seed=None, env_seed="11"), 11)

    def test_empty_env_returns_none(self):
        self.assertIsNone(resolve_seed(cli_seed=None, env_seed=""))

    def test_invalid_env_raises(self):
        with self.assertRaises(ValueError):
            resolve_seed(cli_seed=None, env_seed="abc")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPYCACHEPREFIX=/tmp/pycache PYTHONPATH=src python -m unittest tests/test_seed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'joker_model.seed'`.

**Step 3: Write minimal implementation**

```python
# src/joker_model/seed.py

def resolve_seed(cli_seed, env_seed=None):
    if cli_seed is not None:
        return cli_seed
    if not env_seed:
        return None
    try:
        return int(env_seed)
    except ValueError as exc:
        raise ValueError("JOKER_SEED must be an integer") from exc
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPYCACHEPREFIX=/tmp/pycache PYTHONPATH=src python -m unittest tests/test_seed.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/joker_model/seed.py tests/test_seed.py
git commit -m "feat: add seed resolver"
```

### Task 2: Deterministic picks with seeded RNG

**Files:**
- Modify: `src/joker_model/picks.py`
- Modify: `tests/test_picks.py`

**Step 1: Write the failing test**

```python
# tests/test_picks.py
    def test_generate_picks_deterministic_with_seed(self):
        draws = [
            ([1, 2, 3, 4, 5], 1),
            ([6, 7, 8, 9, 10], 2),
            ([11, 12, 13, 14, 15], 3),
        ]
        lines_a = generate_picks(draws, rng=random.Random(0))
        lines_b = generate_picks(draws, rng=random.Random(0))
        self.assertEqual(lines_a, lines_b)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPYCACHEPREFIX=/tmp/pycache PYTHONPATH=src python -m unittest tests/test_picks.py -v`
Expected: FAIL if `generate_picks` does not pass the RNG through to strategy selection.

**Step 3: Write minimal implementation**

```python
# src/joker_model/picks.py
    best = pick_best_strategy(draws, rng=rng)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPYCACHEPREFIX=/tmp/pycache PYTHONPATH=src python -m unittest tests/test_picks.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/joker_model/picks.py tests/test_picks.py
git commit -m "feat: make seeded picks deterministic"
```

### Task 3: CLI + env wiring and README

**Files:**
- Modify: `scripts/generate_joker_picks.py`
- Modify: `README.md`

**Step 1: Write the failing test**

No new test; CLI wiring uses `resolve_seed` (already tested).

**Step 2: Write minimal implementation**

```python
# scripts/generate_joker_picks.py
import argparse
import os

from joker_model.seed import resolve_seed

# ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Set deterministic RNG seed")
    args = parser.parse_args()

    seed = None
    try:
        seed = resolve_seed(args.seed, os.getenv("JOKER_SEED"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
```

Update README with `--seed` and `JOKER_SEED` examples and remove the roadmap item for seeds.

**Step 3: Run tests to verify they pass**

Run: `PYTHONPYCACHEPREFIX=/tmp/pycache PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v`
Expected: PASS

**Step 4: Commit**

```bash
git add scripts/generate_joker_picks.py README.md
git commit -m "feat: add seed options for reproducible picks"
```
