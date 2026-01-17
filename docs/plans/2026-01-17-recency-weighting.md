# Recency Weighting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Apply exponential recency weighting (default half-life 50 draws) across all heuristic strategies and ensure Thu/Sun pick generation backfills latest results before generating picks.

**Architecture:** Add a shared recency helper for half-life config and draw weights. Thread half-life through frequency builders, statistical strategies, and advanced strategies so all historical aggregates use weighted counts. Update CLI scripts to accept `--half-life`/`RECENCY_HALF_LIFE`, and update the GitHub workflow to backfill before picks.

**Tech Stack:** Python 3.x (stdlib only), `unittest`, GitHub Actions.

### Task 1: Add recency helper + tests

**Files:**
- Create: `src/shared/recency.py`
- Test: `tests/test_recency.py`

**Step 1: Write the failing test**

```python
# tests/test_recency.py
import unittest

from shared.recency import draw_weights, resolve_half_life, DEFAULT_HALF_LIFE


class TestRecencyWeights(unittest.TestCase):
    def test_draw_weights_half_life(self):
        weights = draw_weights(11, 10.0)
        self.assertAlmostEqual(weights[-1], 1.0, places=6)
        self.assertAlmostEqual(weights[0], 0.5, places=6)

    def test_draw_weights_monotonic(self):
        weights = draw_weights(5, 10.0)
        self.assertGreater(weights[-1], weights[0])

    def test_resolve_half_life_default_and_env(self):
        self.assertEqual(resolve_half_life(None, None), DEFAULT_HALF_LIFE)
        self.assertEqual(resolve_half_life(None, "25"), 25.0)

    def test_resolve_half_life_invalid(self):
        with self.assertRaises(ValueError):
            resolve_half_life(None, "0")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_recency.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbols.

**Step 3: Write minimal implementation**

```python
# src/shared/recency.py
import math

DEFAULT_HALF_LIFE = 50.0


def resolve_half_life(cli_value, env_value, default: float = DEFAULT_HALF_LIFE) -> float:
    if cli_value is not None:
        raw = cli_value
    elif env_value is not None:
        raw = env_value
    else:
        raw = default

    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("RECENCY_HALF_LIFE must be a positive number") from exc

    if value <= 0:
        raise ValueError("RECENCY_HALF_LIFE must be a positive number")

    return value


def draw_weights(draw_count: int, half_life: float) -> list[float]:
    if draw_count <= 0:
        return []
    return [0.5 ** ((draw_count - 1 - idx) / half_life) for idx in range(draw_count)]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_recency.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/shared/recency.py tests/test_recency.py
git commit -m "feat: add recency weight helpers"
```

### Task 2: Weight shared frequency builder + tests

**Files:**
- Modify: `src/shared/game_strategies.py`
- Test: `tests/test_shared_game_strategies.py`

**Step 1: Write the failing test**

```python
# tests/test_shared_game_strategies.py
import unittest

from shared.game_config import GameConfig
from shared.game_strategies import build_frequency


class TestGameStrategies(unittest.TestCase):
    def test_build_frequency_weights_recent(self):
        config = GameConfig(
            name="test",
            pool_min=1,
            pool_max=3,
            numbers_drawn=2,
            numbers_to_pick=2,
        )
        draws = [
            [1, 2],  # oldest
            [2, 3],  # newest
        ]

        freq = build_frequency(config, draws, half_life=1.0)

        self.assertAlmostEqual(freq[1], 0.5, places=6)
        self.assertAlmostEqual(freq[2], 1.5, places=6)
        self.assertAlmostEqual(freq[3], 1.0, places=6)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_shared_game_strategies.py -v`
Expected: FAIL (missing half_life support or incorrect weights).

**Step 3: Write minimal implementation**

```python
# src/shared/game_strategies.py
from .recency import draw_weights, DEFAULT_HALF_LIFE


def build_frequency(
    config: GameConfig,
    draws: list[list[int]],
    half_life: float = DEFAULT_HALF_LIFE,
) -> dict[int, float]:
    freq = {n: 0.0 for n in config.pool_range}
    if not draws:
        return freq

    weights = draw_weights(len(draws), half_life)
    for main, weight in zip(draws, weights):
        for n in main:
            if n in freq:
                freq[n] += weight
    return freq


def generate_frequency_picks(
    config: GameConfig,
    draws: list[list[int]],
    count: int,
    rng: random.Random | None = None,
    half_life: float = DEFAULT_HALF_LIFE,
) -> list[list[int]]:
    rng = rng or random.SystemRandom()
    freq = build_frequency(config, draws, half_life=half_life)
    numbers = list(config.pool_range)
    weights = [freq.get(n, 0) + 1 for n in numbers]
    ...
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_shared_game_strategies.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/shared/game_strategies.py tests/test_shared_game_strategies.py
git commit -m "feat: weight shared frequency picks"
```

### Task 3: Weight per-game frequency builders + update tests

**Files:**
- Modify: `src/joker_model/strategies.py`
- Modify: `src/loto_649_model/strategies.py`
- Modify: `src/loto_540_model/strategies.py`
- Modify: `src/joker_model/backtest.py`
- Modify: `src/loto_649_model/backtest.py`
- Modify: `src/loto_540_model/backtest.py`
- Modify: `src/joker_model/picks.py`
- Modify: `src/loto_649_model/picks.py`
- Modify: `src/loto_540_model/picks.py`
- Test: `tests/test_strategies.py`
- Test: `tests/test_loto_649_strategies.py`
- Test: `tests/test_backtest.py`
- Test: `tests/test_loto_649_backtest.py`

**Step 1: Write the failing tests**

```python
# tests/test_strategies.py
from shared.recency import draw_weights, DEFAULT_HALF_LIFE

    def test_build_frequency_counts(self):
        draws = [
            ([1, 2, 3, 4, 5], 1),
            ([1, 2, 10, 11, 12], 2),
        ]
        weights = draw_weights(len(draws), DEFAULT_HALF_LIFE)
        freq = build_frequency(draws)
        self.assertAlmostEqual(freq[1], weights[0] + weights[1], places=6)
        self.assertAlmostEqual(freq[2], weights[0] + weights[1], places=6)
        self.assertAlmostEqual(freq[3], weights[0], places=6)
        self.assertEqual(freq[45], 0.0)
```

```python
# tests/test_loto_649_strategies.py
from shared.recency import draw_weights, DEFAULT_HALF_LIFE

    def test_build_frequency_counts(self):
        draws = [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 10, 11, 12, 13],
        ]
        weights = draw_weights(len(draws), DEFAULT_HALF_LIFE)
        freq = build_frequency(draws)
        self.assertAlmostEqual(freq[1], weights[0] + weights[1], places=6)
        self.assertAlmostEqual(freq[2], weights[0] + weights[1], places=6)
        self.assertAlmostEqual(freq[3], weights[0], places=6)
        self.assertEqual(freq[49], 0.0)
```

```python
# tests/test_backtest.py
        with patch("joker_model.backtest.build_frequency") as build_frequency:
            build_frequency.side_effect = lambda draws, **_: {n: 0 for n in range(1, 46)}
```

```python
# tests/test_loto_649_backtest.py
        with patch("loto_649_model.backtest.build_frequency") as build_frequency:
            build_frequency.side_effect = lambda draws, **_: {n: 0 for n in range(1, 50)}
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests/test_strategies.py tests/test_loto_649_strategies.py tests/test_backtest.py tests/test_loto_649_backtest.py -v`
Expected: FAIL (frequency counts or patch signatures).

**Step 3: Write minimal implementation**

```python
# src/joker_model/strategies.py
from shared.recency import draw_weights, DEFAULT_HALF_LIFE


def build_frequency(draws, half_life: float = DEFAULT_HALF_LIFE):
    freq = {n: 0.0 for n in range(1, 46)}
    if not draws:
        return freq
    weights = draw_weights(len(draws), half_life)
    for (main, _), weight in zip(draws, weights):
        for n in main:
            freq[n] += weight
    return freq
```

```python
# src/loto_649_model/strategies.py
from shared.recency import draw_weights, DEFAULT_HALF_LIFE


def build_frequency(draws, half_life: float = DEFAULT_HALF_LIFE):
    freq = {n: 0.0 for n in range(1, 50)}
    if not draws:
        return freq
    weights = draw_weights(len(draws), half_life)
    for main, weight in zip(draws, weights):
        for n in main:
            freq[n] += weight
    return freq
```

```python
# src/loto_540_model/strategies.py
from shared.recency import draw_weights, DEFAULT_HALF_LIFE


def build_frequency(draws, half_life: float = DEFAULT_HALF_LIFE):
    freq = {n: 0.0 for n in range(1, 41)}
    if not draws:
        return freq
    weights = draw_weights(len(draws), half_life)
    for main, weight in zip(draws, weights):
        for n in main:
            freq[n] += weight
    return freq
```

```python
# src/joker_model/backtest.py
from shared.recency import DEFAULT_HALF_LIFE


def _score_frequency(draws, rng, half_life: float = DEFAULT_HALF_LIFE):
    ...
    freq = build_frequency(draws[:idx], half_life=half_life)
```

```python
# src/joker_model/picks.py
from shared.recency import DEFAULT_HALF_LIFE


def generate_picks(draws, count=2, rng=None, half_life: float = DEFAULT_HALF_LIFE):
    ...
    best = pick_best_strategy(draws, rng=rng, half_life=half_life)
    ...
    if best == "frequency":
        freq = build_frequency(draws, half_life=half_life)
```

Repeat the `half_life` plumbing for `loto_649_model` and `loto_540_model` backtest/picks.

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests/test_strategies.py tests/test_loto_649_strategies.py tests/test_backtest.py tests/test_loto_649_backtest.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/joker_model/strategies.py src/loto_649_model/strategies.py src/loto_540_model/strategies.py \
  src/joker_model/backtest.py src/loto_649_model/backtest.py src/loto_540_model/backtest.py \
  src/joker_model/picks.py src/loto_649_model/picks.py src/loto_540_model/picks.py \
  tests/test_strategies.py tests/test_loto_649_strategies.py tests/test_backtest.py tests/test_loto_649_backtest.py
git commit -m "feat: weight per-game frequency strategies"
```

### Task 4: Weight shared stats strategies + tests

**Files:**
- Modify: `src/shared/stats.py`
- Test: `tests/test_shared_stats.py`

**Step 1: Write the failing test**

```python
# tests/test_shared_stats.py
from shared.recency import draw_weights

    def test_build_delta_distribution_weighted(self):
        draws = [
            [1, 2, 3],  # deltas: [1, 1]
            [1, 3, 5],  # deltas: [2, 2]
        ]
        weights = draw_weights(len(draws), 1.0)
        dist = build_delta_distribution(draws, weights=weights)
        self.assertAlmostEqual(dist[1], 2 * weights[0], places=6)
        self.assertAlmostEqual(dist[2], 2 * weights[1], places=6)
```

Update `HotColdStrategy` tests to use the new half-life parameter:

```python
# tests/test_shared_stats.py
    def test_compute_heat_scores(self):
        strategy = HotColdStrategy(number_pool=10, numbers_to_pick=3, half_life=1.0)
        ...
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_shared_stats.py -v`
Expected: FAIL (weights param not supported or constructor mismatch).

**Step 3: Write minimal implementation**

```python
# src/shared/stats.py
from .recency import draw_weights, DEFAULT_HALF_LIFE


def build_delta_distribution(draws, weights: list[float] | None = None) -> dict[int, float]:
    delta_counts: dict[int, float] = {}
    if weights is None:
        weights = [1.0] * len(draws)
    for main, weight in zip(draws, weights):
        sorted_main = sorted(main)
        for delta in compute_deltas(sorted_main):
            delta_counts[delta] = delta_counts.get(delta, 0.0) + weight
    return delta_counts
```

Add `half_life` attributes to strategy classes and pass weights where distributions are built:

```python
class DeltaStrategy:
    def __init__(..., half_life: float = DEFAULT_HALF_LIFE, ...):
        self.half_life = half_life

    def get_probabilities(self, draws):
        weights = draw_weights(len(draws), self.half_life)
        ... use weights for score accumulation ...

    def generate(self, draws, count, rng):
        weights = draw_weights(len(draws), self.half_life)
        delta_dist = build_delta_distribution(draws, weights=weights)
```

Apply the same pattern to:
- `HotColdStrategy` (use `half_life`, remove fixed decay)
- `PairStrategy` (weighted pair counts)
- `SkipGapStrategy` (weighted expected gaps)
- `SumConstraintStrategy` (weighted mean/std)
- `BalanceStrategy` (weighted odd/even + high/low distributions)

Add optional `weights` parameters (default `None`) to:
- `compute_odd_even_distribution`
- `compute_high_low_distribution`
- `compute_consecutive_distribution`
- `build_position_frequency`

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_shared_stats.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/shared/stats.py tests/test_shared_stats.py
git commit -m "feat: apply recency weights to stats strategies"
```

### Task 5: Weight advanced strategies + tests

**Files:**
- Modify: `src/shared/advanced_strategies.py`
- Modify: `src/shared/features.py`
- Test: `tests/test_shared_advanced_strategies.py`

**Step 1: Write the failing test**

```python
# tests/test_shared_advanced_strategies.py
import unittest

from shared.advanced_strategies import compute_composite_scores
from shared.game_config import GameConfig


class TestAdvancedStrategies(unittest.TestCase):
    def test_composite_scores_weighted_frequency(self):
        config = GameConfig(
            name="test",
            pool_min=1,
            pool_max=4,
            numbers_drawn=2,
            numbers_to_pick=2,
        )
        draws = [
            [1, 2],  # oldest
            [3, 4],  # newest
        ]
        weights = {
            "frequency": 1.0,
            "recency": 0.0,
            "gap": 0.0,
            "position": 0.0,
            "trend": 0.0,
            "balance": 0.0,
        }
        scores = compute_composite_scores(config, draws, weights=weights, half_life=1.0)
        self.assertGreater(scores[3], scores[1])
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_shared_advanced_strategies.py -v`
Expected: FAIL (missing half_life or unweighted frequency).

**Step 3: Write minimal implementation**

```python
# src/shared/advanced_strategies.py
from .recency import draw_weights, DEFAULT_HALF_LIFE


def compute_composite_scores(..., half_life: float = DEFAULT_HALF_LIFE):
    ...
    weights_by_draw = draw_weights(len(draws), half_life)
    # weighted frequency and recency using weights_by_draw
```

Update `compute_position_frequency` in `src/shared/features.py` to accept optional `weights` and use them in the position counts. Use `weights_by_draw` when computing position score in `compute_composite_scores`.

Update `generate_optimal_picks`, `generate_coverage_picks`, `generate_pattern_picks`, and `generate_smart_picks` to accept `half_life` and pass through to `compute_composite_scores`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_shared_advanced_strategies.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/shared/advanced_strategies.py src/shared/features.py tests/test_shared_advanced_strategies.py
git commit -m "feat: weight advanced strategy scoring"
```

### Task 6: Thread half-life through CLI scripts

**Files:**
- Modify: `scripts/generate_joker_picks.py`
- Modify: `scripts/generate_loto_649_picks.py`
- Modify: `scripts/generate_loto_540_picks.py`

**Step 1: Write minimal implementation**

```python
# scripts/generate_*.py
from shared.recency import resolve_half_life

parser.add_argument("--half-life", type=float, help="Recency half-life in draws")
...
half_life = resolve_half_life(args.half_life, os.getenv("RECENCY_HALF_LIFE"))
...
DeltaStrategy(..., half_life=half_life)
HotColdStrategy(..., half_life=half_life)
...
lines = generate_smart_picks(..., half_life=half_life)
```

Ensure `get_strategy_by_name` / `get_all_strategies` accept a `half_life` argument and use it. Pass `half_life` into `generate_picks` (auto) and ensemble strategy lists.

**Step 2: Run a quick smoke test**

Run: `PYTHONPATH=src python scripts/generate_joker_picks.py -n 1 --half-life 10 -v`
Expected: Prints one line without error.

**Step 3: Commit**

```bash
git add scripts/generate_joker_picks.py scripts/generate_loto_649_picks.py scripts/generate_loto_540_picks.py
git commit -m "feat: add half-life config to pick scripts"
```

### Task 7: Backfill before Thu/Sun picks

**Files:**
- Modify: `.github/workflows/generate-picks.yml`

**Step 1: Update workflow**

Add a step after Python setup:

```yaml
      - name: Backfill latest historical draws
        run: PYTHONPATH=src python scripts/backfill_noroc_chior.py
```

**Step 2: Commit**

```bash
git add .github/workflows/generate-picks.yml
git commit -m "chore: backfill draws before picks"
```

---
