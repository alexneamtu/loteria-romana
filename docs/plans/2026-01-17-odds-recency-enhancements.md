# Odds and Recency Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add official prize tier odds to README for Loto 6/49, Joker, and Loto 5/40, and implement recency weighting that supports draw-count or day-based half-life across strategies and neural training.

**Architecture:** Centralize recency weight computation in `shared/recency.py` with `draws` and `days` modes. Thread optional draw dates and mode through strategy and neural training APIs. Neural training uses per-sample weights aligned to the target (next draw) date. README uses combinatorial odds (no scraping).

**Tech Stack:** Python 3 (stdlib only), `unittest`, GitHub Actions (existing workflows).

### Task 1: Recency mode support (draw-count vs days)

**Files:**
- Modify: `tests/test_recency.py`
- Modify: `src/shared/recency.py`

**Step 1: Write the failing tests**

```python
from shared.recency import draw_weights, resolve_half_life, DEFAULT_HALF_LIFE
from shared.recency import resolve_half_life_mode, DEFAULT_HALF_LIFE_MODE

    def test_resolve_half_life_mode_default_and_env(self):
        self.assertEqual(resolve_half_life_mode(None, None), DEFAULT_HALF_LIFE_MODE)
        self.assertEqual(resolve_half_life_mode(None, "days"), "days")

    def test_resolve_half_life_mode_invalid(self):
        with self.assertRaises(ValueError):
            resolve_half_life_mode(None, "weeks")

    def test_draw_weights_days_mode(self):
        draw_dates = ["2024-01-01", "2024-01-11"]
        weights = draw_weights(len(draw_dates), 10.0, mode="days", draw_dates=draw_dates)
        self.assertAlmostEqual(weights[0], 0.5, places=6)
        self.assertAlmostEqual(weights[1], 1.0, places=6)

    def test_draw_weights_days_mode_requires_dates(self):
        with self.assertRaises(ValueError):
            draw_weights(2, 10.0, mode="days")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_recency.py -v`  
Expected: FAIL with missing `resolve_half_life_mode` or unexpected keyword errors.

**Step 3: Write minimal implementation**

```python
from datetime import date

DEFAULT_HALF_LIFE_MODE = "draws"
VALID_HALF_LIFE_MODES = {"draws", "days"}


def resolve_half_life_mode(cli_value, env_value, default: str = DEFAULT_HALF_LIFE_MODE) -> str:
    mode = cli_value if cli_value is not None else env_value if env_value is not None else default
    if mode not in VALID_HALF_LIFE_MODES:
        raise ValueError("RECENCY_HALF_LIFE_MODE must be 'draws' or 'days'")
    return mode


def _parse_draw_dates(draw_dates: list[str]) -> list[date]:
    try:
        return [date.fromisoformat(d) for d in draw_dates]
    except (TypeError, ValueError) as exc:
        raise ValueError("draw_dates must be ISO YYYY-MM-DD strings") from exc


def draw_weights(
    draw_count: int,
    half_life: float,
    draw_dates: list[str] | None = None,
    mode: str = DEFAULT_HALF_LIFE_MODE,
) -> list[float]:
    if draw_count <= 0:
        return []
    if mode == "draws":
        return [0.5 ** ((draw_count - 1 - idx) / half_life) for idx in range(draw_count)]
    if not draw_dates or len(draw_dates) != draw_count:
        raise ValueError("draw_dates must be provided for RECENCY_HALF_LIFE_MODE=days")
    parsed = _parse_draw_dates(draw_dates)
    latest = parsed[-1]
    return [0.5 ** (((latest - d).days) / half_life) for d in parsed]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_recency.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/shared/recency.py tests/test_recency.py
git commit -m "feat: add day-based recency weighting mode"
```

### Task 2: Weighted gap/trend scoring in advanced strategies

**Files:**
- Modify: `tests/test_shared_advanced_strategies.py`
- Modify: `src/shared/features.py`
- Modify: `src/shared/advanced_strategies.py`

**Step 1: Write the failing tests**

```python
from shared.features import compute_weighted_gap_averages

    def test_weighted_gap_averages_use_recent_gaps(self):
        draws = [[1], [1], [2], [2], [1]]
        weights = [0.1, 0.2, 0.3, 0.6, 1.0]
        gaps = compute_weighted_gap_averages(draws, pool_size=2, weights=weights)
        expected = (1 * 0.2 + 3 * 1.0) / (0.2 + 1.0)
        self.assertAlmostEqual(gaps[1], expected, places=6)

    def test_trend_weighting_favors_recent_draw(self):
        config = GameConfig(
            name="test",
            pool_min=1,
            pool_max=2,
            numbers_drawn=1,
            numbers_to_pick=1,
        )
        draws = [[2], [1], [1], [2]]
        weights = {
            "frequency": 0.0,
            "recency": 0.0,
            "gap": 0.0,
            "position": 0.0,
            "trend": 1.0,
            "balance": 0.0,
        }
        scores = compute_composite_scores(config, draws, weights=weights, half_life=0.5)
        self.assertGreater(scores[2], scores[1])
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_shared_advanced_strategies.py -v`  
Expected: FAIL with missing `compute_weighted_gap_averages` or trend assertion failure.

**Step 3: Write minimal implementation**

```python
def compute_weighted_gap_averages(
    draws: list[list[int]],
    pool_size: int,
    weights: list[float] | None = None,
) -> dict[int, float]:
    if weights is None:
        weights = [1.0] * len(draws)
    gap_totals = {n: 0.0 for n in range(1, pool_size + 1)}
    weight_totals = {n: 0.0 for n in range(1, pool_size + 1)}
    last_seen = {n: -1 for n in range(1, pool_size + 1)}
    for idx, (main, weight) in enumerate(zip(draws, weights)):
        for num in main:
            if last_seen[num] >= 0:
                gap = idx - last_seen[num]
                gap_totals[num] += gap * weight
                weight_totals[num] += weight
            last_seen[num] = idx
    return {
        n: (gap_totals[n] / weight_totals[n] if weight_totals[n] > 0 else 0.0)
        for n in range(1, pool_size + 1)
    }
```

Update `compute_composite_scores` to accept `draw_dates` and `half_life_mode`, use
`draw_weights(..., draw_dates=draw_dates, mode=half_life_mode)`; use
`compute_weighted_gap_averages` for gap scoring; and compute trend using weighted
first/second halves.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_shared_advanced_strategies.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/shared/features.py src/shared/advanced_strategies.py tests/test_shared_advanced_strategies.py
git commit -m "feat: apply recency weights to gap and trend scoring"
```

### Task 3: Shared strategy APIs accept draw dates + half-life mode

**Files:**
- Modify: `tests/test_shared_game_strategies.py`
- Modify: `src/shared/game_strategies.py`
- Modify: `src/shared/stats.py`
- Modify: `src/shared/ensemble.py`

**Step 1: Write the failing test**

```python
    def test_build_frequency_days_mode(self):
        config = GameConfig(
            name="test",
            pool_min=1,
            pool_max=3,
            numbers_drawn=2,
            numbers_to_pick=2,
        )
        draws = [[1, 2], [2, 3]]
        draw_dates = ["2024-01-01", "2024-01-11"]
        freq = build_frequency(
            config,
            draws,
            half_life=10.0,
            half_life_mode="days",
            draw_dates=draw_dates,
        )
        self.assertAlmostEqual(freq[1], 0.5, places=6)
        self.assertAlmostEqual(freq[2], 1.5, places=6)
        self.assertAlmostEqual(freq[3], 1.0, places=6)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_shared_game_strategies.py -v`  
Expected: FAIL with unexpected keyword args or missing day-mode support.

**Step 3: Write minimal implementation**

- Update `build_frequency` and `generate_frequency_picks` to accept `draw_dates` and
  `half_life_mode` and call `draw_weights(..., draw_dates=draw_dates, mode=half_life_mode)`.
- Add `half_life_mode` to strategy constructors in `src/shared/stats.py` and pass
  `draw_dates` through `get_probabilities`/`generate`.
- Update `EnsembleVoter.combine_probabilities`, `generate`, and `generate_diverse`
  to accept optional `draw_dates` and `half_life_mode` and pass to strategy methods.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_shared_game_strategies.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/shared/game_strategies.py src/shared/stats.py src/shared/ensemble.py tests/test_shared_game_strategies.py
git commit -m "feat: thread draw dates and half-life mode through strategies"
```

### Task 4: Sample-weighted neural training

**Files:**
- Modify: `tests/test_shared_neural.py`
- Modify: `tests/test_neural.py`
- Modify: `src/shared/neural_base.py`
- Modify: `src/shared/neural_strategies.py`
- Modify: `src/joker_model/neural.py`
- Modify: `src/loto_649_model/neural.py`
- Modify: `src/loto_540_model/neural.py`

**Step 1: Write the failing tests**

```python
    def test_mlp_train_sample_weights_all_ones_matches_unweighted(self):
        rng = random.Random(0)
        inputs = [[1, 0], [0, 1]]
        targets = [[1, 0], [0, 1]]
        mlp1 = MLPModel([2, 3, 2], rng=random.Random(0))
        mlp2 = MLPModel([2, 3, 2], rng=random.Random(0))
        losses_unweighted = mlp1.train(inputs, targets, epochs=5, learning_rate=0.1)
        losses_weighted = mlp2.train(inputs, targets, epochs=5, learning_rate=0.1, sample_weights=[1.0, 1.0])
        for a, b in zip(losses_unweighted, losses_weighted):
            self.assertAlmostEqual(a, b, places=9)

    def test_lstm_train_sample_weights_all_ones_matches_unweighted(self):
        lstm1 = LSTMModel(input_size=2, hidden_size=3, output_size=2, num_layers=1, rng=random.Random(0))
        lstm2 = LSTMModel(input_size=2, hidden_size=3, output_size=2, num_layers=1, rng=random.Random(0))
        sequences = [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]
        targets = [[1, 0], [0, 1]]
        losses_unweighted = lstm1.train(sequences, targets, epochs=5, learning_rate=0.1)
        losses_weighted = lstm2.train(sequences, targets, epochs=5, learning_rate=0.1, sample_weights=[1.0, 1.0])
        for a, b in zip(losses_unweighted, losses_weighted):
            self.assertAlmostEqual(a, b, places=9)
```

```python
    def test_softmax_train_respects_sample_weights(self):
        model = SoftmaxModel(input_size=2, output_size=2, rng=random.Random(0))
        inputs = [[1, 0], [0, 1]]
        targets = [[1, 0], [0, 1]]
        before = [row[:] for row in model.weights]
        model.train(inputs, targets, epochs=1, lr=0.5, sample_weights=[0.0, 0.0])
        self.assertEqual(before, model.weights)
```

**Step 2: Run tests to verify they fail**

Run:  
`PYTHONPATH=src python -m unittest tests/test_shared_neural.py -v`  
`PYTHONPATH=src python -m unittest tests/test_neural.py -v`  
Expected: FAIL with unexpected keyword `sample_weights`.

**Step 3: Write minimal implementation**

- Add `sample_weights` to `MLPModel.train_step` and `MLPModel.train`, scaling
  loss and gradients by weight (and scale L2 updates by weight).
- Add `sample_weights` to `LSTMModel.train`, scaling loss and gradient by weight.
- Add `sample_weights` to `shared.neural_strategies.SoftmaxModel.train`, scaling
  per-sample error by weight.
- Update `shared.neural_strategies.generate_neural_picks` to accept
  `half_life`, `half_life_mode`, and `draw_dates`; compute
  `weights_by_draw = draw_weights(...)` and pass `sample_weights = weights_by_draw[1:]`.
- Update `shared.neural_strategies.score_neural_strategy` to pass dates and mode.
- Update per-game softmax models (`joker_model/neural.py`, `loto_649_model/neural.py`,
  `loto_540_model/neural.py`) to accept `sample_weights` in `train` and (optionally)
  in `loss`, and apply weights to gradient updates.
- Update per-game `generate_neural_lines` to accept `draw_dates`, `half_life`,
  `half_life_mode`, compute sample weights aligned to target draw dates, and pass
  to `train`.

**Step 4: Run tests to verify they pass**

Run:  
`PYTHONPATH=src python -m unittest tests/test_shared_neural.py -v`  
`PYTHONPATH=src python -m unittest tests/test_neural.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/shared/neural_base.py src/shared/neural_strategies.py \
  src/joker_model/neural.py src/loto_649_model/neural.py src/loto_540_model/neural.py \
  tests/test_shared_neural.py tests/test_neural.py
git commit -m "feat: add sample-weighted neural training"
```

### Task 5: Wire draw dates and half-life mode through picks/backtests/CLI

**Files:**
- Modify: `tests/test_picks.py`
- Modify: `tests/test_backtest.py`
- Modify: `tests/test_loto_649_backtest.py`
- Modify: `src/joker_model/strategies.py`
- Modify: `src/loto_649_model/strategies.py`
- Modify: `src/loto_540_model/strategies.py`
- Modify: `src/joker_model/backtest.py`
- Modify: `src/loto_649_model/backtest.py`
- Modify: `src/loto_540_model/backtest.py`
- Modify: `src/joker_model/picks.py`
- Modify: `src/loto_649_model/picks.py`
- Modify: `src/loto_540_model/picks.py`
- Modify: `scripts/generate_joker_picks.py`
- Modify: `scripts/generate_loto_649_picks.py`
- Modify: `scripts/generate_loto_540_picks.py`

**Step 1: Write the failing tests**

```python
from shared.recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE

            pick_best.assert_called_once_with(
                draws,
                rng=rng,
                half_life=DEFAULT_HALF_LIFE,
                half_life_mode=DEFAULT_HALF_LIFE_MODE,
                draw_dates=None,
            )
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_picks.py -v`  
Expected: FAIL due to missing parameters or mismatched call.

**Step 3: Write minimal implementation**

- Update per-game `build_frequency` helpers to accept `draw_dates`/`half_life_mode`
  and pass them into `draw_weights`.
- Update `pick_best_strategy` in each backtest module to accept `draw_dates` and
  `half_life_mode`, and pass to `build_frequency` and `generate_neural_lines`.
- Update `generate_picks` in each `*_model/picks.py` to accept `draw_dates` and
  `half_life_mode`, and pass through to `pick_best_strategy`, `build_frequency`,
  and `generate_neural_lines`.
- Update scripts to parse `--half-life-mode` (choices: `draws`, `days`), read
  `RECENCY_HALF_LIFE_MODE`, build `draw_dates = [d.date for d in draws]`,
  and pass `draw_dates` + `half_life_mode` through strategy calls (including
  `EnsembleVoter.generate` and advanced strategies).

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_picks.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add src/joker_model src/loto_649_model src/loto_540_model \
  scripts/generate_joker_picks.py scripts/generate_loto_649_picks.py scripts/generate_loto_540_picks.py \
  tests/test_picks.py tests/test_backtest.py tests/test_loto_649_backtest.py
git commit -m "feat: pass draw dates and half-life mode through pipelines"
```

### Task 6: README odds tables and CLI docs

**Files:**
- Modify: `README.md`

**Step 1: Compute odds for reference**

Run:
```bash
python - <<'PY'
import math
def comb(n,k): return math.comb(n,k)

# 6/49
D649 = comb(49,6)
odds_649 = {
    "6": comb(6,6)*comb(43,0)/D649,
    "5": comb(6,5)*comb(43,1)/D649,
    "4": comb(6,4)*comb(43,2)/D649,
    "3": comb(6,3)*comb(43,3)/D649,
}

# 5/40 (pick 5, draw 6)
D540 = comb(40,6)
odds_540 = {
    "5": comb(5,5)*comb(35,1)/D540,
    "4": comb(5,4)*comb(35,2)/D540,
    "3": comb(5,3)*comb(35,3)/D540,
}

# Joker
Djoker = comb(45,5)
main = {k: comb(5,k)*comb(40,5-k)/Djoker for k in range(0,6)}
pJ = 1/20
pN = 19/20
odds_joker = {
    "5+J": main[5]*pJ,
    "5": main[5]*pN,
    "4+J": main[4]*pJ,
    "4": main[4]*pN,
    "3+J": main[3]*pJ,
    "3": main[3]*pN,
    "2+J": main[2]*pJ,
    "1+J": main[1]*pJ,
    "0+J": main[0]*pJ,
}

def fmt(p):
    return f\"1 in {round(1/p):,}\", f\"{p*100:.6f}%\"

print(\"6/49\")\nprint({k: fmt(v) for k,v in odds_649.items()})\nprint(\"any\", sum(odds_649.values())*100)\nprint(\"5/40\")\nprint({k: fmt(v) for k,v in odds_540.items()})\nprint(\"any\", sum(odds_540.values())*100)\nprint(\"Joker\")\nprint({k: fmt(v) for k,v in odds_joker.items()})\nprint(\"any\", sum(odds_joker.values())*100)\nPY
```

Expected key values (rounded):

- **Loto 6/49**  
  - 6: `1 in 13,983,816` (0.000007%)  
  - 5: `1 in 54,201` (0.001845%)  
  - 4: `1 in 1,032` (0.096862%)  
  - 3: `1 in 57` (1.765040%)  
  - Any prize: 1.863755%

- **Loto 5/40**  
  - 5: `1 in 109,668` (0.000912%)  
  - 4: `1 in 1,290` (0.077507%)  
  - 3: `1 in 59` (1.705146%)  
  - Any prize: 1.783565%

- **Joker**  
  - 5+J: `1 in 24,435,180` (0.000004%)  
  - 5: `1 in 1,286,062` (0.000078%)  
  - 4+J: `1 in 122,176` (0.000818%)  
  - 4: `1 in 6,430` (0.015551%)  
  - 3+J: `1 in 3,133` (0.031921%)  
  - 3: `1 in 165` (0.606503%)  
  - 2+J: `1 in 247` (0.404335%)  
  - 1+J: `1 in 53` (1.870050%)  
  - 0+J: `1 in 37` (2.692872%)  
  - Any prize: 5.622132%

**Step 2: Update README**

- Add a new “Odds & Prize Tiers” section (base games only) with:
  - Official Prize Tiers table (rule, category, odds, percent).
  - Simplified Win Rules table (plain-language).
  - Any-prize and jackpot summary lines.
  - Note that odds are fixed by game rules and not affected by historical data.
- Update the CLI options section to include:
  - `--half-life` (recency half-life)
  - `--half-life-mode {draws,days}`
  - Environment `RECENCY_HALF_LIFE_MODE`.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add prize tier odds tables to README"
```

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-17-odds-recency-enhancements.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration  
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
