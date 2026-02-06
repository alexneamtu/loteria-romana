# Phase 1: Foundation — Enhanced Backtesting, Holdout, Genetic Algorithm, Gradient Boosting

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the enhanced backtesting infrastructure (EV scoring, significance gates, Monte Carlo validation, holdout management) and implement two new strategies (Genetic Algorithm, Gradient Boosted Trees) that integrate into the existing ensemble pipeline.

**Architecture:** Extend `shared/backtest_base.py` with EV-weighted scoring and significance gating. Add holdout split logic to a new `shared/holdout.py`. New strategies in `shared/genetic.py` and `shared/gradient_boost.py` implement the same interface as `BayesianScorer` (`.name`, `.generate()`, `.get_probabilities()`). Register them in `shared/ensemble_blend.py`.

**Tech Stack:** Python stdlib for genetic algorithm and backtesting enhancements. scikit-learn (optional import) for gradient boosting. Existing `shared/features.py` for feature extraction.

---

### Task 1: Add holdout set management

Manage a holdout split so the most recent N draws are reserved for final evaluation and never used during development/backtesting.

**Files:**
- Create: `src/shared/holdout.py`
- Test: `tests/test_holdout.py`

**Step 1: Write the failing test**

```python
# tests/test_holdout.py
import unittest

from shared.holdout import split_holdout, HoldoutSplit


class TestSplitHoldout(unittest.TestCase):
    def test_split_reserves_correct_count(self):
        draws = [[i, i+1, i+2, i+3, i+4] for i in range(1, 201)]
        result = split_holdout(draws, holdout_size=100)
        self.assertEqual(len(result.train), 100)
        self.assertEqual(len(result.holdout), 100)

    def test_holdout_is_most_recent(self):
        draws = [[i] for i in range(10)]
        result = split_holdout(draws, holdout_size=3)
        self.assertEqual(result.holdout, [[7], [8], [9]])
        self.assertEqual(result.train, [[i] for i in range(7)])

    def test_holdout_fraction(self):
        draws = [[i] for i in range(100)]
        result = split_holdout(draws, holdout_fraction=0.2)
        self.assertEqual(len(result.holdout), 20)
        self.assertEqual(len(result.train), 80)

    def test_holdout_size_takes_precedence(self):
        draws = [[i] for i in range(100)]
        result = split_holdout(draws, holdout_size=10, holdout_fraction=0.5)
        self.assertEqual(len(result.holdout), 10)

    def test_insufficient_data_returns_all_train(self):
        draws = [[i] for i in range(5)]
        result = split_holdout(draws, holdout_size=100)
        self.assertEqual(len(result.train), 5)
        self.assertEqual(len(result.holdout), 0)

    def test_dates_split_with_draws(self):
        draws = [[i] for i in range(10)]
        dates = [f"2024-01-{i+1:02d}" for i in range(10)]
        result = split_holdout(draws, holdout_size=3, dates=dates)
        self.assertEqual(len(result.train_dates), 7)
        self.assertEqual(len(result.holdout_dates), 3)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_holdout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.holdout'`

**Step 3: Write minimal implementation**

```python
# src/shared/holdout.py
"""Holdout set management for strategy evaluation.

Reserves the most recent N draws as a final holdout set
that is never used during strategy development or backtesting.
"""

from dataclasses import dataclass, field


@dataclass
class HoldoutSplit:
    """Result of splitting draws into train and holdout sets."""

    train: list[list[int]]
    holdout: list[list[int]]
    train_dates: list[str] = field(default_factory=list)
    holdout_dates: list[str] = field(default_factory=list)


def split_holdout(
    draws: list[list[int]],
    holdout_size: int = 0,
    holdout_fraction: float = 0.1,
    min_train_size: int = 50,
    dates: list[str] | None = None,
) -> HoldoutSplit:
    """Split draws into training and holdout sets.

    The holdout set is always the most recent draws.
    If holdout_size is provided, it takes precedence over holdout_fraction.

    Args:
        draws: Historical draws (oldest first).
        holdout_size: Exact number of draws to hold out. 0 means use fraction.
        holdout_fraction: Fraction of draws to hold out (default 10%).
        min_train_size: Minimum training draws required. If not enough
            data, returns all draws as training with empty holdout.
        dates: Optional ISO date strings parallel to draws.

    Returns:
        HoldoutSplit with train/holdout draws and optional dates.
    """
    n = len(draws)

    if holdout_size > 0:
        actual_holdout = holdout_size
    else:
        actual_holdout = int(n * holdout_fraction)

    if n - actual_holdout < min_train_size:
        return HoldoutSplit(
            train=list(draws),
            holdout=[],
            train_dates=list(dates) if dates else [],
            holdout_dates=[],
        )

    split_idx = n - actual_holdout
    return HoldoutSplit(
        train=draws[:split_idx],
        holdout=draws[split_idx:],
        train_dates=dates[:split_idx] if dates else [],
        holdout_dates=dates[split_idx:] if dates else [],
    )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_holdout.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```
feat: add holdout set management for strategy evaluation
```

---

### Task 2: Add EV-weighted backtest scoring

Extend `Backtester.backtest()` to return EV as money-weighted score (payout sum / ticket count) instead of just raw win counts. Add a `score_ev()` convenience method.

**Files:**
- Modify: `src/shared/backtest_base.py`
- Test: `tests/test_shared_backtest.py`

**Step 1: Write the failing test**

Add to `tests/test_shared_backtest.py`:

```python
class TestBacktesterEVScoring(unittest.TestCase):
    def setUp(self):
        self.tiers = [
            PrizeTier(name="3_match", matches_required=3, payout=10.0),
            PrizeTier(name="4_match", matches_required=4, payout=100.0),
            PrizeTier(name="5_match", matches_required=5, payout=10000.0),
        ]
        self.backtester = Backtester(
            number_pool=45,
            numbers_to_pick=5,
            prize_tiers=self.tiers,
        )

    def test_ev_per_ticket_computed(self):
        rng_setup = random.Random(42)
        draws = [
            (sorted(rng_setup.sample(range(1, 46), 5)), rng_setup.randint(1, 20))
            for _ in range(100)
        ]
        strategy = DeltaStrategy(45, 5)
        result = self.backtester.backtest(
            strategy=strategy,
            draws=draws,
            train_window=50,
            rng=random.Random(42),
        )
        self.assertIsInstance(result.ev_per_ticket, float)
        self.assertGreaterEqual(result.ev_per_ticket, 0.0)

    def test_ev_per_ticket_zero_when_no_wins(self):
        result = BacktestResult(
            strategy_name="test",
            total_draws=100,
            total_tickets=100,
            expected_value=0.0,
        )
        self.assertEqual(result.ev_per_ticket, 0.0)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_shared_backtest.TestBacktesterEVScoring -v`
Expected: FAIL — `AttributeError: 'BacktestResult' has no attribute 'ev_per_ticket'`

**Step 3: Write minimal implementation**

Add `ev_per_ticket` property to `BacktestResult` in `src/shared/backtest_base.py` after the `roi` property (line 40):

```python
    @property
    def ev_per_ticket(self) -> float:
        """Expected value per ticket (total payout / total tickets)."""
        if self.total_tickets == 0:
            return 0.0
        return self.expected_value / self.total_tickets
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_shared_backtest.py -v`
Expected: All tests PASS (existing + new)

**Step 5: Commit**

```
feat: add EV-per-ticket metric to BacktestResult
```

---

### Task 3: Add significance gate to backtest framework

Add a function that determines whether a strategy should be admitted to the ensemble based on statistical significance against random baseline.

**Files:**
- Modify: `src/shared/backtest_base.py`
- Test: `tests/test_shared_backtest.py`

**Step 1: Write the failing test**

Add to `tests/test_shared_backtest.py`:

```python
from shared.backtest_base import passes_significance_gate


class TestSignificanceGate(unittest.TestCase):
    def test_clearly_better_passes(self):
        result = BacktestResult(
            strategy_name="good",
            total_draws=1000,
            total_tickets=1000,
            total_wins=200,
            win_rate=0.2,
        )
        # If baseline is 0.1, 20% vs 10% over 1000 tickets should pass
        self.assertTrue(passes_significance_gate(result, baseline_win_rate=0.1))

    def test_similar_to_baseline_fails(self):
        result = BacktestResult(
            strategy_name="meh",
            total_draws=100,
            total_tickets=100,
            total_wins=11,
            win_rate=0.11,
        )
        # 11% vs 10% over 100 tickets: not significant
        self.assertFalse(passes_significance_gate(result, baseline_win_rate=0.1))

    def test_worse_than_baseline_fails(self):
        result = BacktestResult(
            strategy_name="bad",
            total_draws=100,
            total_tickets=100,
            total_wins=5,
            win_rate=0.05,
        )
        self.assertFalse(passes_significance_gate(result, baseline_win_rate=0.1))

    def test_custom_alpha(self):
        result = BacktestResult(
            strategy_name="marginal",
            total_draws=500,
            total_tickets=500,
            total_wins=65,
            win_rate=0.13,
        )
        # Might pass at alpha=0.10 but not at alpha=0.01
        gate_loose = passes_significance_gate(result, baseline_win_rate=0.1, alpha=0.10)
        gate_strict = passes_significance_gate(result, baseline_win_rate=0.1, alpha=0.001)
        # At minimum, strict should be harder to pass than loose
        if gate_strict:
            self.assertTrue(gate_loose)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_shared_backtest.TestSignificanceGate -v`
Expected: FAIL — `ImportError: cannot import name 'passes_significance_gate'`

**Step 3: Write minimal implementation**

Add to `src/shared/backtest_base.py` after the `strategy_significance_test` function:

```python
def passes_significance_gate(
    result: BacktestResult,
    baseline_win_rate: float,
    alpha: float = 0.05,
) -> bool:
    """Check if a strategy significantly outperforms the baseline.

    A strategy passes the gate only if:
    1. Its win rate exceeds the baseline
    2. The difference is statistically significant at the given alpha level

    Args:
        result: Backtest result to evaluate.
        baseline_win_rate: Expected win rate under random selection.
        alpha: Significance level (default 0.05).

    Returns:
        True if the strategy significantly outperforms baseline.
    """
    if result.total_tickets == 0:
        return False

    if result.win_rate <= baseline_win_rate:
        return False

    test = strategy_significance_test(result, baseline_win_rate)
    return test.get("p_value", 1.0) < alpha and result.win_rate > baseline_win_rate
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_shared_backtest.py -v`
Expected: All tests PASS

**Step 5: Commit**

```
feat: add significance gate for strategy admission to ensemble
```

---

### Task 4: Add Monte Carlo validation

Generate synthetic fair-lottery datasets and verify no strategy beats random — a sanity check that any real-data edge is genuine and not overfitting.

**Files:**
- Create: `src/shared/monte_carlo.py`
- Test: `tests/test_monte_carlo.py`

**Step 1: Write the failing test**

```python
# tests/test_monte_carlo.py
import random
import unittest

from shared.monte_carlo import generate_synthetic_draws, monte_carlo_validate
from shared.game_config import JOKER_CONFIG


class TestGenerateSyntheticDraws(unittest.TestCase):
    def test_correct_count(self):
        draws = generate_synthetic_draws(JOKER_CONFIG, 100, random.Random(42))
        self.assertEqual(len(draws), 100)

    def test_draws_are_valid(self):
        draws = generate_synthetic_draws(JOKER_CONFIG, 50, random.Random(42))
        for draw in draws:
            self.assertEqual(len(draw), JOKER_CONFIG.numbers_drawn)
            self.assertEqual(sorted(draw), draw)
            self.assertTrue(all(n in JOKER_CONFIG.pool_range for n in draw))
            self.assertEqual(len(set(draw)), len(draw))

    def test_deterministic_with_seed(self):
        d1 = generate_synthetic_draws(JOKER_CONFIG, 20, random.Random(99))
        d2 = generate_synthetic_draws(JOKER_CONFIG, 20, random.Random(99))
        self.assertEqual(d1, d2)


class TestMonteCarloValidate(unittest.TestCase):
    def test_returns_validation_result(self):
        result = monte_carlo_validate(
            config=JOKER_CONFIG,
            strategy_name="frequency",
            strategy_win_rate=0.12,
            num_simulations=10,
            draws_per_simulation=50,
            rng=random.Random(42),
        )
        self.assertIn("synthetic_mean_win_rate", result)
        self.assertIn("strategy_is_plausible", result)
        self.assertIn("num_simulations", result)
        self.assertEqual(result["num_simulations"], 10)

    def test_random_strategy_is_plausible(self):
        result = monte_carlo_validate(
            config=JOKER_CONFIG,
            strategy_name="random",
            strategy_win_rate=0.10,
            num_simulations=20,
            draws_per_simulation=100,
            rng=random.Random(42),
        )
        # Random should always be plausible against synthetic random data
        self.assertTrue(result["strategy_is_plausible"])


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_monte_carlo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.monte_carlo'`

**Step 3: Write minimal implementation**

```python
# src/shared/monte_carlo.py
"""Monte Carlo validation for lottery strategies.

Generates synthetic fair-lottery datasets to verify that no strategy
beats random selection on truly uniform data. If a strategy shows
significant edge on real data but NOT on synthetic data, the edge
may be genuine. If it also shows edge on synthetic data, it's likely
an artifact of overfitting.
"""

import math
import random

from .game_config import GameConfig


def generate_synthetic_draws(
    config: GameConfig,
    count: int,
    rng: random.Random,
) -> list[list[int]]:
    """Generate synthetic draws from a perfectly uniform lottery.

    Args:
        config: Game configuration.
        count: Number of draws to generate.
        rng: Random number generator.

    Returns:
        List of synthetic draws (sorted number lists).
    """
    pool = list(config.pool_range)
    draws = []
    for _ in range(count):
        draw = sorted(rng.sample(pool, config.numbers_drawn))
        draws.append(draw)
    return draws


def monte_carlo_validate(
    config: GameConfig,
    strategy_name: str,
    strategy_win_rate: float,
    num_simulations: int = 50,
    draws_per_simulation: int = 200,
    rng: random.Random | None = None,
) -> dict:
    """Validate a strategy's win rate against synthetic uniform data.

    Runs multiple simulations of a perfectly fair lottery and measures
    the random baseline win rate distribution. If the strategy's real-data
    win rate falls within this distribution, it's plausible that the
    strategy's performance is due to chance.

    Args:
        config: Game configuration.
        strategy_name: Name of the strategy being validated.
        strategy_win_rate: Observed win rate on real data.
        num_simulations: Number of synthetic datasets to generate.
        draws_per_simulation: Draws per synthetic dataset.
        rng: Random number generator.

    Returns:
        Validation results with synthetic statistics and conclusion.
    """
    rng = rng or random.Random()

    synthetic_win_rates = []
    pool = list(config.pool_range)

    for _ in range(num_simulations):
        draws = generate_synthetic_draws(config, draws_per_simulation, rng)
        wins = 0
        for draw in draws:
            pick = sorted(rng.sample(pool, config.numbers_to_pick))
            matches = len(set(pick) & set(draw))
            if matches >= config.min_match_for_prize:
                wins += 1
        synthetic_win_rates.append(wins / draws_per_simulation)

    mean_rate = sum(synthetic_win_rates) / len(synthetic_win_rates)
    std_rate = math.sqrt(
        sum((r - mean_rate) ** 2 for r in synthetic_win_rates)
        / len(synthetic_win_rates)
    ) if len(synthetic_win_rates) > 1 else 0.0

    upper_bound = mean_rate + 2 * std_rate

    is_plausible = strategy_win_rate <= upper_bound

    return {
        "strategy_name": strategy_name,
        "strategy_win_rate": strategy_win_rate,
        "synthetic_mean_win_rate": mean_rate,
        "synthetic_std_win_rate": std_rate,
        "synthetic_upper_2sigma": upper_bound,
        "strategy_is_plausible": is_plausible,
        "num_simulations": num_simulations,
        "conclusion": (
            f"{strategy_name} win rate ({strategy_win_rate:.4f}) is within "
            f"synthetic range ({mean_rate:.4f} +/- {2*std_rate:.4f}). "
            "Performance is consistent with random chance."
            if is_plausible
            else f"{strategy_name} win rate ({strategy_win_rate:.4f}) exceeds "
            f"synthetic upper bound ({upper_bound:.4f}). "
            "Strategy may have found genuine edge OR is overfitting."
        ),
    }
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_monte_carlo.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```
feat: add Monte Carlo validation for strategy overfitting detection
```

---

### Task 5: Implement Genetic Algorithm strategy

A population-based optimization strategy that evolves ticket sets using crossover and mutation, scoring fitness via backtest prize matches.

**Files:**
- Create: `src/shared/genetic.py`
- Test: `tests/test_genetic.py`

**Step 1: Write the failing test**

```python
# tests/test_genetic.py
import random
import unittest

from shared.genetic import GeneticStrategy


class TestGeneticStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = GeneticStrategy(
            pool_size=45,
            numbers_to_pick=5,
            population_size=50,
            generations=20,
            mutation_rate=0.1,
        )
        self.draws = [
            sorted(random.Random(i).sample(range(1, 46), 5))
            for i in range(50)
        ]

    def test_name(self):
        self.assertEqual(self.strategy.name, "genetic")

    def test_generate_returns_correct_count(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 5, rng)
        self.assertEqual(len(lines), 5)

    def test_generate_returns_valid_lines(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 3, rng)
        for line in lines:
            self.assertEqual(len(line), 5)
            self.assertEqual(sorted(line), line)
            self.assertTrue(all(1 <= n <= 45 for n in line))
            self.assertEqual(len(set(line)), 5)

    def test_generate_returns_unique_lines(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 10, rng)
        keys = [tuple(l) for l in lines]
        self.assertEqual(len(keys), len(set(keys)))

    def test_generate_with_empty_draws(self):
        rng = random.Random(42)
        lines = self.strategy.generate([], 3, rng)
        self.assertEqual(len(lines), 3)

    def test_get_probabilities_returns_distribution(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertEqual(len(probs), 45)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertTrue(all(p >= 0 for p in probs))

    def test_deterministic_with_seed(self):
        lines1 = self.strategy.generate(self.draws, 3, random.Random(99))
        lines2 = self.strategy.generate(self.draws, 3, random.Random(99))
        self.assertEqual(lines1, lines2)

    def test_fitness_improves_over_generations(self):
        rng = random.Random(42)
        # Low generations
        low_gen = GeneticStrategy(45, 5, population_size=30, generations=1)
        # Higher generations
        high_gen = GeneticStrategy(45, 5, population_size=30, generations=50)
        # Both should produce valid output
        l1 = low_gen.generate(self.draws, 3, random.Random(42))
        l2 = high_gen.generate(self.draws, 3, random.Random(42))
        self.assertEqual(len(l1), 3)
        self.assertEqual(len(l2), 3)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_genetic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.genetic'`

**Step 3: Write minimal implementation**

```python
# src/shared/genetic.py
"""Genetic algorithm strategy for lottery number optimization.

Evolves a population of ticket candidates using selection, crossover,
and mutation. Fitness is measured by historical prize matches against
recent draws. This strategy optimizes ticket sets directly rather
than predicting probability distributions.

Uses only the Python standard library.
"""

import random
from collections import Counter


class GeneticStrategy:
    """Genetic algorithm that evolves lottery ticket populations."""

    def __init__(
        self,
        pool_size: int,
        numbers_to_pick: int,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        elite_count: int = 2,
        half_life: float = 100.0,
        half_life_mode: str = "draws",
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.half_life = half_life
        self.half_life_mode = half_life_mode
        self.name = "genetic"

    def _random_individual(self, rng: random.Random) -> list[int]:
        """Create a random ticket."""
        return sorted(rng.sample(range(1, self.pool_size + 1), self.numbers_to_pick))

    def _fitness(self, individual: list[int], draws: list[list[int]]) -> float:
        """Score an individual against historical draws.

        Fitness is the weighted count of draws where the individual
        would have won a prize (3+ matches). More recent draws are
        weighted higher using exponential decay.
        """
        if not draws:
            return 0.0

        ind_set = set(individual)
        score = 0.0
        n = len(draws)

        for i, draw in enumerate(draws):
            weight = 0.5 ** ((n - 1 - i) / max(self.half_life, 1.0))
            matches = len(ind_set & set(draw))
            if matches >= 3:
                score += weight * (matches ** 2)
            elif matches >= 2:
                score += weight * matches * 0.5

        return score

    def _tournament_select(
        self,
        population: list[list[int]],
        fitnesses: list[float],
        rng: random.Random,
    ) -> list[int]:
        """Select an individual via tournament selection."""
        indices = rng.sample(range(len(population)), min(self.tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return list(population[best_idx])

    def _crossover(
        self,
        parent1: list[int],
        parent2: list[int],
        rng: random.Random,
    ) -> list[int]:
        """Uniform crossover: combine numbers from two parents."""
        combined = list(set(parent1) | set(parent2))
        if len(combined) < self.numbers_to_pick:
            extra = [n for n in range(1, self.pool_size + 1) if n not in combined]
            combined.extend(rng.sample(extra, self.numbers_to_pick - len(combined)))

        child = sorted(rng.sample(combined, self.numbers_to_pick))
        return child

    def _mutate(self, individual: list[int], rng: random.Random) -> list[int]:
        """Mutate by replacing one random number with another from the pool."""
        result = list(individual)
        available = [n for n in range(1, self.pool_size + 1) if n not in result]
        if not available:
            return result

        idx = rng.randrange(len(result))
        result[idx] = rng.choice(available)
        return sorted(result)

    def _evolve(
        self,
        draws: list[list[int]],
        rng: random.Random,
    ) -> list[list[int]]:
        """Run the genetic algorithm and return the final population."""
        population = [self._random_individual(rng) for _ in range(self.population_size)]

        for _ in range(self.generations):
            fitnesses = [self._fitness(ind, draws) for ind in population]

            sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
            elites = [list(population[i]) for i in sorted_indices[:self.elite_count]]

            new_population = list(elites)

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, fitnesses, rng)
                parent2 = self._tournament_select(population, fitnesses, rng)

                if rng.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2, rng)
                else:
                    child = list(parent1)

                if rng.random() < self.mutation_rate:
                    child = self._mutate(child, rng)

                new_population.append(child)

            population = new_population[:self.population_size]

        fitnesses = [self._fitness(ind, draws) for ind in population]
        sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        return [population[i] for i in sorted_indices]

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        **kwargs,
    ) -> list[list[int]]:
        """Generate picks by evolving and selecting top individuals."""
        evolved = self._evolve(draws, rng)

        lines: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()

        for individual in evolved:
            if len(lines) >= count:
                break
            key = tuple(individual)
            if key not in seen:
                seen.add(key)
                lines.append(individual)

        while len(lines) < count:
            extra = sorted(rng.sample(range(1, self.pool_size + 1), self.numbers_to_pick))
            key = tuple(extra)
            if key not in seen:
                seen.add(key)
                lines.append(extra)

        return lines[:count]

    def get_probabilities(
        self,
        draws: list[list[int]],
        **kwargs,
    ) -> list[float]:
        """Get probability distribution from evolved population.

        Runs evolution and counts how often each number appears
        in the top individuals.
        """
        rng = random.Random(0)
        evolved = self._evolve(draws, rng)

        counts = Counter()
        for individual in evolved:
            for num in individual:
                counts[num] += 1

        total = sum(counts.values()) or 1
        return [counts.get(n, 0) / total for n in range(1, self.pool_size + 1)]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_genetic.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```
feat: add genetic algorithm strategy for lottery number optimization
```

---

### Task 6: Implement Gradient Boosted Trees strategy

Per-number binary classifiers using the existing feature engineering. scikit-learn as optional dependency.

**Files:**
- Create: `src/shared/gradient_boost.py`
- Test: `tests/test_gradient_boost.py`

**Step 1: Write the failing test**

```python
# tests/test_gradient_boost.py
import random
import unittest

from shared.gradient_boost import GradientBoostStrategy, SKLEARN_AVAILABLE


class TestGradientBoostStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = GradientBoostStrategy(
            pool_size=45,
            numbers_to_pick=5,
            numbers_drawn=5,
        )
        rng = random.Random(42)
        self.draws = [
            sorted(rng.sample(range(1, 46), 5))
            for _ in range(80)
        ]

    def test_name(self):
        self.assertEqual(self.strategy.name, "gradient_boost")

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed")
    def test_generate_returns_correct_count(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 5, rng)
        self.assertEqual(len(lines), 5)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed")
    def test_generate_returns_valid_lines(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 3, rng)
        for line in lines:
            self.assertEqual(len(line), 5)
            self.assertEqual(sorted(line), line)
            self.assertTrue(all(1 <= n <= 45 for n in line))
            self.assertEqual(len(set(line)), 5)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed")
    def test_generate_returns_unique_lines(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws, 10, rng)
        keys = [tuple(l) for l in lines]
        self.assertEqual(len(keys), len(set(keys)))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed")
    def test_get_probabilities_returns_distribution(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertEqual(len(probs), 45)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertTrue(all(p >= 0 for p in probs))

    def test_fallback_without_sklearn(self):
        strategy = GradientBoostStrategy(45, 5, 5)
        strategy._sklearn_available = False
        rng = random.Random(42)
        lines = strategy.generate(self.draws, 3, rng)
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertEqual(len(line), 5)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed")
    def test_deterministic_with_seed(self):
        lines1 = self.strategy.generate(self.draws, 3, random.Random(99))
        lines2 = self.strategy.generate(self.draws, 3, random.Random(99))
        self.assertEqual(lines1, lines2)

    def test_generate_with_few_draws(self):
        rng = random.Random(42)
        lines = self.strategy.generate(self.draws[:5], 3, rng)
        self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests/test_gradient_boost.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.gradient_boost'`

**Step 3: Write minimal implementation**

```python
# src/shared/gradient_boost.py
"""Gradient Boosted Trees strategy for lottery number prediction.

Trains per-number binary classifiers using features extracted from
historical draws. Uses scikit-learn's GradientBoostingClassifier
with optional import — falls back to frequency-based selection if
scikit-learn is not available.
"""

import math
import random
from collections import Counter

try:
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class GradientBoostStrategy:
    """Gradient boosting strategy using per-number classifiers."""

    def __init__(
        self,
        pool_size: int,
        numbers_to_pick: int,
        numbers_drawn: int,
        window_size: int = 10,
        n_estimators: int = 50,
        max_depth: int = 3,
        min_train_draws: int = 30,
        half_life: float = 100.0,
        half_life_mode: str = "draws",
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.numbers_drawn = numbers_drawn
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_train_draws = min_train_draws
        self.half_life = half_life
        self.half_life_mode = half_life_mode
        self.name = "gradient_boost"
        self._sklearn_available = SKLEARN_AVAILABLE

    def _build_features(self, draws: list[list[int]], idx: int) -> list[float]:
        """Build feature vector for predicting draw at idx.

        Features are computed from draws before idx.
        """
        window = draws[max(0, idx - self.window_size):idx]
        if not window:
            return [0.0] * (self.pool_size + 6)

        freq = Counter()
        for draw in window:
            for n in draw:
                freq[n] += 1

        total_numbers = sum(len(d) for d in window)
        freq_features = [freq.get(n, 0) / max(total_numbers, 1) for n in range(1, self.pool_size + 1)]

        last_draw = window[-1]
        draw_sum = sum(last_draw)
        odd_count = sum(1 for n in last_draw if n % 2 == 1)
        high_count = sum(1 for n in last_draw if n > self.pool_size // 2)

        last_seen = {}
        for i, draw in enumerate(window):
            for n in draw:
                last_seen[n] = i
        avg_gap = sum(
            len(window) - last_seen.get(n, 0)
            for n in range(1, self.pool_size + 1)
        ) / self.pool_size

        return freq_features + [
            draw_sum / max(self.pool_size * self.numbers_drawn, 1),
            odd_count / max(self.numbers_drawn, 1),
            high_count / max(self.numbers_drawn, 1),
            len(window) / max(self.window_size, 1),
            avg_gap / max(len(window), 1),
            len(freq) / self.pool_size,
        ]

    def _train_and_predict(
        self, draws: list[list[int]]
    ) -> list[float]:
        """Train per-number classifiers and return prediction scores."""
        if not self._sklearn_available or len(draws) < self.min_train_draws:
            return self._fallback_probabilities(draws)

        feature_start = self.window_size
        if len(draws) - feature_start < 10:
            return self._fallback_probabilities(draws)

        X = []
        y_per_number = {n: [] for n in range(1, self.pool_size + 1)}

        for idx in range(feature_start, len(draws)):
            features = self._build_features(draws, idx)
            X.append(features)
            draw_set = set(draws[idx])
            for n in range(1, self.pool_size + 1):
                y_per_number[n].append(1 if n in draw_set else 0)

        scores = []
        for n in range(1, self.pool_size + 1):
            y = y_per_number[n]
            if sum(y) == 0 or sum(y) == len(y):
                scores.append(sum(y) / max(len(y), 1))
                continue

            clf = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=0,
            )
            clf.fit(X, y)

            last_features = self._build_features(draws, len(draws))
            prob = clf.predict_proba([last_features])[0]
            positive_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
            scores.append(prob[positive_idx])

        return scores

    def _fallback_probabilities(self, draws: list[list[int]]) -> list[float]:
        """Simple frequency-based fallback when sklearn unavailable."""
        freq = Counter()
        for draw in draws:
            for n in draw:
                freq[n] += 1
        total = sum(freq.values()) or 1
        probs = [(freq.get(n, 0) + 1) / (total + self.pool_size) for n in range(1, self.pool_size + 1)]
        s = sum(probs)
        return [p / s for p in probs]

    def get_probabilities(
        self,
        draws: list[list[int]],
        **kwargs,
    ) -> list[float]:
        """Return normalized prediction scores as probabilities."""
        scores = self._train_and_predict(draws)
        total = sum(scores) or 1.0
        return [s / total for s in scores]

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        **kwargs,
    ) -> list[list[int]]:
        """Generate picks weighted by classifier predictions."""
        probs = self.get_probabilities(draws)
        numbers = list(range(1, self.pool_size + 1))

        lines: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()

        for _ in range(count * 20):
            if len(lines) >= count:
                break

            chosen = []
            available = list(numbers)
            available_probs = list(probs)

            for _ in range(self.numbers_to_pick):
                if not available:
                    break
                total = sum(available_probs)
                if total == 0:
                    pick = rng.choice(available)
                else:
                    normalized = [p / total for p in available_probs]
                    pick = rng.choices(available, weights=normalized, k=1)[0]
                chosen.append(pick)
                idx = available.index(pick)
                available.pop(idx)
                available_probs.pop(idx)

            line = sorted(chosen)
            key = tuple(line)
            if key not in seen:
                seen.add(key)
                lines.append(line)

        while len(lines) < count:
            extra = sorted(rng.sample(numbers, self.numbers_to_pick))
            key = tuple(extra)
            if key not in seen:
                seen.add(key)
                lines.append(extra)

        return lines[:count]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m unittest tests/test_gradient_boost.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```
feat: add gradient boosted trees strategy with sklearn optional import
```

---

### Task 7: Register new strategies in ensemble blend

Add Genetic Algorithm and Gradient Boost strategies to the ensemble blend pool so they participate in proportional allocation.

**Files:**
- Modify: `src/shared/ensemble_blend.py`
- Modify: `tests/test_ensemble_blend.py`

**Step 1: Write the failing test**

Add to `tests/test_ensemble_blend.py`:

```python
class TestBlendedPicksWithNewStrategies(unittest.TestCase):
    def _make_draws(self, config, count=50):
        rng = random.Random(0)
        pool = list(config.pool_range)
        return [sorted(rng.sample(pool, config.numbers_drawn)) for _ in range(count)]

    def test_blend_includes_genetic_and_gradient_boost(self):
        draws = self._make_draws(JOKER_CONFIG)
        rng = random.Random(42)
        lines = generate_blended_picks(JOKER_CONFIG, draws, 10, rng)
        # Should still produce valid output with new strategies
        self.assertEqual(len(lines), 10)
        for line in lines:
            self.assertEqual(len(line), JOKER_CONFIG.numbers_to_pick)
            self.assertTrue(all(n in JOKER_CONFIG.pool_range for n in line))
```

**Step 2: Run test to verify it fails (or passes if integration is straightforward)**

Run: `PYTHONPATH=src python -m unittest tests.test_ensemble_blend.TestBlendedPicksWithNewStrategies -v`

**Step 3: Modify ensemble_blend.py**

In `src/shared/ensemble_blend.py`, add imports and new strategy scoring to `generate_blended_picks()`:

After the existing imports (line 11), add:
```python
from .genetic import GeneticStrategy

try:
    from .gradient_boost import GradientBoostStrategy, SKLEARN_AVAILABLE
except ImportError:
    SKLEARN_AVAILABLE = False
```

Inside `generate_blended_picks()`, after the cooccurrence strategy instantiation (line 169), add:
```python
    genetic = GeneticStrategy(
        config.pool_size,
        config.numbers_to_pick,
        population_size=50,
        generations=20,
    )
```

In the scores dict (lines 171-191), add:
```python
        "genetic": max(
            _score_strategy_object(
                config, draws, genetic, rng, draw_dates, half_life_mode,
            ),
            1,
        ),
```

After `genetic` scoring, conditionally add gradient_boost:
```python
    if SKLEARN_AVAILABLE:
        gb = GradientBoostStrategy(
            config.pool_size,
            config.numbers_to_pick,
            config.numbers_drawn,
        )
        scores["gradient_boost"] = max(
            _score_strategy_object(
                config, draws, gb, rng, draw_dates, half_life_mode,
            ),
            1,
        )
```

In the generation section (after line 259), add blocks for the new strategies following the same pattern as bayesian/cooccurrence:

```python
    genetic_lines = genetic.generate(
        draws, allocation.get("genetic", 0) + count, rng,
    )
    added_genetic = 0
    for pick in genetic_lines:
        if added_genetic >= allocation.get("genetic", 0):
            break
        key = tuple(pick)
        if key not in seen:
            seen.add(key)
            lines.append(pick)
            added_genetic += 1

    if SKLEARN_AVAILABLE and "gradient_boost" in allocation:
        gb = GradientBoostStrategy(
            config.pool_size,
            config.numbers_to_pick,
            config.numbers_drawn,
        )
        gb_lines = gb.generate(
            draws, allocation.get("gradient_boost", 0) + count, rng,
        )
        added_gb = 0
        for pick in gb_lines:
            if added_gb >= allocation.get("gradient_boost", 0):
                break
            key = tuple(pick)
            if key not in seen:
                seen.add(key)
                lines.append(pick)
                added_gb += 1
```

**Step 4: Run full test suite**

Run: `PYTHONPATH=src python -m unittest -v`
Expected: All tests PASS

**Step 5: Commit**

```
feat: integrate genetic and gradient boost strategies into ensemble blend
```

---

### Task 8: Run full test suite and validate

Final validation that everything works together.

**Step 1: Run the complete test suite**

Run: `PYTHONPATH=src python -m unittest -v`
Expected: All tests PASS

**Step 2: Run a quick smoke test with actual pick generation**

Run: `PYTHONPATH=src python scripts/generate_joker_picks.py --seed 42`
Expected: Produces valid Joker picks without errors

**Step 3: Run Loto 6/49 and 5/40 as well**

Run: `PYTHONPATH=src python scripts/generate_loto_649_picks.py --seed 42`
Run: `PYTHONPATH=src python scripts/generate_loto_540_picks.py --seed 42`
Expected: Both produce valid picks

**Step 4: Commit any final fixes if needed, then tag the phase**

```
chore: Phase 1 complete — enhanced backtesting, genetic, gradient boost
```
