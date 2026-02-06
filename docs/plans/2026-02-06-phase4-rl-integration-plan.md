# Phase 4: Reinforcement Learning + Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a REINFORCE policy gradient RL agent as a new ensemble strategy, add significance-gating to the ensemble blend, and create a holdout evaluation script.

**Architecture:** The RL agent uses a small feedforward policy network (PyTorch, optional) that takes the last K draws as state and outputs per-number selection probabilities. Training uses REINFORCE with a match-count reward. Significance-gating uses the existing `passes_significance_gate()` from `backtest_base.py` to filter weak strategies before blending. A holdout evaluation script tests all strategies on reserved data.

**Tech Stack:** Python 3.12, PyTorch (optional import), standard library only for non-DL code

---

## Context

### Project layout
```
src/shared/           — shared strategies, all follow the same pattern
tests/                — test files, run with PYTHONPATH=src python -m pytest tests/ -v
scripts/              — CLI scripts for pick generation
```

### Strategy pattern (follow exactly)
Every strategy in `src/shared/` follows this interface used by `ensemble_blend.py`:
- `name: str` class attribute (e.g., `name = "rl"`)
- `__init__(self, pool_size, numbers_to_pick, ...)` — pool_size is the max number, numbers_to_pick is how many to select
- `get_probabilities(self, draws, **kwargs) -> list[float]` — returns probability vector of length pool_size
- `generate(self, draws, count, rng, **kwargs) -> list[list[int]]` — returns sorted pick lists

Key conventions:
- Input `draws` is `list[list[int]]` — each draw is sorted 1-indexed main numbers
- Output picks are `list[list[int]]` — each pick is sorted 1-indexed numbers
- PyTorch is optional: wrap imports in try/except, fall back to uniform probabilities
- Deduplication: use `seen: set[tuple[int, ...]]` to avoid duplicate picks

### How ensemble_blend.py integrates strategies
In `generate_blended_picks()` at `src/shared/ensemble_blend.py`:
1. Strategies are scored via `_score_strategy_object()` (walk-forward backtest)
2. Scores become softmax weights via `_allocate_counts()`
3. Each strategy generates its allocated picks
4. Portfolio optimization selects final diverse set

To add a new strategy:
- Import it at the top (inside try/except for optional deps)
- Add scoring call to the `scores` dict
- Add generation block after the scoring section

### Running tests
```bash
PYTHONPATH=src python -m pytest tests/ -v              # all tests
PYTHONPATH=src python -m pytest tests/test_rl_agent.py -v   # single file
```

---

### Task 1: RL Agent Module

**Files:**
- Create: `src/shared/rl_agent.py`
- Create: `tests/test_rl_agent.py`

**Step 1: Write the failing tests**

Create `tests/test_rl_agent.py`:

```python
import unittest
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestRLAgent(unittest.TestCase):
    def setUp(self):
        from shared.rl_agent import RLAgent
        self.rng = random.Random(42)
        self.agent = RLAgent(pool_size=10, numbers_to_pick=3)
        self.draws = [
            sorted(random.Random(i).sample(range(1, 11), 3))
            for i in range(50)
        ]

    def test_name(self):
        self.assertEqual(self.agent.name, "rl")

    def test_generate_returns_correct_count(self):
        picks = self.agent.generate(self.draws, 3, self.rng)
        self.assertEqual(len(picks), 3)

    def test_generate_returns_sorted_numbers(self):
        picks = self.agent.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(pick, sorted(pick))

    def test_generate_numbers_in_range(self):
        picks = self.agent.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(len(pick), 3)
            for n in pick:
                self.assertGreaterEqual(n, 1)
                self.assertLessEqual(n, 10)

    def test_generate_unique_picks(self):
        picks = self.agent.generate(self.draws, 5, self.rng)
        keys = [tuple(p) for p in picks]
        self.assertEqual(len(keys), len(set(keys)))

    def test_generate_with_few_draws(self):
        picks = self.agent.generate(self.draws[:3], 2, self.rng)
        self.assertEqual(len(picks), 2)

    def test_get_probabilities_length(self):
        probs = self.agent.get_probabilities(self.draws)
        self.assertEqual(len(probs), 10)

    def test_get_probabilities_sum_to_one(self):
        probs = self.agent.get_probabilities(self.draws)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_get_probabilities_all_positive(self):
        probs = self.agent.get_probabilities(self.draws)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)

    def test_training_episodes_configurable(self):
        from shared.rl_agent import RLAgent
        agent = RLAgent(pool_size=10, numbers_to_pick=3, episodes=5)
        picks = agent.generate(self.draws, 2, self.rng)
        self.assertEqual(len(picks), 2)

    def test_reward_based_on_matches(self):
        from shared.rl_agent import _compute_reward
        # 0 matches = 0 reward
        self.assertEqual(_compute_reward(0, 3), 0.0)
        # partial matches get increasing reward
        r1 = _compute_reward(1, 3)
        r2 = _compute_reward(2, 3)
        r3 = _compute_reward(3, 3)
        self.assertGreater(r2, r1)
        self.assertGreater(r3, r2)


class TestRLAgentFallback(unittest.TestCase):
    def test_fallback_without_torch(self):
        from shared.rl_agent import RLAgent
        agent = RLAgent(pool_size=10, numbers_to_pick=3)
        rng = random.Random(42)
        draws = [sorted(random.Random(i).sample(range(1, 11), 3)) for i in range(5)]
        # Should not raise even if torch unavailable (falls back to uniform)
        picks = agent.generate(draws, 2, rng)
        self.assertEqual(len(picks), 2)


class TestRLAgentImport(unittest.TestCase):
    def test_module_importable(self):
        import shared.rl_agent


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_rl_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.rl_agent'`

**Step 3: Write the implementation**

Create `src/shared/rl_agent.py`:

```python
"""Reinforcement Learning agent for lottery number selection.

Uses REINFORCE (policy gradient) with a small feedforward network.
The agent learns a policy that maps recent draw history to per-number
selection probabilities, trained by sampling actions and weighting
updates by match-count rewards.

Falls back to uniform probabilities when PyTorch is unavailable.
"""

import random

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

MINIMUM_DRAWS = 5


def _compute_reward(matches: int, numbers_to_pick: int) -> float:
    """Compute reward based on how many numbers matched.

    Uses exponential scaling so near-misses are valued but
    full matches are rewarded substantially more.
    """
    if matches == 0:
        return 0.0
    return (matches / numbers_to_pick) ** 2


def _encode_recent_draws(
    draws: list[list[int]], pool_size: int, window: int,
) -> list[float]:
    """Encode the last `window` draws into a flat feature vector.

    For each of the last `window` draws, creates a multi-hot vector
    of length pool_size, then concatenates them. If fewer than `window`
    draws exist, pads with zeros.

    Returns:
        Flat list of length window * pool_size.
    """
    features: list[float] = []
    start = max(0, len(draws) - window)
    for i in range(window):
        idx = start + i
        vec = [0.0] * pool_size
        if idx < len(draws):
            for number in draws[idx]:
                vec[number - 1] = 1.0
        features.extend(vec)
    return features


if TORCH_AVAILABLE:

    class _PolicyNet(nn.Module):
        """Small feedforward policy network.

        Input: flattened recent-draw features (window * pool_size).
        Output: logits over pool_size numbers.
        """

        def __init__(self, input_size: int, pool_size: int, hidden_size: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, pool_size),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


class RLAgent:
    """REINFORCE policy gradient agent for lottery number selection.

    Trains a small policy network that maps recent draw history to
    per-number probabilities. Uses match-count as reward signal.

    Falls back to uniform sampling when PyTorch is unavailable or
    when insufficient historical data is provided.
    """

    name = "rl"

    def __init__(
        self,
        pool_size: int,
        numbers_to_pick: int,
        window: int = 10,
        hidden_size: int = 64,
        episodes: int = 50,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.window = window
        self.hidden_size = hidden_size
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.discount = discount

    def _uniform_probabilities(self) -> list[float]:
        return [1.0 / self.pool_size] * self.pool_size

    def _should_fallback(self, draws: list[list[int]]) -> bool:
        return not TORCH_AVAILABLE or len(draws) < MINIMUM_DRAWS

    def _train_policy(self, draws: list[list[int]]) -> list[float]:
        """Train REINFORCE agent and return final probability vector."""
        input_size = self.window * self.pool_size
        policy = _PolicyNet(input_size, self.pool_size, self.hidden_size)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate)

        min_train = max(self.window, MINIMUM_DRAWS)

        for _ in range(self.episodes):
            log_probs_list = []
            rewards_list = []

            for t in range(min_train, len(draws)):
                state = _encode_recent_draws(draws[:t], self.pool_size, self.window)
                state_tensor = torch.tensor([state], dtype=torch.float32)

                logits = policy(state_tensor).squeeze(0)
                probs = torch.softmax(logits, dim=0)

                # Sample numbers_to_pick numbers without replacement
                selected_indices = []
                episode_log_prob = torch.tensor(0.0)
                remaining_probs = probs.clone()

                for _ in range(self.numbers_to_pick):
                    dist = torch.distributions.Categorical(remaining_probs)
                    idx = dist.sample()
                    episode_log_prob = episode_log_prob + dist.log_prob(idx)
                    selected_indices.append(idx.item())
                    remaining_probs[idx] = 0.0
                    prob_sum = remaining_probs.sum()
                    if prob_sum > 0:
                        remaining_probs = remaining_probs / prob_sum

                pick = sorted(i + 1 for i in selected_indices)
                actual_draw = draws[t]
                matches = len(set(pick) & set(actual_draw))
                reward = _compute_reward(matches, self.numbers_to_pick)

                log_probs_list.append(episode_log_prob)
                rewards_list.append(reward)

            if not log_probs_list:
                continue

            # Compute discounted returns
            returns = []
            g = 0.0
            for r in reversed(rewards_list):
                g = r + self.discount * g
                returns.insert(0, g)

            returns_tensor = torch.tensor(returns, dtype=torch.float32)
            if returns_tensor.std() > 0:
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

            # Policy gradient loss
            loss = torch.tensor(0.0)
            for lp, ret in zip(log_probs_list, returns_tensor):
                loss = loss - lp * ret

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Extract final probabilities from the most recent state
        policy.eval()
        with torch.no_grad():
            state = _encode_recent_draws(draws, self.pool_size, self.window)
            state_tensor = torch.tensor([state], dtype=torch.float32)
            logits = policy(state_tensor).squeeze(0)
            probs = torch.softmax(logits, dim=0).tolist()

        return probs

    def get_probabilities(
        self, draws: list[list[int]], **kwargs,
    ) -> list[float]:
        """Return probability distribution over numbers 1..pool_size."""
        if self._should_fallback(draws):
            return self._uniform_probabilities()
        return self._train_policy(draws)

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        **kwargs,
    ) -> list[list[int]]:
        """Generate unique lottery picks using the trained RL policy."""
        probabilities = self.get_probabilities(draws, **kwargs)
        numbers = list(range(1, self.pool_size + 1))

        lines: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        max_attempts = count * 200

        attempts = 0
        while len(lines) < count and attempts < max_attempts:
            pick = _sample_without_replacement(
                numbers, probabilities, self.numbers_to_pick, rng,
            )
            key = tuple(pick)
            if key not in seen:
                seen.add(key)
                lines.append(pick)
            attempts += 1

        return lines


def _sample_without_replacement(
    numbers: list[int],
    weights: list[float],
    count: int,
    rng: random.Random,
) -> list[int]:
    """Weighted sampling without replacement, returning sorted picks."""
    pool = list(numbers)
    pool_weights = list(weights)
    chosen: list[int] = []

    for _ in range(count):
        if not pool:
            break
        pick = rng.choices(pool, weights=pool_weights, k=1)[0]
        idx = pool.index(pick)
        chosen.append(pick)
        pool.pop(idx)
        pool_weights.pop(idx)

    return sorted(chosen)
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_rl_agent.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/shared/rl_agent.py tests/test_rl_agent.py
git commit -m "feat: add REINFORCE policy gradient RL agent strategy"
```

---

### Task 2: Integrate RL Agent into Ensemble Blend

**Files:**
- Modify: `src/shared/ensemble_blend.py` (lines 30-43 for imports, lines 290-315 for scoring, lines 428-452 for generation)
- Modify: `tests/test_ensemble_blend.py`

**Step 1: Write the failing test**

Add to `tests/test_ensemble_blend.py`:

```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add this test class at the end of the file, before `if __name__`:

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestRLBlendIntegration(unittest.TestCase):
    def _make_draws(self, config, count=50):
        rng = random.Random(0)
        pool = list(config.pool_range)
        return [sorted(rng.sample(pool, config.numbers_drawn)) for _ in range(count)]

    def test_blend_includes_rl_strategy(self):
        draws = self._make_draws(JOKER_CONFIG)
        rng = random.Random(42)
        lines = generate_blended_picks(JOKER_CONFIG, draws, 5, rng)
        self.assertEqual(len(lines), 5)
        for line in lines:
            self.assertEqual(len(line), JOKER_CONFIG.numbers_to_pick)
            self.assertTrue(all(n in JOKER_CONFIG.pool_range for n in line))
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_ensemble_blend.py::TestRLBlendIntegration -v`
Expected: FAIL — test class doesn't exist yet

**Step 3: Modify ensemble_blend.py**

In `src/shared/ensemble_blend.py`, add the RL import alongside other torch strategies (around line 35-43):

```python
try:
    from .lstm_strategy import LSTMStrategy
    from .tcn_strategy import TCNStrategy
    from .transformer_strategy import TransformerStrategy
    from .normalizing_flows import NormalizingFlowStrategy
    from .rl_agent import RLAgent
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
```

In the `generate_blended_picks()` function, add RL scoring after the other torch strategy scoring (around line 304-315):

Inside the `if _TORCH_AVAILABLE:` block that scores strategies, add after the normalizing_flow scoring:

```python
        rl_scoring = RLAgent(
            config.pool_size, config.numbers_to_pick, episodes=5, window=10,
        )
```

And add `("rl", rl_scoring)` to the list of strategies being scored:

```python
        for strat_name, strat in [
            ("lstm", lstm_scoring),
            ("tcn", tcn_scoring),
            ("transformer", xfmr_scoring),
            ("normalizing_flow", nf_scoring),
            ("rl", rl_scoring),
        ]:
```

In the generation section (around line 428-452), add RL generation. Add after `nf_gen`:

```python
        rl_gen = RLAgent(config.pool_size, config.numbers_to_pick)
```

And add `("rl", rl_gen)` to the generation list:

```python
        for strat_name, strat in [
            ("lstm", lstm_gen),
            ("tcn", tcn_gen),
            ("transformer", xfmr_gen),
            ("normalizing_flow", nf_gen),
            ("rl", rl_gen),
        ]:
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_ensemble_blend.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/shared/ensemble_blend.py tests/test_ensemble_blend.py
git commit -m "feat: integrate RL agent into ensemble blend"
```

---

### Task 3: Significance-Gated Strategy Admission

**Files:**
- Modify: `src/shared/ensemble_blend.py`
- Create: `tests/test_significance_gate_integration.py`

**Step 1: Write the failing test**

Create `tests/test_significance_gate_integration.py`:

```python
import random
import unittest

from shared.ensemble_blend import generate_blended_picks, _apply_significance_gate
from shared.game_config import JOKER_CONFIG


class TestSignificanceGateFunction(unittest.TestCase):
    def test_gate_removes_low_scoring_strategies(self):
        scores = {"random": 5, "frequency": 5, "bayesian": 5, "weak": 1}
        baseline_score = 5
        total_draws = 100
        gated = _apply_significance_gate(scores, baseline_score, total_draws)
        # "weak" should be removed since it doesn't outperform baseline
        self.assertNotIn("weak", gated)
        # random is always kept as the baseline
        self.assertIn("random", gated)

    def test_gate_keeps_strong_strategies(self):
        scores = {"random": 5, "frequency": 20, "bayesian": 15}
        baseline_score = 5
        total_draws = 100
        gated = _apply_significance_gate(scores, baseline_score, total_draws)
        self.assertIn("frequency", gated)
        self.assertIn("bayesian", gated)

    def test_gate_always_keeps_random(self):
        scores = {"random": 10}
        gated = _apply_significance_gate(scores, 10, 100)
        self.assertIn("random", gated)

    def test_gate_with_insufficient_data_keeps_all(self):
        scores = {"random": 1, "frequency": 1, "bayesian": 1}
        gated = _apply_significance_gate(scores, 1, 5)
        self.assertEqual(len(gated), 3)


class TestSignificanceGateEndToEnd(unittest.TestCase):
    def _make_draws(self, config, count=50):
        rng = random.Random(0)
        pool = list(config.pool_range)
        return [sorted(rng.sample(pool, config.numbers_drawn)) for _ in range(count)]

    def test_blend_with_significance_gating(self):
        draws = self._make_draws(JOKER_CONFIG, count=100)
        rng = random.Random(42)
        lines = generate_blended_picks(JOKER_CONFIG, draws, 5, rng)
        self.assertEqual(len(lines), 5)
        for line in lines:
            self.assertEqual(len(line), JOKER_CONFIG.numbers_to_pick)
            self.assertTrue(all(n in JOKER_CONFIG.pool_range for n in line))


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_significance_gate_integration.py -v`
Expected: FAIL — `ImportError: cannot import name '_apply_significance_gate'`

**Step 3: Implement significance gating**

In `src/shared/ensemble_blend.py`, add this function before `generate_blended_picks()`:

```python
def _apply_significance_gate(
    scores: dict[str, int],
    baseline_score: int,
    total_draws: int,
    min_draws: int = 30,
) -> dict[str, int]:
    """Remove strategies that don't significantly outperform random.

    Uses a simple binomial proportion test. Strategies must have a win
    rate significantly above baseline to be included.

    Always keeps "random" as the baseline strategy.

    Args:
        scores: Strategy name -> win count from backtesting.
        baseline_score: Win count of the random strategy.
        total_draws: Number of draws in the scoring window.
        min_draws: Minimum draws needed for gating (skip if fewer).

    Returns:
        Filtered scores dict with only significant strategies and random.
    """
    if total_draws < min_draws:
        return dict(scores)

    baseline_rate = baseline_score / total_draws if total_draws > 0 else 0.0

    gated: dict[str, int] = {}
    for name, score in scores.items():
        if name == "random":
            gated[name] = score
            continue

        strategy_rate = score / total_draws if total_draws > 0 else 0.0
        if strategy_rate <= baseline_rate:
            continue

        # Simple z-test for proportion difference
        se = math.sqrt(baseline_rate * (1 - baseline_rate) / total_draws) if baseline_rate > 0 and baseline_rate < 1 else 0.0
        if se > 0:
            z = (strategy_rate - baseline_rate) / se
            if z > 1.645:  # One-tailed p < 0.05
                gated[name] = score
        else:
            gated[name] = score

    if len(gated) < 2:
        return dict(scores)

    return gated
```

Then in `generate_blended_picks()`, after computing all scores and before the softmax, add:

```python
    # Apply significance gate
    baseline_score = scores.get("random", 1)
    scores = _apply_significance_gate(scores, baseline_score, len(scoring_draws))
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_significance_gate_integration.py -v`
Expected: All tests PASS

Also run existing ensemble tests:
Run: `PYTHONPATH=src python -m pytest tests/test_ensemble_blend.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/shared/ensemble_blend.py tests/test_significance_gate_integration.py
git commit -m "feat: add significance-gated strategy admission to ensemble blend"
```

---

### Task 4: Holdout Evaluation Script

**Files:**
- Create: `scripts/evaluate_holdout.py`
- Create: `tests/test_evaluate_holdout.py`

**Step 1: Write the failing test**

Create `tests/test_evaluate_holdout.py`:

```python
import random
import unittest

from shared.holdout_evaluator import (
    evaluate_strategy_on_holdout,
    evaluate_all_strategies,
    HoldoutResult,
)
from shared.game_config import JOKER_CONFIG, LOTO_649_CONFIG, LOTO_540_CONFIG


class TestHoldoutResult(unittest.TestCase):
    def test_result_fields(self):
        result = HoldoutResult(
            strategy_name="test",
            game_name="joker",
            holdout_size=50,
            wins=5,
            win_rate=0.1,
            matches_distribution={2: 3, 3: 2},
        )
        self.assertEqual(result.strategy_name, "test")
        self.assertEqual(result.game_name, "joker")
        self.assertEqual(result.holdout_size, 50)
        self.assertEqual(result.win_rate, 0.1)


class TestEvaluateStrategyOnHoldout(unittest.TestCase):
    def _make_draws(self, config, count=100):
        rng = random.Random(0)
        pool = list(config.pool_range)
        return [sorted(rng.sample(pool, config.numbers_drawn)) for _ in range(count)]

    def test_evaluate_random_on_holdout(self):
        draws = self._make_draws(JOKER_CONFIG, count=100)
        result = evaluate_strategy_on_holdout(
            config=JOKER_CONFIG,
            strategy_name="random",
            train_draws=draws[:80],
            holdout_draws=draws[80:],
            rng=random.Random(42),
        )
        self.assertEqual(result.strategy_name, "random")
        self.assertEqual(result.game_name, "joker")
        self.assertEqual(result.holdout_size, 20)
        self.assertIsInstance(result.win_rate, float)

    def test_evaluate_frequency_on_holdout(self):
        draws = self._make_draws(JOKER_CONFIG, count=100)
        result = evaluate_strategy_on_holdout(
            config=JOKER_CONFIG,
            strategy_name="frequency",
            train_draws=draws[:80],
            holdout_draws=draws[80:],
            rng=random.Random(42),
        )
        self.assertEqual(result.strategy_name, "frequency")
        self.assertGreaterEqual(result.win_rate, 0.0)


class TestEvaluateAllStrategies(unittest.TestCase):
    def _make_draws(self, config, count=80):
        rng = random.Random(0)
        pool = list(config.pool_range)
        return [sorted(rng.sample(pool, config.numbers_drawn)) for _ in range(count)]

    def test_evaluate_all_returns_results_for_each_strategy(self):
        draws = self._make_draws(JOKER_CONFIG, count=80)
        results = evaluate_all_strategies(
            config=JOKER_CONFIG,
            train_draws=draws[:60],
            holdout_draws=draws[60:],
            rng=random.Random(42),
        )
        self.assertGreater(len(results), 0)
        names = [r.strategy_name for r in results]
        self.assertIn("random", names)
        self.assertIn("frequency", names)

    def test_evaluate_all_for_multiple_games(self):
        for config in [JOKER_CONFIG, LOTO_649_CONFIG, LOTO_540_CONFIG]:
            draws = self._make_draws(config, count=80)
            results = evaluate_all_strategies(
                config=config,
                train_draws=draws[:60],
                holdout_draws=draws[60:],
                rng=random.Random(42),
            )
            self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_evaluate_holdout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.holdout_evaluator'`

**Step 3: Write the holdout evaluator module**

Create `src/shared/holdout_evaluator.py`:

```python
"""Holdout evaluation for lottery strategies.

Evaluates each strategy on a reserved holdout set that was never
used during development or backtesting. Produces per-strategy
match statistics.
"""

import random
from dataclasses import dataclass, field
from collections import Counter

from .game_config import GameConfig
from .game_strategies import (
    generate_frequency_picks,
    generate_random_picks,
    is_prize_winner,
)
from .bayesian import BayesianScorer
from .cooccurrence import CooccurrenceStrategy
from .genetic import GeneticStrategy
from .recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE

try:
    from .gradient_boost import GradientBoostStrategy, SKLEARN_AVAILABLE as _GB_AVAILABLE
except ImportError:
    _GB_AVAILABLE = False

try:
    from .lstm_strategy import LSTMStrategy
    from .tcn_strategy import TCNStrategy
    from .transformer_strategy import TransformerStrategy
    from .normalizing_flows import NormalizingFlowStrategy
    from .rl_agent import RLAgent
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class HoldoutResult:
    """Result of evaluating a strategy on holdout data."""

    strategy_name: str
    game_name: str
    holdout_size: int
    wins: int = 0
    win_rate: float = 0.0
    matches_distribution: dict[int, int] = field(default_factory=dict)


def evaluate_strategy_on_holdout(
    config: GameConfig,
    strategy_name: str,
    train_draws: list[list[int]],
    holdout_draws: list[list[int]],
    rng: random.Random,
    half_life: float = DEFAULT_HALF_LIFE,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    draw_dates: list[str] | None = None,
) -> HoldoutResult:
    """Evaluate a single strategy on holdout draws.

    Generates one pick per holdout draw using the training data,
    then scores against the actual holdout draw.
    """
    wins = 0
    match_counts: list[int] = []

    for i, holdout_draw in enumerate(holdout_draws):
        available_draws = train_draws + holdout_draws[:i]
        pick = _generate_single_pick(
            config, strategy_name, available_draws, rng,
            half_life, half_life_mode, draw_dates,
        )
        if pick is None:
            match_counts.append(0)
            continue

        matches = len(set(pick) & set(holdout_draw))
        match_counts.append(matches)
        if is_prize_winner(config, pick, holdout_draw):
            wins += 1

    distribution = dict(Counter(match_counts))
    holdout_size = len(holdout_draws)
    win_rate = wins / holdout_size if holdout_size > 0 else 0.0

    return HoldoutResult(
        strategy_name=strategy_name,
        game_name=config.name,
        holdout_size=holdout_size,
        wins=wins,
        win_rate=win_rate,
        matches_distribution=distribution,
    )


def _generate_single_pick(
    config: GameConfig,
    strategy_name: str,
    draws: list[list[int]],
    rng: random.Random,
    half_life: float,
    half_life_mode: str,
    draw_dates: list[str] | None,
) -> list[int] | None:
    """Generate a single pick using the named strategy."""
    if strategy_name == "random":
        picks = generate_random_picks(config, 1, rng)
    elif strategy_name == "frequency":
        picks = generate_frequency_picks(
            config, draws, 1, rng,
            half_life=half_life, draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
    elif strategy_name == "bayesian":
        scorer = BayesianScorer(
            config.pool_size, config.numbers_to_pick,
            half_life=half_life, half_life_mode=half_life_mode,
        )
        picks = scorer.generate(
            draws, 1, rng,
            draw_dates=draw_dates, half_life_mode=half_life_mode,
        )
    elif strategy_name == "cooccurrence":
        strat = CooccurrenceStrategy(
            config.pool_size, config.numbers_to_pick,
            half_life=half_life, half_life_mode=half_life_mode,
        )
        picks = strat.generate(
            draws, 1, rng,
            draw_dates=draw_dates, half_life_mode=half_life_mode,
        )
    elif strategy_name == "genetic":
        strat = GeneticStrategy(
            config.pool_size, config.numbers_to_pick,
            population_size=20, generations=5,
        )
        picks = strat.generate(draws, 1, rng)
    elif strategy_name == "gradient_boost" and _GB_AVAILABLE:
        strat = GradientBoostStrategy(
            config.pool_size, config.numbers_to_pick, config.numbers_drawn,
        )
        picks = strat.generate(draws, 1, rng)
    elif strategy_name == "lstm" and _TORCH_AVAILABLE:
        strat = LSTMStrategy(config.pool_size, config.numbers_to_pick, epochs=5)
        picks = strat.generate(draws, 1, rng)
    elif strategy_name == "tcn" and _TORCH_AVAILABLE:
        strat = TCNStrategy(config.pool_size, config.numbers_to_pick, epochs=5)
        picks = strat.generate(draws, 1, rng)
    elif strategy_name == "transformer" and _TORCH_AVAILABLE:
        strat = TransformerStrategy(config.pool_size, config.numbers_to_pick, epochs=5)
        picks = strat.generate(draws, 1, rng)
    elif strategy_name == "normalizing_flow" and _TORCH_AVAILABLE:
        strat = NormalizingFlowStrategy(config.pool_size, config.numbers_to_pick, epochs=5)
        picks = strat.generate(draws, 1, rng)
    elif strategy_name == "rl" and _TORCH_AVAILABLE:
        strat = RLAgent(config.pool_size, config.numbers_to_pick, episodes=5)
        picks = strat.generate(draws, 1, rng)
    else:
        return None

    return picks[0] if picks else None


def evaluate_all_strategies(
    config: GameConfig,
    train_draws: list[list[int]],
    holdout_draws: list[list[int]],
    rng: random.Random,
    half_life: float = DEFAULT_HALF_LIFE,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    draw_dates: list[str] | None = None,
) -> list[HoldoutResult]:
    """Evaluate all available strategies on holdout data."""
    strategy_names = ["random", "frequency", "bayesian", "cooccurrence", "genetic"]

    if _GB_AVAILABLE:
        strategy_names.append("gradient_boost")

    if _TORCH_AVAILABLE:
        strategy_names.extend(["lstm", "tcn", "transformer", "normalizing_flow", "rl"])

    results = []
    for name in strategy_names:
        result = evaluate_strategy_on_holdout(
            config=config,
            strategy_name=name,
            train_draws=train_draws,
            holdout_draws=holdout_draws,
            rng=random.Random(rng.randint(0, 2**32 - 1)),
            half_life=half_life,
            half_life_mode=half_life_mode,
            draw_dates=draw_dates,
        )
        results.append(result)

    return results


def format_holdout_report(results: list[HoldoutResult]) -> str:
    """Format holdout evaluation results as a text report."""
    if not results:
        return "No results to report."

    lines = []
    lines.append("=" * 65)
    lines.append("HOLDOUT EVALUATION REPORT")
    lines.append("=" * 65)
    lines.append(f"Game: {results[0].game_name}")
    lines.append(f"Holdout size: {results[0].holdout_size} draws")
    lines.append("")

    lines.append(f"{'Strategy':<20} {'Wins':>6} {'Win Rate':>10} {'Matches':>25}")
    lines.append("-" * 65)

    for r in sorted(results, key=lambda x: x.win_rate, reverse=True):
        match_str = ", ".join(f"{k}:{v}" for k, v in sorted(r.matches_distribution.items()))
        lines.append(
            f"{r.strategy_name:<20} {r.wins:>6} {r.win_rate:>10.2%} {match_str:>25}"
        )

    lines.append("")
    lines.append("=" * 65)
    return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_evaluate_holdout.py -v`
Expected: All tests PASS

**Step 5: Write the CLI script**

Create `scripts/evaluate_holdout.py`:

```python
"""Evaluate all strategies on holdout data for each game.

Usage:
    PYTHONPATH=src python scripts/evaluate_holdout.py
    PYTHONPATH=src python scripts/evaluate_holdout.py --holdout-size 50
    PYTHONPATH=src python scripts/evaluate_holdout.py --game joker
"""

import argparse
import random
from pathlib import Path

from shared.game_config import JOKER_CONFIG, LOTO_649_CONFIG, LOTO_540_CONFIG
from shared.holdout import split_holdout
from shared.holdout_evaluator import evaluate_all_strategies, format_holdout_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate strategies on holdout data")
    parser.add_argument(
        "--holdout-size", type=int, default=20,
        help="Number of most recent draws to hold out (default: 20)",
    )
    parser.add_argument(
        "--game", choices=["joker", "loto649", "loto540", "all"], default="all",
        help="Which game to evaluate (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    games = {
        "joker": (JOKER_CONFIG, Path("data/clean/joker_draws.csv")),
        "loto649": (LOTO_649_CONFIG, Path("data/clean/loto_649_draws.csv")),
        "loto540": (LOTO_540_CONFIG, Path("data/clean/loto_540_draws.csv")),
    }

    if args.game != "all":
        games = {args.game: games[args.game]}

    for game_name, (config, csv_path) in games.items():
        if not csv_path.exists():
            print(f"Skipping {game_name}: {csv_path} not found")
            continue

        draws = _load_main_numbers(csv_path, config)
        if len(draws) < args.holdout_size + 30:
            print(f"Skipping {game_name}: insufficient data ({len(draws)} draws)")
            continue

        split = split_holdout(draws, holdout_size=args.holdout_size)

        rng = random.Random(args.seed)
        results = evaluate_all_strategies(
            config=config,
            train_draws=split.train,
            holdout_draws=split.holdout,
            rng=rng,
        )

        print(format_holdout_report(results))
        print()


def _load_main_numbers(csv_path: Path, config) -> list[list[int]]:
    """Load draws from CSV, extracting only main numbers."""
    import csv

    draws = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            numbers = []
            for i in range(1, config.numbers_drawn + 1):
                key = f"n{i}"
                if key in row:
                    numbers.append(int(row[key]))
            if len(numbers) == config.numbers_drawn:
                draws.append(sorted(numbers))
    return draws


if __name__ == "__main__":
    main()
```

**Step 6: Commit**

```bash
git add src/shared/holdout_evaluator.py tests/test_evaluate_holdout.py scripts/evaluate_holdout.py
git commit -m "feat: add holdout evaluation module and CLI script"
```

---

### Task 5: End-to-End Verification and PR

**Files:**
- No new files

**Step 1: Run full test suite**

Run: `PYTHONPATH=src python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (0 failures)

**Step 2: Verify pick generation still works**

Run all three games:
```bash
PYTHONPATH=src python scripts/generate_joker_picks.py --seed 42
PYTHONPATH=src python scripts/generate_loto_649_picks.py --seed 42
PYTHONPATH=src python scripts/generate_loto_540_picks.py --seed 42
```
Expected: Each prints valid formatted picks

**Step 3: Push and create PR**

```bash
git push -u origin feature/phase4-rl-integration
gh pr create --base feature/phase3-bias-coverage \
  --title "feat: Phase 4 — RL agent, significance gating, holdout evaluation" \
  --body "$(cat <<'EOF'
## Summary
- Add REINFORCE policy gradient RL agent as new ensemble strategy
- Add significance-gated strategy admission to ensemble blend
- Add holdout evaluation module and CLI script for cross-game analysis

## New Modules
- `src/shared/rl_agent.py` — REINFORCE policy gradient agent
- `src/shared/holdout_evaluator.py` — Holdout evaluation across strategies
- `scripts/evaluate_holdout.py` — CLI for holdout analysis

## Test plan
- [ ] RL agent tests pass
- [ ] Ensemble integration tests pass (with RL)
- [ ] Significance gate tests pass
- [ ] Holdout evaluator tests pass
- [ ] Full suite passes with 0 failures
- [ ] End-to-end pick generation verified for all games
EOF
)"
```
