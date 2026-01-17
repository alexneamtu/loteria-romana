"""Ensemble voting framework for combining multiple strategies."""

import random
from typing import Protocol

from .math_utils import softmax


class Strategy(Protocol):
    """Protocol for strategy objects."""

    name: str

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        ...

    def get_probabilities(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[float]:
        ...


class EnsembleVoter:
    """Combine multiple strategies using weighted voting.

    Strategies can be weighted based on:
    - Backtest performance
    - Equal weighting
    - Manual weights
    """

    def __init__(
        self,
        strategies: list[Strategy],
        weights: dict[str, float] | None = None,
        number_pool: int = 45,
        numbers_to_pick: int = 5,
        secondary_pool: int = 20,
    ):
        """Initialize ensemble voter.

        Args:
            strategies: List of strategy objects
            weights: Optional dict mapping strategy name to weight
            number_pool: Total numbers in pool
            numbers_to_pick: Numbers to pick per line
            secondary_pool: Pool for secondary number
        """
        self.strategies = strategies
        self.number_pool = number_pool
        self.numbers_to_pick = numbers_to_pick
        self.secondary_pool = secondary_pool
        self.name = "ensemble"

        # Initialize weights
        if weights:
            self.strategy_weights = weights
        else:
            # Equal weights by default
            self.strategy_weights = {s.name: 1.0 for s in strategies}

        # Normalize weights
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(self.strategy_weights.values())
        if total > 0:
            self.strategy_weights = {
                k: v / total for k, v in self.strategy_weights.items()
            }

    def update_weights(self, backtest_results: dict[str, float]) -> None:
        """Update strategy weights based on backtest performance.

        Args:
            backtest_results: Dict mapping strategy name to performance metric
                             (higher is better, e.g., win rate or expected value)
        """
        for name, performance in backtest_results.items():
            if name in self.strategy_weights:
                # Use softmax-like weighting to prevent extreme weights
                self.strategy_weights[name] = max(0.01, performance)

        self._normalize_weights()

    def combine_probabilities(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[float]:
        """Combine probability distributions from all strategies.

        Returns weighted average of probability distributions.
        """
        combined = [0.0] * self.number_pool

        for strategy in self.strategies:
            weight = self.strategy_weights.get(strategy.name, 0.0)
            if weight <= 0:
                continue

            probs = strategy.get_probabilities(
                draws,
                draw_dates=draw_dates,
                half_life_mode=half_life_mode,
            )
            for i, p in enumerate(probs):
                combined[i] += weight * p

        # Normalize
        total = sum(combined)
        if total > 0:
            combined = [p / total for p in combined]
        else:
            combined = [1.0 / self.number_pool] * self.number_pool

        return combined

    def get_probabilities(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[float]:
        """Alias for combine_probabilities for protocol compatibility."""
        return self.combine_probabilities(
            draws,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        """Generate lines using weighted strategy voting.

        Uses multiple generation methods:
        1. Probability-weighted sampling from combined distribution
        2. Collect lines from individual strategies based on weights
        """
        lines = []
        seen = set()

        # Method 1: Generate some lines from combined probabilities
        combined_count = count // 2
        combined_probs = self.combine_probabilities(
            draws,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
        numbers = list(range(1, self.number_pool + 1))

        for _ in range(combined_count * 10):
            if len(lines) >= combined_count:
                break

            line = self._sample_weighted(numbers, combined_probs, rng)
            key = tuple(line)
            if key not in seen:
                seen.add(key)
                lines.append(line)

        # Method 2: Collect lines from individual strategies
        remaining = count - len(lines)
        if remaining > 0:
            strategy_lines = self._collect_strategy_lines(
                draws,
                remaining,
                rng,
                seen,
                draw_dates=draw_dates,
                half_life_mode=half_life_mode,
            )
            lines.extend(strategy_lines)

        return lines[:count]

    def _sample_weighted(
        self,
        numbers: list[int],
        weights: list[float],
        rng: random.Random,
    ) -> list[int]:
        """Sample numbers weighted without replacement."""
        chosen = []
        pool = list(numbers)
        pool_weights = list(weights)

        for _ in range(self.numbers_to_pick):
            if not pool:
                break

            # Normalize weights for this round
            total = sum(pool_weights)
            if total <= 0:
                pool_weights = [1.0] * len(pool)
                total = len(pool)

            probs = [w / total for w in pool_weights]
            pick = rng.choices(pool, weights=probs, k=1)[0]
            idx = pool.index(pick)

            chosen.append(pick)
            pool.pop(idx)
            pool_weights.pop(idx)

        return sorted(chosen)

    def _collect_strategy_lines(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        seen: set,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        """Collect lines from individual strategies proportional to weights."""
        lines = []

        # Calculate how many lines each strategy should contribute
        strategy_counts = {}
        remaining = count
        sorted_strategies = sorted(
            self.strategies,
            key=lambda s: self.strategy_weights.get(s.name, 0),
            reverse=True,
        )

        for i, strategy in enumerate(sorted_strategies):
            weight = self.strategy_weights.get(strategy.name, 0)
            if weight <= 0:
                continue

            # Proportional allocation
            if i == len(sorted_strategies) - 1:
                strategy_counts[strategy.name] = remaining
            else:
                n = max(1, int(count * weight))
                strategy_counts[strategy.name] = min(n, remaining)
                remaining -= strategy_counts[strategy.name]

        # Generate lines from each strategy
        for strategy in self.strategies:
            n = strategy_counts.get(strategy.name, 0)
            if n <= 0:
                continue

            strategy_lines = strategy.generate(
                draws,
                n * 2,
                rng,
                draw_dates=draw_dates,
                half_life_mode=half_life_mode,
            )
            for main in strategy_lines:
                key = tuple(main)
                if key not in seen:
                    seen.add(key)
                    lines.append(main)
                    if len(lines) >= count:
                        break

            if len(lines) >= count:
                break

        return lines

    def generate_diverse(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        """Generate diverse lines by ensuring representation from all strategies.

        Unlike regular generate(), this method guarantees at least one line
        from each strategy if possible.
        """
        lines = []
        seen = set()

        # First, get at least one line from each strategy
        for strategy in self.strategies:
            strategy_lines = strategy.generate(
                draws,
                3,
                rng,
                draw_dates=draw_dates,
                half_life_mode=half_life_mode,
            )
            for main in strategy_lines:
                key = tuple(main)
                if key not in seen:
                    seen.add(key)
                    lines.append(main)
                    break

        # Fill remaining with weighted generation
        remaining = count - len(lines)
        if remaining > 0:
            additional = self.generate(
                draws,
                remaining * 2,
                rng,
                draw_dates=draw_dates,
                half_life_mode=half_life_mode,
            )
            for main in additional:
                key = tuple(main)
                if key not in seen:
                    seen.add(key)
                    lines.append(main)
                    if len(lines) >= count:
                        break

        return lines[:count]


class StrategySelector:
    """Automatically select best strategy based on recent performance."""

    def __init__(
        self,
        strategies: list[Strategy],
        lookback: int = 50,
        number_pool: int = 45,
        numbers_to_pick: int = 5,
        secondary_pool: int = 20,
    ):
        self.strategies = strategies
        self.lookback = lookback
        self.number_pool = number_pool
        self.numbers_to_pick = numbers_to_pick
        self.secondary_pool = secondary_pool
        self.name = "selector"

        # Performance tracking
        self.recent_performance: dict[str, list[float]] = {
            s.name: [] for s in strategies
        }

    def evaluate_strategy(
        self,
        strategy: Strategy,
        draws: list[list[int]],
        n_eval: int = 20,
        rng: random.Random | None = None,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> float:
        """Evaluate strategy on recent draws.

        Returns average match rate (matches per ticket).
        """
        rng = rng or random.Random()

        if len(draws) < n_eval + 10:
            return 0.0

        total_matches = 0
        total_tickets = 0

        for i in range(len(draws) - n_eval, len(draws)):
            training = draws[:i]
            actual_main = draws[i]

            tickets = strategy.generate(
                training,
                1,
                rng,
                draw_dates=draw_dates[:i] if draw_dates else None,
                half_life_mode=half_life_mode,
            )
            for main in tickets:
                matches = len(set(main) & set(actual_main))
                total_matches += matches
                total_tickets += 1

        return total_matches / total_tickets if total_tickets > 0 else 0.0

    def select_best(
        self,
        draws: list[list[int]],
        rng: random.Random | None = None,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> Strategy:
        """Select the best performing strategy."""
        rng = rng or random.Random()

        scores = {}
        for strategy in self.strategies:
            score = self.evaluate_strategy(
                strategy,
                draws,
                rng=rng,
                draw_dates=draw_dates,
                half_life_mode=half_life_mode,
            )
            scores[strategy.name] = score
            self.recent_performance[strategy.name].append(score)

            # Keep only recent history
            if len(self.recent_performance[strategy.name]) > self.lookback:
                self.recent_performance[strategy.name].pop(0)

        best_name = max(scores, key=scores.get)
        return next(s for s in self.strategies if s.name == best_name)

    def get_probabilities(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[float]:
        """Get probabilities from the best strategy."""
        best = self.select_best(
            draws,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
        return best.get_probabilities(
            draws,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        """Generate using the best strategy."""
        best = self.select_best(
            draws,
            rng,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
        return best.generate(
            draws,
            count,
            rng,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
