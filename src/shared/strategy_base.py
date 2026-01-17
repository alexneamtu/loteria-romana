"""Base classes and protocols for lottery prediction strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import random


@runtime_checkable
class Strategy(Protocol):
    """Protocol for lottery number generation strategies.

    All strategies must implement:
    - name: A string identifier for the strategy
    - generate(): Generate lottery lines from historical draws
    - get_probabilities(): Get probability distribution over numbers
    """

    name: str

    def generate(
        self, draws: list[tuple[list[int], int]], count: int, rng: random.Random
    ) -> list[tuple[list[int], int]]:
        """Generate lottery lines.

        Args:
            draws: Historical draws as list of (main_numbers, secondary_number) tuples
            count: Number of lines to generate
            rng: Random number generator for reproducibility

        Returns:
            List of (main_numbers, secondary_number) tuples
        """
        ...

    def get_probabilities(
        self, draws: list[tuple[list[int], int]]
    ) -> list[float]:
        """Get probability distribution over main numbers.

        Args:
            draws: Historical draws

        Returns:
            List of probabilities for each number (1 to number_pool)
        """
        ...


class BaseStrategy(ABC):
    """Abstract base class for strategies with common functionality."""

    def __init__(self, number_pool: int, numbers_to_pick: int, secondary_pool: int):
        """Initialize strategy.

        Args:
            number_pool: Range of main numbers (e.g., 45 for Joker)
            numbers_to_pick: How many main numbers per line (e.g., 5 for Joker)
            secondary_pool: Range of secondary number (e.g., 20 for Joker bonus)
        """
        self.number_pool = number_pool
        self.numbers_to_pick = numbers_to_pick
        self.secondary_pool = secondary_pool

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        ...

    @abstractmethod
    def generate(
        self, draws: list[tuple[list[int], int]], count: int, rng: random.Random
    ) -> list[tuple[list[int], int]]:
        """Generate lottery lines."""
        ...

    @abstractmethod
    def get_probabilities(
        self, draws: list[tuple[list[int], int]]
    ) -> list[float]:
        """Get probability distribution over main numbers."""
        ...

    def _generate_secondary(self, rng: random.Random) -> int:
        """Generate secondary number (bonus/joker/noroc).

        Override this method for strategies that model the secondary number.
        Default implementation uses uniform random.
        """
        return rng.randint(1, self.secondary_pool)

    def _sample_weighted_without_replacement(
        self,
        numbers: list[int],
        weights: list[float],
        count: int,
        rng: random.Random,
    ) -> list[int]:
        """Sample numbers weighted by probabilities without replacement.

        Args:
            numbers: Available numbers to sample from
            weights: Weights for each number (higher = more likely)
            count: How many numbers to sample
            rng: Random number generator

        Returns:
            Sorted list of sampled numbers
        """
        chosen = []
        pool = list(numbers)
        pool_weights = list(weights)

        for _ in range(count):
            if not pool:
                break
            pick = rng.choices(pool, weights=pool_weights, k=1)[0]
            idx = pool.index(pick)
            chosen.append(pick)
            pool.pop(idx)
            pool_weights.pop(idx)

        return sorted(chosen)

    def _ensure_unique_lines(
        self,
        generator_fn,
        count: int,
        rng: random.Random,
        max_attempts: int = 1000,
    ) -> list[tuple[list[int], int]]:
        """Generate unique lines using a generator function.

        Args:
            generator_fn: Function that generates a single line tuple
            count: Number of unique lines to generate
            rng: Random number generator
            max_attempts: Maximum attempts before giving up

        Returns:
            List of unique lines
        """
        lines = []
        seen = set()
        attempts = 0

        while len(lines) < count and attempts < max_attempts:
            main, secondary = generator_fn(rng)
            key = tuple(main) + (secondary,)
            if key not in seen:
                seen.add(key)
                lines.append((main, secondary))
            attempts += 1

        return lines


@dataclass
class StrategyResult:
    """Result from a strategy for ensemble combination."""

    name: str
    lines: list[tuple[list[int], int]]
    probabilities: list[float]
    confidence: float = 1.0
    backtest_score: float = 0.0
