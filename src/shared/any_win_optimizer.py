"""
Any-Win Optimizer for Lottery Number Selection.

This module implements strategies optimized for maximizing the probability
of winning ANY prize (not just jackpot). Key insights:

1. Most winning combinations have sums in a specific range (115-185 for 6/49)
2. Balanced odd/even and high/low ratios appear more frequently
3. Consecutive numbers are less common in winning draws
4. Certain number patterns appear more frequently in prize-winning tickets

The goal is NOT to beat the house edge (impossible) but to maximize
the entertainment value by hitting small prizes more frequently.
"""

from dataclasses import dataclass
from typing import Optional
from collections import Counter
import random
import math


@dataclass
class OptimizationConfig:
    """Configuration for any-win optimization."""
    pool_size: int = 49
    draw_size: int = 6

    # Sum constraints (based on historical analysis)
    min_sum: int = 115  # ~25th percentile for 6/49
    max_sum: int = 185  # ~75th percentile for 6/49

    # Balance constraints
    min_odd: int = 2
    max_odd: int = 4
    min_low: int = 2  # Numbers below midpoint
    max_low: int = 4

    # Pattern constraints
    max_consecutive: int = 2  # Maximum consecutive number pairs
    min_gap: int = 3  # Minimum average gap between numbers

    # Decade spread (ensure numbers from different ranges)
    require_decade_spread: bool = True
    min_decades: int = 4  # For 6/49: decades 1-10, 11-20, 21-30, 31-40, 41-49


@dataclass
class OptimizedPick:
    """An optimized lottery pick with metadata."""
    numbers: list[int]
    sum_value: int
    odd_count: int
    low_count: int
    consecutive_pairs: int
    decade_spread: int
    score: float  # Optimization score (higher = better)


class AnyWinOptimizer:
    """
    Optimizer for maximizing any-prize probability.

    Uses historical patterns and mathematical constraints to generate
    picks that are more likely to hit small prizes.

    Example:
        optimizer = AnyWinOptimizer(pool_size=49, draw_size=6)
        optimizer.analyze_history(historical_draws)
        picks = optimizer.generate_optimized_picks(5)
        for pick in picks:
            print(f"{pick.numbers} (score: {pick.score:.2f})")
    """

    def __init__(self, pool_size: int = 49, draw_size: int = 6,
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize the optimizer.

        Args:
            pool_size: Total numbers in the pool
            draw_size: Numbers per draw
            config: Optional custom configuration
        """
        self.pool_size = pool_size
        self.draw_size = draw_size
        self.config = config or self._default_config()

        # Historical analysis results
        self.frequency_weights: dict[int, float] = {}
        self.pair_weights: dict[tuple[int, int], float] = {}
        self.sum_distribution: dict[int, float] = {}
        self.position_weights: list[dict[int, float]] = []

    def _default_config(self) -> OptimizationConfig:
        """Create default config based on pool/draw size."""
        # Calculate expected sum range
        min_possible = sum(range(1, self.draw_size + 1))
        max_possible = sum(range(self.pool_size - self.draw_size + 1, self.pool_size + 1))
        expected_sum = self.draw_size * (self.pool_size + 1) / 2
        std_sum = math.sqrt(self.draw_size * (self.pool_size + 1) * (self.pool_size - self.draw_size) / 12)

        return OptimizationConfig(
            pool_size=self.pool_size,
            draw_size=self.draw_size,
            min_sum=int(expected_sum - std_sum),
            max_sum=int(expected_sum + std_sum),
            min_odd=self.draw_size // 2 - 1,
            max_odd=self.draw_size // 2 + 1,
            min_low=self.draw_size // 2 - 1,
            max_low=self.draw_size // 2 + 1,
        )

    def analyze_history(self, draws: list[list[int]],
                       prize_draws: Optional[list[list[int]]] = None) -> dict:
        """
        Analyze historical draws to learn patterns.

        Args:
            draws: All historical winning draws
            prize_draws: Optional subset of draws that won prizes (for weighting)

        Returns:
            Analysis summary dictionary
        """
        if not draws:
            return {"error": "No draws provided"}

        # Frequency analysis
        freq = Counter()
        for draw in draws:
            for num in draw:
                freq[num] += 1

        total = sum(freq.values())
        self.frequency_weights = {num: count / total for num, count in freq.items()}

        # Pair co-occurrence analysis
        pair_freq = Counter()
        for draw in draws:
            sorted_draw = sorted(draw)
            for i in range(len(sorted_draw)):
                for j in range(i + 1, len(sorted_draw)):
                    pair_freq[(sorted_draw[i], sorted_draw[j])] += 1

        pair_total = sum(pair_freq.values())
        self.pair_weights = {pair: count / pair_total for pair, count in pair_freq.items()}

        # Sum distribution
        sums = [sum(draw) for draw in draws]
        sum_freq = Counter(sums)
        sum_total = len(sums)
        self.sum_distribution = {s: count / sum_total for s, count in sum_freq.items()}

        # Position frequency (for sorted draws)
        self.position_weights = [{} for _ in range(self.draw_size)]
        for draw in draws:
            sorted_draw = sorted(draw)
            for pos, num in enumerate(sorted_draw):
                if num not in self.position_weights[pos]:
                    self.position_weights[pos][num] = 0
                self.position_weights[pos][num] += 1

        # Normalize position weights
        for pos in range(self.draw_size):
            total = sum(self.position_weights[pos].values())
            if total > 0:
                self.position_weights[pos] = {
                    num: count / total
                    for num, count in self.position_weights[pos].items()
                }

        # Calculate optimal ranges
        sorted_sums = sorted(sums)
        n = len(sorted_sums)
        self.config.min_sum = sorted_sums[int(n * 0.25)]
        self.config.max_sum = sorted_sums[int(n * 0.75)]

        return {
            "draws_analyzed": len(draws),
            "most_frequent": freq.most_common(10),
            "least_frequent": freq.most_common()[-10:],
            "sum_range": (min(sums), max(sums)),
            "optimal_sum_range": (self.config.min_sum, self.config.max_sum),
            "mean_sum": sum(sums) / len(sums)
        }

    def score_pick(self, numbers: list[int]) -> OptimizedPick:
        """
        Score a pick based on optimization criteria.

        Args:
            numbers: List of lottery numbers

        Returns:
            OptimizedPick with score and metadata
        """
        sorted_nums = sorted(numbers)

        # Calculate metrics
        sum_value = sum(sorted_nums)
        odd_count = sum(1 for n in sorted_nums if n % 2 == 1)
        midpoint = self.pool_size // 2
        low_count = sum(1 for n in sorted_nums if n <= midpoint)

        # Count consecutive pairs
        consecutive = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i + 1] - sorted_nums[i] == 1:
                consecutive += 1

        # Decade spread
        decades = set()
        for n in sorted_nums:
            decades.add((n - 1) // 10)
        decade_spread = len(decades)

        # Calculate score
        score = 100.0

        # Sum constraint (most important)
        if self.config.min_sum <= sum_value <= self.config.max_sum:
            score += 20
        else:
            distance = min(abs(sum_value - self.config.min_sum),
                          abs(sum_value - self.config.max_sum))
            score -= distance * 0.5

        # Odd/even balance
        if self.config.min_odd <= odd_count <= self.config.max_odd:
            score += 15
        else:
            score -= 10

        # High/low balance
        if self.config.min_low <= low_count <= self.config.max_low:
            score += 15
        else:
            score -= 10

        # Consecutive penalty
        if consecutive > self.config.max_consecutive:
            score -= (consecutive - self.config.max_consecutive) * 10

        # Decade spread bonus
        if decade_spread >= self.config.min_decades:
            score += 10
        else:
            score -= (self.config.min_decades - decade_spread) * 5

        # Frequency weighting (slight bonus for historically frequent numbers)
        if self.frequency_weights:
            freq_score = sum(self.frequency_weights.get(n, 0) for n in sorted_nums)
            expected_freq = self.draw_size / self.pool_size
            score += (freq_score - expected_freq * self.draw_size) * 50

        # Pair co-occurrence bonus
        if self.pair_weights:
            pair_score = 0
            for i in range(len(sorted_nums)):
                for j in range(i + 1, len(sorted_nums)):
                    pair_score += self.pair_weights.get((sorted_nums[i], sorted_nums[j]), 0)
            score += pair_score * 10

        return OptimizedPick(
            numbers=sorted_nums,
            sum_value=sum_value,
            odd_count=odd_count,
            low_count=low_count,
            consecutive_pairs=consecutive,
            decade_spread=decade_spread,
            score=score
        )

    def generate_optimized_picks(self, count: int = 5,
                                 rng: Optional[random.Random] = None,
                                 attempts_per_pick: int = 1000) -> list[OptimizedPick]:
        """
        Generate optimized lottery picks.

        Uses rejection sampling with scoring to find high-quality picks.

        Args:
            count: Number of picks to generate
            rng: Random number generator (for reproducibility)
            attempts_per_pick: Maximum attempts to find good pick

        Returns:
            List of OptimizedPick objects sorted by score
        """
        if rng is None:
            rng = random.Random()

        picks = []
        seen = set()

        for _ in range(count):
            best_pick = None
            best_score = float('-inf')

            for _ in range(attempts_per_pick):
                # Generate candidate
                candidate = self._generate_candidate(rng)
                candidate_tuple = tuple(candidate)

                if candidate_tuple in seen:
                    continue

                pick = self.score_pick(candidate)

                if pick.score > best_score:
                    best_score = pick.score
                    best_pick = pick

                # Early exit if we find an excellent pick
                if pick.score > 150:
                    break

            if best_pick:
                picks.append(best_pick)
                seen.add(tuple(best_pick.numbers))

        return sorted(picks, key=lambda p: -p.score)

    def _generate_candidate(self, rng: random.Random) -> list[int]:
        """Generate a candidate pick using weighted sampling."""
        if self.frequency_weights:
            # Use frequency-weighted sampling
            numbers = list(range(1, self.pool_size + 1))
            weights = [self.frequency_weights.get(n, 1/self.pool_size) for n in numbers]

            # Add small noise to avoid always picking same numbers
            weights = [w + rng.uniform(0, 0.01) for w in weights]

            selected = []
            available = list(zip(numbers, weights))

            for _ in range(self.draw_size):
                total = sum(w for _, w in available)
                r = rng.uniform(0, total)
                cumulative = 0
                for i, (num, weight) in enumerate(available):
                    cumulative += weight
                    if r <= cumulative:
                        selected.append(num)
                        available.pop(i)
                        break

            return sorted(selected)
        else:
            # Pure random
            return sorted(rng.sample(range(1, self.pool_size + 1), self.draw_size))

    def generate_coverage_picks(self, count: int = 10,
                                rng: Optional[random.Random] = None) -> list[OptimizedPick]:
        """
        Generate picks optimized for coverage across the number space.

        Ensures picks are diverse and cover different regions of the pool.

        Args:
            count: Number of picks to generate
            rng: Random number generator

        Returns:
            List of diverse, high-coverage picks
        """
        if rng is None:
            rng = random.Random()

        picks = []
        used_numbers = Counter()

        for _ in range(count):
            # Generate candidates that favor less-used numbers
            best_pick = None
            best_combined_score = float('-inf')

            for _ in range(500):
                candidate = self._generate_coverage_candidate(rng, used_numbers)
                pick = self.score_pick(candidate)

                # Bonus for using less-frequent numbers in our picks
                coverage_bonus = sum(1 / (used_numbers[n] + 1) for n in candidate)
                combined_score = pick.score + coverage_bonus * 10

                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_pick = pick

            if best_pick:
                picks.append(best_pick)
                for n in best_pick.numbers:
                    used_numbers[n] += 1

        return picks

    def _generate_coverage_candidate(self, rng: random.Random,
                                     used: Counter) -> list[int]:
        """Generate candidate favoring unused numbers."""
        numbers = list(range(1, self.pool_size + 1))

        # Weight by inverse of usage
        weights = [1 / (used[n] + 1) for n in numbers]

        # Also factor in historical frequency
        if self.frequency_weights:
            weights = [w * (1 + self.frequency_weights.get(n, 0)) for n, w in zip(numbers, weights)]

        selected = []
        available = list(zip(numbers, weights))

        for _ in range(self.draw_size):
            total = sum(w for _, w in available)
            r = rng.uniform(0, total)
            cumulative = 0
            for i, (num, weight) in enumerate(available):
                cumulative += weight
                if r <= cumulative:
                    selected.append(num)
                    available.pop(i)
                    break

        return sorted(selected)


def create_optimizer_for_game(game: str) -> AnyWinOptimizer:
    """
    Create optimizer configured for a specific Romanian lottery game.

    Args:
        game: One of "loto_649", "loto_540", "joker"

    Returns:
        Configured AnyWinOptimizer
    """
    configs = {
        "loto_649": OptimizationConfig(
            pool_size=49,
            draw_size=6,
            min_sum=115,
            max_sum=185,
            min_odd=2,
            max_odd=4,
            min_low=2,
            max_low=4,
            min_decades=4
        ),
        "loto_540": OptimizationConfig(
            pool_size=40,
            draw_size=5,
            min_sum=85,
            max_sum=135,
            min_odd=2,
            max_odd=3,
            min_low=2,
            max_low=3,
            min_decades=3
        ),
        "joker": OptimizationConfig(
            pool_size=45,
            draw_size=5,
            min_sum=95,
            max_sum=145,
            min_odd=2,
            max_odd=3,
            min_low=2,
            max_low=3,
            min_decades=4
        )
    }

    config = configs.get(game, configs["loto_649"])
    return AnyWinOptimizer(config.pool_size, config.draw_size, config)


def quick_optimize(draws: list[list[int]], game: str = "loto_649",
                   num_picks: int = 10) -> list[OptimizedPick]:
    """
    Quick optimization helper function.

    Args:
        draws: Historical draws for analysis
        game: Game type
        num_picks: Number of picks to generate

    Returns:
        List of optimized picks
    """
    optimizer = create_optimizer_for_game(game)
    optimizer.analyze_history(draws)
    return optimizer.generate_optimized_picks(num_picks)
