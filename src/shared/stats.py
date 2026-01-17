"""Statistical analysis strategies for lottery prediction."""

import math
import random
from itertools import combinations
from dataclasses import dataclass

from .math_utils import mean, std_dev
from .recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE, draw_weights


def compute_deltas(sorted_numbers: list[int]) -> list[int]:
    """Compute differences between consecutive sorted numbers.

    Args:
        sorted_numbers: Numbers in ascending order

    Returns:
        List of deltas (differences between adjacent numbers)
    """
    return [
        sorted_numbers[i + 1] - sorted_numbers[i]
        for i in range(len(sorted_numbers) - 1)
    ]


def build_delta_distribution(
    draws: list[list[int]],
    weights: list[float] | None = None,
) -> dict[int, float]:
    """Build frequency distribution of deltas from historical draws.

    Args:
        draws: List of main_numbers lists

    Returns:
        Dict mapping delta values to their occurrence counts
    """
    delta_counts: dict[int, float] = {}
    if weights is None:
        weights = [1.0] * len(draws)
    for main, weight in zip(draws, weights):
        sorted_main = sorted(main)
        for delta in compute_deltas(sorted_main):
            delta_counts[delta] = delta_counts.get(delta, 0.0) + weight
    return delta_counts


class DeltaStrategy:
    """Generate numbers using delta pattern analysis.

    Research shows ~60% of deltas are ≤6, and ~15% are exactly 1.
    This strategy samples deltas from historical distribution.
    """

    def __init__(
        self,
        number_pool: int,
        numbers_to_pick: int,
        secondary_pool: int = 20,
        half_life: float = DEFAULT_HALF_LIFE,
        half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    ):
        self.number_pool = number_pool
        self.numbers_to_pick = numbers_to_pick
        self.secondary_pool = secondary_pool
        self.half_life = half_life
        self.half_life_mode = half_life_mode
        self.name = "delta"

    def get_probabilities(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[float]:
        """Get probability distribution based on delta analysis."""
        if not draws:
            return [1.0 / self.number_pool] * self.number_pool

        # Convert to number probabilities based on delta patterns
        mode = half_life_mode or self.half_life_mode
        weights = draw_weights(
            len(draws),
            self.half_life,
            draw_dates=draw_dates,
            mode=mode,
        )
        probs = [0.0] * self.number_pool
        for n in range(self.number_pool):
            # Score based on how often this number appears at valid delta positions
            score = 1.0
            for main, weight in zip(draws, weights):
                if n + 1 in main:
                    score += weight
            probs[n] = score

        total_prob = sum(probs)
        return [p / total_prob for p in probs]

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        """Generate lines using sampled deltas from historical distribution."""
        mode = half_life_mode or self.half_life_mode
        weights = draw_weights(
            len(draws),
            self.half_life,
            draw_dates=draw_dates,
            mode=mode,
        )
        delta_dist = build_delta_distribution(draws, weights=weights)

        if not delta_dist:
            # Fallback to uniform deltas if no history
            delta_dist = {i: 1 for i in range(1, 10)}

        total_deltas = sum(delta_dist.values())
        delta_probs = {d: c / total_deltas for d, c in delta_dist.items()}

        lines = []
        max_attempts = count * 200
        attempts = 0
        seen = set()

        while len(lines) < count and attempts < max_attempts:
            line = self._generate_from_deltas(delta_probs, rng)
            if line:
                key = tuple(line)
                if key not in seen:
                    seen.add(key)
                    lines.append(line)
            attempts += 1

        return lines

    def _generate_from_deltas(
        self, delta_probs: dict[int, float], rng: random.Random
    ) -> list[int] | None:
        """Generate a single line using sampled deltas."""
        deltas = sorted(delta_probs.keys())
        weights = [delta_probs.get(d, 0) + 0.01 for d in deltas]

        # Sample deltas for the line
        chosen_deltas = []
        for _ in range(self.numbers_to_pick - 1):
            d = rng.choices(deltas, weights=weights, k=1)[0]
            chosen_deltas.append(d)

        # Calculate valid starting range
        total_delta = sum(chosen_deltas)
        max_start = self.number_pool - total_delta
        if max_start < 1:
            return None

        # Generate starting number and build line
        start = rng.randint(1, max_start)
        numbers = [start]
        for delta in chosen_deltas:
            numbers.append(numbers[-1] + delta)

        # Validate all numbers are in range
        if any(n > self.number_pool or n < 1 for n in numbers):
            return None

        return sorted(numbers)


class HotColdStrategy:
    """Track hot/cold numbers with time-weighted frequency.

    Hot numbers: appeared more frequently than average recently
    Cold numbers: appeared less frequently than average recently
    Research suggests mixing hot, cold, and neutral performs best.
    """

    def __init__(
        self,
        number_pool: int,
        numbers_to_pick: int,
        secondary_pool: int = 20,
        half_life: float = DEFAULT_HALF_LIFE,
        half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    ):
        self.number_pool = number_pool
        self.numbers_to_pick = numbers_to_pick
        self.secondary_pool = secondary_pool
        self.half_life = half_life
        self.half_life_mode = half_life_mode
        self.name = "hotcold"

    def compute_heat_scores(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> dict[int, float]:
        """Compute time-weighted frequency scores.

        More recent draws have higher weight (exponential decay).
        """
        scores = {n: 0.0 for n in range(1, self.number_pool + 1)}
        mode = half_life_mode or self.half_life_mode
        weights = draw_weights(
            len(draws),
            self.half_life,
            draw_dates=draw_dates,
            mode=mode,
        )
        for main, weight in zip(draws, weights):
            for n in main:
                scores[n] += weight

        return scores

    def classify_numbers(
        self, scores: dict[int, float]
    ) -> tuple[list[int], list[int], list[int]]:
        """Classify numbers as hot, cold, or neutral.

        Hot: score > mean + std
        Cold: score < mean - std
        Neutral: everything else
        """
        values = list(scores.values())
        if not values:
            return [], [], list(range(1, self.number_pool + 1))

        m = mean(values)
        s = std_dev(values) if len(values) > 1 else m * 0.1

        hot = [n for n, score in scores.items() if score > m + s]
        cold = [n for n, score in scores.items() if score < m - s]
        neutral = [n for n, score in scores.items() if m - s <= score <= m + s]

        return hot, cold, neutral

    def get_probabilities(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[float]:
        """Get probability distribution based on heat scores."""
        scores = self.compute_heat_scores(
            draws,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
        total = sum(scores.values()) or 1.0
        return [scores.get(n, 0) / total for n in range(1, self.number_pool + 1)]

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        hot_ratio: float = 0.4,
        cold_ratio: float = 0.2,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        """Generate lines with specified hot/cold/neutral mix."""
        scores = self.compute_heat_scores(
            draws,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
        hot, cold, neutral = self.classify_numbers(scores)

        # Calculate target counts
        hot_count = max(1, int(self.numbers_to_pick * hot_ratio))
        cold_count = max(1, int(self.numbers_to_pick * cold_ratio))
        neutral_count = self.numbers_to_pick - hot_count - cold_count

        lines = []
        seen = set()

        for _ in range(count * 10):  # Allow some retries
            if len(lines) >= count:
                break

            line = []

            # Pick from hot numbers
            available_hot = [n for n in hot if n not in line]
            if available_hot and hot_count > 0:
                line.extend(rng.sample(available_hot, min(hot_count, len(available_hot))))

            # Pick from cold numbers
            available_cold = [n for n in cold if n not in line]
            if available_cold and cold_count > 0:
                line.extend(rng.sample(available_cold, min(cold_count, len(available_cold))))

            # Fill remaining from neutral (or any available)
            remaining = self.numbers_to_pick - len(line)
            if remaining > 0:
                available = [
                    n
                    for n in range(1, self.number_pool + 1)
                    if n not in line
                ]
                if available:
                    line.extend(rng.sample(available, min(remaining, len(available))))

            if len(line) == self.numbers_to_pick:
                key = tuple(sorted(line))
                if key not in seen:
                    seen.add(key)
                    lines.append(sorted(line))

        return lines[:count]


class PairStrategy:
    """Analyze and use number pair co-occurrence patterns.

    Some number pairs appear together more frequently than expected.
    This strategy favors historically strong pairs.
    """

    def __init__(
        self,
        number_pool: int,
        numbers_to_pick: int,
        secondary_pool: int = 20,
        half_life: float = DEFAULT_HALF_LIFE,
        half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    ):
        self.number_pool = number_pool
        self.numbers_to_pick = numbers_to_pick
        self.secondary_pool = secondary_pool
        self.half_life = half_life
        self.half_life_mode = half_life_mode
        self.name = "pairs"

    def build_pair_matrix(
        self,
        draws: list[list[int]],
        weights: list[float] | None = None,
    ) -> dict[tuple[int, int], float]:
        """Build pair co-occurrence counts."""
        pair_counts: dict[tuple[int, int], float] = {}
        if weights is None:
            weights = [1.0] * len(draws)
        for main, weight in zip(draws, weights):
            for pair in combinations(sorted(main), 2):
                pair_counts[pair] = pair_counts.get(pair, 0.0) + weight
        return pair_counts

    def get_top_pairs(
        self, pair_counts: dict[tuple[int, int], int], top_n: int = 20
    ) -> list[tuple[int, int]]:
        """Get most frequently occurring pairs."""
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        return [pair for pair, _ in sorted_pairs[:top_n]]

    def get_probabilities(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[float]:
        """Get probability distribution based on pair frequency."""
        mode = half_life_mode or self.half_life_mode
        weights = draw_weights(
            len(draws),
            self.half_life,
            draw_dates=draw_dates,
            mode=mode,
        )
        pair_counts = self.build_pair_matrix(draws, weights=weights)

        # Score each number by how often it appears in strong pairs
        scores = {n: 1.0 for n in range(1, self.number_pool + 1)}
        for (a, b), count in pair_counts.items():
            scores[a] += count
            scores[b] += count

        total = sum(scores.values())
        return [scores.get(n, 0) / total for n in range(1, self.number_pool + 1)]

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        """Generate lines favoring strong pairs."""
        mode = half_life_mode or self.half_life_mode
        weights = draw_weights(
            len(draws),
            self.half_life,
            draw_dates=draw_dates,
            mode=mode,
        )
        pair_counts = self.build_pair_matrix(draws, weights=weights)

        if not pair_counts:
            # Fallback: generate random pairs
            pair_counts = {
                (i, j): 1
                for i in range(1, self.number_pool + 1)
                for j in range(i + 1, self.number_pool + 1)
            }

        lines = []
        seen = set()

        for _ in range(count * 20):
            if len(lines) >= count:
                break

            # Start with a strong pair
            pairs = list(pair_counts.keys())
            weights = [pair_counts[p] + 1 for p in pairs]
            chosen_pair = rng.choices(pairs, weights=weights, k=1)[0]

            line = list(chosen_pair)

            # Add numbers that pair well with existing
            while len(line) < self.numbers_to_pick:
                candidates = [
                    n for n in range(1, self.number_pool + 1) if n not in line
                ]
                if not candidates:
                    break

                candidate_scores = []
                for c in candidates:
                    score = sum(
                        pair_counts.get(tuple(sorted([c, existing])), 0)
                        for existing in line
                    )
                    candidate_scores.append(score + 1)

                next_num = rng.choices(candidates, weights=candidate_scores, k=1)[0]
                line.append(next_num)

            if len(line) == self.numbers_to_pick:
                key = tuple(sorted(line))
                if key not in seen:
                    seen.add(key)
                    lines.append(sorted(line))

        return lines[:count]


class SkipGapStrategy:
    """Track gaps (skips) since each number last appeared.

    Numbers that haven't appeared for longer than their average gap
    are considered "overdue" and weighted higher.
    """

    def __init__(
        self,
        number_pool: int,
        numbers_to_pick: int,
        secondary_pool: int = 20,
        half_life: float = DEFAULT_HALF_LIFE,
        half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    ):
        self.number_pool = number_pool
        self.numbers_to_pick = numbers_to_pick
        self.secondary_pool = secondary_pool
        self.half_life = half_life
        self.half_life_mode = half_life_mode
        self.name = "skip"

    def compute_gaps(self, draws: list[list[int]]) -> dict[int, int]:
        """Compute draws since each number last appeared."""
        gaps = {n: len(draws) for n in range(1, self.number_pool + 1)}

        for age, main in enumerate(reversed(draws)):
            for n in main:
                if gaps[n] == len(draws):  # First occurrence found
                    gaps[n] = age

        return gaps

    def compute_expected_gaps(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> dict[int, float]:
        """Compute average gap for each number historically."""
        gap_totals: dict[int, float] = {n: 0.0 for n in range(1, self.number_pool + 1)}
        weight_totals: dict[int, float] = {n: 0.0 for n in range(1, self.number_pool + 1)}
        last_seen: dict[int, int] = {n: -1 for n in range(1, self.number_pool + 1)}
        mode = half_life_mode or self.half_life_mode
        weights = draw_weights(
            len(draws),
            self.half_life,
            draw_dates=draw_dates,
            mode=mode,
        )

        for idx, main in enumerate(draws):
            weight = weights[idx] if idx < len(weights) else 1.0
            for n in main:
                if last_seen[n] >= 0:
                    gap = idx - last_seen[n]
                    gap_totals[n] += gap * weight
                    weight_totals[n] += weight
                last_seen[n] = idx

        # Default expected gap based on probability
        default_gap = self.number_pool / self.numbers_to_pick

        return {
            n: (gap_totals[n] / weight_totals[n] if weight_totals[n] > 0 else default_gap)
            for n in range(1, self.number_pool + 1)
        }

    def get_probabilities(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[float]:
        """Get probability distribution based on overdue scores."""
        current_gaps = self.compute_gaps(draws)
        expected_gaps = self.compute_expected_gaps(
            draws,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )

        # Score: how overdue is each number
        overdue_scores = {
            n: current_gaps[n] / (expected_gaps[n] + 0.1)
            for n in range(1, self.number_pool + 1)
        }

        total = sum(overdue_scores.values()) or 1.0
        return [
            overdue_scores.get(n, 0) / total
            for n in range(1, self.number_pool + 1)
        ]

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        """Generate lines favoring overdue numbers."""
        current_gaps = self.compute_gaps(draws)
        expected_gaps = self.compute_expected_gaps(
            draws,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )

        # Score: how overdue is each number
        overdue_scores = {
            n: current_gaps[n] / (expected_gaps[n] + 0.1)
            for n in range(1, self.number_pool + 1)
        }

        numbers = list(range(1, self.number_pool + 1))
        weights = [overdue_scores[n] + 0.1 for n in numbers]

        lines = []
        seen = set()

        for _ in range(count * 10):
            if len(lines) >= count:
                break

            line = self._sample_weighted(numbers, weights, self.numbers_to_pick, rng)
            key = tuple(line)
            if key not in seen:
                seen.add(key)
                lines.append(line)

        return lines[:count]

    def _sample_weighted(
        self,
        numbers: list[int],
        weights: list[float],
        count: int,
        rng: random.Random,
    ) -> list[int]:
        """Sample numbers weighted by scores without replacement."""
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


class SumConstraintStrategy:
    """Filter/generate based on total sum distribution.

    Historical sums follow a bell curve. In 6/49, ~71% of winners
    have sums in the range 115-185. This strategy generates lines
    with sums within a configurable range of the historical mean.
    """

    def __init__(
        self,
        number_pool: int,
        numbers_to_pick: int,
        secondary_pool: int = 20,
        half_life: float = DEFAULT_HALF_LIFE,
        half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    ):
        self.number_pool = number_pool
        self.numbers_to_pick = numbers_to_pick
        self.secondary_pool = secondary_pool
        self.half_life = half_life
        self.half_life_mode = half_life_mode
        self.name = "sum"

    def compute_sum_distribution(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> tuple[float, float]:
        """Compute mean and std of historical sums."""
        sums = [sum(main) for main in draws]

        if not sums:
            # Theoretical expected sum for uniform distribution
            expected = self.numbers_to_pick * (1 + self.number_pool) / 2
            return expected, expected * 0.15

        mode = half_life_mode or self.half_life_mode
        weights = draw_weights(
            len(sums),
            self.half_life,
            draw_dates=draw_dates,
            mode=mode,
        )
        total_weight = sum(weights)
        if total_weight <= 0:
            m = mean(sums)
            s = std_dev(sums) if len(sums) > 1 else m * 0.1
            return m, max(s, 1.0)

        m = sum(value * weight for value, weight in zip(sums, weights)) / total_weight
        variance = (
            sum(weight * (value - m) ** 2 for value, weight in zip(sums, weights))
            / total_weight
            if len(sums) > 1
            else 0.0
        )
        s = math.sqrt(variance) if variance > 0 else m * 0.1

        return m, max(s, 1.0)  # Ensure non-zero std

    def get_probabilities(self, draws: list[list[int]]) -> list[float]:
        """Get uniform probability distribution (sum constraint is applied during generation)."""
        return [1.0 / self.number_pool] * self.number_pool

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        sigma_range: float = 1.5,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        """Generate lines with sums within sigma_range of historical mean."""
        mean_sum, std_sum = self.compute_sum_distribution(
            draws,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
        min_sum = mean_sum - sigma_range * std_sum
        max_sum = mean_sum + sigma_range * std_sum

        lines = []
        seen = set()
        max_attempts = count * 500
        attempts = 0

        while len(lines) < count and attempts < max_attempts:
            # Generate random line
            line = sorted(
                rng.sample(range(1, self.number_pool + 1), self.numbers_to_pick)
            )
            line_sum = sum(line)

            if min_sum <= line_sum <= max_sum:
                key = tuple(line)
                if key not in seen:
                    seen.add(key)
                    lines.append(line)

            attempts += 1

        return lines[:count]


# Utility functions for pattern analysis

def compute_odd_even_distribution(
    draws: list[list[int]],
    weights: list[float] | None = None,
) -> dict[int, float]:
    """Count how many odd numbers per draw historically."""
    dist: dict[int, float] = {}
    if weights is None:
        weights = [1.0] * len(draws)
    for main, weight in zip(draws, weights):
        odd_count = sum(1 for n in main if n % 2 == 1)
        dist[odd_count] = dist.get(odd_count, 0.0) + weight
    return dist


def compute_high_low_distribution(
    draws: list[list[int]],
    midpoint: int,
    weights: list[float] | None = None,
) -> dict[int, float]:
    """Count how many high numbers (>= midpoint) per draw."""
    dist: dict[int, float] = {}
    if weights is None:
        weights = [1.0] * len(draws)
    for main, weight in zip(draws, weights):
        high_count = sum(1 for n in main if n >= midpoint)
        dist[high_count] = dist.get(high_count, 0.0) + weight
    return dist


def count_consecutives(sorted_numbers: list[int]) -> int:
    """Count pairs of consecutive numbers."""
    return sum(
        1
        for i in range(len(sorted_numbers) - 1)
        if sorted_numbers[i + 1] - sorted_numbers[i] == 1
    )


def compute_consecutive_distribution(
    draws: list[list[int]],
    weights: list[float] | None = None,
) -> dict[int, float]:
    """Distribution of consecutive pairs in historical draws."""
    dist: dict[int, float] = {}
    if weights is None:
        weights = [1.0] * len(draws)
    for main, weight in zip(draws, weights):
        consec_count = count_consecutives(sorted(main))
        dist[consec_count] = dist.get(consec_count, 0.0) + weight
    return dist


def build_position_frequency(
    draws: list[list[int]],
    numbers_to_pick: int,
    number_pool: int,
    weights: list[float] | None = None,
) -> list[dict[int, float]]:
    """Track which numbers appear at each sorted position.

    Args:
        draws: Historical draws
        numbers_to_pick: Numbers per draw
        number_pool: Total available numbers

    Returns:
        List of dicts, one per position, mapping number to frequency
    """
    position_counts: list[dict[int, float]] = [
        {} for _ in range(numbers_to_pick)
    ]
    if weights is None:
        weights = [1.0] * len(draws)
    for main, weight in zip(draws, weights):
        sorted_main = sorted(main)
        for pos, num in enumerate(sorted_main):
            position_counts[pos][num] = position_counts[pos].get(num, 0.0) + weight

    return position_counts


class BalanceStrategy:
    """Generate lines matching historical odd/even and high/low ratios.

    Research shows winning combinations tend to have balanced distributions:
    - Odd/Even: Most winners have 2-4 odd numbers (not all odd or all even)
    - High/Low: Most winners have 2-4 high numbers (not all high or all low)
    """

    def __init__(
        self,
        number_pool: int,
        numbers_to_pick: int,
        secondary_pool: int = 20,
        half_life: float = DEFAULT_HALF_LIFE,
        half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    ):
        self.number_pool = number_pool
        self.numbers_to_pick = numbers_to_pick
        self.secondary_pool = secondary_pool
        self.name = "balance"
        self.midpoint = (number_pool + 1) // 2
        self.half_life = half_life
        self.half_life_mode = half_life_mode

    def compute_target_ratios(
        self,
        draws: list[list[int]],
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Compute probability distributions for odd/even and high/low counts."""
        mode = half_life_mode or self.half_life_mode
        weights = draw_weights(
            len(draws),
            self.half_life,
            draw_dates=draw_dates,
            mode=mode,
        )
        odd_dist = compute_odd_even_distribution(draws, weights=weights)
        high_dist = compute_high_low_distribution(draws, self.midpoint, weights=weights)

        total_odd = sum(odd_dist.values()) or 1
        total_high = sum(high_dist.values()) or 1

        odd_probs = {k: v / total_odd for k, v in odd_dist.items()}
        high_probs = {k: v / total_high for k, v in high_dist.items()}

        return odd_probs, high_probs

    def get_probabilities(self, draws: list[list[int]]) -> list[float]:
        """Get uniform probability distribution (balance is applied during generation)."""
        return [1.0 / self.number_pool] * self.number_pool

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        draw_dates: list[str] | None = None,
        half_life_mode: str | None = None,
    ) -> list[list[int]]:
        """Generate lines matching historical balance distributions."""
        odd_probs, high_probs = self.compute_target_ratios(
            draws,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )

        if not odd_probs:
            # Default to balanced distribution
            mid = self.numbers_to_pick // 2
            odd_probs = {mid: 0.4, mid - 1: 0.3, mid + 1: 0.3}

        if not high_probs:
            mid = self.numbers_to_pick // 2
            high_probs = {mid: 0.4, mid - 1: 0.3, mid + 1: 0.3}

        odd_targets = list(odd_probs.keys())
        odd_weights = [odd_probs[k] for k in odd_targets]
        high_targets = list(high_probs.keys())
        high_weights = [high_probs[k] for k in high_targets]

        lines = []
        seen = set()
        max_attempts = count * 200

        for _ in range(max_attempts):
            if len(lines) >= count:
                break

            # Sample target odd/even and high/low counts
            target_odd = rng.choices(odd_targets, weights=odd_weights, k=1)[0]
            target_high = rng.choices(high_targets, weights=high_weights, k=1)[0]

            # Generate line matching these targets
            line = self._generate_balanced_line(target_odd, target_high, rng)
            if line:
                key = tuple(line)
                if key not in seen:
                    seen.add(key)
                    lines.append(line)

        return lines[:count]

    def _generate_balanced_line(
        self,
        target_odd: int,
        target_high: int,
        rng: random.Random,
    ) -> list[int] | None:
        """Generate a line with specified odd and high counts."""
        # Categorize numbers
        low_odd = [n for n in range(1, self.midpoint) if n % 2 == 1]
        low_even = [n for n in range(1, self.midpoint) if n % 2 == 0]
        high_odd = [n for n in range(self.midpoint, self.number_pool + 1) if n % 2 == 1]
        high_even = [n for n in range(self.midpoint, self.number_pool + 1) if n % 2 == 0]

        target_even = self.numbers_to_pick - target_odd
        target_low = self.numbers_to_pick - target_high

        # Calculate numbers needed from each category
        # high_odd + high_even = target_high
        # low_odd + low_even = target_low
        # high_odd + low_odd = target_odd
        # high_even + low_even = target_even

        line = []

        # Try different combinations
        for num_high_odd in range(min(target_odd, target_high) + 1):
            num_low_odd = target_odd - num_high_odd
            num_high_even = target_high - num_high_odd
            num_low_even = target_low - num_low_odd

            # Validate counts are non-negative and achievable
            if num_low_odd < 0 or num_high_even < 0 or num_low_even < 0:
                continue
            if len(high_odd) < num_high_odd:
                continue
            if len(low_odd) < num_low_odd:
                continue
            if len(high_even) < num_high_even:
                continue
            if len(low_even) < num_low_even:
                continue

            try:
                line = (
                    rng.sample(high_odd, num_high_odd)
                    + rng.sample(low_odd, num_low_odd)
                    + rng.sample(high_even, num_high_even)
                    + rng.sample(low_even, num_low_even)
                )
                return sorted(line)
            except ValueError:
                continue

        return None
