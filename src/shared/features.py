"""Advanced statistical feature extraction for lottery analysis.

This module provides feature extraction functions that detect patterns
in historical lottery draws. These features can be used to improve
prediction strategies.
"""

import math
from collections import Counter
from typing import NamedTuple

from .game_config import GameConfig


class FeatureSet(NamedTuple):
    """Collection of extracted features from historical draws."""
    digit_frequency: dict[int, int]
    prime_ratio: float
    modular_residues: dict[int, dict[int, int]]
    entropy: float
    gap_distribution: dict[int, list[int]]
    consecutive_frequency: dict[int, int]
    odd_even_ratio: float
    high_low_ratio: float
    sum_statistics: dict[str, float]
    position_frequency: list[dict[int, int]]


def extract_digit_frequency(draws: list[list[int]]) -> dict[int, int]:
    """Extract frequency of individual digits (0-9) across all numbers.

    This can detect if players/machines have biases toward certain digits.

    Example: If '7' appears more often in ones place, digit_freq[7] will be high.
    """
    digit_freq = Counter()
    for main in draws:
        for num in main:
            for digit in str(num):
                digit_freq[int(digit)] += 1
    return dict(digit_freq)


def compute_prime_ratio(draws: list[list[int]]) -> float:
    """Compute the ratio of prime numbers in draws.

    Returns the proportion of drawn numbers that are prime.
    """
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    total = 0
    primes = 0
    for main in draws:
        for num in main:
            total += 1
            if is_prime(num):
                primes += 1

    return primes / total if total > 0 else 0.0


def compute_modular_residues(
    draws: list[list[int]],
    moduli: list[int] | None = None,
) -> dict[int, dict[int, int]]:
    """Compute distribution of numbers modulo various bases.

    This can reveal hidden periodicity patterns.

    Args:
        draws: Historical draws
        moduli: List of moduli to compute (default: [2, 3, 5, 7, 10])

    Returns:
        Dict mapping modulus -> {residue -> count}
    """
    if moduli is None:
        moduli = [2, 3, 5, 7, 10]

    residues = {m: Counter() for m in moduli}

    for main in draws:
        for num in main:
            for m in moduli:
                residues[m][num % m] += 1

    return {m: dict(residues[m]) for m in moduli}


def compute_entropy(draws: list[list[int]], pool_size: int) -> float:
    """Compute Shannon entropy of number frequency distribution.

    Higher entropy = more random (uniform) distribution.
    Lower entropy = some numbers appear much more than others.

    Args:
        draws: Historical draws
        pool_size: Total numbers in the pool

    Returns:
        Entropy value (0 = deterministic, log2(pool_size) = uniform)
    """
    freq = Counter()
    total = 0
    for main in draws:
        for num in main:
            freq[num] += 1
            total += 1

    if total == 0:
        return 0.0

    entropy = 0.0
    for count in freq.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def compute_gap_distribution(
    draws: list[list[int]],
    pool_size: int,
) -> dict[int, list[int]]:
    """Compute gaps (draws between appearances) for each number.

    This helps identify "overdue" numbers.

    Returns:
        Dict mapping number -> list of gaps (draws between appearances)
    """
    gaps = {n: [] for n in range(1, pool_size + 1)}
    last_seen = {n: -1 for n in range(1, pool_size + 1)}

    for idx, main in enumerate(draws):
        for num in main:
            if last_seen[num] >= 0:
                gap = idx - last_seen[num]
                gaps[num].append(gap)
            last_seen[num] = idx

    return gaps


def compute_consecutive_frequency(draws: list[list[int]]) -> dict[int, int]:
    """Count how often consecutive numbers appear together.

    Returns dict mapping count_of_consecutives -> frequency.
    E.g., {0: 50, 1: 30, 2: 15, 3: 5} means 50 draws had no consecutives.
    """
    consecutive_counts = Counter()

    for main in draws:
        sorted_nums = sorted(main)
        consecutives = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i + 1] - sorted_nums[i] == 1:
                consecutives += 1
        consecutive_counts[consecutives] += 1

    return dict(consecutive_counts)


def compute_odd_even_ratio(draws: list[list[int]]) -> float:
    """Compute ratio of odd numbers to total numbers drawn."""
    odd = 0
    total = 0
    for main in draws:
        for num in main:
            total += 1
            if num % 2 == 1:
                odd += 1
    return odd / total if total > 0 else 0.5


def compute_high_low_ratio(draws: list[list[int]], midpoint: int) -> float:
    """Compute ratio of high numbers (>= midpoint) to total.

    Args:
        draws: Historical draws
        midpoint: Number that divides high/low (typically pool_size // 2)
    """
    high = 0
    total = 0
    for main in draws:
        for num in main:
            total += 1
            if num >= midpoint:
                high += 1
    return high / total if total > 0 else 0.5


def compute_sum_statistics(draws: list[list[int]]) -> dict[str, float]:
    """Compute statistics about the sum of drawn numbers.

    Returns min, max, mean, std of sums across all draws.
    """
    sums = []
    for main in draws:
        sums.append(sum(main))

    if not sums:
        return {"min": 0, "max": 0, "mean": 0, "std": 0}

    mean = sum(sums) / len(sums)
    variance = sum((s - mean) ** 2 for s in sums) / len(sums) if len(sums) > 1 else 0
    std = math.sqrt(variance)

    return {
        "min": min(sums),
        "max": max(sums),
        "mean": mean,
        "std": std,
    }


def compute_position_frequency(
    draws: list[list[int]],
    pool_size: int,
    weights: list[float] | None = None,
) -> list[dict[int, float]]:
    """Compute frequency of each number at each position.

    Returns list of dicts, one per position, mapping number -> count.
    """
    if not draws:
        return []

    # Determine number of positions from first draw
    num_positions = len(draws[0])
    position_freq = [{n: 0.0 for n in range(1, pool_size + 1)} for _ in range(num_positions)]
    if weights is None:
        weights = [1.0] * len(draws)

    for main, weight in zip(draws, weights):
        sorted_nums = sorted(main)
        for pos, num in enumerate(sorted_nums):
            position_freq[pos][num] += weight

    return position_freq


def compute_autocorrelation(
    draws: list[list[int]],
    pool_size: int,
    lag: int = 1,
) -> float:
    """Compute autocorrelation of number frequency at given lag.

    This measures if the same numbers tend to appear in consecutive draws.

    Args:
        draws: Historical draws
        pool_size: Total numbers in pool
        lag: Number of draws to look back

    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(draws) <= lag:
        return 0.0

    # Create binary vectors for each draw (1 if number appeared)
    vectors = []
    for main in draws:
        vec = [1 if n in main else 0 for n in range(1, pool_size + 1)]
        vectors.append(vec)

    # Compute correlation between draws at given lag
    correlations = []
    for i in range(lag, len(vectors)):
        v1 = vectors[i]
        v2 = vectors[i - lag]

        # Pearson correlation
        mean1 = sum(v1) / len(v1)
        mean2 = sum(v2) / len(v2)

        num = sum((a - mean1) * (b - mean2) for a, b in zip(v1, v2))
        denom1 = math.sqrt(sum((a - mean1) ** 2 for a in v1))
        denom2 = math.sqrt(sum((b - mean2) ** 2 for b in v2))

        if denom1 > 0 and denom2 > 0:
            correlations.append(num / (denom1 * denom2))

    return sum(correlations) / len(correlations) if correlations else 0.0


def extract_all_features(
    config: GameConfig,
    draws: list[list[int]],
) -> FeatureSet:
    """Extract all available features from historical draws.

    Args:
        config: Game configuration
        draws: Historical draws as main_numbers lists

    Returns:
        FeatureSet with all extracted features
    """
    pool_size = config.pool_size
    midpoint = config.pool_min + pool_size // 2

    return FeatureSet(
        digit_frequency=extract_digit_frequency(draws),
        prime_ratio=compute_prime_ratio(draws),
        modular_residues=compute_modular_residues(draws),
        entropy=compute_entropy(draws, pool_size),
        gap_distribution=compute_gap_distribution(draws, pool_size),
        consecutive_frequency=compute_consecutive_frequency(draws),
        odd_even_ratio=compute_odd_even_ratio(draws),
        high_low_ratio=compute_high_low_ratio(draws, midpoint),
        sum_statistics=compute_sum_statistics(draws),
        position_frequency=compute_position_frequency(draws, pool_size),
    )


def get_overdue_numbers(
    config: GameConfig,
    draws: list[list[int]],
    threshold_factor: float = 1.5,
) -> list[int]:
    """Get numbers that haven't appeared for longer than expected.

    Args:
        config: Game configuration
        draws: Historical draws
        threshold_factor: Multiple of expected gap to consider "overdue"

    Returns:
        List of overdue numbers, sorted by how overdue they are
    """
    if not draws:
        return []

    pool_size = config.pool_size
    numbers_per_draw = config.numbers_drawn

    # Expected gap = pool_size / numbers_per_draw
    expected_gap = pool_size / numbers_per_draw
    threshold = expected_gap * threshold_factor

    # Find current gap for each number
    last_seen = {n: -1 for n in range(1, pool_size + 1)}
    for idx, main in enumerate(draws):
        for num in main:
            last_seen[num] = idx

    current_draw = len(draws)
    overdue = []

    for num in range(1, pool_size + 1):
        if last_seen[num] >= 0:
            gap = current_draw - last_seen[num]
        else:
            gap = current_draw  # Never seen

        if gap > threshold:
            overdue.append((num, gap))

    # Sort by gap descending (most overdue first)
    overdue.sort(key=lambda x: x[1], reverse=True)
    return [num for num, _ in overdue]


def get_hot_numbers(
    config: GameConfig,
    draws: list[list[int]],
    recent_window: int = 10,
) -> list[int]:
    """Get numbers that have appeared frequently in recent draws.

    Args:
        config: Game configuration
        draws: Historical draws
        recent_window: Number of recent draws to consider

    Returns:
        List of hot numbers, sorted by frequency (most frequent first)
    """
    if not draws:
        return []

    recent = draws[-recent_window:] if len(draws) > recent_window else draws
    freq = Counter()

    for main in recent:
        for num in main:
            freq[num] += 1

    # Sort by frequency descending
    sorted_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [num for num, _ in sorted_nums]
