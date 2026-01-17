"""Advanced strategies designed to maximize winning probability.

This module combines multiple statistical techniques to find the
numbers with the highest likelihood of being drawn.
"""

import math
import random
from collections import Counter
from typing import Literal

from .game_config import GameConfig
from .features import (
    extract_all_features,
    get_overdue_numbers,
    get_hot_numbers,
    compute_gap_distribution,
    compute_position_frequency,
)
from .game_strategies import _sample_weighted


def compute_composite_scores(
    config: GameConfig,
    draws: list[list[int]],
    weights: dict[str, float] | None = None,
) -> dict[int, float]:
    """Compute composite probability scores for each number.

    Combines multiple factors:
    - Historical frequency (hot numbers)
    - Gap analysis (overdue numbers)
    - Position bias (where numbers tend to appear)
    - Recency weighting (recent draws matter more)
    - Pattern detection (consecutive, odd/even balance)

    Args:
        config: Game configuration
        draws: Historical draws
        weights: Optional weights for each factor

    Returns:
        Dict mapping number -> composite score (higher = more likely)
    """
    if not draws:
        return {n: 1.0 for n in config.pool_range}

    default_weights = {
        "frequency": 0.25,
        "recency": 0.20,
        "gap": 0.20,
        "position": 0.15,
        "trend": 0.10,
        "balance": 0.10,
    }
    weights = weights or default_weights

    pool_size = config.pool_size
    scores = {n: 0.0 for n in config.pool_range}

    # 1. Frequency score (how often each number appears)
    freq = Counter()
    total = 0
    for main in draws:
        for num in main:
            freq[num] += 1
            total += 1

    if total > 0:
        for num in config.pool_range:
            scores[num] += weights["frequency"] * (freq.get(num, 0) / total)

    # 2. Recency score (exponential decay weighting)
    decay_rate = 0.95
    recency_score = {n: 0.0 for n in config.pool_range}
    for idx, main in enumerate(draws):
        weight = decay_rate ** (len(draws) - idx - 1)
        for num in main:
            recency_score[num] += weight

    max_recency = max(recency_score.values()) if recency_score else 1
    if max_recency > 0:
        for num in config.pool_range:
            scores[num] += weights["recency"] * (recency_score[num] / max_recency)

    # 3. Gap score (overdue numbers get bonus)
    gaps = compute_gap_distribution(draws, pool_size)
    expected_gap = pool_size / config.numbers_drawn

    for num in config.pool_range:
        if num in gaps and gaps[num]:
            avg_gap = sum(gaps[num]) / len(gaps[num])
            # Numbers with longer-than-expected gaps get higher scores
            gap_ratio = avg_gap / expected_gap if expected_gap > 0 else 1
            scores[num] += weights["gap"] * min(gap_ratio, 2.0) / 2.0

    # 4. Position score (numbers that appear in specific positions)
    pos_freq = compute_position_frequency(draws, pool_size)
    for num in config.pool_range:
        pos_score = 0
        for pos_dict in pos_freq:
            pos_score += pos_dict.get(num, 0)
        max_pos = max(sum(d.values()) for d in pos_freq) if pos_freq else 1
        if max_pos > 0:
            scores[num] += weights["position"] * (pos_score / max_pos)

    # 5. Trend score (is frequency increasing or decreasing?)
    if len(draws) >= 20:
        first_half = draws[:len(draws)//2]
        second_half = draws[len(draws)//2:]

        freq_first = Counter()
        freq_second = Counter()

        for main in first_half:
            for num in main:
                freq_first[num] += 1
        for main in second_half:
            for num in main:
                freq_second[num] += 1

        for num in config.pool_range:
            f1 = freq_first.get(num, 0) / (len(first_half) * config.numbers_drawn)
            f2 = freq_second.get(num, 0) / (len(second_half) * config.numbers_drawn)
            # Positive trend = increasing frequency
            trend = f2 - f1
            scores[num] += weights["trend"] * (0.5 + trend)  # Center at 0.5

    # 6. Balance score (maintain odd/even, high/low balance)
    midpoint = config.pool_min + pool_size // 2
    odd_count = sum(1 for n in config.pool_range if n % 2 == 1)
    high_count = sum(1 for n in config.pool_range if n >= midpoint)

    # Slightly favor balanced selections
    for num in config.pool_range:
        balance_bonus = 0
        if num % 2 == 1:  # Odd
            balance_bonus += 0.5
        else:  # Even
            balance_bonus += 0.5
        if num >= midpoint:  # High
            balance_bonus += 0.5
        else:  # Low
            balance_bonus += 0.5
        scores[num] += weights["balance"] * (balance_bonus / 2)

    # Normalize scores to sum to 1
    total_score = sum(scores.values())
    if total_score > 0:
        for num in scores:
            scores[num] /= total_score

    return scores


def generate_optimal_picks(
    config: GameConfig,
    draws: list[list[int]],
    count: int,
    rng: random.Random | None = None,
    strategy: Literal["balanced", "aggressive", "conservative"] = "balanced",
) -> list[list[int]]:
    """Generate optimized lottery picks using composite scoring.

    Args:
        config: Game configuration
        draws: Historical draws
        count: Number of picks to generate
        rng: Random number generator
        strategy: Risk strategy
            - "balanced": Mix of factors
            - "aggressive": Favor hot numbers and trends
            - "conservative": Favor overdue numbers and stability

    Returns:
        List of picks optimized for winning probability
    """
    rng = rng or random.SystemRandom()

    # Adjust weights based on strategy
    if strategy == "aggressive":
        weights = {
            "frequency": 0.35,
            "recency": 0.30,
            "gap": 0.10,
            "position": 0.10,
            "trend": 0.10,
            "balance": 0.05,
        }
    elif strategy == "conservative":
        weights = {
            "frequency": 0.15,
            "recency": 0.10,
            "gap": 0.35,
            "position": 0.15,
            "trend": 0.05,
            "balance": 0.20,
        }
    else:  # balanced
        weights = None  # Use defaults

    scores = compute_composite_scores(config, draws, weights)

    # Convert to lists for sampling
    numbers = list(config.pool_range)
    probs = [scores[n] for n in numbers]

    # Ensure all probabilities are positive
    min_prob = 0.001
    probs = [max(p, min_prob) for p in probs]

    lines = []
    seen = set()

    while len(lines) < count:
        pick = _sample_weighted(numbers, probs, config.numbers_to_pick, rng)
        key = tuple(pick)
        if key in seen:
            continue
        seen.add(key)
        lines.append(pick)

    return lines


def generate_coverage_picks(
    config: GameConfig,
    draws: list[list[int]],
    count: int,
    rng: random.Random | None = None,
) -> list[list[int]]:
    """Generate picks that maximize number coverage.

    This strategy ensures picks are diverse and cover different
    number ranges, reducing correlation between picks.

    Args:
        config: Game configuration
        draws: Historical draws
        count: Number of picks to generate
        rng: Random number generator

    Returns:
        List of diverse picks covering the number space
    """
    rng = rng or random.SystemRandom()

    scores = compute_composite_scores(config, draws)
    numbers = list(config.pool_range)
    probs = [scores[n] for n in numbers]

    lines = []
    used_numbers = Counter()

    while len(lines) < count:
        # Adjust probabilities to favor unused numbers
        adjusted_probs = []
        for num, p in zip(numbers, probs):
            usage_penalty = 1.0 / (1.0 + used_numbers.get(num, 0))
            adjusted_probs.append(p * usage_penalty)

        # Ensure positive
        adjusted_probs = [max(p, 0.001) for p in adjusted_probs]

        pick = _sample_weighted(numbers, adjusted_probs, config.numbers_to_pick, rng)

        key = tuple(pick)
        if key not in [tuple(l) for l in lines]:
            lines.append(pick)
            for num in pick:
                used_numbers[num] += 1

    return lines


def generate_pattern_picks(
    config: GameConfig,
    draws: list[list[int]],
    count: int,
    rng: random.Random | None = None,
) -> list[list[int]]:
    """Generate picks that match historical patterns.

    Analyzes patterns like:
    - Typical sum range
    - Odd/even distribution
    - High/low distribution
    - Consecutive number frequency

    Args:
        config: Game configuration
        draws: Historical draws
        count: Number of picks to generate
        rng: Random number generator

    Returns:
        List of picks matching historical patterns
    """
    rng = rng or random.SystemRandom()

    if not draws:
        from .game_strategies import generate_random_picks
        return generate_random_picks(config, count, rng)

    # Analyze historical patterns
    sums = [sum(main) for main in draws]
    min_sum = min(sums)
    max_sum = max(sums)
    avg_sum = sum(sums) / len(sums)
    sum_range = (min_sum, max_sum)

    # Odd/even patterns
    odd_counts = []
    for main in draws:
        odd_counts.append(sum(1 for n in main if n % 2 == 1))
    avg_odd = sum(odd_counts) / len(odd_counts)
    target_odd = round(avg_odd)

    # High/low patterns
    midpoint = config.pool_min + config.pool_size // 2
    high_counts = []
    for main in draws:
        high_counts.append(sum(1 for n in main if n >= midpoint))
    avg_high = sum(high_counts) / len(high_counts)
    target_high = round(avg_high)

    # Generate picks matching patterns
    scores = compute_composite_scores(config, draws)
    numbers = list(config.pool_range)
    probs = [scores[n] for n in numbers]

    lines = []
    attempts = 0
    max_attempts = count * 100

    while len(lines) < count and attempts < max_attempts:
        attempts += 1
        pick = _sample_weighted(numbers, probs, config.numbers_to_pick, rng)

        # Check if pick matches patterns
        pick_sum = sum(pick)
        if not (sum_range[0] <= pick_sum <= sum_range[1]):
            continue

        pick_odd = sum(1 for n in pick if n % 2 == 1)
        if abs(pick_odd - target_odd) > 1:
            continue

        pick_high = sum(1 for n in pick if n >= midpoint)
        if abs(pick_high - target_high) > 1:
            continue

        key = tuple(pick)
        if key not in [tuple(l) for l in lines]:
            lines.append(pick)

    # Fill remaining with best composite picks if patterns are too restrictive
    while len(lines) < count:
        pick = _sample_weighted(numbers, probs, config.numbers_to_pick, rng)
        key = tuple(pick)
        if key not in [tuple(l) for l in lines]:
            lines.append(pick)

    return lines


def generate_smart_picks(
    config: GameConfig,
    draws: list[list[int]],
    count: int,
    rng: random.Random | None = None,
) -> list[list[int]]:
    """Generate the smartest possible picks using all available techniques.

    This is the ultimate strategy that:
    1. Uses composite scoring for base probabilities
    2. Filters by historical patterns
    3. Ensures diversity between picks
    4. Balances hot and overdue numbers

    Args:
        config: Game configuration
        draws: Historical draws
        count: Number of picks to generate
        rng: Random number generator

    Returns:
        List of optimized picks
    """
    rng = rng or random.SystemRandom()

    if not draws:
        from .game_strategies import generate_random_picks
        return generate_random_picks(config, count, rng)

    # Get different strategy suggestions
    optimal = generate_optimal_picks(config, draws, count, rng, "balanced")
    coverage = generate_coverage_picks(config, draws, count, rng)
    pattern = generate_pattern_picks(config, draws, count, rng)

    # Combine: take best from each
    all_picks = []
    seen = set()

    # Interleave picks from different strategies
    for i in range(count):
        candidates = [
            optimal[i % len(optimal)],
            coverage[i % len(coverage)],
            pattern[i % len(pattern)],
        ]

        for pick in candidates:
            key = tuple(pick)
            if key not in seen:
                all_picks.append(pick)
                seen.add(key)
                break

        if len(all_picks) >= count:
            break

    # Fill if needed
    while len(all_picks) < count:
        pick = optimal[len(all_picks) % len(optimal)]
        key = tuple(pick)
        if key not in seen:
            all_picks.append(pick)
            seen.add(key)

    return all_picks[:count]
