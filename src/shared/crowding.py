"""Heuristics for estimating line crowding risk.

The goal is not predicting draws. The goal is avoiding number patterns
that many human players tend to choose, which can reduce payout sharing
risk conditional on winning.
"""

from collections import Counter


def anti_crowding_score(line: list[int], pool_size: int) -> float:
    """Return a normalized anti-crowding score in [0, 1]."""
    if not line or pool_size <= 0:
        return 0.0

    numbers = sorted(line)
    k = len(numbers)
    if k == 1:
        return 1.0

    penalties = 0.0

    # Birthday bias: many players overuse numbers <= 31.
    low_ratio = sum(1 for n in numbers if n <= 31) / k
    penalties += 0.40 * low_ratio

    # Consecutive pattern bias.
    diffs = [numbers[i + 1] - numbers[i] for i in range(k - 1)]
    consecutive_pairs = sum(1 for d in diffs if d == 1)
    penalties += 0.25 * (consecutive_pairs / (k - 1))

    # Exact arithmetic progression is heavily patterned.
    if len(set(diffs)) == 1:
        penalties += 0.20

    # Same last-digit clustering.
    mod_counts = Counter(n % 10 for n in numbers)
    max_mod = max(mod_counts.values())
    penalties += 0.15 * ((max_mod - 1) / (k - 1))

    # Narrow spread tends to look "patterned"/human.
    spread = (numbers[-1] - numbers[0]) / max(pool_size - 1, 1)
    if spread < 0.5:
        penalties += 0.15 * ((0.5 - spread) / 0.5)

    score = 1.0 - penalties
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def average_anti_crowding_score(lines: list[list[int]], pool_size: int) -> float:
    """Average anti-crowding score across a set of lines."""
    if not lines:
        return 0.0
    return sum(anti_crowding_score(line, pool_size) for line in lines) / len(lines)
