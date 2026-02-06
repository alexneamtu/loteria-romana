"""Regime detection for lottery draw history.

Identifies segments of draw history with different statistical
properties (e.g., different mean sum, different frequency patterns).
This helps backtest strategies within consistent regimes rather than
across regime boundaries where the distribution changed.

Uses binary segmentation with statistical testing.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeBoundary:
    """A detected regime boundary."""

    index: int
    confidence: float
    pre_mean: float
    post_mean: float


def _mean_and_var(values: list[float]) -> tuple[float, float]:
    """Compute mean and variance."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    v = sum((x - m) ** 2 for x in values) / n if n > 1 else 0.0
    return m, v


def _binary_segment_cost(values: list[float]) -> float:
    """Cost function: total squared deviation from segment mean."""
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return sum((x - m) ** 2 for x in values)


def _find_best_split(
    values: list[float],
    min_segment_size: int,
) -> tuple[int, float]:
    """Find the split point that minimizes total cost.

    Returns (best_index, cost_reduction).
    """
    n = len(values)
    if n < 2 * min_segment_size:
        return -1, 0.0

    total_cost = _binary_segment_cost(values)
    best_idx = -1
    best_reduction = 0.0

    for i in range(min_segment_size, n - min_segment_size + 1):
        left_cost = _binary_segment_cost(values[:i])
        right_cost = _binary_segment_cost(values[i:])
        reduction = total_cost - (left_cost + right_cost)

        if reduction > best_reduction:
            best_reduction = reduction
            best_idx = i

    return best_idx, best_reduction


def detect_regimes(
    values: list[float],
    min_segment_size: int = 30,
    significance_threshold: float = 0.01,
    max_regimes: int = 5,
) -> list[RegimeBoundary]:
    """Detect regime boundaries using binary segmentation.

    Recursively splits the sequence at the point that most reduces
    total variance, testing significance at each step.

    Args:
        values: Time-ordered numeric sequence (e.g. draw sums).
        min_segment_size: Minimum draws in a segment.
        significance_threshold: P-value threshold for accepting a split.
        max_regimes: Maximum number of regime boundaries to detect.

    Returns:
        List of RegimeBoundary sorted by index.
    """
    n = len(values)
    if n < 2 * min_segment_size:
        return []

    boundaries: list[RegimeBoundary] = []
    segments = [(0, n)]

    for _ in range(max_regimes):
        best_global_split = -1
        best_global_reduction = 0.0
        best_segment_idx = -1

        for seg_idx, (start, end) in enumerate(segments):
            seg_values = values[start:end]
            split_idx, reduction = _find_best_split(seg_values, min_segment_size)

            if split_idx >= 0 and reduction > best_global_reduction:
                best_global_reduction = reduction
                best_global_split = start + split_idx
                best_segment_idx = seg_idx

        if best_global_split < 0:
            break

        # Statistical significance test (F-test approximation)
        start, end = segments[best_segment_idx]
        seg_values = values[start:end]
        n_seg = len(seg_values)
        total_var = _binary_segment_cost(seg_values)

        if total_var <= 0:
            break

        local_idx = best_global_split - start
        left_cost = _binary_segment_cost(seg_values[:local_idx])
        right_cost = _binary_segment_cost(seg_values[local_idx:])

        f_stat = ((total_var - left_cost - right_cost) / 1.0) / (
            (left_cost + right_cost) / max(n_seg - 2, 1)
        )

        # Approximate p-value
        df1 = 1
        df2 = max(n_seg - 2, 1)
        p_value = _f_test_p_value(f_stat, df1, df2)

        if p_value > significance_threshold:
            break

        pre_mean, _ = _mean_and_var(seg_values[:local_idx])
        post_mean, _ = _mean_and_var(seg_values[local_idx:])

        boundaries.append(RegimeBoundary(
            index=best_global_split,
            confidence=1.0 - p_value,
            pre_mean=pre_mean,
            post_mean=post_mean,
        ))

        # Split the segment
        old_start, old_end = segments.pop(best_segment_idx)
        segments.insert(best_segment_idx, (old_start, best_global_split))
        segments.insert(best_segment_idx + 1, (best_global_split, old_end))

    return sorted(boundaries, key=lambda b: b.index)


def _f_test_p_value(f: float, df1: int, df2: int) -> float:
    """Approximate F-test p-value using normal approximation."""
    if f <= 0 or df1 <= 0 or df2 <= 0:
        return 1.0

    try:
        z = math.sqrt(2.0 * f) - math.sqrt(2.0 * df1 - 1.0) if df1 >= 1 else 0.0
        return 1.0 - _standard_normal_cdf(z)
    except (ValueError, ZeroDivisionError):
        return 1.0


def _standard_normal_cdf(z: float) -> float:
    """Approximate standard normal CDF (Abramowitz & Stegun)."""
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741,
        -1.453152027, 1.061405429,
    )
    p = 0.3275911

    sign = 1 if z >= 0 else -1
    z_abs = abs(z) / math.sqrt(2)

    t = 1.0 / (1.0 + p * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs)

    return 0.5 * (1.0 + sign * y)


def segment_draws_by_regime(
    draws: list[list[int]],
    boundaries: list[RegimeBoundary],
) -> list[list[list[int]]]:
    """Split draws into segments based on detected regime boundaries.

    Args:
        draws: Historical draws in chronological order.
        boundaries: Regime boundaries from detect_regimes().

    Returns:
        List of draw segments, one per regime.
    """
    if not boundaries:
        return [draws]

    sorted_boundaries = sorted(boundaries, key=lambda b: b.index)
    segments: list[list[list[int]]] = []

    prev = 0
    for boundary in sorted_boundaries:
        idx = boundary.index
        if idx > prev:
            segments.append(draws[prev:idx])
        prev = idx

    if prev < len(draws):
        segments.append(draws[prev:])

    return [s for s in segments if s]
