"""Hurst exponent analysis for lottery draw sequences.

The Hurst exponent H measures long-range dependence in a time series:
  - H ~ 0.5: random (no memory)
  - H > 0.5: persistent (trends tend to continue)
  - H < 0.5: anti-persistent (trends tend to reverse)

Uses rescaled range (R/S) analysis with log-log regression to estimate H.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from .analysis_result import AnalysisResult

_MIN_DRAWS = 50
_MIN_GAPS = 20
_H_DEVIATION_THRESHOLD = 0.15


def _compute_gap_series(draws: list[list[int]], number: int) -> list[int]:
    """Compute the gap series for a number: draws between appearances.

    Each gap value is the number of draws between consecutive appearances
    of the given number.
    """
    gaps: list[int] = []
    last_seen: int | None = None

    for index, draw in enumerate(draws):
        if number in draw:
            if last_seen is not None:
                gaps.append(index - last_seen)
            last_seen = index

    return gaps


def _linear_regression_slope(x: list[float], y: list[float]) -> float | None:
    """Compute the OLS slope for simple linear regression y = a + b*x.

    Returns None if the regression is degenerate (e.g. all x values equal).
    """
    n = len(x)
    if n < 2:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = sum((xi - mean_x) ** 2 for xi in x)

    if denominator == 0.0:
        return None

    return numerator / denominator


def rescaled_range_hurst(series: list[float], min_window: int = 8) -> float | None:
    """Estimate the Hurst exponent using rescaled range (R/S) analysis.

    Divides the series into non-overlapping windows of various sizes,
    computes the rescaled range for each, and fits a log-log regression
    to estimate H.

    Args:
        series: Time series data (at least min_window values).
        min_window: Smallest window size to use.

    Returns:
        Estimated Hurst exponent, or None if the series is too short.
    """
    n = len(series)
    if n < min_window:
        return None

    log_sizes: list[float] = []
    log_rs_values: list[float] = []

    window_size = min_window
    while window_size <= n // 2:
        segment_count = n // window_size
        if segment_count < 1:
            break

        rs_values: list[float] = []
        for segment_index in range(segment_count):
            start = segment_index * window_size
            end = start + window_size
            segment = series[start:end]

            mean_seg = sum(segment) / window_size
            deviations = [val - mean_seg for val in segment]

            cumulative = _cumulative_sum(deviations)
            range_val = max(cumulative) - min(cumulative)

            std_val = _standard_deviation(segment)
            if std_val > 0:
                rs_values.append(range_val / std_val)

        if rs_values:
            mean_rs = sum(rs_values) / len(rs_values)
            if mean_rs > 0:
                log_sizes.append(math.log(window_size))
                log_rs_values.append(math.log(mean_rs))

        window_size *= 2

    if len(log_sizes) < 2:
        return None

    return _linear_regression_slope(log_sizes, log_rs_values)


def _cumulative_sum(values: list[float]) -> list[float]:
    """Compute the cumulative sum of a list of values."""
    result: list[float] = []
    running = 0.0
    for val in values:
        running += val
        result.append(running)
    return result


def _standard_deviation(values: list[float]) -> float:
    """Compute sample standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_val = sum(values) / n
    variance = sum((v - mean_val) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def run_hurst_analysis(
    draws: list[list[int]],
    pool_size: int,
    significance: float = 0.01,
) -> list[AnalysisResult]:
    """Run Hurst exponent analysis on lottery draw gap series.

    For each number in the pool, computes the gap series (draws between
    appearances) and estimates the Hurst exponent via R/S analysis.
    Numbers with |H - 0.5| > 0.15 are flagged as potentially non-random.

    Args:
        draws: Historical lottery draws (each a sorted list of ints).
        pool_size: Total numbers in the pool (e.g. 45 for Joker).
        significance: Alpha level (used for threshold context).

    Returns:
        A list containing one aggregate AnalysisResult.
    """
    n = len(draws)

    if n < _MIN_DRAWS:
        return [_inconclusive(n, f"Need >= {_MIN_DRAWS} draws, have {n}")]

    hurst_values: dict[int, float] = {}
    flagged_numbers: list[int] = []

    for number in range(1, pool_size + 1):
        gaps = _compute_gap_series(draws, number)
        if len(gaps) < _MIN_GAPS:
            continue

        gap_floats = [float(g) for g in gaps]
        h = rescaled_range_hurst(gap_floats, min_window=8)
        if h is None:
            continue

        hurst_values[number] = h
        if abs(h - 0.5) > _H_DEVIATION_THRESHOLD:
            flagged_numbers.append(number)

    numbers_analyzed = len(hurst_values)

    if numbers_analyzed == 0:
        return [_inconclusive(n, "No numbers had enough gap data for analysis")]

    all_h = list(hurst_values.values())
    mean_h = sum(all_h) / numbers_analyzed
    std_h = statistics.stdev(all_h) if numbers_analyzed >= 2 else 0.0

    passed = len(flagged_numbers) <= numbers_analyzed // 4

    return [
        AnalysisResult(
            test_name="hurst_aggregate",
            game="all",
            passed=passed,
            p_value=None,
            statistic=mean_h,
            threshold=significance,
            sample_size=n,
            details={
                "mean_hurst": round(mean_h, 4),
                "std_hurst": round(std_h, 4),
                "flagged_numbers": flagged_numbers,
                "numbers_analyzed": numbers_analyzed,
                "h_threshold": _H_DEVIATION_THRESHOLD,
            },
            summary=(
                f"Hurst aggregate: mean_H={mean_h:.4f}, std={std_h:.4f}, "
                f"flagged={len(flagged_numbers)}/{numbers_analyzed} "
                f"({'pass' if passed else 'FAIL'})"
            ),
        ),
    ]


def _inconclusive(sample_size: int, reason: str) -> AnalysisResult:
    """Create an inconclusive AnalysisResult for the Hurst analysis."""
    return AnalysisResult(
        test_name="hurst_aggregate",
        game="all",
        passed=None,
        p_value=None,
        statistic=0.0,
        threshold=0.0,
        sample_size=sample_size,
        details={"reason": reason},
        summary=f"Inconclusive: {reason}",
    )
