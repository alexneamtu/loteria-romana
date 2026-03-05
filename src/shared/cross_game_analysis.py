"""Cross-game correlation analysis.

Detects statistical dependencies between different lottery games
drawn on the same dates. If games are truly independent, their
draw sums and number overlaps should show no significant correlation.
"""

from __future__ import annotations

import itertools
import math
from datetime import date
from typing import Any

from .analysis_result import AnalysisResult

_MIN_COMMON_DATES = 30


def run_cross_game_analysis(
    game_draws: dict[str, list[tuple[date, list[int]]]],
    significance: float = 0.01,
) -> list[AnalysisResult]:
    """Analyze pairwise correlations between different lottery games.

    Args:
        game_draws: Mapping of game name to list of (date, numbers) tuples.
        significance: Alpha level for hypothesis testing.

    Returns:
        A list of AnalysisResult objects, two per game pair
        (sum correlation + number overlap). Empty if fewer than 2 games.
    """
    game_names = sorted(game_draws.keys())
    if len(game_names) < 2:
        return []

    results: list[AnalysisResult] = []
    for name_a, name_b in itertools.combinations(game_names, 2):
        pair_label = f"{name_a} vs {name_b}"
        draws_a = _index_by_date(game_draws[name_a])
        draws_b = _index_by_date(game_draws[name_b])
        common_dates = sorted(set(draws_a.keys()) & set(draws_b.keys()))

        if len(common_dates) < _MIN_COMMON_DATES:
            results.append(_inconclusive(
                "sum_correlation", pair_label, len(common_dates),
                f"Need >= {_MIN_COMMON_DATES} common dates, have {len(common_dates)}",
            ))
            results.append(_inconclusive(
                "number_overlap", pair_label, len(common_dates),
                f"Need >= {_MIN_COMMON_DATES} common dates, have {len(common_dates)}",
            ))
            continue

        sums_a = [sum(draws_a[d]) for d in common_dates]
        sums_b = [sum(draws_b[d]) for d in common_dates]
        results.append(_sum_correlation_test(
            sums_a, sums_b, pair_label, len(common_dates), significance,
        ))

        overlaps = [
            len(set(draws_a[d]) & set(draws_b[d])) for d in common_dates
        ]
        pool_a = _infer_pool_size(game_draws[name_a])
        pool_b = _infer_pool_size(game_draws[name_b])
        pick_a = _infer_pick_count(game_draws[name_a])
        pick_b = _infer_pick_count(game_draws[name_b])
        results.append(_number_overlap_test(
            overlaps, pool_a, pool_b, pick_a, pick_b,
            pair_label, len(common_dates), significance,
        ))

    return results


# ---------------------------------------------------------------------------
# Helper: index draws by date
# ---------------------------------------------------------------------------

def _index_by_date(
    draws: list[tuple[date, list[int]]],
) -> dict[date, list[int]]:
    """Build a date-keyed lookup for a game's draws."""
    return {d: numbers for d, numbers in draws}


# ---------------------------------------------------------------------------
# Helper: infer pool size and pick count
# ---------------------------------------------------------------------------

def _infer_pool_size(draws: list[tuple[date, list[int]]]) -> int:
    """Estimate the pool size from the maximum number observed."""
    max_num = 0
    for _, numbers in draws:
        for n in numbers:
            if n > max_num:
                max_num = n
    return max_num


def _infer_pick_count(draws: list[tuple[date, list[int]]]) -> int:
    """Estimate numbers per draw from the first draw."""
    if not draws:
        return 0
    return len(draws[0][1])


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two sequences.

    Returns 0.0 if variance is zero in either sequence.
    """
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = math.sqrt(var_x * var_y)
    if denominator == 0:
        return 0.0

    return cov / denominator


# ---------------------------------------------------------------------------
# Test 1: Sum correlation (Pearson r + t-test)
# ---------------------------------------------------------------------------

def _sum_correlation_test(
    sums_a: list[float],
    sums_b: list[float],
    pair_label: str,
    sample_size: int,
    significance: float,
) -> AnalysisResult:
    """Test whether draw-sums of two games are correlated."""
    r = _pearson_correlation(
        [float(s) for s in sums_a],
        [float(s) for s in sums_b],
    )

    # t-statistic: t = r * sqrt((n-2) / (1-r^2))
    n = sample_size
    denominator = 1.0 - r * r
    if denominator <= 0:
        # Perfect correlation or anti-correlation
        t_stat = float("inf") if abs(r) > 0 else 0.0
    else:
        t_stat = abs(r) * math.sqrt((n - 2) / denominator)

    # Two-tailed p-value using normal approximation (valid for df > 30)
    p_value = 2.0 * (1.0 - _normal_cdf(t_stat))
    passed = p_value >= significance

    return AnalysisResult(
        test_name="sum_correlation",
        game=pair_label,
        passed=passed,
        p_value=p_value,
        statistic=r,
        threshold=significance,
        sample_size=n,
        details={
            "pearson_r": round(r, 6),
            "t_statistic": round(t_stat, 4) if t_stat != float("inf") else "inf",
        },
        summary=(
            f"Sum correlation: r={r:.4f}, t={t_stat:.2f}, "
            f"p={p_value:.4f} ({'pass' if passed else 'FAIL'})"
        ),
    )


# ---------------------------------------------------------------------------
# Test 2: Number overlap (z-test vs expected)
# ---------------------------------------------------------------------------

def _number_overlap_test(
    overlaps: list[int],
    pool_a: int,
    pool_b: int,
    pick_a: int,
    pick_b: int,
    pair_label: str,
    sample_size: int,
    significance: float,
) -> AnalysisResult:
    """Test whether the mean number overlap exceeds random expectation.

    Under independence, the expected overlap between two draws from
    pools of size P_a and P_b picking k_a and k_b numbers is:
        E[overlap] = k_a * k_b / max(P_a, P_b)
    assuming uniform draws over the shared range.
    """
    shared_pool = min(pool_a, pool_b)
    if shared_pool <= 0 or pick_a <= 0 or pick_b <= 0:
        return _inconclusive(
            "number_overlap", pair_label, sample_size,
            "Cannot compute expected overlap (zero pool or pick count)",
        )

    # Hypergeometric expected overlap: E = pick_a * pick_b / shared_pool
    expected_overlap = pick_a * pick_b / shared_pool
    mean_overlap = sum(overlaps) / len(overlaps)

    # Variance of hypergeometric overlap (approximation):
    # Var = pick_a * pick_b * (shared_pool - pick_a) * (shared_pool - pick_b)
    #       / (shared_pool^2 * (shared_pool - 1))
    numerator = pick_a * pick_b * (shared_pool - pick_a) * (shared_pool - pick_b)
    denominator = shared_pool * shared_pool * max(shared_pool - 1, 1)
    single_draw_variance = numerator / denominator

    if single_draw_variance <= 0:
        return _inconclusive(
            "number_overlap", pair_label, sample_size,
            "Zero variance in expected overlap",
        )

    # Standard error of the mean overlap
    stderr = math.sqrt(single_draw_variance / sample_size)
    if stderr <= 0:
        return _inconclusive(
            "number_overlap", pair_label, sample_size,
            "Zero standard error",
        )

    z = (mean_overlap - expected_overlap) / stderr

    # Two-tailed p-value
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    passed = p_value >= significance

    return AnalysisResult(
        test_name="number_overlap",
        game=pair_label,
        passed=passed,
        p_value=p_value,
        statistic=z,
        threshold=significance,
        sample_size=sample_size,
        details={
            "mean_overlap": round(mean_overlap, 4),
            "expected_overlap": round(expected_overlap, 4),
            "z_statistic": round(z, 4),
        },
        summary=(
            f"Number overlap: mean={mean_overlap:.2f}, "
            f"expected={expected_overlap:.2f}, z={z:.2f}, "
            f"p={p_value:.4f} ({'pass' if passed else 'FAIL'})"
        ),
    )


# ---------------------------------------------------------------------------
# Inconclusive result helper
# ---------------------------------------------------------------------------

def _inconclusive(
    test_name: str,
    pair_label: str,
    sample_size: int,
    reason: str,
) -> AnalysisResult:
    """Create an inconclusive AnalysisResult for a game pair."""
    return AnalysisResult(
        test_name=test_name,
        game=pair_label,
        passed=None,
        p_value=None,
        statistic=0.0,
        threshold=0.0,
        sample_size=sample_size,
        details={"reason": reason},
        summary=f"Inconclusive: {reason}",
    )
