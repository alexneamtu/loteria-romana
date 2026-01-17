"""Statistical validation for lottery randomness verification.

This module provides tools to verify that lottery draws are consistent
with true randomness. These tests confirm the lottery is fair, not
that patterns exist to exploit.
"""

import math
from dataclasses import dataclass


@dataclass
class ChiSquareResult:
    """Result of a chi-square goodness-of-fit test."""

    statistic: float
    degrees_of_freedom: int
    p_value: float
    is_random: bool  # True if p > alpha
    alpha: float = 0.05

    def __str__(self) -> str:
        verdict = "RANDOM" if self.is_random else "NON-RANDOM"
        return (
            f"Chi-Square Test: χ² = {self.statistic:.2f}, "
            f"df = {self.degrees_of_freedom}, "
            f"p = {self.p_value:.4f}, "
            f"verdict = {verdict}"
        )


def chi_square_statistic(observed: list[int], expected: list[float]) -> float:
    """Compute chi-square statistic.

    Args:
        observed: Observed frequencies
        expected: Expected frequencies

    Returns:
        Chi-square test statistic
    """
    if len(observed) != len(expected):
        raise ValueError("Observed and expected must have same length")

    chi_sq = 0.0
    for obs, exp in zip(observed, expected):
        if exp > 0:
            chi_sq += (obs - exp) ** 2 / exp

    return chi_sq


def chi_square_p_value(chi_sq: float, df: int) -> float:
    """Approximate p-value for chi-square distribution.

    Uses Wilson-Hilferty approximation for the chi-square CDF.

    Args:
        chi_sq: Chi-square statistic
        df: Degrees of freedom

    Returns:
        Approximate p-value (1 - CDF)
    """
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")

    if chi_sq <= 0:
        return 1.0

    # Wilson-Hilferty approximation
    z = ((chi_sq / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))

    # Standard normal CDF approximation
    p_value = 1 - _standard_normal_cdf(z)

    return max(0.0, min(1.0, p_value))


def _standard_normal_cdf(z: float) -> float:
    """Approximate standard normal CDF using error function approximation."""
    # Use the approximation: Phi(z) ≈ 0.5 * (1 + erf(z / sqrt(2)))
    return 0.5 * (1 + _erf(z / math.sqrt(2)))


def _erf(x: float) -> float:
    """Approximate error function using Horner's method."""
    # Abramowitz and Stegun approximation 7.1.26
    sign = 1 if x >= 0 else -1
    x = abs(x)

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return sign * y


def test_uniformity(
    draws: list[list[int]],
    pool_size: int,
    numbers_per_draw: int,
    alpha: float = 0.05,
) -> ChiSquareResult:
    """Test if lottery draws follow uniform distribution.

    Uses chi-square goodness-of-fit test to verify that each number
    appears with approximately equal frequency.

    Args:
        draws: List of draw results (each a list of numbers)
        pool_size: Total numbers in the pool (e.g., 49 for 6/49)
        numbers_per_draw: Numbers drawn each time (e.g., 6)
        alpha: Significance level (default 0.05)

    Returns:
        ChiSquareResult with test statistics and conclusion
    """
    # Count occurrences of each number
    counts = [0] * pool_size
    for draw in draws:
        for num in draw:
            if 1 <= num <= pool_size:
                counts[num - 1] += 1

    # Expected count under uniform distribution
    total_numbers = len(draws) * numbers_per_draw
    expected = [total_numbers / pool_size] * pool_size

    # Compute chi-square statistic
    chi_sq = chi_square_statistic(counts, expected)
    df = pool_size - 1
    p_value = chi_square_p_value(chi_sq, df)

    return ChiSquareResult(
        statistic=chi_sq,
        degrees_of_freedom=df,
        p_value=p_value,
        is_random=p_value > alpha,
        alpha=alpha,
    )


def test_independence(
    draws: list[list[int]],
    pool_size: int,
    alpha: float = 0.05,
) -> ChiSquareResult:
    """Test if consecutive draws are independent.

    Uses chi-square test on pair frequencies to check if the
    appearance of a number affects the next draw.

    Args:
        draws: List of draw results (oldest first)
        pool_size: Total numbers in the pool
        alpha: Significance level

    Returns:
        ChiSquareResult with test statistics and conclusion
    """
    if len(draws) < 2:
        return ChiSquareResult(
            statistic=0.0,
            degrees_of_freedom=0,
            p_value=1.0,
            is_random=True,
            alpha=alpha,
        )

    # Count transitions: number n in draw t followed by number m in draw t+1
    transition_counts: dict[tuple[int, int], int] = {}

    for i in range(len(draws) - 1):
        current_draw = set(draws[i])
        next_draw = set(draws[i + 1])

        for n in current_draw:
            for m in next_draw:
                key = (n, m)
                transition_counts[key] = transition_counts.get(key, 0) + 1

    # Expected under independence
    total_transitions = sum(transition_counts.values())
    if total_transitions == 0:
        return ChiSquareResult(
            statistic=0.0,
            degrees_of_freedom=0,
            p_value=1.0,
            is_random=True,
            alpha=alpha,
        )

    # Count marginal frequencies
    from_counts: dict[int, int] = {}
    to_counts: dict[int, int] = {}

    for (n, m), count in transition_counts.items():
        from_counts[n] = from_counts.get(n, 0) + count
        to_counts[m] = to_counts.get(m, 0) + count

    # Compute chi-square for independence
    chi_sq = 0.0
    df_count = 0

    for (n, m), observed in transition_counts.items():
        if from_counts.get(n, 0) > 0 and to_counts.get(m, 0) > 0:
            expected = (from_counts[n] * to_counts[m]) / total_transitions
            if expected > 0:
                chi_sq += (observed - expected) ** 2 / expected
                df_count += 1

    # Degrees of freedom for independence test
    df = max(1, df_count - len(from_counts) - len(to_counts) + 1)
    p_value = chi_square_p_value(chi_sq, df)

    return ChiSquareResult(
        statistic=chi_sq,
        degrees_of_freedom=df,
        p_value=p_value,
        is_random=p_value > alpha,
        alpha=alpha,
    )


def runs_test(draws: list[list[int]], pool_size: int) -> dict:
    """Perform runs test for randomness on number appearances.

    A "run" is a sequence of consecutive draws where a specific
    number either appears or doesn't appear. Too few or too many
    runs indicates non-randomness.

    Args:
        draws: List of draw results
        pool_size: Total numbers in the pool

    Returns:
        Dict with test results for each number
    """
    results: dict[int, dict] = {}

    for num in range(1, pool_size + 1):
        # Create binary sequence: 1 if num appeared, 0 otherwise
        sequence = [1 if num in draw else 0 for draw in draws]

        n = len(sequence)
        n1 = sum(sequence)  # Number of 1s
        n0 = n - n1  # Number of 0s

        if n1 == 0 or n0 == 0:
            results[num] = {
                "runs": 0,
                "expected_runs": 0,
                "z_score": 0,
                "p_value": 1.0,
                "is_random": True,
            }
            continue

        # Count runs
        runs = 1
        for i in range(1, n):
            if sequence[i] != sequence[i - 1]:
                runs += 1

        # Expected runs and variance under randomness
        expected_runs = (2 * n0 * n1) / n + 1
        variance = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))

        if variance > 0:
            z = (runs - expected_runs) / math.sqrt(variance)
            p_value = 2 * (1 - _standard_normal_cdf(abs(z)))
        else:
            z = 0
            p_value = 1.0

        results[num] = {
            "runs": runs,
            "expected_runs": expected_runs,
            "z_score": z,
            "p_value": p_value,
            "is_random": p_value > 0.05,
        }

    return results


def validate_lottery_randomness(
    draws: list[list[int]],
    pool_size: int,
    numbers_per_draw: int,
) -> dict:
    """Comprehensive randomness validation for lottery draws.

    Runs multiple statistical tests to verify the lottery is fair.

    Args:
        draws: List of draw results
        pool_size: Total numbers in the pool
        numbers_per_draw: Numbers drawn each time

    Returns:
        Dict with all test results and overall conclusion
    """
    uniformity = test_uniformity(draws, pool_size, numbers_per_draw)
    independence = test_independence(draws, pool_size)
    runs_results = runs_test(draws, pool_size)

    # Count how many numbers fail runs test
    runs_failures = sum(
        1 for r in runs_results.values() if not r.get("is_random", True)
    )
    runs_pass_rate = 1 - (runs_failures / pool_size) if pool_size > 0 else 1.0

    # Overall assessment
    is_random = (
        uniformity.is_random
        and independence.is_random
        and runs_pass_rate >= 0.95  # Allow 5% of numbers to fail by chance
    )

    return {
        "uniformity_test": uniformity,
        "independence_test": independence,
        "runs_test": {
            "pass_rate": runs_pass_rate,
            "failures": runs_failures,
            "is_random": runs_pass_rate >= 0.95,
        },
        "overall": {
            "is_random": is_random,
            "conclusion": (
                "Lottery draws are consistent with true randomness."
                if is_random
                else "Some deviation from randomness detected (may be statistical noise)."
            ),
        },
    }
