"""Per-number Wald-Wolfowitz runs test for lottery draws.

The runs test detects non-random patterns in binary sequences.
For each number in the pool, we create a binary sequence:
True if the number appeared in that draw, False otherwise.
Then we test whether the number of "runs" (consecutive True or
consecutive False) is consistent with randomness.

Too few runs → clustering (streaks of appearance/absence)
Too many runs → alternating (suspiciously regular)
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RunsReport:
    """Result of a runs test for a single number."""

    number: int
    observed_runs: int
    expected_runs: float
    z_score: float
    p_value: float
    significant: bool


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


def wald_wolfowitz_runs_test(
    sequence: list[bool],
    number: int,
    alpha: float = 0.05,
) -> RunsReport:
    """Perform Wald-Wolfowitz runs test on a binary sequence.

    Args:
        sequence: Binary sequence (True/False for each draw).
        number: The lottery number being tested.
        alpha: Significance level (two-tailed).

    Returns:
        RunsReport with test results.
    """
    n = len(sequence)
    if n < 2:
        return RunsReport(
            number=number, observed_runs=0, expected_runs=0.0,
            z_score=0.0, p_value=1.0, significant=False,
        )

    n1 = sum(1 for x in sequence if x)
    n0 = n - n1

    if n1 == 0 or n0 == 0:
        return RunsReport(
            number=number, observed_runs=1 if n > 0 else 0,
            expected_runs=1.0, z_score=0.0, p_value=1.0,
            significant=False,
        )

    # Count runs
    runs = 1
    for i in range(1, n):
        if sequence[i] != sequence[i - 1]:
            runs += 1

    # Expected runs and variance under H0
    expected = 1.0 + (2.0 * n1 * n0) / n
    variance = (2.0 * n1 * n0 * (2.0 * n1 * n0 - n)) / (n * n * (n - 1.0))

    if variance <= 0:
        return RunsReport(
            number=number, observed_runs=runs,
            expected_runs=expected, z_score=0.0, p_value=1.0,
            significant=False,
        )

    z = (runs - expected) / math.sqrt(variance)
    p_value = 2.0 * (1.0 - _standard_normal_cdf(abs(z)))

    return RunsReport(
        number=number,
        observed_runs=runs,
        expected_runs=expected,
        z_score=z,
        p_value=p_value,
        significant=p_value < alpha,
    )


def per_number_runs_analysis(
    draws: list[list[int]],
    pool_size: int,
    alpha: float = 0.05,
) -> list[RunsReport]:
    """Run the Wald-Wolfowitz runs test for every number in the pool.

    Applies Bonferroni correction for multiple comparisons.

    Args:
        draws: Historical draws (each is a list of drawn numbers).
        pool_size: Total numbers in the pool.
        alpha: Base significance level (before Bonferroni).

    Returns:
        List of RunsReport, one per number.
    """
    adjusted_alpha = alpha / pool_size

    reports = []
    for number in range(1, pool_size + 1):
        sequence = [number in draw for draw in draws]
        report = wald_wolfowitz_runs_test(sequence, number, alpha=adjusted_alpha)
        reports.append(report)

    return reports
