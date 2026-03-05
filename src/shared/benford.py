"""Benford's Law analysis for lottery draw sequences.

Compares the first-digit distribution of drawn numbers against both
Benford's Law and the expected uniform distribution for the pool.
A truly random lottery should follow uniform digit probabilities
(determined by the pool range), NOT Benford's Law. If the data fits
Benford better than uniform, that is a red flag.
"""

from __future__ import annotations

import math
from typing import Any

from .analysis_result import AnalysisResult

_MIN_TOTAL_NUMBERS = 50


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _chi_square_survival(chi2: float, degrees_of_freedom: int) -> float:
    """P-value for chi-square statistic using Wilson-Hilferty approximation.

    Returns the survival function P(X >= chi2) for a chi-square distribution
    with the given degrees of freedom.
    """
    if degrees_of_freedom <= 0:
        return 1.0
    if chi2 <= 0.0:
        return 1.0

    k = float(degrees_of_freedom)
    z = ((chi2 / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(
        2.0 / (9.0 * k)
    )
    return 1.0 - _normal_cdf(z)


def _benford_probs() -> dict[int, float]:
    """Benford's Law first-digit probabilities."""
    return {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def _compute_uniform_digit_probs(pool_size: int) -> dict[int, float]:
    """Expected first-digit probabilities for uniform draws from [1, pool_size].

    Counts how many numbers in [1, pool_size] have each leading digit,
    then normalizes to probabilities.
    """
    digit_counts: dict[int, int] = {d: 0 for d in range(1, 10)}
    for number in range(1, pool_size + 1):
        leading = int(str(number)[0])
        digit_counts[leading] += 1

    total = sum(digit_counts.values())
    return {d: count / total for d, count in digit_counts.items()}


def _extract_first_digits(draws: list[list[int]]) -> list[int]:
    """Extract the leading digit of every number in every draw."""
    digits: list[int] = []
    for draw in draws:
        for number in draw:
            if number > 0:
                digits.append(int(str(number)[0]))
    return digits


def _chi_square_goodness_of_fit(
    observed_counts: dict[int, int],
    expected_probs: dict[int, float],
    total: int,
) -> tuple[float, int]:
    """Compute chi-square statistic and degrees of freedom.

    Returns (chi2, degrees_of_freedom).
    """
    chi2 = 0.0
    active_bins = 0
    for digit in range(1, 10):
        expected = expected_probs.get(digit, 0.0) * total
        observed = observed_counts.get(digit, 0)
        if expected > 0:
            chi2 += (observed - expected) ** 2 / expected
            active_bins += 1

    degrees_of_freedom = max(active_bins - 1, 1)
    return chi2, degrees_of_freedom


def run_benford_analysis(
    draws: list[list[int]],
    pool_size: int,
    significance: float = 0.01,
) -> list[AnalysisResult]:
    """Compare first-digit distribution against Benford's Law and uniform.

    'Passed' means data is consistent with uniform (not Benford-like).
    If data fits Benford better than uniform, the test FAILs (red flag).

    Args:
        draws: Historical lottery draws (each a sorted list of ints).
        pool_size: Total numbers in the pool (e.g. 45 for Joker).
        significance: Alpha level for hypothesis testing.

    Returns:
        A list containing one AnalysisResult.
    """
    first_digits = _extract_first_digits(draws)
    total = len(first_digits)

    if total < _MIN_TOTAL_NUMBERS:
        return [_inconclusive(
            len(draws),
            f"Need >= {_MIN_TOTAL_NUMBERS} total numbers, have {total}",
        )]

    observed_counts: dict[int, int] = {d: 0 for d in range(1, 10)}
    for digit in first_digits:
        observed_counts[digit] += 1

    observed_probs = {d: count / total for d, count in observed_counts.items()}

    benford = _benford_probs()
    uniform = _compute_uniform_digit_probs(pool_size)

    benford_chi2, benford_df = _chi_square_goodness_of_fit(
        observed_counts, benford, total,
    )
    uniform_chi2, uniform_df = _chi_square_goodness_of_fit(
        observed_counts, uniform, total,
    )

    benford_p = _chi_square_survival(benford_chi2, benford_df)
    uniform_p = _chi_square_survival(uniform_chi2, uniform_df)

    fits_benford = benford_p >= significance
    fits_uniform = uniform_p >= significance

    # Pass if data is consistent with uniform.
    # Fail if data fits Benford better than uniform (red flag for lottery).
    if fits_uniform:
        passed = True
    elif fits_benford and not fits_uniform:
        passed = False
    else:
        # Neither fits well; use uniform p-value for verdict
        passed = uniform_p >= significance

    details: dict[str, Any] = {
        "benford_chi2": round(benford_chi2, 4),
        "benford_p": round(benford_p, 6),
        "uniform_chi2": round(uniform_chi2, 4),
        "uniform_p": round(uniform_p, 6),
        "observed_probs": {d: round(p, 4) for d, p in observed_probs.items()},
        "fits_benford": fits_benford,
        "fits_uniform": fits_uniform,
        "benford_df": benford_df,
        "uniform_df": uniform_df,
        "total_digits": total,
    }

    verdict = "pass" if passed else "FAIL"
    summary = (
        f"Benford analysis: uniform_chi2={uniform_chi2:.2f} (p={uniform_p:.4f}), "
        f"benford_chi2={benford_chi2:.2f} (p={benford_p:.4f}) — {verdict}"
    )

    return [AnalysisResult(
        test_name="benford_first_digit",
        game="all",
        passed=passed,
        p_value=uniform_p,
        statistic=float(uniform_chi2),
        threshold=significance,
        sample_size=len(draws),
        details=details,
        summary=summary,
    )]


def _inconclusive(sample_size: int, reason: str) -> AnalysisResult:
    """Create an inconclusive AnalysisResult."""
    return AnalysisResult(
        test_name="benford_first_digit",
        game="all",
        passed=None,
        p_value=None,
        statistic=0.0,
        threshold=0.0,
        sample_size=sample_size,
        details={"reason": reason},
        summary=f"Inconclusive: {reason}",
    )
