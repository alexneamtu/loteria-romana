"""Chi-square bias detection for lottery draws.

Detects whether observed number frequencies deviate significantly
from the expected uniform distribution, using the Haigh (1997)
correction for draws without replacement.
"""

import math
from dataclasses import dataclass


# Chi-square critical values at p=0.05 for common degrees of freedom.
# Source: standard chi-square distribution tables.
_CHI_SQUARE_CRITICAL_005 = {
    19: 30.14,
    29: 42.56,
    39: 54.57,
    44: 60.48,
    48: 65.17,
}


@dataclass(frozen=True)
class BiasReport:
    """Result of a chi-square uniformity test on lottery draws."""

    chi_square: float
    degrees_of_freedom: int
    critical_value: float
    significant: bool
    bias_strength: float


def _lookup_critical_value(df: int) -> float:
    """Look up or interpolate chi-square critical value at p=0.05."""
    if df in _CHI_SQUARE_CRITICAL_005:
        return _CHI_SQUARE_CRITICAL_005[df]

    keys = sorted(_CHI_SQUARE_CRITICAL_005.keys())
    if df < keys[0]:
        return _CHI_SQUARE_CRITICAL_005[keys[0]]
    if df > keys[-1]:
        return _CHI_SQUARE_CRITICAL_005[keys[-1]]

    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i + 1]:
            low_df, high_df = keys[i], keys[i + 1]
            low_cv = _CHI_SQUARE_CRITICAL_005[low_df]
            high_cv = _CHI_SQUARE_CRITICAL_005[high_df]
            fraction = (df - low_df) / (high_df - low_df)
            return low_cv + fraction * (high_cv - low_cv)

    return float(df)


def chi_square_uniformity_test(
    draws: list[list[int]],
    pool_size: int,
    numbers_drawn: int,
) -> BiasReport:
    """Test whether observed number frequencies deviate from uniform.

    Applies the Haigh (1997) correction for sampling without replacement
    within each draw: corrected_chi_sq = naive_chi_sq * (M-1) / (M-m)
    where M = pool_size and m = numbers_drawn.

    Args:
        draws: Historical lottery draws (each is a list of drawn numbers).
        pool_size: Total numbers in the pool (e.g. 45 for Joker).
        numbers_drawn: Numbers drawn per draw (e.g. 5 for Joker).

    Returns:
        BiasReport with test results.
    """
    total_draws = len(draws)
    if total_draws == 0:
        return BiasReport(
            chi_square=0.0,
            degrees_of_freedom=pool_size - 1,
            critical_value=_lookup_critical_value(pool_size - 1),
            significant=False,
            bias_strength=0.0,
        )

    expected_count = total_draws * numbers_drawn / pool_size

    observed = [0] * pool_size
    for draw in draws:
        for num in draw:
            if 1 <= num <= pool_size:
                observed[num - 1] += 1

    naive_chi_sq = sum(
        (obs - expected_count) ** 2 / expected_count
        for obs in observed
    )

    correction = (pool_size - 1) / (pool_size - numbers_drawn)
    chi_sq = naive_chi_sq / correction

    df = pool_size - 1
    critical_value = _lookup_critical_value(df)
    significant = chi_sq > critical_value

    bias_strength = min(chi_sq / critical_value, 1.0) if critical_value > 0 else 0.0

    return BiasReport(
        chi_square=chi_sq,
        degrees_of_freedom=df,
        critical_value=critical_value,
        significant=significant,
        bias_strength=bias_strength,
    )
