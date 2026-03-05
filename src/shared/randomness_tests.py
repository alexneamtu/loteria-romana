"""NIST SP 800-22 randomness tests adapted for lottery draw sequences.

Implements 6 tests applicable at typical lottery sample sizes (~1000 draws).
Each draw is converted to per-number binary sequences (1=appeared, 0=absent)
and statistical tests assess whether the sequences are consistent with
uniform random draws.
"""

from __future__ import annotations

import math
from typing import Any

from .analysis_result import AnalysisResult

# ---------------------------------------------------------------------------
# Minimum draw counts for each test
# ---------------------------------------------------------------------------
_MIN_DRAWS_FREQUENCY = 1
_MIN_DRAWS_RUNS = 100
_MIN_DRAWS_LONGEST_RUN = 200
_MIN_DRAWS_CUMULATIVE_SUMS = 100
_MIN_DRAWS_SERIAL = 500
_MIN_DRAWS_APPROXIMATE_ENTROPY = 500


# ---------------------------------------------------------------------------
# Math helpers (standard-library only)
# ---------------------------------------------------------------------------

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
    # Wilson-Hilferty cube-root transformation
    z = ((chi2 / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(
        2.0 / (9.0 * k)
    )
    return 1.0 - _normal_cdf(z)


def _erfc(x: float) -> float:
    """Complementary error function via math.erfc."""
    return math.erfc(x)


# ---------------------------------------------------------------------------
# Per-number binary sequence builder
# ---------------------------------------------------------------------------

def _build_binary_sequences(
    draws: list[list[int]], pool_size: int,
) -> dict[int, list[int]]:
    """Build a binary presence sequence for each number in the pool.

    Returns a dict mapping number -> list of 0/1 values across draws.
    """
    sequences: dict[int, list[int]] = {
        num: [0] * len(draws) for num in range(1, pool_size + 1)
    }
    for draw_index, draw in enumerate(draws):
        for num in draw:
            if 1 <= num <= pool_size:
                sequences[num][draw_index] = 1
    return sequences


# ---------------------------------------------------------------------------
# Test 1: Frequency monobit (chi-square goodness of fit)
# ---------------------------------------------------------------------------

def _frequency_monobit(
    draws: list[list[int]],
    pool_size: int,
    significance: float,
) -> AnalysisResult:
    """Chi-square test: is each number's observed frequency consistent
    with the uniform expectation?"""
    n = len(draws)
    if n == 0:
        return _inconclusive("frequency_monobit", n, "No draws provided")

    # Expected count per number: n * (numbers_per_draw / pool_size)
    numbers_per_draw = len(draws[0]) if draws else 0
    expected = n * numbers_per_draw / pool_size

    if expected <= 0:
        return _inconclusive("frequency_monobit", n, "Expected frequency is zero")

    # Count observed frequency of each number
    observed = [0] * (pool_size + 1)
    for draw in draws:
        for num in draw:
            observed[num] += 1

    chi2 = sum(
        (observed[num] - expected) ** 2 / expected
        for num in range(1, pool_size + 1)
    )
    degrees_of_freedom = pool_size - 1
    p_value = _chi_square_survival(chi2, degrees_of_freedom)
    passed = p_value >= significance

    return AnalysisResult(
        test_name="frequency_monobit",
        game="all",
        passed=passed,
        p_value=p_value,
        statistic=float(chi2),
        threshold=significance,
        sample_size=n,
        details={
            "expected_per_number": round(expected, 4),
            "degrees_of_freedom": degrees_of_freedom,
        },
        summary=(
            f"Chi-square={chi2:.2f}, df={degrees_of_freedom}, "
            f"p={p_value:.4f} ({'pass' if passed else 'FAIL'})"
        ),
    )


# ---------------------------------------------------------------------------
# Test 2: Runs test
# ---------------------------------------------------------------------------

def _runs_test(
    draws: list[list[int]],
    pool_size: int,
    significance: float,
) -> AnalysisResult:
    """Per-number runs test: are consecutive appearance streaks consistent
    with random expectation?"""
    n = len(draws)
    if n < _MIN_DRAWS_RUNS:
        return _inconclusive(
            "runs", n, f"Need >= {_MIN_DRAWS_RUNS} draws, have {n}",
        )

    sequences = _build_binary_sequences(draws, pool_size)

    p_values: list[float] = []
    skipped = 0

    for num in range(1, pool_size + 1):
        bits = sequences[num]
        proportion = sum(bits) / n

        # Pre-test: skip if proportion too far from expected
        tau = 2.0 / math.sqrt(n)
        numbers_per_draw = len(draws[0]) if draws else 0
        expected_proportion = numbers_per_draw / pool_size

        if abs(proportion - expected_proportion) > tau:
            skipped += 1
            continue

        if proportion == 0.0 or proportion == 1.0:
            skipped += 1
            continue

        # Count runs
        run_count = 1
        for i in range(1, n):
            if bits[i] != bits[i - 1]:
                run_count += 1

        # Expected runs and variance under independence
        expected_runs = 1.0 + 2.0 * n * proportion * (1.0 - proportion)
        denominator = 2.0 * math.sqrt(2.0 * n) * proportion * (1.0 - proportion)

        if denominator == 0:
            skipped += 1
            continue

        z = abs(run_count - expected_runs) / denominator
        p_val = _erfc(z / math.sqrt(2.0))
        p_values.append(p_val)

    if not p_values:
        return _inconclusive("runs", n, "All numbers skipped in pre-test")

    # Combine: use the minimum p-value with Bonferroni-like interpretation
    min_p = min(p_values)
    median_p = sorted(p_values)[len(p_values) // 2]

    # Use median p-value for overall assessment (more robust than minimum)
    passed = median_p >= significance

    return AnalysisResult(
        test_name="runs",
        game="all",
        passed=passed,
        p_value=median_p,
        statistic=float(min_p),
        threshold=significance,
        sample_size=n,
        details={
            "numbers_tested": len(p_values),
            "numbers_skipped": skipped,
            "min_p_value": round(min_p, 6),
            "median_p_value": round(median_p, 6),
        },
        summary=(
            f"Runs test: median_p={median_p:.4f}, min_p={min_p:.4f}, "
            f"tested={len(p_values)}/{pool_size} "
            f"({'pass' if passed else 'FAIL'})"
        ),
    )


# ---------------------------------------------------------------------------
# Test 3: Longest run
# ---------------------------------------------------------------------------

def _longest_run_test(
    draws: list[list[int]],
    pool_size: int,
    significance: float,
) -> AnalysisResult:
    """Per-number longest-run test: are the longest appearance streaks
    consistent with random expectation?"""
    n = len(draws)
    if n < _MIN_DRAWS_LONGEST_RUN:
        return _inconclusive(
            "longest_run", n,
            f"Need >= {_MIN_DRAWS_LONGEST_RUN} draws, have {n}",
        )

    sequences = _build_binary_sequences(draws, pool_size)
    numbers_per_draw = len(draws[0]) if draws else 0
    expected_proportion = numbers_per_draw / pool_size

    longest_runs: list[int] = []
    for num in range(1, pool_size + 1):
        bits = sequences[num]
        max_run = 0
        current_run = 0
        for bit in bits:
            if bit == 1:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0
        longest_runs.append(max_run)

    # Under independence with probability p, expected longest run ~ log(n)/log(1/p)
    p = expected_proportion
    if p <= 0 or p >= 1:
        return _inconclusive("longest_run", n, "Degenerate proportion")

    expected_longest = math.log(n) / math.log(1.0 / p)

    # Standard deviation approximation for longest run
    # Using the Erdos-Renyi law approximation
    sigma_longest = 1.0 / math.log(1.0 / p)

    # Chi-square-like statistic: compare distribution of longest runs
    # across all numbers against expected
    chi2 = sum(
        (lr - expected_longest) ** 2 / max(expected_longest, 1.0)
        for lr in longest_runs
    )
    degrees_of_freedom = pool_size - 1
    p_value = _chi_square_survival(chi2, degrees_of_freedom)
    passed = p_value >= significance

    avg_longest = sum(longest_runs) / len(longest_runs) if longest_runs else 0

    return AnalysisResult(
        test_name="longest_run",
        game="all",
        passed=passed,
        p_value=p_value,
        statistic=float(chi2),
        threshold=significance,
        sample_size=n,
        details={
            "expected_longest_run": round(expected_longest, 2),
            "average_observed_longest": round(avg_longest, 2),
            "max_observed_longest": max(longest_runs) if longest_runs else 0,
            "degrees_of_freedom": degrees_of_freedom,
        },
        summary=(
            f"Longest run: expected~{expected_longest:.1f}, "
            f"avg_observed={avg_longest:.1f}, "
            f"chi2={chi2:.2f}, p={p_value:.4f} "
            f"({'pass' if passed else 'FAIL'})"
        ),
    )


# ---------------------------------------------------------------------------
# Test 4: Cumulative sums
# ---------------------------------------------------------------------------

def _cumulative_sums_test(
    draws: list[list[int]],
    pool_size: int,
    significance: float,
) -> AnalysisResult:
    """Cumulative sums test: detect persistent frequency drift over time."""
    n = len(draws)
    if n < _MIN_DRAWS_CUMULATIVE_SUMS:
        return _inconclusive(
            "cumulative_sums", n,
            f"Need >= {_MIN_DRAWS_CUMULATIVE_SUMS} draws, have {n}",
        )

    sequences = _build_binary_sequences(draws, pool_size)
    numbers_per_draw = len(draws[0]) if draws else 0
    expected_proportion = numbers_per_draw / pool_size

    p_values: list[float] = []

    for num in range(1, pool_size + 1):
        bits = sequences[num]

        # Convert to centered walk: +1 if appeared (above expectation), -1 if not
        # Centered around expected proportion
        walk = [0.0]
        for bit in bits:
            step = bit - expected_proportion
            walk.append(walk[-1] + step)

        # Maximum absolute excursion
        max_excursion = max(abs(w) for w in walk)

        if max_excursion == 0:
            p_values.append(1.0)
            continue

        # Under null hypothesis, the walk is a sum of iid centered variables
        # Variance of each step: p*(1-p)
        variance_per_step = expected_proportion * (1.0 - expected_proportion)
        if variance_per_step <= 0:
            continue

        # Normalized maximum: z = max_excursion / sqrt(n * var)
        z = max_excursion / math.sqrt(n * variance_per_step)

        # P-value from Kolmogorov-Smirnov-like distribution
        # Approximate: P(max|S_k/sqrt(n*var)| >= z) ~ 2*exp(-2*z^2)
        # for large n (Brownian motion reflection principle)
        p_val = 2.0 * math.exp(-2.0 * z * z) if z > 0 else 1.0
        p_val = min(p_val, 1.0)
        p_values.append(p_val)

    if not p_values:
        return _inconclusive("cumulative_sums", n, "No valid sequences")

    median_p = sorted(p_values)[len(p_values) // 2]
    min_p = min(p_values)
    passed = median_p >= significance

    return AnalysisResult(
        test_name="cumulative_sums",
        game="all",
        passed=passed,
        p_value=median_p,
        statistic=float(min_p),
        threshold=significance,
        sample_size=n,
        details={
            "numbers_tested": len(p_values),
            "min_p_value": round(min_p, 6),
            "median_p_value": round(median_p, 6),
        },
        summary=(
            f"Cumulative sums: median_p={median_p:.4f}, min_p={min_p:.4f}, "
            f"tested={len(p_values)}/{pool_size} "
            f"({'pass' if passed else 'FAIL'})"
        ),
    )


# ---------------------------------------------------------------------------
# Test 5: Serial test (pairwise sequential dependencies)
# ---------------------------------------------------------------------------

def _serial_test(
    draws: list[list[int]],
    pool_size: int,
    significance: float,
) -> AnalysisResult:
    """Serial test: detect pairwise sequential dependencies between
    consecutive draws."""
    n = len(draws)
    if n < _MIN_DRAWS_SERIAL:
        return _inconclusive(
            "serial", n,
            f"Need >= {_MIN_DRAWS_SERIAL} draws, have {n}",
        )

    sequences = _build_binary_sequences(draws, pool_size)

    p_values: list[float] = []

    for num in range(1, pool_size + 1):
        bits = sequences[num]

        # Count 2-bit overlapping patterns: 00, 01, 10, 11
        counts = [0, 0, 0, 0]
        for i in range(n - 1):
            pattern = bits[i] * 2 + bits[i + 1]
            counts[pattern] += 1

        total_pairs = n - 1
        if total_pairs == 0:
            continue

        # Count single-bit frequencies
        n1 = sum(bits)
        n0 = n - n1

        # Under independence: P(ab) = P(a)*P(b)
        p1 = n1 / n
        p0 = n0 / n

        expected = [
            total_pairs * p0 * p0,  # 00
            total_pairs * p0 * p1,  # 01
            total_pairs * p1 * p0,  # 10
            total_pairs * p1 * p1,  # 11
        ]

        # Chi-square for serial dependence
        chi2 = 0.0
        valid = True
        for j in range(4):
            if expected[j] < 1.0:
                valid = False
                break
            chi2 += (counts[j] - expected[j]) ** 2 / expected[j]

        if not valid:
            continue

        # 1 degree of freedom for 2x2 table of (bit_t, bit_{t+1})
        p_val = _chi_square_survival(chi2, 1)
        p_values.append(p_val)

    if not p_values:
        return _inconclusive("serial", n, "No valid sequences for serial test")

    median_p = sorted(p_values)[len(p_values) // 2]
    min_p = min(p_values)
    passed = median_p >= significance

    return AnalysisResult(
        test_name="serial",
        game="all",
        passed=passed,
        p_value=median_p,
        statistic=float(min_p),
        threshold=significance,
        sample_size=n,
        details={
            "numbers_tested": len(p_values),
            "min_p_value": round(min_p, 6),
            "median_p_value": round(median_p, 6),
        },
        summary=(
            f"Serial test: median_p={median_p:.4f}, min_p={min_p:.4f}, "
            f"tested={len(p_values)}/{pool_size} "
            f"({'pass' if passed else 'FAIL'})"
        ),
    )


# ---------------------------------------------------------------------------
# Test 6: Approximate entropy
# ---------------------------------------------------------------------------

def _approximate_entropy_test(
    draws: list[list[int]],
    pool_size: int,
    significance: float,
) -> AnalysisResult:
    """Approximate entropy test: detect regularity in per-number binary
    sequences by comparing observed pattern frequencies to those expected
    under independence."""
    n = len(draws)
    if n < _MIN_DRAWS_APPROXIMATE_ENTROPY:
        return _inconclusive(
            "approximate_entropy", n,
            f"Need >= {_MIN_DRAWS_APPROXIMATE_ENTROPY} draws, have {n}",
        )

    sequences = _build_binary_sequences(draws, pool_size)
    numbers_per_draw = len(draws[0]) if draws else 0
    expected_p = numbers_per_draw / pool_size

    # Use block size m=2 (appropriate for sequence lengths ~500-2000)
    m = 2

    p_values: list[float] = []

    for num in range(1, pool_size + 1):
        bits = sequences[num]

        # Observed proportion for this number
        obs_p = sum(bits) / n if n > 0 else expected_p
        if obs_p <= 0 or obs_p >= 1:
            continue

        phi_m = _compute_phi(bits, m, n)
        phi_m1 = _compute_phi(bits, m + 1, n)

        observed_apen = phi_m - phi_m1

        # Theoretical ApEn for iid Bernoulli(obs_p):
        # phi(m) for iid = m * H, where H = p*log(p) + (1-p)*log(1-p)
        # So ApEn_theory = phi(m) - phi(m+1) = -H
        h = obs_p * math.log(obs_p) + (1.0 - obs_p) * math.log(1.0 - obs_p)
        expected_apen = -h

        # Chi-square statistic comparing observed vs expected
        chi2 = 2.0 * n * (expected_apen - observed_apen)

        # Clamp to non-negative (numerical noise can make it slightly negative)
        chi2 = max(chi2, 0.0)

        # Degrees of freedom = 2^m
        df = 2 ** m
        p_val = _chi_square_survival(chi2, df)
        p_values.append(p_val)

    if not p_values:
        return _inconclusive(
            "approximate_entropy", n, "No valid sequences",
        )

    median_p = sorted(p_values)[len(p_values) // 2]
    min_p = min(p_values)
    passed = median_p >= significance

    return AnalysisResult(
        test_name="approximate_entropy",
        game="all",
        passed=passed,
        p_value=median_p,
        statistic=float(min_p),
        threshold=significance,
        sample_size=n,
        details={
            "block_size": m,
            "numbers_tested": len(p_values),
            "min_p_value": round(min_p, 6),
            "median_p_value": round(median_p, 6),
        },
        summary=(
            f"Approximate entropy (m={m}): median_p={median_p:.4f}, "
            f"min_p={min_p:.4f}, tested={len(p_values)}/{pool_size} "
            f"({'pass' if passed else 'FAIL'})"
        ),
    )


def _compute_phi(bits: list[int], block_len: int, n: int) -> float:
    """Compute the phi statistic for approximate entropy.

    Counts overlapping patterns of length block_len in the (circular)
    bit sequence and returns sum(c_i * log(c_i)) where c_i are the
    relative frequencies.
    """
    if block_len == 0:
        return 0.0

    # Extend for circularity
    extended = bits + bits[: block_len - 1]

    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        pattern = tuple(extended[i: i + block_len])
        counts[pattern] = counts.get(pattern, 0) + 1

    phi = 0.0
    for count in counts.values():
        frequency = count / n
        if frequency > 0:
            phi += frequency * math.log(frequency)

    return phi


# ---------------------------------------------------------------------------
# Inconclusive result helper
# ---------------------------------------------------------------------------

def _inconclusive(test_name: str, sample_size: int, reason: str) -> AnalysisResult:
    """Create an inconclusive AnalysisResult."""
    return AnalysisResult(
        test_name=test_name,
        game="all",
        passed=None,
        p_value=None,
        statistic=0.0,
        threshold=0.0,
        sample_size=sample_size,
        details={"reason": reason},
        summary=f"Inconclusive: {reason}",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_randomness_tests(
    draws: list[list[int]],
    pool_size: int,
    significance: float = 0.01,
) -> list[AnalysisResult]:
    """Run a subset of NIST SP 800-22 randomness tests on lottery draws.

    Each draw is a sorted list of main numbers. Tests convert draws to
    per-number binary sequences and assess uniformity and independence.

    Args:
        draws: Historical lottery draws (each a sorted list of ints).
        pool_size: Total numbers in the pool (e.g. 45 for Joker).
        significance: Alpha level for hypothesis testing (default 0.01).

    Returns:
        A list of 6 AnalysisResult objects, one per test.  Tests that
        require more data than available return passed=None (inconclusive).
    """
    return [
        _frequency_monobit(draws, pool_size, significance),
        _runs_test(draws, pool_size, significance),
        _longest_run_test(draws, pool_size, significance),
        _cumulative_sums_test(draws, pool_size, significance),
        _serial_test(draws, pool_size, significance),
        _approximate_entropy_test(draws, pool_size, significance),
    ]
