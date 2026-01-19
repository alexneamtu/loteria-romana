"""NIST SP 800-22 Statistical Test Suite for Randomness.

This module implements the 15 statistical tests from NIST Special Publication
800-22 Revision 1a for testing random and pseudorandom number generators.

Reference: https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

Note: These tests are designed for binary sequences. Lottery draws must be
converted to binary format before testing.
"""

import math
from dataclasses import dataclass
from typing import Callable


@dataclass
class TestResult:
    """Result from a single NIST test."""

    test_name: str
    p_value: float
    is_random: bool  # p_value >= significance_level
    statistic: float
    description: str


def _erfc(x: float) -> float:
    """Complementary error function approximation."""
    # Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return 1 - sign * y


def _igamc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function.

    Approximation using continued fraction expansion.
    """
    if x == 0:
        return 1.0
    if x < 0 or a <= 0:
        return 0.0

    if x < a + 1:
        # Use series expansion
        return 1.0 - _gammainc_series(a, x)
    else:
        # Use continued fraction
        return _gammainc_cf(a, x)


def _gammainc_series(a: float, x: float, max_iter: int = 200) -> float:
    """Incomplete gamma function using series expansion."""
    if x == 0:
        return 0.0

    ap = a
    sum_val = 1.0 / a
    delta = sum_val

    for _ in range(max_iter):
        ap += 1
        delta *= x / ap
        sum_val += delta
        if abs(delta) < abs(sum_val) * 1e-10:
            break

    return sum_val * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammainc_cf(a: float, x: float, max_iter: int = 200) -> float:
    """Incomplete gamma function using continued fraction."""
    b = x + 1 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d

    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-10:
            break

    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


class NISTTestSuite:
    """Complete NIST SP 800-22 randomness test suite.

    Usage:
        suite = NISTTestSuite(significance_level=0.01)
        results = suite.run_all(binary_sequence)

        for result in results:
            print(f"{result.test_name}: p={result.p_value:.4f}, random={result.is_random}")
    """

    def __init__(self, significance_level: float = 0.01):
        """Initialize test suite.

        Args:
            significance_level: Alpha level for hypothesis testing (default 0.01)
        """
        self.significance_level = significance_level

    def frequency_test(self, bits: list[int]) -> TestResult:
        """Test 1: Frequency (Monobit) Test.

        Tests the proportion of ones and zeros in the sequence.
        For a truly random sequence, this should be approximately equal.
        """
        n = len(bits)
        if n == 0:
            return TestResult(
                test_name="Frequency",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description="Empty sequence",
            )

        # Calculate S_n = sum of (2*bit - 1)
        s_n = sum(2 * b - 1 for b in bits)

        # Test statistic
        s_obs = abs(s_n) / math.sqrt(n)

        # P-value
        p_value = _erfc(s_obs / math.sqrt(2))

        return TestResult(
            test_name="Frequency",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=s_obs,
            description=f"Tests proportion of 1s and 0s. S_n={s_n}, n={n}",
        )

    def block_frequency_test(self, bits: list[int], block_size: int = 128) -> TestResult:
        """Test 2: Frequency Test within a Block.

        Tests the proportion of ones within M-bit blocks.
        """
        n = len(bits)
        m = block_size

        if n < m:
            return TestResult(
                test_name="Block Frequency",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Sequence too short (n={n} < M={m})",
            )

        # Number of blocks
        num_blocks = n // m

        # Calculate proportion of ones in each block
        chi_sq = 0.0
        for i in range(num_blocks):
            block = bits[i * m : (i + 1) * m]
            pi = sum(block) / m
            chi_sq += (pi - 0.5) ** 2

        chi_sq *= 4 * m

        # P-value using incomplete gamma function
        p_value = _igamc(num_blocks / 2, chi_sq / 2)

        return TestResult(
            test_name="Block Frequency",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=chi_sq,
            description=f"Tests proportion within {m}-bit blocks. Blocks={num_blocks}",
        )

    def runs_test(self, bits: list[int]) -> TestResult:
        """Test 3: Runs Test.

        Tests the total number of runs (uninterrupted sequences of identical bits).
        """
        n = len(bits)
        if n == 0:
            return TestResult(
                test_name="Runs",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description="Empty sequence",
            )

        # Pre-test: frequency
        pi = sum(bits) / n
        tau = 2 / math.sqrt(n)

        if abs(pi - 0.5) >= tau:
            return TestResult(
                test_name="Runs",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Failed frequency pre-test (pi={pi:.4f})",
            )

        # Count runs
        v_obs = 1
        for i in range(1, n):
            if bits[i] != bits[i - 1]:
                v_obs += 1

        # P-value
        numerator = abs(v_obs - 2 * n * pi * (1 - pi))
        denominator = 2 * math.sqrt(2 * n) * pi * (1 - pi)

        if denominator == 0:
            p_value = 0.0
        else:
            p_value = _erfc(numerator / denominator)

        return TestResult(
            test_name="Runs",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=v_obs,
            description=f"Tests number of runs. V_obs={v_obs}, pi={pi:.4f}",
        )

    def longest_run_test(self, bits: list[int]) -> TestResult:
        """Test 4: Longest Run of Ones in a Block.

        Tests the longest run of ones within M-bit blocks.
        """
        n = len(bits)

        # Determine parameters based on sequence length
        if n < 128:
            return TestResult(
                test_name="Longest Run",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Sequence too short (n={n} < 128)",
            )
        elif n < 6272:
            m, k = 8, 3
            pi = [0.2148, 0.3672, 0.2305, 0.1875]
            v_values = [1, 2, 3, 4]  # <=1, 2, 3, >=4
        elif n < 750000:
            m, k = 128, 5
            pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
            v_values = [4, 5, 6, 7, 8, 9]  # <=4, 5, 6, 7, 8, >=9
        else:
            m, k = 10000, 6
            pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
            v_values = [10, 11, 12, 13, 14, 15, 16]

        num_blocks = n // m

        # Count longest runs in each block
        v_counts = [0] * len(pi)

        for i in range(num_blocks):
            block = bits[i * m : (i + 1) * m]

            # Find longest run of ones
            max_run = 0
            current_run = 0
            for bit in block:
                if bit == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0

            # Categorize
            if n < 6272:
                if max_run <= 1:
                    v_counts[0] += 1
                elif max_run == 2:
                    v_counts[1] += 1
                elif max_run == 3:
                    v_counts[2] += 1
                else:
                    v_counts[3] += 1
            elif n < 750000:
                if max_run <= 4:
                    v_counts[0] += 1
                elif max_run <= 8:
                    v_counts[max_run - 4] += 1
                else:
                    v_counts[5] += 1
            else:
                if max_run <= 10:
                    v_counts[0] += 1
                elif max_run <= 15:
                    v_counts[max_run - 10] += 1
                else:
                    v_counts[6] += 1

        # Chi-square statistic
        chi_sq = sum(
            (v_counts[i] - num_blocks * pi[i]) ** 2 / (num_blocks * pi[i])
            for i in range(len(pi))
            if num_blocks * pi[i] > 0
        )

        p_value = _igamc(k / 2, chi_sq / 2)

        return TestResult(
            test_name="Longest Run",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=chi_sq,
            description=f"Tests longest run of 1s in {m}-bit blocks",
        )

    def matrix_rank_test(self, bits: list[int], m: int = 32, q: int = 32) -> TestResult:
        """Test 5: Binary Matrix Rank Test.

        Tests the rank of disjoint sub-matrices of the sequence.
        """
        n = len(bits)
        num_matrices = n // (m * q)

        if num_matrices < 38:
            return TestResult(
                test_name="Matrix Rank",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Too few matrices ({num_matrices} < 38)",
            )

        # Count matrices by rank
        fm = 0  # Full rank (m)
        fm1 = 0  # Rank m-1
        remaining = 0  # Rank < m-1

        for i in range(num_matrices):
            # Extract matrix
            start = i * m * q
            matrix = []
            for row in range(m):
                matrix.append(bits[start + row * q : start + (row + 1) * q])

            # Compute rank using Gaussian elimination
            rank = self._matrix_rank(matrix, m, q)

            if rank == m:
                fm += 1
            elif rank == m - 1:
                fm1 += 1
            else:
                remaining += 1

        # Expected probabilities for 32x32 matrices
        p_m = 0.2888  # P(rank = m)
        p_m1 = 0.5776  # P(rank = m-1)
        p_rest = 0.1336  # P(rank < m-1)

        # Chi-square statistic
        chi_sq = (
            (fm - num_matrices * p_m) ** 2 / (num_matrices * p_m)
            + (fm1 - num_matrices * p_m1) ** 2 / (num_matrices * p_m1)
            + (remaining - num_matrices * p_rest) ** 2 / (num_matrices * p_rest)
        )

        p_value = math.exp(-chi_sq / 2)

        return TestResult(
            test_name="Matrix Rank",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=chi_sq,
            description=f"Tests rank of {m}x{q} matrices. N={num_matrices}",
        )

    def _matrix_rank(self, matrix: list[list[int]], m: int, q: int) -> int:
        """Compute binary matrix rank using Gaussian elimination."""
        # Make a copy
        mat = [row[:] for row in matrix]
        rank = 0

        for col in range(min(m, q)):
            # Find pivot
            pivot_row = None
            for row in range(rank, m):
                if mat[row][col] == 1:
                    pivot_row = row
                    break

            if pivot_row is None:
                continue

            # Swap rows
            mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]

            # Eliminate
            for row in range(m):
                if row != rank and mat[row][col] == 1:
                    mat[row] = [(mat[row][j] + mat[rank][j]) % 2 for j in range(q)]

            rank += 1

        return rank

    def dft_test(self, bits: list[int]) -> TestResult:
        """Test 6: Discrete Fourier Transform (Spectral) Test.

        Tests for periodic features in the sequence.
        Uses NumPy FFT for efficiency if available, falls back to naive DFT.
        """
        n = len(bits)
        if n < 1000:
            return TestResult(
                test_name="DFT",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Sequence too short (n={n} < 1000)",
            )

        # Convert to +1/-1
        x = [2 * b - 1 for b in bits]

        half_n = n // 2
        threshold = math.sqrt(math.log(1 / 0.05) * n)

        # Count peaks above threshold
        n0 = 0.95 * half_n  # Expected number below threshold

        # Try NumPy FFT first (much faster)
        try:
            import numpy as np
            x_array = np.array(x)
            fft_result = np.fft.fft(x_array)
            magnitudes = np.abs(fft_result[:half_n])
            n1 = np.sum(magnitudes < threshold)
        except ImportError:
            # Fallback to naive DFT (slow for large sequences)
            n1 = 0
            for k in range(half_n):
                real_part = sum(x[j] * math.cos(2 * math.pi * k * j / n) for j in range(n))
                imag_part = sum(x[j] * math.sin(2 * math.pi * k * j / n) for j in range(n))
                magnitude = math.sqrt(real_part**2 + imag_part**2)
                if magnitude < threshold:
                    n1 += 1

        # Test statistic
        d = (n1 - n0) / math.sqrt(n * 0.95 * 0.05 / 4)
        p_value = _erfc(abs(d) / math.sqrt(2))

        return TestResult(
            test_name="DFT",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=d,
            description=f"Tests for periodic features. Peaks below threshold: {n1}/{half_n}",
        )

    def non_overlapping_template_test(
        self, bits: list[int], template: list[int] | None = None, block_size: int = 8
    ) -> TestResult:
        """Test 7: Non-overlapping Template Matching Test.

        Tests for occurrences of pre-specified target strings.
        """
        n = len(bits)
        if template is None:
            template = [0, 0, 0, 0, 0, 0, 0, 0, 1]  # Default template

        m = len(template)
        if n < 100:
            return TestResult(
                test_name="Non-overlapping Template",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Sequence too short (n={n} < 100)",
            )

        num_blocks = 8  # Standard: 8 blocks
        block_len = n // num_blocks

        # Count template occurrences in each block (non-overlapping)
        counts = []
        for i in range(num_blocks):
            block = bits[i * block_len : (i + 1) * block_len]
            count = 0
            j = 0
            while j <= len(block) - m:
                if block[j : j + m] == template:
                    count += 1
                    j += m  # Non-overlapping
                else:
                    j += 1
            counts.append(count)

        # Theoretical mean and variance
        mu = (block_len - m + 1) / (2**m)
        sigma_sq = block_len * (1 / (2**m) - (2 * m - 1) / (2 ** (2 * m)))

        # Chi-square statistic
        if sigma_sq > 0:
            chi_sq = sum((c - mu) ** 2 / sigma_sq for c in counts)
            p_value = _igamc(num_blocks / 2, chi_sq / 2)
        else:
            chi_sq = 0
            p_value = 1.0

        return TestResult(
            test_name="Non-overlapping Template",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=chi_sq,
            description=f"Tests for template occurrences. Mean count: {sum(counts)/len(counts):.2f}",
        )

    def overlapping_template_test(
        self, bits: list[int], template_len: int = 9, block_size: int = 1032
    ) -> TestResult:
        """Test 8: Overlapping Template Matching Test.

        Tests for occurrences of all-ones template using overlapping search.
        """
        n = len(bits)
        m = template_len
        template = [1] * m

        if n < 1000000:
            # Use smaller parameters for shorter sequences
            block_size = min(block_size, n // 10)

        num_blocks = n // block_size
        if num_blocks < 1:
            return TestResult(
                test_name="Overlapping Template",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Sequence too short",
            )

        # Count template occurrences in each block (overlapping)
        v = [0] * 6  # Categories: 0, 1, 2, 3, 4, >=5

        for i in range(num_blocks):
            block = bits[i * block_size : (i + 1) * block_size]
            count = 0
            for j in range(len(block) - m + 1):
                if block[j : j + m] == template:
                    count += 1

            if count <= 4:
                v[count] += 1
            else:
                v[5] += 1

        # Theoretical probabilities (for m=9, K=5)
        pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]

        # Chi-square
        chi_sq = sum(
            (v[i] - num_blocks * pi[i]) ** 2 / (num_blocks * pi[i])
            for i in range(6)
            if num_blocks * pi[i] > 0
        )

        p_value = _igamc(5 / 2, chi_sq / 2)

        return TestResult(
            test_name="Overlapping Template",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=chi_sq,
            description=f"Tests for all-ones template. Blocks: {num_blocks}",
        )

    def universal_test(self, bits: list[int], block_size: int = 7, init_blocks: int = 1280) -> TestResult:
        """Test 9: Maurer's Universal Statistical Test.

        Tests for compressibility of the sequence.
        """
        n = len(bits)
        l = block_size
        q = init_blocks

        if n < (q + 1000) * l:
            return TestResult(
                test_name="Universal",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Sequence too short for L={l}, Q={q}",
            )

        k = n // l - q  # Test blocks

        # Theoretical values for L=7
        expected_value = 6.1962507
        variance = 3.125

        # Initialize table
        table = {}
        for i in range(q):
            block_val = 0
            for j in range(l):
                block_val = block_val * 2 + bits[i * l + j]
            table[block_val] = i + 1

        # Compute sum
        total = 0.0
        for i in range(q, q + k):
            block_val = 0
            for j in range(l):
                block_val = block_val * 2 + bits[i * l + j]

            if block_val in table:
                total += math.log2(i + 1 - table[block_val])
            table[block_val] = i + 1

        fn = total / k

        # P-value
        c = 0.7 - 0.8 / l + (4 + 32 / l) * (k ** (-3 / l)) / 15
        sigma = c * math.sqrt(variance / k)

        p_value = _erfc(abs(fn - expected_value) / (math.sqrt(2) * sigma))

        return TestResult(
            test_name="Universal",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=fn,
            description=f"Tests compressibility. f_n={fn:.4f}, expected={expected_value:.4f}",
        )

    def linear_complexity_test(self, bits: list[int], block_size: int = 500) -> TestResult:
        """Test 10: Linear Complexity Test.

        Tests the linear complexity of the sequence using LFSR.
        """
        n = len(bits)
        m = block_size
        num_blocks = n // m

        if num_blocks < 1:
            return TestResult(
                test_name="Linear Complexity",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Sequence too short for M={m}",
            )

        # Expected mean
        mu = m / 2 + (9 + (-1) ** (m + 1)) / 36 - (m / 3 + 2 / 9) / (2**m)

        # Compute linear complexity for each block
        v = [0] * 7  # Categories for T values

        for i in range(num_blocks):
            block = bits[i * m : (i + 1) * m]
            lc = self._berlekamp_massey(block)

            # T value
            t = (-1) ** m * (lc - mu) + 2 / 9

            # Categorize
            if t <= -2.5:
                v[0] += 1
            elif t <= -1.5:
                v[1] += 1
            elif t <= -0.5:
                v[2] += 1
            elif t <= 0.5:
                v[3] += 1
            elif t <= 1.5:
                v[4] += 1
            elif t <= 2.5:
                v[5] += 1
            else:
                v[6] += 1

        # Theoretical probabilities
        pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]

        # Chi-square
        chi_sq = sum(
            (v[i] - num_blocks * pi[i]) ** 2 / (num_blocks * pi[i])
            for i in range(7)
            if num_blocks * pi[i] > 0
        )

        p_value = _igamc(6 / 2, chi_sq / 2)

        return TestResult(
            test_name="Linear Complexity",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=chi_sq,
            description=f"Tests linear complexity. Blocks: {num_blocks}",
        )

    def _berlekamp_massey(self, bits: list[int]) -> int:
        """Berlekamp-Massey algorithm for linear complexity."""
        n = len(bits)
        c = [0] * n
        b = [0] * n
        c[0] = 1
        b[0] = 1
        l = 0
        m = -1

        for i in range(n):
            # Compute discrepancy
            d = bits[i]
            for j in range(1, l + 1):
                d ^= c[j] & bits[i - j]

            if d == 1:
                t = c[:]
                for j in range(n - i + m):
                    if i - m + j < n:
                        c[i - m + j] ^= b[j]

                if 2 * l <= i:
                    l = i + 1 - l
                    m = i
                    b = t

        return l

    def serial_test(self, bits: list[int], block_size: int = 16) -> TestResult:
        """Test 11: Serial Test.

        Tests the frequency of all possible overlapping m-bit patterns.
        """
        n = len(bits)
        m = block_size

        if n < 100:
            return TestResult(
                test_name="Serial",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Sequence too short",
            )

        # Extend sequence for circularity
        extended = bits + bits[: m - 1]

        def count_patterns(length: int) -> dict[tuple, int]:
            counts: dict[tuple, int] = {}
            for i in range(n):
                pattern = tuple(extended[i : i + length])
                counts[pattern] = counts.get(pattern, 0) + 1
            return counts

        # Compute psi-squared values
        def psi_sq(length: int) -> float:
            if length == 0:
                return 0.0
            counts = count_patterns(length)
            return (2**length / n) * sum(c**2 for c in counts.values()) - n

        psi_m = psi_sq(m)
        psi_m1 = psi_sq(m - 1)
        psi_m2 = psi_sq(m - 2) if m >= 2 else 0

        del_psi = psi_m - psi_m1
        del_psi_2 = psi_m - 2 * psi_m1 + psi_m2

        p_value_1 = _igamc(2 ** (m - 2), del_psi / 2)
        p_value_2 = _igamc(2 ** (m - 3), del_psi_2 / 2)

        # Return the minimum p-value
        p_value = min(p_value_1, p_value_2)

        return TestResult(
            test_name="Serial",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=del_psi,
            description=f"Tests frequency of {m}-bit patterns",
        )

    def approximate_entropy_test(self, bits: list[int], block_size: int = 10) -> TestResult:
        """Test 12: Approximate Entropy Test.

        Tests the frequency of overlapping patterns.
        """
        n = len(bits)
        m = block_size

        if n < 100:
            return TestResult(
                test_name="Approximate Entropy",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Sequence too short",
            )

        def phi(length: int) -> float:
            if length == 0:
                return 0.0

            extended = bits + bits[: length - 1]
            counts: dict[tuple, int] = {}

            for i in range(n):
                pattern = tuple(extended[i : i + length])
                counts[pattern] = counts.get(pattern, 0) + 1

            c = {k: v / n for k, v in counts.items()}
            return sum(c[k] * math.log(c[k]) for k in c if c[k] > 0)

        phi_m = phi(m)
        phi_m1 = phi(m + 1)

        ap_en = phi_m - phi_m1
        chi_sq = 2 * n * (math.log(2) - ap_en)

        p_value = _igamc(2 ** (m - 1), chi_sq / 2)

        return TestResult(
            test_name="Approximate Entropy",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=chi_sq,
            description=f"Tests overlapping patterns. ApEn={ap_en:.4f}",
        )

    def cumulative_sums_test(self, bits: list[int], mode: str = "forward") -> TestResult:
        """Test 13: Cumulative Sums (Cusums) Test.

        Tests the maximal excursion of the random walk defined by cumulative sum.
        """
        n = len(bits)
        if n < 100:
            return TestResult(
                test_name="Cumulative Sums",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Sequence too short",
            )

        # Convert to +1/-1
        x = [2 * b - 1 for b in bits]
        if mode == "backward":
            x = x[::-1]

        # Compute cumulative sums
        s = [0]
        for xi in x:
            s.append(s[-1] + xi)

        z = max(abs(si) for si in s)

        # P-value calculation
        sum1 = 0.0
        sum2 = 0.0

        for k in range(int((-n / z + 1) / 4), int((n / z - 1) / 4) + 1):
            sum1 += _normal_cdf((4 * k + 1) * z / math.sqrt(n)) - _normal_cdf(
                (4 * k - 1) * z / math.sqrt(n)
            )

        for k in range(int((-n / z - 3) / 4), int((n / z - 1) / 4) + 1):
            sum2 += _normal_cdf((4 * k + 3) * z / math.sqrt(n)) - _normal_cdf(
                (4 * k + 1) * z / math.sqrt(n)
            )

        p_value = 1 - sum1 + sum2

        return TestResult(
            test_name=f"Cumulative Sums ({mode})",
            p_value=p_value,
            is_random=p_value >= self.significance_level,
            statistic=z,
            description=f"Tests cumulative sum excursion. z={z}",
        )

    def random_excursions_test(self, bits: list[int]) -> TestResult:
        """Test 14: Random Excursions Test.

        Tests the number of cycles having exactly K visits.
        """
        n = len(bits)
        if n < 1000000:
            # Relaxed version for shorter sequences
            pass

        # Convert to +1/-1 and compute cumulative sum
        x = [2 * b - 1 for b in bits]
        s = [0]
        for xi in x:
            s.append(s[-1] + xi)

        # Find cycles (returns to zero)
        cycles = []
        start = 0
        for i in range(1, len(s)):
            if s[i] == 0:
                if i > start:
                    cycles.append(s[start:i])
                start = i

        j = len(cycles)
        if j < 500:
            return TestResult(
                test_name="Random Excursions",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Too few cycles ({j} < 500)",
            )

        # Test states -4 to 4 (excluding 0)
        states = [-4, -3, -2, -1, 1, 2, 3, 4]
        min_p = 1.0

        for state in states:
            # Count visits to this state in each cycle
            v = [0] * 6  # 0, 1, 2, 3, 4, >=5 visits

            for cycle in cycles:
                count = sum(1 for si in cycle if si == state)
                if count <= 4:
                    v[count] += 1
                else:
                    v[5] += 1

            # Theoretical probabilities
            x_abs = abs(state)
            pi = [
                1 - 1 / (2 * x_abs),
                1 / (4 * x_abs**2),
                1 / (4 * x_abs**2) * (1 - 1 / (2 * x_abs)),
                1 / (4 * x_abs**2) * (1 - 1 / (2 * x_abs)) ** 2,
                1 / (4 * x_abs**2) * (1 - 1 / (2 * x_abs)) ** 3,
                1 / (4 * x_abs**2) * (1 - 1 / (2 * x_abs)) ** 4,
            ]

            # Chi-square
            chi_sq = sum(
                (v[k] - j * pi[k]) ** 2 / (j * pi[k]) for k in range(6) if j * pi[k] > 0
            )

            p_value = _igamc(5 / 2, chi_sq / 2)
            min_p = min(min_p, p_value)

        return TestResult(
            test_name="Random Excursions",
            p_value=min_p,
            is_random=min_p >= self.significance_level,
            statistic=j,
            description=f"Tests cycle visit counts. Cycles: {j}",
        )

    def random_excursions_variant_test(self, bits: list[int]) -> TestResult:
        """Test 15: Random Excursions Variant Test.

        Tests the total number of visits to various states.
        """
        n = len(bits)

        # Convert to +1/-1 and compute cumulative sum
        x = [2 * b - 1 for b in bits]
        s = [0]
        for xi in x:
            s.append(s[-1] + xi)

        # Count zero crossings
        j = sum(1 for i in range(1, len(s)) if s[i] == 0)

        if j < 500:
            return TestResult(
                test_name="Random Excursions Variant",
                p_value=0.0,
                is_random=False,
                statistic=0.0,
                description=f"Too few zero crossings ({j} < 500)",
            )

        # Test states -9 to 9 (excluding 0)
        states = list(range(-9, 0)) + list(range(1, 10))
        min_p = 1.0

        for state in states:
            count = sum(1 for si in s if si == state)
            p_value = _erfc(abs(count - j) / math.sqrt(2 * j * (4 * abs(state) - 2)))
            min_p = min(min_p, p_value)

        return TestResult(
            test_name="Random Excursions Variant",
            p_value=min_p,
            is_random=min_p >= self.significance_level,
            statistic=j,
            description=f"Tests state visit totals. Zero crossings: {j}",
        )

    def run_all(self, bits: list[int]) -> list[TestResult]:
        """Run all 15 NIST tests on the sequence.

        Args:
            bits: Binary sequence (list of 0s and 1s)

        Returns:
            List of TestResult objects for all tests
        """
        results = []

        # Core tests
        results.append(self.frequency_test(bits))
        results.append(self.block_frequency_test(bits))
        results.append(self.runs_test(bits))
        results.append(self.longest_run_test(bits))

        # Matrix test (needs sufficient data)
        if len(bits) >= 32 * 32 * 38:
            results.append(self.matrix_rank_test(bits))

        # Spectral test
        if len(bits) >= 1000:
            results.append(self.dft_test(bits))

        # Template tests
        results.append(self.non_overlapping_template_test(bits))
        results.append(self.overlapping_template_test(bits))

        # Universal test
        if len(bits) >= 387840:
            results.append(self.universal_test(bits))

        # Complexity test
        if len(bits) >= 500:
            results.append(self.linear_complexity_test(bits))

        # Pattern tests
        results.append(self.serial_test(bits))
        results.append(self.approximate_entropy_test(bits))

        # Cumulative sums (both directions)
        results.append(self.cumulative_sums_test(bits, "forward"))
        results.append(self.cumulative_sums_test(bits, "backward"))

        # Random excursions (need many zero crossings)
        results.append(self.random_excursions_test(bits))
        results.append(self.random_excursions_variant_test(bits))

        return results


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# Utility functions for lottery data conversion


def lottery_draws_to_bits(
    draws: list[list[int]], pool_size: int, encoding: str = "presence"
) -> list[int]:
    """Convert lottery draws to binary sequence for NIST testing.

    Args:
        draws: List of lottery draws (each draw is list of numbers)
        pool_size: Total numbers in the pool (e.g., 49 for 6/49)
        encoding: Encoding method:
            - "presence": 1 if number present, 0 if not (pool_size bits per draw)
            - "binary": Binary representation of each number
            - "difference": Binary encoding of differences between numbers

    Returns:
        Binary sequence (list of 0s and 1s)
    """
    bits = []

    if encoding == "presence":
        for draw in draws:
            draw_set = set(draw)
            for n in range(1, pool_size + 1):
                bits.append(1 if n in draw_set else 0)

    elif encoding == "binary":
        num_bits = (pool_size - 1).bit_length()
        for draw in draws:
            for num in sorted(draw):
                for i in range(num_bits - 1, -1, -1):
                    bits.append((num >> i) & 1)

    elif encoding == "difference":
        for draw in draws:
            sorted_draw = sorted(draw)
            prev = 0
            for num in sorted_draw:
                diff = num - prev
                num_bits = max(1, diff.bit_length())
                for i in range(num_bits - 1, -1, -1):
                    bits.append((diff >> i) & 1)
                prev = num

    return bits


def run_nist_analysis(
    draws: list[list[int]], pool_size: int, significance_level: float = 0.01
) -> dict:
    """Run complete NIST analysis on lottery draws.

    Args:
        draws: Historical lottery draws
        pool_size: Total numbers in pool
        significance_level: Alpha level for tests

    Returns:
        Dictionary with analysis results
    """
    suite = NISTTestSuite(significance_level)

    results = {}
    encodings = ["presence", "binary", "difference"]

    for encoding in encodings:
        bits = lottery_draws_to_bits(draws, pool_size, encoding)
        test_results = suite.run_all(bits)

        results[encoding] = {
            "bits_generated": len(bits),
            "tests_run": len(test_results),
            "tests_passed": sum(1 for r in test_results if r.is_random),
            "tests_failed": sum(1 for r in test_results if not r.is_random),
            "details": [
                {
                    "test": r.test_name,
                    "p_value": r.p_value,
                    "passed": r.is_random,
                    "statistic": r.statistic,
                }
                for r in test_results
            ],
        }

    # Overall summary
    total_tests = sum(r["tests_run"] for r in results.values())
    total_passed = sum(r["tests_passed"] for r in results.values())

    results["summary"] = {
        "total_draws": len(draws),
        "total_tests": total_tests,
        "total_passed": total_passed,
        "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
        "conclusion": (
            "RANDOM: All tests passed at significance level"
            if total_passed == total_tests
            else f"INVESTIGATE: {total_tests - total_passed} tests failed"
        ),
    }

    return results
