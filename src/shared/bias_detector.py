"""
Comprehensive Statistical Bias Detection for Lottery Draws.

This module implements rigorous statistical tests to detect any potential
biases in lottery draw data. While lotteries are designed to be random,
these tests can identify:
- Machine/ball degradation over time
- Procedural anomalies
- Drawing mechanism biases
- Temporal patterns

All tests use conservative significance levels with Bonferroni correction
for multiple hypothesis testing.
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import Counter
import math


@dataclass
class BiasReport:
    """Result of a single bias test."""
    test_name: str
    description: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: Optional[float] = None
    details: dict = field(default_factory=dict)
    recommendation: str = ""

    def __str__(self) -> str:
        status = "⚠️ SIGNIFICANT" if self.is_significant else "✓ Pass"
        return f"{self.test_name}: {status} (p={self.p_value:.4f})"


class BiasDetector:
    """
    Comprehensive bias detection for lottery draws.

    Implements multiple statistical tests to detect potential biases
    in lottery number generation. Uses Bonferroni correction to control
    for family-wise error rate when running multiple tests.

    Example:
        detector = BiasDetector(significance_level=0.01)
        draws = [[1, 5, 12, 23, 34, 45], [3, 7, 19, 28, 33, 41], ...]
        reports = detector.full_report(draws, pool_size=49, dates=dates)
        for report in reports:
            if report.is_significant:
                print(f"Potential bias: {report.test_name}")
    """

    def __init__(self, significance_level: float = 0.01):
        """
        Initialize the bias detector.

        Args:
            significance_level: Base alpha level before Bonferroni correction.
                               Default 0.01 for conservative testing.
        """
        self.significance_level = significance_level
        self.num_tests = 8  # Number of tests for Bonferroni correction
        self.adjusted_alpha = significance_level / self.num_tests

    def frequency_uniformity(self, draws: list[list[int]], pool_size: int) -> BiasReport:
        """
        Test if all numbers appear with equal frequency.

        Uses chi-square goodness-of-fit test against uniform distribution.

        Args:
            draws: List of draws, each draw is a list of numbers
            pool_size: Total numbers in the pool (e.g., 49 for Loto 6/49)

        Returns:
            BiasReport with chi-square statistic and significance
        """
        # Count frequency of each number
        counts = Counter()
        total_numbers = 0
        for draw in draws:
            for num in draw:
                counts[num] += 1
                total_numbers += 1

        # Expected frequency under uniform distribution
        expected = total_numbers / pool_size

        # Chi-square calculation
        chi_square = 0.0
        observed_list = []
        for num in range(1, pool_size + 1):
            observed = counts.get(num, 0)
            observed_list.append(observed)
            chi_square += (observed - expected) ** 2 / expected

        # Degrees of freedom
        df = pool_size - 1

        # P-value from chi-square distribution
        p_value = self._chi_square_p_value(chi_square, df)

        # Effect size (Cramér's V)
        effect_size = math.sqrt(chi_square / (total_numbers * (pool_size - 1)))

        # Find most and least frequent numbers
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        most_frequent = sorted_counts[:5] if sorted_counts else []
        least_frequent = sorted_counts[-5:] if len(sorted_counts) >= 5 else sorted_counts

        return BiasReport(
            test_name="Frequency Uniformity",
            description="Tests if all numbers appear with equal frequency",
            statistic=chi_square,
            p_value=p_value,
            is_significant=p_value < self.adjusted_alpha,
            effect_size=effect_size,
            details={
                "expected_frequency": expected,
                "most_frequent": most_frequent,
                "least_frequent": least_frequent,
                "total_numbers": total_numbers,
                "degrees_of_freedom": df
            },
            recommendation="If significant, investigate physical drawing mechanism"
        )

    def position_uniformity(self, draws: list[list[int]], pool_size: int) -> BiasReport:
        """
        Test if numbers are uniformly distributed across positions.

        For sorted draws, tests if each position has uniform distribution.
        Detects if certain numbers tend to appear in specific positions.

        Args:
            draws: List of draws (assumed sorted within each draw)
            pool_size: Total numbers in the pool

        Returns:
            BiasReport with aggregated chi-square across positions
        """
        if not draws:
            return BiasReport(
                test_name="Position Uniformity",
                description="Tests if numbers are uniform across positions",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                details={"error": "No draws provided"}
            )

        num_positions = len(draws[0])
        position_counts = [Counter() for _ in range(num_positions)]

        for draw in draws:
            sorted_draw = sorted(draw)
            for pos, num in enumerate(sorted_draw):
                position_counts[pos][num] += 1

        # Aggregate chi-square across positions
        total_chi_square = 0.0
        total_df = 0
        position_stats = []

        for pos in range(num_positions):
            counts = position_counts[pos]
            n_draws = len(draws)

            # For each position, calculate chi-square against theoretical distribution
            # In sorted draws, position i should have different expected distribution
            chi_sq = 0.0
            observed_nums = sum(counts.values())
            unique_nums = len(counts)

            if unique_nums > 1 and observed_nums > 0:
                expected = observed_nums / unique_nums
                for count in counts.values():
                    chi_sq += (count - expected) ** 2 / expected

                df = unique_nums - 1
                total_chi_square += chi_sq
                total_df += df

                position_stats.append({
                    "position": pos + 1,
                    "chi_square": chi_sq,
                    "unique_numbers": unique_nums
                })

        p_value = self._chi_square_p_value(total_chi_square, total_df) if total_df > 0 else 1.0

        return BiasReport(
            test_name="Position Uniformity",
            description="Tests if numbers are uniformly distributed across sorted positions",
            statistic=total_chi_square,
            p_value=p_value,
            is_significant=p_value < self.adjusted_alpha,
            details={
                "position_stats": position_stats,
                "total_degrees_of_freedom": total_df
            },
            recommendation="If significant, certain balls may have physical biases"
        )

    def temporal_correlation(self, draws: list[list[int]]) -> BiasReport:
        """
        Test for serial correlation between consecutive draws.

        Uses Durbin-Watson-like statistic adapted for lottery draws.
        Checks if knowing the previous draw helps predict the next.

        Args:
            draws: List of draws in chronological order

        Returns:
            BiasReport with correlation statistic
        """
        if len(draws) < 10:
            return BiasReport(
                test_name="Temporal Correlation",
                description="Tests for patterns between consecutive draws",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                details={"error": "Insufficient draws for temporal analysis"}
            )

        # Calculate overlap between consecutive draws
        overlaps = []
        for i in range(1, len(draws)):
            prev_set = set(draws[i-1])
            curr_set = set(draws[i])
            overlap = len(prev_set & curr_set)
            overlaps.append(overlap)

        # Calculate statistics
        mean_overlap = sum(overlaps) / len(overlaps)
        variance = sum((o - mean_overlap) ** 2 for o in overlaps) / len(overlaps)
        std_dev = math.sqrt(variance) if variance > 0 else 0.001

        # Expected overlap under independence (hypergeometric)
        # For 6 from 49: E[overlap] ≈ 6*6/49 ≈ 0.735
        draw_size = len(draws[0])
        pool_estimate = max(draws[0]) if draws else 49  # Estimate pool size
        expected_overlap = (draw_size * draw_size) / pool_estimate

        # Z-score
        z_score = (mean_overlap - expected_overlap) / (std_dev / math.sqrt(len(overlaps)))

        # Two-tailed p-value
        p_value = 2 * (1 - self._standard_normal_cdf(abs(z_score)))

        # Autocorrelation of overlaps
        autocorr = self._autocorrelation(overlaps, lag=1)

        return BiasReport(
            test_name="Temporal Correlation",
            description="Tests for serial correlation between consecutive draws",
            statistic=z_score,
            p_value=p_value,
            is_significant=p_value < self.adjusted_alpha,
            effect_size=autocorr,
            details={
                "mean_overlap": mean_overlap,
                "expected_overlap": expected_overlap,
                "std_deviation": std_dev,
                "lag1_autocorrelation": autocorr,
                "num_comparisons": len(overlaps)
            },
            recommendation="If significant, draw mechanism may have memory effects"
        )

    def day_of_week_bias(self, draws: list[list[int]],
                         dates: list[tuple[int, int, int]]) -> BiasReport:
        """
        Test if draw outcomes vary by day of week.

        Some lotteries are drawn on specific days; this tests if
        the day affects the distribution of numbers.

        Args:
            draws: List of draws
            dates: List of (year, month, day) tuples corresponding to draws

        Returns:
            BiasReport with day-of-week analysis
        """
        if len(draws) != len(dates):
            return BiasReport(
                test_name="Day of Week Bias",
                description="Tests if outcomes vary by drawing day",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                details={"error": "Draws and dates length mismatch"}
            )

        # Group draws by day of week
        day_sums = {i: [] for i in range(7)}  # 0=Monday, 6=Sunday

        for draw, date in zip(draws, dates):
            try:
                # Calculate day of week using Zeller's formula (simplified)
                year, month, day = date
                dow = self._day_of_week(year, month, day)
                day_sums[dow].append(sum(draw))
            except (ValueError, TypeError):
                continue

        # Filter out days with no draws
        active_days = {d: sums for d, sums in day_sums.items() if sums}

        if len(active_days) < 2:
            return BiasReport(
                test_name="Day of Week Bias",
                description="Tests if outcomes vary by drawing day",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                details={"error": "Draws on fewer than 2 different days"}
            )

        # Kruskal-Wallis H test (non-parametric ANOVA)
        all_sums = []
        group_sizes = []
        for sums in active_days.values():
            all_sums.extend(sums)
            group_sizes.append(len(sums))

        n_total = len(all_sums)

        # Rank all values
        sorted_with_idx = sorted(enumerate(all_sums), key=lambda x: x[1])
        ranks = [0] * n_total
        i = 0
        while i < n_total:
            j = i
            while j < n_total and sorted_with_idx[j][1] == sorted_with_idx[i][1]:
                j += 1
            avg_rank = (i + j + 1) / 2  # Average rank for ties
            for k in range(i, j):
                ranks[sorted_with_idx[k][0]] = avg_rank
            i = j

        # Calculate H statistic
        h_stat = 0.0
        idx = 0
        for size in group_sizes:
            group_ranks = ranks[idx:idx + size]
            rank_sum = sum(group_ranks)
            h_stat += (rank_sum ** 2) / size
            idx += size

        h_stat = (12 / (n_total * (n_total + 1))) * h_stat - 3 * (n_total + 1)

        # P-value from chi-square with k-1 degrees of freedom
        df = len(active_days) - 1
        p_value = self._chi_square_p_value(h_stat, df)

        # Day names
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
        day_stats = {day_names[d]: {"count": len(sums), "mean_sum": sum(sums)/len(sums)}
                     for d, sums in active_days.items()}

        return BiasReport(
            test_name="Day of Week Bias",
            description="Tests if draw outcomes vary by day of week",
            statistic=h_stat,
            p_value=p_value,
            is_significant=p_value < self.adjusted_alpha,
            details={
                "day_statistics": day_stats,
                "degrees_of_freedom": df,
                "total_draws": n_total
            },
            recommendation="If significant, investigate drawing schedule/conditions"
        )

    def consecutive_pairs(self, draws: list[list[int]], pool_size: int) -> BiasReport:
        """
        Test if consecutive number pairs appear with expected frequency.

        Under randomness, consecutive pairs (like 23-24) should appear
        with predictable frequency based on combinatorics.

        Args:
            draws: List of draws
            pool_size: Total numbers in the pool

        Returns:
            BiasReport with consecutive pair analysis
        """
        # Count consecutive pairs
        pair_count = 0
        total_possible_pairs = 0

        for draw in draws:
            sorted_draw = sorted(draw)
            for i in range(len(sorted_draw) - 1):
                total_possible_pairs += 1
                if sorted_draw[i+1] - sorted_draw[i] == 1:
                    pair_count += 1

        if total_possible_pairs == 0:
            return BiasReport(
                test_name="Consecutive Pairs",
                description="Tests if consecutive pairs appear at expected rate",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                details={"error": "No draws provided"}
            )

        # Expected probability of consecutive pair
        # For 6 numbers from 49, each adjacent position has prob ≈ 1/(49-k) of consecutive
        draw_size = len(draws[0])

        # Simplified expected rate
        # P(consecutive) for one pair ≈ (pool_size - 1) / C(pool_size, 2)
        expected_rate = (pool_size - 1) / ((pool_size * (pool_size - 1)) / 2)
        expected_pairs = total_possible_pairs * expected_rate

        # Binomial test approximation using normal
        observed_rate = pair_count / total_possible_pairs
        std_error = math.sqrt(expected_rate * (1 - expected_rate) / total_possible_pairs)

        if std_error > 0:
            z_score = (observed_rate - expected_rate) / std_error
            p_value = 2 * (1 - self._standard_normal_cdf(abs(z_score)))
        else:
            z_score = 0.0
            p_value = 1.0

        return BiasReport(
            test_name="Consecutive Pairs",
            description="Tests if consecutive number pairs appear at expected rate",
            statistic=z_score,
            p_value=p_value,
            is_significant=p_value < self.adjusted_alpha,
            details={
                "observed_pairs": pair_count,
                "expected_pairs": expected_pairs,
                "observed_rate": observed_rate,
                "expected_rate": expected_rate,
                "total_adjacent_positions": total_possible_pairs
            },
            recommendation="If significant, ball mixing may be inadequate"
        )

    def sum_distribution(self, draws: list[list[int]], pool_size: int) -> BiasReport:
        """
        Test if draw sums follow expected distribution.

        The sum of drawn numbers should follow a specific distribution
        based on hypergeometric sampling.

        Args:
            draws: List of draws
            pool_size: Total numbers in the pool

        Returns:
            BiasReport with sum distribution analysis
        """
        if not draws:
            return BiasReport(
                test_name="Sum Distribution",
                description="Tests if draw sums follow expected distribution",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                details={"error": "No draws provided"}
            )

        draw_size = len(draws[0])
        sums = [sum(draw) for draw in draws]

        # Calculate observed statistics
        observed_mean = sum(sums) / len(sums)
        observed_variance = sum((s - observed_mean) ** 2 for s in sums) / len(sums)
        observed_std = math.sqrt(observed_variance) if observed_variance > 0 else 0

        # Expected mean and variance for sum of k numbers from 1 to n without replacement
        # E[sum] = k * (n+1) / 2
        # Var[sum] = k * (n+1) * (n-k) / 12
        expected_mean = draw_size * (pool_size + 1) / 2
        expected_variance = draw_size * (pool_size + 1) * (pool_size - draw_size) / 12
        expected_std = math.sqrt(expected_variance)

        # Z-test for mean
        std_error = expected_std / math.sqrt(len(sums))
        z_score = (observed_mean - expected_mean) / std_error if std_error > 0 else 0
        p_value = 2 * (1 - self._standard_normal_cdf(abs(z_score)))

        # Shapiro-Wilk style normality indicator (simplified)
        # Bin the sums and check distribution shape
        min_sum = draw_size * (draw_size + 1) // 2  # Minimum possible sum
        max_sum = sum(range(pool_size - draw_size + 1, pool_size + 1))

        # Create histogram
        num_bins = min(20, len(draws) // 10 + 1)
        bin_width = (max_sum - min_sum) / num_bins
        bins = [0] * num_bins

        for s in sums:
            bin_idx = min(int((s - min_sum) / bin_width), num_bins - 1)
            bins[bin_idx] += 1

        return BiasReport(
            test_name="Sum Distribution",
            description="Tests if draw sums follow expected statistical distribution",
            statistic=z_score,
            p_value=p_value,
            is_significant=p_value < self.adjusted_alpha,
            details={
                "observed_mean": observed_mean,
                "expected_mean": expected_mean,
                "observed_std": observed_std,
                "expected_std": expected_std,
                "min_sum": min(sums),
                "max_sum": max(sums),
                "histogram_bins": bins
            },
            recommendation="If significant, draw mechanism may favor certain ranges"
        )

    def change_point_detection(self, draws: list[list[int]]) -> BiasReport:
        """
        Detect if there are significant changes in draw patterns over time.

        Uses CUSUM-like approach to detect regime changes that might
        indicate equipment changes or procedural modifications.

        Args:
            draws: List of draws in chronological order

        Returns:
            BiasReport with change point analysis
        """
        if len(draws) < 50:
            return BiasReport(
                test_name="Change Point Detection",
                description="Detects significant changes in draw patterns over time",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                details={"error": "Insufficient draws for change point analysis"}
            )

        # Use draw sums as the signal
        sums = [sum(draw) for draw in draws]
        n = len(sums)

        # Calculate overall mean
        overall_mean = sum(sums) / n

        # CUSUM calculation
        cusum = [0.0]
        for s in sums:
            cusum.append(cusum[-1] + (s - overall_mean))

        # Find maximum deviation from expected path
        max_dev = 0.0
        max_dev_idx = 0

        for i in range(1, n + 1):
            expected = (i / n) * cusum[n]
            deviation = abs(cusum[i] - expected)
            if deviation > max_dev:
                max_dev = deviation
                max_dev_idx = i

        # Kolmogorov-Smirnov style statistic
        # Normalize by sqrt(n) and variance
        variance = sum((s - overall_mean) ** 2 for s in sums) / n
        std = math.sqrt(variance) if variance > 0 else 1

        ks_stat = max_dev / (std * math.sqrt(n))

        # Approximate p-value using Kolmogorov distribution
        # For large n, use asymptotic approximation
        p_value = 2 * math.exp(-2 * ks_stat ** 2) if ks_stat > 0 else 1.0
        p_value = min(1.0, max(0.0, p_value))

        # Calculate pre and post change point statistics
        if max_dev_idx > 0 and max_dev_idx < n:
            pre_mean = sum(sums[:max_dev_idx]) / max_dev_idx
            post_mean = sum(sums[max_dev_idx:]) / (n - max_dev_idx)
        else:
            pre_mean = post_mean = overall_mean

        return BiasReport(
            test_name="Change Point Detection",
            description="Detects significant changes in draw patterns over time",
            statistic=ks_stat,
            p_value=p_value,
            is_significant=p_value < self.adjusted_alpha,
            details={
                "potential_change_point_index": max_dev_idx,
                "potential_change_point_draw": max_dev_idx,
                "pre_change_mean": pre_mean,
                "post_change_mean": post_mean,
                "overall_mean": overall_mean,
                "max_cusum_deviation": max_dev
            },
            recommendation="If significant, investigate equipment/procedure changes at detected point"
        )

    def rolling_window_scan(self, draws: list[list[int]],
                            pool_size: int,
                            window_size: int = 100) -> list[BiasReport]:
        """
        Scan for localized biases using rolling windows.

        Detects temporary biases that might appear and disappear,
        useful for identifying equipment degradation periods.

        Args:
            draws: List of draws in chronological order
            pool_size: Total numbers in the pool
            window_size: Size of rolling window

        Returns:
            List of BiasReports for windows with detected anomalies
        """
        anomalies = []

        if len(draws) < window_size:
            return [BiasReport(
                test_name="Rolling Window Scan",
                description="Scans for localized biases in rolling windows",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                details={"error": f"Insufficient draws for window size {window_size}"}
            )]

        # Slide window and check for anomalies
        step_size = window_size // 4  # 75% overlap

        for start in range(0, len(draws) - window_size + 1, step_size):
            window = draws[start:start + window_size]

            # Quick frequency check for this window
            counts = Counter()
            for draw in window:
                for num in draw:
                    counts[num] += 1

            total_nums = sum(counts.values())
            expected = total_nums / pool_size

            # Chi-square for this window
            chi_sq = sum((counts.get(i, 0) - expected) ** 2 / expected
                        for i in range(1, pool_size + 1))

            df = pool_size - 1
            p_value = self._chi_square_p_value(chi_sq, df)

            # More stringent threshold for multiple windows
            num_windows = (len(draws) - window_size) // step_size + 1
            window_alpha = self.significance_level / num_windows

            if p_value < window_alpha:
                # Find most anomalous numbers
                deviations = [(i, (counts.get(i, 0) - expected) / math.sqrt(expected))
                             for i in range(1, pool_size + 1)]
                top_deviations = sorted(deviations, key=lambda x: abs(x[1]), reverse=True)[:5]

                anomalies.append(BiasReport(
                    test_name=f"Rolling Window [{start}:{start+window_size}]",
                    description=f"Localized bias detected in draws {start}-{start+window_size}",
                    statistic=chi_sq,
                    p_value=p_value,
                    is_significant=True,
                    details={
                        "window_start": start,
                        "window_end": start + window_size,
                        "top_deviations": top_deviations,
                        "window_alpha": window_alpha
                    },
                    recommendation="Investigate this specific time period"
                ))

        if not anomalies:
            anomalies.append(BiasReport(
                test_name="Rolling Window Scan",
                description="Scans for localized biases in rolling windows",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                details={
                    "windows_scanned": (len(draws) - window_size) // step_size + 1,
                    "window_size": window_size
                },
                recommendation="No localized anomalies detected"
            ))

        return anomalies

    def full_report(self, draws: list[list[int]],
                    pool_size: int,
                    dates: Optional[list[tuple[int, int, int]]] = None) -> list[BiasReport]:
        """
        Run all bias detection tests and generate comprehensive report.

        Args:
            draws: List of draws
            pool_size: Total numbers in the pool (e.g., 49 for Loto 6/49)
            dates: Optional list of (year, month, day) tuples for temporal analysis

        Returns:
            List of all BiasReports from all tests
        """
        reports = []

        # Core statistical tests
        reports.append(self.frequency_uniformity(draws, pool_size))
        reports.append(self.position_uniformity(draws, pool_size))
        reports.append(self.temporal_correlation(draws))
        reports.append(self.consecutive_pairs(draws, pool_size))
        reports.append(self.sum_distribution(draws, pool_size))
        reports.append(self.change_point_detection(draws))

        # Day of week test (if dates provided)
        if dates:
            reports.append(self.day_of_week_bias(draws, dates))

        # Rolling window scan
        window_reports = self.rolling_window_scan(draws, pool_size)
        reports.extend(window_reports)

        return reports

    def summary(self, reports: list[BiasReport]) -> dict:
        """
        Generate a summary of bias detection results.

        Args:
            reports: List of BiasReports from full_report()

        Returns:
            Dictionary with summary statistics and findings
        """
        significant = [r for r in reports if r.is_significant]

        return {
            "total_tests": len(reports),
            "significant_findings": len(significant),
            "significance_level": self.significance_level,
            "adjusted_alpha": self.adjusted_alpha,
            "overall_conclusion": (
                "POTENTIAL BIASES DETECTED - Further investigation recommended"
                if significant else
                "No significant biases detected - Data appears random"
            ),
            "significant_tests": [r.test_name for r in significant],
            "recommendations": [r.recommendation for r in significant]
        }

    # Helper methods

    def _chi_square_p_value(self, chi_sq: float, df: int) -> float:
        """Calculate p-value from chi-square statistic using approximation."""
        if df <= 0 or chi_sq < 0:
            return 1.0

        # Use Wilson-Hilferty approximation for chi-square CDF
        if df > 0:
            z = ((chi_sq / df) ** (1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
            p_value = 1 - self._standard_normal_cdf(z)
            return max(0.0, min(1.0, p_value))
        return 1.0

    def _standard_normal_cdf(self, z: float) -> float:
        """Approximate standard normal CDF using error function approximation."""
        # Abramowitz and Stegun approximation
        a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741,
                              -1.453152027, 1.061405429)
        p = 0.3275911

        sign = 1 if z >= 0 else -1
        z = abs(z) / math.sqrt(2)

        t = 1.0 / (1.0 + p * z)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)

        return 0.5 * (1.0 + sign * y)

    def _autocorrelation(self, series: list[float], lag: int = 1) -> float:
        """Calculate autocorrelation at given lag."""
        if len(series) <= lag:
            return 0.0

        n = len(series)
        mean = sum(series) / n
        variance = sum((x - mean) ** 2 for x in series) / n

        if variance == 0:
            return 0.0

        covariance = sum((series[i] - mean) * (series[i - lag] - mean)
                        for i in range(lag, n)) / (n - lag)

        return covariance / variance

    def _day_of_week(self, year: int, month: int, day: int) -> int:
        """Calculate day of week (0=Monday, 6=Sunday) using Zeller's formula."""
        if month < 3:
            month += 12
            year -= 1

        k = year % 100
        j = year // 100

        h = (day + (13 * (month + 1)) // 5 + k + k // 4 + j // 4 - 2 * j) % 7

        # Convert from Zeller (0=Saturday) to ISO (0=Monday)
        return (h + 5) % 7
