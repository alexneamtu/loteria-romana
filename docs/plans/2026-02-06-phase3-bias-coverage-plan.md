# Phase 3: Bias Detection + Coverage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance the statistical bias detection pipeline with drift detection, runs tests, and regime splitting; add portfolio optimization for ticket diversity; integrate enhanced bias signals into the ensemble blend.

**Architecture:** Phase 3 builds three new modules: (1) `drift_detection.py` — ADWIN-style adaptive windowing for distribution shift detection, (2) `runs_test.py` — per-number Wald-Wolfowitz runs test for streak analysis, (3) `portfolio.py` — mean-variance portfolio optimization to minimize ticket redundancy. A new `regime.py` module detects regime boundaries and exposes regime-segmented draws. All new modules use stdlib + optional scipy. Enhanced bias signals feed into `ensemble_blend.py` to adjust strategy weights dynamically.

**Tech Stack:** Python stdlib, optional scipy.optimize for portfolio optimization, existing `bias_detection.py` and `bias_detector.py` infrastructure.

---

## Existing Code Context

**Important files to understand before starting:**

- `src/shared/bias_detection.py` — `chi_square_uniformity_test()` returns `BiasReport(chi_square, degrees_of_freedom, critical_value, significant, bias_strength)`. Used by `ensemble_blend.py` to boost random weight when no bias detected.
- `src/shared/bias_detector.py` — `BiasDetector` class with `frequency_uniformity()`, `position_uniformity()`, `temporal_correlation()`, `consecutive_pairs()`, `sum_distribution()`, `change_point_detection()`, `rolling_window_scan()`, `full_report()`. Has internal helpers `_chi_square_p_value()`, `_standard_normal_cdf()`, `_autocorrelation()`.
- `src/shared/ensemble_blend.py` — `generate_blended_picks()` calls `chi_square_uniformity_test()` and uses `bias.significant` and `bias.bias_strength` to adjust random weight.
- `src/shared/wheeling.py` — `WheelGenerator`, `OptimizedWheelGenerator` with greedy set-cover, `verify_wheel_coverage()`, `compute_coverage_statistics()`.
- `src/shared/game_config.py` — `GameConfig(name, pool_min, pool_max, numbers_drawn, numbers_to_pick, ...)` with `pool_size` and `pool_range` properties.
- `src/shared/game_strategies.py` — `generate_random_picks()`, `generate_frequency_picks()`, `is_prize_winner()`, `_sample_weighted()`.
- `src/shared/genetic.py` — Pattern for strategy interface: `name` attribute, `generate(draws, count, rng, **kwargs)`, `get_probabilities(draws, **kwargs)`.

**Test command:** `PYTHONPATH=src python -m pytest tests/ -v`
**Run single test:** `PYTHONPATH=src python -m pytest tests/test_file.py -v`

---

### Task 1: ADWIN Drift Detection

Implement adaptive windowing to detect distribution shifts over time. ADWIN shrinks/grows a window to detect when the mean of a statistic changes significantly — useful for detecting equipment changes or procedural modifications.

**Files:**
- Create: `src/shared/drift_detection.py`
- Create: `tests/test_drift_detection.py`

**Step 1: Write the failing tests**

Create `tests/test_drift_detection.py`:

```python
import unittest
import random

from shared.drift_detection import (
    adwin_detect_drift,
    cusum_detect_drift,
    DriftReport,
)


class TestDriftReport(unittest.TestCase):
    def test_drift_report_fields(self):
        report = DriftReport(
            method="adwin",
            drift_detected=True,
            drift_points=[50],
            statistic=2.5,
            details={"window_size": 30},
        )
        self.assertEqual(report.method, "adwin")
        self.assertTrue(report.drift_detected)
        self.assertEqual(report.drift_points, [50])


class TestADWINDetectDrift(unittest.TestCase):
    def test_no_drift_uniform(self):
        rng = random.Random(42)
        values = [rng.gauss(100, 10) for _ in range(200)]
        report = adwin_detect_drift(values)
        self.assertFalse(report.drift_detected)
        self.assertEqual(report.method, "adwin")

    def test_detects_mean_shift(self):
        rng = random.Random(42)
        before = [rng.gauss(100, 10) for _ in range(200)]
        after = [rng.gauss(150, 10) for _ in range(200)]
        values = before + after
        report = adwin_detect_drift(values)
        self.assertTrue(report.drift_detected)
        self.assertTrue(len(report.drift_points) >= 1)
        # Drift point should be near index 200
        self.assertTrue(any(150 < p < 250 for p in report.drift_points))

    def test_empty_input(self):
        report = adwin_detect_drift([])
        self.assertFalse(report.drift_detected)

    def test_short_input(self):
        report = adwin_detect_drift([1.0, 2.0, 3.0])
        self.assertFalse(report.drift_detected)

    def test_custom_delta(self):
        rng = random.Random(42)
        before = [rng.gauss(100, 5) for _ in range(200)]
        after = [rng.gauss(110, 5) for _ in range(200)]
        values = before + after
        report = adwin_detect_drift(values, delta=0.01)
        self.assertIsInstance(report, DriftReport)


class TestCUSUMDetectDrift(unittest.TestCase):
    def test_no_drift(self):
        rng = random.Random(42)
        values = [rng.gauss(100, 10) for _ in range(200)]
        report = cusum_detect_drift(values)
        self.assertFalse(report.drift_detected)
        self.assertEqual(report.method, "cusum")

    def test_detects_shift(self):
        rng = random.Random(42)
        before = [rng.gauss(100, 10) for _ in range(200)]
        after = [rng.gauss(150, 10) for _ in range(200)]
        values = before + after
        report = cusum_detect_drift(values, threshold=5.0)
        self.assertTrue(report.drift_detected)

    def test_returns_drift_report(self):
        report = cusum_detect_drift([1.0] * 100)
        self.assertIsInstance(report, DriftReport)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_drift_detection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.drift_detection'`

**Step 3: Implement drift detection**

Create `src/shared/drift_detection.py`:

```python
"""Drift detection for lottery draw distributions.

Detects distribution shifts over time using:
- ADWIN (ADaptive WINdowing): automatically adjusts window size
  to detect changes in the mean of a stream.
- CUSUM (CUmulative SUM): cumulative deviation from expected mean.

Both methods help identify equipment changes, procedural modifications,
or other regime shifts in lottery draw data.
"""

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DriftReport:
    """Result of a drift detection analysis."""

    method: str
    drift_detected: bool
    drift_points: list[int] = field(default_factory=list)
    statistic: float = 0.0
    details: dict = field(default_factory=dict)


def adwin_detect_drift(
    values: list[float],
    delta: float = 0.002,
    min_window: int = 30,
) -> DriftReport:
    """Detect drift using ADWIN (ADaptive WINdowing).

    Scans the data for the split point that maximizes the
    statistical difference between the two resulting sub-windows.
    Uses Hoeffding bound to decide significance.

    Args:
        values: Time-ordered numeric sequence (e.g. draw sums).
        delta: Confidence parameter (lower = stricter). Default 0.002.
        min_window: Minimum sub-window size to consider.

    Returns:
        DriftReport with detected change points.
    """
    n = len(values)
    if n < 2 * min_window:
        return DriftReport(method="adwin", drift_detected=False)

    drift_points: list[int] = []
    max_statistic = 0.0

    for split in range(min_window, n - min_window + 1):
        left = values[:split]
        right = values[split:]

        n_left = len(left)
        n_right = len(right)

        mean_left = sum(left) / n_left
        mean_right = sum(right) / n_right

        m = 1.0 / (1.0 / n_left + 1.0 / n_right)
        diff = abs(mean_left - mean_right)

        # Hoeffding bound
        var_left = sum((x - mean_left) ** 2 for x in left) / n_left
        var_right = sum((x - mean_right) ** 2 for x in right) / n_right
        pooled_var = (var_left * n_left + var_right * n_right) / n

        if pooled_var <= 0:
            continue

        epsilon = math.sqrt(2.0 * pooled_var * math.log(2.0 / delta) / m)

        statistic = diff / max(epsilon, 1e-10)
        if statistic > max_statistic:
            max_statistic = statistic

        if diff > epsilon:
            drift_points.append(split)

    # Deduplicate: keep only locally maximal drift points
    if drift_points:
        deduped = [drift_points[0]]
        for p in drift_points[1:]:
            if p - deduped[-1] > min_window:
                deduped.append(p)
        drift_points = deduped

    return DriftReport(
        method="adwin",
        drift_detected=len(drift_points) > 0,
        drift_points=drift_points,
        statistic=max_statistic,
        details={"delta": delta, "min_window": min_window, "n": n},
    )


def cusum_detect_drift(
    values: list[float],
    threshold: float = 4.0,
    drift_magnitude: float | None = None,
) -> DriftReport:
    """Detect drift using CUSUM (CUmulative SUM control chart).

    Tracks cumulative deviation from the target mean. When the
    cumulative sum exceeds the threshold, a drift is signaled.

    Args:
        values: Time-ordered numeric sequence.
        threshold: Decision threshold h (in units of std dev).
        drift_magnitude: Expected shift magnitude. If None, uses 1 std dev.

    Returns:
        DriftReport with detected change points.
    """
    n = len(values)
    if n < 10:
        return DriftReport(method="cusum", drift_detected=False)

    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / n
    std_val = math.sqrt(variance) if variance > 0 else 1.0

    k = drift_magnitude if drift_magnitude is not None else std_val * 0.5
    h = threshold * std_val

    s_pos = 0.0
    s_neg = 0.0
    drift_points: list[int] = []
    max_statistic = 0.0

    for i, x in enumerate(values):
        s_pos = max(0.0, s_pos + (x - mean_val) - k)
        s_neg = max(0.0, s_neg - (x - mean_val) - k)

        stat = max(s_pos, s_neg)
        if stat > max_statistic:
            max_statistic = stat

        if s_pos > h or s_neg > h:
            drift_points.append(i)
            s_pos = 0.0
            s_neg = 0.0

    return DriftReport(
        method="cusum",
        drift_detected=len(drift_points) > 0,
        drift_points=drift_points,
        statistic=max_statistic,
        details={"threshold": threshold, "mean": mean_val, "std": std_val},
    )
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_drift_detection.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/shared/drift_detection.py tests/test_drift_detection.py
git commit -m "feat: add ADWIN and CUSUM drift detection"
```

---

### Task 2: Per-Number Runs Test

Implement Wald-Wolfowitz runs test per number to detect non-random streaks (a number appearing or not appearing for suspiciously long runs).

**Files:**
- Create: `src/shared/runs_test.py`
- Create: `tests/test_runs_test.py`

**Step 1: Write the failing tests**

Create `tests/test_runs_test.py`:

```python
import unittest
import random

from shared.runs_test import (
    wald_wolfowitz_runs_test,
    per_number_runs_analysis,
    RunsReport,
)


class TestRunsReport(unittest.TestCase):
    def test_fields(self):
        report = RunsReport(
            number=5,
            observed_runs=10,
            expected_runs=12.0,
            z_score=-0.5,
            p_value=0.6,
            significant=False,
        )
        self.assertEqual(report.number, 5)
        self.assertFalse(report.significant)


class TestWaldWolfowitzRunsTest(unittest.TestCase):
    def test_alternating_sequence(self):
        # Perfectly alternating: maximum runs
        sequence = [True, False] * 50
        report = wald_wolfowitz_runs_test(sequence, number=1)
        self.assertFalse(report.significant)
        self.assertEqual(report.observed_runs, 100)

    def test_clustered_sequence(self):
        # Highly clustered: very few runs
        sequence = [True] * 50 + [False] * 50
        report = wald_wolfowitz_runs_test(sequence, number=1)
        self.assertTrue(report.significant)
        self.assertEqual(report.observed_runs, 2)

    def test_random_sequence_not_significant(self):
        rng = random.Random(42)
        sequence = [rng.random() < 0.5 for _ in range(200)]
        report = wald_wolfowitz_runs_test(sequence, number=1)
        self.assertFalse(report.significant)

    def test_empty_sequence(self):
        report = wald_wolfowitz_runs_test([], number=1)
        self.assertFalse(report.significant)

    def test_all_same(self):
        report = wald_wolfowitz_runs_test([True] * 100, number=1)
        self.assertFalse(report.significant)  # Can't test with one type


class TestPerNumberRunsAnalysis(unittest.TestCase):
    def test_returns_reports_for_all_numbers(self):
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 11), 3)) for _ in range(100)]
        reports = per_number_runs_analysis(draws, pool_size=10)
        self.assertEqual(len(reports), 10)

    def test_report_numbers_match_pool(self):
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 11), 3)) for _ in range(100)]
        reports = per_number_runs_analysis(draws, pool_size=10)
        numbers = {r.number for r in reports}
        self.assertEqual(numbers, set(range(1, 11)))

    def test_random_draws_mostly_not_significant(self):
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 46), 5)) for _ in range(500)]
        reports = per_number_runs_analysis(draws, pool_size=45)
        significant_count = sum(1 for r in reports if r.significant)
        # At most ~5% should be significant by chance (with Bonferroni, even fewer)
        self.assertLessEqual(significant_count, 10)

    def test_biased_number_detected(self):
        # Number 1 appears in every draw for first 50, then never
        draws = []
        rng = random.Random(42)
        for i in range(50):
            pick = sorted(rng.sample(range(2, 11), 2) + [1])
            draws.append(pick)
        for i in range(50):
            pick = sorted(rng.sample(range(2, 11), 3))
            draws.append(pick)
        reports = per_number_runs_analysis(draws, pool_size=10)
        report_1 = next(r for r in reports if r.number == 1)
        self.assertTrue(report_1.significant)

    def test_custom_significance(self):
        rng = random.Random(42)
        draws = [sorted(rng.sample(range(1, 11), 3)) for _ in range(100)]
        reports = per_number_runs_analysis(draws, pool_size=10, alpha=0.001)
        self.assertEqual(len(reports), 10)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_runs_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.runs_test'`

**Step 3: Implement runs test**

Create `src/shared/runs_test.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_runs_test.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/shared/runs_test.py tests/test_runs_test.py
git commit -m "feat: add per-number Wald-Wolfowitz runs test"
```

---

### Task 3: Regime Detection

Detect regimes (segments of draw history with different statistical properties) and expose regime-segmented data for backtesting.

**Files:**
- Create: `src/shared/regime.py`
- Create: `tests/test_regime.py`

**Step 1: Write the failing tests**

Create `tests/test_regime.py`:

```python
import unittest
import random

from shared.regime import (
    detect_regimes,
    segment_draws_by_regime,
    RegimeBoundary,
)


class TestRegimeBoundary(unittest.TestCase):
    def test_fields(self):
        b = RegimeBoundary(index=50, confidence=0.95, pre_mean=100.0, post_mean=130.0)
        self.assertEqual(b.index, 50)
        self.assertAlmostEqual(b.confidence, 0.95)


class TestDetectRegimes(unittest.TestCase):
    def test_no_regime_change(self):
        rng = random.Random(42)
        values = [rng.gauss(100, 10) for _ in range(200)]
        boundaries = detect_regimes(values)
        self.assertEqual(len(boundaries), 0)

    def test_single_regime_change(self):
        rng = random.Random(42)
        before = [rng.gauss(100, 10) for _ in range(200)]
        after = [rng.gauss(150, 10) for _ in range(200)]
        values = before + after
        boundaries = detect_regimes(values)
        self.assertGreaterEqual(len(boundaries), 1)
        # Boundary should be near index 200
        self.assertTrue(any(150 < b.index < 250 for b in boundaries))

    def test_multiple_regime_changes(self):
        rng = random.Random(42)
        seg1 = [rng.gauss(100, 10) for _ in range(150)]
        seg2 = [rng.gauss(150, 10) for _ in range(150)]
        seg3 = [rng.gauss(80, 10) for _ in range(150)]
        values = seg1 + seg2 + seg3
        boundaries = detect_regimes(values)
        self.assertGreaterEqual(len(boundaries), 2)

    def test_empty_input(self):
        boundaries = detect_regimes([])
        self.assertEqual(len(boundaries), 0)

    def test_short_input(self):
        boundaries = detect_regimes([1.0, 2.0])
        self.assertEqual(len(boundaries), 0)

    def test_custom_min_segment(self):
        rng = random.Random(42)
        values = [rng.gauss(100, 10) for _ in range(200)]
        boundaries = detect_regimes(values, min_segment_size=100)
        self.assertEqual(len(boundaries), 0)


class TestSegmentDrawsByRegime(unittest.TestCase):
    def test_no_boundaries(self):
        draws = [[1, 2, 3]] * 100
        segments = segment_draws_by_regime(draws, [])
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]), 100)

    def test_single_boundary(self):
        draws = [[1, 2, 3]] * 100
        boundaries = [RegimeBoundary(index=50, confidence=0.95, pre_mean=10.0, post_mean=20.0)]
        segments = segment_draws_by_regime(draws, boundaries)
        self.assertEqual(len(segments), 2)
        self.assertEqual(len(segments[0]), 50)
        self.assertEqual(len(segments[1]), 50)

    def test_multiple_boundaries(self):
        draws = [[1, 2, 3]] * 150
        boundaries = [
            RegimeBoundary(index=50, confidence=0.9, pre_mean=10.0, post_mean=20.0),
            RegimeBoundary(index=100, confidence=0.9, pre_mean=20.0, post_mean=30.0),
        ]
        segments = segment_draws_by_regime(draws, boundaries)
        self.assertEqual(len(segments), 3)
        self.assertEqual(len(segments[0]), 50)
        self.assertEqual(len(segments[1]), 50)
        self.assertEqual(len(segments[2]), 50)

    def test_boundary_at_edges(self):
        draws = [[1, 2, 3]] * 100
        boundaries = [RegimeBoundary(index=0, confidence=0.9, pre_mean=10.0, post_mean=20.0)]
        segments = segment_draws_by_regime(draws, boundaries)
        # Should handle edge gracefully
        self.assertTrue(all(len(s) > 0 for s in segments if s))


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_regime.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.regime'`

**Step 3: Implement regime detection**

Create `src/shared/regime.py`:

```python
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

    # Use the approximation: sqrt(2*F*df1/df2) - sqrt(2*df1-1) ~ N(0,1)
    # for large df2
    try:
        z = math.sqrt(2.0 * f) - math.sqrt(2.0 * df1 - 1.0) if df1 >= 1 else 0.0
        # Standard normal CDF
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
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_regime.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/shared/regime.py tests/test_regime.py
git commit -m "feat: add regime detection with binary segmentation"
```

---

### Task 4: Portfolio Optimization

Implement mean-variance portfolio optimization to maximize coverage diversity across a ticket set, minimizing redundancy (overlapping numbers).

**Files:**
- Create: `src/shared/portfolio.py`
- Create: `tests/test_portfolio.py`

**Step 1: Write the failing tests**

Create `tests/test_portfolio.py`:

```python
import unittest
import random

from shared.portfolio import (
    compute_ticket_covariance,
    optimize_ticket_portfolio,
    diversity_score,
)
from shared.game_config import JOKER_CONFIG, LOTO_649_CONFIG


class TestComputeTicketCovariance(unittest.TestCase):
    def test_identical_tickets_high_covariance(self):
        tickets = [[1, 2, 3, 4, 5]] * 3
        cov = compute_ticket_covariance(tickets, pool_size=45)
        # Diagonal should be positive
        self.assertGreater(cov[0][0], 0)
        # Off-diagonal should equal diagonal (identical tickets)
        self.assertAlmostEqual(cov[0][1], cov[0][0])

    def test_disjoint_tickets_low_covariance(self):
        tickets = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        cov = compute_ticket_covariance(tickets, pool_size=45)
        # Off-diagonal should be near zero (no overlap)
        self.assertAlmostEqual(cov[0][1], 0.0)

    def test_matrix_dimensions(self):
        tickets = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        cov = compute_ticket_covariance(tickets, pool_size=10)
        self.assertEqual(len(cov), 3)
        self.assertEqual(len(cov[0]), 3)

    def test_symmetry(self):
        tickets = [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [8, 9, 10, 11, 12]]
        cov = compute_ticket_covariance(tickets, pool_size=45)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(cov[i][j], cov[j][i])


class TestOptimizeTicketPortfolio(unittest.TestCase):
    def test_selects_correct_count(self):
        rng = random.Random(42)
        candidates = [
            sorted(rng.sample(range(1, 46), 5)) for _ in range(20)
        ]
        selected = optimize_ticket_portfolio(candidates, select_count=5, pool_size=45)
        self.assertEqual(len(selected), 5)

    def test_selected_from_candidates(self):
        rng = random.Random(42)
        candidates = [
            sorted(rng.sample(range(1, 46), 5)) for _ in range(20)
        ]
        candidate_set = {tuple(c) for c in candidates}
        selected = optimize_ticket_portfolio(candidates, select_count=5, pool_size=45)
        for s in selected:
            self.assertIn(tuple(s), candidate_set)

    def test_diversity_better_than_random(self):
        rng = random.Random(42)
        # Create candidates with both diverse and redundant tickets
        diverse = [sorted(range(i * 5 + 1, i * 5 + 6)) for i in range(8)]
        redundant = [[1, 2, 3, 4, 5]] * 12
        candidates = diverse + redundant
        selected = optimize_ticket_portfolio(candidates, select_count=5, pool_size=45)
        # Selected should have good diversity
        score = diversity_score(selected, pool_size=45)
        self.assertGreater(score, 0.5)

    def test_fewer_candidates_than_requested(self):
        candidates = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        selected = optimize_ticket_portfolio(candidates, select_count=5, pool_size=45)
        self.assertEqual(len(selected), 2)

    def test_empty_candidates(self):
        selected = optimize_ticket_portfolio([], select_count=5, pool_size=45)
        self.assertEqual(len(selected), 0)


class TestDiversityScore(unittest.TestCase):
    def test_identical_tickets_low_score(self):
        tickets = [[1, 2, 3, 4, 5]] * 5
        score = diversity_score(tickets, pool_size=45)
        self.assertLess(score, 0.2)

    def test_diverse_tickets_high_score(self):
        tickets = [
            [1, 2, 3, 4, 5],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
            [40, 41, 42, 43, 44],
        ]
        score = diversity_score(tickets, pool_size=45)
        self.assertGreater(score, 0.8)

    def test_single_ticket(self):
        score = diversity_score([[1, 2, 3, 4, 5]], pool_size=45)
        self.assertEqual(score, 1.0)

    def test_empty_list(self):
        score = diversity_score([], pool_size=45)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_portfolio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.portfolio'`

**Step 3: Implement portfolio optimization**

Create `src/shared/portfolio.py`:

```python
"""Portfolio optimization for lottery ticket selection.

Treats ticket selection as a mean-variance portfolio problem.
Each ticket is characterized by its number coverage. The goal
is to select a subset of tickets that maximizes coverage diversity
and minimizes redundancy (overlapping numbers).

Uses a greedy algorithm that iteratively selects the ticket with
the lowest covariance to the already-selected set.
"""

import math


def compute_ticket_covariance(
    tickets: list[list[int]],
    pool_size: int,
) -> list[list[float]]:
    """Compute covariance matrix between tickets based on number overlap.

    Each ticket is represented as a binary vector over the number pool.
    Covariance measures how much two tickets share the same numbers.

    Args:
        tickets: List of tickets (each a sorted list of numbers).
        pool_size: Total numbers in the pool.

    Returns:
        n x n covariance matrix where n = len(tickets).
    """
    n = len(tickets)
    if n == 0:
        return []

    # Convert tickets to binary vectors
    vectors = []
    for ticket in tickets:
        vec = [0.0] * pool_size
        for num in ticket:
            if 1 <= num <= pool_size:
                vec[num - 1] = 1.0
        vectors.append(vec)

    # Compute mean vector
    mean_vec = [0.0] * pool_size
    for vec in vectors:
        for j in range(pool_size):
            mean_vec[j] += vec[j]
    for j in range(pool_size):
        mean_vec[j] /= n

    # Compute covariance matrix
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = sum(
                (vectors[i][k] - mean_vec[k]) * (vectors[j][k] - mean_vec[k])
                for k in range(pool_size)
            ) / pool_size
            cov[i][j] = val
            cov[j][i] = val

    return cov


def _overlap_fraction(ticket_a: list[int], ticket_b: list[int], pool_size: int) -> float:
    """Compute overlap fraction between two tickets."""
    set_a = set(ticket_a)
    set_b = set(ticket_b)
    overlap = len(set_a & set_b)
    max_possible = min(len(ticket_a), len(ticket_b))
    return overlap / max_possible if max_possible > 0 else 0.0


def optimize_ticket_portfolio(
    candidates: list[list[int]],
    select_count: int,
    pool_size: int,
) -> list[list[int]]:
    """Select a diverse subset of tickets minimizing redundancy.

    Uses greedy selection: at each step, add the candidate with
    the lowest average overlap to the already-selected tickets.

    Args:
        candidates: Pool of candidate tickets to choose from.
        select_count: Number of tickets to select.
        pool_size: Total numbers in the pool.

    Returns:
        Selected tickets optimized for diversity.
    """
    if not candidates:
        return []

    if len(candidates) <= select_count:
        return list(candidates)

    # Start with the ticket covering the most unique numbers
    # (approximation: pick from different regions of the number space)
    selected_indices: list[int] = []
    selected_sets: list[set[int]] = []

    # Pick first ticket (most central or arbitrary)
    best_first = 0
    best_score = -1.0
    for i, ticket in enumerate(candidates):
        # Score by how "spread" the numbers are
        if len(ticket) >= 2:
            spread = ticket[-1] - ticket[0]
        else:
            spread = 0
        if spread > best_score:
            best_score = spread
            best_first = i

    selected_indices.append(best_first)
    selected_sets.append(set(candidates[best_first]))

    # Greedily add tickets with minimum overlap
    while len(selected_indices) < select_count:
        best_idx = -1
        best_avg_overlap = float("inf")

        for i, ticket in enumerate(candidates):
            if i in selected_indices:
                continue

            ticket_set = set(ticket)
            total_overlap = 0.0
            for s_set in selected_sets:
                overlap = len(ticket_set & s_set)
                total_overlap += overlap

            avg_overlap = total_overlap / len(selected_sets)

            if avg_overlap < best_avg_overlap:
                best_avg_overlap = avg_overlap
                best_idx = i

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        selected_sets.append(set(candidates[best_idx]))

    return [candidates[i] for i in selected_indices]


def diversity_score(
    tickets: list[list[int]],
    pool_size: int,
) -> float:
    """Compute a diversity score for a set of tickets.

    Score ranges from 0 (all identical) to 1 (no overlap).
    Based on the ratio of unique numbers covered to theoretical
    maximum, and penalized by pairwise overlap.

    Args:
        tickets: List of tickets.
        pool_size: Total numbers in the pool.

    Returns:
        Diversity score between 0 and 1.
    """
    if not tickets:
        return 0.0

    if len(tickets) == 1:
        return 1.0

    # Coverage component: what fraction of the pool is covered
    all_numbers = set()
    for ticket in tickets:
        all_numbers.update(ticket)
    total_numbers = sum(len(t) for t in tickets)
    max_unique = min(total_numbers, pool_size)
    coverage = len(all_numbers) / max_unique if max_unique > 0 else 0.0

    # Overlap penalty: average pairwise overlap
    n = len(tickets)
    total_overlap = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            overlap = _overlap_fraction(tickets[i], tickets[j], pool_size)
            total_overlap += overlap
            pair_count += 1

    avg_overlap = total_overlap / pair_count if pair_count > 0 else 0.0

    # Diversity = coverage * (1 - avg_overlap)
    return coverage * (1.0 - avg_overlap)
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_portfolio.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/shared/portfolio.py tests/test_portfolio.py
git commit -m "feat: add portfolio optimization for ticket diversity"
```

---

### Task 5: Enhanced Bias Integration into Ensemble Blend

Wire the new drift detection, runs test, and regime analysis into the ensemble blend to dynamically adjust strategy weights based on detected statistical anomalies.

**Files:**
- Modify: `src/shared/ensemble_blend.py`
- Modify: `tests/test_ensemble_blend.py`

**Step 1: Write the failing tests**

Add to `tests/test_ensemble_blend.py`:

```python
class TestEnhancedBiasIntegration(unittest.TestCase):
    def _make_draws(self, config, count=50):
        rng = random.Random(0)
        pool = list(config.pool_range)
        return [sorted(rng.sample(pool, config.numbers_drawn)) for _ in range(count)]

    def test_blend_runs_with_drift_detection(self):
        """Ensemble should complete even with drift detection active."""
        draws = self._make_draws(JOKER_CONFIG, count=100)
        rng = random.Random(42)
        lines = generate_blended_picks(JOKER_CONFIG, draws, 5, rng)
        self.assertEqual(len(lines), 5)

    def test_blend_handles_regime_aware_scoring(self):
        """Ensemble should work with regime-aware scoring window."""
        draws = self._make_draws(JOKER_CONFIG, count=200)
        rng = random.Random(42)
        lines = generate_blended_picks(JOKER_CONFIG, draws, 5, rng)
        self.assertEqual(len(lines), 5)
        for line in lines:
            self.assertEqual(len(line), JOKER_CONFIG.numbers_to_pick)
            self.assertTrue(all(n in JOKER_CONFIG.pool_range for n in line))
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_ensemble_blend.py::TestEnhancedBiasIntegration -v`
Expected: FAIL — `NameError: name 'TestEnhancedBiasIntegration' is not defined` (since we haven't added the class yet)

**Step 3: Implement enhanced bias integration**

Modify `src/shared/ensemble_blend.py` to add these changes:

1. Add imports at the top (after existing imports):

```python
from .drift_detection import adwin_detect_drift
from .runs_test import per_number_runs_analysis
from .regime import detect_regimes, segment_draws_by_regime
```

2. Add a new function `_compute_enhanced_bias`:

```python
def _compute_enhanced_bias(
    draws: list[list[int]],
    pool_size: int,
    numbers_drawn: int,
) -> dict[str, float]:
    """Compute enhanced bias signals from multiple detectors.

    Returns a dict of bias adjustments:
    - 'drift_factor': 0.0 to 1.0 (1.0 = strong drift detected)
    - 'runs_factor': 0.0 to 1.0 (1.0 = many non-random numbers)
    - 'regime_recency': fraction of draws in most recent regime
    """
    result = {"drift_factor": 0.0, "runs_factor": 0.0, "regime_recency": 1.0}

    if len(draws) < 30:
        return result

    # Drift detection on draw sums
    sums = [sum(d) for d in draws]
    drift = adwin_detect_drift(sums)
    if drift.drift_detected:
        result["drift_factor"] = min(drift.statistic / 3.0, 1.0)

    # Runs test: fraction of numbers with significant streaks
    runs_reports = per_number_runs_analysis(draws, pool_size)
    significant_count = sum(1 for r in runs_reports if r.significant)
    result["runs_factor"] = significant_count / pool_size

    # Regime detection: focus on most recent regime
    boundaries = detect_regimes(sums, min_segment_size=30)
    if boundaries:
        last_boundary = max(b.index for b in boundaries)
        recent_fraction = (len(draws) - last_boundary) / len(draws)
        result["regime_recency"] = recent_fraction

    return result
```

3. In `generate_blended_picks()`, after the existing `bias = chi_square_uniformity_test(...)` call, add enhanced bias computation and use it to adjust weights:

After the line `bias = chi_square_uniformity_test(draws, config.pool_size, config.numbers_drawn,)`, add:

```python
    enhanced = _compute_enhanced_bias(draws, config.pool_size, config.numbers_drawn)
```

Then in the weight adjustment section (after softmax), replace the existing random boost logic:

Replace:
```python
    if not bias.significant:
        random_boost = 0.3 * (1.0 - bias.bias_strength)
        weights["random"] += random_boost
```

With:
```python
    if not bias.significant:
        random_boost = 0.3 * (1.0 - bias.bias_strength)
        weights["random"] += random_boost

    # Enhanced bias adjustments
    if enhanced["drift_factor"] > 0.5:
        # Strong drift: boost recent-data strategies, reduce long-history ones
        for strat in ["frequency", "bayesian"]:
            if strat in weights:
                weights[strat] *= 1.0 + enhanced["drift_factor"] * 0.3

    if enhanced["runs_factor"] > 0.1:
        # Non-random streaks detected: boost pattern-aware strategies
        for strat in ["cooccurrence", "genetic"]:
            if strat in weights:
                weights[strat] *= 1.0 + enhanced["runs_factor"] * 0.5
```

Also, use regime_recency to adjust the scoring window. In the scoring draws selection, replace:

```python
    scoring_draws = draws[-100:] if len(draws) > 100 else draws
    scoring_dates = draw_dates[-100:] if draw_dates and len(draw_dates) > 100 else draw_dates
```

With:

```python
    # Use regime-aware scoring window
    regime_window = int(len(draws) * enhanced["regime_recency"])
    scoring_window = max(min(regime_window, 100), 30)
    scoring_draws = draws[-scoring_window:] if len(draws) > scoring_window else draws
    scoring_dates = draw_dates[-scoring_window:] if draw_dates and len(draw_dates) > scoring_window else draw_dates
```

4. Add the test class to `tests/test_ensemble_blend.py`.

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_ensemble_blend.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/shared/ensemble_blend.py tests/test_ensemble_blend.py
git commit -m "feat: integrate drift, runs, and regime detection into ensemble blend"
```

---

### Task 6: Portfolio-Optimized Pick Generation

Add a post-processing step that applies portfolio optimization to the final pick set, ensuring maximum diversity.

**Files:**
- Modify: `src/shared/ensemble_blend.py`
- Modify: `tests/test_ensemble_blend.py`

**Step 1: Write the failing tests**

Add to `tests/test_ensemble_blend.py`:

```python
class TestPortfolioOptimizedPicks(unittest.TestCase):
    def _make_draws(self, config, count=50):
        rng = random.Random(0)
        pool = list(config.pool_range)
        return [sorted(rng.sample(pool, config.numbers_drawn)) for _ in range(count)]

    def test_optimized_picks_correct_count(self):
        draws = self._make_draws(JOKER_CONFIG)
        rng = random.Random(42)
        lines = generate_blended_picks(JOKER_CONFIG, draws, 10, rng)
        self.assertEqual(len(lines), 10)

    def test_optimized_picks_diverse(self):
        """Picks should have reasonable diversity."""
        from shared.portfolio import diversity_score
        draws = self._make_draws(JOKER_CONFIG, count=100)
        rng = random.Random(42)
        lines = generate_blended_picks(JOKER_CONFIG, draws, 10, rng)
        score = diversity_score(lines, JOKER_CONFIG.pool_size)
        # Should have at least moderate diversity
        self.assertGreater(score, 0.3)

    def test_all_games_produce_valid_picks(self):
        for config in [JOKER_CONFIG, LOTO_649_CONFIG, LOTO_540_CONFIG]:
            draws = self._make_draws(config, count=50)
            rng = random.Random(42)
            lines = generate_blended_picks(config, draws, 5, rng)
            self.assertEqual(len(lines), 5)
            for line in lines:
                self.assertEqual(len(line), config.numbers_to_pick)
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_ensemble_blend.py::TestPortfolioOptimizedPicks -v`
Expected: FAIL

**Step 3: Implement portfolio optimization post-processing**

Add import in `src/shared/ensemble_blend.py`:

```python
from .portfolio import optimize_ticket_portfolio
```

At the end of `generate_blended_picks()`, before `return lines[:count]`, add portfolio optimization:

Replace:
```python
    return lines[:count]
```

With:
```python
    # Apply portfolio optimization when we have more candidates than needed
    if len(lines) > count:
        lines = optimize_ticket_portfolio(lines, count, config.pool_size)

    return lines[:count]
```

Also, generate extra candidates for portfolio optimization to select from. Change the `while len(lines) < count:` loop to generate more than needed:

Replace:
```python
    while len(lines) < count:
        extra = generate_random_picks(config, 1, rng)
        key = tuple(extra[0])
        if key not in seen:
            seen.add(key)
            lines.append(extra[0])

    return lines[:count]
```

With:
```python
    # Generate extra random candidates for portfolio selection
    target = count + max(count // 2, 3)
    while len(lines) < target:
        extra = generate_random_picks(config, 1, rng)
        key = tuple(extra[0])
        if key not in seen:
            seen.add(key)
            lines.append(extra[0])

    # Apply portfolio optimization for diversity
    if len(lines) > count:
        lines = optimize_ticket_portfolio(lines, count, config.pool_size)

    return lines[:count]
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_ensemble_blend.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `PYTHONPATH=src python -m pytest tests/ -v`
Expected: All previously-passing tests still pass.

**Step 6: Commit**

```bash
git add src/shared/ensemble_blend.py tests/test_ensemble_blend.py
git commit -m "feat: add portfolio optimization for diverse pick selection"
```

---

### Task 7: End-to-End Verification and PR

Verify everything works end-to-end, then push and create a PR.

**Step 1: Run full test suite**

Run: `PYTHONPATH=src python -m pytest tests/ -v`
Expected: All pass (except any pre-existing failures).

**Step 2: Test pick generation scripts**

Run:
```bash
PYTHONPATH=src python scripts/generate_joker_picks.py --seed 42
PYTHONPATH=src python scripts/generate_loto_649_picks.py --seed 42
PYTHONPATH=src python scripts/generate_loto_540_picks.py --seed 42
```
Expected: Each script outputs formatted picks without errors.

**Step 3: Push and create PR**

```bash
git push -u origin feature/phase3-bias-coverage
gh pr create --title "feat: Phase 3 — Enhanced bias detection, regime analysis, and portfolio optimization" --body "$(cat <<'EOF'
## Summary
- ADWIN and CUSUM drift detection for distribution shift identification
- Per-number Wald-Wolfowitz runs test for streak analysis
- Binary segmentation regime detection with draw segmenting
- Mean-variance portfolio optimization for ticket diversity
- Enhanced bias signals integrated into ensemble blend weight adjustment
- Regime-aware scoring window for walk-forward backtesting

## Test plan
- [ ] All new modules have comprehensive unit tests
- [ ] Full test suite passes
- [ ] End-to-end pick generation verified for all three games
- [ ] Portfolio diversity scores above threshold for generated picks
EOF
)"
```
