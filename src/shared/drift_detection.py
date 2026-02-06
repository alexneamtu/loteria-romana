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
        drift_magnitude: Expected shift magnitude. If None, uses 0.5 std dev.

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
