"""Calibration metrics for probability predictions.

Measures whether predicted probabilities match observed frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalibrationResult:
    strategy: str
    brier_score: float
    expected_calibration_error: float
    bins: list[dict] = field(default_factory=list)


def compute_calibration(
    predicted_probs: list[float],
    observed: list[bool],
    n_bins: int = 10,
    strategy: str = "",
) -> CalibrationResult:
    """Compute Brier score and Expected Calibration Error.

    Args:
        predicted_probs: Predicted probability for each instance.
        observed: Whether each instance was a positive outcome.
        n_bins: Number of bins for ECE computation.
        strategy: Strategy name for labeling.
    """
    n = len(predicted_probs)
    if n == 0:
        return CalibrationResult(strategy=strategy, brier_score=0.0, expected_calibration_error=0.0)

    # Brier score: mean squared error of probability predictions
    brier = sum(
        (p - (1.0 if o else 0.0)) ** 2
        for p, o in zip(predicted_probs, observed)
    ) / n

    # Expected Calibration Error (ECE)
    bin_width = 1.0 / n_bins
    bins: list[dict] = []
    ece = 0.0

    for b in range(n_bins):
        lo = b * bin_width
        hi = lo + bin_width

        bin_preds = []
        bin_obs = []
        for p, o in zip(predicted_probs, observed):
            if lo <= p < hi or (b == n_bins - 1 and p == hi):
                bin_preds.append(p)
                bin_obs.append(1.0 if o else 0.0)

        if bin_preds:
            pred_mean = sum(bin_preds) / len(bin_preds)
            obs_freq = sum(bin_obs) / len(bin_obs)
            count = len(bin_preds)
            ece += (count / n) * abs(obs_freq - pred_mean)
            bins.append({
                "predicted_mean": pred_mean,
                "observed_freq": obs_freq,
                "count": count,
                "bin_range": (lo, hi),
            })

    return CalibrationResult(
        strategy=strategy,
        brier_score=brier,
        expected_calibration_error=ece,
        bins=bins,
    )
