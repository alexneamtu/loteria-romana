"""Holdout set management for strategy evaluation.

Reserves the most recent N draws as a final holdout set
that is never used during strategy development or backtesting.
"""

from dataclasses import dataclass, field


@dataclass
class HoldoutSplit:
    """Result of splitting draws into train and holdout sets."""

    train: list[list[int]]
    holdout: list[list[int]]
    train_dates: list[str] = field(default_factory=list)
    holdout_dates: list[str] = field(default_factory=list)


def split_holdout(
    draws: list[list[int]],
    holdout_size: int = 0,
    holdout_fraction: float = 0.1,
    min_train_size: int = 2,
    dates: list[str] | None = None,
) -> HoldoutSplit:
    """Split draws into training and holdout sets.

    The holdout set is always the most recent draws.
    If holdout_size is provided, it takes precedence over holdout_fraction.

    Args:
        draws: Historical draws (oldest first).
        holdout_size: Exact number of draws to hold out. 0 means use fraction.
        holdout_fraction: Fraction of draws to hold out (default 10%).
        min_train_size: Minimum training draws required. If not enough
            data, returns all draws as training with empty holdout.
        dates: Optional ISO date strings parallel to draws.

    Returns:
        HoldoutSplit with train/holdout draws and optional dates.
    """
    n = len(draws)

    if holdout_size > 0:
        actual_holdout = holdout_size
    else:
        actual_holdout = int(n * holdout_fraction)

    if n - actual_holdout < min_train_size:
        return HoldoutSplit(
            train=list(draws),
            holdout=[],
            train_dates=list(dates) if dates else [],
            holdout_dates=[],
        )

    split_idx = n - actual_holdout
    return HoldoutSplit(
        train=draws[:split_idx],
        holdout=draws[split_idx:],
        train_dates=dates[:split_idx] if dates else [],
        holdout_dates=dates[split_idx:] if dates else [],
    )
