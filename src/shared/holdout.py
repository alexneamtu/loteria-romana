"""Holdout set management for strategy evaluation.

Reserves the most recent N draws as a final holdout set
that is never used during strategy development or backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


@dataclass
class TemporalSplit:
    """Result of a temporal train/holdout split on draw objects."""

    train: list
    holdout: list
    holdout_size: int
    split_date: Any  # date string of first holdout draw


def temporal_holdout_split(
    draws: list, holdout_size: int = 100
) -> TemporalSplit:
    """Split draw objects into train and holdout by chronological order.

    Holdout is always the last *holdout_size* draws. If holdout_size
    exceeds 80 % of the data the holdout is capped at 20 % to preserve
    sufficient training data.  No shuffling is performed.

    Args:
        draws: Draw objects ordered oldest-first. Each must have a
            ``date`` attribute.
        holdout_size: Number of most-recent draws to reserve.

    Returns:
        A ``TemporalSplit`` with disjoint train / holdout lists.
    """
    total = len(draws)

    if holdout_size > int(total * 0.8):
        holdout_size = int(total * 0.2)

    split_index = total - holdout_size
    train = draws[:split_index]
    holdout = draws[split_index:]

    split_date = str(holdout[0].date) if holdout else None

    return TemporalSplit(
        train=train,
        holdout=holdout,
        holdout_size=holdout_size,
        split_date=split_date,
    )
