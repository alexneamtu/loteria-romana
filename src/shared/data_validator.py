"""Data validator for lottery draw datasets.

Validates draw data for structural correctness and schedule consistency.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import median

from shared.game_config import GameConfig


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: str  # "error" | "warning"
    game: str
    description: str
    draw_date: str | None = None
    details: dict = field(default_factory=dict)


def _check_duplicate_dates(draws, game_name):
    seen = {}
    issues = []
    for draw in draws:
        if draw.date in seen:
            issues.append(ValidationIssue(
                severity="error",
                game=game_name,
                description=f"Duplicate date found: {draw.date}",
                draw_date=draw.date,
                details={"date": draw.date},
            ))
        else:
            seen[draw.date] = True
    return issues


def _check_number_range(draws, config):
    issues = []
    pool_min = config.pool_min
    pool_max = config.pool_max
    for draw in draws:
        out_of_range = [
            n for n in draw.main_numbers
            if n < pool_min or n > pool_max
        ]
        if out_of_range:
            issues.append(ValidationIssue(
                severity="error",
                game=config.name,
                description=(
                    f"Numbers out of range [{pool_min}, {pool_max}]: "
                    f"{out_of_range}"
                ),
                draw_date=draw.date,
                details={"out_of_range": out_of_range},
            ))
    return issues


def _check_number_count(draws, config):
    issues = []
    expected = config.numbers_drawn
    for draw in draws:
        actual = len(draw.main_numbers)
        if actual != expected:
            issues.append(ValidationIssue(
                severity="error",
                game=config.name,
                description=(
                    f"Wrong count of numbers: expected {expected}, "
                    f"got {actual}"
                ),
                draw_date=draw.date,
                details={"expected": expected, "actual": actual},
            ))
    return issues


def _check_duplicate_numbers(draws, config):
    issues = []
    for draw in draws:
        if len(draw.main_numbers) != len(set(draw.main_numbers)):
            duplicates = [
                n for n in set(draw.main_numbers)
                if draw.main_numbers.count(n) > 1
            ]
            issues.append(ValidationIssue(
                severity="error",
                game=config.name,
                description=(
                    f"Duplicate numbers within draw: {duplicates}"
                ),
                draw_date=draw.date,
                details={"duplicates": duplicates},
            ))
    return issues


def _check_chronological_order(draws, config):
    issues = []
    for i in range(1, len(draws)):
        if draws[i].date < draws[i - 1].date:
            issues.append(ValidationIssue(
                severity="error",
                game=config.name,
                description=(
                    f"Non-chronological order: {draws[i - 1].date} "
                    f"followed by {draws[i].date}"
                ),
                draw_date=draws[i].date,
                details={
                    "previous": draws[i - 1].date,
                    "current": draws[i].date,
                },
            ))
    return issues


_MIN_GAP_THRESHOLD_DAYS = 14


def _check_schedule_gaps(draws, config):
    if len(draws) < 3:
        return []

    gaps = []
    for i in range(1, len(draws)):
        current = date.fromisoformat(draws[i].date)
        previous = date.fromisoformat(draws[i - 1].date)
        gap = (current - previous).days
        gaps.append((i, gap))

    gap_values = [g for _, g in gaps]
    median_gap = median(gap_values)
    threshold = max(median_gap * 2.5, _MIN_GAP_THRESHOLD_DAYS)

    issues = []
    for i, gap in gaps:
        if gap > threshold:
            issues.append(ValidationIssue(
                severity="warning",
                game=config.name,
                description=(
                    f"Schedule gap of {gap} days between "
                    f"{draws[i - 1].date} and {draws[i].date} "
                    f"(threshold: {threshold:.0f} days)"
                ),
                draw_date=draws[i].date,
                details={
                    "gap_days": gap,
                    "median_gap": median_gap,
                    "threshold": threshold,
                },
            ))
    return issues


def validate_draws(draws, game_config):
    """Validate a list of draw objects against a game configuration.

    Returns a list of ValidationIssue instances describing any problems found.
    """
    issues = []
    issues.extend(_check_duplicate_dates(draws, game_config.name))
    issues.extend(_check_number_range(draws, game_config))
    issues.extend(_check_number_count(draws, game_config))
    issues.extend(_check_duplicate_numbers(draws, game_config))
    issues.extend(_check_chronological_order(draws, game_config))
    issues.extend(_check_schedule_gaps(draws, game_config))
    return issues
