"""Unified strategy implementations for all lottery games.

This module provides game-agnostic strategy implementations that use
GameConfig to adapt to different lottery rules.
"""

import random
from typing import Protocol

from .game_config import GameConfig


class StrategyProtocol(Protocol):
    """Protocol for lottery number generation strategies."""

    def generate(
        self,
        config: GameConfig,
        draws: list,
        count: int,
        rng: random.Random | None = None,
    ) -> list[list[int]]:
        """Generate lottery number picks."""
        ...


def _sample_weighted(
    numbers: list[int],
    weights: list[float],
    count: int,
    rng: random.Random,
) -> list[int]:
    """Sample numbers without replacement, weighted by probabilities.

    This is the single implementation used by all games, eliminating
    the duplicated versions in joker_model, loto_649_model, loto_540_model.
    """
    chosen = []
    pool = list(numbers)
    pool_weights = list(weights)
    for _ in range(count):
        pick = rng.choices(pool, weights=pool_weights, k=1)[0]
        idx = pool.index(pick)
        chosen.append(pick)
        pool.pop(idx)
        pool_weights.pop(idx)
    return sorted(chosen)


def generate_random_picks(
    config: GameConfig,
    count: int,
    rng: random.Random | None = None,
) -> list[list[int]]:
    """Generate random lottery picks for any game.

    Args:
        config: Game configuration
        count: Number of picks to generate
        rng: Random number generator (uses SystemRandom if None)

    Returns:
        List of picks, each pick is a sorted list of numbers
    """
    rng = rng or random.SystemRandom()
    lines = []
    seen = set()
    numbers = list(config.pool_range)

    while len(lines) < count:
        pick = sorted(rng.sample(numbers, config.numbers_to_pick))
        key = tuple(pick)
        if key in seen:
            continue
        seen.add(key)
        lines.append(pick)

    return lines


def build_frequency(
    config: GameConfig,
    draws: list[list[int]],
) -> dict[int, int]:
    """Build frequency map from historical draws.

    Args:
        config: Game configuration
        draws: List of main_numbers lists

    Returns:
        Dictionary mapping number -> frequency count
    """
    freq = {n: 0 for n in config.pool_range}
    for main in draws:
        for n in main:
            if n in freq:
                freq[n] += 1
    return freq


def generate_frequency_picks(
    config: GameConfig,
    draws: list[list[int]],
    count: int,
    rng: random.Random | None = None,
) -> list[list[int]]:
    """Generate frequency-weighted lottery picks.

    Numbers that appear more frequently in historical draws
    are more likely to be selected.

    Args:
        config: Game configuration
        draws: Historical draws as main_numbers lists
        count: Number of picks to generate
        rng: Random number generator

    Returns:
        List of picks, each pick is a sorted list of numbers
    """
    rng = rng or random.SystemRandom()
    freq = build_frequency(config, draws)
    numbers = list(config.pool_range)
    weights = [freq.get(n, 0) + 1 for n in numbers]  # +1 smoothing

    lines = []
    seen = set()

    while len(lines) < count:
        pick = _sample_weighted(numbers, weights, config.numbers_to_pick, rng)
        key = tuple(pick)
        if key in seen:
            continue
        seen.add(key)
        lines.append(pick)

    return lines


def is_prize_winner(
    config: GameConfig,
    pick: list[int],
    drawn: list[int],
    bonus_match: bool = False,
) -> bool:
    """Check if a pick wins a prize.

    Args:
        config: Game configuration
        pick: Player's numbers
        drawn: Lottery's drawn numbers
        bonus_match: Whether bonus number matched (for Joker)

    Returns:
        True if the pick wins any prize tier
    """
    matches = len(set(pick) & set(drawn))

    # For Joker, bonus match alone can win
    if config.has_bonus and bonus_match:
        return True

    return matches >= config.min_match_for_prize


def generate_bonus_number(
    config: GameConfig,
    rng: random.Random | None = None,
) -> int | None:
    """Generate a bonus number if the game has one.

    Args:
        config: Game configuration
        rng: Random number generator

    Returns:
        Bonus number or None if game doesn't have bonus
    """
    if not config.has_bonus:
        return None
    rng = rng or random.SystemRandom()
    return rng.randint(config.bonus_min, config.bonus_max)
