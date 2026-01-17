"""Game configuration for lottery games.

This module provides a unified configuration system for all lottery games,
eliminating the need for hardcoded values throughout the codebase.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Configuration for a lottery game.

    Attributes:
        name: Human-readable game name
        pool_min: Minimum number in the pool (usually 1)
        pool_max: Maximum number in the pool (e.g., 45, 49, 40)
        numbers_drawn: How many numbers the lottery draws
        numbers_to_pick: How many numbers the player picks
        has_bonus: Whether there's a bonus number (e.g., Joker)
        bonus_min: Minimum bonus number (if applicable)
        bonus_max: Maximum bonus number (if applicable)
        min_match_for_prize: Minimum matches needed to win
    """
    name: str
    pool_min: int
    pool_max: int
    numbers_drawn: int
    numbers_to_pick: int
    has_bonus: bool = False
    bonus_min: int = 0
    bonus_max: int = 0
    min_match_for_prize: int = 3

    @property
    def pool_size(self) -> int:
        """Total numbers in the pool."""
        return self.pool_max - self.pool_min + 1

    @property
    def pool_range(self) -> range:
        """Range of valid numbers."""
        return range(self.pool_min, self.pool_max + 1)


# Pre-defined game configurations
JOKER_CONFIG = GameConfig(
    name="Joker",
    pool_min=1,
    pool_max=45,
    numbers_drawn=5,
    numbers_to_pick=5,
    has_bonus=True,
    bonus_min=1,
    bonus_max=20,
    min_match_for_prize=2,  # 2+ main numbers for a prize
)

LOTO_649_CONFIG = GameConfig(
    name="Loto 6/49",
    pool_min=1,
    pool_max=49,
    numbers_drawn=6,
    numbers_to_pick=6,
    has_bonus=False,
    min_match_for_prize=3,  # 3+ matches for a prize
)

LOTO_540_CONFIG = GameConfig(
    name="Loto 5/40",
    pool_min=1,
    pool_max=40,
    numbers_drawn=6,  # Lottery draws 6 numbers
    numbers_to_pick=5,  # Player picks 5 numbers
    has_bonus=False,
    min_match_for_prize=4,  # 4+ matches for a prize
)


def get_config(game_name: str) -> GameConfig:
    """Get configuration by game name."""
    configs = {
        "joker": JOKER_CONFIG,
        "loto649": LOTO_649_CONFIG,
        "loto_649": LOTO_649_CONFIG,
        "6/49": LOTO_649_CONFIG,
        "loto540": LOTO_540_CONFIG,
        "loto_540": LOTO_540_CONFIG,
        "5/40": LOTO_540_CONFIG,
    }
    return configs[game_name.lower()]
