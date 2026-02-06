"""Monte Carlo validation for lottery strategies.

Generates synthetic fair-lottery datasets to verify that no strategy
beats random selection on truly uniform data. If a strategy shows
significant edge on real data but NOT on synthetic data, the edge
may be genuine. If it also shows edge on synthetic data, it's likely
an artifact of overfitting.
"""

import math
import random

from .game_config import GameConfig


def generate_synthetic_draws(
    config: GameConfig,
    count: int,
    rng: random.Random,
) -> list[list[int]]:
    """Generate synthetic draws from a perfectly uniform lottery."""
    pool = list(config.pool_range)
    draws = []
    for _ in range(count):
        draw = sorted(rng.sample(pool, config.numbers_drawn))
        draws.append(draw)
    return draws


def monte_carlo_validate(
    config: GameConfig,
    strategy_name: str,
    strategy_win_rate: float,
    num_simulations: int = 50,
    draws_per_simulation: int = 200,
    rng: random.Random | None = None,
) -> dict:
    """Validate a strategy's win rate against synthetic uniform data."""
    rng = rng or random.Random()

    synthetic_win_rates = []
    pool = list(config.pool_range)

    for _ in range(num_simulations):
        draws = generate_synthetic_draws(config, draws_per_simulation, rng)
        wins = 0
        for draw in draws:
            pick = sorted(rng.sample(pool, config.numbers_to_pick))
            matches = len(set(pick) & set(draw))
            if matches >= config.min_match_for_prize:
                wins += 1
        synthetic_win_rates.append(wins / draws_per_simulation)

    mean_rate = sum(synthetic_win_rates) / len(synthetic_win_rates)
    std_rate = math.sqrt(
        sum((r - mean_rate) ** 2 for r in synthetic_win_rates)
        / len(synthetic_win_rates)
    ) if len(synthetic_win_rates) > 1 else 0.0

    upper_bound = mean_rate + 2 * std_rate
    is_plausible = strategy_win_rate <= upper_bound

    return {
        "strategy_name": strategy_name,
        "strategy_win_rate": strategy_win_rate,
        "synthetic_mean_win_rate": mean_rate,
        "synthetic_std_win_rate": std_rate,
        "synthetic_upper_2sigma": upper_bound,
        "strategy_is_plausible": is_plausible,
        "num_simulations": num_simulations,
        "conclusion": (
            f"{strategy_name} win rate ({strategy_win_rate:.4f}) is within "
            f"synthetic range ({mean_rate:.4f} +/- {2*std_rate:.4f}). "
            "Performance is consistent with random chance."
            if is_plausible
            else f"{strategy_name} win rate ({strategy_win_rate:.4f}) exceeds "
            f"synthetic upper bound ({upper_bound:.4f}). "
            "Strategy may have found genuine edge OR is overfitting."
        ),
    }
