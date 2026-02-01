"""Ensemble blending strategy for lottery number generation.

Instead of picking a single best strategy (winner-takes-all), this
module allocates picks proportionally across multiple strategies
weighted by their backtest performance.  A chi-square bias detector
dynamically adjusts how aggressively non-random strategies are used.
"""

import math
import random

from .bayesian import BayesianScorer
from .bias_detection import chi_square_uniformity_test
from .cooccurrence import CooccurrenceStrategy
from .game_config import GameConfig
from .game_strategies import (
    generate_frequency_picks,
    generate_random_picks,
    is_prize_winner,
    _sample_weighted,
)
from .math_utils import softmax
from .recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE


def _score_random(
    config: GameConfig,
    draws: list[list[int]],
    rng: random.Random,
) -> int:
    """Count prize-winning random picks across all draws."""
    wins = 0
    for draw in draws:
        pick = sorted(rng.sample(list(config.pool_range), config.numbers_to_pick))
        if is_prize_winner(config, pick, draw):
            wins += 1
    return wins


def _score_frequency(
    config: GameConfig,
    draws: list[list[int]],
    rng: random.Random,
    half_life: float,
    half_life_mode: str,
    draw_dates: list[str] | None,
) -> int:
    """Count prize-winning frequency picks using walk-forward testing."""
    wins = 0
    for idx in range(len(draws)):
        picks = generate_frequency_picks(
            config,
            draws[:idx],
            1,
            rng,
            half_life=half_life,
            draw_dates=draw_dates[:idx] if draw_dates else None,
            half_life_mode=half_life_mode,
        )
        if picks and is_prize_winner(config, picks[0], draws[idx]):
            wins += 1
    return wins


def _score_strategy_object(
    config: GameConfig,
    draws: list[list[int]],
    strategy,
    rng: random.Random,
    draw_dates: list[str] | None,
    half_life_mode: str,
) -> int:
    """Score a strategy that implements the Strategy protocol."""
    wins = 0
    min_train = 3
    for idx in range(min_train, len(draws)):
        picks = strategy.generate(
            draws[:idx],
            1,
            rng,
            draw_dates=draw_dates[:idx] if draw_dates else None,
            half_life_mode=half_life_mode,
        )
        if picks and is_prize_winner(config, picks[0], draws[idx]):
            wins += 1
    return wins


def _allocate_counts(weights: dict[str, float], total: int) -> dict[str, int]:
    """Allocate integer pick counts proportionally to weights.

    Uses largest-remainder method to ensure counts sum to total.
    """
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        names = list(weights)
        base = total // len(names)
        allocation = {n: base for n in names}
        for i in range(total - base * len(names)):
            allocation[names[i]] += 1
        return allocation

    raw = {name: w / weight_sum * total for name, w in weights.items()}
    floored = {name: int(v) for name, v in raw.items()}
    remainders = {name: raw[name] - floored[name] for name in raw}

    allocated = sum(floored.values())
    deficit = total - allocated

    for name in sorted(remainders, key=remainders.get, reverse=True):
        if deficit <= 0:
            break
        floored[name] += 1
        deficit -= 1

    return floored


def generate_blended_picks(
    config: GameConfig,
    draws: list[list[int]],
    count: int,
    rng: random.Random | None = None,
    half_life: float = DEFAULT_HALF_LIFE,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    draw_dates: list[str] | None = None,
) -> list[list[int]]:
    """Generate picks by blending multiple strategies proportionally.

    Steps:
    1. Run chi-square bias detection to gauge data uniformity.
    2. Backtest each strategy on historical data.
    3. Apply softmax to convert scores into allocation weights.
    4. Adjust weights using bias strength (boost random if no bias).
    5. Allocate pick counts proportionally and generate.

    Args:
        config: Game configuration.
        draws: Historical draws (main numbers only).
        count: Number of picks to generate.
        rng: Random number generator.
        half_life: Recency half-life.
        half_life_mode: "draws" or "days".
        draw_dates: Optional ISO date strings.

    Returns:
        List of blended picks.
    """
    rng = rng or random.SystemRandom()

    if len(draws) < 5:
        return generate_random_picks(config, count, rng)

    bias = chi_square_uniformity_test(
        draws, config.pool_size, config.numbers_drawn,
    )

    bayesian = BayesianScorer(
        config.pool_size,
        config.numbers_to_pick,
        half_life=half_life,
        half_life_mode=half_life_mode,
    )
    cooccurrence = CooccurrenceStrategy(
        config.pool_size,
        config.numbers_to_pick,
        half_life=half_life,
        half_life_mode=half_life_mode,
    )

    scores = {
        "random": max(_score_random(config, draws, rng), 1),
        "frequency": max(
            _score_frequency(
                config, draws, rng, half_life, half_life_mode, draw_dates,
            ),
            1,
        ),
        "bayesian": max(
            _score_strategy_object(
                config, draws, bayesian, rng, draw_dates, half_life_mode,
            ),
            1,
        ),
        "cooccurrence": max(
            _score_strategy_object(
                config, draws, cooccurrence, rng, draw_dates, half_life_mode,
            ),
            1,
        ),
    }

    raw_weights = softmax([float(s) for s in scores.values()])
    weights = dict(zip(scores.keys(), raw_weights))

    if not bias.significant:
        random_boost = 0.3 * (1.0 - bias.bias_strength)
        weights["random"] += random_boost

    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    allocation = _allocate_counts(weights, count)

    lines: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    random_lines = generate_random_picks(config, allocation.get("random", 0) + count, rng)
    added_random = 0
    for pick in random_lines:
        if added_random >= allocation.get("random", 0):
            break
        key = tuple(pick)
        if key not in seen:
            seen.add(key)
            lines.append(pick)
            added_random += 1

    freq_lines = generate_frequency_picks(
        config, draws, allocation.get("frequency", 0) + count, rng,
        half_life=half_life, draw_dates=draw_dates, half_life_mode=half_life_mode,
    )
    added_freq = 0
    for pick in freq_lines:
        if added_freq >= allocation.get("frequency", 0):
            break
        key = tuple(pick)
        if key not in seen:
            seen.add(key)
            lines.append(pick)
            added_freq += 1

    bayes_lines = bayesian.generate(
        draws, allocation.get("bayesian", 0) + count, rng,
        draw_dates=draw_dates, half_life_mode=half_life_mode,
    )
    added_bayes = 0
    for pick in bayes_lines:
        if added_bayes >= allocation.get("bayesian", 0):
            break
        key = tuple(pick)
        if key not in seen:
            seen.add(key)
            lines.append(pick)
            added_bayes += 1

    cooc_lines = cooccurrence.generate(
        draws, allocation.get("cooccurrence", 0) + count, rng,
        draw_dates=draw_dates, half_life_mode=half_life_mode,
    )
    added_cooc = 0
    for pick in cooc_lines:
        if added_cooc >= allocation.get("cooccurrence", 0):
            break
        key = tuple(pick)
        if key not in seen:
            seen.add(key)
            lines.append(pick)
            added_cooc += 1

    while len(lines) < count:
        extra = generate_random_picks(config, 1, rng)
        key = tuple(extra[0])
        if key not in seen:
            seen.add(key)
            lines.append(extra[0])

    return lines[:count]
