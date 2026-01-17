import random

from .backtest import pick_best_strategy
from .neural import generate_neural_lines
from .strategies import build_frequency, generate_random_lines, generate_frequency_lines
from shared.recency import DEFAULT_HALF_LIFE


def generate_picks(draws, count=2, rng=None, half_life: float = DEFAULT_HALF_LIFE):
    rng = rng or random.SystemRandom()
    best = pick_best_strategy(draws, rng=rng, half_life=half_life)

    if best == "neural":
        return generate_neural_lines(draws, count, rng=rng)
    if best == "frequency":
        freq = build_frequency(draws, half_life=half_life)
        return generate_frequency_lines(count, freq, rng=rng)
    return generate_random_lines(count, rng=rng)
