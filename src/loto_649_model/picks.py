import random

from .backtest import pick_best_strategy
from .neural import generate_neural_lines
from .strategies import build_frequency, generate_random_lines, generate_frequency_lines


def generate_picks(draws, count=2, rng=None):
    rng = rng or random.SystemRandom()
    best = pick_best_strategy(draws, rng=rng)

    if best == "neural":
        return generate_neural_lines(draws, count, rng=rng)
    if best == "frequency":
        freq = build_frequency(draws)
        return generate_frequency_lines(count, freq, rng=rng)
    return generate_random_lines(count, rng=rng)
