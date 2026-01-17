import random

from .backtest import pick_best_strategy
from .neural import generate_neural_lines
from .strategies import build_frequency, generate_random_lines, generate_frequency_lines


def generate_picks(draws, count=2, rng=None, include_super_noroc: bool = True):
    rng = rng or random.SystemRandom()
    best = pick_best_strategy(draws, rng=rng, include_super_noroc=include_super_noroc)

    if best == "neural":
        return generate_neural_lines(draws, count, rng=rng, include_super_noroc=include_super_noroc)
    if best == "frequency":
        freq = build_frequency(draws)
        return generate_frequency_lines(count, freq, rng=rng, include_super_noroc=include_super_noroc)
    return generate_random_lines(count, rng=rng, include_super_noroc=include_super_noroc)
