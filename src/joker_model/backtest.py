import random

from .metrics import is_joker_prize
from .neural import generate_neural_lines
from .strategies import build_frequency, generate_random_lines, generate_frequency_lines


def _score_strategy(draws, generator, rng):
    wins = 0
    for main, joker in draws:
        lines = generator(1, rng=rng)
        line_main, line_joker = lines[0]
        main_matches = len(set(line_main) & set(main))
        joker_match = line_joker == joker
        if is_joker_prize(main_matches, joker_match):
            wins += 1
    return wins


def _score_frequency(draws, rng):
    wins = 0
    for idx, (main, joker) in enumerate(draws):
        freq = build_frequency(draws[:idx])
        lines = generate_frequency_lines(1, freq, rng=rng)
        line_main, line_joker = lines[0]
        main_matches = len(set(line_main) & set(main))
        joker_match = line_joker == joker
        if is_joker_prize(main_matches, joker_match):
            wins += 1
    return wins


def pick_best_strategy(draws, rng=None):
    rng = rng or random.Random()
    scores = {
        "random": _score_strategy(draws, generate_random_lines, rng),
        "frequency": _score_frequency(draws, rng),
    }

    if len(draws) >= 2:
        scores["neural"] = _score_strategy(draws, lambda c, rng=None: generate_neural_lines(draws, c, rng=rng), rng)
    else:
        scores["neural"] = scores["random"]
    return max(scores, key=scores.get)
