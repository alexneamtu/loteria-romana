import random

from .metrics import is_loto_649_prize
from .neural import generate_neural_lines
from .strategies import build_frequency, generate_random_lines, generate_frequency_lines


def _score_strategy(draws, generator, rng, include_noroc: bool):
    wins = 0
    for main, noroc in draws:
        lines = generator(1, rng=rng, include_noroc=include_noroc)
        line_main, line_noroc = lines[0]
        main_matches = len(set(line_main) & set(main))
        noroc_match = line_noroc == noroc if include_noroc else False
        if is_loto_649_prize(main_matches, noroc_match, include_noroc=include_noroc):
            wins += 1
    return wins


def _score_frequency(draws, rng, include_noroc: bool):
    wins = 0
    for idx, (main, noroc) in enumerate(draws):
        freq = build_frequency(draws[:idx])
        lines = generate_frequency_lines(1, freq, rng=rng, include_noroc=include_noroc)
        line_main, line_noroc = lines[0]
        main_matches = len(set(line_main) & set(main))
        noroc_match = line_noroc == noroc if include_noroc else False
        if is_loto_649_prize(main_matches, noroc_match, include_noroc=include_noroc):
            wins += 1
    return wins


def pick_best_strategy(draws, rng=None, include_noroc: bool = True):
    rng = rng or random.Random()
    scores = {
        "random": _score_strategy(draws, generate_random_lines, rng, include_noroc),
        "frequency": _score_frequency(draws, rng, include_noroc),
    }

    if len(draws) >= 2:
        scores["neural"] = _score_strategy(
            draws,
            lambda c, rng=None, include_noroc=True: generate_neural_lines(
                draws, c, rng=rng, include_noroc=include_noroc
            ),
            rng,
            include_noroc,
        )
    else:
        scores["neural"] = scores["random"]
    return max(scores, key=scores.get)
