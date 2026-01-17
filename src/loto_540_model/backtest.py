import random

from .metrics import is_loto_540_prize
from .neural import generate_neural_lines
from .strategies import build_frequency, generate_random_lines, generate_frequency_lines


def _score_strategy(draws, generator, rng, include_super_noroc: bool):
    """Score a strategy by counting wins.

    In 5/40: player picks 5, lottery draws 6 from 1-40.
    Win by matching 4+ of the 6 drawn numbers.
    """
    wins = 0
    for main, super_noroc in draws:
        lines = generator(1, rng=rng, include_super_noroc=include_super_noroc)
        line_main, line_super_noroc = lines[0]
        # Player's 5 picks vs lottery's 6 drawn numbers
        main_matches = len(set(line_main) & set(main))
        super_noroc_match = line_super_noroc == super_noroc if include_super_noroc else False
        if is_loto_540_prize(main_matches, super_noroc_match, include_super_noroc=include_super_noroc):
            wins += 1
    return wins


def _score_frequency(draws, rng, include_super_noroc: bool):
    wins = 0
    for idx, (main, super_noroc) in enumerate(draws):
        freq = build_frequency(draws[:idx])
        lines = generate_frequency_lines(1, freq, rng=rng, include_super_noroc=include_super_noroc)
        line_main, line_super_noroc = lines[0]
        main_matches = len(set(line_main) & set(main))
        super_noroc_match = line_super_noroc == super_noroc if include_super_noroc else False
        if is_loto_540_prize(main_matches, super_noroc_match, include_super_noroc=include_super_noroc):
            wins += 1
    return wins


def pick_best_strategy(draws, rng=None, include_super_noroc: bool = True):
    rng = rng or random.Random()
    scores = {
        "random": _score_strategy(draws, generate_random_lines, rng, include_super_noroc),
        "frequency": _score_frequency(draws, rng, include_super_noroc),
    }

    if len(draws) >= 2:
        scores["neural"] = _score_strategy(
            draws,
            lambda c, rng=None, include_super_noroc=True: generate_neural_lines(
                draws, c, rng=rng, include_super_noroc=include_super_noroc
            ),
            rng,
            include_super_noroc,
        )
    else:
        scores["neural"] = scores["random"]
    return max(scores, key=scores.get)
