import random

from .metrics import is_loto_649_prize
from .neural import generate_neural_lines
from .strategies import build_frequency, generate_random_lines, generate_frequency_lines
from shared.recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE


def _score_strategy(draws, generator, rng):
    wins = 0
    for main in draws:
        lines = generator(1, rng=rng)
        line_main = lines[0]
        main_matches = len(set(line_main) & set(main))
        if is_loto_649_prize(main_matches):
            wins += 1
    return wins


def _score_frequency(
    draws,
    rng,
    half_life: float = DEFAULT_HALF_LIFE,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    draw_dates: list[str] | None = None,
):
    wins = 0
    for idx, main in enumerate(draws):
        freq = build_frequency(
            draws[:idx],
            half_life=half_life,
            draw_dates=draw_dates[:idx] if draw_dates else None,
            half_life_mode=half_life_mode,
        )
        lines = generate_frequency_lines(1, freq, rng=rng)
        line_main = lines[0]
        main_matches = len(set(line_main) & set(main))
        if is_loto_649_prize(main_matches):
            wins += 1
    return wins


def pick_best_strategy(
    draws,
    rng=None,
    half_life: float = DEFAULT_HALF_LIFE,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    draw_dates: list[str] | None = None,
):
    rng = rng or random.Random()
    scores = {
        "random": _score_strategy(draws, generate_random_lines, rng),
        "frequency": _score_frequency(
            draws,
            rng,
            half_life=half_life,
            half_life_mode=half_life_mode,
            draw_dates=draw_dates,
        ),
    }

    if len(draws) >= 2:
        scores["neural"] = _score_strategy(
            draws,
            lambda c, rng=None: generate_neural_lines(
                draws,
                c,
                rng=rng,
                draw_dates=draw_dates,
                half_life=half_life,
                half_life_mode=half_life_mode,
            ),
            rng,
        )
    else:
        scores["neural"] = scores["random"]
    return max(scores, key=scores.get)
