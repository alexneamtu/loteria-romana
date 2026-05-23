import random

from .metrics import is_loto_540_prize
from .neural import generate_neural_lines
from .strategies import build_frequency, generate_random_lines, generate_frequency_lines
from shared.recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE


def _score_strategy(draws, generator, rng):
    """Score a strategy by counting wins.

    In 5/40: player picks 5, lottery draws 6 from 1-40.
    Win by matching 4+ of the 6 drawn numbers.
    """
    wins = 0
    for main in draws:
        lines = generator(1, rng=rng)
        line_main = lines[0]
        # Player's 5 picks vs lottery's 6 drawn numbers
        main_matches = len(set(line_main) & set(main))
        if is_loto_540_prize(main_matches):
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
        if is_loto_540_prize(main_matches):
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

    if len(draws) >= 3:
        scores["neural"] = _score_neural(
            draws,
            rng,
            half_life=half_life,
            half_life_mode=half_life_mode,
            draw_dates=draw_dates,
        )
    else:
        scores["neural"] = scores["random"]
    return max(scores, key=scores.get)


def _score_neural(
    draws,
    rng,
    half_life: float = DEFAULT_HALF_LIFE,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    draw_dates: list[str] | None = None,
):
    """Walk-forward score: train neural on draws[:idx] and predict draws[idx]."""
    wins = 0
    for idx, main in enumerate(draws):
        if idx < 2:
            continue
        lines = generate_neural_lines(
            draws[:idx],
            1,
            rng=rng,
            draw_dates=draw_dates[:idx] if draw_dates else None,
            half_life=half_life,
            half_life_mode=half_life_mode,
        )
        if not lines:
            continue
        line_main = lines[0]
        main_matches = len(set(line_main) & set(main))
        if is_loto_540_prize(main_matches):
            wins += 1
    return wins
