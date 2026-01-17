import random

from .backtest import pick_best_strategy
from .neural import generate_neural_lines
from .strategies import build_frequency, generate_random_lines, generate_frequency_lines
from shared.recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE


def generate_picks(
    draws,
    count=2,
    rng=None,
    half_life: float = DEFAULT_HALF_LIFE,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    draw_dates: list[str] | None = None,
):
    rng = rng or random.SystemRandom()
    best = pick_best_strategy(
        draws,
        rng=rng,
        half_life=half_life,
        half_life_mode=half_life_mode,
        draw_dates=draw_dates,
    )

    if best == "neural":
        return generate_neural_lines(
            draws,
            count,
            rng=rng,
            draw_dates=draw_dates,
            half_life=half_life,
            half_life_mode=half_life_mode,
        )
    if best == "frequency":
        freq = build_frequency(
            draws,
            half_life=half_life,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
        return generate_frequency_lines(count, freq, rng=rng)
    return generate_random_lines(count, rng=rng)
