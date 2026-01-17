import random

from shared.recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE, draw_weights


def _sample_weighted(numbers, weights, count, rng):
    chosen = []
    pool = list(numbers)
    pool_weights = list(weights)
    for _ in range(count):
        pick = rng.choices(pool, weights=pool_weights, k=1)[0]
        idx = pool.index(pick)
        chosen.append(pick)
        pool.pop(idx)
        pool_weights.pop(idx)
    return sorted(chosen)


def generate_random_lines(count: int, rng=None):
    rng = rng or random.SystemRandom()
    lines = []
    seen = set()
    while len(lines) < count:
        main = sorted(rng.sample(range(1, 46), 5))
        joker = rng.randint(1, 20)
        key = tuple(main) + (joker,)
        if key in seen:
            continue
        seen.add(key)
        lines.append((main, joker))
    return lines


def build_frequency(
    draws,
    half_life: float = DEFAULT_HALF_LIFE,
    draw_dates: list[str] | None = None,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
):
    freq = {n: 0.0 for n in range(1, 46)}
    if not draws:
        return freq

    weights = draw_weights(
        len(draws),
        half_life,
        draw_dates=draw_dates,
        mode=half_life_mode,
    )
    for (main, _), weight in zip(draws, weights):
        for n in main:
            freq[n] += weight
    return freq


def generate_frequency_lines(count: int, freq: dict[int, int], rng=None):
    rng = rng or random.SystemRandom()
    numbers = list(range(1, 46))
    weights = [freq.get(n, 0) + 1 for n in numbers]
    lines = []
    seen = set()
    while len(lines) < count:
        main = _sample_weighted(numbers, weights, 5, rng)
        joker = rng.randint(1, 20)
        key = tuple(main) + (joker,)
        if key in seen:
            continue
        seen.add(key)
        lines.append((main, joker))
    return lines
