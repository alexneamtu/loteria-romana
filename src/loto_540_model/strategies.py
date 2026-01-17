import random

SUPER_NOROC_MAX = 999_999


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


def generate_random_lines(count: int, rng=None, include_super_noroc: bool = True):
    """Generate random Loto 5/40 picks.

    Players pick 5 numbers from 1-40.
    """
    rng = rng or random.SystemRandom()
    lines = []
    seen = set()
    while len(lines) < count:
        main = sorted(rng.sample(range(1, 41), 5))  # 5 numbers from 1-40
        super_noroc = rng.randint(0, SUPER_NOROC_MAX) if include_super_noroc else None
        key = tuple(main) if super_noroc is None else tuple(main) + (super_noroc,)
        if key in seen:
            continue
        seen.add(key)
        lines.append((main, super_noroc))
    return lines


def build_frequency(draws):
    """Build frequency map from historical draws.

    Note: draws contain 6 drawn numbers, we track all of them.
    """
    freq = {n: 0 for n in range(1, 41)}
    for main, _ in draws:
        for n in main:
            freq[n] += 1
    return freq


def generate_frequency_lines(count: int, freq: dict[int, int], rng=None, include_super_noroc: bool = True):
    """Generate frequency-weighted Loto 5/40 picks.

    Players pick 5 numbers from 1-40, weighted by historical frequency.
    """
    rng = rng or random.SystemRandom()
    numbers = list(range(1, 41))
    weights = [freq.get(n, 0) + 1 for n in numbers]
    lines = []
    seen = set()
    while len(lines) < count:
        main = _sample_weighted(numbers, weights, 5, rng)  # Pick 5 numbers
        super_noroc = rng.randint(0, SUPER_NOROC_MAX) if include_super_noroc else None
        key = tuple(main) if super_noroc is None else tuple(main) + (super_noroc,)
        if key in seen:
            continue
        seen.add(key)
        lines.append((main, super_noroc))
    return lines
