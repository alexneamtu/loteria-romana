import random


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
        main = sorted(rng.sample(range(1, 50), 6))
        key = tuple(main)
        if key in seen:
            continue
        seen.add(key)
        lines.append(main)
    return lines


def build_frequency(draws):
    freq = {n: 0 for n in range(1, 50)}
    for main in draws:
        for n in main:
            freq[n] += 1
    return freq


def generate_frequency_lines(count: int, freq: dict[int, int], rng=None):
    rng = rng or random.SystemRandom()
    numbers = list(range(1, 50))
    weights = [freq.get(n, 0) + 1 for n in numbers]
    lines = []
    seen = set()
    while len(lines) < count:
        main = _sample_weighted(numbers, weights, 6, rng)
        key = tuple(main)
        if key in seen:
            continue
        seen.add(key)
        lines.append(main)
    return lines
