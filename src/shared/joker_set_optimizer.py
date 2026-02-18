"""Helpers for improving Joker ticket-set quality.

Phase 1 focuses on two practical levers:
- maximize Joker-number coverage across a ticket set
- reduce duplicated/redundant main-number lines
"""

import random

from .crowding import average_anti_crowding_score
from .portfolio import optimize_ticket_portfolio, diversity_score


def assign_max_coverage_jokers(
    count: int,
    rng: random.Random | None = None,
    joker_pool: int = 20,
) -> list[int]:
    """Assign Joker numbers with maximal unique coverage before repeats."""
    if count <= 0:
        return []
    if joker_pool <= 0:
        raise ValueError("joker_pool must be positive")

    rng = rng or random.SystemRandom()
    assigned: list[int] = []
    base_pool = list(range(1, joker_pool + 1))

    while len(assigned) < count:
        cycle = base_pool.copy()
        rng.shuffle(cycle)
        assigned.extend(cycle)

    return assigned[:count]


def optimize_main_ticket_set(
    candidates: list[list[int]],
    select_count: int,
    pool_size: int,
    numbers_to_pick: int,
    rng: random.Random | None = None,
    anti_crowding_weight: float = 0.0,
) -> list[list[int]]:
    """Select a diverse, de-duplicated subset of main-number tickets."""
    if select_count <= 0:
        return []
    if pool_size <= 0 or numbers_to_pick <= 0:
        raise ValueError("pool_size and numbers_to_pick must be positive")
    if numbers_to_pick > pool_size:
        raise ValueError("numbers_to_pick cannot exceed pool_size")

    rng = rng or random.SystemRandom()

    unique_candidates: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for line in candidates:
        normalized = sorted(line)
        if len(normalized) != numbers_to_pick:
            continue
        if any(n < 1 or n > pool_size for n in normalized):
            continue
        key = tuple(normalized)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(normalized)

    target_candidates = max(select_count * 3, select_count)
    while len(unique_candidates) < target_candidates:
        generated = sorted(rng.sample(range(1, pool_size + 1), numbers_to_pick))
        key = tuple(generated)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(generated)

    selected = optimize_ticket_portfolio(
        unique_candidates,
        select_count=select_count,
        pool_size=pool_size,
    )

    if anti_crowding_weight > 0 and selected:
        def objective(lines: list[list[int]]) -> float:
            return (
                diversity_score(lines, pool_size)
                + anti_crowding_weight
                * average_anti_crowding_score(lines, pool_size)
            )

        current_score = objective(selected)
        selected_keys = {tuple(line) for line in selected}
        candidate_pool = [
            line for line in unique_candidates if tuple(line) not in selected_keys
        ]

        for idx in range(len(selected)):
            best_score = current_score
            best_candidate = None
            for candidate in candidate_pool:
                trial = list(selected)
                trial[idx] = candidate
                if len({tuple(line) for line in trial}) < len(trial):
                    continue
                score = objective(trial)
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate is None:
                continue

            previous = selected[idx]
            selected[idx] = best_candidate
            candidate_pool.remove(best_candidate)
            candidate_pool.append(previous)
            current_score = best_score

    # Defensive fill if selector returns short output for any reason.
    selected_keys = {tuple(line) for line in selected}
    while len(selected) < select_count:
        generated = sorted(rng.sample(range(1, pool_size + 1), numbers_to_pick))
        key = tuple(generated)
        if key in selected_keys:
            continue
        selected_keys.add(key)
        selected.append(generated)

    return selected[:select_count]
