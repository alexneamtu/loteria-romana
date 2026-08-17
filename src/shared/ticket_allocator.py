"""Ticket-level budget allocator.

Enumerates feasible (n_joker, n_649, n_540) ticket combinations under
a RON budget and ranks by P(any win) across the combined independent
tickets. Replaces game_recommender.optimize_budget which ranked by
per-variant tickets using stale pricing.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

from .pricing import VARIANTS_PER_TICKET, compute_ticket_cost


_GAMES = ("joker", "loto_649", "loto_540")


@dataclass(frozen=True)
class TicketAllocation:
    tickets: dict[str, int]
    total_cost: float
    p_any_win: float = 0.0

    def __post_init__(self) -> None:
        for g in _GAMES:
            if self.tickets.get(g, 0) < 0:
                raise ValueError(f"negative ticket count for {g}")

    def recompute_total_cost(self) -> float:
        return sum(
            compute_ticket_cost(game) * count
            for game, count in self.tickets.items()
            if count > 0
        )


def _variant_win_prob(game: str) -> float:
    if game == "joker":
        total = comb(45, 5)
        main_probs = {m: comb(5, m) * comb(40, 5 - m) / total for m in range(6)}
        p_joker = 1 / 20
        return (
            main_probs[3] + main_probs[4] + main_probs[5]
            + main_probs[2] * p_joker + main_probs[1] * p_joker
        )
    if game == "loto_649":
        total = comb(49, 6)
        return sum(comb(6, m) * comb(43, 6 - m) / total for m in range(3, 7))
    # loto_540: pick 5 from 40, 6 drawn, win at 4+ main matches
    total = comb(40, 5)
    return sum(comb(6, m) * comb(34, 5 - m) / total for m in range(4, 6))


def _p_ticket_any_win(game: str) -> float:
    p = _variant_win_prob(game)
    # From pricing, not a local copy: costs already come from there, so a
    # local variant count that drifted would price a ticket one way and score
    # its odds another.
    v = VARIANTS_PER_TICKET[game]
    return 1.0 - (1.0 - p) ** v


def _p_any_win(tickets: dict[str, int]) -> float:
    p_none = 1.0
    for game, count in tickets.items():
        if count <= 0:
            continue
        p_loss = (1.0 - _p_ticket_any_win(game)) ** count
        p_none *= p_loss
    return 1.0 - p_none


def enumerate_allocations(
    budget_ron: float,
    allowed_games: set[str] | None = None,
    max_per_game: int = 8,
) -> list[TicketAllocation]:
    allowed = set(_GAMES) if allowed_games is None else set(allowed_games) & set(_GAMES)
    costs = {g: compute_ticket_cost(g) for g in _GAMES}

    out: list[TicketAllocation] = []
    for nj in range(max_per_game + 1):
        if "joker" not in allowed and nj > 0:
            continue
        for n9 in range(max_per_game + 1):
            if "loto_649" not in allowed and n9 > 0:
                continue
            for n5 in range(max_per_game + 1):
                if "loto_540" not in allowed and n5 > 0:
                    continue
                total = nj + n9 + n5
                if total == 0:
                    continue
                cost = costs["joker"] * nj + costs["loto_649"] * n9 + costs["loto_540"] * n5
                if cost > budget_ron + 1e-9:
                    continue
                tickets = {"joker": nj, "loto_649": n9, "loto_540": n5}
                out.append(
                    TicketAllocation(
                        tickets=tickets,
                        total_cost=round(cost, 2),
                        p_any_win=_p_any_win(tickets),
                    )
                )
    return out


def best_allocation(
    budget_ron: float,
    allowed_games: set[str] | None = None,
    max_per_game: int = 8,
) -> TicketAllocation:
    allocs = enumerate_allocations(
        budget_ron=budget_ron, allowed_games=allowed_games, max_per_game=max_per_game,
    )
    if not allocs:
        return TicketAllocation(
            tickets={"joker": 0, "loto_649": 0, "loto_540": 0},
            total_cost=0.0,
            p_any_win=0.0,
        )
    return max(
        allocs,
        key=lambda a: (
            a.p_any_win,
            sum(a.tickets.values()),
            a.total_cost,
        ),
    )


def best_per_game_allocation(
    budget_buckets: dict[str, float],
) -> TicketAllocation:
    """Independent per-game allocation with separate budgets.

    Each game's bucket is spent on that game alone — the allocator runs
    per-bucket with `allowed_games={game}`. The combined result preserves
    each game's picks and sums the costs.

    This trades `P(any win)` for production coverage of strategies that
    don't dominate on per-RON probability (CoreShare on 6/49, Wheel on 5/40
    etc.), so we accumulate real data on them instead of always buying the
    cheapest-per-probability game.
    """
    tickets: dict[str, int] = {g: 0 for g in _GAMES}
    total_cost = 0.0
    for game, bucket in budget_buckets.items():
        if game not in _GAMES or bucket <= 0:
            continue
        sub = best_allocation(budget_ron=bucket, allowed_games={game})
        tickets[game] = sub.tickets.get(game, 0)
        total_cost += sub.total_cost
    return TicketAllocation(
        tickets=tickets,
        total_cost=round(total_cost, 2),
        p_any_win=_p_any_win(tickets),
    )
