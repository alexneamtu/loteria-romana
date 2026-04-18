"""Metrics for ticket-level backtesting.

Reports jackpot-shaped metrics instead of hit-rate:
- median_roi: typical per-draw net outcome
- p_best_match_at_least(N): rate of approach-the-jackpot hits
- sample_skewness: heavier right tail = more jackpot-tilted
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from .ev_calculator import EVCalculator
from .ticket import Ticket


@dataclass(frozen=True)
class TicketOutcome:
    payout: float
    cost: float
    best_match: int

    @property
    def net(self) -> float:
        return self.payout - self.cost


_CALC = EVCalculator()
_JOKER_GAME = _CALC.create_joker()
_649_GAME = _CALC.create_loto_649()
_540_GAME = _CALC.create_loto_540()


def _joker_prize_ron(best_match: int, best_has_joker: bool, jackpot: float) -> float:
    for tier in _JOKER_GAME.prize_tiers:
        if tier.matches_required == best_match and tier.bonus_required == best_has_joker:
            if tier.fixed_prize is not None:
                return tier.fixed_prize
            if tier.prize_pool_percentage is not None:
                if best_match == 5 and best_has_joker:
                    return jackpot
                return 1000.0 * tier.prize_pool_percentage
    return 0.0


def _649_prize_ron(best_match: int, jackpot: float) -> float:
    for tier in _649_GAME.prize_tiers:
        if tier.matches_required == best_match:
            if tier.fixed_prize is not None:
                return tier.fixed_prize
            if best_match == 6:
                return jackpot
            return 1000.0 * (tier.prize_pool_percentage or 0.0)
    return 0.0


def _540_prize_ron(best_match: int, jackpot: float) -> float:
    for tier in _540_GAME.prize_tiers:
        if tier.matches_required == best_match:
            if tier.fixed_prize is not None:
                return tier.fixed_prize
            if best_match == 5:
                return jackpot
            return 1000.0 * (tier.prize_pool_percentage or 0.0)
    return 0.0


def estimate_ticket_payout(
    ticket: Ticket,
    winning_main: list[int] | tuple[int, ...],
    winning_joker: int | None,
    winning_side: str | None,
    jackpot: float = 1_000_000.0,
) -> float:
    """Estimate RON payout for a ticket on a given drawn set.

    Takes the best-performing variant; ignores side game matches for
    now (Plan C refines side-game prize ladders).
    """
    best_payout = 0.0
    for v in ticket.variants:
        matches = v.count_main_matches(winning_main)
        if ticket.game == "joker":
            has_joker = v.bonus_number is not None and v.bonus_number == winning_joker
            payout = _joker_prize_ron(matches, has_joker, jackpot)
        elif ticket.game == "loto_649":
            payout = _649_prize_ron(matches, jackpot)
        else:
            payout = _540_prize_ron(matches, jackpot)
        if payout > best_payout:
            best_payout = payout
    return best_payout


def median_roi(outcomes: list[TicketOutcome]) -> float:
    if not outcomes:
        return 0.0
    return statistics.median(o.net for o in outcomes)


def p_best_match_at_least(outcomes: list[TicketOutcome], threshold: int) -> float:
    if not outcomes:
        return 0.0
    hits = sum(1 for o in outcomes if o.best_match >= threshold)
    return hits / len(outcomes)


def sample_skewness(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    m3 = sum((v - mean) ** 3 for v in values) / n
    if m2 == 0:
        return 0.0
    return m3 / (m2 ** 1.5)


def summarize(outcomes: list[TicketOutcome]) -> dict[str, float]:
    return {
        "n": len(outcomes),
        "median_roi": median_roi(outcomes),
        "p_best_match_ge_3": p_best_match_at_least(outcomes, 3),
        "p_best_match_ge_4": p_best_match_at_least(outcomes, 4),
        "p_best_match_ge_5": p_best_match_at_least(outcomes, 5),
        "skewness": sample_skewness([o.net for o in outcomes]),
        "mean_roi": statistics.fmean([o.net for o in outcomes]) if outcomes else 0.0,
    }
