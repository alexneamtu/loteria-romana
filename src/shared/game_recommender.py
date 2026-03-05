"""Game recommender for optimal budget allocation across lottery games.

Calculates exact win probabilities and finds the ticket allocation
that maximizes P(any win) for a given budget in RON.

Analysis shows Joker is the best game per RON spent (0.366%/RON),
followed by Loto 6/49 (0.311%/RON). Loto 5/40 is poor value with
the 4+ match threshold (0.020%/RON).
"""

from dataclasses import dataclass, field
from math import comb


TICKET_COSTS = {
    "joker": 8.0,
    "loto_649": 6.0,
    "loto_540": 4.0,
}


@dataclass(frozen=True)
class GameProbability:
    """Win probability details for a single game."""

    name: str
    win_rate: float
    ticket_cost: float

    @property
    def win_rate_per_ron(self) -> float:
        return self.win_rate / self.ticket_cost


@dataclass
class BudgetAllocation:
    """Optimal ticket allocation across games for a given budget."""

    tickets: dict[str, int] = field(default_factory=dict)
    total_cost: float = 0.0
    p_any_win: float = 0.0
    budget: float = 0.0


def calculate_win_probability(game_name: str) -> GameProbability:
    """Calculate exact win probability for a game using combinatorics.

    Returns a GameProbability with the chance of winning any prize
    on a single ticket.
    """
    calculators = {
        "joker": _joker_win_probability,
        "loto_649": _loto_649_win_probability,
        "loto_540": _loto_540_win_probability,
    }
    calculator = calculators.get(game_name)
    if calculator is None:
        raise ValueError(f"Unknown game: {game_name}")

    win_rate = calculator()
    return GameProbability(
        name=game_name,
        win_rate=win_rate,
        ticket_cost=TICKET_COSTS[game_name],
    )


def _joker_win_probability() -> float:
    """P(any prize) for Joker: 5 from 45 + 1 from 20.

    Prize tiers: 3+ main, OR 2+joker, OR 1+joker.
    """
    total = comb(45, 5)
    p_joker = 1.0 / 20

    main_probs = {}
    for m in range(6):
        main_probs[m] = comb(5, m) * comb(40, 5 - m) / total

    p_main_3plus = main_probs[3] + main_probs[4] + main_probs[5]
    p_2_with_joker = main_probs[2] * p_joker
    p_1_with_joker = main_probs[1] * p_joker

    return p_main_3plus + p_2_with_joker + p_1_with_joker


def _loto_649_win_probability() -> float:
    """P(any prize) for Loto 6/49: 6 from 49, win at 3+ matches."""
    total = comb(49, 6)
    p_win = 0.0
    for m in range(3, 7):
        p_win += comb(6, m) * comb(43, 6 - m) / total
    return p_win


def _loto_540_win_probability() -> float:
    """P(any prize) for Loto 5/40: pick 5 from 40, 6 drawn, win at 4+."""
    total = comb(40, 5)
    p_win = 0.0
    for m in range(4, 6):
        p_win += comb(6, m) * comb(34, 5 - m) / total
    return p_win


def _p_any_win(probabilities: dict[str, float], tickets: dict[str, int]) -> float:
    """P(at least one ticket wins) across independent tickets."""
    p_no_win = 1.0
    for game, count in tickets.items():
        if count > 0:
            p_no_win *= (1 - probabilities[game]) ** count
    return 1 - p_no_win


def optimize_budget(budget_ron: float) -> BudgetAllocation:
    """Find the ticket allocation maximizing P(any win) for a budget.

    Uses brute-force search across all feasible combinations of
    Joker, Loto 6/49, and Loto 5/40 tickets.
    """
    if budget_ron < min(TICKET_COSTS.values()):
        return BudgetAllocation(budget=budget_ron)

    probabilities = {
        "joker": _joker_win_probability(),
        "loto_649": _loto_649_win_probability(),
        "loto_540": _loto_540_win_probability(),
    }

    best = BudgetAllocation(budget=budget_ron)
    budget = int(budget_ron)

    max_joker = budget // 8
    for n_joker in range(max_joker + 1):
        remaining_after_joker = budget - n_joker * 8
        max_649 = remaining_after_joker // 6
        for n_649 in range(max_649 + 1):
            remaining = remaining_after_joker - n_649 * 6
            n_540 = remaining // 4

            total_tickets = n_joker + n_649 + n_540
            if total_tickets == 0:
                continue

            tickets = {"joker": n_joker, "loto_649": n_649, "loto_540": n_540}
            cost = n_joker * 8 + n_649 * 6 + n_540 * 4
            p = _p_any_win(probabilities, tickets)

            if p > best.p_any_win:
                best = BudgetAllocation(
                    tickets=tickets,
                    total_cost=float(cost),
                    p_any_win=p,
                    budget=budget_ron,
                )

    return best


def top_diverse_allocations(budget_ron: float, count: int = 3) -> list[BudgetAllocation]:
    """Find top N allocations with diverse game mixes.

    Returns allocations sorted by P(any win), ensuring each has a
    different set of active games (games with tickets > 0).
    """
    if budget_ron < min(TICKET_COSTS.values()):
        return []

    probabilities = {
        "joker": _joker_win_probability(),
        "loto_649": _loto_649_win_probability(),
        "loto_540": _loto_540_win_probability(),
    }

    # Collect all valid allocations grouped by active game set
    by_mix: dict[frozenset[str], BudgetAllocation] = {}
    budget = int(budget_ron)

    max_joker = budget // 8
    for n_joker in range(max_joker + 1):
        remaining_after_joker = budget - n_joker * 8
        max_649 = remaining_after_joker // 6
        for n_649 in range(max_649 + 1):
            remaining = remaining_after_joker - n_649 * 6
            n_540 = remaining // 4

            tickets = {"joker": n_joker, "loto_649": n_649, "loto_540": n_540}
            active = frozenset(g for g, c in tickets.items() if c > 0)
            if not active:
                continue

            cost = n_joker * 8 + n_649 * 6 + n_540 * 4
            p = _p_any_win(probabilities, tickets)

            prev = by_mix.get(active)
            if prev is None or p > prev.p_any_win:
                by_mix[active] = BudgetAllocation(
                    tickets=tickets,
                    total_cost=float(cost),
                    p_any_win=p,
                    budget=budget_ron,
                )

    # Sort by P(any win) descending, take top N
    ranked = sorted(by_mix.values(), key=lambda a: a.p_any_win, reverse=True)
    return ranked[:count]


def format_recommendation(allocation: BudgetAllocation) -> str:
    """Format a budget allocation as a human-readable recommendation."""
    if allocation.p_any_win == 0:
        return f"Budget of {allocation.budget:.0f} RON is too small for any ticket."

    probabilities = {
        "joker": calculate_win_probability("joker"),
        "loto_649": calculate_win_probability("loto_649"),
        "loto_540": calculate_win_probability("loto_540"),
    }

    display_names = {
        "joker": "Joker",
        "loto_649": "Loto 6/49",
        "loto_540": "Loto 5/40",
    }

    lines = [
        f"Budget: {allocation.budget:.0f} RON",
        f"P(any win): {allocation.p_any_win * 100:.2f}%",
        "",
        "Allocation:",
    ]

    for game, count in allocation.tickets.items():
        if count > 0:
            gp = probabilities[game]
            cost = count * gp.ticket_cost
            lines.append(
                f"  {display_names[game]}: {count} ticket(s) "
                f"({cost:.0f} RON, {gp.win_rate * 100:.3f}% each)"
            )

    unused = allocation.budget - allocation.total_cost
    if unused > 0:
        lines.append(f"  Unspent: {unused:.0f} RON")

    return "\n".join(lines)
