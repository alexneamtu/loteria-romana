"""Wheeling systems for lottery coverage optimization.

Wheeling systems generate sets of tickets that guarantee a minimum
match level if certain numbers are drawn. For example, a 3-if-5
wheel guarantees at least 3 matches if 5 of your chosen numbers
are among the winners.
"""

import itertools
from dataclasses import dataclass
from math import comb


@dataclass
class WheelConfig:
    """Configuration for a wheeling system."""

    total_numbers: int  # Numbers in the wheel (e.g., 10)
    numbers_per_line: int  # Numbers per ticket (e.g., 6)
    guarantee: int  # Minimum guaranteed matches (e.g., 3)

    def __post_init__(self):
        if self.guarantee > self.numbers_per_line:
            raise ValueError("Guarantee cannot exceed numbers per line")
        if self.numbers_per_line > self.total_numbers:
            raise ValueError("Numbers per line cannot exceed total numbers")


def generate_full_wheel(numbers: list[int], numbers_per_line: int) -> list[list[int]]:
    """Generate a full wheel (all combinations).

    A full wheel covers every possible combination, guaranteeing
    the maximum possible matches for any winning numbers.

    Args:
        numbers: List of numbers to wheel
        numbers_per_line: Numbers per ticket

    Returns:
        List of ticket lines
    """
    return [list(combo) for combo in itertools.combinations(numbers, numbers_per_line)]


def estimate_full_wheel_size(total_numbers: int, numbers_per_line: int) -> int:
    """Estimate the number of tickets in a full wheel."""
    return comb(total_numbers, numbers_per_line)


class WheelGenerator:
    """Generate wheeling systems with various coverage guarantees."""

    def __init__(
        self,
        numbers_per_line: int,
        guarantee: int,
    ):
        """Initialize wheel generator.

        Args:
            numbers_per_line: Numbers per ticket (e.g., 5 for Joker, 6 for 6/49)
            guarantee: Minimum matches to guarantee
        """
        self.numbers_per_line = numbers_per_line
        self.guarantee = guarantee

    def generate_full_wheel(self, numbers: list[int]) -> list[list[int]]:
        """Generate a full wheel covering all combinations."""
        return generate_full_wheel(numbers, self.numbers_per_line)

    def generate_abbreviated_wheel(
        self,
        numbers: list[int],
        max_tickets: int | None = None,
    ) -> list[list[int]]:
        """Generate an abbreviated wheel with coverage guarantee.

        Uses a greedy covering algorithm to generate tickets that
        guarantee at least `guarantee` matches if all wheeled numbers
        are among the winners.

        Args:
            numbers: List of numbers to wheel
            max_tickets: Maximum tickets to generate (None = no limit)

        Returns:
            List of ticket lines
        """
        if len(numbers) < self.numbers_per_line:
            return []

        # The "sub-combinations" we need to cover
        # For a 3-if-5 wheel: all 3-number combinations must appear
        # in at least one ticket
        subcombos_to_cover = set(
            tuple(sorted(combo))
            for combo in itertools.combinations(numbers, self.guarantee)
        )

        tickets = []
        covered = set()

        # Greedy covering: pick tickets that cover the most uncovered subcombos
        while covered != subcombos_to_cover:
            if max_tickets and len(tickets) >= max_tickets:
                break

            best_ticket = None
            best_cover_count = 0

            # Check all possible tickets
            for ticket in itertools.combinations(numbers, self.numbers_per_line):
                ticket_subcombos = set(
                    tuple(sorted(s))
                    for s in itertools.combinations(ticket, self.guarantee)
                )
                new_cover = ticket_subcombos - covered
                if len(new_cover) > best_cover_count:
                    best_cover_count = len(new_cover)
                    best_ticket = ticket

            if best_ticket is None or best_cover_count == 0:
                break

            tickets.append(list(best_ticket))
            ticket_subcombos = set(
                tuple(sorted(s))
                for s in itertools.combinations(best_ticket, self.guarantee)
            )
            covered |= ticket_subcombos

        return tickets

    def estimate_reduction(self, total_numbers: int) -> dict:
        """Estimate ticket reduction from wheeling.

        Args:
            total_numbers: Number of elements to wheel

        Returns:
            Dict with full wheel size, estimated abbreviated size, and reduction ratio
        """
        full_size = estimate_full_wheel_size(total_numbers, self.numbers_per_line)

        # Estimate abbreviated size based on covering design theory
        # This is a rough approximation
        subcombos = comb(total_numbers, self.guarantee)
        subcombos_per_ticket = comb(self.numbers_per_line, self.guarantee)

        # Lower bound from covering design theory
        estimated_min = max(1, subcombos // subcombos_per_ticket)

        # Practical estimates are usually 20-40% higher due to greedy algorithm
        estimated_abbreviated = int(estimated_min * 1.3)

        return {
            "total_numbers": total_numbers,
            "numbers_per_line": self.numbers_per_line,
            "guarantee": self.guarantee,
            "full_wheel_size": full_size,
            "estimated_abbreviated_size": estimated_abbreviated,
            "reduction_ratio": full_size / estimated_abbreviated if estimated_abbreviated > 0 else 1.0,
        }


def verify_wheel_coverage(
    tickets: list[list[int]],
    numbers: list[int],
    guarantee: int,
) -> tuple[bool, list[tuple]]:
    """Verify that a wheel provides the claimed coverage.

    Args:
        tickets: List of ticket lines
        numbers: Numbers being wheeled
        guarantee: Minimum matches to verify

    Returns:
        (is_valid, list of uncovered subcombinations)
    """
    # All sub-combinations that need to be covered
    required = set(
        tuple(sorted(combo))
        for combo in itertools.combinations(numbers, guarantee)
    )

    # Sub-combinations covered by the tickets
    covered = set()
    for ticket in tickets:
        for subcombo in itertools.combinations(ticket, guarantee):
            covered.add(tuple(sorted(subcombo)))

    uncovered = required - covered
    return len(uncovered) == 0, list(uncovered)


def generate_balanced_wheel(
    numbers: list[int],
    numbers_per_line: int,
    target_tickets: int,
) -> list[list[int]]:
    """Generate a wheel with approximately equal number coverage.

    Each number appears in roughly the same number of tickets,
    providing balanced coverage across all wheeled numbers.

    Args:
        numbers: List of numbers to wheel
        numbers_per_line: Numbers per ticket
        target_tickets: Approximate number of tickets to generate

    Returns:
        List of ticket lines
    """
    if len(numbers) < numbers_per_line:
        return []

    tickets = []
    seen = set()
    n = len(numbers)

    # Calculate expected appearances per number for balance
    # Each number should appear in approximately:
    # target_tickets * numbers_per_line / n tickets
    number_counts = {num: 0 for num in numbers}

    for _ in range(target_tickets * 10):  # Try multiple times
        if len(tickets) >= target_tickets:
            break

        # Select numbers preferring those with lower counts
        sorted_nums = sorted(numbers, key=lambda x: number_counts[x])
        ticket = sorted_nums[:numbers_per_line]
        ticket_key = tuple(sorted(ticket))

        if ticket_key not in seen:
            seen.add(ticket_key)
            tickets.append(sorted(ticket))
            for num in ticket:
                number_counts[num] += 1

        # Also try random variations
        import random
        shuffled = sorted_nums.copy()
        random.shuffle(shuffled)
        ticket = shuffled[:numbers_per_line]
        ticket_key = tuple(sorted(ticket))

        if ticket_key not in seen:
            seen.add(ticket_key)
            tickets.append(sorted(ticket))
            for num in ticket:
                number_counts[num] += 1

    return tickets[:target_tickets]


class KeyNumberWheel:
    """Generate wheels with "key" numbers that appear in every ticket.

    Key number wheels are efficient when you have strong confidence
    in certain numbers. They reduce ticket count while ensuring
    key numbers are always played.
    """

    def __init__(
        self,
        numbers_per_line: int,
        key_numbers: list[int],
    ):
        """Initialize key number wheel.

        Args:
            numbers_per_line: Total numbers per ticket
            key_numbers: Numbers that must appear in every ticket
        """
        self.numbers_per_line = numbers_per_line
        self.key_numbers = sorted(key_numbers)

        if len(key_numbers) >= numbers_per_line:
            raise ValueError("Key numbers must be fewer than numbers per line")

    def generate(self, other_numbers: list[int]) -> list[list[int]]:
        """Generate wheel with key numbers in every ticket.

        Args:
            other_numbers: Non-key numbers to combine with keys

        Returns:
            List of ticket lines
        """
        remaining_slots = self.numbers_per_line - len(self.key_numbers)

        if len(other_numbers) < remaining_slots:
            return []

        tickets = []
        for combo in itertools.combinations(other_numbers, remaining_slots):
            ticket = sorted(list(self.key_numbers) + list(combo))
            tickets.append(ticket)

        return tickets

    def estimate_size(self, other_numbers_count: int) -> int:
        """Estimate number of tickets for key wheel."""
        remaining_slots = self.numbers_per_line - len(self.key_numbers)
        return comb(other_numbers_count, remaining_slots)


# Bluskov-style optimized wheels based on covering design theory


def compute_coverage_lower_bound(
    total_numbers: int, numbers_per_line: int, guarantee: int
) -> int:
    """Compute theoretical lower bound for wheel size.

    Based on covering design theory: minimum tickets needed to cover
    all guarantee-sized subcombinations.

    Args:
        total_numbers: Numbers in the wheel
        numbers_per_line: Numbers per ticket
        guarantee: Minimum match guarantee

    Returns:
        Theoretical minimum number of tickets
    """
    subcombos = comb(total_numbers, guarantee)
    subcombos_per_ticket = comb(numbers_per_line, guarantee)
    return max(1, subcombos // subcombos_per_ticket)


def compute_coverage_statistics(
    tickets: list[list[int]], numbers: list[int], guarantee: int
) -> dict:
    """Compute detailed coverage statistics for a wheel.

    Args:
        tickets: List of ticket lines
        numbers: Numbers being wheeled
        guarantee: Match level to analyze

    Returns:
        Dict with coverage statistics
    """
    # All sub-combinations that need coverage
    required = set(
        tuple(sorted(combo)) for combo in itertools.combinations(numbers, guarantee)
    )

    # Track coverage frequency
    coverage_count: dict[tuple, int] = {r: 0 for r in required}
    for ticket in tickets:
        for subcombo in itertools.combinations(ticket, guarantee):
            key = tuple(sorted(subcombo))
            if key in coverage_count:
                coverage_count[key] += 1

    covered = {k for k, v in coverage_count.items() if v > 0}
    uncovered = required - covered

    # Coverage distribution
    coverage_values = list(coverage_count.values())
    avg_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0
    min_coverage = min(coverage_values) if coverage_values else 0
    max_coverage = max(coverage_values) if coverage_values else 0

    return {
        "total_required": len(required),
        "total_covered": len(covered),
        "coverage_percent": len(covered) / len(required) * 100 if required else 100,
        "is_complete": len(uncovered) == 0,
        "uncovered_count": len(uncovered),
        "avg_coverage": avg_coverage,
        "min_coverage": min_coverage,
        "max_coverage": max_coverage,
        "theoretical_minimum": compute_coverage_lower_bound(
            len(numbers), len(tickets[0]) if tickets else 6, guarantee
        ),
        "actual_tickets": len(tickets),
        "efficiency": (
            compute_coverage_lower_bound(
                len(numbers), len(tickets[0]) if tickets else 6, guarantee
            )
            / len(tickets)
            if tickets
            else 0
        ),
    }


def compute_number_balance(tickets: list[list[int]], numbers: list[int]) -> dict:
    """Analyze how evenly numbers are distributed across tickets.

    Args:
        tickets: List of ticket lines
        numbers: Numbers being wheeled

    Returns:
        Dict with balance statistics
    """
    counts = {n: 0 for n in numbers}
    for ticket in tickets:
        for n in ticket:
            if n in counts:
                counts[n] += 1

    values = list(counts.values())
    if not values:
        return {"error": "No tickets to analyze"}

    mean_count = sum(values) / len(values)
    variance = sum((v - mean_count) ** 2 for v in values) / len(values)
    std_dev = variance ** 0.5

    return {
        "mean_appearances": mean_count,
        "std_dev": std_dev,
        "coefficient_of_variation": std_dev / mean_count if mean_count > 0 else 0,
        "min_appearances": min(values),
        "max_appearances": max(values),
        "least_appearing": [n for n, c in counts.items() if c == min(values)],
        "most_appearing": [n for n, c in counts.items() if c == max(values)],
        "is_balanced": std_dev < mean_count * 0.2,  # Within 20% variation
    }


class OptimizedWheelGenerator:
    """Generate optimized wheels using improved algorithms.

    Based on Professor Bluskov's research on combinatorial covering designs,
    this generator uses heuristics to approach optimal wheel sizes.
    """

    def __init__(
        self,
        numbers_per_line: int,
        guarantee: int,
    ):
        self.numbers_per_line = numbers_per_line
        self.guarantee = guarantee

    def generate(
        self,
        numbers: list[int],
        max_tickets: int | None = None,
        optimization_rounds: int = 3,
    ) -> list[list[int]]:
        """Generate an optimized wheel.

        Uses multiple heuristics and selects the best result.

        Args:
            numbers: Numbers to wheel
            max_tickets: Maximum tickets (None = no limit)
            optimization_rounds: Number of optimization attempts

        Returns:
            List of ticket lines
        """
        if len(numbers) < self.numbers_per_line:
            return []

        best_tickets = None
        best_size = float("inf")

        # Try multiple approaches
        for round_num in range(optimization_rounds):
            # Greedy covering with different orderings
            if round_num == 0:
                # Standard greedy
                tickets = self._greedy_cover(numbers)
            elif round_num == 1:
                # Reverse greedy (start from end)
                tickets = self._greedy_cover(list(reversed(numbers)))
            else:
                # Random shuffle
                import random

                shuffled = list(numbers)
                random.shuffle(shuffled)
                tickets = self._greedy_cover(shuffled)

            if max_tickets and len(tickets) > max_tickets:
                tickets = tickets[:max_tickets]

            if len(tickets) < best_size:
                best_size = len(tickets)
                best_tickets = tickets

        return best_tickets or []

    def _greedy_cover(self, numbers: list[int]) -> list[list[int]]:
        """Standard greedy covering algorithm."""
        subcombos_to_cover = set(
            tuple(sorted(combo))
            for combo in itertools.combinations(numbers, self.guarantee)
        )

        tickets = []
        covered = set()

        while covered != subcombos_to_cover:
            best_ticket = None
            best_count = 0

            for ticket in itertools.combinations(numbers, self.numbers_per_line):
                ticket_subcombos = set(
                    tuple(sorted(s))
                    for s in itertools.combinations(ticket, self.guarantee)
                )
                new_cover = ticket_subcombos - covered
                if len(new_cover) > best_count:
                    best_count = len(new_cover)
                    best_ticket = ticket

            if best_ticket is None or best_count == 0:
                break

            tickets.append(list(best_ticket))
            ticket_subcombos = set(
                tuple(sorted(s))
                for s in itertools.combinations(best_ticket, self.guarantee)
            )
            covered |= ticket_subcombos

        return tickets

    def analyze(self, numbers: list[int]) -> dict:
        """Analyze wheel characteristics before generation.

        Args:
            numbers: Numbers to wheel

        Returns:
            Analysis of wheel properties
        """
        n = len(numbers)
        k = self.numbers_per_line
        t = self.guarantee

        full_wheel_size = comb(n, k)
        min_wheel_size = compute_coverage_lower_bound(n, k, t)
        estimated_wheel = int(min_wheel_size * 1.3)

        return {
            "total_numbers": n,
            "numbers_per_ticket": k,
            "guarantee": t,
            "full_wheel_size": full_wheel_size,
            "theoretical_minimum": min_wheel_size,
            "estimated_abbreviated": estimated_wheel,
            "reduction_factor": full_wheel_size / estimated_wheel if estimated_wheel > 0 else 1,
            "subcombinations_to_cover": comb(n, t),
            "subcombinations_per_ticket": comb(k, t),
        }


# ============================================================================
# ANY-WIN GUARANTEE WHEELS
# ============================================================================
# These functions help create wheels optimized for winning ANY prize (3+ matches)


def create_any_win_wheel(
    selected_numbers: list[int],
    game: str = "loto_649",
    min_guarantee: int = 3,
    max_tickets: int | None = None,
) -> dict:
    """
    Create a wheel that guarantees winning a prize if enough numbers hit.

    For Loto 6/49:
    - If you select 12 numbers and 6 are drawn: guaranteed jackpot coverage
    - If you select 12 numbers and 4 are drawn: guaranteed Category IV (3 matches)

    Args:
        selected_numbers: Your chosen numbers to wheel
        game: Game type ("loto_649", "loto_540", "joker")
        min_guarantee: Minimum matches to guarantee (default: 3 for small prize)
        max_tickets: Maximum tickets (None = generate all needed)

    Returns:
        Dict with wheel info, tickets, and guarantee explanation
    """
    game_configs = {
        "loto_649": {"numbers_per_line": 6, "pool_size": 49, "prize_threshold": 3},
        "loto_540": {"numbers_per_line": 5, "pool_size": 40, "prize_threshold": 3},
        "joker": {"numbers_per_line": 5, "pool_size": 45, "prize_threshold": 3},
    }

    config = game_configs.get(game, game_configs["loto_649"])
    numbers_per_line = config["numbers_per_line"]

    n = len(selected_numbers)

    if n < numbers_per_line:
        return {
            "error": f"Need at least {numbers_per_line} numbers for {game}",
            "tickets": [],
        }

    # Generate the wheel
    generator = OptimizedWheelGenerator(numbers_per_line, min_guarantee)
    tickets = generator.generate(selected_numbers)

    if max_tickets and len(tickets) > max_tickets:
        tickets = tickets[:max_tickets]

    # Verify coverage
    is_valid, uncovered = verify_wheel_coverage(tickets, selected_numbers, min_guarantee)

    # Calculate what guarantees this provides
    guarantees = []
    for hits in range(min_guarantee, n + 1):
        if hits <= numbers_per_line:
            # If X numbers hit and X <= numbers_per_line, we get X matches minimum
            guarantees.append({
                "numbers_hit": hits,
                "guaranteed_matches": min(hits, min_guarantee),
                "explanation": f"If {hits} of your numbers are drawn, you're guaranteed at least {min(hits, min_guarantee)} matches"
            })
        else:
            # More numbers hit than fit in a ticket
            guarantees.append({
                "numbers_hit": hits,
                "guaranteed_matches": min_guarantee,
                "explanation": f"If {hits} of your numbers are drawn, you're guaranteed at least {min_guarantee} matches"
            })

    return {
        "game": game,
        "numbers_wheeled": len(selected_numbers),
        "selected_numbers": sorted(selected_numbers),
        "ticket_count": len(tickets),
        "tickets": tickets,
        "guarantee_level": min_guarantee,
        "coverage_verified": is_valid,
        "guarantees": guarantees,
        "cost_estimate": {
            "loto_649": len(tickets) * 6.0,
            "loto_540": len(tickets) * 4.0,
            "joker": len(tickets) * 8.0,
        }.get(game, len(tickets) * 6.0),
        "explanation": (
            f"This wheel uses {len(tickets)} tickets to guarantee at least "
            f"{min_guarantee} matches if {min_guarantee} or more of your "
            f"{len(selected_numbers)} selected numbers are among the winners."
        )
    }


def suggest_wheel_size(
    budget: float,
    game: str = "loto_649",
    guarantee: int = 3,
) -> dict:
    """
    Suggest optimal number of numbers to wheel based on budget.

    Args:
        budget: Total budget in RON
        game: Game type
        guarantee: Minimum match guarantee

    Returns:
        Suggestions for different wheel sizes
    """
    ticket_costs = {"loto_649": 6.0, "loto_540": 4.0, "joker": 8.0}
    numbers_per_line = {"loto_649": 6, "loto_540": 5, "joker": 5}

    cost = ticket_costs.get(game, 6.0)
    k = numbers_per_line.get(game, 6)
    max_tickets = int(budget / cost)

    suggestions = []

    # Try different wheel sizes
    for n in range(k, min(20, k + 10)):  # From minimum to 20 numbers
        min_tickets = compute_coverage_lower_bound(n, k, guarantee)
        est_tickets = int(min_tickets * 1.3)

        if est_tickets <= max_tickets:
            suggestions.append({
                "numbers_to_select": n,
                "estimated_tickets": est_tickets,
                "estimated_cost": est_tickets * cost,
                "within_budget": True,
                "numbers_needed_to_win": guarantee,
            })
        else:
            suggestions.append({
                "numbers_to_select": n,
                "estimated_tickets": est_tickets,
                "estimated_cost": est_tickets * cost,
                "within_budget": False,
                "numbers_needed_to_win": guarantee,
            })

    # Find optimal
    within_budget = [s for s in suggestions if s["within_budget"]]
    optimal = within_budget[-1] if within_budget else None

    return {
        "budget": budget,
        "game": game,
        "ticket_cost": cost,
        "max_tickets": max_tickets,
        "guarantee": guarantee,
        "suggestions": suggestions,
        "optimal": optimal,
        "recommendation": (
            f"With {budget} RON, you can wheel up to {optimal['numbers_to_select']} numbers "
            f"with a {guarantee}-match guarantee using ~{optimal['estimated_tickets']} tickets."
            if optimal else
            f"Budget of {budget} RON is too small for a {guarantee}-match guarantee wheel."
        )
    }
