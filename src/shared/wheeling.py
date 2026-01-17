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
