from dataclasses import dataclass


@dataclass(frozen=True)
class Loto540Draw:
    """A single Loto 5/40 draw result.

    The game draws 6 numbers from 1-40. Players pick 5 numbers and win
    by matching 4 or 5 of the 6 drawn numbers. super_noroc is the
    6-digit side game printed on the same ticket.
    """
    date: str  # YYYY-MM-DD format
    main_numbers: list[int]  # 6 sorted numbers (1-40)
    super_noroc: str | None = None

    def __hash__(self) -> int:
        return hash((self.date, tuple(self.main_numbers), self.super_noroc))
