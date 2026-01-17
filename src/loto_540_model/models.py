from dataclasses import dataclass


@dataclass(frozen=True)
class Loto540Draw:
    """Represents a single Loto 5/40 draw result.

    The game draws 6 numbers from 1-40. Players pick 5 numbers
    and win by matching 4 or 5 of the 6 drawn numbers.
    Super Noroc is an optional 6-digit bonus number.
    """
    date: str  # YYYY-MM-DD format
    main_numbers: list[int]  # 6 sorted numbers (1-40)
    super_noroc: int | None  # 6-digit number (0-999999) or None
