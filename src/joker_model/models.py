from dataclasses import dataclass


@dataclass(frozen=True)
class JokerDraw:
    date: str
    main_numbers: list[int]
    joker: int
