from dataclasses import dataclass


@dataclass(frozen=True)
class Loto649Draw:
    date: str
    main_numbers: list[int]
