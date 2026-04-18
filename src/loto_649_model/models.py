from dataclasses import dataclass


@dataclass(frozen=True)
class Loto649Draw:
    date: str
    main_numbers: list[int]
    noroc: str | None = None

    def __hash__(self) -> int:
        return hash((self.date, tuple(self.main_numbers), self.noroc))
