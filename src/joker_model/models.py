from dataclasses import dataclass, field


@dataclass(frozen=True)
class JokerDraw:
    date: str
    main_numbers: list[int]
    joker: int
    noroc_plus: str | None = None

    def __hash__(self) -> int:
        return hash((self.date, tuple(self.main_numbers), self.joker, self.noroc_plus))
