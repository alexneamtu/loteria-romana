"""Ticket and Variant dataclasses matching loto.ro physical tickets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Game = Literal["joker", "loto_649", "loto_540"]

_MAIN_COUNT: dict[str, int] = {"joker": 5, "loto_649": 6, "loto_540": 5}
_MAIN_POOL_MAX: dict[str, int] = {"joker": 45, "loto_649": 49, "loto_540": 40}
_JOKER_POOL_MAX = 20


@dataclass(frozen=True)
class Variant:
    main_numbers: tuple[int, ...]
    bonus_number: int | None
    game: str

    def __post_init__(self) -> None:
        if self.game not in _MAIN_COUNT:
            raise ValueError(f"unknown game: {self.game!r}")

        expected = _MAIN_COUNT[self.game]
        if len(self.main_numbers) != expected:
            raise ValueError(
                f"{self.game} variant expects {expected} main numbers, "
                f"got {len(self.main_numbers)}"
            )

        if len(set(self.main_numbers)) != expected:
            raise ValueError(f"{self.game} variant has duplicate main numbers")

        if list(self.main_numbers) != sorted(self.main_numbers):
            raise ValueError(f"{self.game} variant main numbers must be sorted ascending")

        pool_max = _MAIN_POOL_MAX[self.game]
        for n in self.main_numbers:
            if not 1 <= n <= pool_max:
                raise ValueError(
                    f"{self.game} main number {n} out of range 1..{pool_max}"
                )

        if self.game == "joker":
            if self.bonus_number is None:
                raise ValueError("joker variant requires a bonus_number (1..20)")
            if not 1 <= self.bonus_number <= _JOKER_POOL_MAX:
                raise ValueError(
                    f"joker bonus_number {self.bonus_number} out of range 1..20"
                )
        else:
            if self.bonus_number is not None:
                raise ValueError(f"{self.game} variant must not carry a bonus_number")

    def count_main_matches(self, winning: tuple[int, ...] | list[int]) -> int:
        return len(set(self.main_numbers) & set(winning))
