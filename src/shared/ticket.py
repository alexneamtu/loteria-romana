"""Ticket and Variant dataclasses matching loto.ro physical tickets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .pricing import VARIANTS_PER_TICKET

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


@dataclass(frozen=True)
class Ticket:
    game: str
    variants: tuple[Variant, ...]
    side_game_number: str
    strategy: str
    cost_ron: float

    def __post_init__(self) -> None:
        if self.game not in VARIANTS_PER_TICKET:
            raise ValueError(f"unknown game: {self.game!r}")

        expected_variants = VARIANTS_PER_TICKET[self.game]
        if len(self.variants) != expected_variants:
            raise ValueError(
                f"{self.game} ticket requires {expected_variants} variants, "
                f"got {len(self.variants)}"
            )

        for v in self.variants:
            if v.game != self.game:
                raise ValueError(
                    f"variant game={v.game!r} does not match ticket game={self.game!r}"
                )

        if not isinstance(self.side_game_number, str):
            raise ValueError("side_game_number must be a string (leading zeros matter)")

        if self.cost_ron < 0:
            raise ValueError("cost_ron must be non-negative")

    def best_main_match(self, winning: tuple[int, ...] | list[int]) -> int:
        return max(v.count_main_matches(winning) for v in self.variants)
