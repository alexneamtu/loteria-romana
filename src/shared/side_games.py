"""Side-game number helpers for Noroc / Super Noroc / Noroc Plus.

Romanian loto.ro ticket format:
- Loto 6/49 ticket carries a Noroc number: 7 digits, leading zeros allowed.
- Loto 5/40 ticket carries a Super Noroc number: 6 digits, leading zeros allowed.
- Joker ticket carries a Noroc Plus number: prefix "NP" + integer 1..20.

Numbers are always strings to preserve leading zeros.
"""
from __future__ import annotations

import random
import re

NOROC_DIGITS = 7
SUPER_NOROC_DIGITS = 6
NOROC_PLUS_MIN = 1
NOROC_PLUS_MAX = 20

_NOROC_RE = re.compile(rf"^\d{{{NOROC_DIGITS}}}$")
_SUPER_NOROC_RE = re.compile(rf"^\d{{{SUPER_NOROC_DIGITS}}}$")
_NOROC_PLUS_RE = re.compile(r"^NP(\d{1,2})$")


def generate_noroc(rng: random.Random) -> str:
    return f"{rng.randint(0, 10**NOROC_DIGITS - 1):0{NOROC_DIGITS}d}"


def generate_super_noroc(rng: random.Random) -> str:
    return f"{rng.randint(0, 10**SUPER_NOROC_DIGITS - 1):0{SUPER_NOROC_DIGITS}d}"


def generate_noroc_plus(rng: random.Random) -> str:
    n = rng.randint(NOROC_PLUS_MIN, NOROC_PLUS_MAX)
    return f"NP{n:02d}"


def validate_side_game_number(game: str, value: str) -> bool:
    if game == "loto_649":
        return bool(_NOROC_RE.match(value))
    if game == "loto_540":
        return bool(_SUPER_NOROC_RE.match(value))
    if game == "joker":
        match = _NOROC_PLUS_RE.match(value)
        if not match:
            return False
        return NOROC_PLUS_MIN <= int(match.group(1)) <= NOROC_PLUS_MAX
    return False
