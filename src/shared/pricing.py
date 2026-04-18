"""Ticket pricing for loto.ro games.

Prices confirmed by user 2026-04-18:

    Joker variant:     7.0 RON   (2 variants per ticket)
    Loto 6/49 variant: 8.0 RON   (3 variants per ticket)
    Loto 5/40 variant: 5.0 RON   (4 variants per ticket)
    Processing fee:    0.5 RON   (flat, one per main-game ticket)
    Noroc Plus stake:  3.0 RON   (optional Joker side game)
    Noroc stake:       4.0 RON   (optional 6/49 side game)
    Super Noroc stake: 2.0 RON   (optional 5/40 side game)

Full ticket prices (all fields combined):
    Joker     = 2*7.0 + 0.5 + 3.0 = 17.5 RON
    Loto 6/49 = 3*8.0 + 0.5 + 4.0 = 28.5 RON
    Loto 5/40 = 4*5.0 + 0.5 + 2.0 = 22.5 RON

Legacy `game_recommender.TICKET_COSTS` holds incorrect per-variant
values (8/6/4) and is used by ev_calculator break-even math and the
current budget allocator. Plan C migrates callers to this module and
removes the legacy constants.
"""
from __future__ import annotations

PRICE_PER_VARIANT: dict[str, float] = {
    "joker": 7.0,
    "loto_649": 8.0,
    "loto_540": 5.0,
}

VARIANTS_PER_TICKET: dict[str, int] = {
    "joker": 2,
    "loto_649": 3,
    "loto_540": 4,
}

SIDE_GAME_PRICE: dict[str, float] = {
    "joker": 3.0,      # Noroc Plus
    "loto_649": 4.0,   # Noroc
    "loto_540": 2.0,   # Super Noroc
}

PROCESSING_FEE_RON: float = 0.5

TICKET_PRICE_NEEDS_VERIFICATION: bool = False


def compute_ticket_cost(
    game: str,
    variants: int | None = None,
    include_side_game: bool = True,
    include_fee: bool = True,
) -> float:
    """Return ticket cost in RON.

    Args:
        game: "joker" | "loto_649" | "loto_540"
        variants: number of variants on the ticket. Defaults to
            VARIANTS_PER_TICKET[game] (full ticket).
        include_side_game: add SIDE_GAME_PRICE[game] when True.
        include_fee: add PROCESSING_FEE_RON when True. The fee is paid
            once per physical ticket regardless of variant count.
    """
    if game not in PRICE_PER_VARIANT:
        raise KeyError(f"Unknown game: {game}")
    n_variants = VARIANTS_PER_TICKET[game] if variants is None else variants
    cost = PRICE_PER_VARIANT[game] * n_variants
    if include_fee:
        cost += PROCESSING_FEE_RON
    if include_side_game:
        cost += SIDE_GAME_PRICE[game]
    return cost
