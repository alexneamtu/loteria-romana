"""Romanian tax on gambling winnings (Legea 141/2025, in force 2025-08-01).

Withheld per payment, progressive:

    <= 10,000 RON          4%
    10,000 - 66,750 RON    400 + 20% of the excess over 10,000
    > 66,750 RON           11,750 + 40% of the excess over 66,750

The brackets are continuous (4% of 10,000 = 400; 400 + 20% of 56,750 = 11,750),
so `gross_for_net` inverts cleanly.

A flat rate cannot represent this: a 30 RON Category III prize keeps 96% while a
3M RON jackpot keeps ~60%, and breakeven math needs both.
"""
from __future__ import annotations

# (upper bound of bracket, marginal rate, tax accumulated at the lower bound)
_BRACKETS = (
    (10_000.0, 0.04, 0.0),
    (66_750.0, 0.20, 400.0),
    (float("inf"), 0.40, 11_750.0),
)
_LOWER = (0.0, 10_000.0, 66_750.0)


def tax_on(prize: float) -> float:
    """Tax withheld on a single prize payment."""
    if prize <= 0:
        return 0.0
    for (upper, rate, base), lower in zip(_BRACKETS, _LOWER):
        if prize <= upper:
            return base + rate * (prize - lower)
    return 0.0  # unreachable: last bracket is unbounded


def net_of_tax(prize: float) -> float:
    """What the winner actually receives."""
    return prize - tax_on(prize)


def gross_for_net(net: float) -> float:
    """Prize whose after-tax value is `net` — the inverse of net_of_tax."""
    if net <= 0:
        return 0.0
    gross = 0.0
    for (upper, rate, base), lower in zip(_BRACKETS, _LOWER):
        # net = prize - base - rate*(prize - lower)
        gross = (net + base - rate * lower) / (1 - rate)
        if gross <= upper:
            break
    return gross


def demo() -> None:
    assert tax_on(10_000) == 400.0
    assert tax_on(66_750) == 11_750.0
    assert abs(tax_on(100_000) - (11_750 + 0.4 * 33_250)) < 1e-9
    # Continuity across both boundaries.
    assert abs(tax_on(10_000.01) - tax_on(10_000)) < 0.01
    assert abs(tax_on(66_750.01) - tax_on(66_750)) < 0.01
    # Round trip through every bracket.
    for prize in (500.0, 10_000.0, 25_000.0, 66_750.0, 445_294.0, 3_356_991.0):
        assert abs(gross_for_net(net_of_tax(prize)) - prize) < 1e-6, prize
    print("tax: ok")


if __name__ == "__main__":
    demo()
