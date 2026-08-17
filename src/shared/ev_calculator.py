"""
Expected Value Calculator for Romanian Lottery Games.

This module calculates the true expected value (EV) of lottery tickets,
accounting for:
- All prize tiers and their probabilities
- Progressive jackpots and roll-overs
- Ticket costs
- Tax implications (if applicable)

The key insight: EV is almost always negative, but certain conditions
(large jackpots, roll-down events) can create +EV situations.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from .tax import gross_for_net, net_of_tax

# Share of eligible stakes assigned to the prize fund. The regulation sets a
# 40% minimum, and three Loto Prono bulletins (5/40 and 6/49 13.07.2025, Joker
# 08.05.2025) reconcile to exactly 0.400000 once report carry-ins are removed.
# Those bulletins were not retrieved first-hand, and back-solving the fraction
# from individual draw reports scatters it across roughly 0.27-0.64 — the
# per-draw sales reconstruction is too loose to pin it down.
#
# So: unverified but reasonable, and deliberately walled off from the spend
# decision. Only _parimutuel_prize uses it, and the breakeven that gates
# spending counts declared prizes only.
PARIMUTUEL_PAYOUT_FRACTION = 0.40


@dataclass
class PrizeTier:
    """Definition of a lottery prize tier."""
    name: str
    matches_required: int  # Main numbers required
    bonus_required: bool = False  # Whether bonus/joker match is required
    fixed_prize: Optional[float] = None  # Fixed amount (if not pari-mutuel)
    prize_pool_percentage: Optional[float] = None  # % of pool for pari-mutuel
    probability: float = 0.0  # Will be calculated


@dataclass
class LotteryGame:
    """Configuration for a lottery game."""
    name: str
    pool_size: int  # Numbers to choose from (e.g., 49)
    numbers_drawn: int  # Numbers drawn (e.g., 6)
    numbers_picked: int  # Numbers player picks (usually same as drawn)
    ticket_cost: float
    # Lines (variants) a full ticket buys. ticket_cost is the whole-ticket
    # cost but prize probabilities are per-line, so breakeven math must
    # spread the cost across the lines the ticket actually plays.
    lines_per_ticket: int = 1
    # Stake per line that funds prizes. NOT ticket_cost/lines: the 0.50 RON
    # processing fee is charged once per ticket and funds nothing.
    stake_per_line: float = 0.0
    prize_tiers: list[PrizeTier] = field(default_factory=list)
    has_bonus: bool = False
    bonus_pool_size: int = 0
    jackpot_seed: float = 0.0  # Starting jackpot amount
    rollover_percentage: float = 0.0  # % of jackpot that rolls over


@dataclass
class EVResult:
    """Result of expected value calculation."""
    game_name: str
    ticket_cost: float
    expected_value: float
    return_percentage: float  # EV / cost as percentage
    is_positive_ev: bool
    tier_breakdown: list[dict] = field(default_factory=list)
    jackpot_for_positive_ev: Optional[float] = None
    analysis: str = ""


class EVCalculator:
    """
    Expected Value calculator for lottery games.

    Calculates true mathematical EV accounting for all prize tiers
    and jackpot amounts. Can determine jackpot threshold for +EV.

    Example:
        calc = EVCalculator()
        game = calc.create_loto_649()
        result = calc.calculate_ev(game, jackpot=10_000_000)
        print(f"EV: {result.expected_value:.2f} RON")
    """

    # Romanian lottery prize structures (as of 2024)
    # Note: These should be updated periodically as rules change

    @staticmethod
    def create_loto_649(ticket_cost: float | None = None) -> LotteryGame:
        """
        Create Loto 6/49 game configuration.

        Prize structure (verified against the 16.08.2026 report):
        - Category IV: 3 matches, fixed 50 RON, paid off the top
        - Categories I/II/III: 60/20/20 of what remains, all pari-mutuel

        The default ticket_cost excludes the Noroc side-game stake: EV
        tiers model main-game prizes only; side games have separate
        prize ladders and are not in scope here.
        """
        from .pricing import PRICE_PER_VARIANT, VARIANTS_PER_TICKET, compute_ticket_cost
        if ticket_cost is None:
            ticket_cost = compute_ticket_cost("loto_649", include_side_game=False)
        game = LotteryGame(
            name="Loto 6/49",
            pool_size=49,
            numbers_drawn=6,
            numbers_picked=6,
            ticket_cost=ticket_cost,
            lines_per_ticket=VARIANTS_PER_TICKET["loto_649"],
            stake_per_line=PRICE_PER_VARIANT["loto_649"],
            has_bonus=False,
            jackpot_seed=100_000,  # Minimum jackpot
            rollover_percentage=1.0  # Full rollover
        )

        game.prize_tiers = [
            PrizeTier(
                name="Category I (6 matches)",
                matches_required=6,
                prize_pool_percentage=0.60,
            ),
            PrizeTier(
                name="Category II (5 matches)",
                matches_required=5,
                prize_pool_percentage=0.20,
            ),
            PrizeTier(
                name="Category III (4 matches)",
                matches_required=4,
                prize_pool_percentage=0.20,  # was "fixed 80"; report shows 345.57
            ),
            PrizeTier(
                name="Category IV (3 matches)",
                matches_required=3,
                fixed_prize=50.0,  # was 20.0; report shows 50.00/winner
            ),
        ]

        # Calculate probabilities
        EVCalculator._calculate_probabilities(game)

        return game

    @staticmethod
    def create_loto_540(ticket_cost: float | None = None) -> LotteryGame:
        """
        Create Loto 5/40 game configuration.

        Special rule: 6 numbers are drawn, player picks 5.
        Prize structure based on matches out of 6 drawn.

        The default ticket_cost excludes the Super Noroc side-game stake: EV
        tiers model main-game prizes only; side games have separate
        prize ladders and are not in scope here.
        """
        from .pricing import PRICE_PER_VARIANT, VARIANTS_PER_TICKET, compute_ticket_cost
        if ticket_cost is None:
            ticket_cost = compute_ticket_cost("loto_540", include_side_game=False)
        game = LotteryGame(
            name="Loto 5/40",
            pool_size=40,
            numbers_drawn=6,  # 6 are drawn
            numbers_picked=5,  # Player picks 5
            ticket_cost=ticket_cost,
            lines_per_ticket=VARIANTS_PER_TICKET["loto_540"],
            stake_per_line=PRICE_PER_VARIANT["loto_540"],
            has_bonus=False,
            jackpot_seed=50_000,
            rollover_percentage=1.0
        )

        game.prize_tiers = [
            PrizeTier(
                name="Category I (5+1 matches)",
                matches_required=6,  # All 5 picks match, plus the 6th drawn
                prize_pool_percentage=0.50,
            ),
            PrizeTier(
                name="Category II (5 matches)",
                matches_required=5,
                prize_pool_percentage=0.25,
            ),
            PrizeTier(
                name="Category III (4 matches)",
                matches_required=4,
                prize_pool_percentage=0.25,  # was "fixed 30"; report shows 830.15
            ),
            # Loto 5/40 has three categories only — 3 matches pays nothing.
        ]

        EVCalculator._calculate_probabilities_540(game)

        return game

    @staticmethod
    def create_joker(ticket_cost: float | None = None) -> LotteryGame:
        """
        Create Joker game configuration.

        Structure: 5 main numbers from 1-45, plus 1 Joker from 1-20.

        The default ticket_cost excludes the Noroc Plus side-game stake: EV
        tiers model main-game prizes only; side games have separate
        prize ladders and are not in scope here.
        """
        from .pricing import PRICE_PER_VARIANT, VARIANTS_PER_TICKET, compute_ticket_cost
        if ticket_cost is None:
            ticket_cost = compute_ticket_cost("joker", include_side_game=False)
        game = LotteryGame(
            name="Joker",
            pool_size=45,
            numbers_drawn=5,
            numbers_picked=5,
            ticket_cost=ticket_cost,
            lines_per_ticket=VARIANTS_PER_TICKET["joker"],
            stake_per_line=PRICE_PER_VARIANT["joker"],
            has_bonus=True,
            bonus_pool_size=20,
            jackpot_seed=200_000,
            rollover_percentage=1.0
        )

        game.prize_tiers = [
            PrizeTier(
                name="Category I (5+Joker)",
                matches_required=5,
                bonus_required=True,
                prize_pool_percentage=0.37,
            ),
            PrizeTier(
                name="Category II (5 matches)",
                matches_required=5,
                bonus_required=False,
                prize_pool_percentage=0.08,
            ),
            PrizeTier(
                name="Category III (4+Joker)",
                matches_required=4,
                bonus_required=True,
                prize_pool_percentage=0.08,
            ),
            PrizeTier(
                name="Category IV (4 matches)",
                matches_required=4,
                bonus_required=False,
                prize_pool_percentage=0.04,
            ),
            PrizeTier(
                name="Category V (3+Joker)",
                matches_required=3,
                bonus_required=True,
                prize_pool_percentage=0.05,
            ),
            PrizeTier(
                name="Category VI (3 matches)",
                matches_required=3,
                bonus_required=False,
                prize_pool_percentage=0.13,
            ),
            PrizeTier(
                name="Category VII (2+Joker)",
                matches_required=2,
                bonus_required=True,
                prize_pool_percentage=0.06,
            ),
            PrizeTier(
                name="Category VIII (1+Joker)",
                matches_required=1,
                bonus_required=True,
                prize_pool_percentage=0.19,
            ),
        ]

        EVCalculator._calculate_probabilities_joker(game)

        return game

    @staticmethod
    def _calculate_probabilities(game: LotteryGame) -> None:
        """Calculate win probabilities for standard lottery (6/49 style)."""
        n = game.pool_size
        k = game.numbers_drawn
        total_combinations = EVCalculator._combinations(n, k)

        for tier in game.prize_tiers:
            m = tier.matches_required
            # Probability = C(k,m) * C(n-k, k-m) / C(n,k)
            ways_to_match = EVCalculator._combinations(k, m)
            ways_to_not_match = EVCalculator._combinations(n - k, k - m)
            tier.probability = (ways_to_match * ways_to_not_match) / total_combinations

    @staticmethod
    def _calculate_probabilities_540(game: LotteryGame) -> None:
        """Calculate probabilities for Loto 5/40 (5 from 40, 6 drawn)."""
        n = game.pool_size  # 40
        drawn = game.numbers_drawn  # 6
        picked = game.numbers_picked  # 5

        total_combinations = EVCalculator._combinations(n, picked)

        for tier in game.prize_tiers:
            m = tier.matches_required

            if m == 6:
                # Category I is NOT "5 picks among the 6 drawn" — loto.ro
                # defines it as the 5 picks equalling the FIRST five balls
                # drawn ("5 numere din primele 5 extrase"). Exactly one of
                # the C(40,5) sets qualifies. Modelling it as C(6,5)/C(40,5)
                # made the jackpot 6x too likely and the breakeven jackpot
                # 6x too low, which is what drove the EV gate to boost.
                tier.probability = 1 / total_combinations
            elif m == picked:
                # Category II: the other five 5-subsets of the six drawn.
                ways = EVCalculator._combinations(drawn, picked) - 1
                tier.probability = ways / total_combinations
            else:
                # m matches out of our 5 picks
                # P = C(6,m) * C(34, 5-m) / C(40,5)
                ways_to_match = EVCalculator._combinations(drawn, m)
                ways_to_not_match = EVCalculator._combinations(n - drawn, picked - m)
                tier.probability = (ways_to_match * ways_to_not_match) / total_combinations

    @staticmethod
    def _calculate_probabilities_joker(game: LotteryGame) -> None:
        """Calculate probabilities for Joker (5/45 + 1/20)."""
        n = game.pool_size  # 45
        k = game.numbers_drawn  # 5
        bonus_size = game.bonus_pool_size  # 20

        total_main = EVCalculator._combinations(n, k)
        total_combinations = total_main * bonus_size  # Main * Joker combinations

        for tier in game.prize_tiers:
            m = tier.matches_required
            ways_main = EVCalculator._combinations(k, m) * EVCalculator._combinations(n - k, k - m)

            if tier.bonus_required:
                # Must match joker (1/20)
                tier.probability = ways_main / total_combinations
            else:
                # Must NOT match joker (19/20)
                tier.probability = (ways_main * (bonus_size - 1)) / total_combinations

    @staticmethod
    def _combinations(n: int, r: int) -> int:
        """Calculate C(n,r) = n! / (r! * (n-r)!)"""
        if r < 0 or r > n:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)  # Optimization
        result = 1
        for i in range(r):
            result = result * (n - i) // (i + 1)
        return result

    def calculate_ev(self, game: LotteryGame,
                     jackpot: Optional[float] = None,
                     expected_winners: float = 1.0,
                     tax_rate: Optional[float] = None) -> EVResult:
        """
        Calculate expected value for a lottery ticket.

        Args:
            game: Lottery game configuration
            jackpot: Current jackpot amount (uses seed if not provided)
            expected_winners: Expected number of jackpot winners (for splitting)
            tax_rate: Flat rate on winnings (0-1), or None (default) for
                the Romanian progressive schedule

        Returns:
            EVResult with detailed breakdown
        """
        if jackpot is None:
            jackpot = game.jackpot_seed

        total_ev = 0.0
        tier_breakdown = []

        jackpot_tier = self._jackpot_tier(game)

        for tier in game.prize_tiers:
            if tier is jackpot_tier:
                prize = jackpot / expected_winners
            elif tier.fixed_prize is not None:
                prize = tier.fixed_prize
            else:
                prize = self._parimutuel_prize(game, tier)

            net_prize = self._net_prize(prize, tax_rate)
            ev_contribution = tier.probability * net_prize
            total_ev += ev_contribution

            tier_breakdown.append({
                "tier": tier.name,
                "probability": tier.probability,
                "odds": f"1 in {1/tier.probability:,.0f}" if tier.probability > 0 else "N/A",
                "prize": prize,
                "net_prize": net_prize,
                "ev_contribution": ev_contribution
            })

        # total_ev is per line (tier probabilities are per line) but
        # ticket_cost buys lines_per_ticket of them. Scale before subtracting,
        # or a 4-line 5/40 ticket looks 3 lines' worth of returns short.
        ticket_ev = total_ev * game.lines_per_ticket
        net_ev = ticket_ev - game.ticket_cost
        return_pct = (ticket_ev / game.ticket_cost) * 100

        # Calculate jackpot needed for +EV
        jackpot_for_positive = self._calculate_positive_ev_jackpot(
            game, expected_winners, tax_rate
        )

        analysis = self._generate_analysis(game, jackpot, net_ev, jackpot_for_positive)

        return EVResult(
            game_name=game.name,
            ticket_cost=game.ticket_cost,
            expected_value=net_ev,
            return_percentage=return_pct,
            is_positive_ev=net_ev > 0,
            tier_breakdown=tier_breakdown,
            jackpot_for_positive_ev=jackpot_for_positive,
            analysis=analysis
        )

    @staticmethod
    def _jackpot_tier(game: LotteryGame) -> Optional[PrizeTier]:
        """The single top tier.

        For a bonus game that is "all main numbers + the bonus"; otherwise
        "all numbers drawn". Testing `matches_required == numbers_drawn`
        alone also matched Joker's Category II (5 main, no Joker), which was
        then paid the entire jackpot in calculate_ev.
        """
        for tier in game.prize_tiers:
            if game.has_bonus:
                if tier.bonus_required and tier.matches_required == game.numbers_picked:
                    return tier
            elif tier.matches_required == game.numbers_drawn:
                return tier
        return None

    @staticmethod
    def _parimutuel_prize(game: LotteryGame, tier: PrizeTier) -> float:
        """Estimate a non-jackpot pari-mutuel prize.

        A tier's pool is `pct` of the prize fund, the fund is
        `PARIMUTUEL_PAYOUT_FRACTION` of sales, and the winners in that tier
        are `sales_lines * p`. Sales cancel:

            prize = pct * payout_fraction * price_per_line / p

        The old `1000.0 * pct` dropped the `1/p`, which is the whole point —
        a rarer tier pays more. That understated Category II prizes by two to
        three orders of magnitude.

        ponytail: payout_fraction is one national average, not per-game and
        not per-draw. Checked against Aug-2026 loto.ro reports it lands within
        ~2-3x of actual; swap in scraped per-draw pools if the gate ever needs
        better than that.
        """
        if tier.probability <= 0:
            return 0.0
        pct = tier.prize_pool_percentage or 0.05
        return pct * EVCalculator._distributable_per_line(game) / tier.probability

    @staticmethod
    def _distributable_per_line(game: LotteryGame) -> float:
        """Prize fund per line left for the pari-mutuel shares.

        Fixed prizes are paid off the top and the percentages apply to what
        remains. Verified on the 16.08.2026 6/49 report: Category IV paid
        12,018 x 50 = 600,900, and Categories I/II/III came to exactly
        60/20/20 of the 1,173,210 that was left.
        """
        fund = PARIMUTUEL_PAYOUT_FRACTION * game.stake_per_line
        fixed = sum(
            t.probability * t.fixed_prize
            for t in game.prize_tiers
            if t.fixed_prize is not None
        )
        return max(0.0, fund - fixed)

    @staticmethod
    def _net_prize(prize: float, tax_rate: Optional[float]) -> float:
        """After-tax prize. `tax_rate=None` uses the Romanian schedule."""
        if tax_rate is None:
            return net_of_tax(prize)
        return prize * (1 - tax_rate)

    @staticmethod
    def _gross_prize(net: float, tax_rate: Optional[float]) -> float:
        """Inverse of _net_prize."""
        if tax_rate is None:
            return gross_for_net(net)
        if tax_rate >= 1:
            return float("inf")
        return net / (1 - tax_rate)

    def _calculate_positive_ev_jackpot(self, game: LotteryGame,
                                       expected_winners: float,
                                       tax_rate: Optional[float]) -> Optional[float]:
        """
        Calculate minimum jackpot needed for positive EV.

        Returns None if +EV is impossible (no jackpot tier).
        """
        jackpot_tier = self._jackpot_tier(game)

        if jackpot_tier is None or jackpot_tier.probability == 0:
            return None

        # Declared prize amounts only. `_parimutuel_prize` is a long-run
        # approximation — it ignores tier carry-ins, replaces the winner count
        # with its expectation, and rests on an assumed payout fraction. This
        # number gates real spending, and crediting estimated upside here can
        # only ever turn "block" into "spend". A false pass costs a ticket run;
        # a false block costs nothing but the delay, so the bias goes that way.
        #
        # After the tier data was corrected against the 16.08.2026 reports,
        # 6/49 Category IV (50 RON) is the only genuinely fixed prize left in
        # any game, so 5/40 and Joker now gate on the jackpot alone.
        fixed_ev = 0.0
        for tier in game.prize_tiers:
            if tier is jackpot_tier or tier.fixed_prize is None:
                continue
            fixed_ev += tier.probability * self._net_prize(tier.fixed_prize, tax_rate)

        # A ticket buys `lines_per_ticket` independent lines, but the tier
        # probabilities and fixed_ev above are per line. Spread the whole-
        # ticket cost across those lines so cost and odds are both per line
        # (equivalently: charge the full cost against lines*P). Charging the
        # full multi-variant cost against a single line's P overstated
        # breakeven by roughly the variant count.
        # EV/line = fixed_ev + P(jackpot) * jackpot / winners - cost/lines > 0
        # jackpot > (cost/lines - fixed_ev) * winners / (P(jackpot) * (1-tax))
        cost_per_line = game.ticket_cost / game.lines_per_ticket
        needed_ev = cost_per_line - fixed_ev
        if jackpot_tier.probability <= 0:
            return None

        # Net *share* that closes the gap, grossed back up, then multiplied by
        # the number of winners the jackpot is split between. Tax is
        # progressive, so this cannot be a `/(1 - rate)` division — and the
        # split applies once, to the gross total, not also to the per-winner
        # share (that doubled breakeven at expected_winners=2).
        needed_net_share = needed_ev / jackpot_tier.probability
        gross_share = self._gross_prize(needed_net_share, tax_rate)
        return max(0.0, gross_share * expected_winners)

    def _generate_analysis(self, game: LotteryGame, jackpot: float,
                          net_ev: float, jackpot_for_positive: Optional[float]) -> str:
        """Generate human-readable analysis of EV calculation."""
        lines = []

        lines.append(f"Analysis for {game.name}:")
        lines.append(f"- Ticket cost: {game.ticket_cost:.2f} RON")
        lines.append(f"- Current jackpot: {jackpot:,.0f} RON")
        lines.append(f"- Expected value: {net_ev:.4f} RON per ticket")

        if net_ev < 0:
            loss_pct = (-net_ev / game.ticket_cost) * 100
            lines.append(f"- Expected loss: {loss_pct:.1f}% of ticket cost")
        else:
            gain_pct = (net_ev / game.ticket_cost) * 100
            lines.append(f"- Expected gain: {gain_pct:.1f}% of ticket cost")

        if jackpot_for_positive is not None:
            if jackpot >= jackpot_for_positive:
                lines.append("\n⚠️ POSITIVE EV DETECTED!")
                lines.append("This is a mathematically profitable bet.")
            else:
                lines.append(f"\nJackpot needed for +EV: {jackpot_for_positive:,.0f} RON")
                lines.append(f"Current jackpot is {(jackpot/jackpot_for_positive)*100:.1f}% of target")

        return "\n".join(lines)

    def simulate_long_term(self, game: LotteryGame,
                          num_tickets: int,
                          jackpot: float,
                          simulations: int = 10000) -> dict:
        """
        Simulate long-term outcomes of playing the lottery.

        Args:
            game: Lottery game configuration
            num_tickets: Number of tickets to simulate per "lifetime"
            jackpot: Jackpot amount for calculation
            simulations: Number of simulations to run

        Returns:
            Dictionary with simulation statistics
        """
        import random

        results = []
        jackpot_wins = 0

        for _ in range(simulations):
            total_spent = num_tickets * game.ticket_cost
            total_won = 0.0

            for _ in range(num_tickets):
                # Simulate each ticket
                for tier in game.prize_tiers:
                    if random.random() < tier.probability:
                        if tier.fixed_prize is not None:
                            total_won += tier.fixed_prize
                        else:
                            # Jackpot win
                            total_won += jackpot
                            jackpot_wins += 1
                        break  # Only win highest tier

            net_result = total_won - total_spent
            results.append(net_result)

        results.sort()
        mean_result = sum(results) / len(results)
        median_result = results[len(results) // 2]

        # Calculate percentiles
        p5 = results[int(len(results) * 0.05)]
        p25 = results[int(len(results) * 0.25)]
        p75 = results[int(len(results) * 0.75)]
        p95 = results[int(len(results) * 0.95)]

        return {
            "total_tickets": num_tickets,
            "total_cost": num_tickets * game.ticket_cost,
            "simulations": simulations,
            "mean_net_result": mean_result,
            "median_net_result": median_result,
            "percentile_5": p5,
            "percentile_25": p25,
            "percentile_75": p75,
            "percentile_95": p95,
            "jackpot_win_rate": jackpot_wins / simulations,
            "probability_of_profit": sum(1 for r in results if r > 0) / len(results),
            "worst_outcome": results[0],
            "best_outcome": results[-1]
        }

    def compare_strategies(self, game: LotteryGame,
                          jackpot: float,
                          budget: float) -> dict:
        """
        Compare different betting strategies for given budget.

        Strategies:
        1. Single ticket
        2. Multiple tickets (increase coverage)
        3. Wait for higher jackpot

        Args:
            game: Lottery game configuration
            jackpot: Current jackpot
            budget: Total budget to spend

        Returns:
            Dictionary comparing strategies
        """
        num_tickets = int(budget / game.ticket_cost)

        single_ev = self.calculate_ev(game, jackpot)

        # Multiple tickets - EV scales linearly (no benefit unless wheeling)
        multi_ev = single_ev.expected_value * num_tickets

        # Calculate breakeven jackpot
        breakeven = self._calculate_positive_ev_jackpot(game, 1.0, None)

        return {
            "current_jackpot": jackpot,
            "ticket_cost": game.ticket_cost,
            "budget": budget,
            "max_tickets": num_tickets,
            "strategies": {
                "single_ticket": {
                    "tickets": 1,
                    "cost": game.ticket_cost,
                    "expected_value": single_ev.expected_value,
                    "return_pct": single_ev.return_percentage
                },
                "multiple_tickets": {
                    "tickets": num_tickets,
                    "cost": num_tickets * game.ticket_cost,
                    "expected_value": multi_ev,
                    "return_pct": single_ev.return_percentage,  # Same percentage
                    "note": "EV scales linearly; no mathematical advantage"
                },
                "wait_for_jackpot": {
                    "breakeven_jackpot": breakeven,
                    "current_vs_breakeven": jackpot / breakeven if breakeven else 0,
                    "note": "Waiting only helps if jackpot grows faster than ticket sales"
                }
            },
            "recommendation": self._generate_recommendation(game, jackpot, budget, breakeven)
        }

    def _generate_recommendation(self, game: LotteryGame,
                                jackpot: float,
                                budget: float,
                                breakeven: Optional[float]) -> str:
        """Generate strategy recommendation."""
        if breakeven and jackpot >= breakeven:
            return ("RARE OPPORTUNITY: Current jackpot creates positive EV. "
                    "If you choose to play, this is mathematically the best time. "
                    "However, remember that even +EV bets can result in losses.")

        if breakeven and jackpot >= breakeven * 0.8:
            return ("Jackpot approaching +EV threshold. "
                    "From a pure math perspective, waiting is slightly better. "
                    "Entertainment value may justify current play.")

        return ("Current jackpot does not justify play from EV perspective. "
                "If playing for entertainment, set a strict budget you can afford to lose. "
                f"Expected loss: {-self.calculate_ev(game, jackpot).expected_value:.2f} RON per ticket.")

    def kelly_criterion(self, game: LotteryGame, jackpot: float,
                       bankroll: float) -> dict:
        """
        Calculate optimal bet size using Kelly Criterion.

        Kelly Criterion: f* = (bp - q) / b
        where b = odds, p = probability of winning, q = probability of losing

        For lottery, this almost always recommends not playing.

        Args:
            game: Lottery game configuration
            jackpot: Current jackpot
            bankroll: Total available bankroll

        Returns:
            Dictionary with Kelly analysis
        """
        ev_result = self.calculate_ev(game, jackpot)

        # Find jackpot tier probability
        jackpot_prob = 0.0
        for tier in game.prize_tiers:
            if tier.prize_pool_percentage and tier.matches_required >= game.numbers_drawn - 1:
                jackpot_prob = tier.probability
                break

        if jackpot_prob == 0:
            return {
                "kelly_fraction": 0.0,
                "optimal_bet": 0.0,
                "recommendation": "Cannot calculate - no jackpot tier found"
            }

        # Kelly calculation
        b = (jackpot / game.ticket_cost) - 1  # Net odds
        p = jackpot_prob
        q = 1 - p

        kelly_fraction = (b * p - q) / b if b > 0 else 0
        kelly_fraction = max(0, kelly_fraction)  # Can't bet negative

        optimal_bet = kelly_fraction * bankroll

        return {
            "edge": ev_result.expected_value / game.ticket_cost,
            "kelly_fraction": kelly_fraction,
            "optimal_bet": optimal_bet,
            "recommended_tickets": int(optimal_bet / game.ticket_cost),
            "analysis": (
                f"Kelly suggests betting {kelly_fraction*100:.6f}% of bankroll.\n"
                f"For {bankroll:,.0f} RON bankroll: {optimal_bet:.2f} RON optimal bet.\n"
                f"This is {optimal_bet/game.ticket_cost:.1f} tickets.\n"
                "Note: Kelly assumes you can play indefinitely at these odds."
            )
        }
