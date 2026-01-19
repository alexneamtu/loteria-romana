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

from dataclasses import dataclass, field
from typing import Optional
import math


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
    def create_loto_649(ticket_cost: float = 6.0) -> LotteryGame:
        """
        Create Loto 6/49 game configuration.

        Prize structure:
        - Category I: 6 matches (jackpot, pari-mutuel)
        - Category II: 5 matches (pari-mutuel)
        - Category III: 4 matches (fixed)
        - Category IV: 3 matches (fixed)
        """
        game = LotteryGame(
            name="Loto 6/49",
            pool_size=49,
            numbers_drawn=6,
            numbers_picked=6,
            ticket_cost=ticket_cost,
            has_bonus=False,
            jackpot_seed=100_000,  # Minimum jackpot
            rollover_percentage=1.0  # Full rollover
        )

        game.prize_tiers = [
            PrizeTier(
                name="Category I (6 matches)",
                matches_required=6,
                prize_pool_percentage=0.50,  # 50% of prize fund
            ),
            PrizeTier(
                name="Category II (5 matches)",
                matches_required=5,
                prize_pool_percentage=0.15,  # 15% of prize fund
            ),
            PrizeTier(
                name="Category III (4 matches)",
                matches_required=4,
                fixed_prize=80.0,
            ),
            PrizeTier(
                name="Category IV (3 matches)",
                matches_required=3,
                fixed_prize=20.0,
            ),
        ]

        # Calculate probabilities
        EVCalculator._calculate_probabilities(game)

        return game

    @staticmethod
    def create_loto_540(ticket_cost: float = 4.0) -> LotteryGame:
        """
        Create Loto 5/40 game configuration.

        Special rule: 6 numbers are drawn, player picks 5.
        Prize structure based on matches out of 6 drawn.
        """
        game = LotteryGame(
            name="Loto 5/40",
            pool_size=40,
            numbers_drawn=6,  # 6 are drawn
            numbers_picked=5,  # Player picks 5
            ticket_cost=ticket_cost,
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
                prize_pool_percentage=0.20,
            ),
            PrizeTier(
                name="Category III (4 matches)",
                matches_required=4,
                fixed_prize=30.0,
            ),
            PrizeTier(
                name="Category IV (3 matches)",
                matches_required=3,
                fixed_prize=6.0,
            ),
        ]

        EVCalculator._calculate_probabilities_540(game)

        return game

    @staticmethod
    def create_joker(ticket_cost: float = 8.0) -> LotteryGame:
        """
        Create Joker game configuration.

        Structure: 5 main numbers from 1-45, plus 1 Joker from 1-20.
        """
        game = LotteryGame(
            name="Joker",
            pool_size=45,
            numbers_drawn=5,
            numbers_picked=5,
            ticket_cost=ticket_cost,
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
                prize_pool_percentage=0.45,
            ),
            PrizeTier(
                name="Category II (5 matches)",
                matches_required=5,
                bonus_required=False,
                prize_pool_percentage=0.10,
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
                fixed_prize=100.0,
            ),
            PrizeTier(
                name="Category V (3+Joker)",
                matches_required=3,
                bonus_required=True,
                fixed_prize=50.0,
            ),
            PrizeTier(
                name="Category VI (3 matches)",
                matches_required=3,
                bonus_required=False,
                fixed_prize=14.0,
            ),
            PrizeTier(
                name="Category VII (2+Joker)",
                matches_required=2,
                bonus_required=True,
                fixed_prize=14.0,
            ),
            PrizeTier(
                name="Category VIII (1+Joker)",
                matches_required=1,
                bonus_required=True,
                fixed_prize=8.0,
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
                # Special case: all 5 picks among the 6 drawn (5+1)
                # This means 5 of our picks match 5 of the 6 drawn
                # P = C(6,5) * C(34,0) / C(40,5) = 6 / C(40,5)
                ways = EVCalculator._combinations(drawn, picked)
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
                     tax_rate: float = 0.0) -> EVResult:
        """
        Calculate expected value for a lottery ticket.

        Args:
            game: Lottery game configuration
            jackpot: Current jackpot amount (uses seed if not provided)
            expected_winners: Expected number of jackpot winners (for splitting)
            tax_rate: Tax rate on winnings (0-1)

        Returns:
            EVResult with detailed breakdown
        """
        if jackpot is None:
            jackpot = game.jackpot_seed

        total_ev = 0.0
        tier_breakdown = []

        for tier in game.prize_tiers:
            if tier.fixed_prize is not None:
                prize = tier.fixed_prize
            elif tier.matches_required == game.numbers_drawn or \
                 (tier.matches_required == game.numbers_picked and
                  tier.bonus_required and game.has_bonus):
                # Jackpot tier - share among expected winners
                prize = jackpot / expected_winners
            else:
                # Other pari-mutuel tiers (estimate based on percentage)
                # Rough estimate: assume prize pool is ~50% of ticket sales
                prize = 1000.0 * (tier.prize_pool_percentage or 0.05)

            # Apply tax
            net_prize = prize * (1 - tax_rate)

            # Calculate EV contribution
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

        # Final EV is total expected returns minus ticket cost
        net_ev = total_ev - game.ticket_cost
        return_pct = (total_ev / game.ticket_cost) * 100

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

    def _calculate_positive_ev_jackpot(self, game: LotteryGame,
                                       expected_winners: float,
                                       tax_rate: float) -> Optional[float]:
        """
        Calculate minimum jackpot needed for positive EV.

        Returns None if +EV is impossible (no jackpot tier).
        """
        # Find jackpot tier
        jackpot_tier = None
        for tier in game.prize_tiers:
            if tier.matches_required == game.numbers_drawn or \
               (tier.matches_required == game.numbers_picked and
                tier.bonus_required and game.has_bonus):
                jackpot_tier = tier
                break

        if jackpot_tier is None or jackpot_tier.probability == 0:
            return None

        # Calculate fixed EV from non-jackpot tiers
        fixed_ev = 0.0
        for tier in game.prize_tiers:
            if tier is not jackpot_tier and tier.fixed_prize is not None:
                fixed_ev += tier.probability * tier.fixed_prize * (1 - tax_rate)

        # EV = fixed_ev + P(jackpot) * jackpot / winners - cost > 0
        # jackpot > (cost - fixed_ev) * winners / (P(jackpot) * (1-tax))
        needed_ev = game.ticket_cost - fixed_ev
        denominator = jackpot_tier.probability * (1 - tax_rate)

        if denominator > 0:
            min_jackpot = (needed_ev * expected_winners) / denominator
            return max(0, min_jackpot)

        return None

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
        breakeven = self._calculate_positive_ev_jackpot(game, 1.0, 0.0)

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
