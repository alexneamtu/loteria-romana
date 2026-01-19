"""
Paper Trading System for Lottery Strategy Evaluation.

This module implements a simulated trading system that:
- Tests strategies against historical draws without real money
- Tracks performance metrics over time
- Provides statistical significance testing
- Detects overfitting through walk-forward analysis

Key concept: Any strategy must prove itself in paper trading before
considering real deployment. The bar is high: statistically significant
positive results over an extended period.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime
import math
import json


@dataclass
class Trade:
    """Record of a single paper trade."""
    draw_id: int
    draw_date: Optional[tuple[int, int, int]]  # (year, month, day)
    strategy_name: str
    tickets: list[list[int]]  # Selected numbers for each ticket
    actual_draw: list[int]
    ticket_cost: float
    prize_won: float
    matches_per_ticket: list[int]
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    @property
    def net_result(self) -> float:
        """Net profit/loss for this trade."""
        return self.prize_won - (len(self.tickets) * self.ticket_cost)

    @property
    def is_winner(self) -> bool:
        """Whether any ticket won a prize."""
        return self.prize_won > 0


@dataclass
class StrategyStats:
    """Aggregated statistics for a strategy."""
    strategy_name: str
    total_trades: int
    total_tickets: int
    total_spent: float
    total_won: float
    winning_trades: int
    max_drawdown: float
    peak_balance: float
    sharpe_ratio: float
    win_rate: float
    roi: float
    profit_factor: float
    expected_value_per_ticket: float
    confidence_interval_95: tuple[float, float]
    is_statistically_significant: bool
    p_value: float


class PaperTrader:
    """
    Paper trading system for lottery strategy evaluation.

    Simulates real trading conditions to evaluate strategy performance
    without risking real money. Implements rigorous statistical testing
    to detect genuine edge vs random luck.

    Example:
        trader = PaperTrader(
            initial_balance=10000,
            ticket_cost=6.0,
            prize_calculator=calculate_649_prizes
        )

        for draw in historical_draws:
            picks = strategy.generate_picks(previous_draws)
            trader.execute_trade(
                draw_id=draw.id,
                strategy_name="frequency",
                tickets=picks,
                actual_draw=draw.numbers
            )

        stats = trader.get_stats("frequency")
        print(f"ROI: {stats.roi:.2%}")
    """

    def __init__(self,
                 initial_balance: float = 10000.0,
                 ticket_cost: float = 6.0,
                 prize_calculator: Optional[Callable[[list[int], list[int]], float]] = None,
                 game_name: str = "Loto 6/49"):
        """
        Initialize paper trader.

        Args:
            initial_balance: Starting balance for paper trading
            ticket_cost: Cost per ticket
            prize_calculator: Function(picks, actual) -> prize amount
            game_name: Name of lottery game being traded
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.ticket_cost = ticket_cost
        self.prize_calculator = prize_calculator or self._default_prize_calculator
        self.game_name = game_name

        self.trades: list[Trade] = []
        self.balance_history: list[tuple[int, float]] = [(0, initial_balance)]
        self.strategy_trades: dict[str, list[Trade]] = {}

    def execute_trade(self,
                      draw_id: int,
                      strategy_name: str,
                      tickets: list[list[int]],
                      actual_draw: list[int],
                      draw_date: Optional[tuple[int, int, int]] = None) -> Trade:
        """
        Execute a paper trade.

        Args:
            draw_id: Unique identifier for the draw
            strategy_name: Name of strategy being tested
            tickets: List of ticket selections (each is list of numbers)
            actual_draw: Actual winning numbers
            draw_date: Optional (year, month, day) tuple

        Returns:
            Trade record with results
        """
        cost = len(tickets) * self.ticket_cost

        # Check if we can afford the trade
        if self.balance < cost:
            raise ValueError(f"Insufficient balance: {self.balance:.2f} < {cost:.2f}")

        # Calculate prizes and matches
        total_prize = 0.0
        matches_per_ticket = []

        for ticket in tickets:
            matches = len(set(ticket) & set(actual_draw))
            matches_per_ticket.append(matches)
            prize = self.prize_calculator(ticket, actual_draw)
            total_prize += prize

        # Update balance
        self.balance -= cost
        self.balance += total_prize

        # Create trade record
        trade = Trade(
            draw_id=draw_id,
            draw_date=draw_date,
            strategy_name=strategy_name,
            tickets=tickets,
            actual_draw=actual_draw,
            ticket_cost=self.ticket_cost,
            prize_won=total_prize,
            matches_per_ticket=matches_per_ticket
        )

        # Record trade
        self.trades.append(trade)
        self.balance_history.append((len(self.trades), self.balance))

        if strategy_name not in self.strategy_trades:
            self.strategy_trades[strategy_name] = []
        self.strategy_trades[strategy_name].append(trade)

        return trade

    def get_stats(self, strategy_name: Optional[str] = None) -> StrategyStats:
        """
        Calculate comprehensive statistics for a strategy.

        Args:
            strategy_name: Strategy to analyze (None for all trades)

        Returns:
            StrategyStats with all metrics
        """
        if strategy_name:
            trades = self.strategy_trades.get(strategy_name, [])
        else:
            trades = self.trades
            strategy_name = "all_strategies"

        if not trades:
            return StrategyStats(
                strategy_name=strategy_name,
                total_trades=0,
                total_tickets=0,
                total_spent=0,
                total_won=0,
                winning_trades=0,
                max_drawdown=0,
                peak_balance=self.initial_balance,
                sharpe_ratio=0,
                win_rate=0,
                roi=0,
                profit_factor=0,
                expected_value_per_ticket=0,
                confidence_interval_95=(0, 0),
                is_statistically_significant=False,
                p_value=1.0
            )

        # Basic metrics
        total_trades = len(trades)
        total_tickets = sum(len(t.tickets) for t in trades)
        total_spent = sum(len(t.tickets) * t.ticket_cost for t in trades)
        total_won = sum(t.prize_won for t in trades)
        winning_trades = sum(1 for t in trades if t.is_winner)

        # Calculate running balance for drawdown
        running_balance = self.initial_balance
        peak = self.initial_balance
        max_drawdown = 0

        for trade in trades:
            running_balance += trade.net_result
            peak = max(peak, running_balance)
            drawdown = (peak - running_balance) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        # Per-trade returns for Sharpe ratio
        returns = [t.net_result / (len(t.tickets) * t.ticket_cost) for t in trades]
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_return = math.sqrt(variance) if variance > 0 else 0.001

        # Sharpe ratio (assuming risk-free rate of 0 for lottery)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # ROI
        roi = (total_won - total_spent) / total_spent if total_spent > 0 else 0

        # Profit factor
        gross_profit = sum(t.prize_won for t in trades if t.prize_won > 0)
        gross_loss = sum(len(t.tickets) * t.ticket_cost for t in trades if t.prize_won == 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expected value per ticket
        ev_per_ticket = (total_won - total_spent) / total_tickets if total_tickets > 0 else 0

        # Statistical significance testing
        # H0: Strategy has zero edge (EV = 0)
        # Using t-test for mean return
        n = len(returns)
        se = std_return / math.sqrt(n) if n > 0 else 0.001
        t_stat = mean_return / se if se > 0 else 0

        # Two-tailed p-value (approximate using normal for large n)
        p_value = 2 * (1 - self._standard_normal_cdf(abs(t_stat)))

        # 95% confidence interval for mean return
        ci_margin = 1.96 * se
        ci_lower = mean_return - ci_margin
        ci_upper = mean_return + ci_margin

        # Convert CI to ROI terms
        ci_lower_roi = ci_lower * total_trades * self.ticket_cost / total_spent if total_spent > 0 else 0
        ci_upper_roi = ci_upper * total_trades * self.ticket_cost / total_spent if total_spent > 0 else 0

        # Significant if p < 0.05 and positive return
        is_significant = p_value < 0.05 and mean_return > 0

        return StrategyStats(
            strategy_name=strategy_name,
            total_trades=total_trades,
            total_tickets=total_tickets,
            total_spent=total_spent,
            total_won=total_won,
            winning_trades=winning_trades,
            max_drawdown=max_drawdown,
            peak_balance=peak,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            roi=roi,
            profit_factor=profit_factor,
            expected_value_per_ticket=ev_per_ticket,
            confidence_interval_95=(ci_lower_roi, ci_upper_roi),
            is_statistically_significant=is_significant,
            p_value=p_value
        )

    def walk_forward_analysis(self,
                             strategy_func: Callable[[list[list[int]], int], list[list[int]]],
                             all_draws: list[list[int]],
                             train_size: int = 500,
                             test_size: int = 100,
                             step_size: int = 50) -> dict:
        """
        Perform walk-forward analysis to detect overfitting.

        Tests strategy on rolling out-of-sample windows to ensure
        performance persists beyond training data.

        Args:
            strategy_func: Function(historical_draws, num_picks) -> picks
            all_draws: Complete historical draws
            train_size: Number of draws for training
            test_size: Number of draws for testing
            step_size: Window step size

        Returns:
            Dictionary with walk-forward results
        """
        results = []
        window_stats = []

        start_idx = 0
        while start_idx + train_size + test_size <= len(all_draws):
            train_end = start_idx + train_size
            test_end = train_end + test_size

            train_draws = all_draws[start_idx:train_end]
            test_draws = all_draws[train_end:test_end]

            # Generate picks using training data only
            window_trades = []
            temp_balance = self.initial_balance

            for i, actual_draw in enumerate(test_draws):
                picks = strategy_func(train_draws, 1)
                prize = self.prize_calculator(picks[0], actual_draw)

                net_result = prize - self.ticket_cost
                temp_balance += net_result

                window_trades.append({
                    "draw_index": train_end + i,
                    "prize": prize,
                    "net_result": net_result
                })

            # Calculate window statistics
            window_returns = [t["net_result"] / self.ticket_cost for t in window_trades]
            mean_return = sum(window_returns) / len(window_returns)
            total_won = sum(t["prize"] for t in window_trades)
            total_spent = len(window_trades) * self.ticket_cost

            window_stats.append({
                "window_start": start_idx,
                "train_end": train_end,
                "test_end": test_end,
                "mean_return": mean_return,
                "roi": (total_won - total_spent) / total_spent,
                "win_count": sum(1 for t in window_trades if t["prize"] > 0)
            })

            results.extend(window_trades)
            start_idx += step_size

        # Aggregate statistics
        all_returns = [r["net_result"] / self.ticket_cost for r in results]
        mean_return = sum(all_returns) / len(all_returns) if all_returns else 0
        variance = sum((r - mean_return) ** 2 for r in all_returns) / len(all_returns) if all_returns else 0
        std_return = math.sqrt(variance)

        # Check for consistency across windows
        window_rois = [w["roi"] for w in window_stats]
        positive_windows = sum(1 for r in window_rois if r > 0)
        consistency_ratio = positive_windows / len(window_stats) if window_stats else 0

        return {
            "num_windows": len(window_stats),
            "total_test_draws": len(results),
            "overall_mean_return": mean_return,
            "overall_std": std_return,
            "positive_windows": positive_windows,
            "consistency_ratio": consistency_ratio,
            "window_details": window_stats,
            "is_overfitting": consistency_ratio < 0.5,  # Weak threshold
            "recommendation": self._generate_walk_forward_recommendation(
                consistency_ratio, mean_return
            )
        }

    def _generate_walk_forward_recommendation(self,
                                             consistency: float,
                                             mean_return: float) -> str:
        """Generate recommendation based on walk-forward results."""
        if consistency < 0.3:
            return ("HIGH OVERFITTING RISK: Strategy performs inconsistently "
                    "across different time periods. Results likely due to "
                    "curve fitting rather than genuine edge.")

        if consistency < 0.5:
            return ("MODERATE OVERFITTING RISK: Strategy shows mixed results. "
                    "May have some validity but needs more testing.")

        if mean_return < 0:
            return ("NEGATIVE EXPECTED VALUE: Despite consistency, "
                    "strategy has negative expected returns. Not viable.")

        if consistency >= 0.7 and mean_return > 0:
            return ("PROMISING RESULTS: Strategy shows consistent positive "
                    "returns across time periods. Continue monitoring in "
                    "live paper trading before any real deployment.")

        return ("INCONCLUSIVE: More data needed for reliable assessment.")

    def monte_carlo_significance(self,
                                strategy_results: list[float],
                                num_simulations: int = 10000) -> dict:
        """
        Test strategy results against random baseline using Monte Carlo.

        Generates random strategy results to establish null distribution
        and calculates probability of observed results under null.

        Args:
            strategy_results: Actual returns from strategy
            num_simulations: Number of random simulations

        Returns:
            Dictionary with Monte Carlo test results
        """
        import random

        observed_total = sum(strategy_results)
        n_trades = len(strategy_results)

        # Generate null distribution
        null_totals = []
        for _ in range(num_simulations):
            # Shuffle results to break any strategy-specific patterns
            shuffled = strategy_results.copy()
            random.shuffle(shuffled)
            null_totals.append(sum(shuffled))

        # Calculate p-value (proportion of null results >= observed)
        p_value = sum(1 for n in null_totals if n >= observed_total) / num_simulations

        # Calculate percentile of observed result
        percentile = sum(1 for n in null_totals if n < observed_total) / num_simulations * 100

        return {
            "observed_total": observed_total,
            "null_mean": sum(null_totals) / len(null_totals),
            "null_std": math.sqrt(sum((n - sum(null_totals)/len(null_totals))**2
                                      for n in null_totals) / len(null_totals)),
            "p_value": p_value,
            "percentile": percentile,
            "is_significant": p_value < 0.05,
            "interpretation": self._interpret_monte_carlo(p_value, percentile)
        }

    def _interpret_monte_carlo(self, p_value: float, percentile: float) -> str:
        """Interpret Monte Carlo test results."""
        if p_value < 0.01:
            return ("HIGHLY SIGNIFICANT (p<0.01): Results very unlikely under null. "
                    f"Strategy at {percentile:.1f}th percentile vs random.")
        if p_value < 0.05:
            return ("SIGNIFICANT (p<0.05): Results unlikely under null. "
                    f"Strategy at {percentile:.1f}th percentile vs random.")
        if p_value < 0.10:
            return ("MARGINALLY SIGNIFICANT (p<0.10): Some evidence of non-randomness. "
                    "Need more data to confirm.")
        return ("NOT SIGNIFICANT: Results consistent with random chance. "
                "No evidence of genuine edge.")

    def export_trades(self, filepath: str) -> None:
        """Export all trades to JSON file for analysis."""
        export_data = {
            "metadata": {
                "game_name": self.game_name,
                "initial_balance": self.initial_balance,
                "ticket_cost": self.ticket_cost,
                "export_time": datetime.now().isoformat()
            },
            "trades": [
                {
                    "draw_id": t.draw_id,
                    "draw_date": t.draw_date,
                    "strategy": t.strategy_name,
                    "tickets": t.tickets,
                    "actual_draw": t.actual_draw,
                    "prize_won": t.prize_won,
                    "matches": t.matches_per_ticket,
                    "net_result": t.net_result
                }
                for t in self.trades
            ],
            "balance_history": self.balance_history
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

    def import_trades(self, filepath: str) -> None:
        """Import trades from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.initial_balance = data["metadata"]["initial_balance"]
        self.ticket_cost = data["metadata"]["ticket_cost"]
        self.game_name = data["metadata"]["game_name"]

        self.trades = []
        self.strategy_trades = {}
        self.balance = self.initial_balance

        for trade_data in data["trades"]:
            trade = Trade(
                draw_id=trade_data["draw_id"],
                draw_date=tuple(trade_data["draw_date"]) if trade_data["draw_date"] else None,
                strategy_name=trade_data["strategy"],
                tickets=trade_data["tickets"],
                actual_draw=trade_data["actual_draw"],
                ticket_cost=self.ticket_cost,
                prize_won=trade_data["prize_won"],
                matches_per_ticket=trade_data["matches"]
            )

            self.trades.append(trade)
            self.balance += trade.net_result

            if trade.strategy_name not in self.strategy_trades:
                self.strategy_trades[trade.strategy_name] = []
            self.strategy_trades[trade.strategy_name].append(trade)

        self.balance_history = data.get("balance_history", [(0, self.initial_balance)])

    def generate_report(self) -> str:
        """Generate comprehensive paper trading report."""
        lines = [
            "=" * 60,
            "PAPER TRADING REPORT",
            f"Game: {self.game_name}",
            f"Period: {len(self.trades)} trades",
            "=" * 60,
            ""
        ]

        # Overall summary
        overall = self.get_stats()
        lines.extend([
            "OVERALL PERFORMANCE",
            "-" * 40,
            f"Initial Balance: {self.initial_balance:,.2f}",
            f"Current Balance: {self.balance:,.2f}",
            f"Net P&L: {self.balance - self.initial_balance:,.2f}",
            f"ROI: {overall.roi:.2%}",
            f"Max Drawdown: {overall.max_drawdown:.2%}",
            f"Sharpe Ratio: {overall.sharpe_ratio:.3f}",
            f"Win Rate: {overall.win_rate:.2%}",
            ""
        ])

        # Per-strategy breakdown
        if self.strategy_trades:
            lines.extend([
                "STRATEGY BREAKDOWN",
                "-" * 40
            ])

            for strategy_name in self.strategy_trades:
                stats = self.get_stats(strategy_name)
                lines.extend([
                    f"\n{strategy_name}:",
                    f"  Trades: {stats.total_trades}",
                    f"  Tickets: {stats.total_tickets}",
                    f"  Spent: {stats.total_spent:,.2f}",
                    f"  Won: {stats.total_won:,.2f}",
                    f"  ROI: {stats.roi:.2%}",
                    f"  Win Rate: {stats.win_rate:.2%}",
                    f"  EV/Ticket: {stats.expected_value_per_ticket:.4f}",
                    f"  P-Value: {stats.p_value:.4f}",
                    f"  Significant: {'Yes' if stats.is_statistically_significant else 'No'}",
                    f"  95% CI: [{stats.confidence_interval_95[0]:.2%}, {stats.confidence_interval_95[1]:.2%}]"
                ])

        # Conclusion
        lines.extend([
            "",
            "=" * 60,
            "CONCLUSION",
            "=" * 60
        ])

        best_strategy = None
        best_roi = float('-inf')

        for name in self.strategy_trades:
            stats = self.get_stats(name)
            if stats.roi > best_roi and stats.is_statistically_significant:
                best_roi = stats.roi
                best_strategy = name

        if best_strategy:
            lines.append(f"Best performing strategy: {best_strategy} (ROI: {best_roi:.2%})")
            lines.append("Note: Even statistically significant results should be")
            lines.append("verified with extended paper trading before real deployment.")
        else:
            lines.append("No strategy shows statistically significant positive returns.")
            lines.append("This is expected for truly random lottery draws.")
            lines.append("Continue paper trading to gather more data.")

        return "\n".join(lines)

    def _default_prize_calculator(self, picks: list[int], actual: list[int]) -> float:
        """Default prize calculator for Loto 6/49."""
        matches = len(set(picks) & set(actual))
        prizes = {
            6: 1_000_000,  # Jackpot (placeholder)
            5: 5_000,
            4: 80,
            3: 20,
            2: 0,
            1: 0,
            0: 0
        }
        return prizes.get(matches, 0)

    def _standard_normal_cdf(self, z: float) -> float:
        """Approximate standard normal CDF."""
        a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741,
                              -1.453152027, 1.061405429)
        p = 0.3275911

        sign = 1 if z >= 0 else -1
        z = abs(z) / math.sqrt(2)

        t = 1.0 / (1.0 + p * z)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)

        return 0.5 * (1.0 + sign * y)


def create_loto_649_prize_calculator(jackpot: float = 1_000_000) -> Callable:
    """
    Create prize calculator for Loto 6/49.

    Args:
        jackpot: Current jackpot amount

    Returns:
        Prize calculator function
    """
    def calculator(picks: list[int], actual: list[int]) -> float:
        matches = len(set(picks) & set(actual))
        prizes = {
            6: jackpot,
            5: 5000,
            4: 80,
            3: 20
        }
        return prizes.get(matches, 0)

    return calculator


def create_loto_540_prize_calculator(jackpot: float = 500_000) -> Callable:
    """
    Create prize calculator for Loto 5/40.

    Player picks 5, 6 are drawn. Prizes based on matches.
    """
    def calculator(picks: list[int], actual: list[int]) -> float:
        matches = len(set(picks) & set(actual))
        # In 5/40, player picks 5, 6 drawn
        # 5 matches = Category II, 6 matches (5+1) = jackpot
        prizes = {
            6: jackpot,  # All 5 picks in the 6 drawn (5+1)
            5: 2000,     # 5 matches
            4: 30,
            3: 6
        }
        return prizes.get(matches, 0)

    return calculator


def create_joker_prize_calculator(jackpot: float = 2_000_000) -> Callable:
    """
    Create prize calculator for Joker.

    5 main numbers + 1 joker number.
    """
    def calculator(picks: list[int], actual: list[int],
                  joker_pick: int = 0, joker_actual: int = 0) -> float:
        main_matches = len(set(picks[:5]) & set(actual[:5]))
        joker_match = (joker_pick == joker_actual) if joker_pick and joker_actual else False

        # Simplified prize structure
        if main_matches == 5 and joker_match:
            return jackpot
        elif main_matches == 5:
            return 50000
        elif main_matches == 4 and joker_match:
            return 5000
        elif main_matches == 4:
            return 100
        elif main_matches == 3 and joker_match:
            return 50
        elif main_matches == 3:
            return 14
        elif main_matches == 2 and joker_match:
            return 14
        elif main_matches == 1 and joker_match:
            return 8

        return 0

    return calculator
