"""Enhanced backtesting framework with prize tiers and statistics."""

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Any


@dataclass
class PrizeTier:
    """Configuration for a prize tier."""

    name: str
    matches_required: int
    bonus_required: bool = False
    payout: float = 0.0  # Multiplier of ticket price


@dataclass
class BacktestResult:
    """Comprehensive results from backtesting a strategy."""

    strategy_name: str
    total_draws: int
    tickets_per_draw: int = 1
    wins_by_tier: dict[str, int] = field(default_factory=dict)
    expected_value: float = 0.0
    win_rate: float = 0.0
    max_drawdown: int = 0  # Longest losing streak
    recent_performance: float = 0.0  # Performance in last N draws
    confidence_interval: tuple[float, float] = (0.0, 0.0)  # Wilson score CI
    total_tickets: int = 0
    total_wins: int = 0

    @property
    def roi(self) -> float:
        """Return on investment as a percentage."""
        if self.total_tickets == 0:
            return 0.0
        return (self.expected_value - self.total_tickets) / self.total_tickets * 100


def wilson_score_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion.

    More accurate than normal approximation for small samples or extreme proportions.
    """
    if trials == 0:
        return (0.0, 0.0)

    p_hat = successes / trials
    z = 1.96  # 95% confidence

    denominator = 1 + z ** 2 / trials
    center = (p_hat + z ** 2 / (2 * trials)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * trials)) / trials) / denominator

    return (max(0.0, center - spread), min(1.0, center + spread))


def compute_max_drawdown(win_history: list[bool]) -> int:
    """Compute the longest consecutive losing streak."""
    max_streak = 0
    current_streak = 0

    for won in win_history:
        if won:
            current_streak = 0
        else:
            current_streak += 1
            max_streak = max(max_streak, current_streak)

    return max_streak


class Strategy(Protocol):
    """Protocol for strategy objects."""

    name: str

    def generate(
        self,
        draws: list[tuple[list[int], int]],
        count: int,
        rng: random.Random,
    ) -> list[tuple[list[int], int]]:
        ...


class Backtester:
    """Enhanced backtester with prize tier support."""

    def __init__(
        self,
        number_pool: int,
        numbers_to_pick: int,
        prize_tiers: list[PrizeTier] | None = None,
    ):
        self.number_pool = number_pool
        self.numbers_to_pick = numbers_to_pick
        self.prize_tiers = prize_tiers or self._default_prize_tiers()

    def _default_prize_tiers(self) -> list[PrizeTier]:
        """Default prize tiers based on numbers_to_pick."""
        tiers = []
        for matches in range(3, self.numbers_to_pick + 1):
            # Rough payout multipliers (actual values vary by game)
            payout = 10 ** (matches - 2)  # 10, 100, 1000, etc.
            tier = PrizeTier(
                name=f"{matches}_match",
                matches_required=matches,
                bonus_required=False,
                payout=payout,
            )
            tiers.append(tier)

        # Jackpot tier (all numbers + bonus)
        tiers.append(
            PrizeTier(
                name="jackpot",
                matches_required=self.numbers_to_pick,
                bonus_required=True,
                payout=1_000_000.0,
            )
        )
        return tiers

    def evaluate_ticket(
        self,
        ticket: tuple[list[int], int],
        winning_numbers: list[int],
        winning_bonus: int,
    ) -> dict[str, bool]:
        """Evaluate a ticket against winning numbers.

        Returns dict mapping prize tier names to whether they were won.
        """
        main_numbers, bonus = ticket
        matches = len(set(main_numbers) & set(winning_numbers))
        bonus_match = bonus == winning_bonus

        results = {}
        for tier in self.prize_tiers:
            if matches >= tier.matches_required:
                if tier.bonus_required:
                    results[tier.name] = bonus_match
                else:
                    results[tier.name] = True
            else:
                results[tier.name] = False

        return results

    def backtest(
        self,
        strategy: Strategy,
        draws: list[tuple[list[int], int]],
        tickets_per_draw: int = 1,
        train_window: int = 50,
        recent_window: int = 20,
        rng: random.Random | None = None,
    ) -> BacktestResult:
        """Run backtest on historical draws.

        Args:
            strategy: Strategy to test
            draws: Historical draws (oldest first)
            tickets_per_draw: Number of tickets to generate per draw
            train_window: Minimum draws needed before generating tickets
            recent_window: Number of recent draws for performance calculation
            rng: Random number generator

        Returns:
            BacktestResult with comprehensive statistics
        """
        rng = rng or random.Random()

        wins_by_tier: dict[str, int] = {tier.name: 0 for tier in self.prize_tiers}
        total_payout = 0.0
        win_history: list[bool] = []
        recent_wins = 0

        total_tickets = 0
        total_wins = 0

        for i in range(train_window, len(draws)):
            # Use draws before this one for training
            training_draws = draws[:i]
            actual_draw = draws[i]
            actual_main, actual_bonus = actual_draw

            # Generate tickets
            tickets = strategy.generate(training_draws, tickets_per_draw, rng)
            total_tickets += len(tickets)

            draw_won = False
            for ticket in tickets:
                results = self.evaluate_ticket(ticket, actual_main, actual_bonus)

                for tier_name, won in results.items():
                    if won:
                        wins_by_tier[tier_name] += 1
                        tier = next(t for t in self.prize_tiers if t.name == tier_name)
                        total_payout += tier.payout
                        total_wins += 1
                        draw_won = True

            win_history.append(draw_won)

            # Track recent performance
            if i >= len(draws) - recent_window:
                if draw_won:
                    recent_wins += 1

        # Calculate statistics
        total_draws = len(draws) - train_window
        win_rate = total_wins / total_tickets if total_tickets > 0 else 0.0
        max_drawdown = compute_max_drawdown(win_history)
        confidence_interval = wilson_score_interval(total_wins, total_tickets)
        recent_performance = recent_wins / min(recent_window, total_draws) if total_draws > 0 else 0.0

        return BacktestResult(
            strategy_name=strategy.name,
            total_draws=total_draws,
            tickets_per_draw=tickets_per_draw,
            wins_by_tier=wins_by_tier,
            expected_value=total_payout,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            recent_performance=recent_performance,
            confidence_interval=confidence_interval,
            total_tickets=total_tickets,
            total_wins=total_wins,
        )


class CrossValidator:
    """Rolling window cross-validation for strategy evaluation."""

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 50,
    ):
        self.n_splits = n_splits
        self.min_train_size = min_train_size

    def split(
        self, draws: list[tuple[list[int], int]]
    ) -> list[tuple[list[tuple[list[int], int]], list[tuple[list[int], int]]]]:
        """Generate train/test splits using rolling window.

        Returns list of (train_draws, test_draws) tuples.
        """
        n = len(draws)
        if n < self.min_train_size + self.n_splits:
            return []

        test_size = (n - self.min_train_size) // self.n_splits
        splits = []

        for i in range(self.n_splits):
            train_end = self.min_train_size + i * test_size
            test_end = train_end + test_size

            if test_end > n:
                test_end = n

            train = draws[:train_end]
            test = draws[train_end:test_end]

            if train and test:
                splits.append((train, test))

        return splits

    def evaluate(
        self,
        strategy: Strategy,
        draws: list[tuple[list[int], int]],
        backtester: Backtester,
        tickets_per_draw: int = 1,
        rng: random.Random | None = None,
    ) -> dict[str, Any]:
        """Evaluate strategy using cross-validation.

        Returns dict with aggregate statistics across all folds.
        """
        rng = rng or random.Random()
        splits = self.split(draws)

        if not splits:
            return {"error": "Insufficient data for cross-validation"}

        fold_results: list[BacktestResult] = []

        for train_draws, test_draws in splits:
            # Create a combined set for backtesting
            combined = train_draws + test_draws
            result = backtester.backtest(
                strategy=strategy,
                draws=combined,
                tickets_per_draw=tickets_per_draw,
                train_window=len(train_draws),
                rng=rng,
            )
            fold_results.append(result)

        # Aggregate results
        total_wins = sum(r.total_wins for r in fold_results)
        total_tickets = sum(r.total_tickets for r in fold_results)
        avg_win_rate = total_wins / total_tickets if total_tickets > 0 else 0.0
        win_rates = [r.win_rate for r in fold_results]
        max_drawdowns = [r.max_drawdown for r in fold_results]

        return {
            "n_folds": len(fold_results),
            "avg_win_rate": avg_win_rate,
            "win_rate_std": self._std_dev(win_rates),
            "avg_max_drawdown": sum(max_drawdowns) / len(max_drawdowns) if max_drawdowns else 0,
            "total_tickets": total_tickets,
            "total_wins": total_wins,
            "fold_results": fold_results,
        }

    def _std_dev(self, values: list[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
