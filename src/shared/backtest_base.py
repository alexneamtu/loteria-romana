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

    @property
    def ev_per_ticket(self) -> float:
        """Expected value per ticket (total payout / total tickets)."""
        if self.total_tickets == 0:
            return 0.0
        return self.expected_value / self.total_tickets


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
        ticket: tuple[list[int], int] | list[int],
        winning_numbers: list[int],
        winning_bonus: int,
    ) -> dict[str, bool]:
        """Evaluate a ticket against winning numbers.

        Returns dict mapping prize tier names to whether they were won.
        Ticket can be either a tuple (main_numbers, bonus) or just main_numbers list.
        """
        if isinstance(ticket, tuple) and len(ticket) == 2 and isinstance(ticket[1], int):
            main_numbers, bonus = ticket
        else:
            main_numbers = ticket
            bonus = None
        matches = len(set(main_numbers) & set(winning_numbers))
        bonus_match = bonus is not None and bonus == winning_bonus

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

            # Extract just main numbers for strategy training
            training_main = [draw[0] for draw in training_draws]
            # Generate tickets
            tickets = strategy.generate(training_main, tickets_per_draw, rng)
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


def strategy_significance_test(
    strategy_result: BacktestResult,
    baseline_win_rate: float,
) -> dict:
    """Test if a strategy is significantly different from baseline.

    Uses binomial test approximation to determine if the observed
    win rate is significantly different from the expected baseline.

    Args:
        strategy_result: Result from backtesting
        baseline_win_rate: Expected win rate under random selection

    Returns:
        Dict with test statistics and conclusion
    """
    n = strategy_result.total_tickets
    k = strategy_result.total_wins
    p0 = baseline_win_rate

    if n == 0:
        return {"error": "No tickets to analyze"}

    # Observed proportion
    p_hat = k / n

    # Standard error under null hypothesis
    se = math.sqrt(p0 * (1 - p0) / n)

    # Z-statistic
    if se > 0:
        z = (p_hat - p0) / se
    else:
        z = 0.0

    # Two-tailed p-value (approximate using normal)
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    return {
        "observed_rate": p_hat,
        "baseline_rate": p0,
        "z_statistic": z,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "conclusion": (
            "Strategy shows significant difference from baseline"
            if p_value < 0.05
            else "No significant difference from random selection"
        ),
    }


def passes_significance_gate(
    result: BacktestResult,
    baseline_win_rate: float,
    alpha: float = 0.05,
) -> bool:
    """Check if a strategy significantly outperforms the baseline.

    A strategy passes the gate only if:
    1. Its win rate exceeds the baseline
    2. The difference is statistically significant at the given alpha level

    Args:
        result: Backtest result to evaluate.
        baseline_win_rate: Expected win rate under random selection.
        alpha: Significance level (default 0.05).

    Returns:
        True if the strategy significantly outperforms baseline.
    """
    if result.total_tickets == 0:
        return False

    if result.win_rate <= baseline_win_rate:
        return False

    test = strategy_significance_test(result, baseline_win_rate)
    return test.get("p_value", 1.0) < alpha and result.win_rate > baseline_win_rate


def _normal_cdf(z: float) -> float:
    """Approximate standard normal CDF."""
    return 0.5 * (1 + _erf(z / math.sqrt(2)))


def _erf(x: float) -> float:
    """Approximate error function."""
    sign = 1 if x >= 0 else -1
    x = abs(x)

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return sign * y


def compare_strategies(results: list[BacktestResult]) -> dict:
    """Compare multiple strategies statistically.

    Tests whether any strategy is significantly better than others.

    Args:
        results: List of backtest results for different strategies

    Returns:
        Dict with comparison statistics
    """
    if len(results) < 2:
        return {"error": "Need at least 2 strategies to compare"}

    # Calculate win rates
    win_rates = [(r.strategy_name, r.win_rate) for r in results]
    sorted_rates = sorted(win_rates, key=lambda x: x[1], reverse=True)

    # Best vs worst
    best_name, best_rate = sorted_rates[0]
    worst_name, worst_rate = sorted_rates[-1]

    # Chi-square test for homogeneity
    total_tickets = sum(r.total_tickets for r in results)
    total_wins = sum(r.total_wins for r in results)

    if total_tickets == 0:
        return {"error": "No tickets to analyze"}

    pooled_rate = total_wins / total_tickets

    chi_sq = 0.0
    for r in results:
        expected_wins = r.total_tickets * pooled_rate
        expected_losses = r.total_tickets * (1 - pooled_rate)
        actual_losses = r.total_tickets - r.total_wins

        if expected_wins > 0:
            chi_sq += (r.total_wins - expected_wins) ** 2 / expected_wins
        if expected_losses > 0:
            chi_sq += (actual_losses - expected_losses) ** 2 / expected_losses

    df = len(results) - 1
    p_value = _chi_square_p_value(chi_sq, df)

    return {
        "ranking": sorted_rates,
        "best_strategy": best_name,
        "worst_strategy": worst_name,
        "rate_difference": best_rate - worst_rate,
        "chi_square": chi_sq,
        "degrees_of_freedom": df,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "conclusion": (
            f"Significant difference found (p={p_value:.4f})"
            if p_value < 0.05
            else "No significant difference between strategies (as expected for random lottery)"
        ),
    }


def _chi_square_p_value(chi_sq: float, df: int) -> float:
    """Approximate p-value for chi-square distribution."""
    if df <= 0 or chi_sq <= 0:
        return 1.0

    # Wilson-Hilferty approximation
    z = ((chi_sq / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return max(0.0, min(1.0, 1 - _normal_cdf(z)))


def analyze_drawdown_distribution(win_history: list[bool]) -> dict:
    """Analyze the distribution of losing streaks.

    Args:
        win_history: Boolean list of win/loss outcomes

    Returns:
        Dict with drawdown statistics
    """
    if not win_history:
        return {"error": "No history to analyze"}

    # Find all losing streaks
    streaks = []
    current_streak = 0

    for won in win_history:
        if won:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
        else:
            current_streak += 1

    # Don't forget the last streak if it's ongoing
    if current_streak > 0:
        streaks.append(current_streak)

    if not streaks:
        return {
            "total_streaks": 0,
            "max_streak": 0,
            "mean_streak": 0,
            "conclusion": "No losing streaks found (all wins)",
        }

    return {
        "total_streaks": len(streaks),
        "max_streak": max(streaks),
        "min_streak": min(streaks),
        "mean_streak": sum(streaks) / len(streaks),
        "median_streak": sorted(streaks)[len(streaks) // 2],
        "streak_distribution": {
            "1-5": sum(1 for s in streaks if 1 <= s <= 5),
            "6-10": sum(1 for s in streaks if 6 <= s <= 10),
            "11-25": sum(1 for s in streaks if 11 <= s <= 25),
            "26-50": sum(1 for s in streaks if 26 <= s <= 50),
            "51+": sum(1 for s in streaks if s > 50),
        },
    }


def calculate_expected_drawdown(win_rate: float, trials: int) -> dict:
    """Calculate expected maximum drawdown for a given win rate.

    Uses analytical approximation for expected longest run of failures.

    Args:
        win_rate: Probability of winning each trial
        trials: Total number of trials

    Returns:
        Dict with expected drawdown statistics
    """
    if win_rate <= 0 or win_rate >= 1:
        return {"error": "Win rate must be between 0 and 1"}

    loss_rate = 1 - win_rate

    # Expected longest run approximation: log_q(n) where q = loss_rate
    # E[longest run] ≈ log(n) / log(1/q) - 0.5
    if loss_rate > 0 and loss_rate < 1:
        expected_max = math.log(trials) / math.log(1 / loss_rate) - 0.5
    else:
        expected_max = trials

    # Standard deviation approximation
    std_dev = expected_max * 0.3  # Rough approximation

    return {
        "win_rate": win_rate,
        "trials": trials,
        "expected_max_drawdown": expected_max,
        "std_dev": std_dev,
        "95_percentile": expected_max + 1.65 * std_dev,
        "99_percentile": expected_max + 2.33 * std_dev,
    }


def generate_backtest_report(results: list[BacktestResult]) -> str:
    """Generate a formatted backtest report.

    Args:
        results: List of backtest results

    Returns:
        Formatted string report
    """
    if not results:
        return "No results to report."

    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST RESULTS REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Summary table
    lines.append(f"{'Strategy':<15} {'Win Rate':>10} {'CI Low':>8} {'CI High':>8} {'Drawdown':>10}")
    lines.append("-" * 60)

    for r in sorted(results, key=lambda x: x.win_rate, reverse=True):
        lines.append(
            f"{r.strategy_name:<15} {r.win_rate:>10.2%} "
            f"{r.confidence_interval[0]:>8.2%} {r.confidence_interval[1]:>8.2%} "
            f"{r.max_drawdown:>10}"
        )

    lines.append("")

    # Statistical comparison
    comparison = compare_strategies(results)
    lines.append("STATISTICAL ANALYSIS")
    lines.append("-" * 60)
    lines.append(f"Chi-square statistic: {comparison.get('chi_square', 0):.2f}")
    lines.append(f"P-value: {comparison.get('p_value', 1):.4f}")
    lines.append(f"Conclusion: {comparison.get('conclusion', 'N/A')}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("Note: For truly random lotteries, no strategy should")
    lines.append("significantly outperform random selection.")
    lines.append("=" * 60)

    return "\n".join(lines)
