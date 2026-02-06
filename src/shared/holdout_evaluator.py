"""Holdout evaluation for lottery strategies.

Evaluates each strategy on a reserved holdout set that was never
used during development or backtesting. Produces per-strategy
match statistics.
"""

import random
from dataclasses import dataclass, field
from collections import Counter

from .game_config import GameConfig
from .game_strategies import (
    generate_frequency_picks,
    generate_random_picks,
    is_prize_winner,
)
from .bayesian import BayesianScorer
from .cooccurrence import CooccurrenceStrategy
from .genetic import GeneticStrategy
from .recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE

try:
    from .gradient_boost import GradientBoostStrategy, SKLEARN_AVAILABLE as _GB_AVAILABLE
except ImportError:
    _GB_AVAILABLE = False

try:
    from .lstm_strategy import LSTMStrategy
    from .tcn_strategy import TCNStrategy
    from .transformer_strategy import TransformerStrategy
    from .normalizing_flows import NormalizingFlowStrategy
    from .rl_agent import RLAgent
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class HoldoutResult:
    """Result of evaluating a strategy on holdout data."""

    strategy_name: str
    game_name: str
    holdout_size: int
    wins: int = 0
    win_rate: float = 0.0
    matches_distribution: dict[int, int] = field(default_factory=dict)


def evaluate_strategy_on_holdout(
    config: GameConfig,
    strategy_name: str,
    train_draws: list[list[int]],
    holdout_draws: list[list[int]],
    rng: random.Random,
    half_life: float = DEFAULT_HALF_LIFE,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    draw_dates: list[str] | None = None,
) -> HoldoutResult:
    """Evaluate a single strategy on holdout draws."""
    wins = 0
    match_counts: list[int] = []

    for i, holdout_draw in enumerate(holdout_draws):
        available_draws = train_draws + holdout_draws[:i]
        pick = _generate_single_pick(
            config, strategy_name, available_draws, rng,
            half_life, half_life_mode, draw_dates,
        )
        if pick is None:
            match_counts.append(0)
            continue

        matches = len(set(pick) & set(holdout_draw))
        match_counts.append(matches)
        if is_prize_winner(config, pick, holdout_draw):
            wins += 1

    distribution = dict(Counter(match_counts))
    holdout_size = len(holdout_draws)
    win_rate = wins / holdout_size if holdout_size > 0 else 0.0

    return HoldoutResult(
        strategy_name=strategy_name,
        game_name=config.name.lower().replace(" ", "_").replace("/", ""),
        holdout_size=holdout_size,
        wins=wins,
        win_rate=win_rate,
        matches_distribution=distribution,
    )


def _generate_single_pick(
    config: GameConfig,
    strategy_name: str,
    draws: list[list[int]],
    rng: random.Random,
    half_life: float,
    half_life_mode: str,
    draw_dates: list[str] | None,
) -> list[int] | None:
    """Generate a single pick using the named strategy."""
    if strategy_name == "random":
        picks = generate_random_picks(config, 1, rng)
    elif strategy_name == "frequency":
        picks = generate_frequency_picks(
            config, draws, 1, rng,
            half_life=half_life, draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
    elif strategy_name == "bayesian":
        scorer = BayesianScorer(
            config.pool_size, config.numbers_to_pick,
            half_life=half_life, half_life_mode=half_life_mode,
        )
        picks = scorer.generate(
            draws, 1, rng,
            draw_dates=draw_dates, half_life_mode=half_life_mode,
        )
    elif strategy_name == "cooccurrence":
        strat = CooccurrenceStrategy(
            config.pool_size, config.numbers_to_pick,
            half_life=half_life, half_life_mode=half_life_mode,
        )
        picks = strat.generate(
            draws, 1, rng,
            draw_dates=draw_dates, half_life_mode=half_life_mode,
        )
    elif strategy_name == "genetic":
        strat = GeneticStrategy(
            config.pool_size, config.numbers_to_pick,
            population_size=20, generations=5,
        )
        picks = strat.generate(draws, 1, rng)
    elif strategy_name == "gradient_boost" and _GB_AVAILABLE:
        strat = GradientBoostStrategy(
            config.pool_size, config.numbers_to_pick, config.numbers_drawn,
        )
        picks = strat.generate(draws, 1, rng)
    elif strategy_name == "lstm" and _TORCH_AVAILABLE:
        strat = LSTMStrategy(config.pool_size, config.numbers_to_pick, epochs=5)
        picks = strat.generate(draws, 1, rng)
    elif strategy_name == "tcn" and _TORCH_AVAILABLE:
        strat = TCNStrategy(config.pool_size, config.numbers_to_pick, epochs=5)
        picks = strat.generate(draws, 1, rng)
    elif strategy_name == "transformer" and _TORCH_AVAILABLE:
        strat = TransformerStrategy(config.pool_size, config.numbers_to_pick, epochs=5)
        picks = strat.generate(draws, 1, rng)
    elif strategy_name == "normalizing_flow" and _TORCH_AVAILABLE:
        strat = NormalizingFlowStrategy(config.pool_size, config.numbers_to_pick, epochs=5)
        picks = strat.generate(draws, 1, rng)
    elif strategy_name == "rl" and _TORCH_AVAILABLE:
        strat = RLAgent(config.pool_size, config.numbers_to_pick, episodes=5)
        picks = strat.generate(draws, 1, rng)
    else:
        return None

    return picks[0] if picks else None


def evaluate_all_strategies(
    config: GameConfig,
    train_draws: list[list[int]],
    holdout_draws: list[list[int]],
    rng: random.Random,
    half_life: float = DEFAULT_HALF_LIFE,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    draw_dates: list[str] | None = None,
) -> list[HoldoutResult]:
    """Evaluate all available strategies on holdout data."""
    strategy_names = ["random", "frequency", "bayesian", "cooccurrence", "genetic"]

    if _GB_AVAILABLE:
        strategy_names.append("gradient_boost")

    if _TORCH_AVAILABLE:
        strategy_names.extend(["lstm", "tcn", "transformer", "normalizing_flow", "rl"])

    results = []
    for name in strategy_names:
        result = evaluate_strategy_on_holdout(
            config=config,
            strategy_name=name,
            train_draws=train_draws,
            holdout_draws=holdout_draws,
            rng=random.Random(rng.randint(0, 2**32 - 1)),
            half_life=half_life,
            half_life_mode=half_life_mode,
            draw_dates=draw_dates,
        )
        results.append(result)

    return results


def format_holdout_report(results: list[HoldoutResult]) -> str:
    """Format holdout evaluation results as a text report."""
    if not results:
        return "No results to report."

    lines = []
    lines.append("=" * 65)
    lines.append("HOLDOUT EVALUATION REPORT")
    lines.append("=" * 65)
    lines.append(f"Game: {results[0].game_name}")
    lines.append(f"Holdout size: {results[0].holdout_size} draws")
    lines.append("")

    lines.append(f"{'Strategy':<20} {'Wins':>6} {'Win Rate':>10} {'Matches':>25}")
    lines.append("-" * 65)

    for r in sorted(results, key=lambda x: x.win_rate, reverse=True):
        match_str = ", ".join(f"{k}:{v}" for k, v in sorted(r.matches_distribution.items()))
        lines.append(
            f"{r.strategy_name:<20} {r.wins:>6} {r.win_rate:>10.2%} {match_str:>25}"
        )

    lines.append("")
    lines.append("=" * 65)
    return "\n".join(lines)
