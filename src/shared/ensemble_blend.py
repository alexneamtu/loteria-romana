"""Ensemble blending strategy for lottery number generation.

Instead of picking a single best strategy (winner-takes-all), this
module allocates picks proportionally across multiple strategies
weighted by their backtest performance.  A chi-square bias detector
dynamically adjusts how aggressively non-random strategies are used.
"""

import math
import random

from .bayesian import BayesianScorer
from .bias_detection import chi_square_uniformity_test
from .cooccurrence import CooccurrenceStrategy
from .game_config import GameConfig
from .game_strategies import (
    generate_frequency_picks,
    generate_random_picks,
    is_prize_winner,
    _sample_weighted,
)
from .genetic import GeneticStrategy
from .math_utils import softmax
from .recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE
from .drift_detection import adwin_detect_drift
from .runs_test import per_number_runs_analysis
from .regime import detect_regimes
from .portfolio import optimize_ticket_portfolio
from .backtest_base import correct_significance as _bh_correct_significance, BacktestResult as _BacktestResult

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


def _score_random(
    config: GameConfig,
    draws: list[list[int]],
    rng: random.Random,
) -> int:
    """Count prize-winning random picks across all draws."""
    wins = 0
    for draw in draws:
        pick = sorted(rng.sample(list(config.pool_range), config.numbers_to_pick))
        if is_prize_winner(config, pick, draw):
            wins += 1
    return wins


def _score_frequency(
    config: GameConfig,
    draws: list[list[int]],
    rng: random.Random,
    half_life: float,
    half_life_mode: str,
    draw_dates: list[str] | None,
) -> int:
    """Count prize-winning frequency picks using walk-forward testing."""
    wins = 0
    for idx in range(len(draws)):
        picks = generate_frequency_picks(
            config,
            draws[:idx],
            1,
            rng,
            half_life=half_life,
            draw_dates=draw_dates[:idx] if draw_dates else None,
            half_life_mode=half_life_mode,
        )
        if picks and is_prize_winner(config, picks[0], draws[idx]):
            wins += 1
    return wins


def _score_strategy_object(
    config: GameConfig,
    draws: list[list[int]],
    strategy,
    rng: random.Random,
    draw_dates: list[str] | None,
    half_life_mode: str,
) -> int:
    """Score a strategy that implements the Strategy protocol."""
    wins = 0
    min_train = 3
    for idx in range(min_train, len(draws)):
        picks = strategy.generate(
            draws[:idx],
            1,
            rng,
            draw_dates=draw_dates[:idx] if draw_dates else None,
            half_life_mode=half_life_mode,
        )
        if picks and is_prize_winner(config, picks[0], draws[idx]):
            wins += 1
    return wins


def _allocate_counts(weights: dict[str, float], total: int) -> dict[str, int]:
    """Allocate integer pick counts proportionally to weights.

    Uses largest-remainder method to ensure counts sum to total.
    """
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        names = list(weights)
        base = total // len(names)
        allocation = {n: base for n in names}
        for i in range(total - base * len(names)):
            allocation[names[i]] += 1
        return allocation

    raw = {name: w / weight_sum * total for name, w in weights.items()}
    floored = {name: int(v) for name, v in raw.items()}
    remainders = {name: raw[name] - floored[name] for name in raw}

    allocated = sum(floored.values())
    deficit = total - allocated

    for name in sorted(remainders, key=remainders.get, reverse=True):
        if deficit <= 0:
            break
        floored[name] += 1
        deficit -= 1

    return floored


def _compute_enhanced_bias(
    draws: list[list[int]],
    pool_size: int,
    numbers_drawn: int,
) -> dict[str, float]:
    """Compute enhanced bias signals from multiple detectors.

    Returns a dict of bias adjustments:
    - 'drift_factor': 0.0 to 1.0 (1.0 = strong drift detected)
    - 'runs_factor': 0.0 to 1.0 (1.0 = many non-random numbers)
    - 'regime_recency': fraction of draws in most recent regime
    """
    result = {"drift_factor": 0.0, "runs_factor": 0.0, "regime_recency": 1.0}

    if len(draws) < 30:
        return result

    # Drift detection on draw sums
    sums = [sum(d) for d in draws]
    drift = adwin_detect_drift(sums)
    if drift.drift_detected:
        result["drift_factor"] = min(drift.statistic / 3.0, 1.0)

    # Runs test: fraction of numbers with significant streaks
    runs_reports = per_number_runs_analysis(draws, pool_size)
    significant_count = sum(1 for r in runs_reports if r.significant)
    result["runs_factor"] = significant_count / pool_size

    # Regime detection: focus on most recent regime
    boundaries = detect_regimes(sums, min_segment_size=30)
    if boundaries:
        last_boundary = max(b.index for b in boundaries)
        recent_fraction = (len(draws) - last_boundary) / len(draws)
        result["regime_recency"] = recent_fraction

    return result


def _apply_significance_gate(
    scores: dict[str, int],
    baseline_score: int,
    total_draws: int,
    min_draws: int = 30,
) -> dict[str, int]:
    """Remove strategies that don't significantly outperform random.

    Uses a simple binomial proportion test. Strategies must have a win
    rate significantly above baseline to be included.

    Always keeps "random" as the baseline strategy.

    Args:
        scores: Strategy name -> win count from backtesting.
        baseline_score: Win count of the random strategy.
        total_draws: Number of draws in the scoring window.
        min_draws: Minimum draws needed for gating (skip if fewer).

    Returns:
        Filtered scores dict with only significant strategies and random.
    """
    if total_draws < min_draws:
        return dict(scores)

    baseline_rate = baseline_score / total_draws if total_draws > 0 else 0.0

    gated: dict[str, int] = {}
    for name, score in scores.items():
        if name == "random":
            gated[name] = score
            continue

        strategy_rate = score / total_draws if total_draws > 0 else 0.0
        if strategy_rate < baseline_rate:
            continue

        if strategy_rate == baseline_rate:
            gated[name] = score
            continue

        # Simple z-test for proportion difference
        se = math.sqrt(baseline_rate * (1 - baseline_rate) / total_draws) if baseline_rate > 0 and baseline_rate < 1 else 0.0
        if se > 0:
            z = (strategy_rate - baseline_rate) / se
            if z > 1.645:  # One-tailed p < 0.05
                gated[name] = score
        else:
            gated[name] = score

    if len(gated) < 2:
        return dict(scores)

    return gated


def _apply_corrected_significance_gate(
    scores: dict[str, int],
    baseline_score: int,
    total_draws: int,
    min_draws: int = 30,
    fdr_threshold: float = 0.10,
    min_effect_size: float = 0.01,
) -> dict[str, int]:
    """Apply BH-corrected significance gate with effect size filtering.

    Falls back to simple gate if fewer than 3 non-random strategies.
    """
    if total_draws < min_draws:
        return dict(scores)

    non_random = {k: v for k, v in scores.items() if k != "random"}
    if len(non_random) < 3:
        return _apply_significance_gate(scores, baseline_score, total_draws, min_draws)

    baseline_rate = baseline_score / total_draws if total_draws > 0 else 0.0

    bt_results = []
    for name, score in non_random.items():
        bt_results.append(_BacktestResult(
            strategy_name=name,
            total_draws=total_draws,
            total_tickets=total_draws,
            total_wins=score,
            win_rate=score / total_draws if total_draws > 0 else 0.0,
        ))

    corrected = _bh_correct_significance(
        bt_results, baseline_rate, fdr_threshold, min_effect_size,
    )

    gated = {"random": scores.get("random", baseline_score)}
    for cr in corrected:
        if cr.verdict == "included":
            gated[cr.strategy] = scores[cr.strategy]

    if len(gated) < 2:
        return dict(scores)

    return gated


def generate_blended_picks(
    config: GameConfig,
    draws: list[list[int]],
    count: int,
    rng: random.Random | None = None,
    half_life: float = DEFAULT_HALF_LIFE,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
    draw_dates: list[str] | None = None,
) -> list[list[int]]:
    """Generate picks by blending multiple strategies proportionally.

    Steps:
    1. Run chi-square bias detection to gauge data uniformity.
    2. Backtest each strategy on historical data.
    3. Apply softmax to convert scores into allocation weights.
    4. Adjust weights using bias strength (boost random if no bias).
    5. Allocate pick counts proportionally and generate.

    Args:
        config: Game configuration.
        draws: Historical draws (main numbers only).
        count: Number of picks to generate.
        rng: Random number generator.
        half_life: Recency half-life.
        half_life_mode: "draws" or "days".
        draw_dates: Optional ISO date strings.

    Returns:
        List of blended picks.
    """
    rng = rng or random.SystemRandom()

    if len(draws) < 5:
        return generate_random_picks(config, count, rng)

    bias = chi_square_uniformity_test(
        draws, config.pool_size, config.numbers_drawn,
    )

    enhanced = _compute_enhanced_bias(draws, config.pool_size, config.numbers_drawn)

    bayesian = BayesianScorer(
        config.pool_size,
        config.numbers_to_pick,
        half_life=half_life,
        half_life_mode=half_life_mode,
    )
    cooccurrence = CooccurrenceStrategy(
        config.pool_size,
        config.numbers_to_pick,
        half_life=half_life,
        half_life_mode=half_life_mode,
    )
    genetic_scoring = GeneticStrategy(
        config.pool_size,
        config.numbers_to_pick,
        population_size=20,
        generations=5,
    )
    genetic = GeneticStrategy(
        config.pool_size,
        config.numbers_to_pick,
        population_size=50,
        generations=20,
    )

    # Use regime-aware scoring window
    regime_window = int(len(draws) * enhanced["regime_recency"])
    scoring_window = max(min(regime_window, 100), 30)
    scoring_draws = draws[-scoring_window:] if len(draws) > scoring_window else draws
    scoring_dates = draw_dates[-scoring_window:] if draw_dates and len(draw_dates) > scoring_window else draw_dates

    scores = {
        "random": max(_score_random(config, scoring_draws, rng), 1),
        "frequency": max(
            _score_frequency(
                config, scoring_draws, rng, half_life, half_life_mode, scoring_dates,
            ),
            1,
        ),
        "bayesian": max(
            _score_strategy_object(
                config, scoring_draws, bayesian, rng, scoring_dates, half_life_mode,
            ),
            1,
        ),
        "cooccurrence": max(
            _score_strategy_object(
                config, scoring_draws, cooccurrence, rng, scoring_dates, half_life_mode,
            ),
            1,
        ),
        "genetic": max(
            _score_strategy_object(
                config, scoring_draws, genetic_scoring, rng, scoring_dates, half_life_mode,
            ),
            1,
        ),
    }

    if _GB_AVAILABLE:
        gb = GradientBoostStrategy(
            config.pool_size,
            config.numbers_to_pick,
            config.numbers_drawn,
        )
        scores["gradient_boost"] = max(
            _score_strategy_object(
                config, scoring_draws, gb, rng, scoring_dates, half_life_mode,
            ),
            1,
        )

    if _TORCH_AVAILABLE:
        lstm_scoring = LSTMStrategy(
            config.pool_size, config.numbers_to_pick, epochs=5, seq_len=10,
        )
        tcn_scoring = TCNStrategy(
            config.pool_size, config.numbers_to_pick, epochs=5, seq_len=10,
        )
        xfmr_scoring = TransformerStrategy(
            config.pool_size, config.numbers_to_pick, epochs=5, seq_len=10,
        )
        nf_scoring = NormalizingFlowStrategy(
            config.pool_size, config.numbers_to_pick, epochs=10,
        )
        rl_scoring = RLAgent(
            config.pool_size, config.numbers_to_pick, episodes=5, window=10,
        )

        for strat_name, strat in [
            ("lstm", lstm_scoring),
            ("tcn", tcn_scoring),
            ("transformer", xfmr_scoring),
            ("normalizing_flow", nf_scoring),
            ("rl", rl_scoring),
        ]:
            scores[strat_name] = max(
                _score_strategy_object(
                    config, scoring_draws, strat, rng, scoring_dates, half_life_mode,
                ),
                1,
            )

    # Apply significance gate
    baseline_score = scores.get("random", 1)
    scores = _apply_corrected_significance_gate(scores, baseline_score, len(scoring_draws))

    raw_weights = softmax([float(s) for s in scores.values()])
    weights = dict(zip(scores.keys(), raw_weights))

    if not bias.significant:
        random_boost = 0.3 * (1.0 - bias.bias_strength)
        weights["random"] += random_boost

    # Enhanced bias adjustments
    if enhanced["drift_factor"] > 0.5:
        for strat in ["frequency", "bayesian"]:
            if strat in weights:
                weights[strat] *= 1.0 + enhanced["drift_factor"] * 0.3

    if enhanced["runs_factor"] > 0.1:
        for strat in ["cooccurrence", "genetic"]:
            if strat in weights:
                weights[strat] *= 1.0 + enhanced["runs_factor"] * 0.5

    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    allocation = _allocate_counts(weights, count)

    lines: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    random_lines = generate_random_picks(config, allocation.get("random", 0) + count, rng)
    added_random = 0
    for pick in random_lines:
        if added_random >= allocation.get("random", 0):
            break
        key = tuple(pick)
        if key not in seen:
            seen.add(key)
            lines.append(pick)
            added_random += 1

    freq_lines = generate_frequency_picks(
        config, draws, allocation.get("frequency", 0) + count, rng,
        half_life=half_life, draw_dates=draw_dates, half_life_mode=half_life_mode,
    )
    added_freq = 0
    for pick in freq_lines:
        if added_freq >= allocation.get("frequency", 0):
            break
        key = tuple(pick)
        if key not in seen:
            seen.add(key)
            lines.append(pick)
            added_freq += 1

    bayes_lines = bayesian.generate(
        draws, allocation.get("bayesian", 0) + count, rng,
        draw_dates=draw_dates, half_life_mode=half_life_mode,
    )
    added_bayes = 0
    for pick in bayes_lines:
        if added_bayes >= allocation.get("bayesian", 0):
            break
        key = tuple(pick)
        if key not in seen:
            seen.add(key)
            lines.append(pick)
            added_bayes += 1

    cooc_lines = cooccurrence.generate(
        draws, allocation.get("cooccurrence", 0) + count, rng,
        draw_dates=draw_dates, half_life_mode=half_life_mode,
    )
    added_cooc = 0
    for pick in cooc_lines:
        if added_cooc >= allocation.get("cooccurrence", 0):
            break
        key = tuple(pick)
        if key not in seen:
            seen.add(key)
            lines.append(pick)
            added_cooc += 1

    genetic_lines = genetic.generate(
        draws, allocation.get("genetic", 0) + count, rng,
    )
    added_genetic = 0
    for pick in genetic_lines:
        if added_genetic >= allocation.get("genetic", 0):
            break
        key = tuple(pick)
        if key not in seen:
            seen.add(key)
            lines.append(pick)
            added_genetic += 1

    if _GB_AVAILABLE and "gradient_boost" in allocation:
        gb_gen = GradientBoostStrategy(
            config.pool_size,
            config.numbers_to_pick,
            config.numbers_drawn,
        )
        gb_lines = gb_gen.generate(
            draws, allocation.get("gradient_boost", 0) + count, rng,
        )
        added_gb = 0
        for pick in gb_lines:
            if added_gb >= allocation.get("gradient_boost", 0):
                break
            key = tuple(pick)
            if key not in seen:
                seen.add(key)
                lines.append(pick)
                added_gb += 1

    if _TORCH_AVAILABLE:
        lstm_gen = LSTMStrategy(config.pool_size, config.numbers_to_pick)
        tcn_gen = TCNStrategy(config.pool_size, config.numbers_to_pick)
        xfmr_gen = TransformerStrategy(config.pool_size, config.numbers_to_pick)
        nf_gen = NormalizingFlowStrategy(config.pool_size, config.numbers_to_pick)
        rl_gen = RLAgent(config.pool_size, config.numbers_to_pick)

        for strat_name, strat in [
            ("lstm", lstm_gen),
            ("tcn", tcn_gen),
            ("transformer", xfmr_gen),
            ("normalizing_flow", nf_gen),
            ("rl", rl_gen),
        ]:
            if strat_name in allocation:
                strat_lines = strat.generate(
                    draws, allocation.get(strat_name, 0) + count, rng,
                )
                added = 0
                for pick in strat_lines:
                    if added >= allocation.get(strat_name, 0):
                        break
                    key = tuple(pick)
                    if key not in seen:
                        seen.add(key)
                        lines.append(pick)
                        added += 1

    # Generate extra random candidates for portfolio selection
    target = count + max(count // 2, 3)
    while len(lines) < target:
        extra = generate_random_picks(config, 1, rng)
        key = tuple(extra[0])
        if key not in seen:
            seen.add(key)
            lines.append(extra[0])

    # Apply portfolio optimization for diversity
    if len(lines) > count:
        lines = optimize_ticket_portfolio(lines, count, config.pool_size)

    return lines[:count]
