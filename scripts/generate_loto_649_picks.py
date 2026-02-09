import argparse
import os
import random
from pathlib import Path

from loto_649_model.fetch import update_dataset
from loto_649_model.storage import load_draws
from loto_649_model.picks import generate_picks
from loto_649_model.seed import resolve_seed

from shared.stats import (
    DeltaStrategy,
    HotColdStrategy,
    PairStrategy,
    SkipGapStrategy,
    SumConstraintStrategy,
    BalanceStrategy,
)
from shared.ensemble import EnsembleVoter
from shared.wheeling import WheelGenerator, verify_wheel_coverage
from shared.game_config import LOTO_649_CONFIG
from shared.advanced_strategies import (
    generate_smart_picks,
    generate_optimal_picks,
    generate_coverage_picks,
    generate_pattern_picks,
)
from shared.ensemble_blend import generate_blended_picks
from shared.portfolio import optimize_ticket_portfolio
from shared.recency import resolve_recency_settings
from shared.bayesian import BayesianScorer
from shared.cooccurrence import CooccurrenceStrategy
from shared.genetic import GeneticStrategy

try:
    from shared.gradient_boost import GradientBoostStrategy
except ImportError:
    GradientBoostStrategy = None

try:
    from shared.lstm_strategy import LSTMStrategy
    from shared.tcn_strategy import TCNStrategy
    from shared.transformer_strategy import TransformerStrategy
    from shared.normalizing_flows import NormalizingFlowStrategy
    from shared.rl_agent import RLAgent
except ImportError:
    LSTMStrategy = TCNStrategy = TransformerStrategy = None
    NormalizingFlowStrategy = RLAgent = None

# Loto 6/49 game parameters
NUMBER_POOL = 49
NUMBERS_TO_PICK = 6


def get_strategy_by_name(name: str, half_life: float, half_life_mode: str):
    """Get strategy instance by name."""
    strategies = {
        "delta": DeltaStrategy(
            NUMBER_POOL,
            NUMBERS_TO_PICK,
            half_life=half_life,
            half_life_mode=half_life_mode,
        ),
        "hotcold": HotColdStrategy(
            NUMBER_POOL,
            NUMBERS_TO_PICK,
            half_life=half_life,
            half_life_mode=half_life_mode,
        ),
        "pairs": PairStrategy(
            NUMBER_POOL,
            NUMBERS_TO_PICK,
            half_life=half_life,
            half_life_mode=half_life_mode,
        ),
        "skip": SkipGapStrategy(
            NUMBER_POOL,
            NUMBERS_TO_PICK,
            half_life=half_life,
            half_life_mode=half_life_mode,
        ),
        "sum": SumConstraintStrategy(
            NUMBER_POOL,
            NUMBERS_TO_PICK,
            half_life=half_life,
            half_life_mode=half_life_mode,
        ),
        "balance": BalanceStrategy(
            NUMBER_POOL,
            NUMBERS_TO_PICK,
            half_life=half_life,
            half_life_mode=half_life_mode,
        ),
    }
    return strategies.get(name)


def get_all_strategies(half_life: float, half_life_mode: str):
    """Get all available strategies."""
    return [
        DeltaStrategy(
            NUMBER_POOL,
            NUMBERS_TO_PICK,
            half_life=half_life,
            half_life_mode=half_life_mode,
        ),
        HotColdStrategy(
            NUMBER_POOL,
            NUMBERS_TO_PICK,
            half_life=half_life,
            half_life_mode=half_life_mode,
        ),
        PairStrategy(
            NUMBER_POOL,
            NUMBERS_TO_PICK,
            half_life=half_life,
            half_life_mode=half_life_mode,
        ),
        SkipGapStrategy(
            NUMBER_POOL,
            NUMBERS_TO_PICK,
            half_life=half_life,
            half_life_mode=half_life_mode,
        ),
        SumConstraintStrategy(
            NUMBER_POOL,
            NUMBERS_TO_PICK,
            half_life=half_life,
            half_life_mode=half_life_mode,
        ),
        BalanceStrategy(
            NUMBER_POOL,
            NUMBERS_TO_PICK,
            half_life=half_life,
            half_life_mode=half_life_mode,
        ),
    ]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate Loto 6/49 picks using various strategies"
    )
    parser.add_argument("--seed", type=int, help="Set deterministic RNG seed")
    parser.add_argument(
        "-n", "--count", type=int, default=2, help="Number of lines to generate"
    )
    parser.add_argument(
        "-s", "--strategy",
        choices=["auto", "blend", "bayesian", "cooccurrence", "genetic", "gradient_boost",
                 "lstm", "tcn", "transformer", "normalizing_flow", "rl",
                 "smart", "optimal", "coverage", "pattern",
                 "delta", "hotcold", "pairs", "skip", "sum", "balance", "ensemble"],
        default="blend",
        help="Strategy to use for number selection (blend is recommended)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed strategy information"
    )
    parser.add_argument(
        "--half-life", type=float, help="Recency half-life in draws (or days)"
    )
    parser.add_argument(
        "--half-life-mode",
        choices=["draws", "days"],
        help="Recency half-life mode (default: draws)",
    )
    parser.add_argument(
        "--wheel", type=int, metavar="N",
        help="Generate wheeling system with N numbers (e.g., --wheel 12)"
    )
    parser.add_argument(
        "--wheel-guarantee", type=int, default=4,
        help="Minimum match guarantee for wheeling (default: 4)"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    url = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/649_si_noroc/rezultate_extragere.html"
    cache_path = Path("data/raw/loto_649_results.html")
    csv_path = Path("data/clean/loto_649_draws.csv")

    update_dataset(url, cache_path, csv_path)
    draws = load_draws(csv_path)

    try:
        seed = resolve_seed(args.seed, os.getenv("LOTO_649_SEED"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        half_life, half_life_mode = resolve_recency_settings(
            args.half_life,
            os.getenv("RECENCY_HALF_LIFE"),
            args.half_life_mode,
            os.getenv("RECENCY_HALF_LIFE_MODE"),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    draw_tuples = [d.main_numbers for d in draws]
    draw_dates = [d.date for d in draws]

    if args.verbose:
        print(f"Loaded {len(draws)} historical draws")
        print(f"Using strategy: {args.strategy}")
        print()

    # Wheeling mode
    if args.wheel:
        if args.verbose:
            print(f"Generating wheel with {args.wheel} numbers, guarantee {args.wheel_guarantee}")

        # First get top numbers using ensemble
        ensemble = EnsembleVoter(
            get_all_strategies(half_life, half_life_mode),
            number_pool=NUMBER_POOL,
            numbers_to_pick=NUMBERS_TO_PICK,
        )
        probs = ensemble.combine_probabilities(
            draw_tuples,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )

        # Select top N numbers
        number_probs = [(i + 1, p) for i, p in enumerate(probs)]
        number_probs.sort(key=lambda x: x[1], reverse=True)
        wheel_numbers = [n for n, _ in number_probs[:args.wheel]]

        if args.verbose:
            print(f"Selected wheel numbers: {sorted(wheel_numbers)}")

        generator = WheelGenerator(
            numbers_per_line=NUMBERS_TO_PICK,
            guarantee=args.wheel_guarantee,
        )
        tickets = generator.generate_abbreviated_wheel(wheel_numbers)

        # Verify coverage
        is_valid, _ = verify_wheel_coverage(tickets, wheel_numbers, args.wheel_guarantee)

        print(f"\nWheel: {args.wheel_guarantee}-if-{args.wheel} coverage")
        print(f"Numbers: {sorted(wheel_numbers)}")
        print(f"Tickets: {len(tickets)} (vs {generator.estimate_reduction(args.wheel)['full_wheel_size']} full)")
        print(f"Coverage verified: {is_valid}")
        print()

        for idx, ticket in enumerate(tickets, 1):
            print(f"{idx}. {', '.join(str(n) for n in ticket)}")
        return

    # Strategy mode
    if args.strategy == "blend":
        lines = generate_blended_picks(
            LOTO_649_CONFIG,
            draw_tuples,
            args.count,
            rng,
            half_life=half_life,
            half_life_mode=half_life_mode,
            draw_dates=draw_dates,
        )
    elif args.strategy == "smart":
        lines = generate_smart_picks(
            LOTO_649_CONFIG,
            draw_tuples,
            args.count,
            rng,
            half_life=half_life,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
    elif args.strategy == "optimal":
        lines = generate_optimal_picks(
            LOTO_649_CONFIG,
            draw_tuples,
            args.count,
            rng,
            half_life=half_life,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
    elif args.strategy == "coverage":
        lines = generate_coverage_picks(
            LOTO_649_CONFIG,
            draw_tuples,
            args.count,
            rng,
            half_life=half_life,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
    elif args.strategy == "pattern":
        lines = generate_pattern_picks(
            LOTO_649_CONFIG,
            draw_tuples,
            args.count,
            rng,
            half_life=half_life,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
    elif args.strategy == "bayesian":
        strat = BayesianScorer(LOTO_649_CONFIG.pool_size, LOTO_649_CONFIG.numbers_to_pick,
                               half_life=half_life, half_life_mode=half_life_mode)
        candidates = strat.generate(draw_tuples, args.count * 3, rng,
                                    draw_dates=draw_dates, half_life_mode=half_life_mode)
        lines = optimize_ticket_portfolio(candidates, args.count, LOTO_649_CONFIG.pool_size)
    elif args.strategy == "cooccurrence":
        strat = CooccurrenceStrategy(LOTO_649_CONFIG.pool_size, LOTO_649_CONFIG.numbers_to_pick,
                                     half_life=half_life, half_life_mode=half_life_mode)
        candidates = strat.generate(draw_tuples, args.count * 3, rng,
                                    draw_dates=draw_dates, half_life_mode=half_life_mode)
        lines = optimize_ticket_portfolio(candidates, args.count, LOTO_649_CONFIG.pool_size)
    elif args.strategy == "genetic":
        strat = GeneticStrategy(LOTO_649_CONFIG.pool_size, LOTO_649_CONFIG.numbers_to_pick)
        candidates = strat.generate(draw_tuples, args.count * 3, rng)
        lines = optimize_ticket_portfolio(candidates, args.count, LOTO_649_CONFIG.pool_size)
    elif args.strategy == "gradient_boost" and GradientBoostStrategy:
        strat = GradientBoostStrategy(LOTO_649_CONFIG.pool_size, LOTO_649_CONFIG.numbers_to_pick,
                                      LOTO_649_CONFIG.numbers_drawn)
        candidates = strat.generate(draw_tuples, args.count * 3, rng)
        lines = optimize_ticket_portfolio(candidates, args.count, LOTO_649_CONFIG.pool_size)
    elif args.strategy == "lstm" and LSTMStrategy:
        strat = LSTMStrategy(LOTO_649_CONFIG.pool_size, LOTO_649_CONFIG.numbers_to_pick)
        candidates = strat.generate(draw_tuples, args.count * 3, rng)
        lines = optimize_ticket_portfolio(candidates, args.count, LOTO_649_CONFIG.pool_size)
    elif args.strategy == "tcn" and TCNStrategy:
        strat = TCNStrategy(LOTO_649_CONFIG.pool_size, LOTO_649_CONFIG.numbers_to_pick)
        candidates = strat.generate(draw_tuples, args.count * 3, rng)
        lines = optimize_ticket_portfolio(candidates, args.count, LOTO_649_CONFIG.pool_size)
    elif args.strategy == "transformer" and TransformerStrategy:
        strat = TransformerStrategy(LOTO_649_CONFIG.pool_size, LOTO_649_CONFIG.numbers_to_pick)
        candidates = strat.generate(draw_tuples, args.count * 3, rng)
        lines = optimize_ticket_portfolio(candidates, args.count, LOTO_649_CONFIG.pool_size)
    elif args.strategy == "normalizing_flow" and NormalizingFlowStrategy:
        strat = NormalizingFlowStrategy(LOTO_649_CONFIG.pool_size, LOTO_649_CONFIG.numbers_to_pick)
        candidates = strat.generate(draw_tuples, args.count * 3, rng)
        lines = optimize_ticket_portfolio(candidates, args.count, LOTO_649_CONFIG.pool_size)
    elif args.strategy == "rl" and RLAgent:
        strat = RLAgent(LOTO_649_CONFIG.pool_size, LOTO_649_CONFIG.numbers_to_pick)
        candidates = strat.generate(draw_tuples, args.count * 3, rng)
        lines = optimize_ticket_portfolio(candidates, args.count, LOTO_649_CONFIG.pool_size)
    elif args.strategy == "auto":
        lines = generate_picks(
            draw_tuples,
            count=args.count,
            rng=rng,
            half_life=half_life,
            half_life_mode=half_life_mode,
            draw_dates=draw_dates,
        )
    elif args.strategy == "ensemble":
        ensemble = EnsembleVoter(
            get_all_strategies(half_life, half_life_mode),
            number_pool=NUMBER_POOL,
            numbers_to_pick=NUMBERS_TO_PICK,
        )
        lines = ensemble.generate(
            draw_tuples,
            count=args.count,
            rng=rng,
            draw_dates=draw_dates,
            half_life_mode=half_life_mode,
        )
    else:
        strategy = get_strategy_by_name(args.strategy, half_life, half_life_mode)
        if strategy:
            lines = strategy.generate(
                draw_tuples,
                count=args.count,
                rng=rng,
                draw_dates=draw_dates,
                half_life_mode=half_life_mode,
            )
        else:
            lines = generate_picks(
                draw_tuples,
                count=args.count,
                rng=rng,
                half_life=half_life,
                half_life_mode=half_life_mode,
                draw_dates=draw_dates,
            )

    for idx, main in enumerate(lines, 1):
        print(f"{idx}. {', '.join(str(n) for n in main)}")


if __name__ == "__main__":
    main()
