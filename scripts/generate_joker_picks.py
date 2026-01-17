import argparse
import os
import random
from pathlib import Path

from joker_model.fetch import update_dataset
from joker_model.storage import load_draws
from joker_model.picks import generate_picks
from joker_model.seed import resolve_seed

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
from shared.game_config import JOKER_CONFIG
from shared.advanced_strategies import (
    generate_smart_picks,
    generate_optimal_picks,
    generate_coverage_picks,
    generate_pattern_picks,
)

# Joker game parameters
NUMBER_POOL = 45
NUMBERS_TO_PICK = 5
SECONDARY_POOL = 20


def get_strategy_by_name(name: str):
    """Get strategy instance by name."""
    strategies = {
        "delta": DeltaStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SECONDARY_POOL),
        "hotcold": HotColdStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SECONDARY_POOL),
        "pairs": PairStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SECONDARY_POOL),
        "skip": SkipGapStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SECONDARY_POOL),
        "sum": SumConstraintStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SECONDARY_POOL),
        "balance": BalanceStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SECONDARY_POOL),
    }
    return strategies.get(name)


def get_all_strategies():
    """Get all available strategies."""
    return [
        DeltaStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SECONDARY_POOL),
        HotColdStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SECONDARY_POOL),
        PairStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SECONDARY_POOL),
        SkipGapStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SECONDARY_POOL),
        SumConstraintStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SECONDARY_POOL),
        BalanceStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SECONDARY_POOL),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Joker lottery picks using various strategies"
    )
    parser.add_argument("--seed", type=int, help="Set deterministic RNG seed")
    parser.add_argument(
        "-n", "--count", type=int, default=2, help="Number of lines to generate"
    )
    parser.add_argument(
        "-s", "--strategy",
        choices=["auto", "smart", "optimal", "coverage", "pattern", "delta", "hotcold", "pairs", "skip", "sum", "balance", "ensemble"],
        default="smart",
        help="Strategy to use for number selection (smart is recommended)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed strategy information"
    )
    parser.add_argument(
        "--wheel", type=int, metavar="N",
        help="Generate wheeling system with N numbers (e.g., --wheel 10)"
    )
    parser.add_argument(
        "--wheel-guarantee", type=int, default=3,
        help="Minimum match guarantee for wheeling (default: 3)"
    )
    args = parser.parse_args()

    url = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html"
    cache_path = Path("data/raw/joker_results.html")
    csv_path = Path("data/clean/joker_draws.csv")

    update_dataset(url, cache_path, csv_path)
    draws = load_draws(csv_path)

    try:
        seed = resolve_seed(args.seed, os.getenv("JOKER_SEED"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    draw_tuples = [(d.main_numbers, d.joker) for d in draws]
    draw_main_only = [d.main_numbers for d in draws]  # For shared strategies

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
            get_all_strategies(),
            number_pool=NUMBER_POOL,
            numbers_to_pick=NUMBERS_TO_PICK,
        )
        probs = ensemble.combine_probabilities(draw_main_only)

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
            joker = rng.randint(1, SECONDARY_POOL)
            print(f"{idx}. {', '.join(str(n) for n in ticket)} + J{joker}")
        return

    # Strategy mode
    if args.strategy == "smart":
        # Best strategy - combines all techniques
        main_picks = generate_smart_picks(JOKER_CONFIG, draw_main_only, args.count, rng)
        lines = [(main, rng.randint(1, SECONDARY_POOL)) for main in main_picks]
    elif args.strategy == "optimal":
        main_picks = generate_optimal_picks(JOKER_CONFIG, draw_main_only, args.count, rng)
        lines = [(main, rng.randint(1, SECONDARY_POOL)) for main in main_picks]
    elif args.strategy == "coverage":
        main_picks = generate_coverage_picks(JOKER_CONFIG, draw_main_only, args.count, rng)
        lines = [(main, rng.randint(1, SECONDARY_POOL)) for main in main_picks]
    elif args.strategy == "pattern":
        main_picks = generate_pattern_picks(JOKER_CONFIG, draw_main_only, args.count, rng)
        lines = [(main, rng.randint(1, SECONDARY_POOL)) for main in main_picks]
    elif args.strategy == "auto":
        lines = generate_picks(draw_tuples, count=args.count, rng=rng)
    elif args.strategy == "ensemble":
        ensemble = EnsembleVoter(
            get_all_strategies(),
            number_pool=NUMBER_POOL,
            numbers_to_pick=NUMBERS_TO_PICK,
        )
        main_picks = ensemble.generate(draw_main_only, count=args.count, rng=rng)
        lines = [(main, rng.randint(1, SECONDARY_POOL)) for main in main_picks]
    else:
        strategy = get_strategy_by_name(args.strategy)
        if strategy:
            main_picks = strategy.generate(draw_main_only, count=args.count, rng=rng)
            lines = [(main, rng.randint(1, SECONDARY_POOL)) for main in main_picks]
        else:
            lines = generate_picks(draw_tuples, count=args.count, rng=rng)

    for idx, (main, joker) in enumerate(lines, 1):
        print(f"{idx}. {', '.join(str(n) for n in main)} + J{joker}")


if __name__ == "__main__":
    main()
