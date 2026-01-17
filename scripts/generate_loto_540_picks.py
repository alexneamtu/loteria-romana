import argparse
import os
import random
from pathlib import Path

from loto_540_model.fetch import update_dataset
from loto_540_model.storage import load_draws
from loto_540_model.picks import generate_picks
from loto_540_model.seed import resolve_seed

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
from shared.game_config import LOTO_540_CONFIG
from shared.advanced_strategies import (
    generate_smart_picks,
    generate_optimal_picks,
    generate_coverage_picks,
    generate_pattern_picks,
)

# Loto 5/40 game parameters
NUMBER_POOL = 40
NUMBERS_TO_PICK = 5
SUPER_NOROC_MAX = 999999  # 6 digit number


def get_strategy_by_name(name: str):
    """Get strategy instance by name."""
    strategies = {
        "delta": DeltaStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SUPER_NOROC_MAX),
        "hotcold": HotColdStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SUPER_NOROC_MAX),
        "pairs": PairStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SUPER_NOROC_MAX),
        "skip": SkipGapStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SUPER_NOROC_MAX),
        "sum": SumConstraintStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SUPER_NOROC_MAX),
        "balance": BalanceStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SUPER_NOROC_MAX),
    }
    return strategies.get(name)


def get_all_strategies():
    """Get all available strategies."""
    return [
        DeltaStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SUPER_NOROC_MAX),
        HotColdStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SUPER_NOROC_MAX),
        PairStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SUPER_NOROC_MAX),
        SkipGapStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SUPER_NOROC_MAX),
        SumConstraintStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SUPER_NOROC_MAX),
        BalanceStrategy(NUMBER_POOL, NUMBERS_TO_PICK, SUPER_NOROC_MAX),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Loto 5/40 picks using various strategies"
    )
    parser.add_argument("--seed", type=int, help="Set deterministic RNG seed")
    parser.add_argument("--no-super-noroc", action="store_true", help="Omit Super Noroc from picks")
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
        "--wheel-guarantee", type=int, default=4,
        help="Minimum match guarantee for wheeling (default: 4)"
    )
    args = parser.parse_args()

    url = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/540_si_super_noroc/rezultate_extrageri.html"
    cache_path = Path("data/raw/loto_540_results.html")
    csv_path = Path("data/clean/loto_540_draws.csv")

    update_dataset(url, cache_path, csv_path)
    draws = load_draws(csv_path)

    try:
        seed = resolve_seed(args.seed, os.getenv("LOTO_540_SEED"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    include_super_noroc = not args.no_super_noroc
    draw_tuples = [(d.main_numbers, d.super_noroc) for d in draws]

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
        probs = ensemble.combine_probabilities(draw_tuples)

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
            if include_super_noroc:
                super_noroc = rng.randint(0, SUPER_NOROC_MAX)
                print(f"{idx}. {', '.join(str(n) for n in ticket)} + SN{super_noroc:06d}")
            else:
                print(f"{idx}. {', '.join(str(n) for n in ticket)}")
        return

    # Strategy mode
    if args.strategy == "smart":
        main_picks = generate_smart_picks(LOTO_540_CONFIG, draw_tuples, args.count, rng)
        lines = [(main, rng.randint(0, SUPER_NOROC_MAX) if include_super_noroc else None) for main in main_picks]
    elif args.strategy == "optimal":
        main_picks = generate_optimal_picks(LOTO_540_CONFIG, draw_tuples, args.count, rng)
        lines = [(main, rng.randint(0, SUPER_NOROC_MAX) if include_super_noroc else None) for main in main_picks]
    elif args.strategy == "coverage":
        main_picks = generate_coverage_picks(LOTO_540_CONFIG, draw_tuples, args.count, rng)
        lines = [(main, rng.randint(0, SUPER_NOROC_MAX) if include_super_noroc else None) for main in main_picks]
    elif args.strategy == "pattern":
        main_picks = generate_pattern_picks(LOTO_540_CONFIG, draw_tuples, args.count, rng)
        lines = [(main, rng.randint(0, SUPER_NOROC_MAX) if include_super_noroc else None) for main in main_picks]
    elif args.strategy == "auto":
        lines = generate_picks(draw_tuples, count=args.count, rng=rng, include_super_noroc=include_super_noroc)
    elif args.strategy == "ensemble":
        ensemble = EnsembleVoter(
            get_all_strategies(),
            number_pool=NUMBER_POOL,
            numbers_to_pick=NUMBERS_TO_PICK,
        )
        lines = ensemble.generate(draw_tuples, count=args.count, rng=rng)
        # Add super noroc if needed
        if include_super_noroc:
            lines = [(main, rng.randint(0, SUPER_NOROC_MAX)) for main, _ in lines]
    else:
        strategy = get_strategy_by_name(args.strategy)
        if strategy:
            lines = strategy.generate(draw_tuples, count=args.count, rng=rng)
            # Add super noroc if needed
            if include_super_noroc:
                lines = [(main, rng.randint(0, SUPER_NOROC_MAX)) for main, _ in lines]
        else:
            lines = generate_picks(draw_tuples, count=args.count, rng=rng, include_super_noroc=include_super_noroc)

    for idx, (main, super_noroc) in enumerate(lines, 1):
        if include_super_noroc and super_noroc is not None:
            super_noroc_str = f"{super_noroc:06d}"
            print(f"{idx}. {', '.join(str(n) for n in main)} + SN{super_noroc_str}")
        else:
            print(f"{idx}. {', '.join(str(n) for n in main)}")


if __name__ == "__main__":
    main()
