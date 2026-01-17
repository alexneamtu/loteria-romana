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
from shared.recency import resolve_half_life

# Loto 5/40 game parameters
NUMBER_POOL = 40
NUMBERS_TO_PICK = 5


def get_strategy_by_name(name: str, half_life: float):
    """Get strategy instance by name."""
    strategies = {
        "delta": DeltaStrategy(NUMBER_POOL, NUMBERS_TO_PICK, half_life=half_life),
        "hotcold": HotColdStrategy(NUMBER_POOL, NUMBERS_TO_PICK, half_life=half_life),
        "pairs": PairStrategy(NUMBER_POOL, NUMBERS_TO_PICK, half_life=half_life),
        "skip": SkipGapStrategy(NUMBER_POOL, NUMBERS_TO_PICK, half_life=half_life),
        "sum": SumConstraintStrategy(NUMBER_POOL, NUMBERS_TO_PICK, half_life=half_life),
        "balance": BalanceStrategy(NUMBER_POOL, NUMBERS_TO_PICK, half_life=half_life),
    }
    return strategies.get(name)


def get_all_strategies(half_life: float):
    """Get all available strategies."""
    return [
        DeltaStrategy(NUMBER_POOL, NUMBERS_TO_PICK, half_life=half_life),
        HotColdStrategy(NUMBER_POOL, NUMBERS_TO_PICK, half_life=half_life),
        PairStrategy(NUMBER_POOL, NUMBERS_TO_PICK, half_life=half_life),
        SkipGapStrategy(NUMBER_POOL, NUMBERS_TO_PICK, half_life=half_life),
        SumConstraintStrategy(NUMBER_POOL, NUMBERS_TO_PICK, half_life=half_life),
        BalanceStrategy(NUMBER_POOL, NUMBERS_TO_PICK, half_life=half_life),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Loto 5/40 picks using various strategies"
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
        "--half-life", type=float, help="Recency half-life in draws"
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
    try:
        half_life = resolve_half_life(args.half_life, os.getenv("RECENCY_HALF_LIFE"))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    draw_tuples = [d.main_numbers for d in draws]

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
            get_all_strategies(half_life),
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
            print(f"{idx}. {', '.join(str(n) for n in ticket)}")
        return

    # Strategy mode
    if args.strategy == "smart":
        lines = generate_smart_picks(LOTO_540_CONFIG, draw_tuples, args.count, rng, half_life=half_life)
    elif args.strategy == "optimal":
        lines = generate_optimal_picks(LOTO_540_CONFIG, draw_tuples, args.count, rng, half_life=half_life)
    elif args.strategy == "coverage":
        lines = generate_coverage_picks(LOTO_540_CONFIG, draw_tuples, args.count, rng, half_life=half_life)
    elif args.strategy == "pattern":
        lines = generate_pattern_picks(LOTO_540_CONFIG, draw_tuples, args.count, rng, half_life=half_life)
    elif args.strategy == "auto":
        lines = generate_picks(draw_tuples, count=args.count, rng=rng, half_life=half_life)
    elif args.strategy == "ensemble":
        ensemble = EnsembleVoter(
            get_all_strategies(half_life),
            number_pool=NUMBER_POOL,
            numbers_to_pick=NUMBERS_TO_PICK,
        )
        lines = ensemble.generate(draw_tuples, count=args.count, rng=rng)
    else:
        strategy = get_strategy_by_name(args.strategy, half_life)
        if strategy:
            lines = strategy.generate(draw_tuples, count=args.count, rng=rng)
        else:
            lines = generate_picks(draw_tuples, count=args.count, rng=rng)

    for idx, main in enumerate(lines, 1):
        print(f"{idx}. {', '.join(str(n) for n in main)}")


if __name__ == "__main__":
    main()
