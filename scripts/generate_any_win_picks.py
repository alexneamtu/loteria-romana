#!/usr/bin/env python3
"""
Generate Any-Win Optimized Picks for Romanian Lottery.

This script generates lottery picks optimized for maximizing the chance
of winning ANY prize (not just the jackpot).

Usage:
    PYTHONPATH=src python scripts/generate_any_win_picks.py [options]

Options:
    --game GAME         Game: loto_649, loto_540, joker (default: loto_649)
    --count N           Number of picks to generate (default: 5)
    --wheel N           Generate wheel with N numbers (optional)
    --budget AMOUNT     Budget for wheeling in RON (optional)
    --seed SEED         Random seed for reproducibility
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate Any-Win Optimized Picks")
    parser.add_argument("--game", choices=["loto_649", "loto_540", "joker"],
                       default="loto_649", help="Game to generate picks for")
    parser.add_argument("--count", type=int, default=5,
                       help="Number of picks to generate")
    parser.add_argument("--wheel", type=int, default=None,
                       help="Generate wheel with N numbers")
    parser.add_argument("--budget", type=float, default=None,
                       help="Budget for wheeling in RON")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Import data loaders
    data_dir = Path(__file__).parent.parent / "data" / "clean"

    def load_draws(game):
        if game == "loto_649":
            from loto_649_model.storage import load_draws as load_649
            draws_obj = load_649(data_dir / "loto_649_draws.csv")
            draws = [list(d.main_numbers) for d in draws_obj]
            return draws, None, 49, 6
        elif game == "loto_540":
            from loto_540_model.storage import load_draws as load_540
            draws_obj = load_540(data_dir / "loto_540_draws.csv")
            draws = [list(d.main_numbers) for d in draws_obj]
            return draws, None, 40, 5
        else:  # joker
            from joker_model.storage import load_draws as load_joker
            draws_obj = load_joker(data_dir / "joker_draws.csv")
            draws = [list(d.main_numbers) for d in draws_obj]
            return draws, None, 45, 5
    from shared.any_win_optimizer import AnyWinOptimizer, OptimizationConfig
    from shared.wheeling import create_any_win_wheel, suggest_wheel_size
    import random

    # Game configurations
    game_configs = {
        "loto_649": {"pool": 49, "draw": 6, "min_sum": 115, "max_sum": 185, "cost": 6.0},
        "loto_540": {"pool": 40, "draw": 5, "min_sum": 85, "max_sum": 135, "cost": 4.0},
        "joker": {"pool": 45, "draw": 5, "min_sum": 95, "max_sum": 145, "cost": 8.0},
    }

    cfg = game_configs[args.game]

    print(f"{'='*60}")
    print(f"ANY-WIN OPTIMIZED PICKS: {args.game.upper()}")
    print(f"{'='*60}")

    # Load historical data
    draws, dates, pool_size, draw_size = load_draws(args.game)

    # Handle Loto 5/40 (6 drawn, pick 5)
    if args.game == "loto_540":
        draws = [d[:5] for d in draws]

    print(f"Analyzed {len(draws)} historical draws\n")

    # Create optimizer
    config = OptimizationConfig(
        pool_size=cfg["pool"],
        draw_size=cfg["draw"],
        min_sum=cfg["min_sum"],
        max_sum=cfg["max_sum"]
    )
    optimizer = AnyWinOptimizer(cfg["pool"], cfg["draw"], config)
    analysis = optimizer.analyze_history(draws)

    # Set seed if provided
    rng = random.Random(args.seed) if args.seed else None

    # Generate picks
    print("OPTIMIZED PICKS:")
    print("-" * 40)

    picks = optimizer.generate_optimized_picks(args.count, rng=rng)

    for i, pick in enumerate(picks, 1):
        joker_str = ""
        if args.game == "joker":
            # Generate random joker number
            joker = random.randint(1, 20) if not rng else rng.randint(1, 20)
            joker_str = f" + J{joker}"

        print(f"{i}. {', '.join(map(str, pick.numbers))}{joker_str}")
        print(f"   Sum: {pick.sum_value} | Odd/Even: {pick.odd_count}/{cfg['draw']-pick.odd_count} | "
              f"Decades: {pick.decade_spread} | Score: {pick.score:.1f}")

    # Wheeling
    if args.wheel or args.budget:
        print(f"\n{'='*60}")
        print("WHEELING SYSTEM")
        print(f"{'='*60}")

        if args.budget:
            suggestion = suggest_wheel_size(args.budget, args.game, guarantee=3)
            print(f"\nBudget: {args.budget} RON")
            print(suggestion["recommendation"])

            if suggestion["optimal"]:
                wheel_size = suggestion["optimal"]["numbers_to_select"]
            else:
                print("\nInsufficient budget for wheeling.")
                return
        else:
            wheel_size = args.wheel

        # Get top numbers for wheel
        top_picks = optimizer.generate_optimized_picks(wheel_size, rng=rng)
        all_numbers = set()
        for p in top_picks:
            all_numbers.update(p.numbers)

        # Take top wheel_size numbers
        wheel_numbers = sorted(list(all_numbers))[:wheel_size]

        wheel = create_any_win_wheel(wheel_numbers, args.game, min_guarantee=3)

        print(f"\nWheel Numbers ({len(wheel_numbers)}): {wheel_numbers}")
        print(f"Tickets: {wheel['ticket_count']}")
        print(f"Cost: {wheel['cost_estimate']:.0f} RON")
        print(f"\n{wheel['explanation']}")

        print("\nTICKETS:")
        for i, ticket in enumerate(wheel["tickets"], 1):
            joker_str = ""
            if args.game == "joker":
                joker = random.randint(1, 20) if not rng else rng.randint(1, 20)
                joker_str = f" + J{joker}"
            print(f"  {i}. {', '.join(map(str, ticket))}{joker_str}")

    print(f"\n{'='*60}")
    print("NOTE: No method can guarantee winning. Play responsibly.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
