"""Evaluate all strategies on holdout data for each game.

Usage:
    PYTHONPATH=src python scripts/evaluate_holdout.py
    PYTHONPATH=src python scripts/evaluate_holdout.py --holdout-size 50
    PYTHONPATH=src python scripts/evaluate_holdout.py --game joker
"""

import argparse
import csv
import random
from pathlib import Path

from shared.game_config import JOKER_CONFIG, LOTO_649_CONFIG, LOTO_540_CONFIG
from shared.holdout import split_holdout
from shared.holdout_evaluator import evaluate_all_strategies, format_holdout_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate strategies on holdout data")
    parser.add_argument(
        "--holdout-size", type=int, default=20,
        help="Number of most recent draws to hold out (default: 20)",
    )
    parser.add_argument(
        "--game", choices=["joker", "loto649", "loto540", "all"], default="all",
        help="Which game to evaluate (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    games = {
        "joker": (JOKER_CONFIG, Path("data/clean/joker_draws.csv")),
        "loto649": (LOTO_649_CONFIG, Path("data/clean/loto_649_draws.csv")),
        "loto540": (LOTO_540_CONFIG, Path("data/clean/loto_540_draws.csv")),
    }

    if args.game != "all":
        games = {args.game: games[args.game]}

    for game_name, (config, csv_path) in games.items():
        if not csv_path.exists():
            print(f"Skipping {game_name}: {csv_path} not found")
            continue

        draws = _load_main_numbers(csv_path, config)
        if len(draws) < args.holdout_size + 30:
            print(f"Skipping {game_name}: insufficient data ({len(draws)} draws)")
            continue

        split = split_holdout(draws, holdout_size=args.holdout_size)

        rng = random.Random(args.seed)
        results = evaluate_all_strategies(
            config=config,
            train_draws=split.train,
            holdout_draws=split.holdout,
            rng=rng,
        )

        print(format_holdout_report(results))
        print()


def _load_main_numbers(csv_path: Path, config) -> list[list[int]]:
    """Load draws from CSV, extracting only main numbers."""
    draws = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            numbers = []
            for i in range(1, config.numbers_drawn + 1):
                for prefix in (f"n{i}", f"main_{i}"):
                    if prefix in row:
                        numbers.append(int(row[prefix]))
                        break
            if len(numbers) == config.numbers_drawn:
                draws.append(sorted(numbers))
    return draws


if __name__ == "__main__":
    main()
