#!/usr/bin/env python3
"""Run randomness analysis across lottery games and produce a JSON report.

Usage:
    PYTHONPATH=src python scripts/analyze_randomness.py
    PYTHONPATH=src python scripts/analyze_randomness.py --game joker
    PYTHONPATH=src python scripts/analyze_randomness.py --output report.json
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from joker_model.storage import load_draws as load_joker_draws
from loto_649_model.storage import load_draws as load_loto649_draws
from loto_540_model.storage import load_draws as load_loto540_draws
from shared.analysis_result import AnalysisResult
from shared.benford import run_benford_analysis
from shared.cross_game_analysis import run_cross_game_analysis
from shared.game_config import JOKER_CONFIG, LOTO_540_CONFIG, LOTO_649_CONFIG
from shared.hurst import run_hurst_analysis
from shared.randomness_tests import run_randomness_tests

GAME_REGISTRY = {
    "joker": {
        "config": JOKER_CONFIG,
        "csv_path": Path("data/clean/joker_draws.csv"),
        "loader": load_joker_draws,
    },
    "loto_649": {
        "config": LOTO_649_CONFIG,
        "csv_path": Path("data/clean/loto_649_draws.csv"),
        "loader": load_loto649_draws,
    },
    "loto_540": {
        "config": LOTO_540_CONFIG,
        "csv_path": Path("data/clean/loto_540_draws.csv"),
        "loader": load_loto540_draws,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run randomness analysis on lottery draw history",
    )
    parser.add_argument(
        "--game",
        choices=list(GAME_REGISTRY.keys()),
        default=None,
        help="Analyze a single game (default: all games)",
    )
    parser.add_argument(
        "--output",
        default="data/analysis/randomness_report.json",
        help="Output path for JSON report (default: data/analysis/randomness_report.json)",
    )
    return parser.parse_args()


def load_game_draws(game_key: str) -> tuple[list[list[int]], list[str]]:
    """Load draws for a game and return (main_numbers_list, date_strings)."""
    entry = GAME_REGISTRY[game_key]
    csv_path = entry["csv_path"]
    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found")
        return [], []

    draws = entry["loader"](csv_path)
    if not draws:
        print(f"  [skip] No draws loaded from {csv_path}")
        return [], []

    main_numbers = [d.main_numbers for d in draws]
    dates = [d.date for d in draws]
    return main_numbers, dates


def run_per_game_analysis(
    game_key: str,
    main_numbers: list[list[int]],
) -> list[AnalysisResult]:
    """Run all four analysis modules for one game."""
    config = GAME_REGISTRY[game_key]["config"]
    pool_size = config.pool_size
    game_label = config.name

    results: list[AnalysisResult] = []

    randomness_results = run_randomness_tests(main_numbers, pool_size)
    for result in randomness_results:
        result.game = game_label
    results.extend(randomness_results)

    benford_results = run_benford_analysis(main_numbers, pool_size)
    for result in benford_results:
        result.game = game_label
    results.extend(benford_results)

    hurst_results = run_hurst_analysis(main_numbers, pool_size)
    for result in hurst_results:
        result.game = game_label
    results.extend(hurst_results)

    return results


def build_cross_game_input(
    loaded_games: dict[str, tuple[list[list[int]], list[str]]],
) -> dict[str, list[tuple[date, list[int]]]]:
    """Convert loaded game data into the format expected by cross_game_analysis."""
    cross_game_data: dict[str, list[tuple[date, list[int]]]] = {}
    for game_key, (main_numbers, date_strings) in loaded_games.items():
        game_label = GAME_REGISTRY[game_key]["config"].name
        dated_draws: list[tuple[date, list[int]]] = []
        for date_str, numbers in zip(date_strings, main_numbers):
            draw_date = date.fromisoformat(date_str)
            dated_draws.append((draw_date, numbers))
        cross_game_data[game_label] = dated_draws
    return cross_game_data


def print_summary(results: list[AnalysisResult]) -> None:
    """Print a terminal-friendly summary with pass/fail counts."""
    passed = sum(1 for r in results if r.passed is True)
    failed = sum(1 for r in results if r.passed is False)
    inconclusive = sum(1 for r in results if r.passed is None)

    print()
    print("=" * 70)
    print("RANDOMNESS ANALYSIS SUMMARY")
    print("=" * 70)
    print()

    current_game = None
    for result in results:
        if result.game != current_game:
            current_game = result.game
            print(f"--- {current_game} ---")
        print(f"  {result.summary_line()}")

    print()
    print("-" * 70)
    print(f"Total: {len(results)} tests | "
          f"PASS: {passed} | FAIL: {failed} | Inconclusive: {inconclusive}")
    print("-" * 70)


def write_report(results: list[AnalysisResult], output_path: str) -> None:
    """Write the full JSON report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "test_count": len(results),
        "passed": sum(1 for r in results if r.passed is True),
        "failed": sum(1 for r in results if r.passed is False),
        "inconclusive": sum(1 for r in results if r.passed is None),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nReport written to {path}")


def main() -> None:
    args = parse_args()

    game_keys = [args.game] if args.game else list(GAME_REGISTRY.keys())

    loaded_games: dict[str, tuple[list[list[int]], list[str]]] = {}
    all_results: list[AnalysisResult] = []

    for game_key in game_keys:
        game_label = GAME_REGISTRY[game_key]["config"].name
        print(f"Loading {game_label} draws...")
        main_numbers, dates = load_game_draws(game_key)
        if not main_numbers:
            continue

        loaded_games[game_key] = (main_numbers, dates)
        print(f"  {len(main_numbers)} draws loaded")

        print(f"Running analysis for {game_label}...")
        results = run_per_game_analysis(game_key, main_numbers)
        all_results.extend(results)

    if len(loaded_games) >= 2:
        print("Running cross-game analysis...")
        cross_game_data = build_cross_game_input(loaded_games)
        cross_results = run_cross_game_analysis(cross_game_data)
        all_results.extend(cross_results)

    if not all_results:
        print("No analysis results produced. Check that data files exist.")
        raise SystemExit(1)

    print_summary(all_results)
    write_report(all_results, args.output)


if __name__ == "__main__":
    main()
