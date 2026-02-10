"""Generate optimally allocated lottery picks across games for a given budget.

Recommends the best combination of Joker, Loto 6/49, and Loto 5/40 tickets
to maximize the probability of winning any prize.

Usage:
    PYTHONPATH=src python scripts/generate_recommended_picks.py --budget 24
    PYTHONPATH=src python scripts/generate_recommended_picks.py --budget 100 --verbose
    PYTHONPATH=src python scripts/generate_recommended_picks.py --budget 100 --seed 42
"""

import argparse
import os
import random
from pathlib import Path

from shared.game_recommender import (
    calculate_win_probability,
    format_recommendation,
    optimize_budget,
)
from shared.game_config import JOKER_CONFIG, LOTO_540_CONFIG, LOTO_649_CONFIG
from shared.ensemble_blend import generate_blended_picks
from shared.recency import resolve_recency_settings


JOKER_SECONDARY_POOL = 20


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate optimally allocated lottery picks for a budget"
    )
    parser.add_argument(
        "--budget", type=float, required=True,
        help="Budget in RON",
    )
    parser.add_argument(
        "--seed", type=int, help="Set deterministic RNG seed",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed probability breakdown",
    )
    parser.add_argument(
        "--half-life", type=float, help="Recency half-life in draws (or days)",
    )
    parser.add_argument(
        "--half-life-mode", choices=["draws", "days"],
        help="Recency half-life mode (default: draws)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Save picks to files in this directory (joker.txt, loto649.txt, loto540.txt)",
    )
    return parser


def load_joker_draws():
    from joker_model.fetch import update_dataset
    from joker_model.storage import load_draws

    url = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html"
    cache_path = Path("data/raw/joker_results.html")
    csv_path = Path("data/clean/joker_draws.csv")
    update_dataset(url, cache_path, csv_path)
    return load_draws(csv_path)


def load_loto_649_draws():
    from loto_649_model.fetch import update_dataset
    from loto_649_model.storage import load_draws

    url = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/loto_649_si_noroc/rezultate_extrageri.html"
    cache_path = Path("data/raw/loto_649_results.html")
    csv_path = Path("data/clean/loto_649_draws.csv")
    update_dataset(url, cache_path, csv_path)
    return load_draws(csv_path)


def generate_joker_picks(draws, count, rng, half_life, half_life_mode):
    draw_main_only = [d.main_numbers for d in draws]
    draw_dates = [d.date for d in draws]

    main_picks = generate_blended_picks(
        JOKER_CONFIG,
        draw_main_only,
        count,
        rng,
        half_life=half_life,
        half_life_mode=half_life_mode,
        draw_dates=draw_dates,
    )
    return [(main, rng.randint(1, JOKER_SECONDARY_POOL)) for main in main_picks]


def load_loto_540_draws():
    from loto_540_model.fetch import update_dataset
    from loto_540_model.storage import load_draws

    url = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/540_si_super_noroc/rezultate_extrageri.html"
    cache_path = Path("data/raw/loto_540_results.html")
    csv_path = Path("data/clean/loto_540_draws.csv")
    update_dataset(url, cache_path, csv_path)
    return load_draws(csv_path)


def generate_540_picks(draws, count, rng, half_life, half_life_mode):
    draw_main_only = [d.main_numbers for d in draws]
    draw_dates = [d.date for d in draws]

    main_picks = generate_blended_picks(
        LOTO_540_CONFIG,
        draw_main_only,
        count,
        rng,
        half_life=half_life,
        half_life_mode=half_life_mode,
        draw_dates=draw_dates,
    )
    return main_picks


def generate_649_picks(draws, count, rng, half_life, half_life_mode):
    draw_main_only = [d.main_numbers for d in draws]
    draw_dates = [d.date for d in draws]

    main_picks = generate_blended_picks(
        LOTO_649_CONFIG,
        draw_main_only,
        count,
        rng,
        half_life=half_life,
        half_life_mode=half_life_mode,
        draw_dates=draw_dates,
    )
    return main_picks


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        half_life, half_life_mode = resolve_recency_settings(
            args.half_life,
            os.getenv("RECENCY_HALF_LIFE"),
            args.half_life_mode,
            os.getenv("RECENCY_HALF_LIFE_MODE"),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rng = random.Random(args.seed) if args.seed is not None else random.SystemRandom()

    allocation = optimize_budget(args.budget)
    print(format_recommendation(allocation))

    if args.verbose:
        print()
        print("Per-game analysis:")
        for game in ["joker", "loto_649", "loto_540"]:
            gp = calculate_win_probability(game)
            print(
                f"  {game}: {gp.win_rate * 100:.4f}% win rate, "
                f"{gp.ticket_cost:.0f} RON, "
                f"{gp.win_rate_per_ron * 100:.4f}%/RON"
            )

    if allocation.p_any_win == 0:
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    n_joker = allocation.tickets.get("joker", 0)
    n_649 = allocation.tickets.get("loto_649", 0)
    n_540 = allocation.tickets.get("loto_540", 0)

    if n_joker > 0:
        print()
        print(f"--- Joker ({n_joker} tickets) ---")
        draws = load_joker_draws()
        lines = generate_joker_picks(draws, n_joker, rng, half_life, half_life_mode)
        joker_lines = []
        for idx, (main, joker) in enumerate(lines, 1):
            line = f"{idx}. {', '.join(str(n) for n in main)} + J{joker}"
            print(line)
            joker_lines.append(line)
        if output_dir:
            (output_dir / "joker.txt").write_text("\n".join(joker_lines) + "\n")

    if n_649 > 0:
        print()
        print(f"--- Loto 6/49 ({n_649} tickets) ---")
        draws = load_loto_649_draws()
        lines = generate_649_picks(draws, n_649, rng, half_life, half_life_mode)
        loto_lines = []
        for idx, main in enumerate(lines, 1):
            line = f"{idx}. {', '.join(str(n) for n in main)}"
            print(line)
            loto_lines.append(line)
        if output_dir:
            (output_dir / "loto649.txt").write_text("\n".join(loto_lines) + "\n")

    if n_540 > 0:
        print()
        print(f"--- Loto 5/40 ({n_540} tickets) ---")
        draws = load_loto_540_draws()
        lines = generate_540_picks(draws, n_540, rng, half_life, half_life_mode)
        loto540_lines = []
        for idx, main in enumerate(lines, 1):
            line = f"{idx}. {', '.join(str(n) for n in main)}"
            print(line)
            loto540_lines.append(line)
        if output_dir:
            (output_dir / "loto540.txt").write_text("\n".join(loto540_lines) + "\n")


if __name__ == "__main__":
    main()
