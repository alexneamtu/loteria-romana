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
    BudgetAllocation,
    TICKET_COSTS,
    calculate_win_probability,
    format_recommendation,
    optimize_budget,
)
from shared.game_config import JOKER_CONFIG, LOTO_540_CONFIG, LOTO_649_CONFIG
from shared.ev_calculator import EVCalculator
from shared.ensemble_blend import generate_blended_picks
from shared.joker_set_optimizer import (
    assign_max_coverage_jokers,
    optimize_main_ticket_set,
)
from shared.recency import resolve_recency_settings
from shared.results_db import persist_generation_run


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
    parser.add_argument(
        "--ev-gate", action="store_true",
        help="Enable EV/jackpot gate before generating picks",
    )
    parser.add_argument(
        "--ev-min-ratio", type=float, default=0.8,
        help="Minimum jackpot/breakeven ratio required when EV gate is enabled",
    )
    parser.add_argument(
        "--joker-jackpot", type=float,
        help="Current Joker jackpot in RON (required for Joker EV gating)",
    )
    parser.add_argument(
        "--loto649-jackpot", type=float,
        help="Current Loto 6/49 jackpot in RON (required for 6/49 EV gating)",
    )
    parser.add_argument(
        "--loto540-jackpot", type=float,
        help="Current Loto 5/40 jackpot in RON (required for 5/40 EV gating)",
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
    optimized_main = optimize_main_ticket_set(
        candidates=main_picks,
        select_count=count,
        pool_size=JOKER_CONFIG.pool_size,
        numbers_to_pick=JOKER_CONFIG.numbers_to_pick,
        rng=rng,
        anti_crowding_weight=0.35,
    )
    jokers = assign_max_coverage_jokers(
        count=len(optimized_main),
        rng=rng,
        joker_pool=JOKER_SECONDARY_POOL,
    )
    return list(zip(optimized_main, jokers))


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


def _optimize_budget_with_allowed_games(
    budget_ron: float,
    allowed_games: set[str],
) -> BudgetAllocation:
    if budget_ron < min(TICKET_COSTS.values()):
        return BudgetAllocation(budget=budget_ron)

    probabilities = {
        game: calculate_win_probability(game).win_rate
        for game in TICKET_COSTS
    }
    budget = int(budget_ron)
    best = BudgetAllocation(budget=budget_ron)

    joker_range = range(budget // 8 + 1) if "joker" in allowed_games else [0]
    loto649_range_template = range(budget // 6 + 1) if "loto_649" in allowed_games else [0]
    loto540_allowed = "loto_540" in allowed_games

    for n_joker in joker_range:
        remaining_after_joker = budget - n_joker * 8
        loto649_range = loto649_range_template if "loto_649" in allowed_games else [0]
        for n_649 in loto649_range:
            cost_649 = n_649 * 6
            if cost_649 > remaining_after_joker:
                continue
            remaining = remaining_after_joker - cost_649
            n_540 = remaining // 4 if loto540_allowed else 0

            tickets = {"joker": n_joker, "loto_649": n_649, "loto_540": n_540}
            total_tickets = sum(tickets.values())
            if total_tickets == 0:
                continue

            p_no_win = 1.0
            for game, count in tickets.items():
                if count <= 0:
                    continue
                p_no_win *= (1 - probabilities[game]) ** count
            p_any = 1 - p_no_win
            cost = n_joker * 8 + n_649 * 6 + n_540 * 4

            if p_any > best.p_any_win:
                best = BudgetAllocation(
                    tickets=tickets,
                    total_cost=float(cost),
                    p_any_win=p_any,
                    budget=budget_ron,
                )

    return best


def apply_ev_gate(
    allocation: BudgetAllocation,
    enabled: bool,
    min_ratio: float,
    jackpots: dict[str, float],
) -> tuple[BudgetAllocation, dict[str, dict[str, float | bool]]]:
    """Filter allocation by jackpot/breakeven ratio when gate is enabled."""
    if not enabled:
        return allocation, {}

    calc = EVCalculator()
    games = {
        "joker": calc.create_joker(ticket_cost=TICKET_COSTS["joker"]),
        "loto_649": calc.create_loto_649(ticket_cost=TICKET_COSTS["loto_649"]),
        "loto_540": calc.create_loto_540(ticket_cost=TICKET_COSTS["loto_540"]),
    }

    details: dict[str, dict[str, float | bool]] = {}
    allowed_games: set[str] = set()

    for game_name, game in games.items():
        jackpot = jackpots.get(game_name)
        breakeven = calc._calculate_positive_ev_jackpot(game, 1.0, 0.0)  # noqa: SLF001
        ratio = 0.0
        if jackpot and breakeven and breakeven > 0:
            ratio = jackpot / breakeven
        passes = ratio >= min_ratio if jackpot is not None else False
        if passes:
            allowed_games.add(game_name)
        details[game_name] = {
            "jackpot": float(jackpot) if jackpot is not None else 0.0,
            "breakeven": float(breakeven) if breakeven is not None else 0.0,
            "ratio": ratio,
            "passes": passes,
        }

    if not allowed_games:
        filtered = BudgetAllocation(
            tickets={"joker": 0, "loto_649": 0, "loto_540": 0},
            total_cost=0.0,
            p_any_win=0.0,
            budget=allocation.budget,
        )
        return filtered, details

    filtered = _optimize_budget_with_allowed_games(
        budget_ron=allocation.budget,
        allowed_games=allowed_games,
    )
    return filtered, details


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
    jackpots = {
        "joker": args.joker_jackpot,
        "loto_649": args.loto649_jackpot,
        "loto_540": args.loto540_jackpot,
    }
    allocation, gate_details = apply_ev_gate(
        allocation=allocation,
        enabled=args.ev_gate,
        min_ratio=args.ev_min_ratio,
        jackpots=jackpots,
    )
    print(format_recommendation(allocation))

    allocation_payload = {
        "tickets": allocation.tickets,
        "total_cost": allocation.total_cost,
        "p_any_win": allocation.p_any_win,
        "budget": allocation.budget,
    }

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
        if args.ev_gate:
            print()
            print(f"EV gate enabled (min jackpot/breakeven ratio: {args.ev_min_ratio:.2f})")
            for game_name in ["joker", "loto_649", "loto_540"]:
                info = gate_details.get(game_name, {})
                status = "pass" if info.get("passes") else "blocked"
                print(
                    f"  {game_name}: {status} "
                    f"(jackpot={info.get('jackpot', 0):.0f}, "
                    f"breakeven={info.get('breakeven', 0):.0f}, "
                    f"ratio={info.get('ratio', 0):.3f})"
                )

    generated_tickets: list[dict[str, object]] = []

    if allocation.p_any_win == 0:
        persist_generation_run(
            budget=args.budget,
            seed=args.seed,
            ev_gate=args.ev_gate,
            ev_min_ratio=args.ev_min_ratio,
            jackpots=jackpots,
            allocation=allocation_payload,
            gate_details=gate_details,
            tickets=generated_tickets,
        )
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
            generated_tickets.append(
                {
                    "game": "joker",
                    "strategy": "recommended",
                    "line_no": idx,
                    "main_numbers": main,
                    "joker_number": joker,
                }
            )
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
            generated_tickets.append(
                {
                    "game": "loto_649",
                    "strategy": "recommended",
                    "line_no": idx,
                    "main_numbers": main,
                    "joker_number": None,
                }
            )
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
            generated_tickets.append(
                {
                    "game": "loto_540",
                    "strategy": "recommended",
                    "line_no": idx,
                    "main_numbers": main,
                    "joker_number": None,
                }
            )
        if output_dir:
            (output_dir / "loto540.txt").write_text("\n".join(loto540_lines) + "\n")

    persist_generation_run(
        budget=args.budget,
        seed=args.seed,
        ev_gate=args.ev_gate,
        ev_min_ratio=args.ev_min_ratio,
        jackpots=jackpots,
        allocation=allocation_payload,
        gate_details=gate_details,
        tickets=generated_tickets,
    )


if __name__ == "__main__":
    main()
