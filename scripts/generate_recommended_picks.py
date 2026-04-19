"""Generate optimally allocated lottery picks across games for a given budget.

Recommends the best combination of Joker, Loto 6/49, and Loto 5/40 tickets
to maximize the probability of winning any prize.

Usage:
    PYTHONPATH=src python scripts/generate_recommended_picks.py --budget 70
    PYTHONPATH=src python scripts/generate_recommended_picks.py --budget 100 --verbose
    PYTHONPATH=src python scripts/generate_recommended_picks.py --budget 100 --seed 42
    PYTHONPATH=src python scripts/generate_recommended_picks.py --budget 70 --strategy core_share
"""

import argparse
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from shared.ticket_allocator import (
    TicketAllocation,
    best_allocation,
    best_per_game_allocation,
    enumerate_allocations,
)
from shared.ticket_builders import (
    BuilderContext,
    CoreShareBuilder,
    IndependentBuilder,
    WheelBuilder,
)
from shared.ticket_io import dump_tickets
from shared.game_config import JOKER_CONFIG, LOTO_540_CONFIG, LOTO_649_CONFIG
from shared.ev_calculator import EVCalculator
from shared.pricing import compute_ticket_cost
from shared.recency import resolve_recency_settings
from shared.results_db import persist_generation_run
from shared.budget_ledger import BudgetLedger


DISPLAY_NAMES = {
    "joker": "Joker",
    "loto_649": "Loto 6/49",
    "loto_540": "Loto 5/40",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate optimally allocated lottery picks for a budget"
    )
    parser.add_argument(
        "--budget", type=float, required=True,
        help="Budget in RON (also the ledger credit on skip, debit cap on boost)",
    )
    parser.add_argument(
        "--bucket-budget", type=str, default=None,
        help=(
            "Per-game budget split, e.g. 'joker=20,loto_649=30,loto_540=20'. "
            "Runs the allocator independently per game so non-dominant games "
            "get exercised in production. Sum should not exceed --budget."
        ),
    )
    parser.add_argument(
        "--seed", type=int, help="Set deterministic RNG seed",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed probability breakdown",
    )
    parser.add_argument(
        "--mixes", type=int, default=1,
        help="Generate N diverse game mixes (default: 1 = single best allocation)",
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
        help="Save picks to files in this directory (tickets.json)",
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
    parser.add_argument(
        "--strategy", default="independent",
        help=(
            "Ticket build strategy: 'independent', 'core_share', "
            "or 'wheel:<pool_size>' (default: independent)"
        ),
    )
    parser.add_argument(
        "--ev-skip-ratio", type=float, default=0.5,
        help="Skip draw if all games' jackpot/breakeven ratio < this",
    )
    parser.add_argument(
        "--ev-boost-ratio", type=float, default=1.2,
        help="Boost budget from ledger if any game's ratio > this",
    )
    parser.add_argument(
        "--ledger-path", type=str, default="data/budget_bank.json",
        help="Path to budget ledger JSON file",
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


def _parse_bucket_budget(raw: str | None) -> dict[str, float] | None:
    """Parse `--bucket-budget` like 'joker=20,loto_649=30,loto_540=25' into a dict.

    Returns None when the flag is unset. Unknown games raise SystemExit so
    typos fail loudly rather than silently falling through to all-Joker.
    """
    if raw is None or not raw.strip():
        return None
    known = {"joker", "loto_649", "loto_540"}
    out: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--bucket-budget entry {part!r} must be 'game=amount'")
        game, amount = part.split("=", 1)
        game = game.strip()
        if game not in known:
            raise SystemExit(f"--bucket-budget unknown game {game!r}; known: {sorted(known)}")
        try:
            out[game] = float(amount)
        except ValueError as exc:
            raise SystemExit(f"--bucket-budget amount {amount!r} is not numeric") from exc
    return out


def load_loto_649_draws():
    from loto_649_model.fetch import update_dataset
    from loto_649_model.storage import load_draws

    url = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/loto_649_si_noroc/rezultate_extrageri.html"
    cache_path = Path("data/raw/loto_649_results.html")
    csv_path = Path("data/clean/loto_649_draws.csv")
    update_dataset(url, cache_path, csv_path)
    return load_draws(csv_path)


def load_loto_540_draws():
    from loto_540_model.fetch import update_dataset
    from loto_540_model.storage import load_draws

    url = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/540_si_super_noroc/rezultate_extrageri.html"
    cache_path = Path("data/raw/loto_540_results.html")
    csv_path = Path("data/clean/loto_540_draws.csv")
    update_dataset(url, cache_path, csv_path)
    return load_draws(csv_path)


def _builder_for_spec(spec: str):
    if spec == "independent":
        return IndependentBuilder(n_tickets=1)
    if spec == "core_share":
        return CoreShareBuilder()
    if spec.startswith("wheel:"):
        return WheelBuilder(pool_size=int(spec.split(":", 1)[1]))
    raise SystemExit(f"unknown strategy: {spec}")


_GAME_LOADERS = {
    "joker": load_joker_draws,
    "loto_649": load_loto_649_draws,
    "loto_540": load_loto_540_draws,
}

_GAME_CONFIGS = {
    "joker": JOKER_CONFIG,
    "loto_649": LOTO_649_CONFIG,
    "loto_540": LOTO_540_CONFIG,
}


def _build_tickets_for_allocation(allocation, rng, half_life, half_life_mode, strategy_spec):
    """Build Ticket objects for every game slot in the allocation."""
    from shared.ticket import Ticket
    tickets: list[Ticket] = []
    for game, n in allocation.tickets.items():
        if n <= 0:
            continue
        draws_list = _GAME_LOADERS[game]()
        draws_main = [d.main_numbers for d in draws_list]
        draw_dates = [d.date for d in draws_list]
        for _ in range(n):
            ctx = BuilderContext(
                game=game,
                config=_GAME_CONFIGS[game],
                draws=draws_main,
                draw_dates=draw_dates,
                rng=rng,
            )
            tickets.extend(_builder_for_spec(strategy_spec).build(ctx))
    return tickets


def _tickets_to_db_rows(tickets, suffix="") -> list[dict]:
    """Flatten tickets to per-variant dicts matching the legacy DB schema."""
    rows: list[dict] = []
    strategy_label = f"recommended{suffix}"
    game_idx: dict[str, int] = {}
    for ticket in tickets:
        game = ticket.game
        for variant in ticket.variants:
            idx = game_idx.get(game, 0) + 1
            game_idx[game] = idx
            rows.append(
                {
                    "game": game,
                    "strategy": strategy_label,
                    "line_no": idx,
                    "main_numbers": list(variant.main_numbers),
                    "joker_number": variant.bonus_number,
                }
            )
    return rows


def _format_allocation(allocation: TicketAllocation, budget: float) -> str:
    lines = [
        f"Budget: {budget:.0f} RON",
        f"P(any win): {allocation.p_any_win * 100:.2f}%",
        "",
        "Allocation:",
    ]
    for game, count in allocation.tickets.items():
        if count > 0:
            cost = compute_ticket_cost(game) * count
            lines.append(
                f"  {DISPLAY_NAMES[game]}: {count} ticket(s) ({cost:.1f} RON)"
            )
    unused = budget - allocation.total_cost
    if unused > 0.01:
        lines.append(f"  Unspent: {unused:.1f} RON")
    return "\n".join(lines)


def _write_outputs(output_dir, tickets, allocation, budget_ron, suffix=""):
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    total_cost = sum(t.cost_ron for t in tickets)
    dump_tickets(
        output_dir / f"tickets{suffix}.json",
        tickets=tickets,
        budget_ron=budget_ron,
        total_cost_ron=round(total_cost, 2),
        allocation=dict(allocation.tickets),
        generated_at=generated_at,
    )


def apply_ev_gate(
    allocation: TicketAllocation,
    enabled: bool,
    min_ratio: float,
    jackpots: dict[str, float],
) -> tuple[TicketAllocation, dict[str, dict[str, float | bool]]]:
    """Filter allocation by jackpot/breakeven ratio when gate is enabled."""
    if not enabled:
        return allocation, {}

    from shared.pricing import compute_ticket_cost as _cost
    calc = EVCalculator()
    games = {
        "joker": calc.create_joker(ticket_cost=_cost("joker")),
        "loto_649": calc.create_loto_649(ticket_cost=_cost("loto_649")),
        "loto_540": calc.create_loto_540(ticket_cost=_cost("loto_540")),
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
        filtered = TicketAllocation(
            tickets={"joker": 0, "loto_649": 0, "loto_540": 0},
            total_cost=0.0,
            p_any_win=0.0,
        )
        return filtered, details

    # Re-optimise using the original allocation's total cost as the budget cap.
    budget_cap = allocation.total_cost or sum(
        compute_ticket_cost(g) * c for g, c in allocation.tickets.items() if c > 0
    )
    filtered = best_allocation(budget_ron=budget_cap, allowed_games=allowed_games)
    return filtered, details


@dataclass
class EVDecision:
    action: str  # "play" | "skip" | "boost"
    reason: str
    extra_budget: float = 0.0


def apply_ev_gate_v2(
    budget: float,
    jackpots: dict[str, float | None],
    skip_ratio: float,
    boost_ratio: float,
) -> EVDecision:
    """Compute per-game jackpot/breakeven ratios and decide skip/boost/play.

    Uses each game's EV-tier ticket cost (main ticket only — side games
    have separate prize ladders not modeled by EVCalculator) so the
    breakeven reflects the prize tiers we actually score against.
    """
    calc = EVCalculator()
    games = {
        "joker": calc.create_joker(),
        "loto_649": calc.create_loto_649(),
        "loto_540": calc.create_loto_540(),
    }
    ratios: dict[str, float] = {}
    breakevens: dict[str, float] = {}
    for game_name, game in games.items():
        jackpot = jackpots.get(game_name)
        breakeven = calc._calculate_positive_ev_jackpot(game, 1.0, 0.0)  # noqa: SLF001
        breakevens[game_name] = float(breakeven or 0)
        if jackpot is None or breakeven is None or breakeven <= 0:
            ratios[game_name] = 0.0
        else:
            ratios[game_name] = jackpot / breakeven

    max_ratio = max(ratios.values())
    ratio_summary = "  ".join(
        f"{g}={ratios[g]:.2f}" for g in ("joker", "loto_649", "loto_540")
    )

    if max_ratio < skip_ratio:
        return EVDecision(
            action="skip",
            reason=f"all ratios < {skip_ratio} ({ratio_summary})",
        )
    if max_ratio > boost_ratio:
        return EVDecision(
            action="boost",
            reason=f"max ratio {max_ratio:.2f} > {boost_ratio} ({ratio_summary})",
            extra_budget=budget,
        )
    return EVDecision(
        action="play",
        reason=f"ratios OK (max {max_ratio:.2f}) ({ratio_summary})",
    )


def _top_n_diverse_allocations(budget_ron: float, n: int) -> list[TicketAllocation]:
    """Top N allocations with diverse active-game mixes, ranked by p_any_win."""
    all_allocs = enumerate_allocations(budget_ron=budget_ron)
    by_mix: dict[frozenset[str], TicketAllocation] = {}
    for alloc in all_allocs:
        active = frozenset(g for g, c in alloc.tickets.items() if c > 0)
        if not active:
            continue
        prev = by_mix.get(active)
        if prev is None or alloc.p_any_win > prev.p_any_win:
            by_mix[active] = alloc
    ranked = sorted(by_mix.values(), key=lambda a: a.p_any_win, reverse=True)
    return ranked[:n]


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

    jackpots = {
        "joker": args.joker_jackpot,
        "loto_649": args.loto649_jackpot,
        "loto_540": args.loto540_jackpot,
    }

    # Auto-fetch jackpots when the EV gate is enabled and any value is missing.
    # CLI args take precedence; scraper fills only the gaps.
    if args.ev_gate and any(v is None for v in jackpots.values()):
        from shared.jackpot_scraper import fetch_jackpots

        print("EV gate enabled; fetching current jackpots from loto.ro...")
        scraped = fetch_jackpots()
        for game, value in scraped.items():
            if jackpots[game] is None and value is not None:
                jackpots[game] = value
                print(f"  {game}: {value:,.0f} RON (scraped)")
            elif jackpots[game] is None:
                print(f"  {game}: unavailable (scraper returned None)")

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    effective_budget = args.budget
    if args.ev_gate:
        ledger_path = Path(args.ledger_path)
        ledger = BudgetLedger(ledger_path)
        decision = apply_ev_gate_v2(
            budget=args.budget,
            jackpots=jackpots,
            skip_ratio=args.ev_skip_ratio,
            boost_ratio=args.ev_boost_ratio,
        )
        draw_date = datetime.now(UTC).date().isoformat()
        if decision.action == "skip":
            ledger.credit_skip(draw_date=draw_date, amount=args.budget, reason=decision.reason)
            print(f"EV gate: SKIP — {decision.reason}. Ledger balance: {ledger.balance():.2f} RON")
            if output_dir:
                skip_msg = (
                    f"🎰 *Lottery Picks - {draw_date}*\n\n"
                    f"_Skipped: {decision.reason}_\n"
                    f"Ledger balance: {ledger.balance():.2f} RON\n"
                )
                (output_dir / "skip_notice.txt").write_text(skip_msg, encoding="utf-8")
            persist_generation_run(
                budget=args.budget,
                seed=args.seed,
                ev_gate=True,
                ev_min_ratio=args.ev_skip_ratio,
                jackpots=jackpots,
                allocation={"tickets": {}, "total_cost": 0.0, "p_any_win": 0.0, "budget": args.budget},
                gate_details={"action": "skip", "reason": decision.reason},
                tickets=[],
            )
            return
        if decision.action == "boost":
            actual = ledger.debit_boost(draw_date=draw_date, amount=decision.extra_budget, reason=decision.reason)
            effective_budget = args.budget + actual
            print(f"EV gate: BOOST +{actual:.2f} → effective {effective_budget:.2f} RON")

    if args.mixes > 1:
        allocations = _top_n_diverse_allocations(effective_budget, args.mixes)
        if args.ev_gate:
            filtered = []
            for alloc in allocations:
                alloc, _ = apply_ev_gate(
                    allocation=alloc,
                    enabled=True,
                    min_ratio=args.ev_min_ratio,
                    jackpots=jackpots,
                )
                if alloc.p_any_win > 0:
                    filtered.append(alloc)
            allocations = filtered

        all_db_rows: list[dict] = []
        for i, alloc in enumerate(allocations, 1):
            games_in_mix = [
                DISPLAY_NAMES[g]
                for g in ["joker", "loto_649", "loto_540"]
                if alloc.tickets.get(g, 0) > 0
            ]
            print(f"\n{'=' * 50}")
            print(f"MIX {i}: {' + '.join(games_in_mix)}")
            print(f"{'=' * 50}")
            print(_format_allocation(alloc, effective_budget))

            tickets = _build_tickets_for_allocation(
                alloc, rng, half_life, half_life_mode, args.strategy,
            )

            _print_tickets(tickets)

            suffix = f"_mix{i}"
            if output_dir:
                _write_outputs(output_dir, tickets, alloc, effective_budget, suffix=suffix)

            all_db_rows.extend(_tickets_to_db_rows(tickets, suffix=suffix))


        best_alloc = allocations[0] if allocations else TicketAllocation(
            tickets={"joker": 0, "loto_649": 0, "loto_540": 0},
            total_cost=0.0,
            p_any_win=0.0,
        )
        persist_generation_run(
            budget=effective_budget,
            seed=args.seed,
            ev_gate=args.ev_gate,
            ev_min_ratio=args.ev_min_ratio,
            jackpots=jackpots,
            allocation={
                "tickets": best_alloc.tickets,
                "total_cost": best_alloc.total_cost,
                "p_any_win": best_alloc.p_any_win,
                "budget": effective_budget,
                "mixes": len(allocations),
            },
            gate_details={},
            tickets=all_db_rows,
        )
    else:
        buckets = _parse_bucket_budget(args.bucket_budget)
        if buckets is not None:
            allocation = best_per_game_allocation(buckets)
        else:
            allocation = best_allocation(budget_ron=effective_budget)
        allocation, gate_details = apply_ev_gate(
            allocation=allocation,
            enabled=args.ev_gate,
            min_ratio=args.ev_min_ratio,
            jackpots=jackpots,
        )
        print(_format_allocation(allocation, effective_budget))

        if args.verbose and args.ev_gate:
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

        if allocation.p_any_win == 0:
            persist_generation_run(
                budget=effective_budget,
                seed=args.seed,
                ev_gate=args.ev_gate,
                ev_min_ratio=args.ev_min_ratio,
                jackpots=jackpots,
                allocation={
                    "tickets": allocation.tickets,
                    "total_cost": allocation.total_cost,
                    "p_any_win": allocation.p_any_win,
                    "budget": effective_budget,
                },
                gate_details=gate_details,
                tickets=[],
            )
            return

        tickets = _build_tickets_for_allocation(
            allocation, rng, half_life, half_life_mode, args.strategy,
        )

        _print_tickets(tickets)

        if output_dir:
            _write_outputs(output_dir, tickets, allocation, effective_budget)

        db_rows = _tickets_to_db_rows(tickets)
        persist_generation_run(
            budget=effective_budget,
            seed=args.seed,
            ev_gate=args.ev_gate,
            ev_min_ratio=args.ev_min_ratio,
            jackpots=jackpots,
            allocation={
                "tickets": allocation.tickets,
                "total_cost": allocation.total_cost,
                "p_any_win": allocation.p_any_win,
                "budget": effective_budget,
            },
            gate_details=gate_details,
            tickets=db_rows,
        )


def _print_tickets(tickets) -> None:
    from shared.pricing import VARIANTS_PER_TICKET
    by_game: dict[str, list] = {}
    for t in tickets:
        by_game.setdefault(t.game, []).append(t)

    for game in ["joker", "loto_649", "loto_540"]:
        game_tickets = by_game.get(game, [])
        if not game_tickets:
            continue
        print()
        print(f"--- {DISPLAY_NAMES[game]} ({len(game_tickets)} ticket(s)) ---")
        idx = 1
        for t in game_tickets:
            for v in t.variants:
                main = ", ".join(str(n) for n in v.main_numbers)
                if game == "joker":
                    print(f"{idx}. {main} + J{v.bonus_number}")
                else:
                    print(f"{idx}. {main}")
                idx += 1




if __name__ == "__main__":
    main()
