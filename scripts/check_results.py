#!/usr/bin/env python3
"""Check lottery results and compare against saved picks from multiple strategies."""

import csv
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from joker_model.fetch import update_dataset as update_joker
from joker_model.storage import load_draws as load_joker
from loto_649_model.fetch import update_dataset as update_loto649
from loto_649_model.storage import load_draws as load_loto649
from loto_540_model.fetch import update_dataset as update_loto540
from loto_540_model.storage import load_draws as load_loto540
from shared.picks_detail_history import append_detail_rows
from shared.results_db import persist_check_results
from shared.telegram_formatter import format_check_results as _fmt_check_results


def write_picks_detail(jsonl_path: Path, rows: list[dict]) -> None:
    append_detail_rows(jsonl_path, rows)


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def is_side_game_ready(draw, attr: str) -> bool:
    """Return True if the draw has a non-None value for the given side-game attribute."""
    return getattr(draw, attr, None) is not None


def parse_tickets_json(path: Path) -> list:
    """Load tickets.json and return list of Ticket objects."""
    from shared.ticket_io import load_tickets
    doc = load_tickets(path)
    return doc["tickets"]


def score_side_game_match(game: str, ticket_side: str, winning_side: str | None) -> tuple[int, int]:
    """Return (exact_match, digits_matched).

    - joker: Noroc Plus is a bonus ball — match/no-match only. digits_matched == exact_match.
    - loto_649 / loto_540: right-aligned digit match (loto.ro Noroc prize ladder scores
      consecutive rightmost matching digits; digit count drives the prize tier).
    """
    if winning_side is None:
        return 0, 0
    if game == "joker":
        exact = int(ticket_side == winning_side)
        return exact, exact
    matched = 0
    for ch1, ch2 in zip(reversed(ticket_side), reversed(winning_side)):
        if ch1 == ch2:
            matched += 1
        else:
            break
    exact = int(ticket_side == winning_side)
    return exact, matched


def parse_picks(text: str) -> list[list[int]]:
    """Parse picks from saved text format."""
    picks = []
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        match = re.match(r"\d+\.\s*([\d,\s]+)", line)
        if match:
            numbers = [int(n.strip()) for n in match.group(1).split(",")]
            picks.append(numbers)
    return picks


def check_matches(picks: list[list[int]], winning_numbers: list[int]) -> list[dict]:
    """Check how many numbers match."""
    results = []
    for pick in picks:
        matched = set(pick) & set(winning_numbers)
        results.append({
            "pick": pick,
            "matched": sorted(matched),
            "count": len(matched),
        })
    return results


def format_pick_result(pick: list[int], matched: list[int], count: int, total: int) -> str:
    """Format a single pick result line."""
    pick_str = ", ".join(str(n) for n in pick)
    if count == 0:
        return f"  {pick_str} - no matches"
    matched_str = ", ".join(str(n) for n in matched)
    return f"  {pick_str} - {count}/{total} [{matched_str}]"


def find_strategy_files(picks_dir: Path, game_prefix: str) -> dict[str, Path]:
    """Find all strategy files for a game. Returns {strategy_name: file_path}."""
    strategies = {}
    # Check for strategy-specific files (e.g. joker_blend.txt)
    for path in sorted(picks_dir.glob(f"{game_prefix}_*.txt")):
        strategy = path.stem.replace(f"{game_prefix}_", "")
        strategies[strategy] = path
    # Check for plain game file (e.g. joker.txt) from budget-optimized picks
    plain_file = picks_dir / f"{game_prefix}.txt"
    if plain_file.exists() and "recommended" not in strategies:
        strategies["recommended"] = plain_file
    return strategies


def check_game_strategies(
    picks_dir: Path,
    game_prefix: str,
    winning_numbers: list[int],
    match_total: int,
) -> dict[str, dict]:
    """Check all strategy files for a game against winning numbers."""
    strategy_files = find_strategy_files(picks_dir, game_prefix)
    strategy_results = {}

    for strategy, path in strategy_files.items():
        content = path.read_text().strip()
        if not content:
            continue
        picks = parse_picks(content)
        results = check_matches(picks, winning_numbers)
        score = sum(r["count"] for r in results)
        best_match = max((r["count"] for r in results), default=0)
        strategy_results[strategy] = {
            "results": results,
            "score": score,
            "best_match": best_match,
        }

    return strategy_results


def build_game_message(
    emoji: str,
    game_label: str,
    winning_str: str,
    draw_date: str,
    strategy_results: dict[str, dict],
    match_total: int,
) -> str:
    """Build a Telegram message for one game with all strategy results."""
    lines = [f"{emoji} *{game_label} Results - {draw_date}*"]
    lines.append(f"Winning: `{winning_str}`")

    if not strategy_results:
        lines.append("\n_No picks found to compare_")
        return "\n".join(lines)

    ranked = sorted(
        strategy_results.items(),
        key=lambda item: (item[1]["score"], item[1]["best_match"]),
        reverse=True,
    )

    for strategy, data in ranked:
        lines.append(f"\n*{strategy}* ({data['score']} matches)")
        lines.append("```")
        for r in data["results"]:
            lines.append(
                format_pick_result(r["pick"], r["matched"], r["count"], match_total)
            )
        lines.append("```")

    return "\n".join(lines)


def build_comparison_message(
    game_results: list[tuple[str, str, dict[str, dict], int]],
) -> str:
    """Build a summary comparison message across all games."""
    lines = ["📊 *Strategy Ranking*"]

    for emoji, game_label, strategy_results, match_total in game_results:
        if not strategy_results:
            continue

        lines.append(f"\n{emoji} *{game_label}*")
        lines.append("```")

        ranked = sorted(
            strategy_results.items(),
            key=lambda item: (item[1]["score"], item[1]["best_match"]),
            reverse=True,
        )

        for i, (strategy, data) in enumerate(ranked):
            best = data["best_match"]
            score = data["score"]
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            lines.append(f"{medal} {strategy}: {score} total (best {best}/{match_total})")

        lines.append("```")

    return "\n".join(lines)


HISTORY_COLUMNS = [
    "date", "game", "strategy", "picks_count",
    "total_matches", "best_match", "match_total", "winning_numbers",
    "ticket_id", "variants_count", "side_game_match", "side_game_digits",
    "winning_side_game", "builder_name",
]


def append_history(
    history_path: Path,
    game_results: list[tuple[str, str, str, dict[str, dict], int]],
) -> None:
    """Append results to CSV history file."""
    file_exists = history_path.exists() and history_path.stat().st_size > 0
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        for draw_date, game_key, winning_str, strategy_results, match_total in game_results:
            if not strategy_results:
                continue
            for strategy, data in strategy_results.items():
                writer.writerow({
                    "date": draw_date,
                    "game": game_key,
                    "strategy": strategy,
                    "picks_count": len(data.get("results", [])),
                    "total_matches": data["score"],
                    "best_match": data["best_match"],
                    "match_total": match_total,
                    "winning_numbers": winning_str,
                    "ticket_id": data.get("ticket_id", ""),
                    "variants_count": data.get("variants_count", ""),
                    "side_game_match": data.get("side_game_match", ""),
                    "side_game_digits": data.get("side_game_digits", ""),
                    "winning_side_game": data.get("winning_side_game", ""),
                    "builder_name": data.get("builder_name", ""),
                })

    log(f"Appended results to {history_path}")


def fetch_results_with_retry(max_retries: int = 3, delay_minutes: int = 1):
    """Fetch results with retry logic."""
    joker_cache = Path("data/raw/joker_results.html")
    joker_csv = Path("data/clean/joker_draws.csv")
    loto_cache = Path("data/raw/loto_649_results.html")
    loto_csv = Path("data/clean/loto_649_draws.csv")
    loto540_cache = Path("data/raw/loto_540_results.html")
    loto540_csv = Path("data/clean/loto_540_draws.csv")

    joker_cache.parent.mkdir(parents=True, exist_ok=True)
    joker_csv.parent.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    log("=== Results Checker Started ===")
    log(f"Current time: {datetime.now().isoformat()}")
    log(f"Today: {today}, Looking for results from: {yesterday}")

    for attempt in range(max_retries):
        log(f"\n--- Attempt {attempt + 1}/{max_retries} ---")

        for cache in [joker_cache, loto_cache, loto540_cache]:
            if cache.exists():
                log(f"Clearing cache: {cache}")
                cache.unlink()

        try:
            log("Fetching Joker results from loto.ro...")
            update_joker(
                "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html",
                joker_cache,
                joker_csv,
            )
            log("Fetching Loto 6/49 results from loto.ro...")
            update_loto649(
                "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/649_si_noroc/rezultate_extragere.html",
                loto_cache,
                loto_csv,
            )
            log("Fetching Loto 5/40 results from loto.ro...")
            update_loto540(
                "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/540_si_super_noroc/rezultate_extrageri.html",
                loto540_cache,
                loto540_csv,
            )
            log("Fetch completed successfully.")
        except Exception as e:
            log(f"ERROR: Fetch failed: {e}")
            if attempt < max_retries - 1:
                log(f"Waiting {delay_minutes} minute(s) before retry...")
                time.sleep(delay_minutes * 60)
                continue
            raise

        log("Loading draws from CSV...")
        joker_draws = load_joker(joker_csv)
        loto_draws = load_loto649(loto_csv)
        loto540_draws = load_loto540(loto540_csv)

        log(f"Joker: {len(joker_draws) if joker_draws else 0} draws")
        log(f"Loto 6/49: {len(loto_draws) if loto_draws else 0} draws")
        log(f"Loto 5/40: {len(loto540_draws) if loto540_draws else 0} draws")

        joker_ok = bool(joker_draws) and str(joker_draws[-1].date) >= yesterday
        loto_ok = bool(loto_draws) and str(loto_draws[-1].date) >= yesterday
        loto540_ok = bool(loto540_draws) and str(loto540_draws[-1].date) >= yesterday

        if joker_ok or loto_ok or loto540_ok:
            # Log-only informational check; never blocks the workflow.
            missing_side = []
            if joker_draws and not is_side_game_ready(joker_draws[-1], "noroc_plus"):
                missing_side.append("joker/noroc_plus")
            if loto_draws and not is_side_game_ready(loto_draws[-1], "noroc"):
                missing_side.append("loto_649/noroc")
            if loto540_draws and not is_side_game_ready(loto540_draws[-1], "super_noroc"):
                missing_side.append("loto_540/super_noroc")

            if missing_side and attempt < max_retries - 1:
                log(f"Side games not yet published: {missing_side}. Waiting {delay_minutes} more minute(s)...")
                time.sleep(delay_minutes * 60)
                continue

            if missing_side:
                log(f"WARNING: side games still missing after retries: {missing_side}. Proceeding with None values.")

            log("\n=== Recent results found! ===")
            return joker_draws, loto_draws, loto540_draws

        if attempt < max_retries - 1:
            log(f"Results not yet available. Waiting {delay_minutes} minute(s)...")
            time.sleep(delay_minutes * 60)

    log(f"\nWARNING: No results from {yesterday} after {max_retries} attempts")
    log("Proceeding with latest available results...")
    return load_joker(joker_csv), load_loto649(loto_csv), load_loto540(loto540_csv)


def process_game(
    game_key: str,
    emoji: str,
    game_label: str,
    draws,
    picks_dir: Path,
    match_total: int,
) -> tuple[str, str, str, dict[str, dict]]:
    """Process one game. Returns (date, winning_str, message, strategy_results)."""
    if not draws:
        log(f"No {game_label} draws available")
        msg = f"{emoji} *{game_label} Results*\n_No results available_"
        return "N/A", "N/A", msg, {}

    latest = draws[-1]
    winning = latest.main_numbers
    draw_date = str(latest.date)

    if hasattr(latest, "joker"):
        winning_str = f"{', '.join(str(n) for n in winning)} + J{latest.joker}"
    else:
        winning_str = ", ".join(str(n) for n in winning)

    log(f"Latest {game_label}: {draw_date} - {winning_str}")

    strategy_results = check_game_strategies(picks_dir, game_key, winning, match_total)

    msg = build_game_message(
        emoji, game_label, winning_str, draw_date,
        strategy_results, match_total,
    )

    return draw_date, winning_str, msg, strategy_results


def _get_winning_side(game_key: str, draws) -> str | None:
    """Extract winning side-game number from the latest draw."""
    if not draws:
        return None
    latest = draws[-1]
    if game_key == "joker":
        return getattr(latest, "noroc_plus", None)
    if game_key == "loto649":
        return getattr(latest, "noroc", None)
    if game_key == "loto540":
        return getattr(latest, "super_noroc", None)
    return None


def _score_ticket_obj(ticket, winning_main, game_key: str, winning_side: str | None) -> dict:
    """Score a Ticket object against winning numbers. Returns strategy_results entry."""
    best_main = ticket.best_main_match(winning_main)
    total_matches = sum(v.count_main_matches(winning_main) for v in ticket.variants)
    # score_side_game_match uses ticket's game name (loto_649/loto_540), not legacy key
    side_exact, side_digits = score_side_game_match(ticket.game, ticket.side_game_number, winning_side)
    return {
        "results": [
            {"pick": list(v.main_numbers), "matched": sorted(set(v.main_numbers) & set(winning_main)), "count": v.count_main_matches(winning_main)}
            for v in ticket.variants
        ],
        "score": total_matches,
        "best_match": best_main,
        "ticket_id": "",  # filled in per-ticket below
        "variants_count": len(ticket.variants),
        "side_game_match": side_exact,
        "side_game_digits": side_digits,
        "winning_side_game": winning_side or "",
        "builder_name": ticket.strategy,
    }


def main():
    """Main entry point."""
    max_retries = int(os.environ.get("MAX_RETRIES", "3"))
    delay_minutes = int(os.environ.get("RETRY_DELAY_MINUTES", "1"))
    picks_dir = Path(os.environ.get("PICKS_DIR", "picks"))
    output_dir = Path(os.environ.get("RESULTS_DIR", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    joker_draws, loto_draws, loto540_draws = fetch_results_with_retry(
        max_retries=max_retries, delay_minutes=delay_minutes,
    )

    log("\n=== Processing Results ===")

    # Map legacy game_key -> (draws, ticket_game_name)
    game_draws_map = {
        "joker": (joker_draws, "joker"),
        "loto649": (loto_draws, "loto_649"),
        "loto540": (loto540_draws, "loto_540"),
    }

    games = [
        ("joker", "🃏", "JOKER", joker_draws, 5),
        ("loto649", "🎱", "LOTO 6/49", loto_draws, 6),
        ("loto540", "🎯", "LOTO 5/40", loto540_draws, 5),
    ]

    history_path = Path(os.environ.get("HISTORY_CSV", "data/results/history.csv"))

    # Prefer tickets.json over legacy .txt files
    tickets_json_path = picks_dir / "tickets.json"
    tickets_by_game: dict[str, list] = {}
    if tickets_json_path.exists():
        log(f"Loading tickets from {tickets_json_path}")
        ticket_objs = parse_tickets_json(tickets_json_path)
        for t in ticket_objs:
            tickets_by_game.setdefault(t.game, []).append(t)

    all_game_results = []
    history_rows = []
    detail_rows: list[dict] = []
    for game_key, emoji, game_label, draws, match_total in games:
        draws_for_game, ticket_game_name = game_draws_map[game_key]
        json_tickets = tickets_by_game.get(ticket_game_name, [])

        if json_tickets:
            # JSON path: score Ticket objects
            if not draws_for_game:
                log(f"No {game_label} draws available")
                msg = f"{emoji} *{game_label} Results*\n_No results available_"
                draw_date, winning_str, winning_main, winning_side, strategy_results = (
                    "N/A", "N/A", [], None, {}
                )
            else:
                latest = draws_for_game[-1]
                winning_main = latest.main_numbers
                draw_date = str(latest.date)
                if hasattr(latest, "joker"):
                    winning_str = f"{', '.join(str(n) for n in winning_main)} + J{latest.joker}"
                else:
                    winning_str = ", ".join(str(n) for n in winning_main)
                winning_side = _get_winning_side(game_key, draws_for_game)
                log(f"Latest {game_label}: {draw_date} - {winning_str}")

                strategy_results = {}
                scored_tickets = []
                for idx, ticket in enumerate(json_tickets):
                    data = _score_ticket_obj(ticket, winning_main, game_key, winning_side)
                    ticket_id = f"{draw_date}-{game_key}-{ticket.strategy}-{idx}"
                    data["ticket_id"] = ticket_id
                    strat_key = f"{ticket.strategy}_{idx}"
                    strategy_results[strat_key] = data
                    scored_tickets.append((ticket, ticket_id, data["best_match"], data["side_game_match"], data["side_game_digits"]))

                ticket_outcomes = []
                for ticket, ticket_id, best_main, side_exact, side_digits in scored_tickets:
                    variants_rendered = []
                    for v in ticket.variants:
                        hits = set(v.main_numbers) & set(winning_main)
                        main_str = ", ".join(str(n) for n in v.main_numbers)
                        hit_str = f" [{len(hits)} hit]"
                        if ticket.game == "joker" and v.bonus_number is not None:
                            line = f"{main_str} + J{v.bonus_number}{hit_str}"
                        else:
                            line = f"{main_str}{hit_str}"
                        variants_rendered.append(line)
                    ticket_outcomes.append({
                        "ticket_id": ticket_id,
                        "strategy": ticket.strategy,
                        "best_main_match": best_main,
                        "side_game_match": side_exact,
                        "side_game_digits_matched": side_digits,
                        "variants_rendered": variants_rendered,
                        "ticket_side": ticket.side_game_number,
                    })
                    detail_rows.append({
                        "draw_date": draw_date,
                        "game": ticket.game,
                        "ticket_id": ticket_id,
                        "strategy": ticket.strategy,
                        "best_main_match": best_main,
                        "variants": [
                            {"main_numbers": list(v.main_numbers), "bonus_number": v.bonus_number}
                            for v in ticket.variants
                        ],
                        "side_game_number": ticket.side_game_number,
                        "winning_side_game": winning_side or "",
                        "side_game_match": side_exact,
                        "side_game_digits_matched": side_digits,
                        "cost_ron": ticket.cost_ron,
                    })

                msg = _fmt_check_results(
                    game=ticket_game_name,
                    draw_date=draw_date,
                    winning_main=winning_main,
                    winning_side=winning_side,
                    ticket_outcomes=ticket_outcomes,
                )
        else:
            # Legacy .txt path
            draw_date, winning_str, msg, strategy_results = process_game(
                game_key, emoji, game_label, draws, picks_dir, match_total,
            )

        (output_dir / f"{game_key}.txt").write_text(msg, encoding="utf-8")
        all_game_results.append((emoji, game_label, strategy_results, match_total))
        history_rows.append((draw_date, game_key, winning_str, strategy_results, match_total))
        log(f"Wrote {game_key} results message")

    comparison = build_comparison_message(all_game_results)
    (output_dir / "comparison.txt").write_text(comparison, encoding="utf-8")
    log("Wrote comparison message")

    detail_path = Path(os.environ.get("PICKS_DETAIL_PATH", "data/results/picks_detail.jsonl"))
    write_picks_detail(detail_path, detail_rows)

    append_history(history_path, history_rows)
    persist_check_results(
        max_retries=max_retries,
        retry_delay_minutes=delay_minutes,
        history_rows=history_rows,
    )

    log("\n=== Results Checker Completed ===")


if __name__ == "__main__":
    main()
