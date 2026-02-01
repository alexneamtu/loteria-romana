#!/usr/bin/env python3
"""Check lottery results and compare against saved picks from multiple strategies."""

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


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


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
    for path in sorted(picks_dir.glob(f"{game_prefix}_*.txt")):
        strategy = path.stem.replace(f"{game_prefix}_", "")
        strategies[strategy] = path
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

    games = [
        ("joker", "🃏", "JOKER", joker_draws, 5),
        ("loto649", "🎱", "LOTO 6/49", loto_draws, 6),
        ("loto540", "🎯", "LOTO 5/40", loto540_draws, 5),
    ]

    all_game_results = []
    for game_key, emoji, game_label, draws, match_total in games:
        draw_date, winning_str, msg, strategy_results = process_game(
            game_key, emoji, game_label, draws, picks_dir, match_total,
        )
        (output_dir / f"{game_key}.txt").write_text(msg, encoding="utf-8")
        all_game_results.append((emoji, game_label, strategy_results, match_total))
        log(f"Wrote {game_key} results message")

    comparison = build_comparison_message(all_game_results)
    (output_dir / "comparison.txt").write_text(comparison, encoding="utf-8")
    log("Wrote comparison message")

    log("\n=== Results Checker Completed ===")


if __name__ == "__main__":
    main()
