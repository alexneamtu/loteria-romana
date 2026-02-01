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


GAME_CONFIGS = {
    "joker": {"label": "JOKER", "match_total": 5},
    "loto649": {"label": "LOTO", "match_total": 6},
    "loto540": {"label": "LOTO540", "match_total": 5},
}


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


def format_result(pick: list[int], matched: list[int], count: int, total: int) -> str:
    """Format a single result line."""
    pick_str = ", ".join(str(n) for n in pick)
    if count == 0:
        return f"  {pick_str} - no matches"
    elif count <= 2:
        matched_str = ", ".join(str(n) for n in matched)
        return f"  {pick_str} - {count}/{total} [{matched_str}]"
    else:
        matched_str = ", ".join(str(n) for n in matched)
        return f"  {pick_str} - {count}/{total} [{matched_str}]"


def find_strategy_files(picks_dir: Path, game_prefix: str) -> dict[str, Path]:
    """Find all strategy files for a game. Returns {strategy_name: file_path}."""
    strategies = {}
    for path in sorted(picks_dir.glob(f"{game_prefix}_*.txt")):
        strategy = path.stem.replace(f"{game_prefix}_", "")
        strategies[strategy] = path
    return strategies


def score_strategy_results(results: list[dict]) -> int:
    """Score a strategy's results: sum of all match counts."""
    return sum(r["count"] for r in results)


def check_game_strategies(
    picks_dir: Path,
    game_prefix: str,
    winning_numbers: list[int],
    match_total: int,
) -> dict[str, dict]:
    """Check all strategy files for a game against winning numbers.

    Returns {strategy: {"results": [...], "score": int, "best_match": int}}.
    """
    strategy_files = find_strategy_files(picks_dir, game_prefix)
    strategy_results = {}

    for strategy, path in strategy_files.items():
        content = path.read_text().strip()
        if not content:
            continue
        picks = parse_picks(content)
        results = check_matches(picks, winning_numbers)
        score = score_strategy_results(results)
        best_match = max((r["count"] for r in results), default=0)
        strategy_results[strategy] = {
            "results": results,
            "score": score,
            "best_match": best_match,
        }

    return strategy_results


def format_strategy_comparison(
    strategy_results: dict[str, dict],
    match_total: int,
) -> str:
    """Format a comparison summary of all strategies for a game."""
    if not strategy_results:
        return "No strategy picks found"

    ranked = sorted(
        strategy_results.items(),
        key=lambda item: (item[1]["score"], item[1]["best_match"]),
        reverse=True,
    )

    lines = []
    for strategy, data in ranked:
        best = data["best_match"]
        score = data["score"]
        marker = "*" if best >= 3 else ""
        lines.append(f"  {marker}{strategy}: {score} total matches (best {best}/{match_total})")

    return "\n".join(lines)


def format_strategy_detail(
    strategy_results: dict[str, dict],
    match_total: int,
) -> str:
    """Format detailed results for each strategy."""
    if not strategy_results:
        return "No strategy picks found"

    ranked = sorted(
        strategy_results.items(),
        key=lambda item: (item[1]["score"], item[1]["best_match"]),
        reverse=True,
    )

    sections = []
    for strategy, data in ranked:
        section_lines = [f"[{strategy}]"]
        for r in data["results"]:
            section_lines.append(
                format_result(r["pick"], r["matched"], r["count"], match_total)
            )
        sections.append("\n".join(section_lines))

    return "\n".join(sections)


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

        # Clear cache to force fresh fetch
        if joker_cache.exists():
            log("Clearing Joker cache...")
            joker_cache.unlink()
        if loto_cache.exists():
            log("Clearing Loto 6/49 cache...")
            loto_cache.unlink()
        if loto540_cache.exists():
            log("Clearing Loto 5/40 cache...")
            loto540_cache.unlink()

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

        # Load and check if we have recent results
        log("Loading draws from CSV...")
        joker_draws = load_joker(joker_csv)
        loto_draws = load_loto649(loto_csv)
        loto540_draws = load_loto540(loto540_csv)

        log(f"Joker draws loaded: {len(joker_draws) if joker_draws else 0}")
        log(f"Loto 6/49 draws loaded: {len(loto_draws) if loto_draws else 0}")
        log(f"Loto 5/40 draws loaded: {len(loto540_draws) if loto540_draws else 0}")

        if joker_draws:
            log(f"Latest Joker draw date: {joker_draws[-1].date}")
        if loto_draws:
            log(f"Latest Loto 6/49 draw date: {loto_draws[-1].date}")
        if loto540_draws:
            log(f"Latest Loto 5/40 draw date: {loto540_draws[-1].date}")

        joker_ok = bool(joker_draws) and str(joker_draws[-1].date) >= yesterday
        loto_ok = bool(loto_draws) and str(loto_draws[-1].date) >= yesterday
        loto540_ok = bool(loto540_draws) and str(loto540_draws[-1].date) >= yesterday

        log(f"Joker results recent enough: {joker_ok}")
        log(f"Loto 6/49 results recent enough: {loto_ok}")
        log(f"Loto 5/40 results recent enough: {loto540_ok}")

        if joker_ok or loto_ok or loto540_ok:
            log("\n=== Recent results found! Proceeding with comparison ===")
            return joker_draws, loto_draws, loto540_draws

        if attempt < max_retries - 1:
            log(f"Results not yet available for {yesterday}. Waiting {delay_minutes} minute(s)...")
            time.sleep(delay_minutes * 60)

    log(f"\nWARNING: Could not find results from {yesterday} after {max_retries} attempts")
    log("Proceeding with latest available results...")
    return load_joker(joker_csv), load_loto649(loto_csv), load_loto540(loto540_csv)


def output_game_results(
    game_key: str,
    label: str,
    draws,
    picks_dir: Path,
    match_total: int,
):
    """Output results for a single game, including multi-strategy comparison."""
    if not draws:
        log(f"No {label} draws available")
        print(f"{label}_DATE=N/A")
        print(f"{label}_WINNING=No results available")
        print(f"{label}_RESULTS=Could not fetch {label} results")
        print(f"{label}_COMPARISON=N/A")
        return

    latest = draws[-1]
    winning = latest.main_numbers
    log(f"Latest {label}: {latest.date} - {winning}")
    print(f"{label}_DATE={latest.date}")

    if hasattr(latest, "joker"):
        print(f"{label}_WINNING={', '.join(str(n) for n in winning)} + J{latest.joker}")
    else:
        print(f"{label}_WINNING={', '.join(str(n) for n in winning)}")

    # Check legacy picks file (backward compatibility)
    legacy_file = picks_dir / f"{game_key}.txt"
    if legacy_file.exists():
        content = legacy_file.read_text().strip()
        if content:
            picks = parse_picks(content)
            results = check_matches(picks, winning)
            output_lines = []
            for r in results:
                output_lines.append(format_result(r["pick"], r["matched"], r["count"], match_total))
            print(f"{label}_RESULTS=" + "\\n".join(output_lines))
        else:
            print(f"{label}_RESULTS=No picks found to compare")
    else:
        print(f"{label}_RESULTS=No picks found to compare")

    # Multi-strategy comparison
    strategy_results = check_game_strategies(picks_dir, game_key, winning, match_total)
    if strategy_results:
        comparison = format_strategy_comparison(strategy_results, match_total)
        detail = format_strategy_detail(strategy_results, match_total)
        log(f"\n{label} Strategy Comparison:\n{comparison}")
        log(f"\n{label} Strategy Detail:\n{detail}")
        print(f"{label}_COMPARISON=" + comparison.replace("\n", "\\n"))
        print(f"{label}_DETAIL=" + detail.replace("\n", "\\n"))
    else:
        print(f"{label}_COMPARISON=No strategy picks found")
        print(f"{label}_DETAIL=No strategy picks found")


def main():
    """Main entry point."""
    max_retries = int(os.environ.get("MAX_RETRIES", "3"))
    delay_minutes = int(os.environ.get("RETRY_DELAY_MINUTES", "1"))
    picks_dir = Path(os.environ.get("PICKS_DIR", "picks"))

    joker_draws, loto_draws, loto540_draws = fetch_results_with_retry(
        max_retries=max_retries, delay_minutes=delay_minutes
    )

    log("\n=== Processing Results ===")

    output_game_results("joker", "JOKER", joker_draws, picks_dir, match_total=5)
    output_game_results("loto649", "LOTO", loto_draws, picks_dir, match_total=6)
    output_game_results("loto540", "LOTO540", loto540_draws, picks_dir, match_total=5)

    log("\n=== Results Checker Completed ===")


if __name__ == "__main__":
    main()
