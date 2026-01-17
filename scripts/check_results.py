#!/usr/bin/env python3
"""Check lottery results and compare against saved picks."""

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


def format_result(pick: list[int], matched: list[int], count: int, total: int) -> str:
    """Format a single result line."""
    pick_str = ", ".join(str(n) for n in pick)
    if count == 0:
        return f"❌ {pick_str} - no matches"
    elif count <= 2:
        matched_str = ", ".join(str(n) for n in matched)
        return f"🔸 {pick_str} - {count}/{total} matched: [{matched_str}]"
    else:
        matched_str = ", ".join(str(n) for n in matched)
        return f"✅ {pick_str} - {count}/{total} matched: [{matched_str}]"


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


def main():
    """Main entry point."""
    # Get settings from environment or use defaults
    max_retries = int(os.environ.get("MAX_RETRIES", "3"))
    delay_minutes = int(os.environ.get("RETRY_DELAY_MINUTES", "1"))
    picks_dir = Path(os.environ.get("PICKS_DIR", "picks"))

    joker_draws, loto_draws, loto540_draws = fetch_results_with_retry(
        max_retries=max_retries, delay_minutes=delay_minutes
    )

    log("\n=== Processing Results ===")

    if not joker_draws:
        log("No Joker draws available")
        print("JOKER_DATE=N/A")
        print("JOKER_WINNING=No results available")
        print("JOKER_RESULTS=Could not fetch Joker results")
    else:
        latest_joker = joker_draws[-1]
        log(f"Latest Joker: {latest_joker.date} - {latest_joker.main_numbers} + J{latest_joker.joker}")
        print(f"JOKER_DATE={latest_joker.date}")
        print(f"JOKER_WINNING={', '.join(str(n) for n in latest_joker.main_numbers)} + J{latest_joker.joker}")

    if not loto_draws:
        log("No Loto 6/49 draws available")
        print("LOTO_DATE=N/A")
        print("LOTO_WINNING=No results available")
        print("LOTO_RESULTS=Could not fetch Loto 6/49 results")
    else:
        latest_loto = loto_draws[-1]
        log(f"Latest Loto 6/49: {latest_loto.date} - {latest_loto.main_numbers}")
        print(f"LOTO_DATE={latest_loto.date}")
        print(f"LOTO_WINNING={', '.join(str(n) for n in latest_loto.main_numbers)}")

    if not loto540_draws:
        log("No Loto 5/40 draws available")
        print("LOTO540_DATE=N/A")
        print("LOTO540_WINNING=No results available")
        print("LOTO540_RESULTS=Could not fetch Loto 5/40 results")
    else:
        latest_540 = loto540_draws[-1]
        log(f"Latest Loto 5/40: {latest_540.date} - {latest_540.main_numbers}")
        print(f"LOTO540_DATE={latest_540.date}")
        print(f"LOTO540_WINNING={', '.join(str(n) for n in latest_540.main_numbers)}")

    # Read saved picks
    log("\n=== Loading Saved Picks ===")
    joker_picks_file = picks_dir / "joker.txt"
    loto_picks_file = picks_dir / "loto649.txt"
    loto540_picks_file = picks_dir / "loto540.txt"

    log(f"Joker picks file exists: {joker_picks_file.exists()}")
    log(f"Loto 6/49 picks file exists: {loto_picks_file.exists()}")
    log(f"Loto 5/40 picks file exists: {loto540_picks_file.exists()}")

    if joker_picks_file.exists():
        content = joker_picks_file.read_text()
        log(f"Joker picks content:\n{content}")
        if joker_draws:
            joker_picks = parse_picks(content)
            log(f"Parsed {len(joker_picks)} Joker picks")
            joker_results = check_matches(joker_picks, joker_draws[-1].main_numbers)

            output_lines = []
            for r in joker_results:
                output_lines.append(format_result(r["pick"], r["matched"], r["count"], 5))
            print("JOKER_RESULTS=" + "\\n".join(output_lines))
        else:
            print("JOKER_RESULTS=No draws to compare against")
    else:
        print("JOKER_RESULTS=No picks found to compare")

    if loto_picks_file.exists():
        content = loto_picks_file.read_text()
        log(f"Loto 6/49 picks content:\n{content}")
        if loto_draws:
            loto_picks = parse_picks(content)
            log(f"Parsed {len(loto_picks)} Loto 6/49 picks")
            loto_results = check_matches(loto_picks, loto_draws[-1].main_numbers)

            output_lines = []
            for r in loto_results:
                output_lines.append(format_result(r["pick"], r["matched"], r["count"], 6))
            print("LOTO_RESULTS=" + "\\n".join(output_lines))
        else:
            print("LOTO_RESULTS=No draws to compare against")
    else:
        print("LOTO_RESULTS=No picks found to compare")

    if loto540_picks_file.exists():
        content = loto540_picks_file.read_text()
        log(f"Loto 5/40 picks content:\n{content}")
        if loto540_draws:
            loto540_picks = parse_picks(content)
            log(f"Parsed {len(loto540_picks)} Loto 5/40 picks")
            # In 5/40, player picks 5 numbers, lottery draws 6
            loto540_results = check_matches(loto540_picks, loto540_draws[-1].main_numbers)

            output_lines = []
            for r in loto540_results:
                # Show matches out of 5 (player's picks)
                output_lines.append(format_result(r["pick"], r["matched"], r["count"], 5))
            print("LOTO540_RESULTS=" + "\\n".join(output_lines))
        else:
            print("LOTO540_RESULTS=No draws to compare against")
    else:
        print("LOTO540_RESULTS=No picks found to compare")

    log("\n=== Results Checker Completed ===")


if __name__ == "__main__":
    main()
