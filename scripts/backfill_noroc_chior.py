import argparse
import csv
from pathlib import Path
from urllib.request import urlopen

from joker_model.fetch import update_dataset as update_joker
from joker_model.models import JokerDraw
from loto_649_model.fetch import update_dataset as update_649
from loto_649_model.models import Loto649Draw
from loto_540_model.fetch import update_dataset as update_540
from loto_540_model.models import Loto540Draw
from shared.noroc_chior import extract_years, parse_archive_draws

JOKER_ARCHIVE = "http://noroc-chior.ro/Loto/joker/arhiva-rezultate.php"
LOTO_649_ARCHIVE = "http://noroc-chior.ro/Loto/6-din-49/arhiva-rezultate.php"
LOTO_540_ARCHIVE = "http://noroc-chior.ro/Loto/5-din-40/arhiva-rezultate.php"

JOKER_RESULTS = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html"
LOTO_649_RESULTS = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/649_si_noroc/rezultate_extragere.html"
LOTO_540_RESULTS = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/540_si_super_noroc/rezultate_extrageri.html"

JOKER_CACHE = Path("data/raw/joker_results.html")
LOTO_649_CACHE = Path("data/raw/loto_649_results.html")
LOTO_540_CACHE = Path("data/raw/loto_540_results.html")

JOKER_CSV = Path("data/clean/joker_draws.csv")
LOTO_649_CSV = Path("data/clean/loto_649_draws.csv")
LOTO_540_CSV = Path("data/clean/loto_540_draws.csv")


def _fetch(url: str) -> str:
    return urlopen(url).read().decode("utf-8", "ignore")


def _write_draws(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collapse_by_date(draws: list[tuple[str, list[int]]]) -> tuple[list[tuple[str, list[int]]], int]:
    by_date: dict[str, list[int]] = {}
    conflicts = 0
    for date, numbers in draws:
        if date in by_date and by_date[date] != numbers:
            conflicts += 1
            continue
        by_date[date] = numbers
    ordered = [(date, by_date[date]) for date in sorted(by_date.keys())]
    return ordered, conflicts


def _rebuild_game(base_url: str, csv_path: Path, build_row, numbers_count: int) -> tuple[int, int]:
    years = extract_years(_fetch(base_url))
    all_draws: list[tuple[str, list[int]]] = []
    for year in years:
        html = _fetch(f"{base_url}?Y={year}")
        all_draws.extend(parse_archive_draws(html, numbers_count=numbers_count))

    collapsed, conflicts = _collapse_by_date(all_draws)
    rows = [build_row(date, numbers) for date, numbers in collapsed]
    if rows:
        _write_draws(csv_path, list(rows[0].keys()), rows)
    return len(rows), conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild CSVs from noroc-chior archives")
    parser.add_argument("--joker", action="store_true", help="Rebuild Joker")
    parser.add_argument("--loto-649", action="store_true", help="Rebuild Loto 6/49")
    parser.add_argument("--loto-540", action="store_true", help="Rebuild Loto 5/40")
    args = parser.parse_args()

    if not any([args.joker, args.loto_649, args.loto_540]):
        args.joker = args.loto_649 = args.loto_540 = True

    if args.joker:
        count, conflicts = _rebuild_game(
            JOKER_ARCHIVE,
            JOKER_CSV,
            lambda d, nums: {
                "date": d,
                "main_1": nums[0],
                "main_2": nums[1],
                "main_3": nums[2],
                "main_4": nums[3],
                "main_5": nums[4],
                "joker": nums[5],
            },
            numbers_count=6,
        )
        print(f"joker: rebuilt {count} draws (conflicts skipped: {conflicts})")

    if args.loto_649:
        count, conflicts = _rebuild_game(
            LOTO_649_ARCHIVE,
            LOTO_649_CSV,
            lambda d, nums: {
                "date": d,
                "main_1": nums[0],
                "main_2": nums[1],
                "main_3": nums[2],
                "main_4": nums[3],
                "main_5": nums[4],
                "main_6": nums[5],
            },
            numbers_count=6,
        )
        print(f"loto_649: rebuilt {count} draws (conflicts skipped: {conflicts})")

    if args.loto_540:
        count, conflicts = _rebuild_game(
            LOTO_540_ARCHIVE,
            LOTO_540_CSV,
            lambda d, nums: {
                "date": d,
                "main_1": nums[0],
                "main_2": nums[1],
                "main_3": nums[2],
                "main_4": nums[3],
                "main_5": nums[4],
                "main_6": nums[5],
            },
            numbers_count=6,
        )
        print(f"loto_540: rebuilt {count} draws (conflicts skipped: {conflicts})")

    if args.joker:
        update_joker(JOKER_RESULTS, JOKER_CACHE, JOKER_CSV)
    if args.loto_649:
        update_649(LOTO_649_RESULTS, LOTO_649_CACHE, LOTO_649_CSV)
    if args.loto_540:
        update_540(LOTO_540_RESULTS, LOTO_540_CACHE, LOTO_540_CSV)


if __name__ == "__main__":
    main()
