"""Backfill draw history from the loto.ro monthly results archive.

The results pages only list the last handful of draws, but the same URLs
accept a POST with select-year/select-month and render that month instead.
Fetching every month from the newest draw on file up to today closes any
gap left by a stalled or failed run.
"""

import argparse
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from joker_model import storage as joker_storage
from joker_model.parser import parse_joker_results
from loto_540_model import storage as loto_540_storage
from loto_540_model.parser import parse_loto_540_results
from loto_649_model import storage as loto_649_storage
from loto_649_model.parser import parse_loto_649_results

BASE = "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/"

GAMES = {
    "joker": (
        BASE + "joker_si_noroc_plus/rezultate_extrageri.html",
        parse_joker_results,
        joker_storage,
        Path("data/clean/joker_draws.csv"),
    ),
    "loto-649": (
        BASE + "649_si_noroc/rezultate_extragere.html",
        parse_loto_649_results,
        loto_649_storage,
        Path("data/clean/loto_649_draws.csv"),
    ),
    "loto-540": (
        BASE + "540_si_super_noroc/rezultate_extrageri.html",
        parse_loto_540_results,
        loto_540_storage,
        Path("data/clean/loto_540_draws.csv"),
    ),
}


def _fetch_month(url: str, year: int, month: int) -> str:
    data = urllib.parse.urlencode({"select-year": year, "select-month": month}).encode()
    request = urllib.request.Request(url, data=data)
    return urllib.request.urlopen(request, timeout=60).read().decode("utf-8", "ignore")


def _to_month(text: str) -> tuple[int, int]:
    year, month = text.split("-")[:2]
    return int(year), int(month)


def _months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    months = []
    year, month = start
    while (year, month) <= end:
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def backfill(name: str, since: str | None, today: date, fetcher=_fetch_month) -> tuple[int, int]:
    url, parse, storage, csv_path = GAMES[name]
    draws = storage.load_draws(csv_path)
    known = {d.date for d in draws}
    if since:
        start = _to_month(since)
    elif known:
        start = _to_month(max(known))
    else:
        start = (today.year, today.month)

    months = _months(start, (today.year, today.month))
    added = 0
    for year, month in months:
        for draw in parse(fetcher(url, year, month)):
            if draw.date not in known:
                known.add(draw.date)
                draws.append(draw)
                added += 1

    draws.sort(key=lambda d: d.date)
    # Rewrite rather than append: the file gets a full header (side-game
    # columns included) and stays date-ordered.
    csv_path.unlink(missing_ok=True)
    storage.append_draws(csv_path, draws)
    return added, len(months)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        metavar="YYYY-MM",
        help="first month to fetch (default: month of the newest draw on file)",
    )
    for name in GAMES:
        parser.add_argument(f"--{name}", action="store_true", help=f"backfill {name} only")
    args = parser.parse_args()

    selected = [name for name in GAMES if getattr(args, name.replace("-", "_"))] or list(GAMES)
    today = date.today()
    for name in selected:
        added, months = backfill(name, args.since, today)
        print(f"{name}: +{added} draws over {months} month(s)")


if __name__ == "__main__":
    main()
