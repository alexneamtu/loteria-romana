import re
from datetime import datetime

from .models import Loto540Draw


def _normalize_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")


def parse_loto_540_results(html: str) -> list[Loto540Draw]:
    """Parse Loto 5/40 results from HTML.

    The game draws 6 numbers from 1-40. Super Noroc is 6 single-digit
    images in /bile/super-noroc/ when present.
    """
    draws = []
    date_pattern = re.compile(
        r"Detalii castiguri\s+la 5/40\s+din\s+<span>(\d{2}\.\d{2}\.\d{4})</span>"
    )

    for match in date_pattern.finditer(html):
        window_start = max(0, match.start() - 2000)
        window = html[window_start:match.start()]

        # Trim back to the nearest opening <div so a late draw's window
        # doesn't catch earlier draws' super-noroc digits (the main-number
        # regex is tolerant of extras because it takes the last 6, but
        # super-noroc requires exactly 6 matches within the current draw).
        last_div = window.rfind("<div")
        if last_div != -1:
            window = window[last_div:]

        main_nums = [int(n) for n in re.findall(r"/bile/(\d{1,2})\.png", window)]
        super_noroc_digits = re.findall(r"/bile/super-noroc/(\d)\.png", window)

        if len(main_nums) < 6:
            continue

        date = _normalize_date(match.group(1))
        main = sorted(main_nums[-6:])
        super_noroc = (
            "".join(super_noroc_digits) if len(super_noroc_digits) == 6 else None
        )
        draws.append(Loto540Draw(date, main, super_noroc))

    return sorted(draws, key=lambda d: d.date)
