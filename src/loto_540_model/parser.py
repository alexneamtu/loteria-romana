import re
from datetime import datetime

from .models import Loto540Draw


def _normalize_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")


def parse_loto_540_results(html: str) -> list[Loto540Draw]:
    """Parse Loto 5/40 results from HTML.

    The game draws 6 numbers from 1-40. Super Noroc is a 6-digit bonus.
    """
    draws = []

    # Pattern for main 5/40 draw dates
    date_pattern = re.compile(
        r"Detalii castiguri\s+la 5/40\s+din\s+<span>(\d{2}\.\d{2}\.\d{4})</span>"
    )

    # Pattern for Super Noroc numbers and dates
    super_noroc_pattern = re.compile(
        r'numere-extrase-noroc"><span>(\d+)</span>.*?'
        r"Detalii castiguri la super noroc din\s+<span>(\d{2}\.\d{2}\.\d{4})</span>",
        re.DOTALL
    )

    # Build a map of date -> super_noroc
    super_noroc_map = {}
    for match in super_noroc_pattern.finditer(html):
        noroc_num = int(match.group(1))
        noroc_date = _normalize_date(match.group(2))
        super_noroc_map[noroc_date] = noroc_num

    # Parse main draws
    for match in date_pattern.finditer(html):
        window_start = max(0, match.start() - 2000)
        window = html[window_start:match.start()]

        # Extract ball numbers from images
        main_nums = [int(n) for n in re.findall(r"/bile/(\d{1,2})\.png", window)]

        if len(main_nums) < 6:
            continue

        date = _normalize_date(match.group(1))
        main = sorted(main_nums[-6:])  # Take last 6 numbers found
        super_noroc = super_noroc_map.get(date)

        draws.append(Loto540Draw(date, main, super_noroc))

    return sorted(draws, key=lambda d: d.date)
