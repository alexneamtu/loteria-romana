import re
from datetime import datetime

from .models import Loto540Draw


def _normalize_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")


def parse_loto_540_results(html: str) -> list[Loto540Draw]:
    """Parse Loto 5/40 results from HTML.

    The game draws 6 numbers from 1-40.
    """
    draws = []

    # Pattern for main 5/40 draw dates
    date_pattern = re.compile(
        r"Detalii castiguri\s+la 5/40\s+din\s+<span>(\d{2}\.\d{2}\.\d{4})</span>"
    )

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

        draws.append(Loto540Draw(date, main))

    return sorted(draws, key=lambda d: d.date)
