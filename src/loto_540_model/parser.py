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
    blocks = re.split(r"<div class=\"[^\"]*rezultate-extrageri-content[^\"]*\">", html)
    date_pattern = re.compile(
        r"Detalii castiguri[^<]*<span>(\d{2}\.\d{2}\.\d{4})</span>", re.IGNORECASE
    )

    main_by_date: dict[str, list[int]] = {}
    super_noroc_by_date: dict[str, str] = {}

    for block in blocks:
        date_match = date_pattern.search(block)
        if not date_match:
            continue
        date = _normalize_date(date_match.group(1))

        numbers = [int(n) for n in re.findall(r"/bile/(\d{1,2})\.png", block)]
        if len(numbers) >= 6:
            main_by_date[date] = sorted(numbers[-6:])

        # Super Noroc lives in its own block, tagged with the same draw date.
        digits = re.findall(r"/bile/super-noroc/(\d)\.png", block)
        if len(digits) == 6:
            super_noroc_by_date[date] = "".join(digits)

    return [
        Loto540Draw(date, main, super_noroc_by_date.get(date))
        for date, main in sorted(main_by_date.items())
    ]
