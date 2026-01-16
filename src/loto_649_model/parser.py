import re
from datetime import datetime

from .models import Loto649Draw


def _normalize_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")


def _extract_noroc_digits(block: str) -> int | None:
    span_match = re.search(r"numere-extrase-noroc[^>]*>\s*<span>(\d+)</span>", block)
    if span_match:
        return int(span_match.group(1))
    digits = re.findall(r"span-nr-\d+\">(\d)</span>", block)
    if digits:
        return int("".join(digits))
    return None


def parse_loto_649_results(html: str) -> list[Loto649Draw]:
    blocks = re.split(r"<div class=\"rezultate-extrageri-content[^\"]*\">", html)
    date_pattern = re.compile(r"Detalii castiguri[^<]*<span>(\d{2}\.\d{2}\.\d{4})</span>", re.IGNORECASE)

    main_by_date: dict[str, list[int]] = {}
    noroc_by_date: dict[str, int] = {}

    for block in blocks:
        date_match = date_pattern.search(block)
        if not date_match:
            continue
        date = _normalize_date(date_match.group(1))

        numbers = [int(n) for n in re.findall(r"/bile/(\d{1,2})\.png", block)]
        if len(numbers) >= 6:
            main_by_date[date] = sorted(numbers[-6:])

        noroc = _extract_noroc_digits(block)
        if noroc is not None:
            noroc_by_date[date] = noroc

    draws = []
    for date, main in sorted(main_by_date.items()):
        noroc = noroc_by_date.get(date)
        if noroc is None:
            continue
        draws.append(Loto649Draw(date, main, noroc))

    return draws
