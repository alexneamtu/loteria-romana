import html
import re
from datetime import datetime

MONTHS = {
    "ianuarie": 1,
    "februarie": 2,
    "martie": 3,
    "aprilie": 4,
    "mai": 5,
    "iunie": 6,
    "iulie": 7,
    "august": 8,
    "septembrie": 9,
    "octombrie": 10,
    "noiembrie": 11,
    "decembrie": 12,
}


def extract_years(html_text: str) -> list[int]:
    years = {int(y) for y in re.findall(r"\?Y=(\d{4})", html_text)}
    return sorted(years)


def _normalize_date(text: str) -> str:
    clean = html.unescape(text)
    if "," in clean:
        clean = clean.split(",", 1)[1].strip()
    match = re.search(r"(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})", clean, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unable to parse date: {text}")
    day = int(match.group(1))
    month_name = match.group(2).lower()
    year = int(match.group(3))
    month = MONTHS[month_name]
    return datetime(year, month, day).strftime("%Y-%m-%d")


def parse_archive_draws(html_text: str, numbers_count: int) -> list[tuple[str, list[int]]]:
    tables = re.findall(r"<table[^>]*class=bilet[^>]*>.*?</table>", html_text, re.S | re.I)
    archive_table = None
    for table in tables:
        if re.search(r"Data<BR>extragerii", table, re.I):
            archive_table = table
            break
    if not archive_table:
        return []

    rows = re.findall(r"<tr[^>]*>.*?</tr>", archive_table, re.S | re.I)
    draws = []
    for row in rows:
        date_match = re.search(r"<td[^>]*class=odd[^>]*nowrap[^>]*>(.*?)</td>", row, re.S | re.I)
        if not date_match:
            continue
        date = _normalize_date(date_match.group(1))
        numbers = [
            int(n)
            for n in re.findall(
                r"<td[^>]*class=(?:\"|')?(?:odd_rounded|red_rounded)(?:\"|')?[^>]*>(\d+)</td>",
                row,
                re.I,
            )
        ]
        if len(numbers) < numbers_count:
            continue
        draws.append((date, numbers[:numbers_count]))
    return draws
