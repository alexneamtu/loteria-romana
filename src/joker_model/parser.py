import re
from datetime import datetime

from .models import JokerDraw


def _normalize_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")


def parse_joker_results(html: str) -> list[JokerDraw]:
    draws = []
    date_pattern = re.compile(r"Detalii castiguri\s+la joker\s+din\s+<span>(\d{2}\.\d{2}\.\d{4})</span>")

    for match in date_pattern.finditer(html):
        window_start = max(0, match.start() - 2000)
        window = html[window_start:match.start()]

        main_nums = [int(n) for n in re.findall(r"/bile/(\d{1,2})\.png", window)]
        joker_nums = [int(n) for n in re.findall(r"/bile/joker/(\d{1,2})\.png", window)]
        noroc_plus_nums = re.findall(r"/bile/noroc-plus/(\d{1,2})\.png", window)

        if len(main_nums) < 5 or not joker_nums:
            continue

        main = sorted(main_nums[-5:])
        joker = joker_nums[-1]
        noroc_plus = f"NP{int(noroc_plus_nums[-1]):02d}" if noroc_plus_nums else None
        draws.append(JokerDraw(_normalize_date(match.group(1)), main, joker, noroc_plus))

    return draws
