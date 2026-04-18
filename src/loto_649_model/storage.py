import csv
from pathlib import Path

from .models import Loto649Draw

_FIELDNAMES = ["date"] + [f"main_{i}" for i in range(1, 7)] + ["noroc"]


def load_draws(path: Path) -> list[Loto649Draw]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            main = [int(row[f"main_{i}"]) for i in range(1, 7)]
            noroc = row.get("noroc") or None
            rows.append(Loto649Draw(row["date"], sorted(main), noroc))
    return rows


def append_draws(path: Path, draws: list[Loto649Draw]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        if not exists:
            writer.writeheader()
        for draw in draws:
            writer.writerow({
                "date": draw.date,
                "main_1": draw.main_numbers[0],
                "main_2": draw.main_numbers[1],
                "main_3": draw.main_numbers[2],
                "main_4": draw.main_numbers[3],
                "main_5": draw.main_numbers[4],
                "main_6": draw.main_numbers[5],
                "noroc": draw.noroc or "",
            })
    return len(draws)
