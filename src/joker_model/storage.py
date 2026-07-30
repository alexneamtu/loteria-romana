import csv
from pathlib import Path

from .models import JokerDraw

_FIELDNAMES = ["date"] + [f"main_{i}" for i in range(1, 6)] + ["joker", "noroc_plus"]


def load_draws(path: Path) -> list[JokerDraw]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            main = [int(row[f"main_{i}"]) for i in range(1, 6)]
            noroc_plus = row.get("noroc_plus") or None
            rows.append(
                JokerDraw(
                    row["date"],
                    sorted(main),
                    int(row["joker"]),
                    noroc_plus,
                )
            )
    return sorted(rows, key=lambda d: d.date)


def append_draws(path: Path, draws: list[JokerDraw]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES, lineterminator="\n")
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
                "joker": draw.joker,
                "noroc_plus": draw.noroc_plus or "",
            })
    return len(draws)
