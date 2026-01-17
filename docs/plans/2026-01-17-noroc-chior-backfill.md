# Noroc-Chior Rebuild Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a one-time rebuild script that fetches noroc-chior archive pages, rewrites local CSVs from that archive (main numbers only, plus Joker bonus), then runs the existing loto.ro update to append any newest draws not present in the archive.

**Architecture:** Create a parser module in `src/shared/noroc_chior.py` for year extraction, Romanian date normalization, and archive table parsing. Implement `scripts/backfill_noroc_chior.py` to rebuild CSVs from noroc-chior and then call the existing `update_dataset()` functions to append recent draws from loto.ro.

**Tech Stack:** Python stdlib (`re`, `html`, `urllib.request`, `datetime`, `pathlib`, `csv`), existing models/storage helpers.

### Task 1: Add fixtures for noroc-chior parsing

**Files:**
- Create: `tests/fixtures/noroc_chior_years.html`
- Create: `tests/fixtures/noroc_chior_649_2024.html`
- Create: `tests/fixtures/noroc_chior_540_2024.html`
- Create: `tests/fixtures/noroc_chior_joker_2024.html`

**Step 1: Write year list fixture**

```html
<h1>Arhiva rezultatelor</h1>
<center>
  Filtrare dupa an: [ <a href='?Y=1993'>1993</a> ][ <a href='?Y=2024'>2024</a> ]
</center>
```

**Step 2: Write 6/49 archive table fixture**

```html
<table class=bilet>
  <tr>
    <td class=th_1 rowspan=2>Data<BR>extragerii</td>
    <td class=th_1 rowspan=2 colspan=6>Numerele<BR>extrase</td>
  </tr>
  <tr>
    <td class=th_2>Numar<br>castiguri</td>
  </tr>
  <tr>
    <td class=odd nowrap>Ma, 31 decembrie 2024</td>
    <td class=odd_rounded>27</td>
    <td class=odd_rounded>5</td>
    <td class=odd_rounded>49</td>
    <td class=odd_rounded>19</td>
    <td class=odd_rounded>11</td>
    <td class=odd_rounded>44</td>
    <td class=odd>REPORT</td>
  </tr>
</table>
```

**Step 3: Write 5/40 archive table fixture**

```html
<table class=bilet>
  <tr>
    <td class=th_1 rowspan=2>Data<BR>extragerii</td>
    <td class=th_1 rowspan=2 colspan=6>Numerele<BR>extrase</td>
  </tr>
  <tr>
    <td class=th_2>Numar<br>castiguri</td>
  </tr>
  <tr>
    <td class=odd nowrap>Ma, 31 decembrie 2024</td>
    <td class=odd_rounded>21</td>
    <td class=odd_rounded>16</td>
    <td class=odd_rounded>2</td>
    <td class=odd_rounded>3</td>
    <td class=odd_rounded>14</td>
    <td class=odd_rounded>17</td>
    <td class=odd>REPORT</td>
  </tr>
</table>
```

**Step 4: Write Joker archive table fixture**

```html
<table class=bilet>
  <tr>
    <td class="th_1 joker_class_01" rowspan=2>Data<BR>extragerii</td>
    <td class="th_1 joker_class_02" colspan=6>Numerele extrase</td>
  </tr>
  <tr>
    <td class="th_2" colspan=5>Primul set</td>
    <td class="th_2">Joker</td>
  </tr>
  <tr>
    <td class=odd nowrap>Ma, 31 decembrie 2024</td>
    <td class=odd_rounded>23</td>
    <td class=odd_rounded>42</td>
    <td class=odd_rounded>24</td>
    <td class=odd_rounded>37</td>
    <td class=odd_rounded>16</td>
    <td class=red_rounded>1</td>
  </tr>
</table>
```

**Step 5: Commit fixtures**

```bash
git add tests/fixtures/noroc_chior_*.html
git commit -m "test: add noroc-chior fixtures"
```

### Task 2: Add parsing tests

**Files:**
- Create: `tests/test_noroc_chior_parser.py`

**Step 1: Write failing tests**

```python
from pathlib import Path

from shared.noroc_chior import extract_years, parse_archive_draws

FIXTURES = Path("tests/fixtures")


def test_extract_years():
    html = (FIXTURES / "noroc_chior_years.html").read_text(encoding="utf-8")
    assert extract_years(html) == [1993, 2024]


def test_parse_loto_649_archive():
    html = (FIXTURES / "noroc_chior_649_2024.html").read_text(encoding="utf-8")
    draws = parse_archive_draws(html, numbers_count=6)
    assert draws == [("2024-12-31", [27, 5, 49, 19, 11, 44])]


def test_parse_loto_540_archive():
    html = (FIXTURES / "noroc_chior_540_2024.html").read_text(encoding="utf-8")
    draws = parse_archive_draws(html, numbers_count=6)
    assert draws == [("2024-12-31", [21, 16, 2, 3, 14, 17])]


def test_parse_joker_archive():
    html = (FIXTURES / "noroc_chior_joker_2024.html").read_text(encoding="utf-8")
    draws = parse_archive_draws(html, numbers_count=6)
    assert draws == [("2024-12-31", [23, 42, 24, 37, 16, 1])]
```

**Step 2: Run tests to verify failures**

Run: `PYTHONPATH=src python -m unittest tests/test_noroc_chior_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shared.noroc_chior'`

**Step 3: Commit tests**

```bash
git add tests/test_noroc_chior_parser.py
git commit -m "test: add noroc-chior parser coverage"
```

### Task 3: Implement noroc-chior parser module

**Files:**
- Create: `src/shared/noroc_chior.py`

**Step 1: Write minimal implementation**

```python
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
        if "Data<BR>extragerii" in table:
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
        numbers = [int(n) for n in re.findall(r"<td[^>]*class=(?:odd_rounded|red_rounded)[^>]*>(\d+)</td>", row, re.I)]
        if len(numbers) < numbers_count:
            continue
        draws.append((date, numbers[:numbers_count]))
    return draws
```

**Step 2: Run tests to verify pass**

Run: `PYTHONPATH=src python -m unittest tests/test_noroc_chior_parser.py -v`
Expected: PASS

**Step 3: Commit parser module**

```bash
git add src/shared/noroc_chior.py
git commit -m "feat: add noroc-chior archive parser"
```

### Task 4: Add rebuild script

**Files:**
- Create: `scripts/backfill_noroc_chior.py`

**Step 1: Write minimal rebuild script**

```python
import argparse
import csv
from pathlib import Path
from urllib.request import urlopen

from joker_model.models import JokerDraw
from loto_649_model.models import Loto649Draw
from loto_540_model.models import Loto540Draw
from joker_model.storage import load_draws as load_joker, append_draws as append_joker
from loto_649_model.storage import load_draws as load_649, append_draws as append_649
from loto_540_model.storage import load_draws as load_540, append_draws as append_540
from joker_model.fetch import update_dataset as update_joker
from loto_649_model.fetch import update_dataset as update_649
from loto_540_model.fetch import update_dataset as update_540
from shared.noroc_chior import extract_years, parse_archive_draws


def _fetch(url: str) -> str:
    return urlopen(url).read().decode("utf-8", "ignore")


def _write_draws(path: Path, fieldnames: list[str], rows: list[dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rebuild_game(base_url, csv_path, build_row, numbers_count):
    years = extract_years(_fetch(base_url))
    all_draws = []
    for year in years:
        html = _fetch(f"{base_url}?Y={year}")
        all_draws.extend(parse_archive_draws(html, numbers_count=numbers_count))

    all_draws.sort(key=lambda item: item[0])
    rows = [build_row(date, numbers) for date, numbers in all_draws]
    if rows:
        _write_draws(csv_path, list(rows[0].keys()), rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Rebuild CSVs from noroc-chior archives")
    parser.add_argument("--joker", action="store_true", help="Rebuild Joker")
    parser.add_argument("--loto-649", action="store_true", help="Rebuild Loto 6/49")
    parser.add_argument("--loto-540", action="store_true", help="Rebuild Loto 5/40")
    args = parser.parse_args()

    if not any([args.joker, args.loto_649, args.loto_540]):
        args.joker = args.loto_649 = args.loto_540 = True

    if args.joker:
        count = _rebuild_game(
            "http://noroc-chior.ro/Loto/joker/arhiva-rezultate.php",
            Path("data/clean/joker_draws.csv"),
            lambda d, nums: {
                "date": d,
                "main_1": nums[0],
                "main_2": nums[1],
                "main_3": nums[2],
                "main_4": nums[3],
                "main_5": nums[4],
                "joker": nums[5],
            },
            numbers_count=6,
        )
        print(f"joker: rebuilt {count} draws")

    if args.loto_649:
        count = _rebuild_game(
            "http://noroc-chior.ro/Loto/6-din-49/arhiva-rezultate.php",
            Path("data/clean/loto_649_draws.csv"),
            lambda d, nums: {
                "date": d,
                "main_1": nums[0],
                "main_2": nums[1],
                "main_3": nums[2],
                "main_4": nums[3],
                "main_5": nums[4],
                "main_6": nums[5],
            },
            numbers_count=6,
        )
        print(f"loto_649: rebuilt {count} draws")

    if args.loto_540:
        count = _rebuild_game(
            "http://noroc-chior.ro/Loto/5-din-40/arhiva-rezultate.php",
            Path("data/clean/loto_540_draws.csv"),
            lambda d, nums: {
                "date": d,
                "main_1": nums[0],
                "main_2": nums[1],
                "main_3": nums[2],
                "main_4": nums[3],
                "main_5": nums[4],
                "main_6": nums[5],
            },
            numbers_count=6,
        )
        print(f"loto_540: rebuilt {count} draws")

    # Append newest draws from loto.ro using existing update flow
    if args.joker:
        update_joker(
            "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html",
            Path("data/raw/joker_results.html"),
            Path("data/clean/joker_draws.csv"),
        )
    if args.loto_649:
        update_649(
            "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/649_si_noroc/rezultate_extragere.html",
            Path("data/raw/loto_649_results.html"),
            Path("data/clean/loto_649_draws.csv"),
        )
    if args.loto_540:
        update_540(
            "https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/540_si_super_noroc/rezultate_extrageri.html",
            Path("data/raw/loto_540_results.html"),
            Path("data/clean/loto_540_draws.csv"),
        )


if __name__ == "__main__":
    main()
```

**Step 2: Run parser tests**

Run: `PYTHONPATH=src python -m unittest tests/test_noroc_chior_parser.py -v`
Expected: PASS

**Step 3: Commit script**

```bash
git add scripts/backfill_noroc_chior.py
git commit -m "feat: add noroc-chior rebuild script"
```

### Task 5: Run full test suite

**Step 1: Run all tests**

Run: `PYTHONPATH=src python -m unittest discover -v -s tests`
Expected: PASS

**Step 2: Commit any fixes**

```bash
git add -A
git commit -m "test: verify noroc-chior rebuild changes"
```
