# Noroc-Chior Backfill Design

## Goal
Add a one-time rebuild path that pulls historical draws from noroc-chior.ro, rewrites the local CSV datasets from that archive, then runs the existing loto.ro update to append any newest draws not present in the archive. The rebuild should focus on main numbers (and Joker bonus) only.

## Data Source
Use the archive pages for each game:
- Loto 6/49: `http://noroc-chior.ro/Loto/6-din-49/arhiva-rezultate.php`
- Loto 5/40: `http://noroc-chior.ro/Loto/5-din-40/arhiva-rezultate.php`
- Joker: `http://noroc-chior.ro/Loto/joker/arhiva-rezultate.php`

Each page exposes a year selector via `?Y=YYYY`. The backfill will extract available years by parsing `?Y=` links from the base page, then fetch each year page.

## Parsing Approach
Each year page contains a large archive table with the header `Data<BR>extragerii`. For each data row:
- Extract the date from the first `td` with `class=odd` and `nowrap`.
- Normalize Romanian date strings like `Ma, 31 decembrie 2024` by mapping month names to numbers and outputting `YYYY-MM-DD`.
- Extract numbers from cells with `class=odd_rounded` and `class=red_rounded`.
  - For Loto 6/49 and 5/40: use the first 6 numbers.
  - For Joker: use the first 6 numbers, split into 5 main + joker bonus (last number).

All parsing will use stdlib only (`re`, `html`, `urllib.request`).

## Rebuild and Conflicts
Rebuild each CSV from the noroc-chior archive by writing all parsed draws in date order. After the rebuild, call the existing loto.ro `update_dataset()` to append any newer draws. Since `update_dataset()` only appends dates that are missing, noroc-chior remains the source for existing dates; conflicts are logged for review if detected during rebuild.

The script will be append-only and will not reorder existing CSVs.

## Scope / Non-Goals
- No changes to the normal `update_dataset()` flow.
- No ingestion of Noroc/Super Noroc.
- No ongoing merge; this is a one-time rebuild + append flow.
