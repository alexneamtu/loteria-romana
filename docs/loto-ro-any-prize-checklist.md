# Loto.ro Any-Prize Checklist (50 RON)

## Scope
- Goal: maximize chance of any prize per draw.
- Scope: loto.ro only; budget can split across games.
- Disclaimer: no edge over randomness, only probability per RON.

## Inputs (update each draw)
- Prices: https://www.loto.ro/?page_id=1068
- Loto 6/49 odds: https://www.loto.ro/?p=3876
- Loto 5/40 odds: https://www.loto.ro/?p=3921
- Joker odds + Noroc Plus rules: https://www.loto.ro/?p=3904

## Quick math
- Any-prize for a game with categories: `P(any) = sum(1 / odds_i)`
- Add-on combined line: `P(A or B) = 1 - (1 - P(A)) * (1 - P(B))`
- Efficiency per RON: `P(any) / price_ron`
- Ticket overhead: 0.5 RON per ticket; group lines to minimize overhead.

## Add-on base probabilities (rules-based)
- Noroc: last 3 digits match + N+3/N-3 => `0.0010002`
- Noroc Plus / Super Noroc: first 2 or last 2 digits match => `0.0199`

## Allocation steps
1. Compute `P(any)` for each base game line.
2. Compute `P(any)` for add-on combos (6/49+Noroc, Joker+Noroc Plus, 5/40+Super Noroc).
3. Rank by efficiency per RON.
4. Buy the most efficient line type until the 50 RON budget is used.
5. Swap check: replace 1-2 top lines with next-best option if it increases total `P(any)`.
6. Tie-break: if within 3-5%, prefer Loto 6/49 for jackpot upside.

## Current snapshot (2026-01-16)
- Prices (RON): 6/49=8, Joker=7, 5/40=5, Noroc=4, Noroc Plus=3, Super Noroc=2
- Any-prize per line:
  - Loto 6/49: 0.01863627
  - Joker: 0.02769393
  - Loto 5/40: 0.00078431
  - Loto 6/49 + Noroc: 0.01961783
  - Joker + Noroc Plus: 0.04704282
  - Loto 5/40 + Super Noroc: 0.02066870
- Recommended allocation (50 RON):
  - 5 x Joker + Noroc Plus (10 RON each)
  - Per-draw any-prize chance: ~21.41%

## Line selection
- Use RNG/quick-pick; keep all lines unique within a game.
- Avoid common human patterns to reduce shared prizes (not to improve odds).

## Draw log (one line per draw)
- Date, games played, number of lines, total cost, result.
