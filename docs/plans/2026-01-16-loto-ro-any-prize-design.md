# Loto.ro Any-Prize Optimizer Design

## Goal
Build a repeatable, code-optional workflow that maximizes the probability of any prize per 50 RON using loto.ro games, while keeping Loto 6/49 as the default choice when options are close.

## Constraints
- Use official loto.ro prices and published odds.
- Allow splitting the 50 RON budget across multiple games in the same draw.
- Do not claim an edge over randomness; only optimize probability per RON.
- Prefer Loto 6/49 when efficiency is within a small margin (about 3-5%).

## Approach Summary
- Collect per-line price and any-prize probability for Loto 6/49, Loto 5/40, Joker, and add-ons (Noroc, Noroc Plus) from loto.ro.
- Compute efficiency as: any_prize_probability / line_price_ron.
- Allocate budget via greedy selection plus a small local swap check to reduce wasted remainder.
- Generate unique random 6/49 lines; for other games, use official quick-pick or game-specific RNG if added later.
- Avoid common human patterns only to reduce shared-prize risk, not to improve odds.

## Current Data Snapshot (2026-01-16)
### Sources
- Prices: https://www.loto.ro/?page_id=1068
- Loto 6/49 odds: https://www.loto.ro/?p=3876
- Loto 5/40 odds: https://www.loto.ro/?p=3921
- Joker odds + Noroc Plus rules: https://www.loto.ro/?p=3904

### Prices (pret varianta simpla)
- Loto 6/49: 8.00 RON
- Joker: 7.00 RON
- Loto 5/40: 5.00 RON
- Noroc: 4.00 RON (only with Loto 6/49)
- Noroc Plus: 3.00 RON (only with Joker)
- Super Noroc: 2.00 RON (only with Loto 5/40)
- Ticket overhead listed: 0.5 RON per ticket; assume lines are grouped to minimize overhead.

### Odds tables (as published)
- Loto 6/49: 1 in 13.983.816 (6/6), 54.200,8 (5/6), 1.032,4 (4/6), 56,66 (3/6)
- Loto 5/40: 1 in 658.008 (5/5 in first 5), 131.602 (5/6), 1.290 (4/6)
- Joker: 1 in 24.435.180, 1.221.759, 122.759, 6.109, 3.140, 157, 240, 60

### Derived any-prize probability per line
- Loto 6/49: 0.01863627 (1.8636%, ~1 in 53.66)
- Loto 5/40: 0.00078431 (0.0784%, ~1 in 1275.00)
- Joker: 0.02769393 (2.7694%, ~1 in 36.11)
- Noroc: 0.0010002 (0.1000%, ~1 in 999.80) from last-3-digits + N+3/N-3 rules
- Noroc Plus / Super Noroc: 0.0199 (1.99%, ~1 in 50.25) from first-2 or last-2 digits

### Efficiency per RON (per line)
- Loto 6/49: 0.00232953
- Joker: 0.00395628
- Loto 5/40: 0.00015686
- Loto 6/49 + Noroc: 0.00163482
- Joker + Noroc Plus: 0.00470428
- Loto 5/40 + Super Noroc: 0.00295267

### Recommended allocation (50 RON budget)
- 5 x Joker + Noroc Plus (10 RON each)
- Per-line any-prize: ~4.704% (~1 in 21.26)
- Per-draw any-prize (5 lines): ~21.41%

## Data Inputs
Each game option should include:
- game_id
- label
- line_price_ron
- any_prize_prob
- number_pool
- numbers_per_line

Use a simple JSON or CSV file (and a spreadsheet mirror) so odds and prices can be updated before each draw.

## Allocation Algorithm
1. Rank games by efficiency.
2. Greedy buy the most efficient option until the next line cannot be afforded.
3. Local swap check: replace 1-2 top-option lines with the next-best option and keep the allocation that yields higher total any-prize probability.
4. If top options are within ~3-5%, prefer Loto 6/49 for jackpot upside.

## Validation and Safety
- Reject invalid inputs (non-positive prices, probabilities outside 0-1).
- If data is missing, fall back to 6/49-only with random unique lines.
- Keep a per-draw log to avoid accidental duplication or overspending.

## References (related sites/projects)
- http://noroc-chior.ro/
- http://noroc-chior.ro/Loto/joker/
- https://www.calculweb.net/loto-generator/
- https://github.com/LoteriaRomana/bilete.loto.ro
- https://github.com/LotoRO-AI/loto-ro-data
- https://github.com/Norb24/RoLoto
- https://github.com/AndreiChristian/LOTERIA_ROMANA_CLI
- https://github.com/Antonia-Tunsoiu/Loto-6-49
- https://github.com/zamsaalbertdaniel/generator-649-pwa
