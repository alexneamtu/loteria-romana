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
