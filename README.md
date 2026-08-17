# Loteria Romana — Lottery Modeling Pipeline

A loto.ro-only research pipeline that ingests historical results, stores clean datasets, generates picks for Joker / Loto 6/49 / Loto 5/40, and posts them to Telegram on a schedule. The goal is transparent experimentation and honest measurement — not guaranteed wins.

## What this is / isn't

- Covers **Joker**, **Loto 6/49**, and **Loto 5/40** on loto.ro.
- Experiments with sampling strategies, wheeling, and EV-aware budget gating.
- Does **not** improve win probability — lottery draws are random.
- Not a predictor, not financial advice.

## How it works (high level)

Two GitHub Actions workflows run the whole pipeline:

| Workflow | Schedule | Purpose |
|---|---|---|
| `generate-picks.yml` | Sun + Thu at 06:00 UTC | Scrape current jackpots, run EV gate, generate ticket(s), post to Telegram |
| `check-results.yml` | Mon + Fri at 06:00 UTC | Fetch drawn numbers, score tickets, update history, post results to Telegram |

Both commit their artifacts back to `main` so historical state survives across runs.

## Ticket structure

A real loto.ro ticket is N variants + an optional side game:

| Game | Variant | Variants/ticket | Side game | Side stake | Processing fee | **Full ticket** |
|---|---|---|---|---|---|---|
| Joker | 5 from 1–45 + 1 Joker from 1–20 | 2 | Noroc Plus | 3.0 RON | 0.5 RON | **17.5 RON** |
| Loto 6/49 | 6 from 1–49 | 3 | Noroc (7-digit) | 4.0 RON | 0.5 RON | **28.5 RON** |
| Loto 5/40 | 5 from 1–40 | 4 | Super Noroc (6-digit) | 2.0 RON | 0.5 RON | **22.5 RON** |

The orchestrator emits one `picks/tickets.json` per run with the allocation, variants, and side-game number. The Telegram formatter renders one message per physical ticket grouped with its side game.

## EV gate

Every scheduled run scrapes the current jackpots from the loto.ro homepage and computes `ratio = jackpot / breakeven` per game, where **breakeven** is the jackpot amount at which a ticket's expected value crosses zero.

- `ratio < 0.10` (skip threshold) → no tickets, credit the day's budget to the ledger.
- `0.10 ≤ ratio ≤ 0.35` → play normally at the scheduled budget, using only the games that clear 0.10.
- `ratio > 0.35` (boost threshold) → debit the ledger to increase the effective budget on high rollovers.

**These thresholds are relative, not break-even.** A ratio of 1.0 means the jackpot has reached the point where a ticket's EV crosses zero — for Joker that is ~346M RON and for 6/49 ~337M RON, roughly 5× the largest jackpots those games have ever paid. Gating at `ratio ≥ 1.0` therefore means never playing, which is the mathematically correct answer and also a pipeline that produces nothing. The thresholds above instead ask "is this jackpot high relative to its own range?" and accept a negative EV. Only Loto 5/40 (breakeven ~2.23M RON) can realistically approach 1.0.

`--ev-skip-ratio` defaults to `--ev-min-ratio` so the global skip gate and the per-game filter cannot disagree. Setting skip lower than min only widens the band where a draw is nominally played but every game is filtered out; that case now credits the ledger and posts a reason rather than silently dropping the budget.

Breakeven is computed from the **main-ticket cost only** (variants + processing fee) — side-game stakes have separate prize ladders not scored by the EV model. Each skip/play/boost decision includes the per-game ratios in its reason (e.g. `all ratios < 0.5 (joker=0.19  loto_649=0.05  loto_540=0.28)`), so the Telegram notice explains *why* the gate fired. The ledger persists at `data/budget_bank.json` and is committed on every run.

## Strategies

Ticket builders (in `src/shared/ticket_builders.py`):

- **IndependentBuilder** (default) — each variant is an independent blended-picks draw. Status-quo baseline.
- **CoreShareBuilder** — all variants in a ticket share a high-signal "core" of K numbers; remaining slots rotate through a petal pool. Concentrates variance into the jackpot tail.
- **WheelBuilder** — abbreviated covering wheel over a pool of K numbers. Guarantees N-match coverage if enough pool numbers are drawn.

`generate_blended_picks` in `shared.ensemble_blend` scores multiple non-torch strategies (frequency, Bayesian, cooccurrence, genetic, markov, gradient boost) via walk-forward backtest and allocates picks proportionally. Torch-backed strategies (LSTM / TCN / Transformer / NF / RL) are imported but skipped in production via `DISABLE_TORCH_STRATEGIES=1` — backtests showed no measurable edge.

Wheeling and anti-crowding helpers live in `shared.wheeling` and `shared.crowding` respectively and are wired into the builders above.

## Odds & prize tiers (game rules, unchanged)

Odds are fixed by the games. No model changes them.

### Loto 6/49

| Category | Match | Odds | Chance |
|---|---|---|---|
| I (jackpot) | 6 of 6 | 1 in 13,983,816 | 0.000007% |
| II | 5 of 6 | 1 in 54,201 | 0.001845% |
| III | 4 of 6 | 1 in 1,032 | 0.096862% |
| IV | 3 of 6 | 1 in 57 | 1.765040% |

Any prize ≈ 1 in 54 per variant.

### Loto 5/40 (6 drawn, player picks 5)

| Category | Match | Odds | Chance |
|---|---|---|---|
| I (jackpot) | 5 of 5 among the 6 drawn | 1 in 109,668 | 0.000912% |
| II | 4 of 5 | 1 in 1,290 | 0.077507% |
| III | 3 of 5 | 1 in 59 | 1.705146% |

Any prize ≈ 1 in 56 per variant.

### Joker (5 main + 1 Joker)

| Category | Match | Odds | Chance |
|---|---|---|---|
| I (jackpot) | 5 + Joker | 1 in 24,435,180 | 0.000004% |
| II | 5 main | 1 in 1,286,062 | 0.000078% |
| III | 4 + Joker | 1 in 122,176 | 0.000818% |
| IV | 4 main | 1 in 6,430 | 0.015551% |
| V | 3 + Joker | 1 in 3,133 | 0.031921% |
| VI | 3 main | 1 in 165 | 0.606503% |
| VII | 2 + Joker | 1 in 247 | 0.404335% |
| VIII | 1 + Joker | 1 in 53 | 1.870050% |
| IX | Joker only | 1 in 37 | 2.692872% |

Any prize ≈ 1 in 18 per variant.

### Breakeven jackpots

EV of a ticket turns positive when the jackpot clears the per-game breakeven:

| Game | Approx breakeven | Typical real jackpot |
|---|---|---|
| Joker | ~346M RON | 20–80M RON |
| Loto 6/49 | ~337M RON | 10–30M RON |
| Loto 5/40 | ~2.2M RON | 400K–1M RON |

Under normal conditions all three games sit well below breakeven, and the EV gate will skip most scheduled runs. A skip is the mathematically correct decision — it preserves budget for the rare rollover that actually clears the threshold.

## CLI

The scheduled workflow invokes `scripts/generate_recommended_picks.py`. Common flags:

| Flag | Default | Purpose |
|---|---|---|
| `--budget` | required | Budget in RON. Allocator picks a ticket combination that fits. |
| `--bucket-budget` | unset | Per-game split, e.g. `joker=20,loto_649=30,loto_540=20`. Runs the allocator independently per game so non-dominant games still get exercised. |
| `--strategy` | `independent` | `independent` \| `core_share` \| `wheel:<pool_size>` |
| `--mixes` | `1` | Emit N diverse allocations as `tickets_mix{i}.json` |
| `--seed` | unset | RNG seed for reproducibility |
| `--output-dir` | — | Directory for `tickets.json` + artifacts |
| `--ev-gate` | off | Enable jackpot/breakeven skip/boost logic |
| `--ev-skip-ratio` | `0.5` | Skip if every game's ratio is below this |
| `--ev-boost-ratio` | `1.2` | Boost from ledger if any game's ratio exceeds this |
| `--ledger-path` | `data/budget_bank.json` | File-backed budget ledger |
| `--joker-jackpot` / `--loto649-jackpot` / `--loto540-jackpot` | auto-scraped | Override auto-fetch from loto.ro |
| `--verbose` | off | Per-game probability breakdown |

Result checker (`scripts/check_results.py`) is invoked by the check-results workflow. It reads `picks/tickets.json` from the last artifact, fetches the latest draws, scores main + side-game matches, and appends to `data/results/history.csv` + `data/results/picks_detail.jsonl`.

## Local development

Run all tests (excluding slow integration tests):

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

Run slow integration tests too:

```bash
SLOW_TESTS=1 PYTHONPATH=src python -m pytest tests/ -q
```

Smoke the orchestrator without the EV gate:

```bash
PYTHONPATH=src python scripts/generate_recommended_picks.py \
  --budget 70 --seed 42 --strategy core_share --output-dir /tmp/picks
cat /tmp/picks/tickets.json | python -m json.tool
```

Legacy per-game CLIs still exist for ad-hoc generation:

```bash
PYTHONPATH=src python scripts/generate_joker_picks.py
PYTHONPATH=src python scripts/generate_loto_649_picks.py
PYTHONPATH=src python scripts/generate_loto_540_picks.py
```

These do not go through the ticket/variant/EV-gate machinery; they emit flat line lists.

## Automation setup

Repository secrets (all optional; without them the workflow just logs):

- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` — Telegram destination
- `DATABASE_URL` — Postgres DSN for cross-run analytics
- `RESULTS_DB_PATH` — SQLite file fallback when no Postgres

When a database is configured, scripts dual-write to both files and the following tables:
`workflow_runs`, `generation_runs`, `generated_tickets`, `check_runs`, `check_results`, `budget_ledger_entries`.

The scheduled runner uses the free `ubuntu-24.04-arm` image and the pip cache, which keeps the generate-picks job under 5 min when the EV gate skips (~3 min) and under 10 min when it plays.

## Data sources

loto.ro result pages (HTML scraped and cached):

- Joker: `https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html`
- Loto 6/49: `https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/649_si_noroc/rezultate_extragere.html`
- Loto 5/40: `https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/540_si_super_noroc/rezultate_extrageri.html`
- Homepage (jackpot scrape): `https://www.loto.ro`

## Repository layout

```
src/
├── joker_model/         # Joker parser, storage, draws model (with noroc_plus)
├── loto_649_model/      # Loto 6/49 (with noroc)
├── loto_540_model/      # Loto 5/40 (with super_noroc)
└── shared/
    ├── pricing.py              # Confirmed ticket prices (variant + fee + side game)
    ├── ticket.py               # Ticket + Variant frozen dataclasses
    ├── side_games.py           # Noroc / Super Noroc / Noroc Plus helpers
    ├── ticket_allocator.py     # Budget-aware allocator over confirmed prices
    ├── ticket_builders.py      # IndependentBuilder / CoreShareBuilder / WheelBuilder
    ├── ticket_metrics.py       # skewness / P(best_match≥N) / median ROI / payout
    ├── ticket_backtester.py    # Walk-forward harness over historical CSVs
    ├── ticket_io.py            # tickets.json read/write
    ├── telegram_formatter.py   # Ticket-aware Telegram messages
    ├── picks_detail_history.py # JSONL append-only ticket outcome log
    ├── budget_ledger.py        # EV-gate skip/boost ledger
    ├── jackpot_scraper.py      # loto.ro homepage jackpot parser
    ├── ev_calculator.py        # Per-game EV math + breakeven
    ├── ensemble_blend.py       # Strategy blending (torch-opt-out via env var)
    ├── game_config.py          # Game pools, numbers drawn/picked
    ├── game_recommender.py     # Legacy optimizer (kept for compat)
    ├── results_db.py           # Postgres/SQLite persistence
    ├── wheeling.py             # Coverage wheels
    ├── crowding.py             # Anti-crowding payout helpers
    └── (strategy modules: bayesian, cooccurrence, genetic, gradient_boost, markov, …)

scripts/
├── generate_recommended_picks.py  # Scheduled orchestrator
├── check_results.py               # Scheduled results checker
├── workflow_messages.py           # NUL-separated Telegram message emitter
├── run_jackpot_backtest.py        # Offline strategy comparison
├── migrate_history_schema.py      # One-shot additive CSV migration
└── generate_{joker,loto_649,loto_540}_picks.py  # Legacy per-game CLIs

tests/                # unittest + pytest suite (SLOW_TESTS=1 for integration)
docs/
├── plans/            # Historical design notes + implementation plans
└── 2026-plans/       # Backtest reports, strategy analyses

data/
├── raw/              # Cached loto.ro HTML
├── clean/            # Parsed draw history per game (CSV)
├── results/
│   ├── history.csv         # Aggregated per-ticket outcomes
│   └── picks_detail.jsonl  # Full-detail replayable log
└── budget_bank.json  # EV-gate ledger (committed across runs)
```

## Limitations

- Lottery outcomes are random; no model improves win probability.
- The backtest signal on current history (1000–1200 draws per game) is **not statistically significant**. CoreShare and Wheel show directional jackpot-tilt vs Independent but sample size is too small for conclusive claims.
- At the default 70 RON budget the allocator picks all-Joker (it dominates `P(any win) / RON` by 3–50×). Use `--bucket-budget joker=X,loto_649=Y,loto_540=Z` to force coverage of non-dominant games.
- Anti-crowding (playing unpopular numbers to split the jackpot less when winning) is available as an opt-in flag on `CoreShareBuilder(anti_crowding=True)` — off by default to keep backtest-validated behavior.
- Treat any spending as entertainment, not investment. Under normal jackpot conditions the EV gate will skip most scheduled runs — that is the mathematically correct behavior.
