# Loteria Romana - Lottery Modeling Pipelines

A loto.ro-only research pipeline that ingests historical results, stores clean datasets, and generates weekly lines using multiple strategies (random, frequency-weighted, and a lightweight neural baseline). The goal is transparent experimentation, not guaranteed wins.

## What this is / isn't

- This is a loto.ro-only pipeline for Joker and Loto 6/49 + Noroc.
- It is an experiment in sampling strategies and simple modeling.
- It does not improve expected value; lottery outcomes remain random.
- It is not a predictor and not financial advice.

## Scope

- Target games:
  - **Joker** (main numbers 1-45, Joker 1-20)
  - **Loto 6/49 + Noroc** (main numbers 1-49, Noroc 7-digit number, optional)
- Data source: official results pages on loto.ro.
- Out of scope: other lotteries/games.

## Current status

Implemented:
- HTML parsers for Joker and Loto 6/49 + Noroc.
- Dataset update flow (cache HTML -> CSV storage).
- Prize rule checks (any prize) for each game.
- Random and frequency-weighted line generation.
- Neural sampler (softmax baseline) for main numbers.
- Backtesting and best-strategy selection.
- Weekly picks scripts that print 2 lines (Variant A/B) per run.

## Data sources

- Joker results:
  - https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html
- Loto 6/49 + Noroc results:
  - https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/649_si_noroc/rezultate_extragere.html

HTML is cached locally to avoid repeated downloads. Parsed draws are stored as CSV to keep the pipeline lightweight and reproducible.

## Pipeline overview

1. Fetch results HTML (or read from cache).
2. Parse draws into structured records.
3. Append new draws to CSV dataset.
4. Generate candidate lines via strategies.
5. Backtest strategies and pick the best.
6. Output 2 variants per run.

## Strategies

- Random: uniform sampling without replacement for main numbers.
- Frequency-weighted: uses full-history counts with +1 smoothing.
- Neural baseline: simple softmax model trained on previous draw -> next draw.

## Quickstart

Run all tests:

```bash
PYTHONPATH=src python -m unittest -v
```

Generate 2 Joker variants before each draw:

```bash
PYTHONPATH=src python scripts/generate_joker_picks.py
```

Reproducible Joker variants with a fixed seed:

```bash
PYTHONPATH=src python scripts/generate_joker_picks.py --seed 123
```

Or via environment variable:

```bash
JOKER_SEED=123 PYTHONPATH=src python scripts/generate_joker_picks.py
```

Generate 2 Loto 6/49 + Noroc variants before each draw:

```bash
PYTHONPATH=src python scripts/generate_loto_649_picks.py
```

Omit Noroc (main numbers only):

```bash
PYTHONPATH=src python scripts/generate_loto_649_picks.py --no-noroc
```

Reproducible Loto 6/49 variants with a fixed seed:

```bash
PYTHONPATH=src python scripts/generate_loto_649_picks.py --seed 123
```

Or via environment variable:

```bash
LOTO_649_SEED=123 PYTHONPATH=src python scripts/generate_loto_649_picks.py
```

Output format:
- Joker: `1. 7, 11, 44, 45, 46 + J13`
- Loto 6/49: `1. 1, 7, 18, 27, 35, 49 + N6026250`

## Repository layout

- `src/joker_model/` - Joker pipeline (parser, storage, strategies, metrics).
- `src/loto_649_model/` - Loto 6/49 + Noroc pipeline.
- `tests/` - unit tests and fixtures.
- `docs/plans/` - design notes and implementation plans.
- `data/` - cached HTML + CSV (created by scripts).

## Limitations and ethics

- Lottery outcomes are random; no model can guarantee wins.
- This project is for research and disciplined experimentation.
- Use at your own risk; treat spending as entertainment, not investment.
