# Loteria Romana - Joker Model Pipeline

A focused, Loto.ro-only research pipeline for the Joker game. It ingests historical results, stores clean datasets, and generates weekly Joker lines using multiple strategies (random, frequency-weighted, and a lightweight neural baseline). The goal is transparent experimentation, not guaranteed wins.

## What this is / isn't

- This is a Joker-only pipeline using loto.ro results.
- It is an experiment in sampling strategies and simple modeling.
- It does not improve expected value; lottery outcomes remain random.
- It is not a predictor and not financial advice.

## Scope

- Target game: **Joker** (main numbers 1-45, Joker 1-20).
- Data source: official results page on loto.ro.
- Out of scope: other lotteries (6/49, 5/40, Noroc, etc.).

## Current status

Implemented:
- HTML parser for Joker draws.
- Dataset update flow (cache HTML -> CSV storage).
- Prize rule check for Joker.
- Random and frequency-weighted line generation.
- Neural sampler (softmax baseline).
- Backtesting and best-strategy selection.
- Weekly picks script that prints 7 Joker lines.

In progress / planned:
- Improve frequency weighting using historical counts by default.
- Add reproducible seeds / config options.

## Data source

Results are parsed from:
- https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html

HTML is cached locally to avoid repeated downloads. Parsed draws are stored as CSV to keep the pipeline lightweight and reproducible.

## Pipeline overview

1. Fetch results HTML (or read from cache).
2. Parse draws into structured records.
3. Append new draws to CSV dataset.
4. Generate candidate lines via strategies.
5. Backtest strategies and pick the best.
6. Output 7 weekly Joker lines.

## Strategies

- Random: uniform sampling without replacement for main numbers.
- Frequency-weighted: samples using a provided frequency table (historical counts wiring is on the roadmap).
- Neural baseline: simple softmax model trained on previous draw -> next draw.

## Quickstart (current)

Run the parser test (verifies parsing + fixture):

```bash
PYTHONPATH=src python -m unittest tests/test_parser.py -v
```

Run all current tests:

```bash
PYTHONPATH=src python -m unittest -v
```

## Repository layout

- `src/joker_model/` - core library (parser, storage, strategies, metrics).
- `tests/` - unit tests and fixtures.
- `docs/plans/` - design notes and implementation plans.
- `data/` - planned location for cached HTML + CSV (created by scripts).

## Roadmap

1. Improve frequency strategy using historical counts.
2. Add reproducible seeds / config options.

## Limitations and ethics

- Lottery outcomes are random; no model can guarantee wins.
- This project is for research and disciplined experimentation.
- Use at your own risk; treat spending as entertainment, not investment.
