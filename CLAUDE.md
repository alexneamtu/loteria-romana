# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run all tests:
```bash
PYTHONPATH=src python -m unittest -v
```

Run a single test file:
```bash
PYTHONPATH=src python -m unittest tests/test_parser.py -v
```

Run a specific test method:
```bash
PYTHONPATH=src python -m unittest tests.test_parser.TestParser.test_parse_snippet -v
```

Generate Joker picks:
```bash
PYTHONPATH=src python scripts/generate_joker_picks.py
PYTHONPATH=src python scripts/generate_joker_picks.py --seed 123  # reproducible
```

Generate Loto 6/49 picks:
```bash
PYTHONPATH=src python scripts/generate_loto_649_picks.py
PYTHONPATH=src python scripts/generate_loto_649_picks.py --no-noroc  # main numbers only
PYTHONPATH=src python scripts/generate_loto_649_picks.py --seed 123  # reproducible
```

Generate Loto 5/40 picks:
```bash
PYTHONPATH=src python scripts/generate_loto_540_picks.py
PYTHONPATH=src python scripts/generate_loto_540_picks.py --no-super-noroc  # main numbers only
PYTHONPATH=src python scripts/generate_loto_540_picks.py --seed 123  # reproducible
```

## Architecture

This is a Python lottery modeling pipeline for loto.ro games (Joker, Loto 6/49, and Loto 5/40). There is no `requirements.txt`; the project uses only the standard library.

### Module structure

All game modules (`src/joker_model/`, `src/loto_649_model/`, `src/loto_540_model/`) follow the same pattern:

- `models.py` - Dataclass for draw results (`JokerDraw`, `Loto649Draw`)
- `parser.py` - HTML parsing from loto.ro pages
- `fetch.py` - Download/cache HTML, update dataset
- `storage.py` - CSV read/write for historical draws
- `strategies.py` - Line generation: `generate_random_lines`, `generate_frequency_lines`
- `neural.py` - Softmax neural baseline: `generate_neural_lines`
- `backtest.py` - `pick_best_strategy` compares strategies on historical data
- `metrics.py` - Prize detection: `is_joker_prize`, `is_loto_649_prize`
- `picks.py` - `generate_picks` orchestrates strategy selection and line generation
- `seed.py` - `resolve_seed` handles CLI arg and env var seed resolution

### Data flow

1. `fetch.update_dataset()` downloads HTML (or uses cache), parses draws, appends to CSV
2. `storage.load_draws()` reads CSV into draw objects
3. `picks.generate_picks()` runs backtesting to select best strategy, then generates lines
4. Scripts output formatted lines to stdout

### Game rules encoded

- Joker: 5 main numbers (1-45), 1 joker (1-20)
- Loto 6/49: 6 main numbers (1-49), 7-digit noroc (optional)
- Loto 5/40: 6 numbers drawn (1-40), player picks 5, 6-digit super noroc (optional)

### Test fixtures

Tests use HTML snippets in `tests/fixtures/` to avoid network calls.

## Git Workflow

**Always create pull requests instead of pushing directly to main.**

1. Create a feature branch from main:
   ```bash
   git checkout -b feature/descriptive-branch-name
   ```

2. Make commits with clear messages

3. Push the branch and create a PR:
   ```bash
   git push -u origin feature/descriptive-branch-name
   gh pr create --fill
   ```

4. After PR is approved and merged, clean up:
   ```bash
   git checkout main
   git pull
   git branch -d feature/descriptive-branch-name
   ```
