# Loteria Romana - Lottery Modeling Pipelines

A loto.ro-only research pipeline that ingests historical results, stores clean datasets, and generates weekly lines using multiple strategies including statistical analysis, neural networks, ensemble methods, and wheeling systems. The goal is transparent experimentation, not guaranteed wins.

## What this is / isn't

- This is a loto.ro-only pipeline for Joker, Loto 6/49 + Noroc, and Loto 5/40 + Super Noroc.
- It is an experiment in sampling strategies and statistical modeling.
- It does not improve expected value; lottery outcomes remain random.
- It is not a predictor and not financial advice.

## Scope

- Target games:
  - **Joker** (main numbers 1-45, Joker 1-20)
  - **Loto 6/49 + Noroc** (main numbers 1-49, Noroc 7-digit number, optional)
  - **Loto 5/40 + Super Noroc** (6 numbers drawn from 1-40, player picks 5, Super Noroc 6-digit number, optional)
- Data source: official results pages on loto.ro.
- Out of scope: other lotteries/games.

## Features

### Statistical Strategies
- **Delta Analysis** - generates numbers based on historical delta (gap) distributions
- **Hot/Cold Numbers** - time-weighted frequency with exponential decay
- **Pair Correlation** - tracks which number pairs appear together frequently
- **Skip/Gap Analysis** - favors "overdue" numbers based on expected gaps
- **Sum Constraints** - filters lines by total sum within historical range
- **Balance Strategy** - matches historical odd/even and high/low ratios

### Neural Networks
- **MLP (Multi-Layer Perceptron)** - configurable hidden layers with L2 regularization
- **LSTM** - sequence learning for temporal patterns in draw history

### Ensemble Methods
- **Ensemble Voter** - combines all strategies with weighted voting
- **Strategy Selector** - automatically selects best-performing strategy

### Wheeling Systems
- **Abbreviated Wheels** - reduce tickets while guaranteeing minimum matches
- **Key Number Wheels** - ensure specific numbers appear in every ticket
- **Coverage Verification** - validates wheel coverage guarantees

### Backtesting
- Prize tier tracking (3-match, 4-match, etc.)
- Wilson score confidence intervals
- Maximum drawdown (longest losing streak)
- Rolling window cross-validation

## Quickstart

Run all tests (174 tests):

```bash
PYTHONPATH=src python -m unittest -v
```

### Basic Usage

Generate 2 Joker picks (auto-selects best strategy):

```bash
PYTHONPATH=src python scripts/generate_joker_picks.py
```

Generate 2 Loto 6/49 picks:

```bash
PYTHONPATH=src python scripts/generate_loto_649_picks.py
```

Generate 2 Loto 5/40 picks:

```bash
PYTHONPATH=src python scripts/generate_loto_540_picks.py
```

### Strategy Selection

Use a specific strategy:

```bash
# Available: auto, delta, hotcold, pairs, skip, sum, balance, ensemble
PYTHONPATH=src python scripts/generate_joker_picks.py -s ensemble -n 5
```

### Wheeling Systems

Generate a wheel with 10 numbers and 3-match guarantee:

```bash
PYTHONPATH=src python scripts/generate_joker_picks.py --wheel 10 --wheel-guarantee 3 -v
```

Generate a Loto 6/49 wheel with 12 numbers and 4-match guarantee:

```bash
PYTHONPATH=src python scripts/generate_loto_649_picks.py --wheel 12 --wheel-guarantee 4 -v
```

### CLI Options

```
-n, --count N          Number of lines to generate (default: 2)
-s, --strategy NAME    Strategy: auto, delta, hotcold, pairs, skip, sum, balance, ensemble
-v, --verbose          Show detailed strategy information
--seed N               Set deterministic RNG seed
--wheel N              Generate wheeling system with N numbers
--wheel-guarantee N    Minimum match guarantee for wheeling
--no-noroc             Omit Noroc from Loto 6/49 picks
--no-super-noroc       Omit Super Noroc from Loto 5/40 picks
```

### Reproducibility

Fixed seed via argument:

```bash
PYTHONPATH=src python scripts/generate_joker_picks.py --seed 123
```

Or via environment variable:

```bash
JOKER_SEED=123 PYTHONPATH=src python scripts/generate_joker_picks.py
LOTO_649_SEED=123 PYTHONPATH=src python scripts/generate_loto_649_picks.py
LOTO_540_SEED=123 PYTHONPATH=src python scripts/generate_loto_540_picks.py
```

### Output Format

- Joker: `1. 7, 11, 44, 45, 46 + J13`
- Loto 6/49: `1. 1, 7, 18, 27, 35, 49 + N6026250`
- Loto 5/40: `1. 3, 4, 7, 20, 36 + SN626628`
- Wheel: Shows coverage info, then numbered tickets

## Data Sources

- Joker results:
  - https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html
- Loto 6/49 + Noroc results:
  - https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/649_si_noroc/rezultate_extragere.html
- Loto 5/40 + Super Noroc results:
  - https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/540_si_super_noroc/rezultate_extrageri.html

HTML is cached locally to avoid repeated downloads. Parsed draws are stored as CSV.

## Repository Layout

```
src/
├── joker_model/      # Joker pipeline (parser, storage, strategies)
├── loto_649_model/   # Loto 6/49 + Noroc pipeline
├── loto_540_model/   # Loto 5/40 + Super Noroc pipeline
└── shared/           # Shared utilities
    ├── math_utils.py     # Softmax, cross-entropy, matrix ops
    ├── config.py         # Game configurations
    ├── strategy_base.py  # Strategy protocol and base class
    ├── neural_base.py    # MLP and LSTM implementations
    ├── stats.py          # Statistical strategies
    ├── ensemble.py       # Ensemble voting and selection
    ├── backtest_base.py  # Backtesting framework
    └── wheeling.py       # Wheeling systems

tests/                # Unit tests (174 tests)
docs/plans/           # Design notes and implementation plans
data/                 # Cached HTML + CSV (created by scripts)
scripts/              # CLI tools for generating picks
```

## Limitations and Ethics

- Lottery outcomes are random; no model can guarantee wins.
- This project is for research and disciplined experimentation.
- Use at your own risk; treat spending as entertainment, not investment.
