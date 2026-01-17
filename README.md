# Loteria Romana - Lottery Modeling Pipelines

A loto.ro-only research pipeline that ingests historical results, stores clean datasets, and generates weekly lines using multiple strategies including statistical analysis, neural networks, ensemble methods, and wheeling systems. The goal is transparent experimentation, not guaranteed wins.

## What this is / isn't

- This is a loto.ro-only pipeline for Joker, Loto 6/49, and Loto 5/40.
- It is an experiment in sampling strategies and statistical modeling.
- It does not improve expected value; lottery outcomes remain random.
- It is not a predictor and not financial advice.

## Scope

- Target games:
  - **Joker** (main numbers 1-45, Joker 1-20)
  - **Loto 6/49** (main numbers 1-49, Noroc optional)
  - **Loto 5/40** (6 numbers drawn from 1-40, player picks 5, Super Noroc optional)
- Data source: official results pages on loto.ro.
- Out of scope: other lotteries/games.

## Features

### Advanced Strategies (Recommended)
- **Smart** - combines all techniques for maximum accuracy (default)
- **Optimal** - composite scoring with frequency, recency, gaps, and trends
- **Coverage** - maximizes number diversity across picks
- **Pattern** - matches historical sum, odd/even, and high/low patterns

### Statistical Strategies
- **Delta Analysis** - generates numbers based on historical delta (gap) distributions
- **Hot/Cold Numbers** - time-weighted frequency with exponential decay
- **Pair Correlation** - tracks which number pairs appear together frequently
- **Skip/Gap Analysis** - favors "overdue" numbers based on expected gaps
- **Sum Constraints** - filters lines by total sum within historical range
- **Balance Strategy** - matches historical odd/even and high/low ratios

### Feature Engineering
- **Digit frequency analysis** - detects patterns in individual digits (0-9)
- **Prime number ratio** - tracks proportion of primes in draws
- **Modular residues** - reveals periodicity patterns (mod 2, 3, 5, 7, 10)
- **Entropy scoring** - measures randomness to detect anomalies
- **Position frequency** - analyzes where numbers appear when sorted
- **Autocorrelation** - measures self-similarity between draws

### Neural Networks
- **MLP (Multi-Layer Perceptron)** - configurable hidden layers with L2 regularization
- **LSTM** - sequence learning for temporal patterns in draw history
- **Softmax regression** - lightweight model for probability prediction

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

Generate 2 Joker picks (uses smart strategy by default):

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
# Advanced strategies (recommended): smart, optimal, coverage, pattern
# Statistical strategies: delta, hotcold, pairs, skip, sum, balance
# Other: auto, ensemble
PYTHONPATH=src python scripts/generate_joker_picks.py -s smart -n 5
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
-s, --strategy NAME    Strategy (default: smart)
                       Advanced: smart, optimal, coverage, pattern
                       Statistical: delta, hotcold, pairs, skip, sum, balance
                       Other: auto, ensemble
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
- Loto 6/49: `1. 1, 7, 18, 27, 35, 49 + N6026250` (or without Noroc if `--no-noroc`)
- Loto 5/40: `1. 3, 4, 7, 20, 36 + SN626628` (or without Super Noroc if `--no-super-noroc`)
- Wheel: Shows coverage info, then numbered tickets

## Automation

The repository includes GitHub Actions workflows that run automatically:

- **Generate Picks** (Sunday & Thursday at 10 AM UTC): Generates picks for all three games using smart strategy and sends them via Telegram
- **Check Results** (Monday & Friday at 8 AM UTC): Compares generated picks against actual draw results

To enable Telegram notifications, set repository secrets:
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID

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
    ├── game_config.py        # Game configurations (pool sizes, rules)
    ├── game_strategies.py    # Unified random/frequency strategies
    ├── advanced_strategies.py # Smart/optimal/coverage/pattern strategies
    ├── features.py           # Statistical feature extraction
    ├── neural_strategies.py  # Neural network strategies
    ├── math_utils.py         # Softmax, cross-entropy, matrix ops
    ├── strategy_base.py      # Strategy protocol and base class
    ├── neural_base.py        # MLP and LSTM implementations
    ├── stats.py              # Statistical strategies (delta, hotcold, etc.)
    ├── ensemble.py           # Ensemble voting and selection
    ├── backtest_base.py      # Backtesting framework
    └── wheeling.py           # Wheeling systems

tests/                # Unit tests
docs/plans/           # Design notes and implementation plans
data/                 # Cached HTML + CSV (created by scripts)
scripts/              # CLI tools for generating picks
```

## Limitations and Ethics

- Lottery outcomes are random; no model can guarantee wins.
- This project is for research and disciplined experimentation.
- Use at your own risk; treat spending as entertainment, not investment.
