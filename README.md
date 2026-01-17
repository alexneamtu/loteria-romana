# Loteria Romana - Lottery Modeling Pipelines

A loto.ro-only research pipeline that ingests historical results, stores clean datasets, and generates weekly lines using multiple strategies including statistical analysis, neural networks, ensemble methods, and wheeling systems. The goal is transparent experimentation, not guaranteed wins.

## What this is / isn't

- This is a loto.ro-only pipeline for Joker, Loto 6/49, and Loto 5/40.
- It is an experiment in sampling strategies and statistical modeling.
- It does not improve expected value; lottery outcomes remain random.
- It is not a predictor and not financial advice.

## Documentation

For comprehensive documentation on lottery prediction methods, statistical analysis, and implementation details, see the [docs/](docs/README.md) directory:

- [Introduction to Lottery Prediction](docs/01-introduction.md) - Odds, expected value, and fundamentals
- [Statistical Methods](docs/02-statistical-methods.md) - Classical approaches and their limitations
- [Machine Learning Methods](docs/03-machine-learning-methods.md) - Neural networks and why they don't work
- [Wheeling Systems](docs/04-wheeling-systems.md) - The only mathematically guaranteed approach
- [Loto.ro Specific Analysis](docs/05-loto-ro-specific.md) - Romanian lottery rules and data
- [Implementation Guide](docs/06-implementation-guide.md) - How to use this codebase
- [Backtesting Results](docs/07-backtesting-results.md) - Strategy comparison data
- [Honest Assessment](docs/08-honest-assessment.md) - Limitations and realistic expectations

## Scope

- Target games:
  - **Joker** (main numbers 1-45, Joker 1-20)
  - **Loto 6/49** (main numbers 1-49)
  - **Loto 5/40** (6 numbers drawn from 1-40, player picks 5)
- Data source: official results pages on loto.ro.
- Out of scope: other lotteries/games.

## Odds & Prize Tiers

Odds are fixed by the game rules and are not influenced by historical data or any model.

### Simplified Win Rules

| Game | You win when... |
| --- | --- |
| Loto 6/49 | Match at least 3 of the 6 drawn numbers. |
| Loto 5/40 | Match at least 3 of your 5 numbers among the 6 drawn numbers. |
| Joker | Match 3+ main numbers, or match the Joker number with any count of main numbers (including 0). |

### Loto 6/49 Prize Tiers

| Category | Match rule | Odds | Chance |
| --- | --- | --- | --- |
| I (6) | Match 6 of 6 | 1 in 13,983,816 | 0.000007% |
| II (5) | Match 5 of 6 | 1 in 54,201 | 0.001845% |
| III (4) | Match 4 of 6 | 1 in 1,032 | 0.096862% |
| IV (3) | Match 3 of 6 | 1 in 57 | 1.765040% |

Any prize: 1 in 54 (1.863755%). Jackpot: 1 in 13,983,816 (0.000007%).

### Loto 5/40 Prize Tiers

| Category | Match rule | Odds | Chance |
| --- | --- | --- | --- |
| I (5) | Match 5 of 5 (from 6 drawn) | 1 in 109,668 | 0.000912% |
| II (4) | Match 4 of 5 | 1 in 1,290 | 0.077507% |
| III (3) | Match 3 of 5 | 1 in 59 | 1.705146% |

Any prize: 1 in 56 (1.783565%). Jackpot: 1 in 109,668 (0.000912%).

### Joker Prize Tiers

| Category | Match rule | Odds | Chance |
| --- | --- | --- | --- |
| I (5+J) | Match 5 main + Joker | 1 in 24,435,180 | 0.000004% |
| II (5) | Match 5 main | 1 in 1,286,062 | 0.000078% |
| III (4+J) | Match 4 main + Joker | 1 in 122,176 | 0.000818% |
| IV (4) | Match 4 main | 1 in 6,430 | 0.015551% |
| V (3+J) | Match 3 main + Joker | 1 in 3,133 | 0.031921% |
| VI (3) | Match 3 main | 1 in 165 | 0.606503% |
| VII (2+J) | Match 2 main + Joker | 1 in 247 | 0.404335% |
| VIII (1+J) | Match 1 main + Joker | 1 in 53 | 1.870050% |
| IX (0+J) | Match Joker only | 1 in 37 | 2.692872% |

Any prize: 1 in 18 (5.622132%). Jackpot: 1 in 24,435,180 (0.000004%).

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

Run all tests:

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
--half-life N          Recency half-life in draws (or days)
--half-life-mode MODE  Recency half-life mode: draws or days
--wheel N              Generate wheeling system with N numbers
--wheel-guarantee N    Minimum match guarantee for wheeling
```

Environment:
- `RECENCY_HALF_LIFE_MODE=draws|days` (defaults to draws)
- `RECENCY_HALF_LIFE` (optional numeric half-life)

### Recency Weighting

Strategies apply exponential decay so newer draws influence scores more. The newest draw always has weight 1.0, and the weight halves at the configured half-life.

- `draws` mode: half-life is measured in number of draws (assumes even spacing).
- `days` mode: half-life is measured in calendar days between draw dates, so uneven gaps are handled correctly.

### Educational EV Summary

This is an educational snapshot using user-provided jackpots and explicit assumptions. It is not gambling advice.

Solid Facts (No Assumptions Needed):
- Expected value is negative for all three games at current jackpots (6/49: 16.24M lei, Joker: 47.99M lei, 5/40: 224K lei Cat I).
- EV-optimal variants = 0 for both Thu and Sun (odds/prices identical by day).
- Any-prize probabilities (per line): 6/49 ≈ 1.864%, Joker ≈ 5.622%, 5/40 ≈ 1.784%.

Assumptions (User-Specified):
- Lower-tier expected return as % of line price: 6/49 = 5%, Joker = 7%, 5/40 = 15%.
- Single-winner jackpot (actual splits reduce EV further).
- Slip fee 0.5 lei amortized per line.

Net EV per Line (Under Assumptions):
- 6/49: -6.94 lei (ROI -81.6%)
- Joker: -5.05 lei (ROI -67.3%)
- 5/40: -4.41 lei (ROI -80.2%)

Implication (Assumption-Dependent):
- Under these assumptions, Joker has the least-negative EV among the three.

Cannot Compute Exactly (Data Gap):
- Probability of net profit per N-line bundle requires exact payout tables (tier payouts or exact % per category).

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
- Loto 6/49: `1. 1, 7, 18, 27, 35, 49`
- Loto 5/40: `1. 3, 4, 7, 20, 36`
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
- Loto 6/49 results:
  - https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/649_si_noroc/rezultate_extragere.html
- Loto 5/40 results:
  - https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/540_si_super_noroc/rezultate_extrageri.html

HTML is cached locally to avoid repeated downloads. Parsed draws are stored as CSV.

## Repository Layout

```
src/
├── joker_model/      # Joker pipeline (parser, storage, strategies)
├── loto_649_model/   # Loto 6/49 pipeline
├── loto_540_model/   # Loto 5/40 pipeline
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

## IF YOU CHOOSE TO PLAY ANYWAY – PURCHASE GUIDE

| Game  | Cost/Line | Slip Fee | Rec. Lines | Total Cost | Exp. Loss   | Break-Even Chance* |
|-------|-----------|----------|------------|------------|-------------|--------------------|
| 6/49  | 8.00 lei  | 0.50 lei | 1-5        | 8.50-40.50 | -6.94-34.70 | ~1-7%              |
| Joker | 7.00 lei  | 0.50 lei | 1-10       | 7.50-70.50 | -5.05-50.50 | ~4-38%             |
| 5/40  | 5.00 lei  | 0.50 lei | 1-3        | 5.50-15.50 | -4.41-13.23 | ~1-4%              |

* Approximate chance of net profit (any prize > total spent), based on your lower-tier assumptions (5%/7%/15% return).

RECOMMENDATION BY ENTERTAINMENT BUDGET:
- Under 10 lei: 1 line Joker (7.50 lei, -5.05 expected loss, 4% break-even chance)
- 10-40 lei: 5 lines Joker (35.50 lei, -25.25 loss, ~22% break-even) OR 1 line each game
- 50+ lei: 10 lines Joker (70.50 lei, -50.50 loss, ~38% break-even) - still negative EV

KEY INSIGHT:
If you're buying a ticket anyway for fun, Joker gives you the most chances to hit something (5.6% vs 1.8% per line), so you'll feel the "entertainment value" more often than 6/49 or 5/40.
