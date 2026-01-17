# Implementation Guide

This document explains how to use the loteria-romana codebase effectively, including CLI usage, strategy selection, wheeling systems, and extending the code.

## Quick Start

### Prerequisites

- Python 3.10+
- No external dependencies (uses standard library only)

### Basic Usage

```bash
# Generate 2 Joker picks
PYTHONPATH=src python scripts/generate_joker_picks.py

# Generate 2 Loto 6/49 picks
PYTHONPATH=src python scripts/generate_loto_649_picks.py

# Generate 2 Loto 5/40 picks
PYTHONPATH=src python scripts/generate_loto_540_picks.py
```

### Running Tests

```bash
# All tests
PYTHONPATH=src python -m unittest -v

# Specific test file
PYTHONPATH=src python -m unittest tests/test_parser.py -v

# Specific test method
PYTHONPATH=src python -m unittest tests.test_parser.TestParser.test_parse_snippet -v
```

## CLI Reference

### Common Options

| Option | Short | Description |
|--------|-------|-------------|
| `--count` | `-n` | Number of lines to generate (default: 2) |
| `--strategy` | `-s` | Strategy to use (default: smart) |
| `--verbose` | `-v` | Show detailed information |
| `--seed` | | Set RNG seed for reproducibility |
| `--wheel` | | Generate wheel with N numbers |
| `--wheel-guarantee` | | Minimum match guarantee for wheel |

### Available Strategies

#### Advanced Strategies (Recommended)

| Strategy | Description |
|----------|-------------|
| `smart` | Combines all techniques (default) |
| `optimal` | Composite scoring with frequency, recency, gaps |
| `coverage` | Maximizes number diversity |
| `pattern` | Matches historical sum, odd/even, high/low |

#### Statistical Strategies

| Strategy | Description |
|----------|-------------|
| `delta` | Uses historical delta (gap) distributions |
| `hotcold` | Time-weighted frequency with decay |
| `pairs` | Favors frequently co-occurring pairs |
| `skip` | Weights "overdue" numbers higher |
| `sum` | Filters by total sum range |
| `balance` | Matches odd/even, high/low ratios |

#### Other Strategies

| Strategy | Description |
|----------|-------------|
| `auto` | Backtests and selects best strategy |
| `ensemble` | Weighted combination of all strategies |
| `random` | Pure random selection |
| `frequency` | Simple historical frequency |

### Examples

```bash
# 5 picks using delta strategy
PYTHONPATH=src python scripts/generate_joker_picks.py -s delta -n 5

# Verbose output with pattern strategy
PYTHONPATH=src python scripts/generate_loto_649_picks.py -s pattern -v

# Reproducible results with seed
PYTHONPATH=src python scripts/generate_loto_540_picks.py --seed 42 -n 3

# Wheeling system
PYTHONPATH=src python scripts/generate_loto_649_picks.py --wheel 10 --wheel-guarantee 4 -v
```

## Strategy Selection Guide

### Decision Tree

```
Do you want reproducibility?
├── Yes → Use --seed option
└── No → Continue

Do you want mathematical guarantees?
├── Yes → Use wheeling (--wheel, --wheel-guarantee)
└── No → Continue

Do you want automatic strategy selection?
├── Yes → Use -s auto
└── No → Continue

Do you want structured selections?
├── Yes → Use advanced strategies (smart, optimal, pattern)
└── No → Use -s random
```

### Strategy Comparison

| Strategy | Complexity | Backtested | Structured | Best For |
|----------|------------|------------|------------|----------|
| smart | High | Yes | Yes | General use |
| optimal | High | Yes | Yes | Score-based |
| coverage | Medium | No | Yes | Diversity |
| pattern | Medium | No | Yes | Matching history |
| delta | Medium | Yes | Yes | Gap analysis |
| hotcold | Medium | Yes | Yes | Trend following |
| auto | Variable | Yes | Yes | Unknown data |
| ensemble | High | Yes | Yes | Aggregation |
| random | Low | No | No | Baseline |

### Recommended Strategy by Use Case

| Use Case | Strategy | Reason |
|----------|----------|--------|
| Regular play | `smart` | Best overall approach |
| Pool/syndicate | `--wheel` | Coverage guarantees |
| Testing | `--seed` + any | Reproducibility |
| Research | `auto` | Strategy comparison |
| Quick pick | `random` | True randomness |

## Wheeling Systems

### When to Use Wheels

1. **Group play** - Multiple people sharing tickets
2. **Systematic approach** - Guaranteed coverage
3. **Budget planning** - Known ticket count

### Wheel Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--wheel N` | Total numbers to wheel | `--wheel 10` |
| `--wheel-guarantee N` | Minimum matches guaranteed | `--wheel-guarantee 4` |

### Examples

```bash
# 10 numbers, guarantee at least 3 matches if 5+ hit
PYTHONPATH=src python scripts/generate_joker_picks.py --wheel 10 --wheel-guarantee 3 -v

# 12 numbers, guarantee at least 4 matches if 6 hit
PYTHONPATH=src python scripts/generate_loto_649_picks.py --wheel 12 --wheel-guarantee 4 -v
```

### Wheel Output

```
=== Wheeling System ===
Numbers in wheel: 10
Numbers per ticket: 6
Guarantee: 4-if-6

Full wheel would require: 210 tickets
Abbreviated wheel uses: 18 tickets
Reduction: 91.4%

Coverage verified: 100% of 4-combinations covered

Tickets:
1. 1, 2, 3, 4, 5, 6
2. 1, 2, 3, 7, 8, 9
...
```

## Output Formats

### Standard Output

```
Joker picks (smart strategy):
1. 7, 11, 24, 33, 45 + J13
2. 3, 18, 29, 38, 42 + J7
```

### Verbose Output (-v)

```
Joker picks using smart strategy

Strategy: smart
Based on: 500 historical draws
Last updated: 2024-01-15

Generated picks:
1. 7, 11, 24, 33, 45 + J13
   Sum: 120 | Odd/Even: 3/2 | High/Low: 3/2
2. 3, 18, 29, 38, 42 + J7
   Sum: 130 | Odd/Even: 2/3 | High/Low: 3/2
```

## Environment Variables

### Seed Control

```bash
# Game-specific seeds
JOKER_SEED=123 PYTHONPATH=src python scripts/generate_joker_picks.py
LOTO_649_SEED=456 PYTHONPATH=src python scripts/generate_loto_649_picks.py
LOTO_540_SEED=789 PYTHONPATH=src python scripts/generate_loto_540_picks.py
```

### Data Directory

```bash
# Custom data directory
DATA_DIR=/path/to/data PYTHONPATH=src python scripts/generate_joker_picks.py
```

## Code Organization

### Directory Structure

```
src/
├── joker_model/      # Joker-specific code
│   ├── models.py     # JokerDraw dataclass
│   ├── parser.py     # HTML parsing
│   ├── fetch.py      # Download/update data
│   ├── storage.py    # CSV read/write
│   ├── strategies.py # Strategy implementations
│   ├── neural.py     # Neural network models
│   ├── backtest.py   # Backtesting
│   ├── metrics.py    # Prize detection
│   ├── picks.py      # Main pick generation
│   └── seed.py       # Seed resolution
├── loto_649_model/   # Loto 6/49 (same structure)
├── loto_540_model/   # Loto 5/40 (same structure)
└── shared/           # Shared utilities
    ├── game_config.py        # Game configurations
    ├── game_strategies.py    # Unified strategies
    ├── advanced_strategies.py # Smart/optimal/coverage
    ├── features.py           # Feature extraction
    ├── neural_strategies.py  # Neural strategies
    ├── math_utils.py         # Math operations
    ├── strategy_base.py      # Strategy protocol
    ├── neural_base.py        # MLP/LSTM base
    ├── stats.py              # Statistical strategies
    ├── ensemble.py           # Ensemble methods
    ├── backtest_base.py      # Backtesting framework
    └── wheeling.py           # Wheeling systems
```

### Key Classes

#### Strategy Protocol

```python
from typing import Protocol
import random

class Strategy(Protocol):
    name: str

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
    ) -> list[list[int]]:
        ...

    def get_probabilities(
        self,
        draws: list[list[int]],
    ) -> list[float]:
        ...
```

#### Backtester

```python
from shared.backtest_base import Backtester, BacktestResult

backtester = Backtester(number_pool=49, numbers_to_pick=6)
result = backtester.backtest(strategy, draws, tickets_per_draw=1)

print(f"Win rate: {result.win_rate:.4f}")
print(f"Max drawdown: {result.max_drawdown}")
```

#### WheelGenerator

```python
from shared.wheeling import WheelGenerator

wheel = WheelGenerator(numbers_per_line=6, guarantee=4)
tickets = wheel.generate_abbreviated_wheel(numbers=[1,2,3,4,5,6,7,8,9,10])
```

## Extending the Code

### Adding a New Strategy

1. Create strategy class:

```python
# src/shared/my_strategy.py

class MyStrategy:
    def __init__(self, number_pool: int, numbers_to_pick: int):
        self.number_pool = number_pool
        self.numbers_to_pick = numbers_to_pick
        self.name = "mystrategy"

    def get_probabilities(self, draws: list[list[int]]) -> list[float]:
        # Return probability for each number
        return [1.0 / self.number_pool] * self.number_pool

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
    ) -> list[list[int]]:
        # Generate `count` lines
        lines = []
        for _ in range(count):
            line = sorted(rng.sample(
                range(1, self.number_pool + 1),
                self.numbers_to_pick
            ))
            lines.append(line)
        return lines
```

2. Register in game strategies:

```python
# src/joker_model/strategies.py

from shared.my_strategy import MyStrategy

STRATEGIES = {
    # ... existing strategies
    'mystrategy': lambda: MyStrategy(45, 5),
}
```

### Adding a New Game

1. Create game directory:

```
src/my_game_model/
├── __init__.py
├── models.py
├── parser.py
├── fetch.py
├── storage.py
├── strategies.py
├── neural.py
├── backtest.py
├── metrics.py
├── picks.py
└── seed.py
```

2. Define game configuration:

```python
# src/shared/game_config.py

MY_GAME_CONFIG = {
    'name': 'my_game',
    'main_pool': 50,
    'main_pick': 5,
    'secondary_pool': 10,
    'secondary_pick': 1,
}
```

3. Create CLI script:

```python
# scripts/generate_my_game_picks.py

from my_game_model.picks import generate_picks
# ... CLI implementation
```

### Adding New Features

1. Define feature extractor:

```python
# src/shared/features.py

def compute_my_feature(draws: list[list[int]], pool_size: int) -> dict[int, float]:
    scores = {}
    for n in range(1, pool_size + 1):
        scores[n] = calculate_score(n, draws)
    return scores
```

2. Integrate into advanced strategies:

```python
# src/shared/advanced_strategies.py

def compute_composite_score(n: int, features: dict) -> float:
    score = 0.0
    score += features['frequency'][n] * 0.25
    score += features['my_feature'][n] * 0.15  # New feature
    # ...
    return score
```

## Data Management

### Updating Data

```python
# Programmatic data update
from joker_model.fetch import update_dataset

update_dataset()  # Downloads latest results
```

### Data Location

```
data/
├── joker/
│   ├── results.html      # Cached HTML
│   └── draws.csv         # Parsed results
├── loto_649/
│   └── ...
└── loto_540/
    └── ...
```

### Forcing Data Refresh

```bash
# Delete cache to force refresh
rm -rf data/joker/*.html
PYTHONPATH=src python scripts/generate_joker_picks.py
```

## Debugging

### Verbose Mode

```bash
PYTHONPATH=src python scripts/generate_joker_picks.py -v
```

### Python Debugging

```python
# Add to script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspecting Strategies

```python
from joker_model.strategies import get_strategy

strategy = get_strategy('smart', number_pool=45, numbers_to_pick=5)
probs = strategy.get_probabilities(historical_draws)

# View top 10 probabilities
sorted_probs = sorted(
    [(n+1, p) for n, p in enumerate(probs)],
    key=lambda x: x[1],
    reverse=True
)
print(sorted_probs[:10])
```

## Performance Tips

### Large Datasets

For backtesting over large datasets:

```python
# Use rolling window to limit memory
from shared.backtest_base import CrossValidator

cv = CrossValidator(n_splits=5, min_train_size=50)
results = cv.evaluate(strategy, draws, backtester)
```

### Parallel Processing

The codebase is single-threaded by design (standard library only), but you can parallelize at the script level:

```bash
# Run in parallel
PYTHONPATH=src python scripts/generate_joker_picks.py -s delta -n 1000 &
PYTHONPATH=src python scripts/generate_joker_picks.py -s hotcold -n 1000 &
wait
```

## Summary

| Task | Command/Approach |
|------|-----------------|
| Quick picks | `python scripts/generate_*_picks.py` |
| Specific strategy | `-s STRATEGY` |
| Reproducibility | `--seed N` |
| Wheeling | `--wheel N --wheel-guarantee M` |
| Verbose output | `-v` |
| Multiple picks | `-n COUNT` |
| Testing | `python -m unittest -v` |
| Extend strategies | Implement Strategy protocol |

## Next Steps

- [Backtesting Results](07-backtesting-results.md) - Strategy comparisons
- [Honest Assessment](08-honest-assessment.md) - Limitations
- [Statistical Methods](02-statistical-methods.md) - Theory behind strategies
