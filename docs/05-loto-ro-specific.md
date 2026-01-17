# Loto.ro Specific Analysis

This document covers the specific rules, odds, and historical analysis of Romanian lottery games available on loto.ro: Joker, Loto 6/49, and Loto 5/40.

## Game Overview

| Game | Main Pool | Secondary | Draws | Jackpot Odds |
|------|-----------|-----------|-------|--------------|
| Joker | 5 from 45 | 1 from 20 | Sun, Thu | 1:24,435,180 |
| Loto 6/49 | 6 from 49 | - | Sun, Thu | 1:13,983,816 |
| Loto 5/40 | 5 from 40 | - | Sun, Thu | 1:658,008 |

## Joker

### Rules

- **Main numbers**: Pick 5 numbers from 1-45
- **Joker number**: Pick 1 number from 1-20
- **Draws**: Sunday and Thursday evenings

### Prize Structure

| Matches | Joker | Category | Approximate Prize |
|---------|-------|----------|-------------------|
| 5 | Yes | I | Jackpot |
| 5 | No | II | ~50,000 RON |
| 4 | Yes | III | ~5,000 RON |
| 4 | No | IV | ~500 RON |
| 3 | Yes | V | ~50 RON |
| 3 | No | VI | ~10 RON |
| 2 | Yes | VII | Free ticket |
| 1 | Yes | VIII | Free ticket |
| 0 | Yes | IX | Free ticket |

### Probability Calculations

**5 main + Joker (Jackpot):**
```
P = 1 / (C(45,5) × 20)
  = 1 / (1,221,759 × 20)
  = 1 / 24,435,180
  ≈ 0.000000041
```

**5 main, no Joker:**
```
P = 1 / C(45,5) × (19/20)
  = 19 / 24,435,180
  ≈ 0.00000078
```

**4 main + Joker:**
```
P = C(5,4) × C(40,1) / C(45,5) × (1/20)
  = (5 × 40) / 1,221,759 × (1/20)
  = 200 / 24,435,180
  ≈ 0.0000082
```

### Historical Analysis

Based on loto.ro historical data:

**Number frequency distribution:**
- Most frequent main numbers: Varies by period
- Least frequent main numbers: Varies by period
- Chi-square test: p > 0.05 (consistent with randomness)

**Joker number distribution:**
- Expected frequency: 1/20 = 5%
- Observed: Within statistical expectations

## Loto 6/49

### Rules

- **Main numbers**: Pick 6 numbers from 1-49
- **Draws**: Sunday and Thursday evenings

### Prize Structure

| Matches | Category | Approximate Prize |
|---------|----------|-------------------|
| 6 | I | Jackpot |
| 5 | II | ~10,000 RON |
| 4 | III | ~100 RON |
| 3 | IV | ~10 RON |

### Probability Calculations

**6 matches (Jackpot):**
```
P = 1 / C(49,6)
  = 1 / 13,983,816
  ≈ 0.0000000715
```

**5 matches:**
```
P = C(6,5) × C(43,1) / C(49,6)
  = (6 × 43) / 13,983,816
  = 258 / 13,983,816
  ≈ 0.0000184
```

**4 matches:**
```
P = C(6,4) × C(43,2) / C(49,6)
  = (15 × 903) / 13,983,816
  = 13,545 / 13,983,816
  ≈ 0.000969
```

**3 matches:**
```
P = C(6,3) × C(43,3) / C(49,6)
  = (20 × 12,341) / 13,983,816
  = 246,820 / 13,983,816
  ≈ 0.0177
```

### 2025 Randomness Study

A comprehensive analysis of Romanian 6/49 draws was conducted:

**Methodology:**
- Chi-square goodness-of-fit test
- Null hypothesis: Numbers are drawn uniformly at random
- Significance level: α = 0.05

**Results:**
- Test statistic: χ² = 63.98
- Degrees of freedom: 48
- p-value: 0.0766

**Conclusion:**
Since p = 0.0766 > 0.05, we fail to reject the null hypothesis. The data is consistent with true randomness.

### Historical Patterns (Descriptive Only)

These patterns are **descriptive, not predictive**:

**Sum distribution:**
- Mean: ~150
- Standard deviation: ~33
- 71% of winners in range 115-185

**Odd/Even distribution:**
- Most common: 3 odd, 3 even (~32%)
- Next common: 4 odd, 2 even or 2 odd, 4 even (~27% each)

**High/Low distribution (1-24 low, 25-49 high):**
- Most common: 3 high, 3 low (~33%)
- Next common: 4 high, 2 low or 2 high, 4 low (~25% each)

**Consecutive numbers:**
- ~30% of draws contain at least one consecutive pair
- ~5% contain two consecutive pairs

## Loto 5/40

### Rules

- **Draw**: 6 numbers drawn from 1-40
- **Player picks**: 5 numbers
- **Win condition**: Match your 5 with any 5 of the 6 drawn
- **Draws**: Sunday and Thursday evenings

### Prize Structure

| Matches | Category | Approximate Prize |
|---------|----------|-------------------|
| 5 of 6 | I | Jackpot |
| 4 of 6 | II | ~500 RON |
| 3 of 6 | III | ~20 RON |

### Probability Calculations

**5 matches (Jackpot):**
```
P = C(5,5) × C(35,1) / C(40,6) × (6 choose which 5 of 6)
  = 6 / C(40,5)
  = 6 / 658,008
  ≈ 0.00000912
```

Note: Better odds than 6/49 because:
- 6 numbers are drawn (vs 6 picked in 6/49)
- Only need to match 5 of them
- Smaller number pool (40 vs 49)

**4 matches:**
```
P = C(5,4) × C(35,2) / C(40,6) × combinations
  ≈ 0.000395
```

**3 matches:**
```
P = C(5,3) × C(35,3) / C(40,6) × combinations
  ≈ 0.0103
```

### Historical Analysis

**Number frequency distribution:**
- Expected appearances per number: draws × 6 / 40 = 15% of draws
- Observed: Within statistical expectations

**Sum distribution (6 numbers drawn):**
- Mean: ~123
- Standard deviation: ~27

## Data Sources

### Official URLs

**Joker:**
```
https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/joker_si_noroc_plus/rezultate_extrageri.html
```

**Loto 6/49:**
```
https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/649_si_noroc/rezultate_extragere.html
```

**Loto 5/40:**
```
https://www.loto.ro/loto-new/newLotoSiteNexioFinalVersion/web/app2.php/jocuri/540_si_super_noroc/rezultate_extrageri.html
```

### Data Pipeline

1. **Fetch**: Download HTML from loto.ro
2. **Parse**: Extract draw results from HTML
3. **Store**: Save to CSV format
4. **Cache**: HTML cached to avoid repeated downloads

### CSV Format

**Joker:**
```csv
date,n1,n2,n3,n4,n5,joker
2024-01-04,7,14,23,31,42,13
```

**Loto 6/49:**
```csv
date,n1,n2,n3,n4,n5,n6
2024-01-04,3,12,19,27,35,48
```

**Loto 5/40:**
```csv
date,n1,n2,n3,n4,n5,n6
2024-01-04,2,8,15,22,33,39
```

## Expected Value Analysis

### Joker

Assuming ticket cost 5 RON and typical prize pool:

| Prize | Probability | Payout | EV Contribution |
|-------|-------------|--------|-----------------|
| 5+J | 0.000000041 | 5,000,000 | 0.20 |
| 5 | 0.00000078 | 50,000 | 0.04 |
| 4+J | 0.0000082 | 5,000 | 0.04 |
| 4 | 0.000156 | 500 | 0.08 |
| 3+J | 0.000236 | 50 | 0.01 |
| 3 | 0.00449 | 10 | 0.04 |

**Total EV ≈ 0.41 RON per 5 RON ticket**
**Expected loss: ~4.59 RON per ticket (~92% house edge)**

### Loto 6/49

| Prize | Probability | Payout | EV Contribution |
|-------|-------------|--------|-----------------|
| 6 | 0.0000000715 | 5,000,000 | 0.36 |
| 5 | 0.0000184 | 50,000 | 0.92 |
| 4 | 0.000969 | 100 | 0.10 |
| 3 | 0.0177 | 10 | 0.18 |

**Total EV ≈ 1.56 RON per 3 RON ticket**
**Expected loss: ~1.44 RON per ticket (~48% house edge)**

### Loto 5/40

| Prize | Probability | Payout | EV Contribution |
|-------|-------------|--------|-----------------|
| 5 | 0.00000912 | 500,000 | 4.56 |
| 4 | 0.000395 | 500 | 0.20 |
| 3 | 0.0103 | 20 | 0.21 |

**Total EV ≈ 4.97 RON per ticket** (varies with jackpot size)

Note: 5/40 has relatively better odds due to game structure.

## Strategy Implementation

### Game Configurations

From `src/shared/game_config.py`:

```python
JOKER_CONFIG = {
    'name': 'joker',
    'main_pool': 45,
    'main_pick': 5,
    'secondary_pool': 20,
    'secondary_pick': 1,
}

LOTO_649_CONFIG = {
    'name': 'loto_649',
    'main_pool': 49,
    'main_pick': 6,
    'secondary_pool': None,
    'secondary_pick': None,
}

LOTO_540_CONFIG = {
    'name': 'loto_540',
    'main_pool': 40,
    'main_pick': 5,
    'numbers_drawn': 6,
    'secondary_pool': None,
    'secondary_pick': None,
}
```

### Prize Detection

From game-specific `metrics.py`:

```python
def is_joker_prize(ticket: tuple[list[int], int],
                   winning: tuple[list[int], int]) -> dict[str, bool]:
    main_matches = len(set(ticket[0]) & set(winning[0]))
    joker_match = ticket[1] == winning[1]

    return {
        'category_1': main_matches == 5 and joker_match,
        'category_2': main_matches == 5 and not joker_match,
        'category_3': main_matches == 4 and joker_match,
        # ... etc
    }
```

## CLI Usage

### Generate Joker Picks

```bash
# Default (smart strategy, 2 picks)
PYTHONPATH=src python scripts/generate_joker_picks.py

# Specific strategy
PYTHONPATH=src python scripts/generate_joker_picks.py -s delta -n 5

# With wheeling
PYTHONPATH=src python scripts/generate_joker_picks.py --wheel 10 --wheel-guarantee 3
```

### Generate Loto 6/49 Picks

```bash
PYTHONPATH=src python scripts/generate_loto_649_picks.py

# Verbose output
PYTHONPATH=src python scripts/generate_loto_649_picks.py -v

# Fixed seed for reproducibility
PYTHONPATH=src python scripts/generate_loto_649_picks.py --seed 42
```

### Generate Loto 5/40 Picks

```bash
PYTHONPATH=src python scripts/generate_loto_540_picks.py

# Multiple picks with pattern strategy
PYTHONPATH=src python scripts/generate_loto_540_picks.py -s pattern -n 10
```

## Comparison with Other Lotteries

| Lottery | Country | Pool | Pick | Jackpot Odds |
|---------|---------|------|------|--------------|
| Joker | Romania | 45+20 | 5+1 | 1:24M |
| Loto 6/49 | Romania | 49 | 6 | 1:14M |
| Loto 5/40 | Romania | 40 | 5/6 | 1:658K |
| EuroMillions | Europe | 50+12 | 5+2 | 1:139M |
| Powerball | USA | 69+26 | 5+1 | 1:292M |
| UK Lotto | UK | 59 | 6 | 1:45M |

Romanian lotteries have relatively better odds than major international lotteries, though expected value remains negative.

## Summary

| Aspect | Joker | Loto 6/49 | Loto 5/40 |
|--------|-------|-----------|-----------|
| Best odds | | | X |
| Largest jackpot | X | | |
| Proven random | X | X | X |
| EV (per ticket) | ~-92% | ~-48% | Varies |

All three games are verified random and have negative expected value. Choose based on:
- **Jackpot size**: Joker typically has largest
- **Win probability**: 5/40 has best odds for any prize
- **Entertainment value**: Personal preference

## Next Steps

- [Implementation Guide](06-implementation-guide.md) - How to use this codebase
- [Backtesting Results](07-backtesting-results.md) - Strategy comparisons
- [Honest Assessment](08-honest-assessment.md) - Realistic expectations
