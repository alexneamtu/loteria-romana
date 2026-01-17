# Statistical Methods for Lottery Analysis

This document covers classical statistical approaches to lottery number selection, their mathematical foundations, implementation details, and why they cannot actually improve your odds.

## Overview of Methods

| Method             | Basis                        | Implementation                  | Effectiveness         |
|--------------------|------------------------------|---------------------------------|-----------------------|
| Frequency Analysis | Historical occurrence counts | `src/shared/game_strategies.py` | None (for prediction) |
| Hot/Cold Numbers   | Time-weighted frequency      | `src/shared/stats.py`           | None (for prediction) |
| Delta Analysis     | Gap distributions            | `src/shared/stats.py`           | None (for prediction) |
| Sum Constraints    | Normal distribution of sums  | `src/shared/stats.py`           | None (for prediction) |
| Pair Correlation   | Co-occurrence patterns       | `src/shared/stats.py`           | None (for prediction) |
| Skip/Gap Analysis  | Time since appearance        | `src/shared/stats.py`           | None (for prediction) |
| Balance Strategy   | Odd/even, high/low ratios    | `src/shared/stats.py`           | None (for prediction) |

## Frequency Analysis

### Concept

Count how often each number has appeared historically, then weight selections toward more (or less) frequent numbers.

### Mathematical Foundation

For each number n in pool [1, N]:
```
frequency(n) = count(n appears in historical draws) / total_draws
```

With Laplace smoothing to avoid zero probabilities:
```
smoothed_frequency(n) = (count(n) + α) / (total_draws + α × N)
```

Where α is typically 1 (add-one smoothing).

### Implementation

```python
def compute_frequencies(draws: list[list[int]], pool_size: int) -> dict[int, float]:
    counts = {n: 0 for n in range(1, pool_size + 1)}
    for draw in draws:
        for n in draw:
            counts[n] += 1
    total = sum(counts.values())
    return {n: c / total for n, c in counts.items()}
```

### Why It Doesn't Work

1. **Law of Large Numbers** - With sufficient draws, all numbers converge to equal frequency
2. **Independence** - Past frequency has no bearing on future draws
3. **Regression to the Mean** - Deviations from expected frequency self-correct naturally

### Research Finding

> Chi-square tests on Romanian 6/49 data (p-value 0.0766) show no significant deviation from uniform distribution.

## Hot/Cold Numbers

### Concept

Numbers are classified as "hot" (appearing frequently recently) or "cold" (appearing rarely recently). Strategies either follow or fade these trends.

### Mathematical Foundation

Time-weighted frequency with exponential decay:
```
heat_score(n) = Σ decay^age × I(n in draw)
```

Where:
- `decay` is typically 0.95
- `age` is draws since that historical draw
- `I(n in draw)` is 1 if n appeared, 0 otherwise

Classification:
```
hot = numbers where score > mean + std
cold = numbers where score < mean - std
neutral = all others
```

### Implementation

From `src/shared/stats.py`:

```python
def compute_heat_scores(draws: list[list[int]],
                        pool_size: int,
                        decay_rate: float = 0.95) -> dict[int, float]:
    scores = {n: 0.0 for n in range(1, pool_size + 1)}
    for age, main in enumerate(reversed(draws)):
        weight = decay_rate ** age
        for n in main:
            scores[n] += weight
    return scores
```

### Typical Mix Strategy

- 40% from hot numbers
- 20% from cold numbers
- 40% from neutral numbers

### Why It Doesn't Work

1. **No Memory** - Draw machines don't "remember" recent results
2. **Mean Reversion Illusion** - Hot numbers cooling off is not predictable timing
3. **Cherry-Picking** - Any historical period can be made to look predictive

## Delta Analysis

### Concept

Analyze the gaps (deltas) between consecutive sorted numbers in winning combinations. Use the historical delta distribution to generate new combinations.

### Mathematical Foundation

For a sorted combination [a₁, a₂, ..., aₖ]:
```
deltas = [a₂ - a₁, a₃ - a₂, ..., aₖ - aₖ₋₁]
```

Build a probability distribution over observed deltas:
```
P(delta = d) = count(d) / total_deltas
```

### Key Finding

Research shows approximately:
- 60% of deltas are ≤ 6
- 15% of deltas are exactly 1 (consecutive numbers)

### Implementation

From `src/shared/stats.py`:

```python
def build_delta_distribution(draws: list[list[int]]) -> dict[int, int]:
    delta_counts = {}
    for main in draws:
        sorted_main = sorted(main)
        for i in range(len(sorted_main) - 1):
            delta = sorted_main[i + 1] - sorted_main[i]
            delta_counts[delta] = delta_counts.get(delta, 0) + 1
    return delta_counts
```

### Why It Doesn't Work

1. **Constrained Space** - Delta patterns emerge from the number pool constraints, not from non-randomness
2. **Distribution is Stable** - The delta distribution is a property of random draws from a finite pool
3. **No Predictive Power** - Knowing past deltas doesn't predict future deltas

## Sum Constraints

### Concept

The sum of winning numbers follows a roughly normal distribution. Filter generated combinations to have sums within the historical range.

### Mathematical Foundation

For n numbers drawn uniformly from [1, N]:
```
Expected sum = n × (N + 1) / 2
Variance ≈ n × (N² - 1) / 12
```

For 6/49:
```
Expected sum = 6 × 50 / 2 = 150
Standard deviation ≈ 33
```

Historical finding: ~71% of 6/49 winners have sums in range 115-185.

### Implementation

From `src/shared/stats.py`:

```python
def generate_sum_constrained(pool_size: int,
                             pick_count: int,
                             min_sum: float,
                             max_sum: float,
                             rng: random.Random) -> list[int]:
    for _ in range(500):  # Max attempts
        line = sorted(rng.sample(range(1, pool_size + 1), pick_count))
        if min_sum <= sum(line) <= max_sum:
            return line
    return None
```

### Why It Doesn't Work

1. **Filtering, Not Prediction** - You're selecting from a subset, not predicting
2. **Most Random Combinations Qualify** - The constraint eliminates few combinations
3. **No Edge** - Jackpot odds remain unchanged

## Pair Correlation

### Concept

Track which number pairs appear together more frequently than expected, then favor these pairs in selections.

### Mathematical Foundation

Expected co-occurrence for pair (a, b):
```
E(co-occur) = total_draws × P(a selected) × P(b selected | a selected)
            = total_draws × (k/N) × ((k-1)/(N-1))
```

Chi-square statistic for pair strength:
```
χ² = (observed - expected)² / expected
```

### Implementation

From `src/shared/stats.py`:

```python
def build_pair_matrix(draws: list[list[int]]) -> dict[tuple[int, int], int]:
    pair_counts = {}
    for main in draws:
        for pair in combinations(sorted(main), 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts
```

### Why It Doesn't Work

1. **Random Fluctuation** - Any pair can appear "strong" by chance
2. **No Persistence** - Strong pairs don't remain strong
3. **Multiple Testing Problem** - With C(49,2) = 1,176 pairs, some will look significant by chance

## Skip/Gap Analysis

### Concept

Track how many draws since each number last appeared. Numbers that haven't appeared for longer than their average gap are "overdue" and weighted higher.

### Mathematical Foundation

Current gap for number n:
```
gap(n) = draws_since_last_appearance(n)
```

Expected gap (for uniform distribution):
```
E(gap) = N / k
```

Where k is numbers per draw.

Overdue score:
```
overdue(n) = gap(n) / E(gap)
```

### Implementation

From `src/shared/stats.py`:

```python
def compute_gaps(draws: list[list[int]], pool_size: int) -> dict[int, int]:
    gaps = {n: len(draws) for n in range(1, pool_size + 1)}
    for age, main in enumerate(reversed(draws)):
        for n in main:
            if gaps[n] == len(draws):
                gaps[n] = age
    return gaps
```

### Why It Doesn't Work

**This is the Gambler's Fallacy in disguise.**

1. **Independence** - The ball machine doesn't know or care when a number last appeared
2. **No "Due" Numbers** - Probability doesn't change based on history
3. **Infinite Expected Wait** - A number can theoretically never appear and still be "normal"

## Balance Strategy

### Concept

Match historical distributions of odd/even and high/low numbers.

### Typical Distributions

For 6/49:
- **Odd/Even**: Most winners have 2-4 odd numbers (not all odd or all even)
- **High/Low**: Most winners have 2-4 high numbers (>25)

### Implementation

```python
def generate_balanced_line(pool_size: int, pick_count: int,
                           target_odd: int, target_high: int,
                           rng: random.Random) -> list[int]:
    mid = (pool_size + 1) // 2

    low_odd = [n for n in range(1, mid) if n % 2 == 1]
    low_even = [n for n in range(1, mid) if n % 2 == 0]
    high_odd = [n for n in range(mid, pool_size + 1) if n % 2 == 1]
    high_even = [n for n in range(mid, pool_size + 1) if n % 2 == 0]

    # Select from each category to match targets
    ...
```

### Why It Doesn't Work

1. **Already Likely** - Most random combinations are already balanced
2. **Marginal Filtering** - Eliminates only extreme combinations
3. **No Predictive Edge** - Doesn't identify the winning combination

## Position Frequency

### Concept

Analyze which numbers appear at each sorted position (1st smallest, 2nd smallest, etc.) and weight accordingly.

### Mathematical Foundation

Position probability:
```
P(n at position p) = count(n at position p) / total_draws
```

### Research Finding

Position distributions for Loto 6/49:
- Position 1 (lowest): heavily weighted toward 1-10
- Position 6 (highest): heavily weighted toward 40-49

This is a mathematical property of sorting, not a predictive feature.

### Why It Doesn't Work

The position distributions are **expected** from random draws. They don't deviate from theoretical expectations in any predictive way.

## Composite Scoring

### Concept

Combine multiple statistical features into a single score for each number.

### Implementation

From `src/shared/advanced_strategies.py`:

```python
def compute_composite_score(n: int, features: dict) -> float:
    score = 0.0
    score += features['frequency'][n] * 0.3
    score += features['recency'][n] * 0.2
    score += features['overdue'][n] * 0.2
    score += features['pair_strength'][n] * 0.15
    score += features['position_fit'][n] * 0.15
    return score
```

### Why It Doesn't Work

Combining multiple ineffective predictors doesn't create an effective predictor. Each component has no predictive power, so their combination has no predictive power.

## Summary Table

| Method    | Mathematical Basis  | Complexity  | Prediction Value |
|-----------|---------------------|-------------|------------------|
| Frequency | Count / Total       | O(n)        | Zero             |
| Hot/Cold  | Exponential decay   | O(n)        | Zero             |
| Delta     | Gap distribution    | O(n)        | Zero             |
| Sum       | Normal distribution | O(1) filter | Zero             |
| Pairs     | Co-occurrence       | O(n²)       | Zero             |
| Skip/Gap  | Time since seen     | O(n)        | Zero             |
| Balance   | Category ratios     | O(1) filter | Zero             |
| Position  | Sorted position     | O(n × k)    | Zero             |
| Composite | Weighted average    | O(n)        | Zero             |

## The Fundamental Problem

All statistical methods share the same flaw: **they analyze past data to predict future random events**.

For prediction to work, there must be:
1. **Dependence** - Future draws depend on past draws ❌
2. **Exploitable patterns** - Patterns that persist and can be captured ❌
3. **Non-randomness** - The generating process is not truly random ❌

None of these conditions hold for properly run lotteries.

## When Statistics Are Useful

Statistical analysis is valuable for:

1. **Verification** - Confirming the lottery is fair and random
2. **Education** - Understanding probability distributions
3. **Entertainment** - Making selections more interesting
4. **Coverage Analysis** - Optimizing wheeling systems

## Next Steps

- [Machine Learning Methods](03-machine-learning-methods.md) - Do neural networks fare any better?
- [Wheeling Systems](04-wheeling-systems.md) - The only mathematically guaranteed approach
- [Honest Assessment](08-honest-assessment.md) - Setting realistic expectations
