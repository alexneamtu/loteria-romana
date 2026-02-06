# Advanced Lottery Algorithm Research

## Goal

Research and implement better algorithms for lottery number selection across all three games (Joker, Loto 6/49, Loto 5/40). Three research areas, prioritized ML-first:

1. **Cutting-edge ML approaches** — LSTM, TCN, RL, genetic algorithms, gradient boosting, improved transformers, normalizing flows
2. **Statistical bias detection** — Autocorrelation, drift detection, position analysis, regime detection
3. **Combinatorial coverage** — Covering designs, portfolio optimization for ticket sets

All algorithms must integrate into the existing strategy protocol and ensemble system with backtesting.

## Current State

Already implemented:
- Random, Frequency (recency-weighted), Bayesian (Dirichlet), Co-occurrence (graph-based)
- Neural (softmax regression), Markov chains, Composite scoring (6 factors)
- Ensemble blend with chi-square bias detection
- Optional: Transformer (encoder-decoder) and VAE (PyTorch)
- Walk-forward backtesting, EV analysis, NIST randomness tests
- ~1,000-1,250 historical draws per game

## New Algorithms

### ML Strategies

#### 1. LSTM Sequence Model
- Sliding window of last N draws (10-20) as input sequences
- Learns temporal patterns: momentum shifts, seasonal effects, multi-draw dependencies
- Output: probability distribution over numbers, sampled to generate picks
- Dependency: PyTorch (optional import)
- Module: `shared/lstm.py`

#### 2. Temporal Convolutional Network (TCN)
- Dilated causal convolutions over draw history
- Faster training than LSTM, better at long-range dependencies
- Explicit control over receptive field size, parallelizable
- Same input/output interface as LSTM
- Dependency: PyTorch (optional import)
- Module: `shared/tcn.py`

#### 3. Genetic Algorithm Optimizer
- Evolve a population of ticket sets over generations
- Fitness: backtest score against historical draws (prize matches)
- Crossover: combine number selections from parent tickets
- Mutation: random number swaps
- Optimizes ticket sets directly against scoring function
- Dependency: stdlib only
- Module: `shared/genetic.py`

#### 4. Reinforcement Learning Agent
- State: recent draw history + current partial ticket
- Action: select next number
- Reward: prize tier achieved
- Training: policy gradient (REINFORCE) or Q-learning
- Learns selection policies rather than probability distributions
- Dependency: PyTorch (optional import)
- Module: `shared/rl_agent.py`

#### 5. Gradient Boosted Trees
- Leverage existing feature engineering from `shared/features.py`
- Per-number binary classifier: "will number X appear in next draw?"
- Features: digit frequency, prime ratio, gap distributions, entropy, consecutive patterns, odd/even, sum stats
- Dependency: scikit-learn (baseline), optional XGBoost
- Module: `shared/gradient_boost.py`

#### 6. Autoregressive Transformer (improved)
- Replace existing encoder-decoder with decoder-only (GPT-style)
- Generate numbers one at a time, conditioned on previous numbers + draw history
- Naturally handles "without replacement" constraint
- Can learn positional preferences
- Dependency: PyTorch (optional import)
- Module: replace/extend `shared/transformer_model.py`

#### 7. Normalizing Flows
- Generative model: transform uniform distribution into empirical draw distribution
- Exact likelihood computation (unlike existing VAE)
- Score how "typical" a candidate ticket is
- Detect subtle distributional shifts over time
- Dependency: PyTorch (optional import)
- Module: `shared/normalizing_flows.py`

### Bias Detection

#### 8. Enhanced Bias Detection Pipeline
Expand beyond current chi-square test:
- **Autocorrelation analysis** — are draws correlated across time lags?
- **Per-number runs tests** — individual number appearance streaks
- **Position analysis** — do numbers cluster in certain draw positions?
- **Drift detection** — CUSUM and ADWIN algorithms to identify distribution shifts (equipment changes, wear)
- **Regime detection** — split data into regimes, backtest within regimes
- Module: extend `shared/bias_detection.py`

### Combinatorial Coverage

#### 9. Covering Designs
- Given N tickets, maximize probability of hitting at least one prize tier
- Lottery wheel / covering design algorithms
- Guarantee: if K chosen numbers are drawn, at least one ticket has 3+ matches
- Greedy set-cover approximation (stdlib), optional ILP solver (scipy.optimize)
- Module: `shared/covering_designs.py`

#### 10. Portfolio Optimization
- Treat ticket selection as mean-variance portfolio problem
- Each ticket has expected return and covariance with other tickets (overlapping numbers)
- Markowitz-style optimization: maximize expected coverage, minimize redundancy
- Dependency: scipy (optional)
- Module: `shared/portfolio.py`

## Integration Architecture

Every new algorithm implements the existing `Strategy` protocol:

```python
class Strategy(Protocol):
    name: str
    def generate(draws, count, rng, ...) -> list[list[int]]
    def get_probabilities(draws, ...) -> list[float]
```

- Heavy dependencies use optional-import pattern: `try: import torch`
- Each strategy lives in `shared/` as a new module
- Ensemble blend extended to include new strategies in its pool
- Backtesting automatically evaluates all available strategies

## Backtesting Enhancements

- **EV scoring** — weight matches by prize tier payouts, not just binary win/loss
- **Significance gates** — strategy must pass binomial test vs random (p < 0.05) to enter ensemble
- **Regime-aware backtesting** — split data by detected regimes, evaluate separately
- **Monte Carlo validation** — generate synthetic fair-lottery data, verify no strategy beats random (overfitting check)

## Anti-Overfitting Measures

With 10+ strategies and ~1,000-1,200 draws, overfitting is the primary risk:
- **Temporal cross-validation** — always train on past, test on future, never shuffle
- **Bonferroni correction** — adjust significance thresholds for multiple comparisons
- **Out-of-sample holdout** — reserve most recent 100 draws, never touch during development
- **Complexity budget** — track parameter count per strategy, penalize unjustified capacity

## Success Metrics

- **Primary**: Any strategy achieves statistically significant positive EV on holdout data
- **Secondary**: Optimized ensemble outperforms uniform-random on prize coverage
- **Tertiary**: Covering designs provide provably better worst-case guarantees

## Implementation Phases

### Phase 1 — Foundation (infra + quick wins)
- Enhance backtesting with EV scoring, significance gates, Monte Carlo validation
- Add holdout set management to storage layer
- Implement Genetic Algorithm optimizer (stdlib)
- Implement Gradient Boosted Trees (scikit-learn + existing features)

### Phase 2 — Deep Learning Models
- LSTM sequence model
- Temporal Convolutional Network
- Autoregressive Transformer (replace existing encoder-decoder)
- Normalizing Flows

### Phase 3 — Bias Detection + Coverage
- Enhanced bias detection pipeline
- Covering design algorithms
- Portfolio optimization for ticket sets

### Phase 4 — Reinforcement Learning + Integration
- RL agent with policy gradient training
- Extend ensemble blend to include all new strategies
- Significance-gated strategy admission
- Final holdout evaluation across all games

### Phase 5 — Validation + Report
- Monte Carlo sanity checks on synthetic data
- Cross-game comparison (do patterns transfer?)
- Comprehensive analysis report with statistical conclusions
- Document findings: which approaches work, which don't, and why
