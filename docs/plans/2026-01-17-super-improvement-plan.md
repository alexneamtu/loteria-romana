# Super Improvement Plan: Loteria Romana

## Executive Summary

This plan outlines a comprehensive strategy to significantly improve the lottery prediction system through better architecture, advanced statistical modeling, enhanced neural networks, and smarter ensemble methods. The goal is to maximize the probability of detecting non-random patterns (if any exist) while maintaining code quality and testability.

---

## Phase 1: Architecture Refactoring (Foundation)

### 1.1 Eliminate Code Duplication with GameConfig

**Problem**: Massive duplication across `joker_model/`, `loto_649_model/`, `loto_540_model/`

**Solution**: Create a unified game configuration system.

```
src/shared/game_config.py
```

| Game | Pool | Pick | Draw |
|------|------|------|------|
| Joker | 1-45 | 5 | 5+Joker |
| Loto 6/49 | 1-49 | 6 | 6 |
| Loto 5/40 | 1-40 | 5 | 6 |

**Note**: Secondary numbers (Noroc, Super Noroc, Noroc Plus) removed to simplify the system and focus on main number prediction.

**Files to create**:
- `src/shared/game_config.py` - GameConfig dataclass with all parameters
- `src/shared/game_pipeline.py` - Generic pipeline using config

**Files to refactor**:
- Remove duplicated `_sample_weighted()` from all 3 game modules
- Remove duplicated `SoftmaxModel` from all 3 game modules
- Consolidate into shared implementations

**Estimated impact**: -400 lines of duplicated code, 1 place to fix bugs

---

### 1.2 Unified Strategy Base

**Current**: Each strategy hardcodes pool sizes (45, 49, 40)

**Solution**: Strategies accept `GameConfig` and adapt automatically.

```python
class BaseStrategy(Protocol):
    def __init__(self, config: GameConfig): ...
    def get_probabilities(self, draws: list) -> list[float]: ...
    def generate(self, draws: list, count: int, rng) -> list[tuple]: ...
```

---

## Phase 2: Advanced Feature Engineering

### 2.1 New Statistical Features

Add to `src/shared/features.py`:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| **Digit Frequency** | Frequency of digits 0-9 across all numbers | Detect visual patterns humans might follow |
| **Prime Ratio** | % of prime numbers in each draw | Mathematical pattern detection |
| **Modular Residues** | Distribution mod 2, 3, 5, 7, 10 | Reveals hidden periodicity |
| **Position Transitions** | How numbers move between positions | Capture sequential dependencies |
| **Entropy Score** | Shannon entropy of recent draws | Detect non-randomness |
| **Autocorrelation** | Self-similarity at various lags | Time-series patterns |
| **Gap Sequences** | Pattern of gaps between same number | Overdue number refinement |
| **Cluster Detection** | Numbers that co-appear frequently | Network analysis |

### 2.2 Joker Bonus Number Modeling (Joker game only)

**Note**: Noroc/Super Noroc/Noroc Plus have been removed from the system.

For the Joker game, the bonus Joker number (1-20) can still be modeled:

```python
class JokerBonusStrategy:
    """Model Joker bonus number patterns (1-20 range)."""

    def analyze_joker_distribution(self, draws) -> dict:
        """Joker numbers 1-20 might have frequency biases."""
        joker_freq = Counter(d.joker for d in draws)
        return {"frequency": joker_freq, "hot": most_common, "cold": least_common}
```

---

## Phase 3: Neural Network Upgrades

### 3.1 Integrate Advanced Neural Models

**Current**: Game modules use basic softmax regression
**Available but unused**: MLP and LSTM in `shared/neural_base.py`

**Action**: Replace game-specific neural.py with shared implementations

```python
# New: src/shared/neural_strategies.py
class NeuralStrategy(BaseStrategy):
    def __init__(self, config: GameConfig, architecture: str = "mlp"):
        self.model = self._build_model(architecture, config)

    def _build_model(self, arch, config):
        if arch == "mlp":
            return MLPModel([config.pool_size, 64, 32, config.pool_size])
        elif arch == "lstm":
            return LSTMModel(config.pool_size, hidden_size=32)
```

### 3.2 Fix LSTM Backpropagation

**Current issue**: LSTM only updates output layer (line 398 comment)

**Solution**: Implement proper BPTT (Backpropagation Through Time)

```python
def _backward_lstm(self, loss_grad, sequence_length):
    """Full gradient flow through cell states."""
    # Accumulate gradients through time steps
    # Update input, forget, cell, output gate weights
```

### 3.3 Training Improvements

| Improvement | Implementation |
|-------------|----------------|
| Learning rate decay | `lr = initial_lr * (0.95 ** epoch)` |
| Early stopping | Monitor validation loss, stop if no improvement for 10 epochs |
| Gradient clipping | `clip_grad_norm(gradients, max_norm=1.0)` |
| Batch normalization | Normalize layer inputs |
| Dropout simulation | Random zeroing during training |

### 3.4 New Architectures (Future)

- **Attention mechanism**: Focus on most relevant historical draws
- **Transformer encoder**: Self-attention for sequence modeling
- **Mixture of Experts**: Different networks for different number ranges

---

## Phase 4: Ensemble Enhancements

### 4.1 Advanced Probability Combination

**Current**: Simple weighted average
**Better approaches**:

```python
class EnhancedEnsemble:
    def combine_logarithmic(self, probs_list, weights):
        """Logarithmic pooling - better for combining experts."""
        log_probs = [w * np.log(p + 1e-10) for w, p in zip(weights, probs_list)]
        return softmax(sum(log_probs))

    def combine_rank(self, probs_list):
        """Rank aggregation - robust to outliers."""
        ranks = [rankdata(-p) for p in probs_list]
        return softmax(-np.mean(ranks, axis=0))
```

### 4.2 Adaptive Weight Learning

**Current**: Fixed weights or simple performance-based updates
**Solution**: Online learning of ensemble weights

```python
class AdaptiveEnsemble:
    def update_weights_online(self, predictions, actual_draw):
        """Update weights based on prediction accuracy."""
        for i, pred in enumerate(predictions):
            accuracy = self._measure_accuracy(pred, actual_draw)
            self.weights[i] *= (1 + self.learning_rate * accuracy)
        self._normalize_weights()
```

### 4.3 Diversity Maximization

Ensure ensemble members make different predictions:

```python
def measure_diversity(self, strategy_predictions):
    """Compute disagreement between strategies."""
    # Entropy of prediction variance
    # Correlation matrix between strategies
    # Q-statistic for classifier diversity
```

---

## Phase 5: Backtesting & Evaluation

### 5.1 New Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Tier Precision** | TP_tier / Predicted_tier | How often predicted tier matches? |
| **Coverage Rate** | Unique_winning_covered / Total_draws | Diversification measure |
| **Sharpe Analog** | (Return - Baseline) / StdDev(Return) | Risk-adjusted performance |
| **Calmar Ratio** | CAGR / Max_Drawdown | Drawdown-adjusted return |
| **Hit Rate by Tier** | Wins_at_tier / Total_predictions | Per-tier accuracy |

### 5.2 Bootstrap Confidence Intervals

```python
def bootstrap_confidence(results, n_bootstrap=1000, alpha=0.05):
    """More robust than Wilson score for small samples."""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(results, size=len(results), replace=True)
        bootstrap_means.append(np.mean(sample))
    return np.percentile(bootstrap_means, [alpha/2*100, (1-alpha/2)*100])
```

### 5.3 Concept Drift Detection

```python
class DriftDetector:
    """Detect when historical patterns become invalid."""

    def detect_distribution_shift(self, old_draws, new_draws):
        """KL divergence between frequency distributions."""

    def detect_performance_degradation(self, rolling_accuracy):
        """Alert when strategy performance drops significantly."""
```

---

## Phase 6: Wheeling Optimization

### 6.1 Mathematical Covering Designs

**Current**: Greedy algorithm, O(n² × 2^k)
**Better**: Use known covering designs from combinatorics

```python
class OptimalWheelGenerator:
    def get_covering_design(self, n, k, t):
        """Use La Jolla Covering Repository designs."""
        # Pre-computed optimal coverings for common parameters
```

### 6.2 Constraint-Aware Wheels

```python
class FilteredWheelGenerator:
    def generate_with_constraints(self, numbers, constraints):
        """Generate wheel, then filter by constraints."""
        tickets = self.generate_base_wheel(numbers)
        return [t for t in tickets if self._satisfies_constraints(t, constraints)]

    def _satisfies_constraints(self, ticket, constraints):
        # Sum in range
        # Balance requirements
        # No consecutive triplets
```

---

## Phase 7: Data & Pipeline

### 7.1 Extended Historical Data

**Current**: Only recent draws from loto.ro HTML
**Enhancement**:
- Scrape historical archives (if available)
- Store in SQLite for faster queries
- Track data quality metrics

### 7.2 Real-Time Monitoring

```python
class DrawMonitor:
    """Track prediction performance over time."""

    def log_prediction(self, draw_date, predicted, actual):
        """Store for analysis."""

    def generate_performance_report(self, period="monthly"):
        """Automated reporting."""
```

---

## Implementation Priority Matrix

| Phase | Effort | Impact | Priority |
|-------|--------|--------|----------|
| 1.1 GameConfig | Medium | High | **P0** |
| 1.2 Strategy Base | Medium | High | **P0** |
| 2.1 New Features | High | Very High | **P1** |
| 2.2 Joker Bonus | Low | Low | P3 |
| 3.1 Integrate Neural | Medium | High | **P1** |
| 3.2 Fix LSTM | High | Medium | P2 |
| 3.3 Training Improvements | Medium | Medium | P2 |
| 4.1 Advanced Ensemble | Medium | High | **P1** |
| 4.2 Adaptive Weights | Medium | Medium | P2 |
| 5.1 New Metrics | Low | Medium | P2 |
| 5.2 Bootstrap CI | Low | Low | P3 |
| 6.1 Optimal Wheels | High | Low | P3 |
| 7.1 Extended Data | Medium | Medium | P2 |

**Simplification**: Noroc, Super Noroc, and Noroc Plus have been removed from all games to focus on main number prediction.

---

## Success Metrics

1. **Code Quality**
   - Reduce duplication by 50%+ (target: <1000 lines removed)
   - All tests still pass after refactoring
   - New features have 90%+ test coverage

2. **Prediction Quality**
   - Backtest win rate improvement: target +10% relative
   - Higher tier hit rate (4+ matches)
   - Better calibration (predicted probability matches actual frequency)

3. **Performance**
   - Wheel generation 2x faster for 12+ numbers
   - Backtesting 3x faster with parallelization
   - Neural training convergence in 50% fewer epochs

---

## Execution Plan

### Sprint 1 (Week 1-2): Foundation
- [ ] Create `GameConfig` and `GamePipeline`
- [ ] Refactor strategies to use config
- [ ] Remove duplication from game modules
- [ ] Update all tests

### Sprint 2 (Week 3-4): Features
- [ ] Implement new statistical features
- [ ] Add secondary number modeling
- [ ] Create feature extraction pipeline
- [ ] Add feature importance analysis

### Sprint 3 (Week 5-6): Neural
- [ ] Integrate `MLPModel` into game pipelines
- [ ] Add training improvements (LR decay, early stopping)
- [ ] Implement proper LSTM backpropagation
- [ ] Benchmark old vs new neural performance

### Sprint 4 (Week 7-8): Ensemble & Evaluation
- [ ] Implement logarithmic pooling
- [ ] Add adaptive weight learning
- [ ] New backtesting metrics
- [ ] Concept drift detection

### Sprint 5 (Week 9-10): Polish
- [ ] Performance optimization
- [ ] Extended test coverage
- [ ] Documentation updates
- [ ] Final benchmarking

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking changes during refactor | Comprehensive test suite, incremental changes |
| Neural improvements don't help | A/B test against baseline, rollback if worse |
| Overfitting to historical data | Cross-validation, holdout test set |
| Complexity increases maintenance | Keep interfaces simple, document thoroughly |

---

## Conclusion

This plan transforms the lottery system from a collection of duplicated game modules into a unified, extensible framework with advanced statistical modeling and neural network capabilities. The phased approach ensures continuous improvement while maintaining stability.

**Expected outcome**: A significantly more sophisticated prediction system that maximizes the probability of detecting any exploitable patterns, while acknowledging that lottery outcomes are fundamentally random.
