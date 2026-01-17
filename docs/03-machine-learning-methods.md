# Machine Learning Methods for Lottery Analysis

This document covers machine learning approaches to lottery number selection, including neural networks (MLP, LSTM), ensemble methods, and why these sophisticated techniques cannot improve prediction accuracy for random processes.

## The Fundamental Limitation

> "A neural network can only learn patterns that exist. In a truly random lottery, there are no patterns to learn."

All machine learning methods documented here will converge to predicting **uniform probability** for all numbers when trained on sufficient lottery data. This is the correct output for a random process.

## Methods Overview

| Method | Architecture | Purpose | Prediction Value |
|--------|-------------|---------|------------------|
| Softmax Regression | Single layer | Baseline probability | Zero |
| MLP | Multi-layer | Feature interaction | Zero |
| LSTM | Recurrent | Sequence patterns | Zero |
| Ensemble | Model voting | Aggregation | Zero |

## Softmax Regression

### Concept

A single-layer neural network that converts feature inputs to probability distributions over the number pool.

### Mathematical Foundation

For input features x and number n:
```
z_n = w_n · x + b_n
P(n) = exp(z_n) / Σ exp(z_i)
```

This is equivalent to logistic regression with multiple classes.

### Implementation

From `src/shared/neural_base.py`:

```python
def softmax(logits: list[float]) -> list[float]:
    """Compute softmax probabilities."""
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_logits)
    return [e / total for e in exp_logits]

class SoftmaxModel:
    def __init__(self, input_size: int, output_size: int):
        self.weights = [[0.0] * input_size for _ in range(output_size)]
        self.biases = [0.0] * output_size

    def forward(self, features: list[float]) -> list[float]:
        logits = []
        for i in range(len(self.biases)):
            z = sum(w * x for w, x in zip(self.weights[i], features))
            z += self.biases[i]
            logits.append(z)
        return softmax(logits)
```

### Training

Uses cross-entropy loss with gradient descent:
```
L = -Σ y_n × log(P(n))
∂L/∂w = (P(n) - y_n) × x
```

### Why It Doesn't Work

The optimal weights for truly random data converge to:
- All weights → 0
- Output → uniform distribution (1/N for each number)

This is mathematically correct - the model learns there's nothing to learn.

## Multi-Layer Perceptron (MLP)

### Concept

Multiple hidden layers allow the network to learn complex non-linear relationships between inputs and outputs.

### Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
    (features)    (ReLU/tanh)     (softmax)
```

### Implementation

From `src/shared/neural_base.py`:

```python
class MLPModel:
    def __init__(self, input_size: int, hidden_sizes: list[int],
                 output_size: int, l2_lambda: float = 0.01):
        self.layers = []
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            layer = {
                'weights': self._init_weights(sizes[i], sizes[i+1]),
                'biases': [0.0] * sizes[i+1]
            }
            self.layers.append(layer)

        self.l2_lambda = l2_lambda

    def forward(self, x: list[float]) -> list[float]:
        for i, layer in enumerate(self.layers[:-1]):
            x = self._matmul(x, layer['weights'])
            x = [xi + bi for xi, bi in zip(x, layer['biases'])]
            x = [max(0, xi) for xi in x]  # ReLU

        # Output layer with softmax
        x = self._matmul(x, self.layers[-1]['weights'])
        x = [xi + bi for xi, bi in zip(x, self.layers[-1]['biases'])]
        return softmax(x)
```

### Regularization

L2 regularization prevents overfitting:
```
L_total = L_cross_entropy + λ × Σ w²
```

### Why It Doesn't Work

1. **No signal to learn** - The labels (winning numbers) are random
2. **Regularization dominates** - L2 pushes weights toward zero
3. **Universal approximation irrelevant** - Can approximate any function, but the true function is uniform random

### Research Finding

> "SmileyNet" (multi-layer for lottery) converged to outputting near-uniform probabilities after sufficient training.

## LSTM (Long Short-Term Memory)

### Concept

Recurrent architecture designed to capture temporal patterns in sequences. Theoretically could capture if there were dependencies between consecutive draws.

### Architecture

```
Draw(t-n) → Draw(t-n+1) → ... → Draw(t-1) → Prediction(t)
    ↓           ↓                   ↓            ↓
  LSTM       LSTM               LSTM       Dense → Softmax
```

### LSTM Cell Mathematics

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     # Input gate
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     # Output gate

c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  # Candidate memory
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t         # New memory
h_t = o_t ⊙ tanh(c_t)                    # New hidden state
```

### Implementation

From `src/shared/neural_base.py`:

```python
class LSTMModel:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.hidden_size = hidden_size

        # Gate weights
        self.Wf = self._init_weights(input_size + hidden_size, hidden_size)
        self.Wi = self._init_weights(input_size + hidden_size, hidden_size)
        self.Wo = self._init_weights(input_size + hidden_size, hidden_size)
        self.Wc = self._init_weights(input_size + hidden_size, hidden_size)

        # Output projection
        self.Wy = self._init_weights(hidden_size, output_size)

    def forward(self, sequence: list[list[float]]) -> list[float]:
        h = [0.0] * self.hidden_size
        c = [0.0] * self.hidden_size

        for x in sequence:
            combined = x + h

            f = self._sigmoid(self._matmul(combined, self.Wf))
            i = self._sigmoid(self._matmul(combined, self.Wi))
            o = self._sigmoid(self._matmul(combined, self.Wo))
            c_tilde = self._tanh(self._matmul(combined, self.Wc))

            c = [f_i * c_i + i_i * ct_i
                 for f_i, c_i, i_i, ct_i in zip(f, c, i, c_tilde)]
            h = [o_i * math.tanh(c_i) for o_i, c_i in zip(o, c)]

        return softmax(self._matmul(h, self.Wy))
```

### Why It Doesn't Work

1. **No temporal dependency** - "This week's winning numbers have no dependence to the previous week's winning numbers" (Markov chain research)
2. **Gates learn nothing** - Input/forget gates converge to balanced values
3. **Sequence is noise** - The model cannot distinguish signal from noise because there is only noise

### Research Finding

> LSTM models on Powerball/Mega Millions: "any sign of patterns or suspicious data was not found" (Ahmad-Alam/Lottery-Prediction)

## Ensemble Methods

### Concept

Combine predictions from multiple models or strategies to potentially achieve better results than any single approach.

### Voting Methods

**Hard Voting:**
```
prediction = mode(model_1, model_2, ..., model_n)
```

**Soft Voting (Weighted):**
```
P(n) = Σ w_i × P_i(n)
```

### Implementation

From `src/shared/ensemble.py`:

```python
class EnsembleVoter:
    def __init__(self, strategies: list, weights: list[float] = None):
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)

    def get_combined_probabilities(self, draws: list[list[int]]) -> list[float]:
        combined = [0.0] * self.pool_size
        total_weight = sum(self.weights)

        for strategy, weight in zip(self.strategies, self.weights):
            probs = strategy.get_probabilities(draws)
            for i, p in enumerate(probs):
                combined[i] += (weight / total_weight) * p

        return combined
```

### Strategy Selection

Automatically select the best-performing strategy based on backtesting:

```python
class StrategySelector:
    def select_best(self, strategies: list, draws: list) -> Strategy:
        results = []
        for strategy in strategies:
            score = self.backtest(strategy, draws)
            results.append((strategy, score))

        return max(results, key=lambda x: x[1])[0]
```

### Why It Doesn't Work

**The Wisdom of Crowds fails when all participants are wrong.**

1. **No diverse information** - All models are looking at the same meaningless data
2. **Averaging noise** - The average of random predictions is still random
3. **False confidence** - Agreement among models doesn't indicate correctness

## Feature Engineering

### Common Features

| Feature | Description | Code Location |
|---------|-------------|---------------|
| Frequency | Historical count | `src/shared/features.py` |
| Recency | Draws since appearance | `src/shared/features.py` |
| Position frequency | Sorted position stats | `src/shared/features.py` |
| Delta patterns | Gap distributions | `src/shared/stats.py` |
| Digit frequency | Individual digit patterns | `src/shared/features.py` |
| Prime ratio | Proportion of primes | `src/shared/features.py` |
| Modular residues | Periodicity (mod 2,3,5,7,10) | `src/shared/features.py` |
| Entropy | Randomness measure | `src/shared/features.py` |
| Autocorrelation | Self-similarity | `src/shared/features.py` |

### Implementation

From `src/shared/features.py`:

```python
def extract_features(draws: list[list[int]], pool_size: int) -> dict:
    return {
        'frequency': compute_frequency_features(draws, pool_size),
        'recency': compute_recency_features(draws, pool_size),
        'position': compute_position_features(draws, pool_size),
        'digit': compute_digit_features(draws),
        'prime': compute_prime_features(draws),
        'modular': compute_modular_features(draws),
        'entropy': compute_entropy_features(draws),
    }
```

### Why Feature Engineering Doesn't Work

**Features describe the past, not the future.**

All features capture properties of historical draws. Since future draws are independent of past draws, these features have no predictive power.

## Hyperparameter Tuning

### Common Hyperparameters

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| Learning rate | 0.001 - 0.1 | Convergence speed |
| Hidden size | 32 - 256 | Model capacity |
| L2 lambda | 0.001 - 0.1 | Regularization strength |
| Dropout | 0.1 - 0.5 | Overfitting prevention |
| Epochs | 10 - 1000 | Training duration |

### Tuning Methods

1. **Grid Search** - Try all combinations
2. **Random Search** - Sample randomly
3. **Bayesian Optimization** - Model the parameter space

### Why Hyperparameter Tuning Doesn't Work

Tuning optimizes for **training performance**, but there's no real pattern to capture. Better training scores don't translate to better predictions.

## Overfitting Analysis

### Signs of Overfitting

1. Training loss decreases while validation loss increases
2. Model memorizes specific winning combinations
3. Predictions become highly concentrated on few numbers

### Prevention Methods

1. **Regularization** - L2, dropout
2. **Early stopping** - Stop when validation loss plateaus
3. **Cross-validation** - Use rolling window validation

### The Irony

With lottery data, both overfitting and underfitting lead to the same result: **useless predictions**. The optimal model for random data is one that predicts uniform probabilities.

## Comparison with Real Applications

| Application | Data Type | Patterns Exist? | ML Works? |
|-------------|-----------|-----------------|-----------|
| Image recognition | Structured | Yes | Yes |
| Natural language | Structured | Yes | Yes |
| Stock prices | Semi-random | Debatable | Debatable |
| Weather | Physical laws | Yes | Yes |
| **Lottery** | **Truly random** | **No** | **No** |

## Published Research Results

### "Predicting Winning Lottery Numbers" (ArXiv)
- Method: Bayesian/Dirichlet approach
- Result: No improvement over random

### "SmileyNet - Reading Tea Leaves with AI" (ArXiv)
- Method: Deep neural network
- Result: "Predictions not guaranteed" due to randomness

### "Lottery Prediction" (Ahmad-Alam/GitHub)
- Method: 4-layer LSTM
- Result: "Any sign of patterns was not found"

### "SSRN: Predicting Seemingly Random Risk-Taking"
- Method: Various ML
- Result: 53% accuracy (barely above 50% random baseline)

## Practical Recommendations

If you still want to use ML approaches:

1. **Accept the limitations** - Use for entertainment, not profit
2. **Use simple models** - Complex models waste computation
3. **Monitor for overfitting** - Though it doesn't matter for prediction
4. **Combine with wheeling** - The only mathematically guaranteed approach

## Summary

| Method | Theoretical Power | Practical Power | Recommendation |
|--------|------------------|-----------------|----------------|
| Softmax | Baseline | None | Skip |
| MLP | Feature interaction | None | Skip |
| LSTM | Sequence learning | None | Skip |
| Ensemble | Aggregation | None | Skip |
| All ML | Pattern recognition | **None for random data** | Accept limitations |

## The Honest Conclusion

> Machine learning is powerful when patterns exist. Lotteries are designed to have no patterns. Using ML for lottery prediction is like using a metal detector to find ghosts - the tool is excellent, but there's nothing to find.

## Next Steps

- [Wheeling Systems](04-wheeling-systems.md) - The only approach with mathematical guarantees
- [Backtesting Results](07-backtesting-results.md) - Empirical comparison of strategies
- [Honest Assessment](08-honest-assessment.md) - Setting realistic expectations
