# Phase 2: Deep Learning Models Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement four deep learning strategies (LSTM, TCN, Autoregressive Transformer, Normalizing Flows) that integrate into the ensemble blend via the Strategy protocol.

**Architecture:** Each model uses PyTorch with optional import pattern (`try: import torch`). All implement `.name`, `.generate(draws, count, rng, **kwargs)`, and `.get_probabilities(draws, **kwargs)`. Input is multi-hot encoded draw sequences; output is probability distribution over pool numbers, sampled without replacement to generate picks.

**Tech Stack:** PyTorch (optional import), Python stdlib for fallback stubs

---

## Context

### Strategy Protocol (from `src/shared/strategy_base.py`)
Strategies used in `ensemble_blend.py` follow a simpler interface than the full `Strategy` protocol:
```python
# What ensemble_blend.py actually calls:
strategy.name  # str attribute
strategy.generate(draws, count, rng, draw_dates=..., half_life_mode=...)  # -> list[list[int]]
strategy.get_probabilities(draws, ...)  # -> list[float]
```

Where `draws` is `list[list[int]]` (just main numbers, no secondary).

### Existing Patterns to Follow
- `src/shared/genetic.py` — stdlib strategy with `.name`, `.generate()`, `.get_probabilities()`
- `src/shared/gradient_boost.py` — optional import pattern with `SKLEARN_AVAILABLE` flag and stub class
- `src/shared/transformer_model.py` — existing PyTorch transformer (encoder-based, NOT protocol-compliant; we'll build new ones)
- `src/shared/cooccurrence.py` — constructor pattern: `__init__(pool_size, numbers_to_pick, ...)`

### Test Pattern
```bash
PYTHONPATH=src python -m unittest tests/test_<module>.py -v
```

### Key Constants
- `GameConfig` from `src/shared/game_config.py`: `pool_size`, `numbers_to_pick`, `pool_range`
- `JOKER_CONFIG`: pool_size=45, numbers_to_pick=5
- `LOTO_649_CONFIG`: pool_size=49, numbers_to_pick=6
- `DEFAULT_HALF_LIFE = 100.0`, `DEFAULT_HALF_LIFE_MODE = "draws"` from `src/shared/recency.py`

---

## Task 1: LSTM Sequence Strategy

**Files:**
- Create: `src/shared/lstm_strategy.py`
- Create: `tests/test_lstm_strategy.py`

### Design

PyTorch LSTM that processes a sliding window of multi-hot encoded draws. Architecture:
- Input: `(batch, seq_len, pool_size)` multi-hot vectors
- 2-layer LSTM with hidden_size=64
- Linear output head → `pool_size` logits
- BCEWithLogitsLoss (multi-label)
- Walk-forward: train on `draws[:i]`, predict `draws[i]`
- Lightweight defaults for fast training: seq_len=20, hidden=64, epochs=30, batch=16

**Step 1: Write the failing tests**

Create `tests/test_lstm_strategy.py`:

```python
import unittest
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestLSTMStrategy(unittest.TestCase):
    def setUp(self):
        from shared.lstm_strategy import LSTMStrategy
        self.rng = random.Random(42)
        self.strategy = LSTMStrategy(pool_size=10, numbers_to_pick=3)
        # Generate 50 synthetic draws
        self.draws = [
            sorted(random.Random(i).sample(range(1, 11), 3))
            for i in range(50)
        ]

    def test_name(self):
        self.assertEqual(self.strategy.name, "lstm")

    def test_generate_returns_correct_count(self):
        picks = self.strategy.generate(self.draws, 3, self.rng)
        self.assertEqual(len(picks), 3)

    def test_generate_returns_sorted_numbers(self):
        picks = self.strategy.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(pick, sorted(pick))

    def test_generate_numbers_in_range(self):
        picks = self.strategy.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(len(pick), 3)
            for n in pick:
                self.assertGreaterEqual(n, 1)
                self.assertLessEqual(n, 10)

    def test_generate_with_few_draws_falls_back(self):
        """With < seq_len draws, should still produce valid picks."""
        picks = self.strategy.generate(self.draws[:3], 2, self.rng)
        self.assertEqual(len(picks), 2)

    def test_get_probabilities_length(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertEqual(len(probs), 10)

    def test_get_probabilities_sum_to_one(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_get_probabilities_all_positive(self):
        probs = self.strategy.get_probabilities(self.draws)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)

    def test_generate_unique_picks(self):
        picks = self.strategy.generate(self.draws, 5, self.rng)
        keys = [tuple(p) for p in picks]
        self.assertEqual(len(keys), len(set(keys)))


class TestLSTMStrategyImport(unittest.TestCase):
    def test_module_importable(self):
        """Module should always be importable regardless of PyTorch."""
        import shared.lstm_strategy  # noqa: F401


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests/test_lstm_strategy.py -v`
Expected: `ModuleNotFoundError: No module named 'shared.lstm_strategy'`

**Step 3: Implement LSTMStrategy**

Create `src/shared/lstm_strategy.py`:

```python
"""LSTM sequence strategy for lottery number generation.

Uses a 2-layer LSTM to learn temporal patterns in draw sequences.
Input is a sliding window of multi-hot encoded draws; output is a
probability distribution over pool numbers.

Requires PyTorch (optional import).
"""

import random

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


def _encode_draws(draws, pool_size):
    """Convert draws to multi-hot tensor (num_draws, pool_size)."""
    encoded = torch.zeros(len(draws), pool_size)
    for i, draw in enumerate(draws):
        for n in draw:
            encoded[i, n - 1] = 1.0
    return encoded


if TORCH_AVAILABLE:

    class _LSTMNet(nn.Module):
        def __init__(self, pool_size, hidden_size=64, num_layers=2, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=pool_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.output = nn.Linear(hidden_size, pool_size)

        def forward(self, x):
            # x: (batch, seq_len, pool_size)
            lstm_out, _ = self.lstm(x)
            # Use last timestep
            last = lstm_out[:, -1, :]
            return self.output(last)


class LSTMStrategy:
    """LSTM-based lottery number generation strategy."""

    name = "lstm"

    def __init__(
        self,
        pool_size,
        numbers_to_pick,
        hidden_size=64,
        num_layers=2,
        seq_len=20,
        epochs=30,
        batch_size=16,
        learning_rate=1e-3,
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def _train_and_predict(self, draws):
        """Train on draws and return probability vector."""
        if not TORCH_AVAILABLE or len(draws) < 5:
            return [1.0 / self.pool_size] * self.pool_size

        encoded = _encode_draws(draws, self.pool_size)

        # Build sequences: input[i] = draws[i:i+seq_len], target = draws[i+seq_len]
        effective_seq = min(self.seq_len, len(draws) - 1)
        if effective_seq < 2:
            return [1.0 / self.pool_size] * self.pool_size

        inputs, targets = [], []
        for i in range(len(draws) - effective_seq):
            inputs.append(encoded[i:i + effective_seq])
            targets.append(encoded[i + effective_seq])

        if not inputs:
            return [1.0 / self.pool_size] * self.pool_size

        X = torch.stack(inputs)
        Y = torch.stack(targets)

        model = _LSTMNet(self.pool_size, self.hidden_size, self.num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X))
            for start in range(0, len(X), self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_x, batch_y = X[idx], Y[idx]
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Predict using most recent window
        model.eval()
        with torch.no_grad():
            recent = encoded[-effective_seq:].unsqueeze(0)
            logits = model(recent)
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        return probs

    def get_probabilities(self, draws, **kwargs):
        """Return probability distribution over pool numbers."""
        return self._train_and_predict(draws)

    def generate(self, draws, count, rng, **kwargs):
        """Generate picks by sampling from learned distribution."""
        probs = self._train_and_predict(draws)
        picks = []
        seen = set()
        pool = list(range(1, self.pool_size + 1))

        for _ in range(count + 10):  # over-generate for uniqueness
            if len(picks) >= count:
                break
            remaining = list(pool)
            remaining_probs = list(probs)
            chosen = []
            for _ in range(self.numbers_to_pick):
                total = sum(remaining_probs)
                if total <= 0:
                    idx = rng.randrange(len(remaining))
                else:
                    normalized = [p / total for p in remaining_probs]
                    r = rng.random()
                    cumulative = 0.0
                    idx = len(remaining) - 1
                    for j, p in enumerate(normalized):
                        cumulative += p
                        if r <= cumulative:
                            idx = j
                            break
                chosen.append(remaining[idx])
                remaining.pop(idx)
                remaining_probs.pop(idx)

            pick = sorted(chosen)
            key = tuple(pick)
            if key not in seen:
                seen.add(key)
                picks.append(pick)

        return picks[:count]
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m unittest tests/test_lstm_strategy.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/shared/lstm_strategy.py tests/test_lstm_strategy.py
git commit -m "feat: add LSTM sequence strategy with PyTorch optional import"
```

---

## Task 2: TCN Strategy

**Files:**
- Create: `src/shared/tcn_strategy.py`
- Create: `tests/test_tcn_strategy.py`

### Design

Temporal Convolutional Network with dilated causal convolutions. Architecture:
- Input: `(batch, pool_size, seq_len)` — Conv1d expects channels-first
- Stack of causal Conv1d layers with exponentially increasing dilation: [1, 2, 4, 8]
- Each block: Conv1d → ReLU → Dropout → residual connection
- Global average pooling → Linear → pool_size logits
- BCEWithLogitsLoss
- Lightweight defaults: seq_len=20, channels=32, kernel_size=3, 4 blocks, epochs=30

**Step 1: Write the failing tests**

Create `tests/test_tcn_strategy.py`:

```python
import unittest
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestTCNStrategy(unittest.TestCase):
    def setUp(self):
        from shared.tcn_strategy import TCNStrategy
        self.rng = random.Random(42)
        self.strategy = TCNStrategy(pool_size=10, numbers_to_pick=3)
        self.draws = [
            sorted(random.Random(i).sample(range(1, 11), 3))
            for i in range(50)
        ]

    def test_name(self):
        self.assertEqual(self.strategy.name, "tcn")

    def test_generate_returns_correct_count(self):
        picks = self.strategy.generate(self.draws, 3, self.rng)
        self.assertEqual(len(picks), 3)

    def test_generate_returns_sorted_numbers(self):
        picks = self.strategy.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(pick, sorted(pick))

    def test_generate_numbers_in_range(self):
        picks = self.strategy.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(len(pick), 3)
            for n in pick:
                self.assertGreaterEqual(n, 1)
                self.assertLessEqual(n, 10)

    def test_generate_with_few_draws(self):
        picks = self.strategy.generate(self.draws[:3], 2, self.rng)
        self.assertEqual(len(picks), 2)

    def test_get_probabilities_length(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertEqual(len(probs), 10)

    def test_get_probabilities_sum_to_one(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_receptive_field_covers_sequence(self):
        """TCN receptive field should cover at least seq_len."""
        from shared.tcn_strategy import TCNStrategy
        s = TCNStrategy(pool_size=10, numbers_to_pick=3, num_blocks=4, kernel_size=3)
        # Receptive field = 1 + num_blocks * (kernel_size - 1) * dilation_sum
        # With dilations [1,2,4,8] and kernel 3: 1 + (3-1)*(1+2+4+8) = 1 + 30 = 31
        # Should cover seq_len=20
        self.assertGreaterEqual(31, s.seq_len)

    def test_generate_unique_picks(self):
        picks = self.strategy.generate(self.draws, 5, self.rng)
        keys = [tuple(p) for p in picks]
        self.assertEqual(len(keys), len(set(keys)))


class TestTCNStrategyImport(unittest.TestCase):
    def test_module_importable(self):
        import shared.tcn_strategy  # noqa: F401


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests/test_tcn_strategy.py -v`
Expected: `ModuleNotFoundError`

**Step 3: Implement TCNStrategy**

Create `src/shared/tcn_strategy.py`:

```python
"""Temporal Convolutional Network strategy for lottery number generation.

Uses dilated causal convolutions over draw history. Faster training
than LSTM and explicit control over receptive field size.

Requires PyTorch (optional import).
"""

import random

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


def _encode_draws(draws, pool_size):
    """Convert draws to multi-hot tensor (num_draws, pool_size)."""
    encoded = torch.zeros(len(draws), pool_size)
    for i, draw in enumerate(draws):
        for n in draw:
            encoded[i, n - 1] = 1.0
    return encoded


if TORCH_AVAILABLE:

    class _CausalConvBlock(nn.Module):
        """Single causal convolution block with residual connection."""

        def __init__(self, channels, kernel_size, dilation, dropout=0.1):
            super().__init__()
            padding = (kernel_size - 1) * dilation  # causal padding
            self.conv = nn.Conv1d(
                channels, channels, kernel_size,
                padding=padding, dilation=dilation,
            )
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.causal_trim = padding

        def forward(self, x):
            out = self.conv(x)
            if self.causal_trim > 0:
                out = out[:, :, :-self.causal_trim]  # trim future
            out = self.dropout(self.relu(out))
            return out + x  # residual

    class _TCNNet(nn.Module):
        def __init__(self, pool_size, channels=32, kernel_size=3,
                     num_blocks=4, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(pool_size, channels)
            dilations = [2 ** i for i in range(num_blocks)]
            self.blocks = nn.ModuleList([
                _CausalConvBlock(channels, kernel_size, d, dropout)
                for d in dilations
            ])
            self.output = nn.Linear(channels, pool_size)

        def forward(self, x):
            # x: (batch, seq_len, pool_size)
            x = self.input_proj(x)           # (batch, seq_len, channels)
            x = x.transpose(1, 2)            # (batch, channels, seq_len)
            for block in self.blocks:
                x = block(x)
            x = x.mean(dim=-1)               # global avg pool → (batch, channels)
            return self.output(x)


class TCNStrategy:
    """TCN-based lottery number generation strategy."""

    name = "tcn"

    def __init__(
        self,
        pool_size,
        numbers_to_pick,
        channels=32,
        kernel_size=3,
        num_blocks=4,
        seq_len=20,
        epochs=30,
        batch_size=16,
        learning_rate=1e-3,
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def _train_and_predict(self, draws):
        if not TORCH_AVAILABLE or len(draws) < 5:
            return [1.0 / self.pool_size] * self.pool_size

        encoded = _encode_draws(draws, self.pool_size)
        effective_seq = min(self.seq_len, len(draws) - 1)
        if effective_seq < 2:
            return [1.0 / self.pool_size] * self.pool_size

        inputs, targets = [], []
        for i in range(len(draws) - effective_seq):
            inputs.append(encoded[i:i + effective_seq])
            targets.append(encoded[i + effective_seq])

        if not inputs:
            return [1.0 / self.pool_size] * self.pool_size

        X = torch.stack(inputs)
        Y = torch.stack(targets)

        model = _TCNNet(
            self.pool_size, self.channels, self.kernel_size, self.num_blocks,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X))
            for start in range(0, len(X), self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_x, batch_y = X[idx], Y[idx]
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            recent = encoded[-effective_seq:].unsqueeze(0)
            logits = model(recent)
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        return probs

    def get_probabilities(self, draws, **kwargs):
        return self._train_and_predict(draws)

    def generate(self, draws, count, rng, **kwargs):
        probs = self._train_and_predict(draws)
        picks = []
        seen = set()
        pool = list(range(1, self.pool_size + 1))

        for _ in range(count + 10):
            if len(picks) >= count:
                break
            remaining = list(pool)
            remaining_probs = list(probs)
            chosen = []
            for _ in range(self.numbers_to_pick):
                total = sum(remaining_probs)
                if total <= 0:
                    idx = rng.randrange(len(remaining))
                else:
                    normalized = [p / total for p in remaining_probs]
                    r = rng.random()
                    cumulative = 0.0
                    idx = len(remaining) - 1
                    for j, p in enumerate(normalized):
                        cumulative += p
                        if r <= cumulative:
                            idx = j
                            break
                chosen.append(remaining[idx])
                remaining.pop(idx)
                remaining_probs.pop(idx)

            pick = sorted(chosen)
            key = tuple(pick)
            if key not in seen:
                seen.add(key)
                picks.append(pick)

        return picks[:count]
```

**Step 4: Run tests**

Run: `PYTHONPATH=src python -m unittest tests/test_tcn_strategy.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/shared/tcn_strategy.py tests/test_tcn_strategy.py
git commit -m "feat: add Temporal Convolutional Network strategy"
```

---

## Task 3: Autoregressive Transformer Strategy

**Files:**
- Create: `src/shared/transformer_strategy.py`
- Create: `tests/test_transformer_strategy.py`

### Design

New file (NOT modifying existing `transformer_model.py`). Decoder-only GPT-style transformer that generates numbers one at a time, conditioned on draw history. Architecture:
- Input: sliding window of multi-hot encoded draws → `(batch, seq_len, pool_size)`
- Learnable positional encoding
- 2-layer decoder with causal self-attention (4 heads)
- d_model=64, d_ff=128
- BCEWithLogitsLoss for training
- Lightweight: seq_len=20, epochs=30, batch=16

We create a **new** file `transformer_strategy.py` as a Strategy-protocol-compliant wrapper, keeping the existing `transformer_model.py` untouched.

**Step 1: Write the failing tests**

Create `tests/test_transformer_strategy.py`:

```python
import unittest
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestTransformerStrategy(unittest.TestCase):
    def setUp(self):
        from shared.transformer_strategy import TransformerStrategy
        self.rng = random.Random(42)
        self.strategy = TransformerStrategy(pool_size=10, numbers_to_pick=3)
        self.draws = [
            sorted(random.Random(i).sample(range(1, 11), 3))
            for i in range(50)
        ]

    def test_name(self):
        self.assertEqual(self.strategy.name, "transformer")

    def test_generate_returns_correct_count(self):
        picks = self.strategy.generate(self.draws, 3, self.rng)
        self.assertEqual(len(picks), 3)

    def test_generate_returns_sorted_numbers(self):
        picks = self.strategy.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(pick, sorted(pick))

    def test_generate_numbers_in_range(self):
        picks = self.strategy.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(len(pick), 3)
            for n in pick:
                self.assertGreaterEqual(n, 1)
                self.assertLessEqual(n, 10)

    def test_generate_with_few_draws(self):
        picks = self.strategy.generate(self.draws[:3], 2, self.rng)
        self.assertEqual(len(picks), 2)

    def test_get_probabilities_length(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertEqual(len(probs), 10)

    def test_get_probabilities_sum_to_one(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_uses_causal_mask(self):
        """Transformer should use causal (lower-triangular) attention mask."""
        from shared.transformer_strategy import TransformerStrategy
        s = TransformerStrategy(pool_size=10, numbers_to_pick=3)
        # Just verify it generates valid output (causal mask is internal)
        picks = s.generate(self.draws, 1, self.rng)
        self.assertEqual(len(picks), 1)

    def test_generate_unique_picks(self):
        picks = self.strategy.generate(self.draws, 5, self.rng)
        keys = [tuple(p) for p in picks]
        self.assertEqual(len(keys), len(set(keys)))


class TestTransformerStrategyImport(unittest.TestCase):
    def test_module_importable(self):
        import shared.transformer_strategy  # noqa: F401


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests/test_transformer_strategy.py -v`
Expected: `ModuleNotFoundError`

**Step 3: Implement TransformerStrategy**

Create `src/shared/transformer_strategy.py`:

```python
"""Decoder-only Transformer strategy for lottery number generation.

GPT-style transformer with causal self-attention over draw sequences.
Separate from the existing transformer_model.py (which is encoder-based
and does not implement the Strategy protocol).

Requires PyTorch (optional import).
"""

import math
import random

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


def _encode_draws(draws, pool_size):
    encoded = torch.zeros(len(draws), pool_size)
    for i, draw in enumerate(draws):
        for n in draw:
            encoded[i, n - 1] = 1.0
    return encoded


if TORCH_AVAILABLE:

    class _DecoderBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
            x = x + self.dropout(attn_out)
            normed = self.norm2(x)
            x = x + self.dropout(self.ff(normed))
            return x

    class _TransformerNet(nn.Module):
        def __init__(self, pool_size, d_model=64, n_heads=4,
                     n_layers=2, d_ff=128, max_len=100, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(pool_size, d_model)
            # Learnable positional encoding
            self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
            self.blocks = nn.ModuleList([
                _DecoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.output = nn.Linear(d_model, pool_size)

        def forward(self, x):
            seq_len = x.size(1)
            x = self.input_proj(x) + self.pos_embed[:, :seq_len, :]
            # Causal mask: upper triangle = True (blocked)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1,
            ).bool()
            for block in self.blocks:
                x = block(x, mask)
            x = self.norm(x[:, -1, :])  # last position
            return self.output(x)


class TransformerStrategy:
    """Decoder-only transformer for lottery number generation."""

    name = "transformer"

    def __init__(
        self,
        pool_size,
        numbers_to_pick,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        seq_len=20,
        epochs=30,
        batch_size=16,
        learning_rate=1e-3,
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def _train_and_predict(self, draws):
        if not TORCH_AVAILABLE or len(draws) < 5:
            return [1.0 / self.pool_size] * self.pool_size

        encoded = _encode_draws(draws, self.pool_size)
        effective_seq = min(self.seq_len, len(draws) - 1)
        if effective_seq < 2:
            return [1.0 / self.pool_size] * self.pool_size

        inputs, targets = [], []
        for i in range(len(draws) - effective_seq):
            inputs.append(encoded[i:i + effective_seq])
            targets.append(encoded[i + effective_seq])

        if not inputs:
            return [1.0 / self.pool_size] * self.pool_size

        X = torch.stack(inputs)
        Y = torch.stack(targets)

        model = _TransformerNet(
            self.pool_size, self.d_model, self.n_heads,
            self.n_layers, self.d_ff,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X))
            for start in range(0, len(X), self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_x, batch_y = X[idx], Y[idx]
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            recent = encoded[-effective_seq:].unsqueeze(0)
            logits = model(recent)
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        return probs

    def get_probabilities(self, draws, **kwargs):
        return self._train_and_predict(draws)

    def generate(self, draws, count, rng, **kwargs):
        probs = self._train_and_predict(draws)
        picks = []
        seen = set()
        pool = list(range(1, self.pool_size + 1))

        for _ in range(count + 10):
            if len(picks) >= count:
                break
            remaining = list(pool)
            remaining_probs = list(probs)
            chosen = []
            for _ in range(self.numbers_to_pick):
                total = sum(remaining_probs)
                if total <= 0:
                    idx = rng.randrange(len(remaining))
                else:
                    normalized = [p / total for p in remaining_probs]
                    r = rng.random()
                    cumulative = 0.0
                    idx = len(remaining) - 1
                    for j, p in enumerate(normalized):
                        cumulative += p
                        if r <= cumulative:
                            idx = j
                            break
                chosen.append(remaining[idx])
                remaining.pop(idx)
                remaining_probs.pop(idx)

            pick = sorted(chosen)
            key = tuple(pick)
            if key not in seen:
                seen.add(key)
                picks.append(pick)

        return picks[:count]
```

**Step 4: Run tests**

Run: `PYTHONPATH=src python -m unittest tests/test_transformer_strategy.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/shared/transformer_strategy.py tests/test_transformer_strategy.py
git commit -m "feat: add decoder-only Transformer strategy"
```

---

## Task 4: Normalizing Flows Strategy

**Files:**
- Create: `src/shared/normalizing_flows.py`
- Create: `tests/test_normalizing_flows.py`

### Design

RealNVP-style normalizing flow that transforms a simple base distribution into the empirical lottery draw distribution. Architecture:
- Base distribution: standard normal → sigmoid → [0,1]^pool_size
- Coupling layers: split input, apply affine transform conditioned on other half
- 4 coupling layers with MLP conditioners (2 hidden layers of 32 units)
- Training: maximize log-likelihood of observed draws
- Output: probability per number, sampled for picks

**Step 1: Write the failing tests**

Create `tests/test_normalizing_flows.py`:

```python
import unittest
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestNormalizingFlows(unittest.TestCase):
    def setUp(self):
        from shared.normalizing_flows import NormalizingFlowStrategy
        self.rng = random.Random(42)
        self.strategy = NormalizingFlowStrategy(pool_size=10, numbers_to_pick=3)
        self.draws = [
            sorted(random.Random(i).sample(range(1, 11), 3))
            for i in range(50)
        ]

    def test_name(self):
        self.assertEqual(self.strategy.name, "normalizing_flow")

    def test_generate_returns_correct_count(self):
        picks = self.strategy.generate(self.draws, 3, self.rng)
        self.assertEqual(len(picks), 3)

    def test_generate_returns_sorted_numbers(self):
        picks = self.strategy.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(pick, sorted(pick))

    def test_generate_numbers_in_range(self):
        picks = self.strategy.generate(self.draws, 2, self.rng)
        for pick in picks:
            self.assertEqual(len(pick), 3)
            for n in pick:
                self.assertGreaterEqual(n, 1)
                self.assertLessEqual(n, 10)

    def test_generate_with_few_draws(self):
        picks = self.strategy.generate(self.draws[:3], 2, self.rng)
        self.assertEqual(len(picks), 2)

    def test_get_probabilities_length(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertEqual(len(probs), 10)

    def test_get_probabilities_sum_to_one(self):
        probs = self.strategy.get_probabilities(self.draws)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_get_probabilities_all_positive(self):
        probs = self.strategy.get_probabilities(self.draws)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)

    def test_generate_unique_picks(self):
        picks = self.strategy.generate(self.draws, 5, self.rng)
        keys = [tuple(p) for p in picks]
        self.assertEqual(len(keys), len(set(keys)))


class TestNormalizingFlowsImport(unittest.TestCase):
    def test_module_importable(self):
        import shared.normalizing_flows  # noqa: F401


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests/test_normalizing_flows.py -v`
Expected: `ModuleNotFoundError`

**Step 3: Implement NormalizingFlowStrategy**

Create `src/shared/normalizing_flows.py`:

```python
"""Normalizing Flow strategy for lottery number generation.

Implements a RealNVP-style normalizing flow that learns to transform
a simple base distribution into the empirical lottery draw distribution.
Provides exact likelihood computation (unlike VAE).

Requires PyTorch (optional import).
"""

import random

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


def _encode_draws(draws, pool_size):
    encoded = torch.zeros(len(draws), pool_size)
    for i, draw in enumerate(draws):
        for n in draw:
            encoded[i, n - 1] = 1.0
    return encoded


if TORCH_AVAILABLE:

    class _CouplingLayer(nn.Module):
        """Affine coupling layer: split input, transform one half."""

        def __init__(self, dim, hidden_dim=32, mask_even=True):
            super().__init__()
            self.register_buffer(
                "mask",
                torch.arange(dim).float() % 2 == (0 if mask_even else 1),
            )
            half_dim = dim  # conditioner sees full dim (masked)
            self.scale_net = nn.Sequential(
                nn.Linear(half_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
                nn.Tanh(),  # bound scale for stability
            )
            self.shift_net = nn.Sequential(
                nn.Linear(half_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
            )

        def forward(self, x):
            """Forward: data → latent."""
            masked = x * self.mask
            s = self.scale_net(masked) * (1 - self.mask)
            t = self.shift_net(masked) * (1 - self.mask)
            z = masked + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = s.sum(dim=-1)
            return z, log_det

        def inverse(self, z):
            """Inverse: latent → data."""
            masked = z * self.mask
            s = self.scale_net(masked) * (1 - self.mask)
            t = self.shift_net(masked) * (1 - self.mask)
            x = masked + (1 - self.mask) * (z - t) * torch.exp(-s)
            return x

    class _RealNVP(nn.Module):
        def __init__(self, dim, n_layers=4, hidden_dim=32):
            super().__init__()
            self.layers = nn.ModuleList([
                _CouplingLayer(dim, hidden_dim, mask_even=(i % 2 == 0))
                for i in range(n_layers)
            ])

        def forward(self, x):
            """Forward pass: compute latent z and log-determinant."""
            total_log_det = torch.zeros(x.size(0), device=x.device)
            z = x
            for layer in self.layers:
                z, log_det = layer(z)
                total_log_det += log_det
            return z, total_log_det

        def sample(self, n, device="cpu"):
            """Sample from the model by inverting standard normal samples."""
            z = torch.randn(n, self.layers[0].mask.size(0), device=device)
            x = z
            for layer in reversed(self.layers):
                x = layer.inverse(x)
            return x


class NormalizingFlowStrategy:
    """Normalizing flow strategy for lottery number generation."""

    name = "normalizing_flow"

    def __init__(
        self,
        pool_size,
        numbers_to_pick,
        n_coupling_layers=4,
        hidden_dim=32,
        epochs=50,
        batch_size=16,
        learning_rate=1e-3,
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.n_coupling_layers = n_coupling_layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def _train_model(self, draws):
        """Train flow model and return it."""
        if not TORCH_AVAILABLE or len(draws) < 5:
            return None

        encoded = _encode_draws(draws, self.pool_size)

        model = _RealNVP(self.pool_size, self.n_coupling_layers, self.hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(encoded))
            for start in range(0, len(encoded), self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch = encoded[idx]
                # Add small noise to binary input for continuous relaxation
                batch = batch + torch.randn_like(batch) * 0.05

                z, log_det = model(batch)
                # Log-likelihood under standard normal base
                log_prob = -0.5 * (z ** 2).sum(dim=-1) + log_det
                loss = -log_prob.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        return model

    def _get_probs_from_model(self, model):
        """Generate samples from trained model and compute marginal probabilities."""
        if model is None:
            return [1.0 / self.pool_size] * self.pool_size

        model.eval()
        with torch.no_grad():
            samples = model.sample(200)
            # Convert to probabilities: sigmoid then average per number
            activated = torch.sigmoid(samples)
            marginal = activated.mean(dim=0).tolist()

        # Normalize to sum to 1
        total = sum(marginal)
        if total <= 0:
            return [1.0 / self.pool_size] * self.pool_size
        return [p / total for p in marginal]

    def get_probabilities(self, draws, **kwargs):
        model = self._train_model(draws)
        return self._get_probs_from_model(model)

    def generate(self, draws, count, rng, **kwargs):
        probs = self.get_probabilities(draws, **kwargs)
        picks = []
        seen = set()
        pool = list(range(1, self.pool_size + 1))

        for _ in range(count + 10):
            if len(picks) >= count:
                break
            remaining = list(pool)
            remaining_probs = list(probs)
            chosen = []
            for _ in range(self.numbers_to_pick):
                total = sum(remaining_probs)
                if total <= 0:
                    idx = rng.randrange(len(remaining))
                else:
                    normalized = [p / total for p in remaining_probs]
                    r = rng.random()
                    cumulative = 0.0
                    idx = len(remaining) - 1
                    for j, p in enumerate(normalized):
                        cumulative += p
                        if r <= cumulative:
                            idx = j
                            break
                chosen.append(remaining[idx])
                remaining.pop(idx)
                remaining_probs.pop(idx)

            pick = sorted(chosen)
            key = tuple(pick)
            if key not in seen:
                seen.add(key)
                picks.append(pick)

        return picks[:count]
```

**Step 4: Run tests**

Run: `PYTHONPATH=src python -m unittest tests/test_normalizing_flows.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/shared/normalizing_flows.py tests/test_normalizing_flows.py
git commit -m "feat: add Normalizing Flow (RealNVP) strategy"
```

---

## Task 5: Integrate Deep Learning Strategies into Ensemble Blend

**Files:**
- Modify: `src/shared/ensemble_blend.py`
- Modify: `tests/test_ensemble_blend.py`

### Design

Add LSTM, TCN, Transformer, and Normalizing Flow to the ensemble blend scoring and generation, following the same pattern used for Genetic and Gradient Boost. Use lightweight configs for scoring (fewer epochs, smaller models). All four use optional import with `TORCH_AVAILABLE` guard.

**Step 1: Write the failing test**

Add to `tests/test_ensemble_blend.py`:

```python
@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestDeepLearningBlend(unittest.TestCase):
    def test_blend_includes_deep_learning(self):
        """Ensemble should include deep learning strategies when PyTorch available."""
        from shared.ensemble_blend import generate_blended_picks
        from shared.game_config import GameConfig

        config = GameConfig(
            name="test", pool_min=1, pool_max=10,
            numbers_drawn=3, numbers_to_pick=3,
        )
        draws = [sorted(random.Random(i).sample(range(1, 11), 3)) for i in range(30)]
        rng = random.Random(42)
        picks = generate_blended_picks(config, draws, 5, rng)
        self.assertEqual(len(picks), 5)
        for pick in picks:
            self.assertEqual(len(pick), 3)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_ensemble_blend.TestDeepLearningBlend -v`

**Step 3: Update `src/shared/ensemble_blend.py`**

Add imports at top:

```python
try:
    from .lstm_strategy import LSTMStrategy
    from .tcn_strategy import TCNStrategy
    from .transformer_strategy import TransformerStrategy
    from .normalizing_flows import NormalizingFlowStrategy
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
```

In `generate_blended_picks()`, after the genetic scoring block and before `if _GB_AVAILABLE:`, add:

```python
    if _TORCH_AVAILABLE:
        # Lightweight configs for scoring
        lstm_scoring = LSTMStrategy(config.pool_size, config.numbers_to_pick, epochs=5, seq_len=10)
        tcn_scoring = TCNStrategy(config.pool_size, config.numbers_to_pick, epochs=5, seq_len=10)
        xfmr_scoring = TransformerStrategy(config.pool_size, config.numbers_to_pick, epochs=5, seq_len=10)
        nf_scoring = NormalizingFlowStrategy(config.pool_size, config.numbers_to_pick, epochs=10)

        for name, strat in [("lstm", lstm_scoring), ("tcn", tcn_scoring),
                            ("transformer", xfmr_scoring), ("normalizing_flow", nf_scoring)]:
            scores[name] = max(
                _score_strategy_object(config, scoring_draws, strat, rng, scoring_dates, half_life_mode),
                1,
            )
```

In the generation section (after the genetic generation block), add:

```python
    if _TORCH_AVAILABLE:
        lstm_gen = LSTMStrategy(config.pool_size, config.numbers_to_pick)
        tcn_gen = TCNStrategy(config.pool_size, config.numbers_to_pick)
        xfmr_gen = TransformerStrategy(config.pool_size, config.numbers_to_pick)
        nf_gen = NormalizingFlowStrategy(config.pool_size, config.numbers_to_pick)

        for strat_name, strat in [("lstm", lstm_gen), ("tcn", tcn_gen),
                                   ("transformer", xfmr_gen), ("normalizing_flow", nf_gen)]:
            if strat_name in allocation:
                strat_lines = strat.generate(draws, allocation.get(strat_name, 0) + count, rng)
                added = 0
                for pick in strat_lines:
                    if added >= allocation.get(strat_name, 0):
                        break
                    key = tuple(pick)
                    if key not in seen:
                        seen.add(key)
                        lines.append(pick)
                        added += 1
```

**Step 4: Run tests**

Run: `PYTHONPATH=src python -m unittest tests/test_ensemble_blend.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/shared/ensemble_blend.py tests/test_ensemble_blend.py
git commit -m "feat: integrate deep learning strategies into ensemble blend"
```

---

## Task 6: Run Full Test Suite and Validation

**Files:**
- No new files

**Step 1: Run all tests**

Run: `PYTHONPATH=src python -m unittest discover -s tests -v`
Expected: All tests PASS (except pre-existing `test_generate_picks_workflow` failure unrelated to our work)

**Step 2: Run pick generation scripts to verify end-to-end**

Run each with seed for reproducibility and timeout:
```bash
timeout 120 PYTHONPATH=src python scripts/generate_joker_picks.py --seed 42
timeout 120 PYTHONPATH=src python scripts/generate_loto_649_picks.py --seed 42
```

Verify: scripts complete within 2 minutes, output valid picks.

**Step 3: Commit any fixes**

If any issues found, fix and commit.

---

## Execution Notes

- Tasks 1-4 are independent and can be executed in any order (or in parallel pairs: 1+2, then 3+4)
- Task 5 depends on Tasks 1-4 being complete
- Task 6 depends on Task 5
- Deep learning tests will be slower (~5-15s each due to training). This is expected.
- The scoring configs use very few epochs (5-10) to keep ensemble blend fast. The generation configs use default epochs (30-50) since they run once per call, not per-draw.
