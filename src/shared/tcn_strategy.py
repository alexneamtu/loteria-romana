"""
Temporal Convolutional Network strategy for lottery number generation.

Uses dilated causal convolutions to model temporal dependencies in
historical draw sequences. The TCN processes multi-hot encoded draw
vectors through a stack of causal convolution blocks with exponentially
increasing dilation, producing per-number probability estimates.

Falls back to uniform probabilities when PyTorch is unavailable or
when insufficient historical data is provided.
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


_MINIMUM_DRAWS = 5


def _encode_draws(draws, pool_size):
    """Encode a list of draws as multi-hot vectors.

    Args:
        draws: List of draws, each a list of 1-indexed numbers.
        pool_size: Total numbers in the pool.

    Returns:
        Tensor of shape (len(draws), pool_size) with 1.0 at drawn positions.
    """
    encoded = torch.zeros(len(draws), pool_size)
    for i, draw in enumerate(draws):
        for n in draw:
            encoded[i, n - 1] = 1.0
    return encoded


if TORCH_AVAILABLE:

    class _CausalConvBlock(nn.Module):
        """Single causal convolution block with residual connection.

        Applies left-padding to ensure the convolution only attends to
        past and present timesteps, then trims any future leakage from
        the right side.
        """

        def __init__(self, channels, kernel_size, dilation, dropout=0.1):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                channels, channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=self.padding,
            )
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            """Apply causal convolution with residual.

            Args:
                x: Tensor of shape (batch, channels, seq_len).

            Returns:
                Tensor of same shape with residual added.
            """
            out = self.conv(x)
            if self.padding > 0:
                out = out[:, :, :-self.padding]
            out = self.relu(out)
            out = self.dropout(out)
            return out + x

    class _TCNNet(nn.Module):
        """Temporal Convolutional Network for multi-label draw prediction.

        Architecture:
            1. Linear projection from pool_size to channels
            2. Stack of causal Conv1d blocks with dilation [1, 2, 4, 8, ...]
            3. Global average pooling over time
            4. Linear projection to pool_size logits
        """

        def __init__(self, pool_size, channels, kernel_size, num_blocks, dropout=0.1):
            super().__init__()
            self.input_projection = nn.Linear(pool_size, channels)
            blocks = []
            for i in range(num_blocks):
                dilation = 2 ** i
                blocks.append(
                    _CausalConvBlock(channels, kernel_size, dilation, dropout)
                )
            self.blocks = nn.Sequential(*blocks)
            self.output_projection = nn.Linear(channels, pool_size)

        def forward(self, x):
            """Forward pass producing per-number logits.

            Args:
                x: Multi-hot encoded input of shape (batch, seq_len, pool_size).

            Returns:
                Logits of shape (batch, pool_size).
            """
            h = self.input_projection(x)
            h = h.transpose(1, 2)
            h = self.blocks(h)
            h = h.mean(dim=2)
            return self.output_projection(h)


class TCNStrategy:
    """Generate lottery picks using a Temporal Convolutional Network.

    Trains a small TCN on the historical draw sequence and samples
    from the predicted probability distribution. When PyTorch is not
    installed or fewer than ``_MINIMUM_DRAWS`` draws are available,
    falls back to a uniform distribution.
    """

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

    def generate(self, draws, count, rng, **kwargs):
        """Generate lottery lines by sampling from TCN probabilities.

        Args:
            draws: Historical draws as list of lists of 1-indexed ints.
            count: Number of lines to produce.
            rng: ``random.Random`` instance for reproducibility.

        Returns:
            List of ``count`` sorted pick lists, each of length
            ``numbers_to_pick``, with no duplicate lines.
        """
        probabilities = self.get_probabilities(draws, **kwargs)
        numbers = list(range(1, self.pool_size + 1))
        lines = []
        seen = set()
        max_attempts = count * 20

        for _ in range(max_attempts):
            if len(lines) >= count:
                break
            pick = _sample_without_replacement(numbers, probabilities, self.numbers_to_pick, rng)
            key = tuple(pick)
            if key not in seen:
                seen.add(key)
                lines.append(pick)

        return lines

    def get_probabilities(self, draws, **kwargs):
        """Compute per-number probabilities from the TCN.

        Returns a list of ``pool_size`` floats summing to 1.0. Falls
        back to uniform when PyTorch is unavailable or data is sparse.

        Args:
            draws: Historical draws.

        Returns:
            Probability for each number from 1 to ``pool_size``.
        """
        if not TORCH_AVAILABLE or len(draws) < _MINIMUM_DRAWS:
            return _uniform_probabilities(self.pool_size)

        return self._train_and_predict(draws)

    def _train_and_predict(self, draws):
        """Train a fresh TCN on *draws* and return probabilities."""
        encoded = _encode_draws(draws, self.pool_size)
        sequences, targets = _build_sequences(encoded, self.seq_len)

        if len(sequences) == 0:
            return _uniform_probabilities(self.pool_size)

        model = _TCNNet(
            self.pool_size,
            self.channels,
            self.kernel_size,
            self.num_blocks,
        )

        _train_model(model, sequences, targets, self.epochs, self.batch_size, self.learning_rate)

        probabilities = _predict(model, encoded, self.seq_len, self.pool_size)
        return probabilities


def _uniform_probabilities(pool_size):
    """Return a uniform distribution over *pool_size* numbers."""
    p = 1.0 / pool_size
    return [p] * pool_size


def _sample_without_replacement(numbers, weights, count, rng):
    """Weighted sampling without replacement using *rng*.

    Args:
        numbers: Available numbers to choose from.
        weights: Weight for each number (same length as numbers).
        count: How many numbers to pick.
        rng: ``random.Random`` instance.

    Returns:
        Sorted list of selected numbers.
    """
    pool = list(numbers)
    pool_weights = list(weights)
    chosen = []

    for _ in range(count):
        if not pool:
            break
        pick = rng.choices(pool, weights=pool_weights, k=1)[0]
        idx = pool.index(pick)
        chosen.append(pick)
        pool.pop(idx)
        pool_weights.pop(idx)

    return sorted(chosen)


if TORCH_AVAILABLE:

    def _build_sequences(encoded, seq_len):
        """Slice encoded draws into overlapping (input, target) pairs.

        Args:
            encoded: Tensor of shape (num_draws, pool_size).
            seq_len: Length of each input window.

        Returns:
            Tuple of (inputs, targets) tensors.  inputs has shape
            (N, seq_len, pool_size), targets has shape (N, pool_size).
        """
        total = encoded.size(0)
        if total <= seq_len:
            return torch.zeros(0), torch.zeros(0)

        inputs = []
        targets = []
        for i in range(total - seq_len):
            inputs.append(encoded[i:i + seq_len])
            targets.append(encoded[i + seq_len])
        return torch.stack(inputs), torch.stack(targets)

    def _train_model(model, sequences, targets, epochs, batch_size, learning_rate):
        """Train *model* on the given sequences with BCE loss."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        dataset_size = sequences.size(0)

        for _ in range(epochs):
            permutation = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                indices = permutation[start:start + batch_size]
                batch_x = sequences[indices]
                batch_y = targets[indices]
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

    def _predict(model, encoded, seq_len, pool_size):
        """Run inference on the most recent window and return probabilities."""
        model.eval()
        with torch.no_grad():
            recent = encoded[-seq_len:].unsqueeze(0)
            if recent.size(1) < seq_len:
                padding = torch.zeros(1, seq_len - recent.size(1), pool_size)
                recent = torch.cat([padding, recent], dim=1)
            logits = model(recent)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)
        return probabilities.tolist()
