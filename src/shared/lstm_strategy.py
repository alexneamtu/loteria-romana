"""LSTM sequence strategy for lottery prediction using PyTorch.

Trains a multi-layer LSTM on sliding windows of multi-hot encoded draws,
then samples from the predicted probability distribution to generate picks.
Falls back to uniform probabilities when PyTorch is unavailable or when
insufficient historical data is provided.
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

MINIMUM_DRAWS = 5


def _encode_draw_as_multi_hot(draw: list[int], pool_size: int) -> list[float]:
    """Convert a draw (1-indexed numbers) to a multi-hot vector."""
    vector = [0.0] * pool_size
    for number in draw:
        vector[number - 1] = 1.0
    return vector


def _build_sliding_windows(
    draws: list[list[int]], pool_size: int, seq_len: int
) -> tuple[list[list[list[float]]], list[list[float]]]:
    """Build training pairs from a sliding window over draws.

    Each sample consists of a sequence of seq_len multi-hot vectors
    (the input window) and the multi-hot vector of the next draw
    (the target).

    Returns:
        Tuple of (input_sequences, targets) where each input sequence
        is a list of multi-hot vectors and each target is a multi-hot vector.
    """
    encoded = [_encode_draw_as_multi_hot(draw, pool_size) for draw in draws]
    sequences = []
    targets = []
    for i in range(seq_len, len(encoded)):
        sequences.append(encoded[i - seq_len:i])
        targets.append(encoded[i])
    return sequences, targets


if TORCH_AVAILABLE:

    class _LSTMNet(nn.Module):
        """PyTorch LSTM that predicts multi-hot draw vectors."""

        def __init__(
            self, pool_size: int, hidden_size: int = 64, num_layers: int = 2
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=pool_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, pool_size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Return raw logits for each number in the pool.

            Args:
                x: Tensor of shape (batch, seq_len, pool_size).

            Returns:
                Logits tensor of shape (batch, pool_size).
            """
            output, _ = self.lstm(x)
            last_hidden = output[:, -1, :]
            return self.head(last_hidden)


class LSTMStrategy:
    """LSTM-based lottery number generation strategy.

    Uses a PyTorch LSTM trained on sliding windows of historical draws.
    When PyTorch is unavailable or insufficient data is provided, falls
    back to a uniform probability distribution.
    """

    name = "lstm"

    def __init__(
        self,
        pool_size: int,
        numbers_to_pick: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        seq_len: int = 20,
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def _uniform_probabilities(self) -> list[float]:
        return [1.0 / self.pool_size] * self.pool_size

    def _should_fallback(self, draws: list[list[int]]) -> bool:
        return not TORCH_AVAILABLE or len(draws) < MINIMUM_DRAWS

    def _train_and_predict(self, draws: list[list[int]]) -> list[float]:
        """Train an LSTM on the draws and return a probability vector."""
        effective_seq_len = min(self.seq_len, len(draws) - 1)
        sequences, targets = _build_sliding_windows(
            draws, self.pool_size, effective_seq_len
        )
        if not sequences:
            return self._uniform_probabilities()

        x_tensor = torch.tensor(sequences, dtype=torch.float32)
        y_tensor = torch.tensor(targets, dtype=torch.float32)

        model = _LSTMNet(
            pool_size=self.pool_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        dataset_size = len(sequences)
        model.train()

        for _ in range(self.epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                x_batch = x_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            recent_window = _encode_draw_as_multi_hot(
                draws[-1], self.pool_size
            )
            encoded_tail = [
                _encode_draw_as_multi_hot(d, self.pool_size)
                for d in draws[-effective_seq_len:]
            ]
            input_tensor = torch.tensor(
                [encoded_tail], dtype=torch.float32
            )
            logits = model(input_tensor).squeeze(0)
            probabilities = torch.softmax(logits, dim=0).tolist()

        return probabilities

    def get_probabilities(
        self, draws: list[list[int]], **kwargs
    ) -> list[float]:
        """Return probability distribution over numbers 1..pool_size.

        Falls back to uniform when PyTorch is unavailable or when
        fewer than MINIMUM_DRAWS are provided.
        """
        if self._should_fallback(draws):
            return self._uniform_probabilities()
        return self._train_and_predict(draws)

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        **kwargs,
    ) -> list[list[int]]:
        """Generate unique lottery picks sampled from LSTM predictions.

        Args:
            draws: Historical draws, each a sorted list of 1-indexed numbers.
            count: How many unique lines to produce.
            rng: Seeded random.Random instance for reproducibility.

        Returns:
            List of sorted pick lists.
        """
        probabilities = self.get_probabilities(draws, **kwargs)
        numbers = list(range(1, self.pool_size + 1))

        lines: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        max_attempts = count * 200

        attempts = 0
        while len(lines) < count and attempts < max_attempts:
            pick = _sample_without_replacement(
                numbers, probabilities, self.numbers_to_pick, rng
            )
            key = tuple(pick)
            if key not in seen:
                seen.add(key)
                lines.append(pick)
            attempts += 1

        return lines


def _sample_without_replacement(
    numbers: list[int],
    weights: list[float],
    count: int,
    rng: random.Random,
) -> list[int]:
    """Weighted sampling without replacement, returning sorted picks."""
    pool = list(numbers)
    pool_weights = list(weights)
    chosen: list[int] = []

    for _ in range(count):
        if not pool:
            break
        pick = rng.choices(pool, weights=pool_weights, k=1)[0]
        idx = pool.index(pick)
        chosen.append(pick)
        pool.pop(idx)
        pool_weights.pop(idx)

    return sorted(chosen)
