"""Decoder-only GPT-style Transformer strategy for lottery prediction.

Trains a causal Transformer on sliding windows of multi-hot encoded draws,
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


def _encode_draws(draws: list[list[int]], pool_size: int) -> "torch.Tensor":
    """Convert draws (1-indexed numbers) to a multi-hot tensor.

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


def _build_sliding_windows(
    encoded: "torch.Tensor", seq_len: int
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Build training pairs from a sliding window over encoded draws.

    Args:
        encoded: Tensor of shape (total_draws, pool_size).
        seq_len: Length of each input window.

    Returns:
        Tuple of (inputs, targets) where inputs has shape
        (num_samples, seq_len, pool_size) and targets has shape
        (num_samples, pool_size).
    """
    inputs = []
    targets = []
    for i in range(len(encoded) - seq_len):
        inputs.append(encoded[i:i + seq_len])
        targets.append(encoded[i + seq_len])
    return torch.stack(inputs), torch.stack(targets)


if TORCH_AVAILABLE:

    class _DecoderBlock(nn.Module):
        """Single decoder block with pre-norm causal self-attention and FFN."""

        def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            self.norm_attention = nn.LayerNorm(d_model)
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                batch_first=True,
            )
            self.norm_ffn = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )

        def forward(self, x: "torch.Tensor", causal_mask: "torch.Tensor") -> "torch.Tensor":
            """Apply causal self-attention and feed-forward network.

            Args:
                x: Input tensor of shape (batch, seq_len, d_model).
                causal_mask: Boolean upper-triangular mask of shape (seq_len, seq_len).

            Returns:
                Output tensor of the same shape as x.
            """
            normed = self.norm_attention(x)
            attended, _ = self.attention(
                normed, normed, normed, attn_mask=causal_mask
            )
            x = x + attended

            normed = self.norm_ffn(x)
            x = x + self.ffn(normed)
            return x

    class _TransformerNet(nn.Module):
        """Decoder-only GPT-style Transformer for multi-hot draw prediction."""

        def __init__(
            self,
            pool_size: int,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            d_ff: int = 128,
            max_len: int = 200,
        ):
            super().__init__()
            self.pool_size = pool_size
            self.input_projection = nn.Linear(pool_size, d_model)
            self.positional_encoding = nn.Parameter(
                torch.randn(1, max_len, d_model) * 0.02
            )
            self.blocks = nn.ModuleList([
                _DecoderBlock(d_model, n_heads, d_ff)
                for _ in range(n_layers)
            ])
            self.final_norm = nn.LayerNorm(d_model)
            self.output_head = nn.Linear(d_model, pool_size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Predict next-draw logits from a sequence of multi-hot draws.

            Args:
                x: Tensor of shape (batch, seq_len, pool_size).

            Returns:
                Logits tensor of shape (batch, pool_size).
            """
            seq_len = x.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()

            x = self.input_projection(x)
            x = x + self.positional_encoding[:, :seq_len, :]

            for block in self.blocks:
                x = block(x, causal_mask)

            x = self.final_norm(x[:, -1, :])
            return self.output_head(x)


class TransformerStrategy:
    """Decoder-only Transformer lottery number generation strategy.

    Uses a causal GPT-style Transformer trained on sliding windows of
    historical draws. When PyTorch is unavailable or insufficient data
    is provided, falls back to a uniform probability distribution.
    """

    name = "transformer"

    def __init__(
        self,
        pool_size: int,
        numbers_to_pick: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        seq_len: int = 20,
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
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

    def _uniform_probabilities(self) -> list[float]:
        return [1.0 / self.pool_size] * self.pool_size

    def _should_fallback(self, draws: list[list[int]]) -> bool:
        return not TORCH_AVAILABLE or len(draws) < MINIMUM_DRAWS

    def _train_and_predict(self, draws: list[list[int]]) -> list[float]:
        """Train the Transformer on draws and return a probability vector."""
        effective_seq_len = min(self.seq_len, len(draws) - 1)
        encoded = _encode_draws(draws, self.pool_size)

        inputs, targets = _build_sliding_windows(encoded, effective_seq_len)
        if len(inputs) == 0:
            return self._uniform_probabilities()

        model = _TransformerNet(
            pool_size=self.pool_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        dataset_size = len(inputs)
        model.train()

        for _ in range(self.epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                x_batch = inputs[batch_idx]
                y_batch = targets[batch_idx]

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            recent_window = encoded[-effective_seq_len:].unsqueeze(0)
            logits = model(recent_window).squeeze(0)
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
        """Generate unique lottery picks sampled from Transformer predictions.

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
