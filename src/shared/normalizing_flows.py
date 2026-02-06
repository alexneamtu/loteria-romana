"""Normalizing Flow (RealNVP) strategy for lottery number prediction.

Learns to transform a standard normal base distribution into the empirical
lottery draw distribution using affine coupling layers. Provides exact
likelihood computation and generates picks by sampling from the learned
distribution.

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
SAMPLE_COUNT = 200


def _encode_draws(draws, pool_size):
    """Convert draws to multi-hot encoded tensor.

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

    class _CouplingLayer(nn.Module):
        """Affine coupling layer for RealNVP.

        Splits input dimensions using a binary mask. The masked portion
        stays fixed while the unmasked portion is transformed via learned
        scale and shift networks conditioned on the fixed portion.
        """

        def __init__(self, dim, hidden_dim, mask):
            super().__init__()
            self.register_buffer("mask", mask)

            self.scale_net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
                nn.Tanh(),
            )

            self.shift_net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
            )

        def forward(self, x):
            """Forward pass: data space to latent space.

            Returns:
                Tuple of (z, log_det_jacobian).
            """
            masked_x = x * self.mask
            scale = self.scale_net(masked_x)
            shift = self.shift_net(masked_x)

            z = masked_x + (1 - self.mask) * (x * torch.exp(scale) + shift)
            log_det = ((1 - self.mask) * scale).sum(dim=-1)
            return z, log_det

        def inverse(self, z):
            """Inverse pass: latent space to data space."""
            masked_z = z * self.mask
            scale = self.scale_net(masked_z)
            shift = self.shift_net(masked_z)

            x = masked_z + (1 - self.mask) * (z - shift) * torch.exp(-scale)
            return x

    class _RealNVP(nn.Module):
        """RealNVP normalizing flow model.

        Stacks multiple coupling layers with alternating even/odd masks
        to enable flexible bijective transformations.
        """

        def __init__(self, dim, hidden_dim, n_coupling_layers):
            super().__init__()
            layers = []
            for i in range(n_coupling_layers):
                mask = torch.zeros(dim)
                if i % 2 == 0:
                    mask[::2] = 1.0
                else:
                    mask[1::2] = 1.0
                layers.append(_CouplingLayer(dim, hidden_dim, mask))
            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            """Transform data to latent space with log determinant.

            Returns:
                Tuple of (z, total_log_det).
            """
            total_log_det = torch.zeros(x.shape[0])
            z = x
            for layer in self.layers:
                z, log_det = layer(z)
                total_log_det = total_log_det + log_det
            return z, total_log_det

        def sample(self, n):
            """Generate samples by passing standard normal through inverse."""
            z = torch.randn(n, self.layers[0].mask.shape[0])
            x = z
            for layer in reversed(self.layers):
                x = layer.inverse(x)
            return x


class NormalizingFlowStrategy:
    """RealNVP normalizing flow strategy for lottery number generation.

    Trains a normalizing flow model on historical draws to learn their
    distribution, then samples from the learned distribution to generate
    picks. Falls back to uniform probabilities when PyTorch is unavailable
    or fewer than MINIMUM_DRAWS are provided.
    """

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

    def _uniform_probabilities(self):
        return [1.0 / self.pool_size] * self.pool_size

    def _should_fallback(self, draws):
        return not TORCH_AVAILABLE or len(draws) < MINIMUM_DRAWS

    def _train_model(self, draws):
        """Train a RealNVP model on multi-hot encoded draws.

        Returns:
            Trained _RealNVP model.
        """
        encoded = _encode_draws(draws, self.pool_size)
        noise = torch.randn_like(encoded) * 0.05
        training_data = encoded + noise

        model = _RealNVP(
            dim=self.pool_size,
            hidden_dim=self.hidden_dim,
            n_coupling_layers=self.n_coupling_layers,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        dataset_size = len(training_data)
        model.train()

        for _ in range(self.epochs):
            permutation = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = permutation[start:start + self.batch_size]
                batch = training_data[batch_idx]

                optimizer.zero_grad()
                z, log_det = model(batch)
                log_likelihood = -0.5 * z.pow(2).sum(dim=-1) + log_det
                loss = -log_likelihood.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        return model

    def _extract_probabilities(self, model):
        """Extract marginal number probabilities from trained model.

        Generates samples, applies sigmoid, and averages per-number
        activations to produce a normalized probability vector.

        Returns:
            List of probabilities, one per number in the pool.
        """
        model.eval()
        with torch.no_grad():
            samples = model.sample(SAMPLE_COUNT)
            activations = torch.sigmoid(samples)
            marginal = activations.mean(dim=0)

        probabilities = marginal.tolist()
        total = sum(probabilities)
        if total <= 0:
            return self._uniform_probabilities()
        return [p / total for p in probabilities]

    def get_probabilities(self, draws, **kwargs):
        """Return normalized probability distribution over pool numbers.

        Falls back to uniform when PyTorch is unavailable or when
        fewer than MINIMUM_DRAWS are provided.
        """
        if self._should_fallback(draws):
            return self._uniform_probabilities()

        model = self._train_model(draws)
        return self._extract_probabilities(model)

    def generate(self, draws, count, rng, **kwargs):
        """Generate unique lottery picks sampled from flow predictions.

        Args:
            draws: Historical draws, each a sorted list of 1-indexed numbers.
            count: How many unique lines to produce.
            rng: Seeded random.Random instance for reproducibility.

        Returns:
            List of sorted pick lists.
        """
        probabilities = self.get_probabilities(draws, **kwargs)
        numbers = list(range(1, self.pool_size + 1))

        lines = []
        seen = set()
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


def _sample_without_replacement(numbers, weights, count, rng):
    """Weighted sampling without replacement, returning sorted picks."""
    pool = list(numbers)
    pool_weights = list(weights)
    chosen = []

    for _ in range(count):
        if not pool:
            break
        total = sum(pool_weights)
        if total == 0:
            pick = rng.choice(pool)
        else:
            normalized = [w / total for w in pool_weights]
            pick = rng.choices(pool, weights=normalized, k=1)[0]
        idx = pool.index(pick)
        chosen.append(pick)
        pool.pop(idx)
        pool_weights.pop(idx)

    return sorted(chosen)
