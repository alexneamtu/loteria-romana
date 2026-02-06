"""Reinforcement Learning agent for lottery number selection.

Uses REINFORCE (policy gradient) with a small feedforward network.
The agent learns a policy that maps recent draw history to per-number
selection probabilities, trained by sampling actions and weighting
updates by match-count rewards.

Falls back to uniform probabilities when PyTorch is unavailable.
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


def _compute_reward(matches: int, numbers_to_pick: int) -> float:
    """Compute reward based on how many numbers matched."""
    if matches == 0:
        return 0.0
    return (matches / numbers_to_pick) ** 2


def _encode_recent_draws(
    draws: list[list[int]], pool_size: int, window: int,
) -> list[float]:
    """Encode the last `window` draws into a flat feature vector."""
    features: list[float] = []
    start = max(0, len(draws) - window)
    for i in range(window):
        idx = start + i
        vec = [0.0] * pool_size
        if idx < len(draws):
            for number in draws[idx]:
                vec[number - 1] = 1.0
        features.extend(vec)
    return features


if TORCH_AVAILABLE:

    class _PolicyNet(nn.Module):
        """Small feedforward policy network."""

        def __init__(self, input_size: int, pool_size: int, hidden_size: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, pool_size),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


class RLAgent:
    """REINFORCE policy gradient agent for lottery number selection."""

    name = "rl"

    def __init__(
        self,
        pool_size: int,
        numbers_to_pick: int,
        window: int = 10,
        hidden_size: int = 64,
        episodes: int = 50,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
    ):
        self.pool_size = pool_size
        self.numbers_to_pick = numbers_to_pick
        self.window = window
        self.hidden_size = hidden_size
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.discount = discount

    def _uniform_probabilities(self) -> list[float]:
        return [1.0 / self.pool_size] * self.pool_size

    def _should_fallback(self, draws: list[list[int]]) -> bool:
        return not TORCH_AVAILABLE or len(draws) < MINIMUM_DRAWS

    def _train_policy(self, draws: list[list[int]]) -> list[float]:
        """Train REINFORCE agent and return final probability vector."""
        input_size = self.window * self.pool_size
        policy = _PolicyNet(input_size, self.pool_size, self.hidden_size)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate)

        min_train = max(self.window, MINIMUM_DRAWS)

        for _ in range(self.episodes):
            log_probs_list = []
            rewards_list = []

            for t in range(min_train, len(draws)):
                state = _encode_recent_draws(draws[:t], self.pool_size, self.window)
                state_tensor = torch.tensor([state], dtype=torch.float32)

                logits = policy(state_tensor).squeeze(0)
                probs = torch.softmax(logits, dim=0)

                selected_indices = []
                episode_log_prob = torch.tensor(0.0)
                mask = torch.ones(self.pool_size, dtype=torch.float32)

                for _ in range(self.numbers_to_pick):
                    masked_probs = probs * mask
                    masked_probs = masked_probs / masked_probs.sum()
                    dist = torch.distributions.Categorical(masked_probs)
                    idx = dist.sample()
                    episode_log_prob = episode_log_prob + dist.log_prob(idx)
                    selected_indices.append(idx.item())
                    mask = mask.clone()
                    mask[idx] = 0.0

                pick = sorted(i + 1 for i in selected_indices)
                actual_draw = draws[t]
                matches = len(set(pick) & set(actual_draw))
                reward = _compute_reward(matches, self.numbers_to_pick)

                log_probs_list.append(episode_log_prob)
                rewards_list.append(reward)

            if not log_probs_list:
                continue

            returns = []
            g = 0.0
            for r in reversed(rewards_list):
                g = r + self.discount * g
                returns.insert(0, g)

            returns_tensor = torch.tensor(returns, dtype=torch.float32)
            if returns_tensor.std() > 0:
                returns_tensor = (
                    (returns_tensor - returns_tensor.mean())
                    / (returns_tensor.std() + 1e-8)
                )

            loss = torch.tensor(0.0)
            for lp, ret in zip(log_probs_list, returns_tensor):
                loss = loss - lp * ret

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        policy.eval()
        with torch.no_grad():
            state = _encode_recent_draws(draws, self.pool_size, self.window)
            state_tensor = torch.tensor([state], dtype=torch.float32)
            logits = policy(state_tensor).squeeze(0)
            probs = torch.softmax(logits, dim=0).tolist()

        return probs

    def get_probabilities(
        self, draws: list[list[int]], **kwargs,
    ) -> list[float]:
        """Return probability distribution over numbers 1..pool_size."""
        if self._should_fallback(draws):
            return self._uniform_probabilities()
        return self._train_policy(draws)

    def generate(
        self,
        draws: list[list[int]],
        count: int,
        rng: random.Random,
        **kwargs,
    ) -> list[list[int]]:
        """Generate unique lottery picks using the trained RL policy."""
        probabilities = self.get_probabilities(draws, **kwargs)
        numbers = list(range(1, self.pool_size + 1))

        lines: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        max_attempts = count * 200

        attempts = 0
        while len(lines) < count and attempts < max_attempts:
            pick = _sample_without_replacement(
                numbers, probabilities, self.numbers_to_pick, rng,
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
