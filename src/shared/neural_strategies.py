"""Unified neural network strategies for lottery prediction.

This module provides neural network-based number generation that adapts
to any game configuration. It uses the MLP implementation from neural_base.py.
"""

import random
from typing import Literal

from .game_config import GameConfig
from .math_utils import softmax, one_hot
from .recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE


class SoftmaxModel:
    """Simple softmax regression model for lottery prediction.

    This is a lightweight model that learns associations between
    previous draws and next draws.
    """

    def __init__(self, input_size: int, output_size: int, rng: random.Random | None = None):
        self.input_size = input_size
        self.output_size = output_size
        self.rng = rng or random.Random()
        self.weights = [
            [self.rng.uniform(-0.01, 0.01) for _ in range(input_size)]
            for _ in range(output_size)
        ]

    def predict_probs(self, inputs: list[float]) -> list[float]:
        """Predict probability distribution over numbers."""
        logits = []
        for row in self.weights:
            logits.append(sum(w * x for w, x in zip(row, inputs)))
        return softmax(logits)

    def train(
        self,
        inputs_list: list[list[float]],
        targets_list: list[list[float]],
        epochs: int = 10,
        lr: float = 0.1,
        sample_weights: list[float] | None = None,
    ) -> None:
        """Train the model using gradient descent."""
        for _ in range(epochs):
            for idx, (inputs, target) in enumerate(zip(inputs_list, targets_list)):
                weight = sample_weights[idx] if sample_weights is not None else 1.0
                if weight == 0.0:
                    continue
                probs = self.predict_probs(inputs)
                for i in range(self.output_size):
                    error = (probs[i] - target[i]) * weight
                    for j in range(self.input_size):
                        self.weights[i][j] -= lr * error * inputs[j]


def _sample_without_replacement(
    weights: list[float],
    count: int,
    rng: random.Random,
) -> list[int]:
    """Sample indices without replacement, weighted by probabilities."""
    pool = list(range(len(weights)))
    chosen = []
    local_weights = list(weights)
    for _ in range(count):
        pick = rng.choices(pool, weights=local_weights, k=1)[0]
        idx = pool.index(pick)
        chosen.append(pick)
        pool.pop(idx)
        local_weights.pop(idx)
    return chosen


def generate_neural_picks(
    config: GameConfig,
    draws: list[tuple],
    count: int,
    rng: random.Random | None = None,
    epochs: int = 10,
    lr: float = 0.1,
    architecture: Literal["softmax", "mlp"] = "softmax",
    half_life: float = DEFAULT_HALF_LIFE,
    draw_dates: list[str] | None = None,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
) -> list[list[int]]:
    """Generate lottery picks using neural network prediction.

    The model learns patterns from historical draws and samples
    new picks based on learned probability distributions.

    Args:
        config: Game configuration
        draws: Historical draws as (main_numbers, bonus) tuples
        count: Number of picks to generate
        rng: Random number generator
        epochs: Training epochs
        lr: Learning rate
        architecture: Model type ("softmax" or "mlp")

    Returns:
        List of picks, each pick is a sorted list of numbers
    """
    rng = rng or random.SystemRandom()

    if len(draws) < 2:
        # Not enough data to train, fall back to random
        from .game_strategies import generate_random_picks
        return generate_random_picks(config, count, rng)

    pool_size = config.pool_size

    if architecture == "softmax":
        model = SoftmaxModel(
            input_size=pool_size,
            output_size=pool_size,
            rng=random.Random(0),  # Deterministic initialization
        )
    else:
        # MLP architecture - use the neural_base implementation
        from .neural_base import MLPModel
        model = MLPModel(
            layer_sizes=[pool_size, 32, pool_size],
            rng=random.Random(0),
        )

    # Prepare training data
    inputs = []
    targets = []

    for prev, nxt in zip(draws[:-1], draws[1:]):
        prev_main, _ = prev
        nxt_main, _ = nxt

        # Convert to one-hot encoding (0-indexed)
        x = one_hot([n - config.pool_min for n in prev_main], pool_size)
        # Target: normalize so drawn numbers share probability
        y = one_hot(
            [n - config.pool_min for n in nxt_main],
            pool_size,
            value=1.0 / config.numbers_drawn,
        )

        inputs.append(x)
        targets.append(y)

    # Train the model
    sample_weights = None
    if draw_dates:
        from .recency import draw_weights

        weights_by_draw = draw_weights(
            len(draw_dates),
            half_life,
            draw_dates=draw_dates,
            mode=half_life_mode,
        )
        sample_weights = weights_by_draw[1:]

    if architecture == "softmax":
        model.train(inputs, targets, epochs=epochs, lr=lr, sample_weights=sample_weights)
    else:
        model.train(inputs, targets, epochs=epochs, learning_rate=lr, sample_weights=sample_weights)

    # Generate picks from the trained model
    last_main, _ = draws[-1]
    last_x = one_hot([n - config.pool_min for n in last_main], pool_size)
    probs = model.predict_probs(last_x)

    lines = []
    seen = set()

    while len(lines) < count:
        # Sample indices based on predicted probabilities
        indices = _sample_without_replacement(probs, config.numbers_to_pick, rng)
        # Convert back to actual numbers
        pick = sorted([i + config.pool_min for i in indices])
        key = tuple(pick)
        if key in seen:
            continue
        seen.add(key)
        lines.append(pick)

    return lines


def score_neural_strategy(
    config: GameConfig,
    draws: list[tuple],
    rng: random.Random,
    draw_dates: list[str] | None = None,
    half_life: float = DEFAULT_HALF_LIFE,
    half_life_mode: str = DEFAULT_HALF_LIFE_MODE,
) -> int:
    """Score the neural strategy on historical data.

    Returns the number of winning picks when testing on each draw.
    """
    from .game_strategies import is_prize_winner

    wins = 0
    for idx in range(2, len(draws)):
        # Train on draws[:idx], predict for draws[idx]
        train_draws = draws[:idx]
        test_draw = draws[idx]

        picks = generate_neural_picks(
            config,
            train_draws,
            1,
            rng=rng,
            half_life=half_life,
            draw_dates=draw_dates[:idx] if draw_dates else None,
            half_life_mode=half_life_mode,
        )
        if picks:
            pick = picks[0]
            test_main, test_bonus = test_draw
            if is_prize_winner(config, pick, test_main):
                wins += 1

    return wins
