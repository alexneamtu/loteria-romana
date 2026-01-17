import math
import random

from shared.recency import DEFAULT_HALF_LIFE, DEFAULT_HALF_LIFE_MODE


def _softmax(logits):
    max_logit = max(logits)
    exp_vals = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_vals)
    return [v / total for v in exp_vals]


class SoftmaxModel:
    def __init__(self, input_size: int, output_size: int, rng=None):
        self.input_size = input_size
        self.output_size = output_size
        self.rng = rng or random.Random()
        self.weights = [
            [self.rng.uniform(-0.01, 0.01) for _ in range(input_size)]
            for _ in range(output_size)
        ]

    def predict_probs(self, inputs):
        logits = []
        for row in self.weights:
            logits.append(sum(w * x for w, x in zip(row, inputs)))
        return _softmax(logits)

    def loss(self, inputs_list, targets_list, sample_weights=None):
        total = 0.0
        for idx, (inputs, target) in enumerate(zip(inputs_list, targets_list)):
            weight = sample_weights[idx] if sample_weights is not None else 1.0
            probs = self.predict_probs(inputs)
            for p, t in zip(probs, target):
                if t:
                    total -= math.log(max(p, 1e-12)) * weight
        total_weight = sum(sample_weights) if sample_weights else len(inputs_list)
        return total / max(1, total_weight)

    def train(self, inputs_list, targets_list, epochs=10, lr=0.1, sample_weights=None):
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


def _one_hot(indices, size, value=1.0):
    vec = [0.0] * size
    for idx in indices:
        vec[idx] = value
    return vec


def _sample_without_replacement(weights, count, rng):
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


def generate_neural_lines(
    draws,
    count,
    rng=None,
    epochs=10,
    lr=0.1,
    draw_dates=None,
    half_life=DEFAULT_HALF_LIFE,
    half_life_mode=DEFAULT_HALF_LIFE_MODE,
):
    """Generate neural network-based Loto 5/40 picks.

    Uses historical draws to train a model, then samples 5 numbers from 1-40.
    """
    rng = rng or random.SystemRandom()
    if len(draws) < 2:
        return []

    input_size = 40
    main_model = SoftmaxModel(input_size=input_size, output_size=40, rng=random.Random(0))

    inputs = []
    main_targets = []

    for prev, nxt in zip(draws[:-1], draws[1:]):
        x = _one_hot([n - 1 for n in prev], 40)
        inputs.append(x)
        # Target is the 6 drawn numbers from next draw
        main_targets.append(_one_hot([n - 1 for n in nxt], 40, value=1.0 / 6.0))

    sample_weights = None
    if draw_dates:
        from shared.recency import draw_weights

        weights_by_draw = draw_weights(
            len(draw_dates),
            half_life,
            draw_dates=draw_dates,
            mode=half_life_mode,
        )
        sample_weights = weights_by_draw[1:]

    main_model.train(inputs, main_targets, epochs=epochs, lr=lr, sample_weights=sample_weights)

    last_x = _one_hot([n - 1 for n in draws[-1]], 40)
    main_probs = main_model.predict_probs(last_x)

    lines = []
    seen = set()
    while len(lines) < count:
        main_idxs = _sample_without_replacement(main_probs, 5, rng)  # Pick 5 numbers
        main = sorted([i + 1 for i in main_idxs])
        key = tuple(main)
        if key in seen:
            continue
        seen.add(key)
        lines.append(main)

    return lines
