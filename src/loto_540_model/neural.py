import math
import random

from .strategies import SUPER_NOROC_MAX


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

    def loss(self, inputs_list, targets_list):
        total = 0.0
        for inputs, target in zip(inputs_list, targets_list):
            probs = self.predict_probs(inputs)
            for p, t in zip(probs, target):
                if t:
                    total -= math.log(max(p, 1e-12))
        return total / max(1, len(inputs_list))

    def train(self, inputs_list, targets_list, epochs=10, lr=0.1):
        for _ in range(epochs):
            for inputs, target in zip(inputs_list, targets_list):
                probs = self.predict_probs(inputs)
                for i in range(self.output_size):
                    error = probs[i] - target[i]
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


def generate_neural_lines(draws, count, rng=None, epochs=10, lr=0.1, include_super_noroc: bool = True):
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
        prev_main, _prev_super_noroc = prev
        x = _one_hot([n - 1 for n in prev_main], 40)
        inputs.append(x)
        # Target is the 6 drawn numbers from next draw
        main_targets.append(_one_hot([n - 1 for n in nxt[0]], 40, value=1.0 / 6.0))

    main_model.train(inputs, main_targets, epochs=epochs, lr=lr)

    last_main, _last_super_noroc = draws[-1]
    last_x = _one_hot([n - 1 for n in last_main], 40)
    main_probs = main_model.predict_probs(last_x)

    lines = []
    seen = set()
    while len(lines) < count:
        main_idxs = _sample_without_replacement(main_probs, 5, rng)  # Pick 5 numbers
        main = sorted([i + 1 for i in main_idxs])
        super_noroc = rng.randint(0, SUPER_NOROC_MAX) if include_super_noroc else None
        key = tuple(main) if super_noroc is None else tuple(main) + (super_noroc,)
        if key in seen:
            continue
        seen.add(key)
        lines.append((main, super_noroc))

    return lines
