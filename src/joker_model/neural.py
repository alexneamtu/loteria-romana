import math
import random


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


def generate_neural_lines(draws, count, rng=None, epochs=10, lr=0.1):
    rng = rng or random.SystemRandom()
    if len(draws) < 2:
        return []

    input_size = 65
    main_model = SoftmaxModel(input_size=input_size, output_size=45, rng=random.Random(0))
    joker_model = SoftmaxModel(input_size=input_size, output_size=20, rng=random.Random(1))

    inputs = []
    main_targets = []
    joker_targets = []

    for prev, nxt in zip(draws[:-1], draws[1:]):
        prev_main, prev_joker = prev
        x = _one_hot([n - 1 for n in prev_main], 45) + _one_hot([prev_joker - 1], 20)
        inputs.append(x)
        main_targets.append(_one_hot([n - 1 for n in nxt[0]], 45, value=1.0 / 5.0))
        joker_targets.append(_one_hot([nxt[1] - 1], 20))

    main_model.train(inputs, main_targets, epochs=epochs, lr=lr)
    joker_model.train(inputs, joker_targets, epochs=epochs, lr=lr)

    last_main, last_joker = draws[-1]
    last_x = _one_hot([n - 1 for n in last_main], 45) + _one_hot([last_joker - 1], 20)
    main_probs = main_model.predict_probs(last_x)
    joker_probs = joker_model.predict_probs(last_x)

    lines = []
    seen = set()
    while len(lines) < count:
        main_idxs = _sample_without_replacement(main_probs, 5, rng)
        main = sorted([i + 1 for i in main_idxs])
        joker = rng.choices(range(1, 21), weights=joker_probs, k=1)[0]
        key = tuple(main) + (joker,)
        if key in seen:
            continue
        seen.add(key)
        lines.append((main, joker))

    return lines
