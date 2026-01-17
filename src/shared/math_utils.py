"""Math utilities for neural networks and statistics (stdlib only)."""

import math


def softmax(logits: list[float]) -> list[float]:
    """Convert logits to probabilities using softmax function.

    Args:
        logits: Raw scores (can be any real numbers)

    Returns:
        Probability distribution (sums to 1.0)
    """
    max_logit = max(logits)
    exp_shifted = [math.exp(x - max_logit) for x in logits]
    total = sum(exp_shifted)
    return [e / total for e in exp_shifted]


def cross_entropy(predictions: list[float], targets: list[float]) -> float:
    """Compute cross-entropy loss.

    Args:
        predictions: Predicted probabilities (should sum to 1)
        targets: Target distribution (should sum to 1)

    Returns:
        Cross-entropy loss value (lower is better)
    """
    epsilon = 1e-15
    return -sum(t * math.log(max(p, epsilon)) for p, t in zip(predictions, targets))


def relu(x: float) -> float:
    """Rectified Linear Unit activation function."""
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    """Derivative of ReLU for backpropagation."""
    return 1.0 if x > 0 else 0.0


def one_hot(indices: list[int], size: int, value: float = 1.0) -> list[float]:
    """Create one-hot encoded vector.

    Args:
        indices: Positions to set to value
        size: Total size of the vector
        value: Value to use at indices (default 1.0)

    Returns:
        One-hot encoded vector
    """
    vec = [0.0] * size
    for idx in indices:
        if 0 <= idx < size:
            vec[idx] = value
    return vec


def matrix_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Multiply matrix by vector.

    Args:
        matrix: 2D list of shape (output_size, input_size)
        vector: 1D list of shape (input_size,)

    Returns:
        Result vector of shape (output_size,)
    """
    return [sum(row[i] * vector[i] for i in range(len(vector))) for row in matrix]


def vector_add(a: list[float], b: list[float]) -> list[float]:
    """Element-wise vector addition."""
    return [x + y for x, y in zip(a, b)]


def vector_subtract(a: list[float], b: list[float]) -> list[float]:
    """Element-wise vector subtraction."""
    return [x - y for x, y in zip(a, b)]


def vector_scale(vector: list[float], scalar: float) -> list[float]:
    """Multiply vector by scalar."""
    return [x * scalar for x in vector]


def dot_product(a: list[float], b: list[float]) -> float:
    """Compute dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def outer_product(a: list[float], b: list[float]) -> list[list[float]]:
    """Compute outer product of two vectors.

    Args:
        a: Column vector of shape (m,)
        b: Row vector of shape (n,)

    Returns:
        Matrix of shape (m, n) where result[i][j] = a[i] * b[j]
    """
    return [[ai * bj for bj in b] for ai in a]


def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def tanh(x: float) -> float:
    """Hyperbolic tangent activation function."""
    return math.tanh(x)


def clip(x: float, min_val: float, max_val: float) -> float:
    """Clip value to range [min_val, max_val]."""
    return max(min_val, min(max_val, x))


def mean(values: list[float]) -> float:
    """Compute arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: list[float]) -> float:
    """Compute population variance."""
    if not values:
        return 0.0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def std_dev(values: list[float]) -> float:
    """Compute population standard deviation."""
    return math.sqrt(variance(values))


def normalize(values: list[float]) -> list[float]:
    """Normalize values to have zero mean and unit variance."""
    if not values:
        return []
    m = mean(values)
    s = std_dev(values)
    if s == 0:
        return [0.0] * len(values)
    return [(x - m) / s for x in values]


def l2_norm(vector: list[float]) -> float:
    """Compute L2 norm (Euclidean length) of vector."""
    return math.sqrt(sum(x ** 2 for x in vector))


def l2_regularization(weights: list[list[float]], lambda_: float) -> float:
    """Compute L2 regularization penalty for weight matrix.

    Args:
        weights: 2D weight matrix
        lambda_: Regularization strength

    Returns:
        Regularization penalty term
    """
    return lambda_ * sum(w ** 2 for row in weights for w in row) / 2
