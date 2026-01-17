"""Neural network implementations for lottery prediction (stdlib only)."""

import math
import random
from dataclasses import dataclass

from .math_utils import (
    softmax,
    cross_entropy,
    relu,
    relu_derivative,
    sigmoid,
    tanh,
    matrix_multiply,
    outer_product,
    vector_add,
    vector_subtract,
    vector_scale,
    l2_regularization,
)


class Layer:
    """Single dense neural network layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
        rng: random.Random = None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.rng = rng or random.Random()

        # Xavier/Glorot initialization
        scale = math.sqrt(2.0 / (input_size + output_size))
        self.weights = [
            [self.rng.gauss(0, scale) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        self.biases = [0.0] * output_size

        # Cache for backpropagation
        self._last_input = None
        self._last_z = None
        self._last_output = None

    def forward(self, inputs: list[float]) -> list[float]:
        """Forward pass through the layer."""
        self._last_input = inputs

        # Linear transformation: z = Wx + b
        z = [
            sum(w * x for w, x in zip(self.weights[i], inputs)) + self.biases[i]
            for i in range(self.output_size)
        ]
        self._last_z = z

        # Apply activation
        if self.activation == "relu":
            outputs = [relu(val) for val in z]
        elif self.activation == "tanh":
            outputs = [tanh(val) for val in z]
        elif self.activation == "sigmoid":
            outputs = [sigmoid(val) for val in z]
        elif self.activation == "softmax":
            outputs = softmax(z)
        else:  # linear
            outputs = z

        self._last_output = outputs
        return outputs

    def backward(self, grad_output: list[float], learning_rate: float) -> list[float]:
        """Backward pass with gradient descent update.

        Args:
            grad_output: Gradient from next layer
            learning_rate: Learning rate for weight updates

        Returns:
            Gradient to pass to previous layer
        """
        # Compute activation derivative
        if self.activation == "relu":
            grad_activation = [
                grad_output[i] * relu_derivative(self._last_z[i])
                for i in range(self.output_size)
            ]
        elif self.activation == "sigmoid":
            s = self._last_output
            grad_activation = [
                grad_output[i] * s[i] * (1 - s[i]) for i in range(self.output_size)
            ]
        elif self.activation == "tanh":
            t = self._last_output
            grad_activation = [
                grad_output[i] * (1 - t[i] ** 2) for i in range(self.output_size)
            ]
        else:  # linear or softmax (softmax gradient handled in loss)
            grad_activation = grad_output

        # Compute gradient for previous layer
        grad_input = [0.0] * self.input_size
        for j in range(self.input_size):
            grad_input[j] = sum(
                self.weights[i][j] * grad_activation[i]
                for i in range(self.output_size)
            )

        # Update weights and biases
        for i in range(self.output_size):
            for j in range(self.input_size):
                self.weights[i][j] -= learning_rate * grad_activation[i] * self._last_input[j]
            self.biases[i] -= learning_rate * grad_activation[i]

        return grad_input


class MLPModel:
    """Multi-layer perceptron with configurable architecture."""

    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[str] = None,
        rng: random.Random = None,
    ):
        """Initialize MLP.

        Args:
            layer_sizes: Sizes of each layer [input, hidden1, hidden2, ..., output]
            activations: Activation for each layer (default: relu for hidden, softmax for output)
            rng: Random number generator
        """
        self.rng = rng or random.Random()
        self.layers = []

        # Default activations: relu for hidden, softmax for output
        if activations is None:
            activations = ["relu"] * (len(layer_sizes) - 2) + ["softmax"]

        for i in range(len(layer_sizes) - 1):
            activation = activations[i] if i < len(activations) else "linear"
            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i + 1], activation, self.rng)
            )

        self.l2_lambda = 0.001

    def forward(self, inputs: list[float]) -> list[float]:
        """Forward pass through all layers."""
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def train_step(
        self, inputs: list[float], targets: list[float], learning_rate: float
    ) -> float:
        """Single training step with backpropagation.

        Args:
            inputs: Input features
            targets: Target distribution
            learning_rate: Learning rate

        Returns:
            Loss value
        """
        # Forward pass
        predictions = self.forward(inputs)

        # Compute loss (cross-entropy for softmax output)
        loss = cross_entropy(predictions, targets)

        # Add L2 regularization to loss
        for layer in self.layers:
            loss += l2_regularization(layer.weights, self.l2_lambda)

        # Backward pass (for softmax + cross-entropy, gradient is pred - target)
        grad = vector_subtract(predictions, targets)

        # Apply L2 regularization gradient and propagate
        for layer in reversed(self.layers):
            # Add L2 gradient to weights
            for i in range(layer.output_size):
                for j in range(layer.input_size):
                    grad_reg = self.l2_lambda * layer.weights[i][j]
                    layer.weights[i][j] -= learning_rate * grad_reg

            grad = layer.backward(grad, learning_rate)

        return loss

    def train(
        self,
        inputs_list: list[list[float]],
        targets_list: list[list[float]],
        epochs: int = 100,
        learning_rate: float = 0.01,
    ) -> list[float]:
        """Train the network.

        Args:
            inputs_list: List of input samples
            targets_list: List of target distributions
            epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            List of loss values per epoch
        """
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0

            # Shuffle training data
            indices = list(range(len(inputs_list)))
            self.rng.shuffle(indices)

            for idx in indices:
                loss = self.train_step(
                    inputs_list[idx], targets_list[idx], learning_rate
                )
                epoch_loss += loss

            avg_loss = epoch_loss / len(inputs_list) if inputs_list else 0.0
            losses.append(avg_loss)

        return losses


class LSTMCell:
    """Single LSTM cell implementation."""

    def __init__(
        self, input_size: int, hidden_size: int, rng: random.Random = None
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rng = rng or random.Random()

        combined_size = input_size + hidden_size
        scale = math.sqrt(2.0 / combined_size)

        # Initialize gate weights
        self.Wf = self._init_weights(combined_size, hidden_size, scale)  # Forget gate
        self.Wi = self._init_weights(combined_size, hidden_size, scale)  # Input gate
        self.Wo = self._init_weights(combined_size, hidden_size, scale)  # Output gate
        self.Wc = self._init_weights(combined_size, hidden_size, scale)  # Cell candidate

        # Biases (forget gate bias = 1.0 for better gradient flow)
        self.bf = [1.0] * hidden_size
        self.bi = [0.0] * hidden_size
        self.bo = [0.0] * hidden_size
        self.bc = [0.0] * hidden_size

    def _init_weights(
        self, in_size: int, out_size: int, scale: float
    ) -> list[list[float]]:
        return [
            [self.rng.gauss(0, scale) for _ in range(in_size)]
            for _ in range(out_size)
        ]

    def forward(
        self,
        x: list[float],
        h_prev: list[float],
        c_prev: list[float],
    ) -> tuple[list[float], list[float]]:
        """Single step forward pass.

        Args:
            x: Input vector
            h_prev: Previous hidden state
            c_prev: Previous cell state

        Returns:
            (new_hidden_state, new_cell_state)
        """
        combined = x + h_prev

        # Compute gates
        f = [
            sigmoid(
                sum(w * c for w, c in zip(self.Wf[i], combined)) + self.bf[i]
            )
            for i in range(self.hidden_size)
        ]
        i = [
            sigmoid(
                sum(w * c for w, c in zip(self.Wi[j], combined)) + self.bi[j]
            )
            for j in range(self.hidden_size)
        ]
        o = [
            sigmoid(
                sum(w * c for w, c in zip(self.Wo[k], combined)) + self.bo[k]
            )
            for k in range(self.hidden_size)
        ]
        c_candidate = [
            tanh(
                sum(w * c for w, c in zip(self.Wc[m], combined)) + self.bc[m]
            )
            for m in range(self.hidden_size)
        ]

        # New cell state: c = f * c_prev + i * c_candidate
        c_new = [
            f[n] * c_prev[n] + i[n] * c_candidate[n]
            for n in range(self.hidden_size)
        ]

        # New hidden state: h = o * tanh(c)
        h_new = [o[n] * tanh(c_new[n]) for n in range(self.hidden_size)]

        return h_new, c_new


class LSTMModel:
    """LSTM network for sequence prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        rng: random.Random = None,
    ):
        """Initialize LSTM.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output (typically number_pool for lottery)
            num_layers: Number of stacked LSTM layers
            rng: Random number generator
        """
        self.rng = rng or random.Random()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stack of LSTM cells
        self.cells = [
            LSTMCell(
                input_size if i == 0 else hidden_size, hidden_size, self.rng
            )
            for i in range(num_layers)
        ]

        # Output layer (hidden -> output with softmax)
        self.output_layer = Layer(hidden_size, output_size, "softmax", self.rng)

        self.l2_lambda = 0.001

    def forward_sequence(self, sequence: list[list[float]]) -> list[float]:
        """Process entire sequence and return final output.

        Args:
            sequence: List of input vectors (one per timestep)

        Returns:
            Output probabilities
        """
        # Initialize hidden and cell states
        h = [[0.0] * self.hidden_size for _ in range(self.num_layers)]
        c = [[0.0] * self.hidden_size for _ in range(self.num_layers)]

        # Process each timestep
        for x in sequence:
            current = x
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell.forward(
                    current, h[layer_idx], c[layer_idx]
                )
                current = h[layer_idx]

        # Output layer on final hidden state
        return self.output_layer.forward(h[-1])

    def train(
        self,
        sequences: list[list[list[float]]],
        targets: list[list[float]],
        epochs: int = 100,
        learning_rate: float = 0.01,
    ) -> list[float]:
        """Train LSTM with truncated backpropagation through time.

        Note: This is a simplified training that only updates output layer weights.
        Full BPTT for LSTM is complex; this provides a working baseline.

        Args:
            sequences: List of input sequences
            targets: List of target distributions
            epochs: Training epochs
            learning_rate: Learning rate

        Returns:
            Loss history
        """
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = list(range(len(sequences)))
            self.rng.shuffle(indices)

            for idx in indices:
                seq = sequences[idx]
                target = targets[idx]

                # Forward pass
                pred = self.forward_sequence(seq)

                # Compute loss
                loss = cross_entropy(pred, target)
                epoch_loss += loss

                # Simplified backprop: only update output layer
                grad = vector_subtract(pred, target)
                self.output_layer.backward(grad, learning_rate)

            avg_loss = epoch_loss / len(sequences) if sequences else 0.0
            losses.append(avg_loss)

        return losses


@dataclass
class NeuralStrategyConfig:
    """Configuration for neural network strategies."""

    architecture: str = "mlp"  # 'softmax', 'mlp', 'lstm'
    hidden_sizes: tuple[int, ...] = (64, 32)
    learning_rate: float = 0.01
    epochs: int = 100
    l2_lambda: float = 0.001
    sequence_length: int = 10  # For LSTM

    def __post_init__(self):
        if isinstance(self.hidden_sizes, list):
            self.hidden_sizes = tuple(self.hidden_sizes)
