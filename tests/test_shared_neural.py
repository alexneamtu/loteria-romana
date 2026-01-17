import unittest
import random
import math

from shared.neural_base import Layer, MLPModel, LSTMCell, LSTMModel
from shared.math_utils import softmax


class TestLayer(unittest.TestCase):
    def test_layer_forward_shape(self):
        layer = Layer(10, 5, activation="relu", rng=random.Random(42))
        inputs = [1.0] * 10
        outputs = layer.forward(inputs)
        self.assertEqual(len(outputs), 5)

    def test_layer_relu_activation(self):
        layer = Layer(3, 3, activation="relu", rng=random.Random(42))
        # Force weights to produce negative and positive outputs
        layer.weights = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
        layer.biases = [0.0, 0.0, 0.0]
        outputs = layer.forward([1.0, 0.0, 0.0])
        self.assertEqual(outputs[0], 1.0)  # relu(1) = 1
        self.assertEqual(outputs[1], 0.0)  # relu(-1) = 0
        self.assertEqual(outputs[2], 0.5)  # relu(0.5) = 0.5

    def test_layer_softmax_sums_to_one(self):
        layer = Layer(5, 3, activation="softmax", rng=random.Random(42))
        inputs = [1.0, 2.0, 3.0, 4.0, 5.0]
        outputs = layer.forward(inputs)
        self.assertAlmostEqual(sum(outputs), 1.0, places=10)

    def test_layer_backward_returns_correct_shape(self):
        layer = Layer(10, 5, activation="relu", rng=random.Random(42))
        inputs = [1.0] * 10
        layer.forward(inputs)
        grad_output = [0.1] * 5
        grad_input = layer.backward(grad_output, learning_rate=0.01)
        self.assertEqual(len(grad_input), 10)


class TestMLPModel(unittest.TestCase):
    def test_mlp_forward_shape(self):
        mlp = MLPModel([10, 8, 5], rng=random.Random(42))
        inputs = [1.0] * 10
        outputs = mlp.forward(inputs)
        self.assertEqual(len(outputs), 5)

    def test_mlp_output_is_probability_distribution(self):
        mlp = MLPModel([10, 8, 5], rng=random.Random(42))
        inputs = [random.random() for _ in range(10)]
        outputs = mlp.forward(inputs)
        self.assertAlmostEqual(sum(outputs), 1.0, places=10)
        for p in outputs:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_mlp_training_reduces_loss(self):
        rng = random.Random(42)
        mlp = MLPModel([5, 10, 3], rng=rng)

        # Create simple training data
        inputs_list = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
        targets_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        losses = mlp.train(inputs_list, targets_list, epochs=50, learning_rate=0.1)

        # Loss should decrease
        self.assertLess(losses[-1], losses[0])

    def test_mlp_multiple_hidden_layers(self):
        mlp = MLPModel([10, 8, 6, 4], activations=["relu", "relu", "softmax"], rng=random.Random(42))
        inputs = [1.0] * 10
        outputs = mlp.forward(inputs)
        self.assertEqual(len(outputs), 4)
        self.assertAlmostEqual(sum(outputs), 1.0, places=10)

    def test_mlp_deterministic_with_seed(self):
        inputs = [1.0, 2.0, 3.0, 4.0, 5.0]

        mlp1 = MLPModel([5, 8, 3], rng=random.Random(123))
        output1 = mlp1.forward(inputs)

        mlp2 = MLPModel([5, 8, 3], rng=random.Random(123))
        output2 = mlp2.forward(inputs)

        for o1, o2 in zip(output1, output2):
            self.assertAlmostEqual(o1, o2, places=10)


class TestLSTMCell(unittest.TestCase):
    def test_lstm_cell_output_shape(self):
        cell = LSTMCell(input_size=10, hidden_size=8, rng=random.Random(42))
        x = [1.0] * 10
        h_prev = [0.0] * 8
        c_prev = [0.0] * 8
        h_new, c_new = cell.forward(x, h_prev, c_prev)
        self.assertEqual(len(h_new), 8)
        self.assertEqual(len(c_new), 8)

    def test_lstm_cell_hidden_state_changes(self):
        cell = LSTMCell(input_size=5, hidden_size=4, rng=random.Random(42))
        x = [1.0, 0.5, -0.5, 0.0, 1.0]
        h_prev = [0.0] * 4
        c_prev = [0.0] * 4
        h_new, c_new = cell.forward(x, h_prev, c_prev)

        # Hidden state should be non-zero with non-zero input
        self.assertFalse(all(h == 0.0 for h in h_new))

    def test_lstm_cell_bounded_output(self):
        cell = LSTMCell(input_size=5, hidden_size=4, rng=random.Random(42))
        x = [10.0, -10.0, 5.0, -5.0, 0.0]
        h_prev = [0.5] * 4
        c_prev = [0.5] * 4
        h_new, c_new = cell.forward(x, h_prev, c_prev)

        # Hidden state should be bounded (tanh output)
        for h in h_new:
            self.assertGreaterEqual(h, -1.0)
            self.assertLessEqual(h, 1.0)


class TestLSTMModel(unittest.TestCase):
    def test_lstm_model_output_shape(self):
        lstm = LSTMModel(
            input_size=10, hidden_size=8, output_size=5, num_layers=1, rng=random.Random(42)
        )
        sequence = [[1.0] * 10 for _ in range(5)]  # 5 timesteps
        outputs = lstm.forward_sequence(sequence)
        self.assertEqual(len(outputs), 5)

    def test_lstm_model_output_is_probability(self):
        lstm = LSTMModel(
            input_size=10, hidden_size=8, output_size=5, num_layers=1, rng=random.Random(42)
        )
        sequence = [[random.random() for _ in range(10)] for _ in range(5)]
        outputs = lstm.forward_sequence(sequence)
        self.assertAlmostEqual(sum(outputs), 1.0, places=10)
        for p in outputs:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_lstm_model_multi_layer(self):
        lstm = LSTMModel(
            input_size=10, hidden_size=8, output_size=5, num_layers=2, rng=random.Random(42)
        )
        sequence = [[1.0] * 10 for _ in range(5)]
        outputs = lstm.forward_sequence(sequence)
        self.assertEqual(len(outputs), 5)
        self.assertAlmostEqual(sum(outputs), 1.0, places=10)

    def test_lstm_model_training_reduces_loss(self):
        rng = random.Random(42)
        lstm = LSTMModel(
            input_size=5, hidden_size=4, output_size=3, num_layers=1, rng=rng
        )

        # Create training data: sequences of length 3
        sequences = [
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
            [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
            [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
        ]
        targets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        losses = lstm.train(sequences, targets, epochs=50, learning_rate=0.1)

        # Loss should decrease
        self.assertLess(losses[-1], losses[0])

    def test_lstm_different_sequence_lengths(self):
        lstm = LSTMModel(
            input_size=5, hidden_size=4, output_size=3, num_layers=1, rng=random.Random(42)
        )

        # Short sequence
        short_seq = [[1.0] * 5 for _ in range(2)]
        output_short = lstm.forward_sequence(short_seq)

        # Long sequence
        long_seq = [[1.0] * 5 for _ in range(10)]
        output_long = lstm.forward_sequence(long_seq)

        # Both should produce valid probability distributions
        self.assertAlmostEqual(sum(output_short), 1.0, places=10)
        self.assertAlmostEqual(sum(output_long), 1.0, places=10)


class TestNeuralIntegration(unittest.TestCase):
    def test_mlp_for_lottery_prediction(self):
        """Test MLP can predict lottery-like distributions."""
        rng = random.Random(42)

        # Simulated lottery: 5 numbers from 1-45
        # Input: one-hot of previous 5 numbers (45 features)
        # Output: probabilities for next 45 numbers
        mlp = MLPModel([45, 32, 45], rng=rng)

        # Create mock training data
        inputs_list = []
        targets_list = []
        for _ in range(20):
            # Random previous draw
            prev_nums = rng.sample(range(45), 5)
            prev_one_hot = [1.0 if i in prev_nums else 0.0 for i in range(45)]
            inputs_list.append(prev_one_hot)

            # Random next draw as target
            next_nums = rng.sample(range(45), 5)
            next_target = [1.0 / 5 if i in next_nums else 0.0 for i in range(45)]
            targets_list.append(next_target)

        # Train
        losses = mlp.train(inputs_list, targets_list, epochs=20, learning_rate=0.05)

        # Should have decreasing loss
        self.assertLess(losses[-1], losses[0])

        # Output should be valid probability distribution
        test_input = [1.0 if i < 5 else 0.0 for i in range(45)]
        probs = mlp.forward(test_input)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)

    def test_lstm_for_sequence_prediction(self):
        """Test LSTM can handle sequence-based lottery prediction."""
        rng = random.Random(42)

        lstm = LSTMModel(
            input_size=45, hidden_size=16, output_size=45, num_layers=1, rng=rng
        )

        # Create mock sequences (10 historical draws)
        sequences = []
        targets = []
        for _ in range(10):
            seq = []
            for _ in range(5):  # 5 draws in history
                nums = rng.sample(range(45), 5)
                one_hot = [1.0 if i in nums else 0.0 for i in range(45)]
                seq.append(one_hot)
            sequences.append(seq)

            # Target: next draw
            target_nums = rng.sample(range(45), 5)
            target = [1.0 / 5 if i in target_nums else 0.0 for i in range(45)]
            targets.append(target)

        # Train
        losses = lstm.train(sequences, targets, epochs=20, learning_rate=0.05)

        # Output should be valid
        test_seq = [
            [1.0 if i < 5 else 0.0 for i in range(45)] for _ in range(5)
        ]
        probs = lstm.forward_sequence(test_seq)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
