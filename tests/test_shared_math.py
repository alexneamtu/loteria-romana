import unittest
import math

from shared.math_utils import (
    softmax,
    cross_entropy,
    relu,
    relu_derivative,
    one_hot,
    matrix_multiply,
    vector_add,
    vector_subtract,
    vector_scale,
    dot_product,
    outer_product,
    sigmoid,
    tanh,
    clip,
    mean,
    variance,
    std_dev,
    normalize,
    l2_norm,
    l2_regularization,
)


class TestSoftmax(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        logits = [1.0, 2.0, 3.0, 4.0]
        probs = softmax(logits)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)

    def test_softmax_all_positive(self):
        logits = [-1.0, 0.0, 1.0]
        probs = softmax(logits)
        for p in probs:
            self.assertGreater(p, 0.0)

    def test_softmax_larger_logit_higher_prob(self):
        logits = [1.0, 2.0, 3.0]
        probs = softmax(logits)
        self.assertLess(probs[0], probs[1])
        self.assertLess(probs[1], probs[2])

    def test_softmax_uniform_input(self):
        logits = [5.0, 5.0, 5.0]
        probs = softmax(logits)
        for p in probs:
            self.assertAlmostEqual(p, 1.0 / 3.0, places=10)

    def test_softmax_numerical_stability(self):
        logits = [1000.0, 1001.0, 1002.0]
        probs = softmax(logits)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)
        for p in probs:
            self.assertFalse(math.isnan(p))
            self.assertFalse(math.isinf(p))


class TestCrossEntropy(unittest.TestCase):
    def test_cross_entropy_perfect_prediction(self):
        predictions = [0.0, 1.0, 0.0]
        targets = [0.0, 1.0, 0.0]
        loss = cross_entropy(predictions, targets)
        self.assertAlmostEqual(loss, 0.0, places=10)

    def test_cross_entropy_imperfect_prediction(self):
        predictions = [0.5, 0.3, 0.2]
        targets = [1.0, 0.0, 0.0]
        loss = cross_entropy(predictions, targets)
        expected = -math.log(0.5)
        self.assertAlmostEqual(loss, expected, places=10)

    def test_cross_entropy_handles_zero_predictions(self):
        predictions = [0.0, 0.5, 0.5]
        targets = [0.0, 1.0, 0.0]
        loss = cross_entropy(predictions, targets)
        self.assertFalse(math.isnan(loss))
        self.assertFalse(math.isinf(loss))


class TestRelu(unittest.TestCase):
    def test_relu_positive(self):
        self.assertEqual(relu(5.0), 5.0)

    def test_relu_negative(self):
        self.assertEqual(relu(-5.0), 0.0)

    def test_relu_zero(self):
        self.assertEqual(relu(0.0), 0.0)

    def test_relu_derivative_positive(self):
        self.assertEqual(relu_derivative(5.0), 1.0)

    def test_relu_derivative_negative(self):
        self.assertEqual(relu_derivative(-5.0), 0.0)


class TestOneHot(unittest.TestCase):
    def test_one_hot_single_index(self):
        result = one_hot([2], 5)
        self.assertEqual(result, [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_one_hot_multiple_indices(self):
        result = one_hot([0, 2, 4], 5)
        self.assertEqual(result, [1.0, 0.0, 1.0, 0.0, 1.0])

    def test_one_hot_custom_value(self):
        result = one_hot([1], 3, value=0.5)
        self.assertEqual(result, [0.0, 0.5, 0.0])

    def test_one_hot_out_of_bounds_ignored(self):
        result = one_hot([10], 5)
        self.assertEqual(result, [0.0, 0.0, 0.0, 0.0, 0.0])


class TestMatrixOperations(unittest.TestCase):
    def test_matrix_multiply(self):
        matrix = [[1, 2], [3, 4], [5, 6]]
        vector = [1, 2]
        result = matrix_multiply(matrix, vector)
        self.assertEqual(result, [5, 11, 17])

    def test_vector_add(self):
        result = vector_add([1, 2, 3], [4, 5, 6])
        self.assertEqual(result, [5, 7, 9])

    def test_vector_subtract(self):
        result = vector_subtract([4, 5, 6], [1, 2, 3])
        self.assertEqual(result, [3, 3, 3])

    def test_vector_scale(self):
        result = vector_scale([1, 2, 3], 2.0)
        self.assertEqual(result, [2, 4, 6])

    def test_dot_product(self):
        result = dot_product([1, 2, 3], [4, 5, 6])
        self.assertEqual(result, 32)

    def test_outer_product(self):
        result = outer_product([1, 2], [3, 4, 5])
        expected = [[3, 4, 5], [6, 8, 10]]
        self.assertEqual(result, expected)


class TestActivations(unittest.TestCase):
    def test_sigmoid_zero(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=10)

    def test_sigmoid_large_positive(self):
        result = sigmoid(100.0)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_sigmoid_large_negative(self):
        result = sigmoid(-100.0)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_sigmoid_range(self):
        for x in [-10, -1, 0, 1, 10]:
            result = sigmoid(x)
            self.assertGreater(result, 0.0)
            self.assertLess(result, 1.0)

    def test_tanh_zero(self):
        self.assertAlmostEqual(tanh(0.0), 0.0, places=10)

    def test_tanh_range(self):
        for x in [-10, -1, 0, 1, 10]:
            result = tanh(x)
            self.assertGreaterEqual(result, -1.0)
            self.assertLessEqual(result, 1.0)


class TestUtilities(unittest.TestCase):
    def test_clip(self):
        self.assertEqual(clip(5.0, 0.0, 10.0), 5.0)
        self.assertEqual(clip(-5.0, 0.0, 10.0), 0.0)
        self.assertEqual(clip(15.0, 0.0, 10.0), 10.0)

    def test_mean(self):
        self.assertEqual(mean([1, 2, 3, 4, 5]), 3.0)
        self.assertEqual(mean([]), 0.0)

    def test_variance(self):
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        self.assertAlmostEqual(variance(values), 4.0, places=10)

    def test_std_dev(self):
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        self.assertAlmostEqual(std_dev(values), 2.0, places=10)

    def test_normalize(self):
        values = [1, 2, 3, 4, 5]
        result = normalize(values)
        self.assertAlmostEqual(mean(result), 0.0, places=10)
        self.assertAlmostEqual(std_dev(result), 1.0, places=10)

    def test_l2_norm(self):
        self.assertAlmostEqual(l2_norm([3, 4]), 5.0, places=10)

    def test_l2_regularization(self):
        weights = [[1, 2], [3, 4]]
        result = l2_regularization(weights, 0.01)
        expected = 0.01 * (1 + 4 + 9 + 16) / 2
        self.assertAlmostEqual(result, expected, places=10)


if __name__ == "__main__":
    unittest.main()
