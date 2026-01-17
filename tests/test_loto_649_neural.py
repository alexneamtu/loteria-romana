import unittest
import random

from loto_649_model.neural import SoftmaxModel, generate_neural_lines


class TestLoto649Neural(unittest.TestCase):
    def test_softmax_probs_sum_to_one(self):
        model = SoftmaxModel(input_size=4, output_size=3, rng=random.Random(0))
        probs = model.predict_probs([1, 0, 0, 0])
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertEqual(len(probs), 3)

    def test_training_reduces_loss(self):
        model = SoftmaxModel(input_size=2, output_size=2, rng=random.Random(0))
        inputs = [[1, 0], [0, 1]]
        targets = [[1, 0], [0, 1]]
        loss_before = model.loss(inputs, targets)
        model.train(inputs, targets, epochs=50, lr=0.5)
        loss_after = model.loss(inputs, targets)
        self.assertLess(loss_after, loss_before)

    def test_generate_neural_lines(self):
        draws = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
        ]
        rng = random.Random(0)
        lines = generate_neural_lines(draws, count=2, rng=rng, epochs=10, lr=0.1)
        self.assertEqual(len(lines), 2)
        for main in lines:
            self.assertEqual(len(main), 6)
            self.assertTrue(all(1 <= n <= 49 for n in main))


if __name__ == "__main__":
    unittest.main()
