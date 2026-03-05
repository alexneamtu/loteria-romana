import unittest

from shared.calibration import compute_calibration, CalibrationResult


class TestCalibration(unittest.TestCase):
    def test_perfect_calibration(self):
        predicted = [0.5] * 100
        observed = [True] * 50 + [False] * 50
        result = compute_calibration(predicted, observed)
        self.assertIsInstance(result, CalibrationResult)
        self.assertLess(result.brier_score, 0.3)

    def test_terrible_calibration(self):
        predicted = [0.9] * 100
        observed = [True] * 10 + [False] * 90
        result = compute_calibration(predicted, observed)
        self.assertGreater(result.brier_score, 0.5)

    def test_ece_perfect(self):
        predicted = [0.3] * 50 + [0.7] * 50
        observed = [True] * 15 + [False] * 35 + [True] * 35 + [False] * 15
        result = compute_calibration(predicted, observed, n_bins=2)
        self.assertLess(result.expected_calibration_error, 0.1)

    def test_bins_structure(self):
        predicted = [0.1 * i for i in range(10)] * 10
        observed = [i % 3 == 0 for i in range(100)]
        result = compute_calibration(predicted, observed, n_bins=5)
        self.assertGreater(len(result.bins), 0)
        for b in result.bins:
            self.assertIn("predicted_mean", b)
            self.assertIn("observed_freq", b)
            self.assertIn("count", b)

    def test_empty_input(self):
        result = compute_calibration([], [])
        self.assertEqual(result.brier_score, 0.0)
        self.assertEqual(result.expected_calibration_error, 0.0)
