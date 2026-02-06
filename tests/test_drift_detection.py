import unittest
import random

from shared.drift_detection import (
    adwin_detect_drift,
    cusum_detect_drift,
    DriftReport,
)


class TestDriftReport(unittest.TestCase):
    def test_drift_report_fields(self):
        report = DriftReport(
            method="adwin",
            drift_detected=True,
            drift_points=[50],
            statistic=2.5,
            details={"window_size": 30},
        )
        self.assertEqual(report.method, "adwin")
        self.assertTrue(report.drift_detected)
        self.assertEqual(report.drift_points, [50])


class TestADWINDetectDrift(unittest.TestCase):
    def test_no_drift_uniform(self):
        rng = random.Random(42)
        values = [rng.gauss(100, 10) for _ in range(200)]
        report = adwin_detect_drift(values)
        self.assertFalse(report.drift_detected)
        self.assertEqual(report.method, "adwin")

    def test_detects_mean_shift(self):
        rng = random.Random(42)
        before = [rng.gauss(100, 10) for _ in range(200)]
        after = [rng.gauss(150, 10) for _ in range(200)]
        values = before + after
        report = adwin_detect_drift(values)
        self.assertTrue(report.drift_detected)
        self.assertTrue(len(report.drift_points) >= 1)
        self.assertTrue(any(150 < p < 250 for p in report.drift_points))

    def test_empty_input(self):
        report = adwin_detect_drift([])
        self.assertFalse(report.drift_detected)

    def test_short_input(self):
        report = adwin_detect_drift([1.0, 2.0, 3.0])
        self.assertFalse(report.drift_detected)

    def test_custom_delta(self):
        rng = random.Random(42)
        before = [rng.gauss(100, 5) for _ in range(200)]
        after = [rng.gauss(110, 5) for _ in range(200)]
        values = before + after
        report = adwin_detect_drift(values, delta=0.01)
        self.assertIsInstance(report, DriftReport)


class TestCUSUMDetectDrift(unittest.TestCase):
    def test_no_drift(self):
        rng = random.Random(42)
        values = [rng.gauss(100, 10) for _ in range(200)]
        report = cusum_detect_drift(values, threshold=5.0)
        self.assertFalse(report.drift_detected)
        self.assertEqual(report.method, "cusum")

    def test_detects_shift(self):
        rng = random.Random(42)
        before = [rng.gauss(100, 10) for _ in range(200)]
        after = [rng.gauss(150, 10) for _ in range(200)]
        values = before + after
        report = cusum_detect_drift(values, threshold=5.0)
        self.assertTrue(report.drift_detected)

    def test_returns_drift_report(self):
        report = cusum_detect_drift([1.0] * 100)
        self.assertIsInstance(report, DriftReport)


if __name__ == "__main__":
    unittest.main()
