import unittest

from shared.bias_detection import (
    BiasReport,
    chi_square_uniformity_test,
    _lookup_critical_value,
)


class TestLookupCriticalValue(unittest.TestCase):
    def test_exact_match(self):
        self.assertAlmostEqual(_lookup_critical_value(44), 60.48)

    def test_interpolation(self):
        cv = _lookup_critical_value(34)
        self.assertGreater(cv, 42.56)
        self.assertLess(cv, 54.57)

    def test_below_range(self):
        cv = _lookup_critical_value(5)
        self.assertAlmostEqual(cv, 30.14)

    def test_above_range(self):
        cv = _lookup_critical_value(100)
        self.assertAlmostEqual(cv, 65.17)


class TestChiSquareUniformityTest(unittest.TestCase):
    def test_empty_draws(self):
        report = chi_square_uniformity_test([], 45, 5)
        self.assertFalse(report.significant)
        self.assertEqual(report.chi_square, 0.0)
        self.assertAlmostEqual(report.bias_strength, 0.0)

    def test_uniform_draws(self):
        draws = []
        for start in range(0, 45, 5):
            draws.append(list(range(start + 1, start + 6)))
        draws *= 20
        report = chi_square_uniformity_test(draws, 45, 5)
        self.assertFalse(report.significant)
        self.assertEqual(report.degrees_of_freedom, 44)

    def test_biased_draws(self):
        draws = [[1, 2, 3, 4, 5]] * 500
        report = chi_square_uniformity_test(draws, 45, 5)
        self.assertTrue(report.significant)
        self.assertAlmostEqual(report.bias_strength, 1.0)

    def test_report_fields(self):
        draws = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        report = chi_square_uniformity_test(draws, 45, 5)
        self.assertIsInstance(report, BiasReport)
        self.assertGreater(report.critical_value, 0)
        self.assertGreaterEqual(report.bias_strength, 0.0)
        self.assertLessEqual(report.bias_strength, 1.0)

    def test_haigh_correction_reduces_chi_square(self):
        draws = [[1, 2, 3, 4, 5]] * 50
        report = chi_square_uniformity_test(draws, 45, 5)
        expected_count = 50 * 5 / 45
        observed = [0] * 45
        for d in draws:
            for n in d:
                observed[n - 1] += 1
        naive = sum((o - expected_count) ** 2 / expected_count for o in observed)
        self.assertLess(report.chi_square, naive)


if __name__ == "__main__":
    unittest.main()
