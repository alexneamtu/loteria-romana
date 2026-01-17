import unittest

from shared.recency import (
    draw_weights,
    resolve_half_life,
    resolve_half_life_mode,
    DEFAULT_HALF_LIFE,
    DEFAULT_HALF_LIFE_MODE,
)


class TestRecencyWeights(unittest.TestCase):
    def test_draw_weights_half_life(self):
        weights = draw_weights(11, 10.0)
        self.assertAlmostEqual(weights[-1], 1.0, places=6)
        self.assertAlmostEqual(weights[0], 0.5, places=6)

    def test_draw_weights_monotonic(self):
        weights = draw_weights(5, 10.0)
        self.assertGreater(weights[-1], weights[0])

    def test_resolve_half_life_default_and_env(self):
        self.assertEqual(resolve_half_life(None, None), DEFAULT_HALF_LIFE)
        self.assertEqual(resolve_half_life(None, "25"), 25.0)

    def test_resolve_half_life_invalid(self):
        with self.assertRaises(ValueError):
            resolve_half_life(None, "0")

    def test_resolve_half_life_mode_default_and_env(self):
        self.assertEqual(resolve_half_life_mode(None, None), DEFAULT_HALF_LIFE_MODE)
        self.assertEqual(resolve_half_life_mode(None, "days"), "days")

    def test_resolve_half_life_mode_invalid(self):
        with self.assertRaises(ValueError):
            resolve_half_life_mode(None, "weeks")

    def test_draw_weights_days_mode(self):
        draw_dates = ["2024-01-01", "2024-01-11"]
        weights = draw_weights(len(draw_dates), 10.0, mode="days", draw_dates=draw_dates)
        self.assertAlmostEqual(weights[0], 0.5, places=6)
        self.assertAlmostEqual(weights[1], 1.0, places=6)

    def test_draw_weights_days_mode_requires_dates(self):
        with self.assertRaises(ValueError):
            draw_weights(2, 10.0, mode="days")


if __name__ == "__main__":
    unittest.main()
